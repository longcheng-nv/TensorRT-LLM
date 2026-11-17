# DSV4.1-Flash 单用户解码优化（megakernel）报告 — M4 Pro 48 GB

日期：2026-11-16（mk-7：2026-11-17）｜ 引擎分支 `deepinfer-mac-dsv41-flash`（DeepInfer-Mac，提交 mk-1 … mk-7，基于 `opt/compute-side`）｜ 专家缓存 `--cache-gb 16`（CLAUDE.md 上限）

## 结论
| 指标 | 本轮开始 | 现在 | 说明 |
|---|---|---|---|
| 计算路径（专家全部命中、无 SSD）| 87 ms/token（11.5 tok/s）| **68 ms/token（14.7 tok/s）** | megakernel 后 GPU 内核合计 ~54 ms（各 kernel 无依赖回放吞吐之和），权重带宽下限 ~41 ms（峰值带宽）；新的两次 dispatch 稀疏注意力再省 2.0 ms/token（4K 上下文 5.4 ms/token；电池供电下同会话 A/B）|
| 无损端到端解码（中位）| 3.4 tok/s（单个提示，`scripts/decode_speed.py`）| **4.7 tok/s**（4 类提示各 200 token；同一提示为 4.58 tok/s）；mk-7 后 HTTP 实测 4.56 tok/s（接电源，与默认模式同法，4 个提示 4.11–4.92）；进程内 4 类提示各 200 token：中位 4.45、按总耗时 4.04 tok/s | SSD 受限；中位会高估实际吞吐 |
| `--fast`（有损，`--sub-rank 2 --spec-k 2`；CLI 默认）| — | **10.2–10.6 tok/s**（4 类提示 200 token 稳态，中位/均值）；**经 HTTP 服务流式实测 10.06 tok/s**（4 个提示各 300 token，含开头预热）；最终代码不加参数（默认模式）实测 **10.13 tok/s**（2026-11-17，电池供电）；**mk-7 后 HTTP 实测 10.53 tok/s**（接电源，4 个提示 10.45 / 9.47 / 11.38 / 10.83）| 达到 TPS=10；质量代价见下；mk-7 见下文 |

**TPS=10 只在有损的「缓存感知路由」下达到。** 16 GiB 缓存时 LRU 命中率约 67%，每 token 约 78 次专家读盘 × 约 2.3–2.4 ms（平均约 14.5 MB / 6 GB/s；3/3/3 位宽层 15.5 MB）；即使未来全知的 Belady 缓存也要约 46 次读盘（约 100–110 ms，已超过 100 ms 预算），所以无损 10 TPS 在这台机器上不可达。kernel 融合只能压缩计算（87→68 ms），压不掉 SSD。

**2026-11-16 起 CLI 默认即 `--fast`**（`serve` / `chat` / `bench`，启动时在 stderr 打印当前模式）；需要确定性、可复现的输出时加 `--lossless`（部署验证脚本已自动加）。Python API `LLM(...)` 仍是无损。

## 做了什么
| 改动 | 无损? | 实测收益 |
|---|---|---|
| mHC megakernel：hc_post + mixes GEMV + Sinkhorn(20 轮) + pre-mix + RMSNorm 一次 dispatch（最后一个 threadgroup 收尾）| 是 | 计算路径 87→75 ms（原来每层 ~160 个微小 kernel）|
| 稀疏注意力 megakernel：window ring + 索引器选中的压缩 KV gather、QK、sink softmax（PV 用 bf16 概率，同参考）、逆 RoPE、fp8 输出量化、ring 更新；两次并行 dispatch（scores：8 头 × 64 key / threadgroup；softmax + PV：8 头 × 64 维 / threadgroup，key 分 8 段）| 是（与参考的差异同旧 kernel，见下方更正）| BS=1 逐层串行时延：短上下文 49 µs vs 参考 138 µs（**2.8×**，逐算子 167 µs 为 3.4×）；4K 上下文（top-512 生效）82 µs vs 133 µs（**1.6×**，逐算子 171 µs 为 2.1×）|
| mxfp8 GEMV：融合 act_quant / RMSNorm prologue、分组行（wo_a）、RoPE epilogue（wq_b）、split-K | 是 | 与 MLX 持平或更快（190–240 GB/s）|
| router + shared expert 两次 dispatch：bf16 gate（省 157 MB 内存）、top-6 + 权重、下一/二层专家预测都在 kernel 内 | 是 | 280→194 µs/层 |
| MoE megakernel：slab 槽位上的 6 个专家 up/down 两次 dispatch，宽读取 + 预缩放激活 | 是 | 2/3/4-bit 全部快于 MLX gather_qmm 链 |
| 索引器精确捷径：压缩 key ≤ 512 时 top-k 就是全部 key | 是（精确）| 短/中上下文省 ~300 kernel/token |
| **GPU hot-wait**：Apple GPU 空闲 ≥1–2 ms 后降频，下一个 kernel 慢 2–4×；等 SSD 时持续提交微小 kernel 保持时钟 | 是 | 无损解码 281→213 ms/token |
| 原生 C 双优先级 IO（需求读独占 SSD，投机读低优先级、可提升、可取消）| 是 | 与旧 IO 持平，去掉 GIL 争用 |
| Engram 行在 token 开始时预取 | 是 | −9.5 ms/token |
| top-p 采样只排序前 2048 候选（60 次随机校验结果一致）| 是 | 采样 11→2 ms/token |
| **缓存感知专家替换** `--sub-rank N`：第 N 名及以后的专家若不在缓存，用缓存里分数最高的未选专家替代，权重重新归一 | **否** | 见下表 |

所有 kernel 均对照 MLX 参考链验证（多数逐位一致，其余在 fp32 求和顺序误差内）；端到端偏差与「只改 fp32 求和分组」的偏差同量级（模型对数值扰动本身是混沌的）。`pytest tests/` 33/33 通过；部署期的 API 端到端套件 `test_api_e2e.py` 在新引擎的无损模式（`--lossless`）上 **28/28 通过**（最终代码 2026-11-17 复测相同）（流式、贪心确定性+logprobs、seed、多轮前缀复用、工具调用、JSON schema+thinking、6K/12K 长上下文 needle、视觉、/v1/messages、/v1/responses）。

**更正（稀疏注意力）**：此前报告的「注意力核心 ~100 µs/层 → 30 µs」是把各层调用无数据依赖地批量回放得到的吞吐，多个 kernel 可以重叠执行，不代表解码时的时延。按解码方式串行测（每次调用的 ring 输出喂给下一次），原来的单 dispatch 版本（64 头 ÷ 每组 4 头 = 16 个 threadgroup，每个串行走完全部 640 个 key 两遍）在短上下文为 86 µs/层，在 4K 上下文为 195 µs/层，比参考算子链（133 µs）还慢。现已改为上表的两次并行 dispatch 版本。逐元素比对（`logs/mk/attn_exactness.log`）：新旧 kernel 在短上下文完全一致，4K 上下文 40 层中有 5 层各差 1 个元素（PV 求和分 8 段）；新 kernel 与参考算子链每层至多 12 / 32768 个输出元素相差一个 fp8 量化步（最差余弦 0.9999996），与旧 kernel 对参考的差异相同（来自 QK 的 fp32 求和顺序），ring 逐位一致；`DEEPINFER_MK_ATTN_SPLIT=0` 可切回旧版。表中其他单项 kernel 数字同样是无依赖回放的吞吐；端到端数字（计算路径 87→68 ms、各解码 tok/s）是解码循环实测，不受影响。数据：`logs/mk/attn_speedup_split2_rerun.log`（逐层时延）、`logs/mk/attn_e2e.log`（解码循环 A/B）。实际解码（4 类短提示各 200 token，同一会话新旧 kernel A/B）：`--fast` 10.37 vs 10.31 tok/s，无损 4.54 vs 4.41 tok/s（中位），在运行间波动之内：短上下文时省下的 ~2 ms/token 被 SSD 等待淹没，收益主要在长上下文（`logs/mk/multi_prompt_split_ab.log`）。这三组新数据都在电池供电下测得（旧 kernel 在电池下为 85.7 / 194.7 µs/层，与首次测量的 85.5 / 192.3 µs 一致）；本报告其余速度数字（含计算路径 68 ms 与各解码 tok/s）仍是此前接电源时的实测。

## mk-7（2026-11-17）：解码循环的重叠（无损，两种模式默认开启）
接电源重新采集时间线后，把每层串行链上不必要的等待挪开。四项都不改数学（每个输出元素的计算方式不变），只改提交顺序；每项可用环境变量关闭以做对照。

| 改动 | 做法 | 环境变量（默认开）|
|---|---|---|
| 路由与共享专家拆开 | `rs_up` / `rs_down` 加 threadgroup 偏移，路由（router 行 + top-k）和共享专家（w13 + w2，约 35 MB）分成两组 dispatch；路由同步只等路由，共享专家排在其后，与主机读回路由、提交命中专家并行 | `DEEPINFER_SPLIT_SHARED` |
| 未命中专家延迟提交 | 有读盘的层用已分配的槽位先构建未命中 MoE 的计算图并立即返回；解码循环先构建下一层的 hc + 注意力图，再等读盘，读完后与下一层注意力一起提交（`ExpertStore.reserve_slots` 把 `get_slots` 拆成两半）| `DEEPINFER_DEFER_MISS` |
| Engram 值投影提前 | 两个 Engram 层的 wkv GEMV（各 162 MB，只依赖 token）移出第 1 / 14 层的路由路径，改在第一个要等新读盘的层提交，填进 GPU 空档 | `DEEPINFER_ENGRAM_EARLY` |
| 等盘按完成信号唤醒 | 原生 IO 新增 `mkio_wait_us`（条件变量限时等待）；hot-wait 改为每 60 µs 提交一个不等待的保温 kernel，其间睡在读盘完成信号上。醒来延迟中位数 80 → 34 µs；保温效果不变（`keepalive2.py`：等盘后的 GEMV 均为 315–340 µs，完全空闲时为 660–1190 µs）| `DEEPINFER_HOT_TICK_US`（0 = 原方式）|
| 先发读盘再提交命中专家 | 有新读盘的层先发读盘 | `DEEPINFER_IO_FIRST`，**默认关**：在上面几项之上测不出收益（−3.7 vs −4.1 ms/token）|

**测法。**不同进程之间的速度波动有 ±3–5%（缓存历史、后台负载、GPU 时钟），会盖过 1–3% 的改动：同一套代码先后两次独立运行的均值就差了 5 ms/token。所以改用同一进程内逐 token 轮换配置（`mac-deploy-tests/mk/ab_interleaved.py`）：6 类提示，每种配置至少 500 个 token，取相邻 token 的配对差，用 bootstrap 估 95% 区间。

| 配置（默认 `--fast`，接电源）| 平均 ms/token | tok/s | 相对 mk-6 |
|---|---|---|---|
| mk-6 | 101.5 | 9.86 | — |
| + Engram 提前 + 快速唤醒 + 延迟提交 | 99.3 | 10.07 | −2.1 ms [−3.4, −0.7] |
| **+ 路由 / 共享专家拆分（mk-7）** | **95.2** | **10.51** | **−6.25 ms [−7.5, −5.0]** |

6 类提示每一类都变快。mk-7 后复测推测宽度（同法）：`--spec-k` 1 / 0 / 3 分别比 2 慢 1.2 / 6.5 / 8.6 ms/token，`--fast` 预设不变。无损模式受 SSD 吞吐限制，测不出变化：+0.4 ms [−4.8, +5.8]（780 个 token）。无损模式经 HTTP 服务实测 4.56 tok/s（接电源，同法；4 个提示 4.88 / 4.11 / 4.92 / 4.35），进程内 4 类提示各 200 token 中位 4.45、按总耗时 4.04 tok/s。HTTP 服务实测（最终代码不加参数，接电源）为 **10.53 tok/s**（4 个提示 10.45 / 9.47 / 11.38 / 10.83；生成代码的提示仍低于 10）。API 端到端：无损 **28/28**，默认 27/28（只挂「贪心两次一致」这项确定性测试，`--fast` 本来就不可复现）。

**正确性。**
- `test_mk_forward.py --cache-gb 6`：关掉推测，让每个未命中层都走延迟提交，新旧两条路径 12 步的 logits 指标逐位相同，与参考实现的 argmax 12/12 一致。
- `test_mk_router_split.py`：拆分后的路由 kernel 与原 kernel 逐位相同（4 层 × 3 组输入 × 3 种推测设置）。
- 无损模式 16 GiB：新旧路径各解码 2 个提示 × 60 token，logits 逐位相同。

无损模式的 logits 本来就不保证逐位可复现。专家按「命中 / 未命中」分两组分别累加，分组取决于缓存状态和读盘时序。旧代码自己连跑两次，也会在某个提示上出现约 1e-2 的相对差，但 token 一致。

## 速度—质量前沿（cache 16 GiB）
质量：官方 API 采样续写（antirez/ds4 V4.1-Flash 向量）的 teacher-forced NLL 与 top-1 一致率，与无损配置逐例配对，95% bootstrap 区间。速度：200 token 贪心稳态（第 20 个 token 之后）。

| 配置 | 解码 tok/s | ΔNLL | Δ API top-1 一致率 |
|---|---|---|---|
| 无损（`--lossless`）| 4.5–4.7 | 0 | 0（噪声地板 ±0.01 NLL）|
| `--sub-rank 4`（替换第 5–6 名）| ~5 | +0.025 [+0.003, +0.047] | −0.8 pp |
| `--sub-rank 3`（替换第 4–6 名）| ~7.6 | +0.053 [+0.013, +0.090] | −2.2 pp |
| `--sub-rank 2` + 第 3 名 gap 0.2（4–6 名不限；内部参数 SUB_FREE_RANK=3，CLI 的 `--sub-gap` 会对第 3–6 名都限 gap）| ~9.1 | +0.065 [+0.024, +0.108] | −3.0 pp |
| **`--fast` = `--sub-rank 2 --spec-k 2`（替换第 3–6 名）** | **10.2–10.6**（服务实测 10.06）| **+0.087 [+0.046, +0.129]**（general 40 例）；**+0.050 [+0.033, +0.068]**（held-out router 60 例，3786 位置）| **−2.6 pp / −2.4 pp** |
| 同上但直接丢弃不替换 | — | +0.144 | −3.8 pp |

规律：速度与质量代价都随「被替换的专家数」近似线性变化；替换优于丢弃；分数差（gap）限制每次替换的质量代价最低，但换来的速度也少。

`--fast` 功能回归：API 端到端套件 **26/28 通过**（最终代码默认模式 2026-11-17 复测相同）——精确答案、生成代码可执行、JSON schema、工具调用、6K/12K 长上下文检索、视觉全部通过；失败的 2 项是「同一贪心请求两次输出一致」与「同 seed 可复现」。这是缓存感知路由的固有性质：替换哪个专家取决于当时缓存里有什么（受之前的请求和 SSD 时序影响），所以 **`--fast` 的输出不可复现**；需要确定性输出时请加 `--lossless`。

## 为什么无损做不到 10 TPS（复核用的数据）
- 权重精度（决定读盘字节）：线性层（注意力投影、共享专家）为 FP8（mxfp8，由官方 FP8 逐位无损转换）；LM head 8 位 affine；门控、norm、embedding 为 BF16。路由专家由官方 FP4（每个约 18.8 MB）重新量化为 2–4 位整数 affine：25 层 3 位、17 层 2/2/3 位、第 39 层 4 位，每个 12.5 / 15.5 / 19.9 MB，平均每次读约 14.5 MB，比 FP4 少约 23%。这里的「无损」指路由与参考实现一致，比较对象是这份本地量化模型，不等于与官方 FP4 权重逐位一致。
- 计算侧：每 token 读 ~11 GB 常驻权重（dense mxfp8 5.2 GB + 共享专家/路由 1.8 GB + 6 个路由专家 3.5 GB + head 0.7 GB）；现在 GPU 内核 ~54 ms（各 kernel 无依赖回放吞吐之和），带宽下限 ~41 ms（峰值带宽）。
- SSD 侧：`logs/mk/traces/trace_c16.npz`（960 个解码 token 的完整路由分数）离线回放：LRU 0.674 命中 → 78 读/token；Belady 0.811 → 46 读/token；LFU / 2Q / 衰减 LFU / 按名次刷新都没有明显优于 LRU（最好的 2Q 每 token 只少约 3%）；缓存 24 GiB → 61 读/token。
- 时序模型 `sim_pipeline.py`：计算压到 48 ms、`--sub-rank 3`、16 GiB 时约 110 ms/token；要到 100 ms 需要再大缓存或更激进的替换。
- 自投机（DSpark MTP）：每轮只接受 ~1.9 token，验证要读所有草稿 token 的专家并集，SSD 受限下反而更慢（2.5–2.8 tok/s），已放弃。

## 怎么用
```bash
cd /Users/loncheng/Models/DeepInfer-Mac
# 默认：--fast（有损，约 10.5 tok/s，输出不可复现；融合 kernel 默认开启）
.venv/bin/python -m deepinfer_mac serve --model /Users/loncheng/Models/dsv41-mac --cache-gb 16 --port 8000
# 无损（确定性、可复现，约 4.5 tok/s）
.venv/bin/python -m deepinfer_mac serve --model /Users/loncheng/Models/dsv41-mac --cache-gb 16 --lossless --port 8000
# 关闭融合 kernel（回到参考 MLX 路径）
... --no-mk
```

## 还能做什么
1. 更大的专家缓存最有效（需要关掉其他应用、放宽 CLAUDE.md 的 16 GiB 上限）：24 GiB 时读盘 −22%，`--sub-rank 3` 有望接近 10 TPS。
2. 外置 Thunderbolt 5 SSD 与内置盘条带化（专家文件分两半），SSD 带宽翻倍是唯一能直接解决无损瓶颈的硬件手段。
3. GPU 侧专家调度（命中专家不等 CPU 同步）：逐层同步的总开销约 6 ms/token（计算路径实测：无逐层同步 61.6 ms vs 有同步 67.8 ms），这是上限；其中路由同步到首个 MoE kernel 的 CPU 侧延迟实测 ~67 µs/层（≈ 2.7 ms/token）。mk-7 的路由 / 共享专家拆分已经让共享专家填进这段延迟。
4. 默认模式下剩下的大头是 SSD 等待（第 1–2 名专家未命中时的需求读，以及来不及读完的推测读）。不改质量时，缓存从 16 GiB 加到 24 GiB 收益最大（需要放宽 CLAUDE.md 上限）。

## 复现
- 工具：`DeepInfer-Mac/mac-deploy-tests/mk/`（`mk_bench.py`、`multi_prompt.py`、`mk_quality.py`、`sim_*.py`、kernel 测试 `test_mk_*.py`、稀疏注意力时延 `bench_attn_speedup.py` 与解码 A/B `bench_attn_e2e.py`）
- 数据：`/Users/loncheng/Models/logs/mk/`（`bench.jsonl`、`quality/*.json`、`traces/`）
