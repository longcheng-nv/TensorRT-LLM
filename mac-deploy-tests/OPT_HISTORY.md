# 推理优化历程（可回放）：DSV4.1-Flash on M4 Pro 48 GB，BS=1，ISL=1024

日期：2026-10-28 ｜ 引擎分支 `opt/compute-side`（每轮一个提交 + 标签 `opt-iterN`）｜ 记录：`logs/opt/journal/`

## 回放方式
```bash
/Users/loncheng/Models/dsv41-mac-tests/opt_replay.sh 3      # 检出 opt-iter3 到独立 worktree，用记录时的配置重跑解码基准
/Users/loncheng/Models/dsv41-mac-tests/opt_replay.sh all
```
每轮记录协议（`opt_iter.sh`）固定：缓存 18 GiB、路由前瞻关、真实 Engram；decode_speed 三次取中位数（14 token 提示、20 token 贪心）、bench ISL=1024 gen 32 两轮取第二轮、parity/engine/server 单测、贪心探针 17×23。原始数据：`iter-N.json`、`iter-N.log`、`iterations.jsonl`、`journal.md`。

验证回放一致性：iter0 记录 306 / 回放 293 ms；iter5 记录 270 / 回放 264 ms（噪声约 ±10 ms）。

## 历程
| 轮 | 改动 | 解码 ms/tok（3 次中位） | ISL1024 解码 tok/s | 结论 |
|---|---|---|---|---|
| 0 | 基线（上游 a4df33c + hf.py/server.py 补丁），SPEC_K 3 | 306 | 3.18 | — |
| 1 | 去掉解码路径索引层的 `mx.eval`，idx/pred 两次同步合一 | 286 | 3.18 | 小幅/噪声边缘 |
| 2 | Block 逐元素链融合为 2 个编译图/层 | 290 | 3.27 | **无增益**：逐元素 kernel 不是主项 |
| 3 | SPEC_K 3→4（推测精度 87%、召回 58%） | 273 | 3.23 | **有效**，约 −6% |
| 4 | 二级推测（l+2 top-k）实现并实测 k=1/2/3：+3/+6/+14% | 270（默认关） | 3.13 | **负结果**：SSD 空闲窗已被填满，多读只会推迟需求读 |
| 5 | MergedLinear：wq_a+wkv、共享 w1+w3 各合成一次 matmul | 270 | 3.06 | **无增益** |
| 6 | I/O 线程 8/16/24 对比；推测置信度过滤离线标定 | 259 | 3.25 | 线程数无增益；过滤是线性权衡，未实现 |

累计：短提示解码 306→259 ms（**−15%**，3.2→3.5–3.8 tok/s）；ISL=1024 解码 3.18→3.25 tok/s（噪声内），prefill 14.7→14.6 s（已在 SSD roofline）。

## 关键诊断（决定了哪些路走不通）
1. **纯 GPU 时间只有 69 ms/token**（`microbench_gpu_throughput.py`：attention 投影 12.8、核心 18.3、路由+共享 9.8、专家 gather 22.9、索引 4.9）。端到端 270 ms = SSD 等待 125 + GPU 69 + **流水线气泡约 76**。气泡来自每层一次主机同步（路由结果→SSD 槽位分配），GPU 队列排空后等 CPU 发下一层。这是 MoE 流式架构的固有依赖，不能无损去除；因此 kernel 融合（轮 2、5）无效。
2. **推测预取的精度曲线**（`spec_accuracy.py`）：按预测名次精度为 97/93/84/73/56/44%，K=4 是收益拐点；两层前瞻 top-2 精度 92% 但实测仍为负（带宽已饱和）。从 MoE 输出而非输入做预测没有更准（attention 决定路由）。
3. **SSD 已接近满载**：6.5 GB/s 随机读上限，1.29 GB/token → 198 ms 传输下限；当前 259–270 ms 对应 SSD 忙碌度 73–77%。剩余差距是气泡期间无可预取的字节。

## 当前位置 vs roofline
| 负载 | 实测 | roofline | 效率 |
|---|---|---|---|
| 短提示解码（缓存 18，命中 0.74） | 259 ms/tok | 1.31 GB / 6.5 GB/s = 202 ms | **78%** |
| ISL=1024 解码（命中 0.62） | 3.25 tok/s | 4.6 tok/s | **71%** |
| ISL=1024 prefill | 14.6 s | 15.5 s | ≈100% |

## 继续提升需要换方法（均超出「无损配置/引擎微调」范围）
- 去掉每层同步：GPU 侧槽位表 + 推测路由带验证（flash-moe 式），复杂且作者实测推测路由开销 38%。
- 更大内存（命中率↑直接降低 SSD 字节）或更快 SSD（Thunderbolt 5 外置 ≈ 内置，无增益）。
- 有损：`--moe-topk 4`（实测 +38%）、轻度专家裁剪（≤1.2×，见 PRUNING_ANALYSIS.md）。

## 交付
- 引擎：`DeepInfer-Mac` 分支 `opt/compute-side`，HEAD = iter6；相对基线保留的有效改动：SPEC_K 默认 4、解码路径同步精简、Block 融合、MergedLinear（无害）、二级推测代码（默认关）。全部通过 33 项 parity/engine/server 单测与贪心探针。
- 服务：`python -m deepinfer_mac serve --cache-gb 18`（spec-k 默认 4），运行于 127.0.0.1:8000。
- 工具：`opt_iter.sh`、`opt_replay.sh`、`microbench_layer.py`、`microbench_gpu_throughput.py`、`cprofile_decode.py`、`spec_accuracy.py`、`sweep_overlap.sh`。
