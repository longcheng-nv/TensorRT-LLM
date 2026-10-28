# 面向 Roofline 的推理优化（不裁模型）：DSV4.1-Flash on M4 Pro 48 GB，BS=1，ISL=1024

日期：2026-10-28 ｜ 目标：达到本机 roofline 的 60% ｜ 结果：**解码 68%，prefill ≈100%**（配置级优化，无损）

## 1. Roofline 定义（本机为 SSD 流式部署，必须按双资源建模）
每 token 解码时间下限：
TPOT_roof(C) = max( miss_bytes(C) / BW_ssd , resident_bytes / BW_mem , T_compute_ideal )
- BW_ssd = 6.5 GB/s（`bench_store.py` 实测随机 15 MB 专家块读取饱和值）
- BW_mem = 273 GB/s（M4 Pro 统一内存），resident_bytes = 8.2 GB dense + 约 2.8 GB 当 token 专家 → 40 ms
- miss_bytes(C) = 240 专家/token × 15.5 MB × (1 − 命中率(C))，命中率随缓存 C 增长

Prefill(ISL=1024) 下限：触达专家总字节 / BW_ssd（1024 个 token 几乎触达每层全部专家，约 100–130 GB → 15–20 s），GPU 算力项（约 33 TFLOP / 15 TFLOPS ≈ 2.3 s）不是瓶颈。

TTFT 和 TPOT 都由 SSD 带宽决定；统一内存消除了 CPU→GPU 拷贝，流水线能做的只是让 SSD 不空闲。

## 2. 基线诊断（缓存 9 GiB）
| 量 | 值 |
|---|---|
| 解码 | 416 ms/token（2.4 tok/s），命中 0.50，未命中 1.57 GB/token |
| SSD 纯传输时间 | 1.57 / 6.5 = 241 ms = roofline(9) |
| 计算侧 | 约 175 ms（attention 134 + 路由/共享专家 63，含 dispatch 开销） |
| SSD 忙碌度 | 58%：每层必须先算 attention 与路由才知道读哪些专家，SSD 在此期间空闲 |
| 效率 | 2.4 / 4.1 = 58% |

## 3. 无损优化手段的实测（`sweep_overlap.sh`，decode_speed，20 token 贪心）
| 缓存 GiB | SPEC_K | LOOKAHEAD_K | ms/token | tok/s | 命中 | GB/token | SSD 等待 ms |
|---|---|---|---|---|---|---|---|
| 9 | 0 | 0 | 425 | 2.35 | 0.55 | 1.57 | 246 |
| 9 | 3 | 0 | **360** | **2.78** | 0.64 | 1.68 | 168 |
| 9 | 0 | 6 | 467 | 2.14 | 0.76 | 2.18 | 276 |
| 18 | 0 | 0 | 353 | 2.83 | 0.69 | 1.10 | 178 |
| 18 | 3 | 0 | **308** | **3.25** | 0.76 | 1.20 | 122 |
| 18 | 3 | 6 | 334 | 3.00 | 0.82 | 1.52 | 171 |
| 22 | 3 | 0 | 297 | 3.36 | 0.78 | 1.08 | 109（swap +370 MB） |

结论：
- **SPEC_K=3（空闲窗口推测预取 3 个下一层专家）**：用 SSD 空闲时间读取，+17–18%。
- **路由前瞻 LOOKAHEAD**：预测错误导致每 token 多读 30–40% 字节，SSD 受限下必然变慢，应关闭。
- **缓存 9→18 GiB**：命中 0.55→0.76，+20%；22 GiB 边际收益 3% 且触发 swap，本机上限取 18。

## 4. 目标负载确认（bench，ISL=1024，gen 32，缓存 18，SPEC_K 3）
| 指标 | 基线（缓存 9） | 优化后 | roofline | 效率 |
|---|---|---|---|---|
| 解码 | 2.37–2.40 tok/s | **3.13–3.16 tok/s** | 命中 0.62 → 未命中 1.41 GB → 217 ms → 4.6 tok/s | **68%** |
| Prefill 1024 | 15.4–18.0 s | 14.8–16.2 s | 101–105 GB / 6.5 GB/s = 15.5–16 s | **≈100%** |
| 每轮专家读取 | 126–129 GB | 101–105 GB | | |

（长 prefill 会把缓存冲成 prefill 分布，解码命中率 0.62 低于短提示词的 0.76，roofline 随之变化；表中按实际命中率计算。）

## 5. 推荐配置
```bash
python -m deepinfer_mac serve --model /Users/loncheng/Models/dsv41-mac --cache-gb 18 --spec-k 3 --port 8000   # lookahead 默认关
```
内存：dense 8 GB + 缓存 18 GiB + 运行开销 ≈ 28 GB；与用户约 28 GB 的其他应用共存时接近 48 GB 上限，若出现 swap 增长请降到 16。

## 6. 从 68% 到 ~100% 的路径（需要改引擎代码，仍然无损）
缓存 18 时 SSD 下限 185–217 ms 与计算侧 186 ms 已相当，继续提升必须压缩计算侧：
1. attention 投影 69 ms + 核心 65 ms 远高于其带宽成本（dense 8 GB / 273 GB/s = 30 ms），主要是 40 层 × 多个小 kernel 的 dispatch 与 Python 调度开销：把每层 attention+路由整体 `mx.compile`、减少 `mx.eval` 同步点、把 numpy 往返（路由索引）留在 GPU。预期计算侧 186 → 100–120 ms。
2. 路由先行：先算第 l 层路由就发起读取，再算共享专家与 attention 的可并行部分（当前顺序是 route+shared 一起算）。
3. I/O 线程 8→12 并把 SPEC_K 自适应为「SSD 空闲时间能装下的专家数」。
达成后 TPOT → max(SSD 185–217, 计算 ~110) ≈ roofline，解码约 4.5–5 tok/s。再往上只有加内存（更高命中率）或换机器。

数据：`logs/opt/bench1024-baseline.log`、`bench1024-c18-spec3.log`、`sweep_overlap.tsv`、`logs/prune/profile_parts.log`、`bench_store.log`。
