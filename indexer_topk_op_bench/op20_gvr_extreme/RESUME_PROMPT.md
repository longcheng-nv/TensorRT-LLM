# op20 — CAMPAIGN CLOSED (2026-07-03)

op20 GVR 极致优化 campaign 已全部收口(tier1/2/3 CLOSED)。本文件仅作状态
存档;如需重启后续工作,从「Post-campaign leads」一节挑方向,先读
`{PLAN,ITERATIONS,LEARNINGS}.md`。

**Branch**: `omni/op20-gvr-extreme`。

## Final scoreboard (all exact, cold-L2 CUDA-graph, in-run 3-way A/B)
| tier | grid | fastest | rival/x gm | 收口 commit |
|---|---|---|---|---|
| tier1 fp32 K512/1024 | 84 | 65/84 (+4 准平局) | 1.345 | @01aa989b4f |
| tier2 fp32 K2048 | 36 | 24/36 | 1.251 | @bd98ac2135 |
| tier3 16-bit (bf16/fp16) | 240 | 121/240 | 1.086 | 本 commit |

采纳杠杆:fused P2+P3 slot-collect(f/nf 后缀 + auto-gate)、fusP4T4 路由
(131K/262K BS1/4,全 dtype)、mc/mcC8 路由(大 N BS16 / 262K 低 BS)、
dispatch 全面重调(fused 使最优 M 4→2)。

## 结构墙(已证伪不可解,勿重试)
- 小 N(4-8K)相位链地板 ~2µs:smem-resident/mc-cache/P1 子采样三路证伪。
- K2048 N8192-16K 墙(cr=1,N/K=4)。
- 16-bit 大 N 残差:radix 字节减半而 GVR 相位链固定成本不缩,parity 比
  fp32 更难;各桶已挂 portfolio 最优 cfg。
- BS16 fusion 崩(bs×P 过展)、fusP8T4 超 16-CTA 上限、iter1 level-2
  sub-histogram(op16 律)。

## Post-campaign leads(未排期)
- op13 cheaper-P2(cand-cut 换根查找器)仍是唯一已识别的真实杠杆。
- B300 交叉验证(PLAN gate 提及,未跑;fp32 op17 已证 HW-invariant)。
- 16-bit 大 N 如需翻盘:需 16-bit 专用 key 直方图(PLAN tier3 方向,未动工
  —— 表调优已把洞从 0.36 收到 0.75+,剩余属结构性)。

## Key files
- `src/gvr_x_op.py` — sandwich + fused P2+P3 + mc/mcC/fusP 路由(全 dtype 共用)
- `results/dispatch_table_{fp32,bf16,fp16}.json`(备份链 .pre_iter*/.pre_tier*)
- `results/{iter0..5,tier2_iter*,tier3_iter*}_*.jsonl` — 全程数据
- `scripts/tier_bench.py`(JSONL 断点续跑)、`tier3_probe.py`(逐桶探针范式)
- `ITERATIONS.md` — 全程记分板与证伪记录
