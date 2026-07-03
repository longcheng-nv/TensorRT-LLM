# op20 RESUME (paste-ready for a fresh session)

继续 op20 GVR 极致优化(omni-kernel 协议)。先读:
`indexer_topk_op_bench/op20_gvr_extreme/{PLAN,ITERATIONS,LEARNINGS}.md`

**Goal**: 单一 GVR 算子平均超越所有对手 + ≥95% cell 最快(tier1 允许 ≤4 输)。
**Priority**: tier1 fp32 K512/1024 > tier2 fp32 K2048 > tier3 16-bit。
**Branch**: `omni/op20-gvr-extreme` @ iter6(tier1 CLOSED)。Node: umbriel-b200-039 (2×B200)。

## State (iter5, tier1 84 cells, all exact)
- **65/84 fastest** vs in-run radix_cutedsl;x/base gm 1.255;rival/x gm 1.345。
- iter4(融合 P2+P3):小 N 假设证伪(墙 = P1+barrier 固定成本,fuse 不敏感);
  但大 N 决定性收益(P3 成本 ~ 候选数而非 N;最优 M 由 4 降 2;15 key 重调,
  262K BS64 达 1.43×)。cfg 后缀 f/nf 显式控制,裸 cfg = auto-gate。
- iter5(fusP4T4 路由):131K/262K BS1/4 共 8 key 切到 op17-v2 P×T fusion,
  probe 1.05–1.15× vs mc/mcC8;K1024-262K 洞 0.909/0.934 → 0.953/0.991。
  BS16 fusion 崩(bs×P 过展,不路由);fusP8T4 无法启动(32>16 CTA 上限)。
- 剩余输区:**15 个小 N 墙 cell**(N4096/8192,0.78–0.88,配置不敏感)+
  4 个准平局(0.95–1.00)。

## Next actions (queued)
- **tier1 已收口(iter6)**:小 N 墙确认结构性(相位链地板;smem-resident/
  mc smem-cache/P1 子采样三条路全部证伪,warm 归因显示 N4096 有 2.05µs 级数
  地板)。最终 65/84 fastest + 4 准平局,rival/x gm 1.345,84/84 exact。
- **tier2(fp32 K2048)**:`tier_bench.py --tier 2`,同 probe→dispatch→全量
  协议;fused f/nf 与 fusP4T4 杠杆可直接复用(注意 K2048 cr=1)。
- **tier3(16-bit)**:洞需 2.1×;方向 = 数据并行 + 单趟 16-bit key 直方图
  阈值(见 PLAN.md)。

## Protocol (硬规则)
- 取舍只认 GPU0 独占全量 `scripts/tier_bench.py --tier 1`(84 cells);
  跨 GPU/并发烟测只用于编译/正确性。
- 未在该 BS 桶探针过的配置不得写进 dispatch 表(iter3 BS16 教训;iter5 再证:
  fusion BS16 崩即未路由)。
- 每 iter:全量 A/B → ITERATIONS.md → `git commit -s`(带 Made-with/Co-Authored-By
  trailers)。里程碑加 nsys 纯核复验 + ×3 seed exact。
- 红线见 LEARNINGS.md(op14/15/16 + iter1 level-2 + smoke 教训)。

## Key files
- `src/gvr_x_op.py` — op19 sandwich + fused P2+P3(iter4)+ mc/mcC/fusP 路由
- `results/dispatch_table_fp32.json`(.pre_iter{2,3a,4b,5} = 备份链)
- `scripts/tier_bench.py`(验收)、`probe_variants.py`(归因)、
  `iter4_smoke.py`/`iter4b_retune.py`/`iter5_probe.py`(iter4/5 验证)
- `results/iter{0,1,2,3a,4,4b,5}_tier1.jsonl` — 逐 iter 全量数据
