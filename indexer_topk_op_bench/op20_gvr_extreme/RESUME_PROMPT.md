# op20 RESUME (paste-ready for a fresh session)

继续 op20 GVR 极致优化(omni-kernel 协议)。先读:
`indexer_topk_op_bench/op20_gvr_extreme/{PLAN,ITERATIONS,LEARNINGS}.md`

**Goal**: 单一 GVR 算子平均超越所有对手 + ≥95% cell 最快(tier1 允许 ≤4 输)。
**Priority**: tier1 fp32 K512/1024 > tier2 fp32 K2048 > tier3 16-bit。
**Branch**: `omni/op20-gvr-extreme` @ iter3 (0cf336a9b0)。Node: umbriel-b200-044 (2×B200)。

## State (iter3, tier1 84 cells, all exact)
- ~62/84 fastest vs in-run radix_cutedsl;x/base gm ~1.19。
- 剩余输区:
  1. **小 N 墙**(16 cells,N4096/8192 全 BS,0.74–0.88):所有 M/G 配置挤在
     13–15µs vs radix 10.4–12.9 ⇒ 固定成本(P1+barrier 链+P4),调参无效。
  2. **K1024 262K BS1/4**(0.88–0.91):mcC8 已到 0.91;需 chunked-sandwich。
  3. 3 个 parity 噪声 cell(0.983–0.996)。

## Next actions (queued)
- **iter4(主攻小 N)**:P2+P3 融合——ladder 的 M 阈值扫描前已知(≠被证伪的
  Opt-L:secant 需收敛后才知阈值),同一趟 N 扫描内计数 + 按最松列收集候选进
  smem(≤kC 上限,溢出回退),省一条 P3 全 N 扫描 + barrier;加 P1 gather 精简
  (采样 K/4 做 stats)。目标 13.1→~10.5µs @N4096 M2R1p4。
- **iter5**:K1024-262K chunked-cluster sandwich(mc 数据并行计数 + sandwich
  band P3/P4;或直接在 mc 内核上加 band 精化)。
- **tier2/tier3**:tier1 达标后按同法推 K2048、16-bit(16-bit 洞需 2.1×,
  数据并行 + 单趟 16-bit key 直方图阈值)。

## Protocol (硬规则)
- 取舍只认 GPU0 独占全量 `scripts/tier_bench.py --tier 1`(84 cells);
  跨 GPU/并发烟测只用于编译/正确性。
- 未在该 BS 桶探针过的配置不得写进 dispatch 表(iter3 BS16 教训)。
- 每 iter:全量 A/B → ITERATIONS.md → `git commit -s`(带 Made-with/Co-Authored-By
  trailers)。里程碑加 nsys 纯核复验 + ×3 seed exact。
- 红线见 LEARNINGS.md(op14/15/16 + iter1 level-2 + smoke 教训)。

## Key files
- `src/gvr_x_op.py` — op19 sandwich + mc/mcC 路由(gvr_sw_auto 入口)
- `results/dispatch_table_fp32.json`(.pre_iter2/.pre_iter3a = 备份)
- `scripts/tier_bench.py`(验收)、`scripts/probe_variants.py`(归因)
- `results/iter{0,1,2,3a}_tier1.jsonl` — 逐 iter 全量数据
