# FALSIFIED.md — op40_omni_gvr

Format: (conclusion, condition domain [K/N/dtype/BS/arch], evidence strength
[host|event|nsys|NCU]) + root-cause class
`structural-wall | measurement-artifact | complexity-backfire`.

## Inherited red-lines (non-KF ledgers, per user firewall ruling)

Checkpoint command (run before every hypothesis; KF-lineage dirs excluded):

```bash
grep -il "<keyword>" FALSIFIED.md WALLS.md \
  ../op{10,12,13,14,15,16,17,18,19,20,21}_*/LEARNINGS.md \
  ../op29_gvr_hbe/{FALSIFIED,WALLS}.md ../op32_gvr_shortrow_regres/{FALSIFIED,WALLS}.md \
  ../op33_*/{FALSIFIED,WALLS}.md ../op34_gvr_beat_sglang/{FALSIFIED,WALLS,LEARNINGS}.md \
  ../op35_apex_topk/{FALSIFIED,WALLS}.md ../op35_gvr_round2/FALSIFIED.md 2>/dev/null
```

Key inherited entries (scoped; re-verify domain before applying):
- (fewer-passes/1-pass secant saves DRAM traffic, BS=1 small-N fp32, nsys) —
  measurement-artifact/structural: inputs ≪ L2, re-reads are L2 hits (op14/op15).
- (smem-resident candidates, BS=1, nsys) — complexity-backfire (op15).
- (dual-threshold speculation, K2048 fp32, nsys) — falsified (op16).
- (BS=1 short-row fp32 is latency-bound, N≲32K, NCU dram 0.06%/issue 15%) —
  structural wall; register-resident/warp-reduce dead (op32).
- (hit-rate-conditioned dispatch, any, design) — forbidden: hit unknowable at
  inference (user rule, memorialized after op26).
- (event-axis timing near launch floor, small N, event) — measurement-artifact;
  nsys mandatory (op32, multiple).

## Campaign entries (append below)
- (R0 ladder widening beats its count-column tax, K512/K1024 real fp32 BS=1,
  nsys full-865) — FALSIFIED, complexity-backfire: v2lad vs v1 gm 0.9894
  (flash 0.9617/pro 0.9669); reproduces the upstream 2026-07-16 audit on the
  real-only envelope. DOMAIN: K2048 is the exception — 4-rung ladder wins
  (+1.8% vs v1 on v32); harvested.
- (p1b_cache=True for fp32 K512/K1024, real fp32 BS=1, nsys) — WASH (<1%).
- (multi-level distributed radix-select as the P4 PERF path, all cs, real
  fp32 BS=1, nsys full-865) — FALSIFIED, complexity-backfire: v3 vs v1 gm
  0.8696; 4-level descent is the common case on continuous keys; 8 cluster
  barriers + 8 candidate scans >> 1 gather + coarse/fine. Revival: hybrid
  (1 distributed level + tiny-class handoff). Radix retained as v4's
  correctness fallback (pathological rows only) — gate GREEN 69/69 there.
- (mt_unroll 4->8 in block_count_ge_multi, all models real fp32 BS=1, nsys
  full-865) — FALSIFIED: gm 0.9625 vs v5best; worst flash 0.863.
