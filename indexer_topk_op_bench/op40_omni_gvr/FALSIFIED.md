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
