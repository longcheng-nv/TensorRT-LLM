# op14 — GVR ≈1-HBM-pass secant (radix compaction) — Iteration log

Base: GVR cuteDSL rank-scatter P4 (op#7). Goal: cut large-N HBM passes 3→~1 via
candidate compaction. Target ~40% avg gain at large N, exact, no regression.
See ALGORITHM_SPEC.md for design + measured ceiling (3 full-N reads baseline;
P2+P3≈80% at 262K ⇒ 40% reachable).

Baseline (rank-scatter, nsys pure-kernel cold-L2, K512 fp32, report data):
N=65536 18.26us | N=131072 24.60us | N=262144 37.07us. (1pass==base pre-impl.)

Harness: scripts/ab.py (--exact ; nsys measure ; --parse-multi). Data byte-identical
to report.html (get_bundle cfg=beta_moderate seed=42).

## Iter 0 — scaffold ✅
- Copied rank-scatter base → src/gvr_topk_decode_1pass.py; added gated flag
  enable_1pass_compaction (+compaction_C) as stored no-op. Op wrapper
  src/gvr_1pass_op.py (gvr_rs_base / gvr_1pass). Smoke exact, A/B ~tie. Baseline
  nsys locked.

## Iter 1 — implement fused count+compact fast path ✅ (exact)
- Added global scratch cand_val[cap]/cand_idx[cap] (cap=16*K, BS-strided) to the
  kernel jit signature + __call__ + op wrapper (both entries; baseline passes None).
- New jit helpers: fused_count_compact (pass-1: stream all N once, count v>=t0
  uncapped → c0, compact survivors via smem atomic slot s_slot capped at cap),
  block_count_ge_scratch / phase2_secant_search_scratch / phase3_collect_scratch
  (operate over cand_val[0:c0] << N). t0 = pmin (s_thr[1]).
- Kernel branch (const_expr on flag): fast path runs fused pass + gate K<=c0<=cap;
  if pass → scratch P2/P3; else FALLBACK to baseline full-N P2/P3. P4 unchanged
  (reads only smem_keys/vals). Flag OFF = byte-identical baseline.
- EXACTNESS: K{512,1024}×{fp32,bf16}×N{65536,131072,262144}×seed{0,1,2}, 3 cfgs
  = 216 cells. 214/216 strict pass. The 2 fails (K1024 bf16 N262144 beta_shallow
  s1) hit BOTH base AND 1pass identically (uniq=1013/1024) — a PRE-EXISTING bf16
  boundary-tie defect (27 elems tie at the K-th bf16 value), NOT a 1pass
  regression. 1pass stays value-equivalent to base on every cell.
