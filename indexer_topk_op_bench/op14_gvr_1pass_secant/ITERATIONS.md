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
- PERF (nsys ×1, K512 fp32): 1pass LOSES +43%/+94%/+102% at N=65536/131072/262144.
  Scales with N. Two issues found: (a) c0=#{v>=pmin} grows with N (9.2k/10.9k at
  131k/262k) > cap=16*K=8192 → large N FELL BACK → wasted pass-1; (b) per-survivor
  smem atomic + scattered global write.

## Iter 2 — 4-way unroll in pass-1 ✅exact / ✗perf
- Gave fused_count_compact the same 4-way unrolled LDG as block_count_ge. No
  change: +43/+95/+103%. The regression is N-proportional, NOT survivor-bound.

## Iter 3 — warp-aggregated emit + cap=32*K ✅exact / ✗perf (FALSIFIED)
- _warp_emit: one warp atomic per (k,j) via vote_ballot+popc+shuffle instead of
  per-survivor atomic; cap 16*K→32*K so large N fires the fast path (c0<cap).
- Exactness: fp32 108/108, bf16 K512 54/54, bf16 K1024 52/54 (2 shared bf16-tie).
- PERF (nsys ×3 median): +68.1% / +93.3% / +106.0% — WORSE at 65K. Stable.
- ROOT-CAUSE / FALSIFICATION (ncu): the premise "baseline = 3 HBM passes" is
  FALSE on B200. B200 L2 = 126.5 MB; the fp32 input at N=262144 is only 1.0 MB
  → fits in L2 ~100×. ncu dram__bytes_read.sum = 1.11 MB for BOTH base and 1pass
  (= ONE input read); base's P2/P3 re-reads are L2 HITS, not HBM. So baseline is
  ALREADY ≈1 HBM pass. Compaction saves zero HBM traffic and only ADDS cost:
  per-element warp-collective (ballot/popc/shuffle) over all N in pass-1 + a
  global scratch write + scratch re-reads. Net = pure loss, scaling with N.
- VERDICT: NO-SHIP at the tested N grid. The optimization can only win when the
  input EXCEEDS L2 (>126 MB ⇒ N > ~33M fp32 elems), far outside the DSv4 decode
  regime (N ≤ 262144). Flag stays OFF (default). fp32 exactness + fallback are
  correct and retained for any future >L2 regime.
