# op37 — GVR (pure-algorithm) vs sglang_v2 on real §7b, N≥32K, fp32

Started 2026-07-20, umbriel-b200-028 (8×B200). Goal (user /goal): on the op26
REPORT §7b real decode-capture axis (V4 Flash/Pro + V3.2, BS × ISL rungs),
fp32, K∈{512,1024,2048}, **indexer N ≥ 32K only**, make GVR beat sglang_v2 —
by improving the GVR algorithm itself. RED LINE: no dispatch-to-other-operator
(no radix escape, no sgl_bx port as the winning arm — op36's ship table is
explicitly out of scope).

## Baseline arithmetic (from op26 rival_long.csv, OLD head @018251950f)

- All-N fp32 composite gvr_pr/sglang: gm 0.745 (275 cells).
- **N≥32K restriction: gm 0.895 (132 cells)** — the 4-16k hole is out of scope.
- Loss map (sgl/pr, >1 = GVR faster), 12 rungs × 11 BS:
  - BS 1-8: gm ~0.72 (48 cells) — DOMINANT deficit.
  - BS 16: 0.85 (transition).
  - BS 32-128: **1.12-1.22 (GVR already wins — sglang mid-BS valley)**.
  - BS 256-1024: ~0.91, dragged by flash-512k 0.50-0.55 + flash-1M 0.77
    (measured PRE-vseed; current head repaired flash-1M big-BS up to 1.43×).
- Feasibility: BS≤8 → 0.90 alone gives composite ≈0.971; + BS16 parity ≈0.985;
  + big-BS repair ≈1.01. Multiple partial wins compose to >1.0.

## Evidence anchors

- p4tt clock64 phase split @e6fdbfac3d (commit b1e439fe35): at cs=8 BS=1,
  P4 select(+tail) = 51-58% of kernel, LEADER-ONLY while 7/8 CTAs wait;
  P2 17-21%, P1 9-17%. → distributing P4 is the main BS≤8 lever.
  Perfect-4× dist_P4 ⇒ kernel ×0.59 ⇒ BS=1 cells 0.67→~1.13.
- op36 A2 dist_p4 (gvrpkg36, 6 cluster syncs): wins only N≈262K
  (flash-1M BS1 1.19, BS256 1.41), loses N=65-131K (sync tax boundary
  ~160K); pro-512k rows contaminated by pre-p4tt tie tax.
- pick_config: N<65536 → cs=1 ⇒ the N=32771 rung runs 1 CTA/row at BS≤8
  (147 idle SMs) — L1 launch-policy probe (forced cs2/cs4) in flight.
- op34 lock: BS=1 GVR-beats-sglang-by-30% infeasible (oracle collect-only UB
  ≈ sglang whole kernel). We need ~parity at BS≤8, NOT +30% — inside the UB.

## Falsification red-lines (do NOT re-propose)

P4-internal reseeding/fine-hist-iteration; fused P2+P3 single-scan (Opt-L);
smem-resident row; sw-pipeline occupancy at BS=1; hit-rate dispatch (hit
unknowable at inference); ms_auto fused count+collect (1.47× slower);
2-way multi-threshold P2. See project_gvr_topk_falsification_history.

## Campaign tracks

- T0 baseline @e6fdbfac3d: src/{ops_op37,sweep_op37,drive_op37}.py|sh →
  results/baseline (12 rungs × 11 BS × {gvr_pr, sglang_v2}, nsys cold-L2,
  2-way workers, cell-resumable). IN FLIGHT.
- L1 cs-policy probe: forced cs2/cs4 at N=32771 rung (gvr_cs2/gvr_cs4 arms)
  → results/l1probe. IN FLIGHT.
- DP4 port: agent splicing gvrpkg36 dist_p4 → variant/gvrpkg37 on prod2
  (current head), exactness battery only. IN FLIGHT. Then my nsys A/B on
  loss cells; then sync-reduction iterations (merge SYNC1 into P3-end
  handoff; single-level wide-hist P4 over candidate range leaning on
  p4_exact_tail for boundary; distribute only O(cand) work).
- Later: BS 256-1024 flash-512k mechanism (re-check at current head first —
  may be vseed-repaired or R0 low-hit fallback scans).

## Results so far (2026-07-20, b200-028, same-node paired)

- **T0 baseline @e6fdbfac3d DONE: composite gm 0.8664** (132 cells, 0 inexact,
  results/baseline). Per-BS: 1-8 → 0.68-0.71; 16 → 0.82; 32/64/128 →
  1.17/1.17/1.09 (wins); 256/512/1024 → 0.91/0.89/0.88.
  flash-1M big-BS repaired vs OLD head (0.77→1.01); **flash-512k big-BS
  collapse PERSISTS (0.48-0.53 @BS≥256)** — separate mechanism, task.
  Composite arithmetic: BS≤8→0.95 (+11%) & BS16→1.0 (+1.8%) & bigBS→0.95
  (+2%) ⇒ ≈1.02. The BS 32-128 win block is the asset to protect.
- **L1 forced-cs probe FALSIFIED** (results/l1probe): cs2/cs4 at N=32771 rung
  loses at every BS (BS=1 0.92-0.98, BS≥64 0.27-0.63). pick_config cs=1 gate
  is correct for the CURRENT cluster path; clustering the 32K rung only
  becomes viable if dp4-v2 cuts the cluster sync tax. Do not re-run as a
  pure launch-policy change.

## Measurement discipline

nsys cold-L2 only (no CUDA-event verdicts); ship verdicts ≤2-way concurrent;
A/B arms paired same-run same-GPU; sglang timed by us_span (PDL 2-kernel);
anchors vs 07-20 §9b canonical (b200-027) med ≤1.05; never resume a
partially-errored nsys batch (rep overwrite); env -u GITHUB_TOKEN -u HF_TOKEN;
*.sqlite/*.nsys-rep never committed.
