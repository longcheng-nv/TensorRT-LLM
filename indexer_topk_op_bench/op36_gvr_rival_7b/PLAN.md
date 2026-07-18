# op36 — GVR vs sglang_v2 on the real §7b axis (BS × ISL, decode-capture)

Campaign dir: `indexer_topk_op_bench/op36_gvr_rival_7b/`. Node: umbriel-b200-047
(8× B200). Base: PR#16457 shipped HEAD `eae374554c`
(worktree `../TensorRT-LLM-gvr-r0`, branch `perf/gvr-topk-r0-histogram-ladder`).

## Goal (user-stated) and measured feasibility

**User target**: optimized GVR ≥ **1.10× geomean over sglang_v2** on the op26
report §7b/§8 REAL decode-capture axis (V4 Flash/Pro + V3.2, BS 1-1024 × ISL
4k-1M, fp32 = the sglang-comparable subset, 275 cells, us_span, nsys cold-L2).

**Baseline arithmetic** (analysis/baseline_7b.py, from the 07-16 b200-081
backfilled grid embedded in the op26 REPORT):

| position | gm vs sglang_v2 |
|---|---|
| A. gvr_pr as-is | **0.745** |
| B. pr × 1.25 uniform lever stack (op35 realistic ceiling) | 0.931 |
| C. best exact dispatch (pr/op26/radix/FI per cell) | 0.827 |
| D. C + sglang-PARITY path at every lost cell | 1.030 |
| E. D + 25% levers on won cells | 1.069 |

By band: ISL 4-16k pr 0.599 (99 cells, the hole) / 32-128k 0.793 / 256k-1M 0.910.

**Feasibility verdict, stated up front**: 1.10 is NOT reachable from levers
2/3/5/6 alone (ceiling ≈0.93), nor from levers + perfect dispatch + parity
(≈1.07). Crossing 1.10 requires beating sglang per-cell on its home turf —
triple-falsified (op34 +30% BS=1, op35 +40%, op35-apex +50% own-build gm 0.51).
The campaign therefore runs a **feasibility pivot gate** (below) instead of
silently chasing an unreachable number.

## Tracks

- **A0 bundle-v2 harvest** (validated): skip_h1 + kb512@K2048 from
  `op35_gvr_round2/variant/gvrpkg35/` → arm `gvr_a0`. Expected full-77 1.039,
  K2048 1.133. First silicon step; also revalidates harness on this node.
- **A1 admission escape** (variance cap, lever 5): R0 already counts seed
  admissions; cold-seed signal → in-kernel bail to secant/hint-free path
  instead of the fallback ladder. Targets the disclosed low-hit regression
  (flash 1024k BS≥128 0.68-0.79× vs base; v32 256k 0.75-0.87). Red line: NO
  host-side hit dispatch (hit unknowable at inference). Lagged-hit feedback
  (kernel emits counter, host uses prev step) allowed as second form.
- **A2 B1 tile rung-class sideband + distP4** (lever 3 + biggest known lever):
  P2 counting pass emits u8/tile rung class → P3 skips whole tiles (exact).
  Account INSTRUCTION-side (issue-bound 81-84%), never bytes. distP4 = kill
  handoff2 value-ship + parallelize leader P4 (P4blk med 37%, zero-P4blk UB
  1.578).
- **A3 multi-CTA C>8 at N≥512k** (lever 6): count-scan scales C8→C64 = 3.7×
  @262k (op34 CRUX-A); 147 idle SMs at BS=1. Shape-keyed dispatch only.
- **B small-N structural path**: ISL 4-16k = 99/275 cells gm 0.599; GVR 1-CTA
  skeleton floor ~9.7µs vs sglang 4.7-6.7µs (8-CTA MLP). Only route to parity+
  = 8-CTA fused histogram+tie exact path dispatched at small N, with the tie
  overflow defect FIXED (unconditional exactness = our moat; sglang caps
  kMaxNumTie=2048, real V3.2 L52 already fails it). Build decision: port+fix
  beats write-from-scratch (apex lesson).

## Feasibility pivot gate (mandatory)

After A0-A3 + B have first nsys verdicts: recompute composite real-§7b gm vs
sglang. If < 1.10 (expected per arithmetic), STOP and present the measured
ceiling + options to the user (accept ~1.0-1.07 composite with exactness moat;
or redefine target axis). Do NOT iterate blindly past the gate.

## Measurement discipline (op35 lessons, binding)

- nsys cold-L2 canonical axis only; CUDA-event cross-run diffs INVALID.
- 8-GPU parallelism for SCREENING shards only; **ship verdicts re-run at ≤2
  concurrent nsys on an idle node** (8-concurrent fabricates ±15% outliers in
  both directions).
- Exactness gates on every variant: harness folded exact check + (before any
  ship claim) the op26 full-grid battery. sglang cells use us_span (PDL).
- Anchor: every batch carries gvr_pr + sglang_v2 re-measured same-process;
  cross-node claims only via anchor transfer with per-batch p95 drift check.
- Long runs under setsid (TaskStop can't kill process trees); background via
  run_in_background (sandbox kills nohup&, exit 144).
- profiling with `env -u GITHUB_TOKEN -u HF_TOKEN`; never commit *.sqlite/*.nsys-rep.

## Red lines (falsification history — never retry)

- Host dispatch on hit-rate (unknowable at inference).
- h-tracked cross-step placement (superseded by within-step P1b histogram).
- Beating sglang at BS=1 fp32 within the GVR 1-CTA skeleton (op34 double lock:
  oracle C=64 collect-only ≈ sglang whole kernel).
- p4_fused_hist, global kNumBins=512, kNumBins=256 (OOB=UB), launch-cfg
  tuning (>2.5%), literal whole-window B1, K2048 tail ladder on PR (vseed
  covers it), hint inside sglang skeleton (4-8× slower).

## Harness

Clone of `op26_r0_upstream_port_report/rival_harness/` (nsys batch sweep,
NVTX c|/w| ranges, cold-L2 512MB evict, cell-resumable jsonl) with new arms
registered in `ops_rival.py` style: gvr_a0/a1/a2/a3/bpath. Real bundles via
`real_data_v4cap.py` / `real_data_v32.py` (op22 loaders). Baseline CSV:
`results/baseline_real_bs.csv` (825 rows incl. 16-bit for context).
