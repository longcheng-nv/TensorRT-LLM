# op34 CRUX-C rung-2 proxy — multi-CTA fused collect + rank tail (node 048, cold)

scripts/crux_c_proxy.py. Multi-CTA collect (Triton: per-block tl.cumsum stream-compaction +
1 atomic/block-iter to reserve slots) + torch.topk tail. EXACT on real pro data (vdiff=0) for
oracle (M=1024) AND hintish (M=5121) thresholds, C∈{16,32,64}.

## Fused collect kernel1 ALONE (pure-kernel NCU cold, µs) — the dominant cost
| ISL (N) | C32 oracle | C64 oracle | C32 hintish | C64 hintish | bare-count C64 |
|---|---|---|---|---|---|
| 256k  (65539)  | 17.34 | **14.40** | 17.15 | 14.24 | 6.75 |
| 1024k (262127) | 33.41 | **20.96** | 32.99 | 20.61 | 9.28 |

- Collect ≈ 2× bare count (Triton's block-wide cumsum + scatter is heavy; a CuTe warp-ballot+popc
  compaction — op#7 lineage — would land far closer to bare count). Threshold-INsensitive (oracle
  vs hintish within noise): the cost is the full-N SCAN, not the # written. ⇒ append cost is a
  fixed ~2× tax in Triton, reducible in CuTe.

## Budget vs sglang (large N, C=64)
| ISL | sglang | goal ≤sgl/1.3 | collect(C64) | tail budget left |
|---|---|---|---|---|
| 256k  | 28.16 | 21.66 | 14.40 | **7.3 µs** |
| 1024k | 39.04 | 30.03 | 20.96 | **9.1 µs** |
A lean single-CTA rank-scatter on ≤kC candidates (op#7 P4 ~5–8µs) FITS the tail budget at both
large-N cells ⇒ **multi-CTA GVR beats sglang by ~30% at ISL≥256k**. GO to build the real kernel.
(NOTE: NCU-sum of collect+torch.topk over-counted the tail 60–90µs because --cache-control all
flushes L2 between torch.topk's 6 sub-kernels — an NCU-multi-kernel artifact; the REAL back-to-back
tail (warm L2) is a few µs. Use nsys or a lean single-kernel tail, never NCU-sum a torch pipeline.)

## Verdict: GO (rung-2 passed)
Large-N (ISL 256k/1024k) is winnable within budget even with the heavy-Triton collect. Small/mid N
expected walled (op32 floor + merge overhead). Build a 2-kernel Triton multi-CTA top-K with a LEAN
exact rank-scatter tail (kernel2), gate 3-track, nsys pure-kernel A/B across the full ISL grid vs
sglang + op26_r0. This proves the algorithm (GVR threshold skeleton, parallelized) and sets up the
CuTe single-launch productization. Then measure grand geomean + win region → dispatch → report.
