# op34 FALSIFIED — seeded from the master GVR falsification ledger (2026-07-14)

Checkpoint rule (Phase 2.5): grep this + WALLS.md before implementing ANY hypothesis.
On a hit: cite the revival condition or drop. Scoped triples (conclusion, domain, evidence).

## Imported red-lines (GVR June+op20-33 record; do NOT re-propose without revival)
| Hypothesis | Verdict | Domain | Evidence | Root cause | Revival |
|---|---|---|---|---|---|
| Fuse P3 collect into count (Opt-L single-scan) | FALSIFIED | all | nsys+FORCE_HAPPY | complexity-backfire (ballot+popc+shfl+atomicAdd ≈ full pass) | cheaper append mech OR a DIFFERENT saving (P4 shrink / issue cut) not measured by Opt-L |
| Model-driven / robust P1 seed (self-loop v1-3) | FALSIFIED | 91k cells | host+silicon | structural (drift ~symmetric, median≈0) | none at P1 |
| 2-way / M-ary multi-threshold refine (Opt-F) | FALSIFIED/WASH | all | nsys | complexity (secant already ~1.46 iter) | — |
| Cluster DSM at high BS (Opt-B/Q5f) | FALSIFIED | BS>~SMs | nsys 0.36-0.45x | structural (GPC wave-cap) | — |
| smem row-residency (op15) | FALSIFIED | BS=1 N<=262K | nsys+warm-L2 | structural (re-reads L2-hot) | — |
| P4-internal reseed/fine-hist/interp | FALSIFIED | all | nsys | complexity (P4 barrier-bound) | — |
| bf16 low-precision passes (op#8) | FALSIFIED | all | nsys | convert = real extra HBM write | smem_cache free convert |
| GVR-turbo >=1.5x vs radix (op#8) | INFEASIBLE | BS=1 all | nsys+NCU, 4 levers | structural (2.5 pass single-CTA vs 1 pass multi-CTA) | change skeleton (excluded) |
| HBE 1-pass fusion when BS*N*4B<~1.5xL2 (op29) | FALSIFIED | fp32 envelope | nsys | structural (2nd pass L2-hot, saved pass never DRAM) | none cold-L2 |
| HBE fused pass at N<=32768 (op29) | FALSIFIED | fp32 N<=32K | nsys | complexity (fixed phases don't amortize short) | REVIVED N>=65536 (cut fixed costs) 1.03-1.13 |
| Hint on sglang cluster (HBE-C, op31) | NO-SHIP in-envelope | N<=256K | nsys | conditional (wins only N>=524288) | envelope-external only |
| hint-quantile columns at K=2048 (op31) | FALSIFIED | V3.2 marginals | crux+nsys | structural (lands ~2000 bins low) | better K2048 hint statistic |
| event-axis ship claims | ARTIFACT | all | protocol | measurement (>=5 fabricated wins) | nsys only |

## op34 own falsifications (append as they occur)
- (kC-diet / P4-shrink / fast-write certain-winners as a path to 2x, {BS=1 fp32 real
  v4cap}, nsys rung-2 kcprobe) FALSIFIED — P4 is a SMALL fraction at BS=1; the two full-N
  scan passes (R0 count + P3 collect, single-CTA 1/148 SM) dominate. Shrinking the candidate
  cap yields <=14% (small N) to NEGATIVE (large N: small kC triggers fb_fix extra passes).
  Data: results/kcprobe. kC-diet stays a known ~4-14% small-N lever, NOT a 2x path. Revival:
  none for the P4 axis at BS=1; the live sub-lever is PASS-FUSION (remove a full-N read).
- (PASS-FUSION / single-scan collect-into-count as a 30% path, {BS=1 fp32 real v4cap COLD-L2},
  NCU iter2b) FALSIFIED on mechanism. NCU CRUX (analysis/NCU_CRUX_048.md): under cold-L2 BOTH
  op26_r0 and sglang read the row exactly ONCE from HBM (~360KB=1x row); GVR's 2nd pass (P3
  collect) is L2-HOT, so fusing it away saves only an L2 read = op29's measured 1.03-1.13x, NOT
  a 2x. The wall is NOT pass count. BOTH kernels sit at <0.2% DRAM AND <1% SM peak = LATENCY-bound;
  sglang's only edge is 8-CTA MLP (8x outstanding loads). Revival: none for pass-fusion; the live
  lever is MLP (multi-CTA-per-row >8, or intra-CTA sw-pipeline).
- (MULTI-CTA single-pass GVR (hint threshold) beats sglang, {BS=1 fp32 real v4cap large-N},
  nsys iter4) FALSIFIED. op34_mcta = 4–8× SLOWER (76–125µs vs sglang 12–19µs). t=hint.min is
  exact but admits M=16K–100K candidates on real data (weak hint at hit_rate<1) → collect
  degenerates to full scan + heavy tail; a tighter hint-quantile rung MISSES exactness (count<K).
  Data: results/harvest_pro. Revival: none — see the UB lock below.
- (ANY GVR-skeleton kernel beats sglang by 30% at BS=1, {fp32 real v4cap, all N}, nsys+UB iter4)
  INFEASIBLE, double-locked. UB probe = oracle-threshold multi-CTA collect-only at C=64 (impossible
  best case) = 16–17µs @1024k / 12µs @256k ≈ sglang's ENTIRE kernel (12–19µs). The mandatory exact
  rank tail then exceeds sglang ⇒ parity is the ceiling, never sglang/1.30. Root cause: cold-L2 both
  do 1 HBM read (latency-bound); sglang's saved-pass advantage is L2-hot (cheap) while GVR pays
  threshold-safety + collect/rank phase-separation that sglang fuses. Domain {BS=1}; revival =
  change skeleton (excluded) OR BS>1 (different MLP calculus, out of scope). analysis/DOUBLE_LOCK_048.md.

