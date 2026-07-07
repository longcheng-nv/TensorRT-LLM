# op22 mechanism study — why `best` (hr 0.90) makes the GVR family slower

> 2026-07-07, umbriel-b200-049 GPU1 (screening runs; canonical timings are the
> GPU0 nsys grid). Artifacts: `mech_check_iters.py` → `mech_check_iters.jsonl`
> (162 cells), `mech_crossover.py` (2×2 swap). Timing sources quoted below:
> parsed nsys `results_b200_op22/{best,real}/seqlen_sweep/results.jsonl`
> (b200-040) + CUDA-event crossover (b200-049, screening).

## Verdict

The `best`-scenario slowdown is **REAL and data-dependent — not a measurement
artifact — but the original tie-density hypothesis is falsified.** The cost
driver is the number of **extra full-row re-scans** the GVR kernels take, and
that number is a **pure function of the preIdx vector** (threshold-init
quality), not of the logits value distribution.

## Evidence chain

1. **Non-convergence never happens** (`count_gvr_iters` host replay, verified
   faithful to the vendored single-CTA kernel): P2 converges 162/162 cells
   across best/worst/real; p2_iters med 1 / max 7; p4 snap med 2-3 / max 23.
   Tie-dense boundary → refine blowup is dead.

2. **`gvr_cutedsl` obeys a clean linear law in replayed eval count** —
   scenario drops out. K2048 fp32 (nsys cold): N=1M ev1→93.3, ev2→~115,
   ev3→136.7 µs; N=524K best ev4 86.2 vs real ev4 86.3 µs (identical).
   Marginal cost ≈ 22 µs per extra full-row scan at N=1M (~190 GB/s
   single-CTA).

3. **Crossover experiment (decisive)** — time {best,real} logits × {best,real}
   preIdx at N=1M fp32 (event, median of 50 cold):

   | op / cell | lg=best,pi=best | lg=best,pi=real | lg=real,pi=best | lg=real,pi=real |
   |---|---|---|---|---|
   | ms_auto K2048 | 247.8 | **40.7** | 229.1 | **40.5** |
   | cutedsl K2048 | 146.0 | **103.1** | 168.0 | **102.3** |
   | ms_auto K512 | **125.0** | 187.0 | **124.5** | 186.2 |
   | cutedsl K512 | 123.9 | 125.7 | 125.5 | 128.0 |

   Cost follows the **preIdx column only**; swapping the logits row changes
   nothing. Replayed evals track it exactly at K2048 (ev3/ev1/ev4/ev1).

4. **Why high hit rate hurts**: GVR seeds its threshold from
   `mean(logits[preIdx])`. With hr→1 the stash *is* the top-K, so pmean ≈ the
   top-K **median** → initial count ≈ K/2 < K → guaranteed undershoot → 1-3
   refine re-scans. With boundary-distributed misses (`real`, and `worst`
   hr .05 whose miss-depth model clusters misses at the selection boundary),
   pmean lands at ≈ the K-th value → first count already in the [K, kC]
   acceptance band. Replay: `worst` = p2 0 iters / ev1 in **54/54** cells →
   prediction: worst-scenario GVR comes out FAST.
   **Early confirmation (worst seqlen K2048 fp32, b200-049)**: cutedsl N=1M
   94.8 µs ≈ real 93.3 (ev1 both ✓), multicta 37.2 ≈ 34.9 ✓ — but ms_auto
   105 µs vs real 28.6: worst's boundary-flat data gives a WIDE candidate
   band (replay cand=4318 → ≈kC) → the slot-overflow fallback (trigger #2)
   fires even at ev1. Both msc triggers now observed in isolation:
   refine-driven (best K2048), band-driven (worst K2048, real K512).

5. **op21 (`gvr_ms_auto`, msc paths) has a second, sharper cliff**: each
   refine (or ladder miss) triggers the fallback at `gvr_msc_op.py:1096` —
   the leader CTA **recounts the full row single-CTA** (~95 µs/pass at N=1M
   fp32) — hence 236 µs (ev3) vs 28.6 µs (ev1) at the headline cell, worse
   than plain cutedsl when the init is poisoned, despite being 3.3× faster
   when it is not.
   The msc fast path needs the stash-quantile ladder (M_thr=4 rungs) to
   bracket the K-th value with band ≤ kC and no slot overflow
   (`gvr_msc_op.py:993-996`). K512 shows the overflow trigger in isolation:
   all 4 crossover combos replay ev2, but real-preIdx yields cand ≈ 4500
   (> slot capacity) vs 654 for best-preIdx → fallback → 186 vs 124 µs, i.e.
   **not monotone in hr** — it is ladder/band geometry, not hit rate per se.

## Consequences for the report

- Rename the axis in prose: scenarios select **threshold-init quality**, not
  "GVR friendliness". `best`(hr .90) is P2-init-adversarial; `worst`(hr .05,
  boundary misses) is P2-init-friendly. hr → GVR-speed is NON-monotone.
- radix/sglang are hr-insensitive (no preIdx use) → flat across scenarios ✓
  (radix ~20 µs at all N/scenarios in current data).
- op21's hugeN dispatch relies on the msc fast path; its worst case
  (init-poisoned, N≥512K) is ~2× slower than plain cutedsl — a real
  robustness gap to call out (per-refine cost is C×-amplified by the leader
  recount).
- Methodology caveat: synthetic fixed-hr preIdx at hr≥0.9 is a *stress*
  construction — real captures never show pmean at the top-K median because
  misses concentrate at the boundary (cf. reference_gvr_realdata_undershoot).
