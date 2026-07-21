# L-J tight_bracket uniform ~30% tax — phase-level diagnosis

Date: 2026-07-21, node umbriel-b200-027, GPU 0 (CUDA_VISIBLE_DEVICES=0),
branch omni/op21-gvr-prod.

Context: L-J nsys verdict (results/lj) = NET LOSS, gvr_pr/gvr_lj composite
0.7221, uniform ~0.65-0.80 tax on ALL warm rungs/BS; only win = cold
flash-512k (hit .057) 1.26-1.33x. This note locates the tax with a
clock64-instrumented clone of gvrpkg37 and two one-flag ctor ablations.

DISCIPLINE: all clock64 numbers below are QUALITATIVE FRACTIONS anchored to
the untimed arm's wall (timed-arm overhead 0-10%, see log). No ship verdict
is claimed from them; any fix must re-run the nsys 132-cell A/B.

## Method

- `variant/gvrpkgtimed37` = `variant/gvrpkg37` + the exact `[ptime]` 8-slot
  clock64 stamps of `p4f1_harness/gvrpkgtimed` (leader CTA, thread 0;
  phase_ts (BS, 8)). Applied by anchor-based insertion (patch-with-fuzz
  mislanded and was discarded).
- `src/measure_phases_lj.py` (clone of measure_phases_bs.py): per cell runs
  arms back-to-back same-GPU, each arm = untimed gvrpkg37 anchor + timed
  clone. Warm 10 / reps 30, 512MB L2 evict per rep, median.
- Cells (L-J verdict representatives): pro/128k L30 BS8 (N=32771 cs4 T512,
  hit .326), pro/512k L30 BS8 (N=131075 cs4 T1024, hit .230), flash/512k
  L22 BS512 (N=131075 cs1 T512, hit .057 — the WIN cell), flash/128k L22
  BS2 (N=32771 cs8 T512, hit .701). All arms exact on all cells.

## Result — per-phase wall-anchored us (tb=off vs tb=on) and cyc ratio

results/phase_lj.csv, logs/phase_lj_full.log. Wall ratio = off/on
(>1 = tb wins); phase cyc ratio = on/off (>1 = tb slower).

| cell (hit) | wall off→on (ratio) | P2 count | P3 collect | P1b rungs | P4 select | P1 gather |
|---|---|---|---|---|---|---|
| pro/128k BS8 (.326)   | 19.46→23.55 (0.826) | 4.32→8.22u **2.00x** | 2.27→3.49u **1.62x** | 0.84→1.31u 1.65x | 9.31→7.96u 0.90x | 1.01x |
| pro/512k BS8 (.230)   | 23.55→31.57 (0.746) | 5.24→12.25u **2.54x** | 3.97→5.47u **1.50x** | 0.64→1.01u 1.70x | 10.78→10.30u 1.04x | 0.94x |
| flash/512k BS512 (.057)| 220.1→179.2 (**1.228 WIN**) | 159.5→96.5u **0.50x** | 41.7→60.8u **1.21x** | 0.85x | 0.97x | 0.98x |
| flash/128k BS2 (.701) | 17.44→21.50 (0.811) | 3.37→7.08u **2.44x** | 1.51→2.09u **1.61x** | 0.55→0.92u 1.95x | 9.54→9.28u 1.13x | 1.01x |

Tax decomposition (warm cells): P2 +3.7-7.0us is 65-85% of the net loss;
P3 +0.6-1.5us is the rest; P1b +0.4-0.5us small; P4 band diet returns
~0 to -1.4us (nothing at hit .23-.70 — cnt_hi is small so need≈K and
band≈full admission). Win cell: P2 halves (admission-miss refine killed,
-63us) while P3 still PAYS +19us — the win would be materially larger
without the P3 tax.

## Identified tax locations (code-confirmed)

1. **P2: rung-ladder width M in `block_count_ge_multi`** (dominant).
   tb_qfracs default = 9 rungs + vseed ⇒ M_thr=10 count columns vs base
   2 (K512: (0.85,)+vseed) / 3 (K1024: (0.85,0.35)+vseed). The full-N
   count pass does M predicated `cnt_frag[m] += (v>=thr_frag[m])` per
   element (gvr_topk_decode.py L1449-1455): 10 cmp+add per element turns
   the BW-bound scan issue-bound. Measured slope ≈ 0.55us/column
   (pro/128k: (8.22-4.32)/7) ≈ 2.8% of wall per column — the original
   "-3..-7% per extra column" model was per-COLUMN correct but the ladder
   added 7-8 columns, hence ~20-25% wall. NOT extra DRAM (same one pass
   over the row), NOT occupancy (P1/P4/stage phases are 0.9-1.1x
   unchanged in the same kernel), NOT extra barriers (none added).
2. **P3: `phase3_collect_band` has no 4-way unroll fast path** (v1
   "correctness first", L1985-1986): 1-way vec loop + scalar tail vs the
   base `phase3_collect_candidates` 4-way unrolled fast path
   (enable_phase3_unroll, L1874-1888). A second full-N pass at ~60-70%
   of base speed ⇒ 1.5-1.6x cyc warm, 1.21x on the BS512 BW-rich cell.
3. **P1b: 9-rung extraction M-loop** in the warp-0 bin walk
   (phase1b_hspace_rungs L1076-1080, M compares per bin per lane) —
   1.6-2.0x on a small phase (+0.4-0.5us). Third-order.

Why the tax is FLAT in N / BS / persists in BW-bound cells: all three
locations are per-element costs of the two full-N passes (P2 count, P3
collect) — they scale WITH N and WITH rows, so the wall FRACTION is
constant; and because they add instructions per loaded byte, a
DRAM-saturated cell shifts to issue-limited instead of hiding them.
Hypotheses ruled out by the same data: (a) no extra full-N pass exists
(phase count identical); (b) sure-set writes don't break P3 (win cell P3
only 1.21x with a large sure set); (c) no occupancy/register cliff
(unchanged phases flat ~1.0x, straggle ~1.02); (d) ladder counted once,
not per iteration (P2 cyc 2-2.5x, not Mx); (e) zero added cluster syncs
(confirmed in code; epilogue 0.85-1.0x).

## Ablations (one-flag ctor recompiles, results/phase_lj_ablate.csv)

- `tb_thin`: tight_bracket=True, tb_qfracs=(0.85, 0.35, 0.05) — base-width
  ladder + one deep rung (M_thr=4). Attributes the P2 tax to ladder width
  and probes whether a thin bracket keeps the cold-cell win.
- `off_nou3`: tb=off, enable_phase3_unroll=False — if its P3 cyc ≈ the
  band-collect cyc, the whole P3 delta is the missing unroll, not the
  dual-threshold classify.

RESULTS (filled after run): see logs/phase_lj_ablate.log.

## Verdict — removable vs intrinsic

- **P3 tax: REMOVABLE.** Port the existing enable_phase3_unroll 4-way
  fast path into `phase3_collect_band` (same structure; the two
  classify-compares and dual write pointers live in the unrolled body
  exactly as in the 1-way loop). Expected to recover most of the
  1.5-1.6x and to enlarge the flash-512k win.
- **P1b tax: mostly intrinsic to M rungs, negligible** (≤0.5us); can be
  shaved by hoisting the qneeds compare out of the per-bin loop if ever
  needed.
- **P2 tax: INTRINSIC to ladder width at ~0.5us/column/32K-elems** —
  counting 10 columns in one pass is the tight-bracket idea itself.
  Reduction options (not elimination):
  (a) trim the ladder (9+1 → 4-6 columns) — linear tax cut, looser
      brackets; tb_thin ablation quantifies both sides;
  (b) per-element branchless binary search over the sorted rungs
      (~4 cmp+select) + interval counter, suffix-sum at the end —
      replaces 10 cmp+add with ~5-6 ops, ~2x cut of the extra P2 cost,
      but a rewrite of the count inner loop with register-indexing risk;
  (c) escape-only form: keep the base 2-3 column ladder ALWAYS, and only
      when the base admission MISSES fall into a bracket-refine that
      reuses the L-J dual-threshold P3/P4 (the win cell is exactly the
      admission-miss regime; warm cells never pay). This preserves the
      1.24-1.33x flash-512k win at zero warm tax and zero new syncs —
      in-kernel, no hit-rate dispatch (PLAN.md L-J NEXT note).

- Arithmetic honesty for "fix + re-A/B" of the full tight_bracket=ON
  shape: even with P3 fully repaired and P2 halved via (b), pro/128k
  BS8 would be ~23.55-1.2-1.95 ≈ 20.4us vs 19.46 base — still a ~5%
  warm loss, and the P4 diet pays nothing at warm hit rates because
  cnt_hi ≪ K. ⇒ recommended minimal fix is (c) escape-only + P3 unroll,
  NOT a tuned always-on ladder.
