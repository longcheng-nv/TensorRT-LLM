# compB BS>1 extension — minimal validation (A + single-wave B)

2026-07-22, umbriel-b200-039 (8xB200, all idle), GPUs 1-6, one cell per GPU.
Canonical timing = nsys cold-L2 kernel-sum (20 cold + 50 warm NVTX reps,
512 MB evict outside range), all 3 arms paired back-to-back per cell:
`gvr_pr` (REPORT-verbatim PR#16457 local anchor), `kf_compB` (shipped compB,
BS sequential launches), `kf_compB_ext` (this extension, one call per batch).
Data = real decode captures, same row replicated to BS (== §7.8 protocol).

**Exactness: 126/126 records exact (rows {0, BS/2, BS-1} tie-robust value-set
check). Verdict: BOTH hypotheses CONFIRMED.**

## A — small/mid-n tiers batched via grid.y (n <= 16896)

4 cells (flash_16k N=4099 small<6>, pro_16k N=4099 small<6>, v32_8k N=8207
K=2048 mid<2>, flash_64k N=16387 mid<4>), BS 1..1024:

- ext time is FLAT to BS=64 (5.8 -> 6.1 us on flash_16k; one CTA per row,
  148 SMs) and grows sub-linearly after (BS=1024: 36-58 us).
- vs sequential compB: 1.9x @BS2 ... 157x @BS1024.
- vs gvr_pr: **1.61-1.80x at EVERY BS** (pooled gm 1.61-1.72 per BS point,
  16/16 wins per BS row incl. BS=1024). The BS=1 win region extends to the
  whole BS axis for small-n rows, immediately.

## B — large-n single-wave row teams (N=131075, team=65 CTAs/row)

2 cells (flash_512k K=512, pro_512k K=1024), BS 1..16. Occupancy cap
measured 296 blocks (active=2/SM, register-bound; smem 40KB would allow 5)
-> rows_per_wave = 4.

| BS | waves | ext/seq (fl/pro) | ext/gvr (fl/pro) | note |
|---|---|---|---|---|
| 1 | 1 | 1.006 / 1.016 | 2.57 / 1.95 | parity: team=65 (no bump-to-148) == shipped grid |
| 2 | 1 | 1.96 / 1.97 | 2.55 / 1.89 | **single wave ~= free** |
| 4 | 1 | 3.74 / 3.61 | 2.49 / 1.81 | 9.7us @BS4 ~= 9.2us @BS1 |
| 8 | 2 | 3.66 / 3.59 | 1.25 / 0.92 | linear in waves, crossover vs PR arm |
| 16 | 4 | 3.68 / 3.56 | 0.65 / 0.48 | beyond capacity -> batched arm wins |

- **Core hypothesis confirmed: within one co-resident wave, extra rows are
  ~free** (BS=4 costs +5% over BS=1). Multi-wave time = waves x single-wave
  (chunked launches, span/kernel tax <= 1.02 everywhere).
- BS=1 parity kills the residual worry that the shipped `blocks =
  max(team, 148)` bump mattered: team-only grid is timing-identical.
- The §7.8 collapse is fully reversed inside the wave-capacity window:
  BS=4 was 0.470x vs PR (sequential), now 2.49x/1.81x.

## Dispatch gate implied (replaces the BS==1 hard gate)

- n <= 16896: batched ext path for ANY BS (uniform win).
- large-n: ext while `BS <= rows_per_wave` (= cap/team, here 4); beyond
  ~2 waves the batched PR arm wins -> fall back (crossover BS ~= 8 at
  N=131K, K-dependent: pro crosses earlier than flash).
- Next lever for B: raise `active` from 2 (register diet on topk_fast, or
  a half-team variant with 2 elts/thread reload) -> rows_per_wave 4 -> 8+,
  pushing the crossover past the production-relevant BS range.

Files: kernel_ext.cu (surgical edits via apply_ext_edits.py; aefm namespace
only, v30 range falls back to sequential), main_ext.cpp, bs_ext.py,
drive_bs_ext.sh, parse_bs_ext.py -> ext_bs.csv. First launch of the driver
was invalidated (missing `grep -E '^(flash|pro|v32) '` list filter -> 3
shards ran the same cell and overwrote one nsys rep; quarantined in
contaminated_run1/, full re-run clean).
