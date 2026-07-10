# op27 RESUME — CAMPAIGN SHIPPED 2026-07-10 ~05:3xZ (b200-027)

ITER3 IN FLIGHT (post-ship regression fixes, user: run to completion):
- Confirmed regressions vs op25: (A) worst bf16 K2048 65K BS64-2048 +262K
  BS1024, tail 8-13% slower same-node (mechanism: 16-bit quantization widens
  the tail-column band; wide-band sandwich > stock all_ge->dist fallback);
  (B) real 16-bit 65K BS1 -6% (msc; rank-space coverage hole (0.048,0.45)).
  Noise class: ALL K512/K1024 cells (bit-identical, pooled med 1.0002);
  worst fp32 512K BS2/4 refuted same-node (0.998/1.005).
- OP27_R2 knob in gvr_ms (R=2 + bAcc=4096, K2048-only, ms path; msc asserts
  R==1). 5-arm matrix ab_iter3.py on GPU1: notail/tail/tail_r2/mid/mid_r2
  (mid = OP25_QFRACS=0.75,0.5,0.048). Parse = same NVTX c|arm|cell pattern.
- After verdict: ship config -> gates (gate_exact_k2048 + replay) -> re-sweep
  arm via launch_op27_027.sh with NEW OUT root op27027b (edit script OUT= and
  updater OP27_ROOT default) -> rerun update_report_op27.py -> verify -> commit.

PREVIOUS SHIP STATE: op27_hls backfilled into op22 REPORT.html (update_report_op27.py,
last-writer over mc/op25/radix/op26/op27; anchor drift med 1.0022 p90 1.0403;
exactness 414/414). K2048 vs base (seqlen BS=1 gm): worst 1.146->1.437 fp32 /
1.085->1.243 bf16 / 1.140->1.377 fp16; real/best unchanged (bs-grid op25/op27
0.991-0.996); K512/K1024 bit-identical (op25/op27 gm 1.0028). 16-bit real
BS=1 large-N same-node re-check: bf16 0.993 / fp16 0.997 (CSV per-cell dips
= double-anchor-transfer noise). Ship config: OP27_K2048_TAIL default-ON in
gvr_ms_op.py. Historical checkpoint below.

READ FIRST: PLAN.md (campaign design + ITER0 falsification), COST.md (token/$
per phase). User directives: (a) NO data-dependent dispatch — algorithm/param
fixes only (shape-keyed N/K gating like existing _config() IS allowed);
(b) zero regression on op25-won cells; (c) final deliverable = HLS-op27 arm
backfilled into op22_temporal_fixed_hr_bench/REPORT.html with byte-identical
op22rr bundles + anchor-transfer (clone update_report_op25.py pattern);
(d) keep COST.md and this checkpoint current.

## State

- ITER0 (host replay, screen_mprobe.py + results_screen.json): M-ary all_ge
  probe FALSIFIED for K512/K1024 (w3a 0.048 col already brackets worst pole;
  fb = 1 pass). K2048 worst = all_ge all N (stock ladder), host fb also 1 pass.
- Loss relocation: op25 small-N tax is DATA-INDEPENDENT (loses on BEST too:
  base/op25 0.87-0.97 at K512 4-16K vs hls 1.04-1.15); REAL small-N: HLS
  family BEATS base 1.0-1.7x and op25 < hls by 8-18% (same tax) => fix = cut
  the tax, keep w3a coverage + big-N wins; real/best/worst all improve.
- Optimization points ledger (user's 1-6):
  1 all_ge M-probe: FALSIFIED (K512/1024); K2048 covered by tail_s1 arm test.
  2 N-adaptive ladder width: pending iter1 (only if w3a tax is real).
  3 P2 M-ary refine: DO NOT DO (Opt-F/op8 wash; logfalsi 1.00 pass).
  4 fp16 native compare: ALREADY SHIPPED (iter9 p2_native, line ~301).
  5 smem_ptcnt guard: deprioritized (noise-level gain, code-mass risk).
  6a ms-dist extension: N/A for BS=1 small-N (single CTA, no cluster).
  6b Step3 low-side placement: iter2 candidate only if placement matters.
  6c K2048 tail column: IN TEST as tail_s1 (0.75,0.45,0.048).

## Running

- iter1 decomposition A/B: setsid ./drive_ab_decomp.sh (GPU0, started
  03:30:09Z). Arms K512/1024: plain/stock_s1/w3a_s1/ship; K2048:
  plain/stock_s1/ship/tail_s1. best/worst/real x fp32 BS=1 x 15 cells.
  Progress file: results/nsys/ab_decomp/ab_<scen>_fp32.out ("done" lines;
  "AB BATCH DONE" = clean). Driver log drive_ab_decomp.log ends with
  "drive_ab_decomp done" when all 3 scenarios finish.
  If dead: re-run `cd op27_hls_allge_probe && setsid ./drive_ab_decomp.sh
  > drive_ab_decomp.log 2>&1 &` (per-scenario .done skip; delete stale
  .done to redo). TaskStop cannot kill it — use pkill on ab_decomp.py +
  nsys, then verify no respawn (env_taskstop_orphan_drivers).

## iter1 partial verdict (best scenario done, 15/15 exact ok)

- w3a tax on BEST is REAL and owned by the LADDER TABLE: w3a_s1/stock_s1 gm
  1.098 (K512) / 1.058 (K1024); peaks 1.13-1.26 at K512 16K/32K — too big for
  compare cost => BAND/COLLECT GEOMETRY tax (w3a column spread changes the
  sandwich band), not tau(M).
- slot2 is a GAIN, not a tax: ship/w3a_s1 gm 0.938 (K512) / 0.980 (K1024) —
  doubled capacity avoids overflow fallbacks. Op25 memory "slot2 free at
  t=512" vindicated; the op25 small-N regression is the w3a table.
- K2048 tail col ~free on best (gm 0.988, i.e. -1.2%, borderline noise).
- Fix hypothesis: small-N ladder PLACEMENT change (keep 0.92 pair01 col +
  0.048 tail col; move/drop the 0.45 middle col, or M=3 per HLS-math Theorem
  "M*=3 both regimes") — decide after worst+real land.

## Next (after iter1 completes)

1. python3 parse_ab_decomp.py  -> per-cell table + geomean taxes:
   w3a tax = w3a_s1/stock_s1, slot2 tax = ship/w3a_s1 (K512/1024),
   tail gain = stock_s1/tail_s1, slot2 tax = ship/stock_s1 (K2048),
   floor = plain/ship. Exactness must be all-ok.
2. Fix design by verdict (in gvr_ms_op.py, knob-gated OP27_*):
   - slot2 owns tax -> tighten slot_scale gate in _compile (shape-keyed,
     e.g. scale=2 only for 8192<=n<65536 or only t bucket that's free) —
     recheck against the op25 memory claim "free at t=512" (contradicted
     by rr backfill data).
   - w3a owns tax -> per-N qfracs table (keep 0.048 tail col coverage;
     consider dropping only the 0.92 col at small N where pair01 cliff
     doesn't bite, or M=3 ladder at small N per HLS math Theorem M*=3).
   - tail_s1 wins K2048 worst without best/real regression -> add K2048
     ladder table entry.
3. Gates: exactness (all arms x K x dtype at BS=1), then P0 grid
   (op25_hls_expand/drive_p0_ship.sh pattern), then full-dtype A/B.
4. Ship arm: add "op27_hls" to sweep_op22rr.py ARMS_EXTRA (OP27 knobs ON,
   gvr_ms_auto), run OP22RR_ARMS="gvr_cutedsl,op27_hls" 81-cell sweep via
   drive_nsys_op22rr.sh on 027, anchor-transfer into REPORT.html by
   cloning op22_temporal_fixed_hr_bench/update_report_op25.py ->
   update_report_op27.py (must re-derive prior backfills; run LAST after
   update_report_radix.py convention — check its self-contained variant).
5. Update COST.md phase rows + this checkpoint at every milestone.

## Env notes

- 027 all 8 GPUs idle at start; iter1 uses GPU0. nsys: env -u GITHUB_TOKEN
  -u HF_TOKEN; *.sqlite/*.nsys-rep gitignored globally.
- Kernel src: op21_gvr_prod/src/gvr_ms_op.py (+gvr_msc_op.py msc path).
  Key lines: _config ~2010 (t/use256 buckets), _compile ~2023 (slot_scale
  n-gate, qfracs default), _qfracs_for ~1997 (OP25_QFRACS env, ship table),
  slot_cap ~295, overflow check ~1846, fallback phase3 override ~1250-1394.
