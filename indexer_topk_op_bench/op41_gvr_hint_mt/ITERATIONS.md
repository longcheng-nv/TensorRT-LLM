# op41 iterations ledger

## phase 0/1 — 2026-07-23 — CLOSED (brief saturated, measured)
Crux 1 (hint_study): rank-hK hint ~= v_K (count 506-512/K on 75/75); 6-rung
order-stat ladder one-pass coverage 75/75 @kC=8K; no fixed rank viable
(h 0.057-0.998, fixed-rank count span 548->189051).
Crux 2 (v3_pass_probe, DBG_PASSES build of r3_v11): stock v3 P2 secant =
0 extra passes + 0 descent on 45/45 hint-path cells — its P1 value-quantile
ladder already one-passes the whole envelope; ladder-quality lever value = 0.
Arm-side adoption blocked by ledger: count-feedback iterate = 2nd DRAM pass
at big N; hint-rank primary w/o feedback = min-hint-blowup domain.
VERDICT in RESULTS.md. Assets: pass-count instrumentation + ladder design
ready if a future envelope shows nonzero pass histograms.

## phase 2 — 2026-07-23 — REOPENED: BS>1 heterogeneous-row verification
User-directed verification of the saturation verdict on hetero batches
(rows = ALL GVR-active layers of each (model,isl), cycled to BS).
FINDING 1 (v3_pass_probe_bs): saturation does NOT hold row-wise on hetero
data — per-row pass histogram over 20400 hint-path rows: p0 18695 / p1 1599 /
p2 106 (8.4% nonzero). Pass count is a LAYER property; the envelope's 3 bench
layers per cell happened to all be p0 (the BS=1 blind spot).
FINDING 2 (straggler_tax, stock v3 paired event axis): straggler layers gate
whole-batch latency where waves don't amortize them: flash_512k 1.10-1.35x
(ONE p2 layer, L10), v32_256k 1.07-1.13, v32_128k up to 1.10, pro_1024k up to
1.12. Noise floor +-3% self-calibrated by all-p0 groups (identical A/B sets).
FINDING 3 (hint_study_all_layers): exact 8-rank order-stat ladder one-passes
542/545 layers (misses = 3 kC-window straddles incl. pro_1024k_L10; secant
fallback covers). Stock v3's bin-edge value-quantile ladder: 91.7% row-wise.
DESIGN (in simulation): keep stage1 trim + stage2 64-bin, ADD stage-3 single
extra pass refining ALL rungs to 32 sub-bins (register hints only, per-rung
warp-parallel scan, ~0.2-0.4us, zero memory traffic) — 32x finer rungs.
sim_stage3.py quantifies coverage before any kernel surgery.

## phase 3 — 2026-07-23 — SHIP (v3mt: per-K rung fractions, zero P2 cost)
Design path: sim_stage3 exonerated bin quantization (offline AR8/HS1 stage-2
replica already 544/545) -> real culprit = SPARSE ladders at BS>1 dispatch
tiers (AR4/HS2-4). sim_dispatch_fracs/2 (dispatch-exact per-group sim) found
per-K optima; no global frac set dominates (h-distribution tracks K/model).
FALSIFIED: (single global AR4 frac set; domain: all-K hetero batches;
evidence: a4_5085 wins v32 but flash 20->16/21, a4_3070 vice versa).
Ship: AR4 K2048->{55,88} gated npad<49152||>98304 (measured exception:
npad~65600 K2048 loses with ALL shifted fracs at BS>=256, stock-frac control
1.00 — frac_scan_64k), K1024->{35,70}, K512->stock {25,65}; AR6 K2048->
{25,50,75,92}, else stock. Constants only; rung count unchanged.
VERDICT (ab_v3mt v2): gate 0 fails (75x3 replicated + hetero per-row);
hetero tax axis 32/32 >= 1.00: v32_256k 1.10-1.18, v32_16k 1.10-1.16,
pro_512k 1.13-1.15, pro_256k 1.18@BS256, pro_1024k 1.07@BS1024, flash ~1.00
(its lone L10 straggler is a kC-window straddle, unfixable by ladder);
replicated REG axis: no cell <0.97, positive outliers up to pro_64k 1.20.
Tradeoff logged: npad gate reverts v32_64k BS>=256 replicated-L54 win (1.22)
to protect the hetero mixture — hetero is the production-realistic axis.
Follow-up (optional): swap v3mt into the op39 combined dispatch and re-run
the 750-cell envelope (expect small positive drift; bench layers mostly p0).

## phase 4 — 2026-07-23 — option-1 executed: v3mt into the op39 combined dispatch
750-cell envelope re-run (bs41_nsys 8-GPU sharded, op38 protocol verbatim,
0/750 inexact): v3mt alone gm 1.3064 / min 0.8367 / 104 losers (v3 record:
1.2928 / 0.6525 / 115) — the worst-case floor rises 28% (straggler bench
layers fixed). NEW COMBINED RECORD: BEST(arm_e6, v3mt) = gm 1.3279 /
mean 1.3655 / min 0.8367 / <1.0 82 (e6 record 1.3179 / 1.3564 / 0.7665 / 90).
Harvest curve: e1 1.3049 -> e2 1.3136 -> e5 1.3150 -> e6 1.3179 -> 1.3279.
Triple-dispatch projection (arm+v3+v3mt) 1.3336 — rejected as ship shape
(two v3 variants for +0.4%; the 36 v3mt<v3 cases are frac-tuning residue).
Remaining named lever: per-case chase of those 36 cells (diminishing).

## phase 5 — 2026-07-23 — option-2 (upstream port): NO-PORT verdict + one gated candidate
Target = production cuteDSL kernel (PR#16457 e612 baseline via op40 gvrpkg40b;
PR branch is read-only from this machine). Its r0_qfracs ARE per-K CCDF-rank
fractions (same semantics as v3's qt) — K2048 default (0.6,0.35)+vseed already
recalibrated 2026-07-19 in the same direction op41 found.
PHASE A (upstream_hetero_probe): production kernel largely ABSORBS row
heterogeneity — BS>=64 tax 0.94-1.03 (vseed adaptive rung + cheap falsi pass;
hetero often FASTER than replicated mean); pockets only at BS16 (flash_256k
1.23, v32_64k 1.16-1.18, pro_256k 1.16).
PHASE B/C (qfrac sweeps, paired, per-row exact):
FALSIFIED: (K512 (0.65,0.25) M=2 port; domain: flash all-npad; evidence:
flash_256k clean win was a pocket — 128k 0.91-0.94, 1024k 0.93, 512k mixed).
FALSIFIED: (K1024 any move; m1lo wash 0.98-1.00, M=2 v3port 0.95 loses the
count column). K2048 (0.9,0.6): NOT Pareto as default (v32_32k BS>=64 -3-5%)
BUT clean at BS16: 5/5 v32 groups 1.04-1.21, zero losers.
VERDICT: no unconditional default change ships — upstream vseed already does
per-row distribution adaptation (the thing v3 lacked). ONE candidate remains:
K2048 && BS<=16 gated qfracs (0.9,0.6) via the pick_config surface (+4-21%);
needs user approval + PR#16457 merge first (one-concern-per-PR; the branch is
another session's). Evidence: results/upstream_{hetero,qfrac_sweep,qfrac_ext}.csv.
