# ITERATIONS.md — op42 GVR BS=1-1024 (base 28dc11f6, incumbent gvrpkg_04a0)

## iter 0 — 2026-07-24 — SETUP + node anchor + measurement artifact
Hypothesis: n/a (P0). Chain bring-up on umbriel-b200-073 (4x B200 idle).
Result: base 28dc11f6 copy compiles + exact on real cells; BS=1 event-axis
anchor flash_128k_L36 bsx/head = 1.62-1.66 (record: gm 1.65) — direction ok.
ARTIFACT FOUND: head (cuteDSL) event-axis numbers include HOST issue latency
— pro_1024k BS<=64 events read ~1.2ms while nsys pure-kernel shows 24us
(matches op37 b200-026 anchor 23.5us). Reproduced on GPU0+GPU1, all layers.
Mechanism: mCTA cluster variant host-side launch cost; BS>=128 variant ~92us
events. VERDICT: event axis BANNED for any ratio involving the head arm;
smoke mode only for bsx-vs-bsx iteration. nsys = only A/B axis (matches
production CUDA-graph deployment where host cost is hidden).
Ledger write-back: FALSIFIED.md M-entry added.
Next: iter 1 nsys screen.

## iter 1 — 2026-07-24 — IN PROGRESS (nsys screen running)
Hypothesis (ledger: FALSIFIED #1 revival = kernel-side row parallelism):
minimal grid.y batching — one cluster per row along grid.y, single launch,
per-row pointer offsets; per-tier CS/TB/MAXV unchanged.
Probe (rung 2, event axis, bsx arm only — valid intra-arm):
  flash_128k_L36 (CS8 tier): BS1-8 flat ~14.4us (seq was 73us @BS8);
  BS64 35.6us; BS1024 383us (seq was 8347us — 22x better, still wave-bound:
  4096 CTAs / 148 SMs = 28 waves x ~13us).
  flash_4k_L36 (direct, 1 CTA/row): wins at every BS (events; nsys pending).
Exactness: 0 inexact on all smokes so far (value-multiset vs torch.topk).
Structure identified for BS>=64: __launch_bounds__(TB,1) + CS CTAs/row =>
waves = BS*CS/148; per-wave ~= BS1 latency. Lever for iter2: BS-aware tier
re-dispatch (minimize CTAs/row at large BS: CS=1 dense tiers w/ TB=1024 or
larger MAXV; K-aware). nsys 12-cell x 11-BS screen sharded on GPU0-3.

## iter 2 — 2026-07-24 — PARTIAL (kept for BS 8-31 band)
Hypothesis: dense min-CS tiers amortize per-wave fixed cost. Result (nsys, 7
cells): BS>=32 lifted (0.23->0.40 @1024) but SATURATES — wall = per-row WORK
(8-rung ALU x secant iters vs head ~1-pass). BS8 dense wins only CS16 tiers.
Ledger: W2 added (per-row work wall). 96/96 exact.

## iter 4 — 2026-07-24 — SHIP-CANDIDATE DIRECTION (superseded by iter5 line)
Hypothesis: throughput form — CS=1, 2 CTA/SM, fused P2-count+P3-collect at
pivot=hmin (count>=K by construction), atomic push, P4 in-CTA. nsys 9 cells:
BS1024 gm 0.86 (0.40 prior); pro_1024k 1.19, flash_4k 1.83; laggards
flash_1024k 0.65 (pivot overflow -> re-stream on low-hit rows), pro_128k 0.63,
v32 0.64-0.75. 60/60 exact. Skeleton intact (P1/P2/P3/P4 + plateau).

## iter 5 — 2026-07-24 — MIXED (bug found, fix queued)
Hypothesis: sampled 8-rung pre-pass calibrates pivot + fused pass R=8->2.
Result (nsys): flash_128k 0.84->1.07-1.13, v32_256k 0.84->1.50 (fat rows
fixed); BUT v32_64k 0.64->0.33, v32_16k 0.75->0.62 (+~1 full-row stream).
ROOT CAUSE (code review): sample stride `i += T*SS` samples the FIRST 512
contiguous float4s of the row (+ tail), not every-SS-th — positional bias
onto the attention-sink region -> est overshoots -> pivot too high ->
count<K -> hmin re-collect stream. Fix: idx = v0 + (j*T + tid)*SS uniform.
Ledger: (positional-biased row sampling fabricates threshold estimates on
real decode data, domain: any sampled-pivot scheme, evidence: nsys iter5
v32 cells + code) — measurement/algorithm artifact. 24/24 exact so far.

## iter 7 — 2026-07-24 — SHIP-CANDIDATE (dispatch baked)
Hypothesis: tp CS>1 clusters fill SMs in mid-BS band (bs*CS<=296).
Result (nsys 5 cells): BS16 ~1.0-1.06, BS32 0.90-1.05 (v32_256k 0.62->1.05);
BS128 CS2 REGRESSES vs CS1 (cluster overhead > parallelism gain near
capacity) -> CS1 for bs>=128. Small-npad cells prefer dense/latency to
bs~64-128. 24/24+40/40+45/45+32/32 gates exact.
Dispatch baked into launcher (per-npad bands + CS=f(bs,npad)); env knobs
now explicit-override-only. Portfolio: direct(<=12288)->bs<256; latency
bs<8; dense [8,16) big-npad / [64,128) small-npad; tp elsewhere.
Next: M1 82-cell stratified screen x full BS ladder (baked dispatch).

## M1 — 2026-07-24 — 82-cell stratified screen DONE (baked dispatch, full BS ladder)
Relaunch note: first launch died with session @22/82; resumed on -048 after
anchor cell matched -073 within 0.5% (RESUME gotcha entry). 82/82, 0 FAIL,
902 pairs, all exact.
VERDICT vs bar 1.40: OVERALL gm 1.3198 (min 0.303, max 2.97) — MISS by ~6%.
By BS: 1/2/4/8 = 1.72/1.65/1.63/1.48; 16-128 = 1.13-1.20; 256-1024 = 1.17-1.19.
By model: flash 1.326 / pro 1.351 / v32 1.285.
Weak mass decomposes into:
(a) 5 PATHOLOGICAL cells (min<0.55 @BS>=16): pro_1024k_L32,
    v32_{32k L03,32k L41,64k L41,256k L03} — excluding them gm = 1.3700.
    All collapse uniformly across BS16-1024 (0.30-0.49) => arm-level failure
    on those layers' distributions (suspect: sampled-pivot overshoot ->
    re-stream, K=2048-heavy), not a capacity effect.
(b) structural bands (patho excluded): 128-512k x BS16-128 weak 39/76;
    32-128k x BS16-1024 weak 48/119. <32k and >=512k mostly healthy.
iter8 targets (ordered): (1) fix patho-layer collapse (re-stream escape /
pivot calibration robustness) — worth +4pp gm alone; (2) mid-npad x mid-BS
band (tp fused U=8 batch loads + __ldcs, 3 CTA/SM at K<=1024, ncu BW attrib).
Data: results/m1_data.csv (fresh parse), analyzer scripts/analyze_m1.py.

## iter 8 — 2026-07-24 — PATHO-CELL FIX SHIPPED (band-first + 2-sigma escape)
Hypothesis (M1 verdict item a): patho collapse = fixed pivot band [1.5K,0.6kC]
rejecting rungs whose TRUE count is inside [K,kC] (ladder coarse on low-hr
layers) -> hmin fallback (count up to 25x kC) -> overflow -> secant re-stream.
Attribution: offline exact replay (scripts/probe_patho.py) of P1 ladder + P2a
sampled pivot on real rows: 4/5 patho cells = secant 4-5 full-row passes vs 1
healthy; est accuracy <10% (NOT sampling noise); worst miss = 32 counts (one
sample) from band edge. Fix evolution:
  iter8a sym-2sigma acceptance: patho fixed, BUT 3 cells regressed (hmin
  est closer to tgt outbid the band pick -> larger C -> heavier P4).
  iter8b band-first + 2-sigma escape only when band EMPTY: offline 0/82
  failures, picks change ONLY on the 4 patho cells.
nsys 10-cell verdict (iter8b vs M1): patho cell-gm 0.607->1.219 (pro_1024k_L32),
0.725->1.070, 0.719->1.023, 0.637->1.013; the 3 iter8a regressors restored to
old (1.258->1.262, 1.067->1.073, 1.199->1.201); controls unchanged (1.737,
1.090). All exact. PROJECTED 82-cell gm 1.3198 -> 1.3531 (min 0.705), weak
142->134 of 902. Bar 1.40 gap now ~3.3%.
WATCH-ITEM: v32_256k_L03 BS16/32 was 0.46-0.49 in M1 but 0.99-1.01 in BOTH
iter8a and iter8b reruns with an (offline-verified) unchanged pick — M1
measurement anomalous or sporadic CS8 DSMEM exchange flakiness; exactness
never failed. Re-check under iter9 (cf reference: cluster_arrive_relaxed race).
Next (iter9): structural band 32-512k x BS16-128 (tp fused U=8 batch loads +
__ldcs, 3 CTA/SM smem diet at K<=1024, ncu BW attribution).

## iter 9 — 2026-07-25 — IN PROGRESS (structural band 32-512k x BS16-128)
ncu attribution (flash_128k_L22/pro_64k_L32 BS32-128, full set + stall ratios):
band is NOT BW-bound — DRAM<=10%, SM<=16%, occ ~25% (1 CTA/SM, smem-capped).
tp kernel duration FLAT vs BS (22-23us at BS32 and BS128) => per-row serial
multi-pass latency is the wall. Stalls split three ways: no_instruction 22%
(icache, big unrolled body) / long_scoreboard 21% (gmem latency) / barrier 18%
/ wait+short_sb 25%. Same-cell tier contrast: pro_64k BS32 reg<4,512,4> 12.8us
vs tp-CS1 22.6us (1.8x per-row path gap).
Dispatch probes (env-only, smoke bsx-only): FALLTHRU (reg tiers at BS>=32)
BLOWS UP (launch_bounds(TB,1) co-residency waves); DENSE16 marginal (2-3% on
128k-ISL only) => dispatch lever EXHAUSTED with existing tiers.
Kernel lever (src_i9, single-variable):
  A capn 16384->8192 (tp CS cap by npad): npad-32832 family BS16-64 -9..-15%
    (32->64+ CTAs; SM idling was the wall there). v32 unaffected (cap 8 hit).
  B U=8 everywhere: wins CS1 big rows (v32_128k BS128 -7%) but LOSES CS>1
    small slices (+5-6%: slice < 8*T falls to scalar tail, zero batching).
  B2 cascaded 8/4/2 tails: recovers some, adds icache, mixed.
  B3 U by CS (CS1->8, CS>1->4) TEMPLATE-PARAMed: combines A+B wins cleanly,
    no regressions on smoke (weak band -4..-15%; big-BS sentinels flat or
    -4%; direct tier untouched).
nsys verdict tag iter9ab3 (15 cells x full BS ladder, 4-GPU shard): RUNNING.
Analyzer: scripts/analyze_i9.py (M1 + iter8b + iter9 patch projection).
