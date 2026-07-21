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

## Mechanism map (results/phase_bs.csv, clock64 splice, BS∈{1,2,8}, 6 cells)

- Warm/mid-hit (hit 0.23-0.70): **P4 select 44-58%** (leader-only), P2 17-24%,
  P1 10-14%. Holds at BS 2 and 8 (cs 8→4). Straggle ≈1.0.
- Cold-hit (flash-512k hit 0.057): **P2 count+admission 45-51%** (rung miss →
  refine rounds, each a cluster sync), P4 27-35%.
- V4 cells cluster at BS≤8 even on the 32K rung (pick_config keys off
  max_seq_len=N*cr, NOT compressed N) — only v32 (cr=1) 32k runs cs=1.
- flash-512k BS-scaling attribution (results/f512k*): pr 2.1-3.1× FASTER than
  base everywhere (R0 repairs). BS≥256 collapse = BW-bound multi-pass wall:
  pr ~0.40µs/row vs sglang ~0.19µs/row at BS=1024 (cold content ⇒ ~4-5
  full-N passes vs sglang's fixed 2). Fix = fewer miss-path passes, not R0
  rollback.
- **NCU DRAM + phase split at BS≥256 (07-20)**: pr 403µs @76.9% DRAM = 2.38GB
  = 4.4× row data; sglang 188µs = 1.9×. Phase split (BS 256/1024, cs=1):
  **P2 = 72.5-72.7%** (≈3.5 full-N-pass equivalents ⇒ the ladder MISSES on
  this real cell and refines ~2.5 extra passes), P3 21%, P4 only 3.4%.
  ⇒ the collapse is a P2 admission-miss problem; L-J's wider/deeper bracket
  (replay M8 band 482 on this cell) should kill the refine → P2 →1 pass →
  projected ~2× at BS≥256 (0.53→~parity). L-J now carries BOTH loss regions
  (warm BS≤8 via P4 diet + cold/big-BS via refine elimination).

## gvr29/HBE scope ruling (2026-07-20)

op29's gvr29_hbe arm achieves sglang parity at BS 1-16 AND 1.3-1.7× at
BS≥1024 (N 131-262K) on the op22 grid — BUT the kernel is a CUDA **fork of
sglang v2** (src/gvr29/gvr29_standalone.cu) with the HBE tier added; its
non-engaged cells literally run sglang code (hence sgl/hbe = 1.00 there).
Under this campaign's red line, porting gvr29 wholesale = shipping a rival
port (out of scope, same as op36 sgl_bx). IN scope: re-implementing the HBE
*algorithmic ideas* inside the production cuteDSL GvrTopKKernel as a new
guess/verify strategy (sampled-rank guess column when hint is cold; fused
collect-with-slack; mini-hist refine; GMEM-atomic multi-CTA cooperation
instead of cluster-barrier chains). Tier-2 lever if dp4-v2 falls short —
note op34's lock says BS=1 ceiling ≈ sglang parity, which matches gvr29's
measured 1.00-1.01 at BS 1-8.

## BS≤8 exhaustive lever ledger (user directive 2026-07-20: 穷尽 GVR 改进空间)

Phase budget at BS≤8 loss cells (phase_bs.csv): P4 44-58% (leader-serial),
P2 17-24% (45-51% cold), P1 gather 7-14%, P1b ~3%, stage/epilogue <1%.

L-A  dist_P4 v1 (A2 splice, 6 syncs) — **VERDICT: NET LOSS at current head**
     (results/dp4, 132 cells 0 inexact, pr/dp4 composite 0.9721): wins only
     N=262K BS1-2 (1.03-1.07), loses BS8-64 mid-N (worst v32-64k 0.77-0.85),
     BS≥128 cs1 control clean 1.00. op36's A2 headroom shrank because the
     current head already dieted P4 (hist-diet + p4tt). Arithmetic: at
     v32-64k BS8 the distributed-work saving (~11µs ideal) is overwhelmed
     by ~16µs of sync/DSMEM-atomic overhead — NCU attribution in flight to
     decide if L-B (4 syncs) is viable or the whole cluster-barrier dist_P4
     family is dead below 262K. DO NOT ship dp4-v1.
L-B  dist_P4 v2: S1→P3-handoff merge + S5+S6 merge (6→4 syncs)   — next
L-C  dist_P4 v3: single-level wide-hist over cand band (4→3 syncs,
     p4_exact_tail absorbs boundary; host-side fire-rate estimate first)
L-D  cs=16 probe at BS 1-2 (ctor bound [1,16]; only AFTER dist_P4 lands —
     leader-serial P4 made cs>8 pointless before)
L-E  P2 refine sync diet: fold rung-count cluster merges into fewer rounds;
     admission already vseed'd — do NOT retune qfracs (silicon-wash, §9c)
L-F  HBE-lite cold guess: P1 samples full-row tiles → order-stat guess
     column when hint is cold (in-kernel, no hit-rate dispatch), replaces
     miss→secant chains; targets cold cells (flash-512k) at ALL BS
L-G  P1 hint-gather distribution/overlap check (7-14%; verify gather is
     cluster-parallel; overlap with P1b via warp split = the "deepest
     untouched lever" intra-CTA pipelining, op32-falsified ONLY at cs=1
     short rows — cluster BS≤8 large-N is unprobed territory)
L-H  Launch micro-tuning — **FALSIFIED as meaningful** (results/t512:
     T512 override on the 6 V4 T1024 rungs = +1.0-1.5% BS1-2 only, loses
     BS8-128 to 0.80; the pre-compression n_per_cta inflation is NOT a
     consequential mis-tune; not worth a shape-gate).
L-J  **v1 nsys VERDICT (results/lj, 132 cells, 0 inexact, b200-027): NET
     LOSS as-implemented** — gvr_pr/gvr_lj composite 0.7221; uniform
     ~0.65-0.80 tax across ALL warm rungs/BS (flat in N and BS; BS1024
     BW-bound cells also ~0.70 ⇒ structural extra memory traffic, not
     the modeled -3..-7% wide-count-column tax). ONLY win = the cold
     flash-512k cell (hit .057): BS256/512/1024 = 1.26/1.24/1.33 vs pr
     (lifts vs sglang 0.48-0.53 → 0.64-0.70) — the P2 admission-miss
     refine root cause IS killed there, as projected. flash-1024k shows
     NO such win (0.63-0.72) ⇒ benefit is cold-cell-specific. Anchor
     sane: pr/sgl this run 0.8640 vs baseline 0.8664. NEXT: phase
     attribution (warm loss cell vs flash-512k win cell) to locate the
     uniform tax; if removable → fix + re-A/B; else L-J degenerates to
     an in-kernel cold-escape form (bracket only when admission misses)
     — never a host-side hit-rate gate (hit unknowable at inference).
     Original design: count MORE rung columns
     in the existing P2 M-ary pass (M=2+vseed today) → post-count pick the
     tightest pair (lo,hi) straddling K → P3 emits ≥hi "sure set" directly
     to output, collects ONLY the (lo,hi) band → P4 ranks band_count ≪
     cand_count elements for the remaining K-cnt_hi slots. Attacks the
     fat-admission P4 input (kC=3072=6×K@K512; flash-1M measured 4408
     admitted) with ZERO new cluster syncs; leader P4 hist/scatter work
     scales down proportionally. Cost = wider count pass (qr2 3-col tax
     was -3..-7%, must A/B col count) + P3 two-threshold classify.
     Host-side replay estimate of band sizes on the 25 real cells FIRST.
L-I  PDL / kernel-prologue overlap (sglang gains from 2-kernel PDL; GVR
     could pre-launch evict/plan under PDL) — framework-adjacent, last
FALSIFIED (do not revisit): forced cs2/4 w/o algorithm change; P4 reseed/
fine-iter; fused P2+P3 (Opt-L/ms_auto); smem-resident; sw-pipeline @cs1;
hit-rate host dispatch; qfracs retune V4.

Ceiling honesty: op34 oracle lock ⇒ BS=1 target is sglang PARITY (±5%),
not a win; composite >1.0 comes from parity at BS≤16 + protected wins at
BS 32-128 + big-BS repair (L-F).

## Measurement discipline

nsys cold-L2 only (no CUDA-event verdicts); ship verdicts ≤2-way concurrent;
A/B arms paired same-run same-GPU; sglang timed by us_span (PDL 2-kernel);
anchors vs 07-20 §9b canonical (b200-027) med ≤1.05; never resume a
partially-errored nsys batch (rep overwrite); env -u GITHUB_TOKEN -u HF_TOKEN;
*.sqlite/*.nsys-rep never committed.
