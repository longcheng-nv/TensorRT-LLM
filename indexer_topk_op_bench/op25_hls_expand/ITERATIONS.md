# op25 — iteration log

Node: umbriel-b200-027 (GPU0-7 healthy, 33-38C idle). All timing = nsys
cold-L2 paired same-process A/B, sorted-value-set exactness.
Session date: 2026-07-08/09.

## Iter 0 — baseline push (Phase 0)

Orphan branch `op25-hls-topk` -> longcheng-nv/TensorRT-LLM @ c91fd020713:
HLS operator (op21 iter16 ship state) + minimal import closure, 11 files,
token-scan clean, code only (per repo policy).

## Iter 1 — S1a host screening (Step 1), 4 rounds

`screen_qfracs.py` (r1-r3, 17 arms) + `screen_r4_m4.py` (r4, M=4 wides).
30,290 rows/round: op22rr fp32 78 + op24 392 + real Pro 29,820.
- The static ladder's real-axis failure is the h>0.75 pair01 cliff (67% of
  real Pro steps): deepest col 0.75 undershoots when hit-rate is high.
- Deep col 0.92 fixes it; 0.95 overflows the fused collect; 0.88 is too
  shallow for the Pro h~0.9 mass (r4: w3e pro-h>=.75 0.711 vs 0.945).
- Low tail col ~0.048 catches the adversarial all_ge pole (worst hr=.05);
  0.06 already misses it entirely (w3b worst 0.000).
- K2048/v32: every deep/wide arm regresses (real .75->.375) x3 screens ->
  keep stock. Root: cr=1 band geometry, not hit rate.
- Collect-cap x2: frees the Pro low-h overflow bucket (fast .07->.94).

## Iter 2 — S1a first silicon + decomposition (Step 2)

First ship attempt (wide4b M=5 + slot_scale=2 everywhere) A/B
(`ab_qfracs.py`, 24 cells x 3 scenarios): worst gm 1.241 WIN but best gm
0.880 LOSS — including K2048 cells whose ladder was UNCHANGED.
`ab_decomp.py` 4-arm decomposition (base/ladder/slots/ship, best-scenario
= pure tax):
- M=5 ladder tax +7..19% on fast rows: the 5th count column rides in the
  hot loop of EVERY slice (tau(4)/tau(3) = +22%/pass, NOT divided by C).
- slot_scale=2: -3% (free) at n<65536 (t=512, 80KB), +12..21% at n>=65536
  (t=1024, 131KB slot smem).
FIX: ship ladder -> M=4 `w3a = (0.92, 0.45, 0.048)` (admission ~= wide4b:
pro 0.957 vs 0.958; zero fast-path tax) + slot_scale N-gated (<65536).
LESSON (extends iter14 "code mass is currency"): ladder COLUMNS are
currency too; admission bought with M must be re-priced through tau x
full-scan, never through tau/C.

## Iter 3 — Step 3 (S1b) closed by replay

`s1b_replay.py`, 29.8k real Pro transitions ON TOP of the S1a ladder:
ema_sym fast 0.851 / ema_asym 0.871 vs static wide 0.958 (oracle 1.000).
Causal EMA placement is strictly dominated — the wide static ladder
absorbed the h-information EMA was buying, and the sigma~0.1
process-variance floor makes EMA windows leak. -0.2..-1.1% model at
envelope N. Step 3 permanently closed (supersedes the iter15.3 deferral;
no kernel built, none needed).

## Iter 4 — Step 4: C=8 probe ships, HLS-MC closed by Amdahl

`ab_c8.py` (72 cells, 3 scenarios, msc C4 vs C8 vs radix on ship kernel):
- C8 gm 1.073/1.075/1.086 (real/best/worst), up to 1.19x @K1024 262K.
- radix/c8 >= 1 in 17-19/24 cells per scenario — most of R2 flips.
- Only K512 65K prefers C4 (0.90-0.98).
- Ship rule: `C=8 iff fp32 && K<2048 && bs*8<=NUM_SMS && (n>=131072 ||
  (n>=65536 && K>=1024))` in gvr_ms_auto.
>8-CTA HLS-MC (gmem count-merge): Amdahl bound from the probe itself —
t(262K,C8) - 2*scan_slice(131K,C8) ~= 17.4us serial floor vs radix 19.3;
C->inf max win ~10%, realizable fraction after gmem-merge overheads ~half
=> ~tie. Closed without building (single-point criterion could not clear
>=1.0 with margin).

## Iter 5 — Step 5: kC diet wash; small-N floor triangulated

`s3a_screen.py`: kC/2 admission-free at N<=32K; kc30 bleeds.
`ab_kc.py` silicon (16 cells real): ship/kc50 gm 0.994 — WASH. op21's
rank-scatter P4 is barrier-bound, not cand-bound (closes the loop with
op12; the op13 ~10-15% win was secant-snap-specific).
S3b (warp pipelining) not re-run: op8 NCU (occupancy structural at BS=1,
pipeline <=4%, fp32 +18% hurt) + op15 warm-L2 (phase-chain latency, not
memory tier) + this kc50 wash (work-reduction insensitivity) triangulate
the same floor. R1 BS=1 small-N stays structural (radix/sglang 2-3x);
product answer remains dispatch.

## Iter 6 — final config re-validation (v2) — DONE

Config: w3a ladder (K512/K1024) + stock (K2048) + slot_scale 2@n<65536 +
C8 rule (bs<=8 after the P0 catch below).

- Gates (final config): synth 54 + real 60 + realxC 180 + adversarial 36
  + band 72 + op22 stress 456 + 16-bit 360 — ALL GREEN (1218 asserts).
- ab_qfracs _v2 (base=iter16-equiv / ship / radix, 24 cells x 3 scen):
  * real : gm base/ship 1.184, radix/ship 1.005 (16/24), 7 flips->WIN.
    K1024 262K BS1 2.105x, K512 65K 1.891x, 131K BS16 1.70-1.80x.
  * worst: gm 1.404 (K512/K1024 half = 1.54; K2048 rows falsified as
    measurement spread by micro-check — arms bit-identical there,
    wall med 40.2 vs 40.3us), radix flips 7 (65-262K BS1 + 8K BS512).
  * best : gm 0.968 — M=4 killed the v1 tax (was 0.880); residual
    small-N BS1 pocket 0.86-0.89 = wider band (0.92->0.45 gap) doubles
    P4 cand where P4 dominates; flips NO rival outcome (R1 both-lose).
- P0 grid: first pass 1.178 14/17 — REAL regression at K1024xBS16
  (0.80-0.98): C8 rule at bs<=18 put 16x8=128 CTAs into GPC wave_cap
  contention (the known cluster-8 wall). Rule tightened to the measured
  domain bs<=8, 3 cells re-run: **1.274, 17/17** (iter16 anchor band
  1.276-1.298, cross-GPU noise). LESSON: dispatch rules stop where the
  data stops; the P0 gate caught exactly the extrapolated region.
- Session /cost at wrap: ~$172.85 (406 msgs; 116M cache-read tokens).

Remote branch op25-hls-topk: c91fd02071 (baseline) -> d5455ed1ca (ship).
Follow-up (not this campaign): op22rr full-grid rescan incl. sglang arm
+ 16-bit perf probe of the new ladder + K2048-worst falsi economics.
