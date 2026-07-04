# op13_gvr_p2cand — GVR cuteDSL P2-secant / candidate-count optimization

**Goal (user):** Starting from GVR (cuteDSL) — the plain **snap-P4** kernel
`ops/cute_vendored/blackwell/top_k/gvr_topk_decode.py` — reduce the Phase-4
candidate count (to cut P3+P4 time) while keeping Phase-2 secant **iterations as
low as possible**, by tuning the secant (kFTarget, kCC, interpolation, init), and
staying **exact + no P2 fallback** across all dtype{fp32,bf16,fp16} × K{512,1024,
2048} × seq-len × data-distribution. Local single-op perf must track integration.

**Base op:** `harness/gvr_cutedsl_op.py` (wrapper) + `GvrTopKKernel`. cr: K512/1024
→ 4, K2048 → 1. Inputs: `harness/synth_data.py` (3 beta cfgs shallow/moderate/deep,
unified preIdx hit-rate 0.6). Canonical timing = cold-L2 (512 MB evict) CUDA-graph
+ cudaEvent, same as `report/sweep.py`.

---

## Iter 0 — host-replay harness + baseline validation  ✅

- `src/p2_replay.py`: parameterized fp32 host replay of P1→P4 secant control flow
  (faithful to `harness/count_gvr_iters.py`, which is validated 108/108 vs real
  kernel). Knobs: `init_mode` {mean|lerp(alpha)|pquantile(q)}, `kFTarget`, `kCC`,
  `f_lo/f_hi/f_iter0_cap`. Uses `searchsorted` on pre-sorted row → O(logN)/eval.
  Reports per row: p2_evals, cand_count, converged(done==1), exact (value-equiv).
- `scripts/validate_replay.py`: **720/720** cells — replay(mean) matches the
  kernel-faithful baseline p2_evals & cand_count and is exact everywhere. Replay
  is trustworthy for cheap param search.

## Iter 1 — host param sweep (cand vs P2-evals tradeoff)  ✅

`scripts/sweep_params.py`, `scripts/pareto.py`. Admissible = 100% exact + 0%
fallback over all (N, cfg, seed). Baseline cand over-collect: **K512 4.05×K,
K1024 3.43×K, K2048 1.21–1.49×K**; baseline p2_evals 2.0 / 2.4 / 3.0–3.25.

Findings:
- **Candidate count CAN be cut to ~1.04×K and stay exact** — but ONLY by
  narrowing `kCC` (acceptance upper bound), which **triples P2 evals** (K512
  2.0→5.4, K1024 2.4→5.3). Each extra eval = one full-N `count_ge` pass.
- **Init-only levers (lerp/pquantile) do NOT help K512/K1024**: pushing the
  initial threshold up *backfires* in the wide default window (a single overshoot
  below K turns it into val_hi, secant then lands HIGHER → cand ↑ to ~5×K). For
  K2048 init shaves cand 1.49→1.40×K at small eval cost (marginal).
- Net: with the current secant, **cand reduction is not free** — fundamentally
  costs P2 full-N passes. This matches op12's "opt-2 added secant passes".

## Iter 2 — DECISIVE phase A/B on the REAL snap kernel  ✅  ⭐

`scripts/phase_ab.py`: clock64 P1–P4 split (on production cold-L2 wall-us) of the
real `GvrTopKKernel`, baseline `kC` vs override `kCC=1.25×K, kFTarget=K`. Built a
`GvrTimedOverride(GvrTopKKernelTimed)` exposing kC/kFTarget post-ctor (both are
plain attrs read as const_expr at compile). **K=512 fp32, beta_moderate, seed 0:**

| N | base P2/P3/P4 (tot) | narrow P2/P3/P4 (tot) | ΔP3+P4 | ΔP2 | ΔTOT |
|---|---|---|---|---|---|
| 4096   | 3.08/1.38/9.32 (16.38) | 6.57/1.46/5.67 (16.38) | −3.56 | +3.49 | **0.00** |
| 8192   | 3.00/1.70/9.45 (16.38) | 6.65/1.55/6.02 (16.35) | −3.58 | +3.65 | −0.03 |
| 16384  | 3.56/2.35/10.15(18.43) | 8.04/2.25/5.84 (18.43) | −4.40 | +4.48 | 0.00 |
| 65536  | 7.64/6.28/10.41(26.59) | 16.91/5.48/0.17(24.58) | −11.04| +9.27 | **−2.02** |
| 262144 | 16.90/17.60/6.27(43.01)| 28.89/9.73/3.08(43.01) | −11.07| +11.99| 0.00 |

**Conclusions (reframe the task):**
1. **Plain-snap P4 IS candidate-count-bound** (P4 9.3→5.7µs @4K, 10.4→0.2µs
   @65K). This **CONTRADICTS op12's "P4 barrier/latency-floor bound"** — but
   op12 measured the **rank-scatter** P4 kernel; the **snap** P4 the user named
   scales with cand. ⇒ the user's premise is correct for THIS base op.
2. **Crude kCC narrowing is a net wash**: extra P2 full-N scans cost ≈ exactly
   the P3+P4 savings → ΔTOT≈0 (only +2µs win at N=65536). P3 also shrinks at
   large N (17.6→9.7 @262K) but P2 explosion (16.9→28.9) eats it.
3. **The win, if any, requires CHEAPER candidate reduction** — fewer extra P2
   evals per unit cand cut. Two unexplored levers:
   - **moderate cand target (≈2×K)**: P4 is steeply cand-bound, so even halving
     cand (~5µs saved @small N) for ~+1 eval (~1.5µs) could net +3µs.
   - **better root-finder for the narrowed window** (converge [K, kCC_narrow]
     in 2–3 evals not 5.4): the secant is the tax; a 2-sided / regula-falsi /
     slope-model step could cut evals at fixed cand.
   - **N-dispatch**: large N is P2-dominated (each eval ~8–12µs) → keep baseline
     there; apply cand reduction only at small/mid N where P4 dominates & evals
     are cheap.

---

## Iter 3 — kCC sweet-spot + CORRECTION of iter-2 measurement flaw  ⭐

**Host pre-pass** (`scripts/kcc_host_prepass.py`): with eval-optimal kFTarget per
kCC, the candidate cut is much CHEAPER than iter-2's crude kCC=1.25×K:
- K=512: **kCC=2×K (kFT=1024)** → +1 eval (2→3) at small N, cand 3.6×K→1.2-1.5×K.
  kCC=3×K (kFT=1280) → +1 eval, cand→1.4-1.9×K. (was +3 evals for 1.04×K.)
- K=1024 similar; K=2048 baseline already lean (kCC barely helps).

**MEASUREMENT FLAW FOUND (invalidates iter-2's "wash"):** `phase_ab.py` derives
absolute µs as `production_wall × modified_kernel_fractions`. The production wall
is ALWAYS the default-kCC op, so ΔTOT was pure cold-L2 noise — the modified
kernel's real wall time was never timed. iter-2's "ΔTOT≈0 wash" is therefore
**not a valid conclusion**; phase_ab is only valid for the *fraction* split.

**Corrected wall-time A/B** (`scripts/kcc_walltime_ab.py`): builds a non-
instrumented `GvrOverride(GvrTopKKernel)` with kC/kFT override (production path,
mirrors gvr_cutedsl_op compile), times it cold-L2 (CUDA-graph + cudaEvent + 512MB
evict) directly, and checks exactness. **K=512 fp32, all variants EXACT:**

| N | base µs | kc2x µs (Δ) | kc3x µs (Δ) |
|---|---|---|---|
| 4096   | 16.38 | 16.38 (~)      | 14.34 (−2.05) |
| 8192   | 16.38 | 16.38 (~)      | 14.34 (−2.05) |
| 16384  | 18.43 | 18.43 (~)      | 18.43 (~) |
| 32768  | 18.43 | 18.43 (~)      | 18.43 (~) |
| 65536  | 26.62 | 24.58 (−2.05)  | 24.58 (−2.05) |
| 131072 | 30.72 | 36.86 (+6.14 LOSS) | 36.86 (+6.14 LOSS) |
| 262144 | 43.01 | 55.30 (+12.3 LOSS) | 47.10 (+4.10 LOSS) |

Findings:
- **Large N (≥131072): kCC reduction clearly LOSES** (+4 to +12µs) — the P2-eval
  explosion (full-N scans) dominates. Robust (deltas ≫ noise). ⇒ **N-dispatch is
  mandatory: never narrow kCC at large N.**
- **Small N: event-timed wall is UNRELIABLE here.** Median quantizes to ~1.024µs
  multiples on a ~16µs launch-overhead floor; the ~2µs predicted win is at the
  floor and inconsistent (kc3x wins @4K but kc2x — fewer cand — does not, which is
  backwards). Pure-kernel nsys is required to resolve the small/mid-N regime
  (report mandates nsys for these memory-bound kernels; event over-estimates,
  nsys/event≈0.88).

## Iter 4 — nsys pure-kernel A/B: small/mid-N WIN confirmed  ⭐✅

`scripts/nsys_kcc_ab.py` (mirrors harness/sweep_nsys protocol: eager + sync in
NVTX range, 512MB evict outside, nvtx_kern_sum / Inst, evict kernel filtered).
**K=512 fp32 cold-L2 pure-kernel (all variants EXACT):**

| N | base µs | kc2x (Δ) | kc3x (Δ) |
|---|---|---|---|
| 4096  | 11.91 | 10.87 (−9%)  | 10.11 (−15%) |
| 8192  | 12.30 | 10.77 (−12%) | 10.82 (−12%) |
| 16384 | 14.32 | 12.83 (−10%) | 12.76 (−11%) |
| 32768 | 13.98 | 14.44 (+3% loss) | 14.08 (~) |
| 65536 | 21.34 | 19.79 (−7%)  | 20.19 (−5%) |

- **Net win ~7–15% at N∈{4K,8K,16K,65K}, exact + no fallback** — once the ~16µs
  CUDA-graph launch floor is stripped (event A/B couldn't see it). The
  candidate-reduction premise IS realizable on the snap kernel, in the small/mid-N
  regime (BS=1 decode's operating range). Magnitude is ~10%, not "drastic".
- N=32768 neutral (kc2x slight loss). Large N ≥131072 LOSES (event A/B: +6–12µs;
  P2-eval explosion) ⇒ **N-dispatch: narrow kCC for N≤~65K, baseline for N≥131K**.
- kc3x (cand~1.4–1.9×K, kFT aims higher→fewer evals) ≈ kc2x; kc3x safer at small N.

## NEXT (iter 5) — confirm + finalize N-dispatch table
1. nsys A/B for K=512 bf16 (cheaper P4 → smaller win? must verify) + K=1024 fp32,
   and K=512 fp32 N∈{131072,262144} to pin the crossover under nsys.
2. Per (K,dtype): pick kCC/kFT for N≤crossover; baseline above. Build the
   N-dispatched table.
3. Copy kernel → `src/gvr_topk_decode_p2c.py` with the dispatch; full-grid
   exactness (3 beta cfgs + seeds, no fallback); nsys A/B vs gvr_cutedsl on the
   report grid (Task 5); wrapper mirrors gvr_cutedsl_op for local==integration.

## Iter 5 — generality across dtype/K: win is fp32-K512-specific  ⭐

nsys pure-kernel A/B extended (`results/nsys/kcc_K*_b.nsys-rep`):

**K=512 bf16** (P4 cheap — bf16 ties collapse candidate values → fast snap):
4096 −4% / 8192 −3% / 16384 ~ / 32768 ~ / 65536 kc2x −3% kc3x +loss. **Win ~3-4%
only, small N.** Confirms: P4 saving (hence the whole win) shrinks when P4 is
already cheap.

**K=1024 fp32** (noisy single-batch; base@8192=17.70 looks like a spike):
4096 loss / 8192 big-WIN(suspect) / 16384 −7% / 32768 −4–9% / 65536 +loss.
Mixed — smaller, noisier than K=512.

**Conclusion — premise VALIDATED but modest + regime-specific:**
- Best case **K=512 fp32, N≤~16K (+65K): ~10% pure-kernel win, exact, no fallback.**
- bf16 ~3%, fp16 untested (expect between). K=1024 small/mixed. K=2048 baseline
  already lean (kCC barely helps) — not worth it.
- NOT "大幅度": P3 is full-N-READ-bound (cand only cuts smem writes, minor) and the
  P2-eval tax (+1 full-N scan) eats much of the P4 saving. The win = (P4 fraction ×
  cand-cut) − (P2-eval cost), maximized for fp32 (expensive P4) × K512 (4×K
  over-collect) × small/mid N (P4-dominant, cheap evals).
- **Large N ≥131K always loses** (P2-eval explosion) → N-dispatch mandatory.

CAVEAT: single-batch nsys has run-to-run variance (≥0.5µs). The ~1.5µs K512-fp32
wins are consistent across 4/5 N, but a ship claim needs repeated batches (median
of ≥3 nsys runs per cell) — see iter-6.

## Iter 6 — ×3-median confirmation + fp16  ✅

nsys pure-kernel, MEDIAN over 3 independent nsys batches (batch-to-batch variance
< 0.1µs → win is robust, not noise). K=512, all variants EXACT:

**fp32** (kc2x=1024/1024, kc3x=1536/1280):
4096 kc3x −1.83(−15%)/kc2x −1.08(−9%); 8192 −1.5(−12%); 16384 −1.6(−11%);
32768 +0.4 loss / ~ ; 65536 kc2x −1.44(−7%)/kc3x −1.08(−5%). **kc3x best/safest.**

**fp16**: 4096 −0.6(−6%); 8192 kc3x −0.66; 16384 kc2x −1.01(−8%); 32768 ~;
65536 kc2x −1.50(−8%). **~5–8%, kc2x slightly better.**

**bf16** (iter 5): ~3–4% small N only.

CONFIRMED: K=512 net win at N≤16K + 65K, robust, exact, no fallback. Magnitude by
dtype fp32(~10%) > fp16(~6%) > bf16(~3%) — tracks P4 cost. N=32768 is a neutral
notch (kc2x slight loss); large N≥131K loses. → N-dispatch + per-dtype variant.

## Iter 6b — SHIP DECISION: production logits are fp32 ⇒ ship-worthy

Production indexer-topk input dtype confirmed **fp32**:
- `dsa.py::warmup_heuristic_topk_decode` allocates `logits = torch.zeros(..., dtype=
  torch.float32)` (dsa.py:97) and `scratch`/`radix_aux_logits` fp32.
- The decode logits fed to `(cute_dsl_)indexer_topk_decode` come from
  `fp8_paged_mqa_logits` / `cute_dsl_fp8_paged_mqa_logits` (dsa.py:2381/2395/2426/
  2433) — fp8 inputs, **fp32 logits output** (dot-product accumulation).
⇒ The strongest win regime (fp32, ~7–15% at N≤16K+65K) is the PRODUCTION regime.
   K=512 = V4 Flash, K=1024 = V4 Pro. **Worth shipping** as an N-dispatched
   kCC/kFTarget tweak (exact, no fallback). bf16/fp16 not the prod path here.

## Iter 7 — BUILD + full validation + A/B  ✅  ⭐ (SHIP: K512 fp32 only)

Built the shipped deliverable and validated it end-to-end:

- **`src/gvr_p2c_op.py`** — N-dispatched op, mirrors `harness/gvr_cutedsl_op.py`
  EXACTLY (same vendored `GvrTopKKernel`, compile flags, fake tensors, launch) so
  local single-op perf == tensorrt_llm integration. Override via `GvrP2C` subclass
  (kC/kFTarget post-ctor, NO vendored-file edit). Dispatch: `dispatch_params(dtype,
  K, N)` → narrow `(kCC,kFT)` for N≤`NARROW_N_MAX`=65536 else baseline.

- **Exactness (`scripts/validate_exactness.py`): 480/480 cells PASS** — value-equiv
  to torch.topk AND host-replay done==1 (no P2 fallback), cand∈[K,kCC]. fp32 grid
  240 cells (120 narrow + 120 baseline) + bf16/fp16 240 cells (all baseline). 3 beta
  cfgs × 4 seeds (fp32) / 2 seeds (half) × all N × K{512,1024,2048}. 0 fails.

- **nsys pure-kernel cold-L2 A/B (`scripts/nsys_p2c_ab.py`, ×3-median, shipped p2c
  op vs baseline gvr_cutedsl op):** (`results/nsys_p2c_ab_medians.txt`)

  **K=512 fp32 (SHIPPED) — robust WIN, no large-N loss:**
  | N | base | p2c | Δ |
  |---|---|---|---|
  | 4096 | 12.28 | 10.48 | **−14.7%** |
  | 8192 | 12.49 | 11.09 | **−11.2%** |
  | 16384 | 15.01 | 13.33 | **−11.2%** |
  | 32768 | 14.59 | 14.75 | +1.1% ~tie |
  | 65536 | 21.53 | 20.34 | **−5.5%** |
  | 131072 | 26.03 | 25.98 | −0.2% ~tie (dispatch→baseline) |
  | 262144 | 38.49 | 38.44 | −0.1% ~tie (dispatch→baseline) |

  N-dispatch boundary CONFIRMED: at N≥131072 the p2c op ties baseline exactly
  (it hands those N to baseline params) → the shipped op never regresses at large N.
  65K wins, 131K ties → crossover pinned in (65536,131072].

- **K=1024 fp32: NOT shipped.** ×3-median is noisy with REAL regressions — N=4096
  **+15.8%** and N=65536 **+12.4%** (despite 8K/16K/32K wins). Matches iter 5/6
  "K1024 small/noisy". A config with real losses fails the falsification bar →
  demoted to baseline. Only fp32 K=512 (= V4 Flash) is in `_NARROW_TABLE`.

- **K=2048 fp32: NOT shipped (now nsys-confirmed, was only a-priori).** ×3-median
  within-op base vs kc2x/kc3x: net wash-to-loss — N=8192 ~−0.3 (tiny), N=16384
  **+0.46/+0.75**, N=32768 ~tie, N=65536 **+1.77/+1.69**. Confirms baseline already
  lean (cand only 1.2–1.5×K → no headroom); narrowing only adds P2-eval tax.

**SHIP STATUS:** `_NARROW_TABLE = {(fp32, 512): (kCC=1536, kFTarget=1280)}`,
`NARROW_N_MAX=65536`. Exact, no fallback, robust ~5–15% small/mid-N win, zero
large-N regression. Ready to wire into production `GvrParams.get` as an N-keyed
override for (fp32, K=512, cr=4). Optional next: more K1024 batches to de-noise (if
a clean win emerges, re-add); wire into production GvrParams; report column.

## Iter 8a — 2026-07-04 — cheaper P2 root-finder: LOG-COUNT interpolation (host replay)  ⭐✅

**Hypothesis (H2 from LEARNINGS)**: the P2 eval tax comes from LINEAR value-space
interpolation `f=(clo-kFT)/(clo-chi)` against a ~exponential CCDF tail — it
systematically under-steps. Interpolating in log-count space
`f=log2(clo/kFT)/log2(clo/chi)` should converge in fewer full-N scans.

**Method**: `p2_replay.py` + `interp_mode` knob {linear, logcount, illinois,
logillinois} (linear regression-checked 720/720 vs baseline);
`scripts/rootfinder_sweep.py` sweeps kFTarget per (window, mode), eval-optimal
pick gated on 100% exact + converged (3 beta cfgs × 4 seeds × all N). Results:
`results/rootfinder_sweep_fp32.{log,json}`.

**Findings (fp32)**:
- **logcount kills the large-N eval inflation** in every regime:
  - K512 kc2x(1024, ft=614): evals **3.00 FLAT all N** (lin 3.0→3.75@262K),
    cand 1.30–1.52×K; strictly ≤ lin everywhere (4K cand 1.93→1.52 at same evals).
  - K1024 kc2x(2048, ft=1024): evals 3.00 flat, **cand 1.04–1.11×K**
    (lin: 3.92@4K, 3.42/3.83 large-N, cand→1.97).
  - K1024 base(5120, ft=1024): **large-N free win** — evals 3.50→2.75@131K,
    cand 3.49→1.59×K@262K.
  - K2048 base(6144, ft=2048): evals **2.00@8K** (lin 3.0), 3.00 flat large-N
    (lin 3.58/3.75), cand 1.13→1.07×K — free −0.58/−0.75 full-N scans at 131K/262K.
- **Illinois == linear everywhere** (initial brackets are fake counts; secant
  rarely hits the same side twice) and logillinois == logcount ⇒ the stateless
  log formula captures ALL the win. No state, no extra smem — pure f-formula swap.
- Narrow@262K tax: +1.75 evals (lin) → **+1.0 eval (log)**; iter2 attribution
  (P3+P4 saving ≈ 11µs@262K vs eval ≈ 8.4µs) ⇒ narrow-log may flip net-positive
  at large N — kernel A/B decides (iter8c).

## Iter 8b — 2026-07-04 — kernel port: GvrP2CLog subclass — exactness 396/396 ✅

**Port**: `src/gvr_p2clog_op.py` — GvrTopKKernel subclass overriding
`phase2_secant_search` (@cute.jit method resolved via MRO at trace time; NO
vendored-file copy/edit — same pattern as iter7 GvrP2C). Interp block uses
`cute.math.log2(·, fastmath=True)` with degenerate-denominator fallback to the
linear formula; clamps/brackets/fallback logic unchanged ⇒ exactness guard
untouched. Smoke 5/5 exact (incl. narrow@262K).

**Validation (`scripts/validate_log_exactness.py`, iter7 two-check protocol):
396/396 PASS** — fp32 5-variant portfolio {K512 (1024,614); K1024 (2048,1024),
(base,1024); K2048 (base,2048), (4096,2048)} × all N × 3 beta cfgs × 4 seeds;
0 value-equiv fails, 0 P2-fallback fails. `results/log_exactness_fp32.log`.

**bf16/fp16 host sweeps**: winners identical to fp32 (P2 secant always computes
fp32; ties only shift cand slightly). 16-bit K2048 base window: cand 1.16–2.22×K
→ 1.06–1.10×K at unchanged evals (pure P4 saving). ⇒ ship shape can be a uniform
per-K table, no dtype branch. `results/rootfinder_sweep_{bf16,fp16}.log`.

**iter8c queued**: `scripts/nsys_p2clog_ab.py` (3-way base/p2c/log per K, log
variants at ALL N — large-N narrow flip is the decisive question). BLOCKED on
GPU co-tenancy (all 8 GPUs ~157GB/25-94% util from another namespace);
idle-watcher armed.

## Iter 8c INCIDENT — 2026-07-04 — b200-019 first run CONTAMINATED, all reps quarantined

**Symptom**: r1/r2 single-batch parses showed base 1.5–7× slower than iter7
references (K512 base@4K 14.7–15.1 vs 12.3µs; @262K 75.7 (r1) → **258.7 (r2)**
vs 38.5µs) with within-run drift and non-monotonic N — temporal contamination
(co-tenant arrival or clock/power-state collapse on 019), NOT a code or HW-SKU
effect (base contains none of our changes). All 5 landed reps →
`results/nsys/contaminated_20260704_019/`; none parsed into conclusions.

**Driver hardened to v2** (`run_iter8c_batches.sh`): wait_free now needs
mem<30GB AND util≤5% on ALL GPUs ×3 consecutive samples; post-batch sanity gate
parses the fresh rep and requires base@minN<20µs AND base@maxN<65µs (known-good
B200 bands) else auto-quarantine + redo (≤3 tries). Old driver on 019 must be
STOPPED before relaunch (file was rewritten in place — running bash reads
incrementally).

**Incident root cause (resolved same day)**: NOT a co-tenant — **b200-019
GPU 0 has broken cooling**: constant 70–74 °C / 227 W at 0 % util & 0 MiB
(GPU 1: 31 °C), lifetime `SW Thermal Slowdown` counter 6.82e9 µs (~1.9 h)
in `nvidia-smi -q -d PERFORMANCE`. Under load it throttles progressively —
exactly the observed within-run drift. mem/util preflight cannot catch it.
Fix: rerun pinned `CUDA_VISIBLE_DEVICES=1` (GPU 1 counters clean). Also fixed
a v2-driver deadlock: wait_free's awk max-accumulator was uninitialized, so a
PERFECTLY idle node (`0, 0` on all GPUs) yielded empty maxmem → `[ "" -lt ]`
error → infinite WAIT; fixed with `BEGIN{m=0;u=0}` + `+0` coercion.

## Iter 8c — 2026-07-04 — nsys ×3-median A/B (GPU1 b200-019) + SHIP  ⭐✅

9/9 batches passed the v2 sanity gate first-try; repeatability <0.5 %
(K512 base@262K: 37.42/37.33/37.27 µs). Full tables:
`results/nsys_p2clog_ab_medians.txt`. Deltas vs base, fp32 pure-kernel cold-L2:

| N | K512 p2c | K512 logn | K1024 logn | K1024 logb | K2048 logb | K2048 logn |
|---|---|---|---|---|---|---|
| 4096  | **−16.6%** | −8.7% | **−3.5%** | +1.2% | — | — |
| 8192  | **−12.4%** | −8.1% | **−32.1%** | +5.5% | +21.8% | **−3.7%** |
| 16384 | **−11.9%** | −10.3% | −0.9% | +22.1% | −6.7% | **−8.3%** |
| 32768 | +0.2% | +2.4% | **−8.8%** | −7.7% | +0.6% | −1.0% |
| 65536 | **−5.7%** | −0.8% | +13.4% | +12.6% | +0.5% | +0.6% |
| 131072 | +0.0% | +7.4% | **−22.0%** | −23.1% | −11.2% | **−11.0%** |
| 262144 | +0.2% | +13.4% | +9.8% | +9.4% | −12.1% | **−12.2%** |

**Ship verdicts** (±0.2 µs = tie; any real regression band stays baseline):
- **K512: log NOT shipped.** logn loses to the iter7 p2c everywhere it wins
  and REGRESSES +7.4 %/+13.4 % at 131K/262K — the host prediction that the
  +1 eval tax (~+8.4 µs) would be repaid by P3+P4 savings at 262K is
  **falsified by nsys** (net +5.0 µs loss). iter7 ship stands unchanged.
- **K1024: ships for the FIRST time, N-dispatched.** logn(2048,1024) at
  N≤32768 and N==131072 (−32.1 % @8K, −8.8 % @32K, −22.0 % @131K — the 8K/131K
  base slow spots are the host-replay eval spikes, reproducible ×3); 65536 and
  262144 stay baseline (real +13.4 %/+9.8 % regressions, same shape that
  killed iter7's blanket narrow).
- **K2048: ships at ALL N≥8192** with logn(4096,2048) — worst cell +0.6 %
  (tie band 32K–65K), wins to −12.2 % @262K (the free −0.58/−0.75-eval
  large-N win; host prediction −10~15 % @131K/262K CONFIRMED). logb rejected
  at 8K (+21.8 %: cand 2.74×K hurts P4, as predicted).
- **16-bit: baseline** (host winners identical to fp32 but zero nsys
  evidence, op20 precedent says 16-bit band boundaries shift, and production
  indexer logits are fp32).

**Landed**: `dispatch_p2c_v2()` + `gvr_cutedsl_p2c_v2()` ship-routing in
`src/gvr_p2clog_op.py` (log path → GvrP2CLog; lin/baseline path → iter7
`gvr_p2c_op`). Routing exactness re-validated on GPU1: 9/9 branches
(every band × both Ks + K512 both routes + bf16 fallback) value-exact
uniq=K valdiff=0. **CAMPAIGN: iter8 CLOSED — cheaper-P2 log-interp shipped
for K1024/K2048; K512 keeps iter7 lin-narrow.**

## (superseded) iter-7 plan — build the N-dispatched kernel + full validation
1. Copy `gvr_topk_decode.py` → `src/gvr_topk_decode_p2c.py`. Make kCC + kFTarget
   N-dispatched (the kernel already has N at compile via the seq dispatch; simplest
   is to pick (kCC,kFT) per cell in the op wrapper's GvrParams override path, since
   N is known at compile/launch). Table: N≤65536 → kc3x (K512: 1536/1280; K1024:
   3072/2560; fp32-only — keep baseline for bf16/fp16 if not shipping those);
   N≥131072 → baseline (kC default). Crossover sits 65K–131K; confirm exact point
   under nsys (parse `results/nsys/kcc_K512_fp32_b.nsys-rep` for 131072/262144).
2. Wrapper mirrors `harness/gvr_cutedsl_op.py` EXACTLY (compile flags, fake tensors,
   launch) so local single-op perf == tensorrt_llm integration.
3. Full-grid exactness: value-equiv torch.topk over 3 beta cfgs × seeds × all N,
   done==1 (no P2 fallback) everywhere.
4. nsys cold-L2 A/B vs gvr_cutedsl on the report grid; add as op column / report.

## (superseded) iter-6 plan — decision: ship the narrow win or pivot
1. Repeat K=512 fp32 nsys A/B ×3 batches → median, confirm the ~10% small/mid-N
   win is robust (not variance). Add fp16.
2. Decide ship-worthiness: a kCC/kFTarget + N-dispatch change is ~free to
   implement and exact; if production V4 indexer logits are **fp32** and N often
   ≤64K, it is worth shipping (K512=V4 Flash). If logits are **bf16**, the ~3%
   win likely is not worth the dispatch complexity → document and stop, or pivot
   to "reduce P2 evals at large N" (the larger, opposite-direction lever).
   ACTION: confirm production indexer logits dtype + typical decode N first.
3. If ship: copy kernel → `src/gvr_topk_decode_p2c.py`, add N-dispatched
   kCC/kFTarget (table from iter 3-5), full-grid exactness (3 beta cfgs+seeds, no
   fallback), nsys A/B vs gvr_cutedsl on the report grid, wrapper mirrors
   gvr_cutedsl_op (local==integration). Then report.

## (superseded) iter-4 plan — nsys pure-kernel decisive small/mid-N A/B
1. nsys pure-kernel cold-L2 A/B (reuse `harness/sweep_nsys.py` /
   `report/parse_nsys_full.py` protocol) for K512/1024 × fp32/bf16 × N∈{4K,8K,16K,
   32K,65K}, baseline vs kCC=2×K and 3×K. Resolve whether the small/mid-N P4
   saving net-beats the +1 eval once the launch floor is stripped.
2. If a net-win regime exists → finalize N-dispatched kCC/kFT table, copy kernel
   to `src/gvr_topk_decode_p2c.py`, full-grid exactness (all beta cfgs+seeds, no
   fallback), nsys A/B vs gvr_cutedsl across the report grid (Task 5).
3. If nsys also shows wash/loss at small N → the candidate-reduction direction is
   falsified for the snap kernel too (P4 saving real but ≤ the P2-eval tax even
   at the floor); write up + pivot to "reduce P2 evals at large N".

## (superseded) iter-3 plan — sweet-spot + cheaper-secant search
1. In `phase_ab.py`, sweep kCC ∈ {1.25,1.5,2,3}×K and measure ΔTOT per N to find
   the cand target that maximizes (P4_save − P2_cost) per N/K/dtype. Hypothesis:
   ~2×K at N≤16K is net-positive; large N wants baseline.
2. Prototype a cheaper P2 root-finder in `src/p2_replay.py` (regula-falsi w/
   Illinois, or use the count-vs-thr slope from two brackets) targeting the
   narrowed window in ≤3 evals; re-run host sweep for (evals, cand, exact).
3. If host predicts a regime where (cand↓ at ≤+1 eval) → port into a copied
   kernel `src/gvr_topk_decode_p2c.py`, verify exact on full grid (all beta
   cfgs, no fallback), then nsys cold-L2 A/B vs gvr_cutedsl (Task 5).
4. Keep wrapper identical to `gvr_cutedsl_op.py` for local==integration parity.
