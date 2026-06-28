# op14 — GVR ≈1-HBM-pass secant (radix-style candidate compaction)

Base op: **GVR cuteDSL rank-scatter P4** (op#7) =
`p4_recursive_digit/src/gvr_topk_decode_p4.py` (copied here as
`src/gvr_topk_decode_1pass.py`), wrapped like `harness/gvr_cutedsl_rs_op.py`.

## 1. The proposed algorithm (user)

For a guessed threshold `t`, maintain a candidate buffer `cand` of size `C·K`
(C tunable, start C=2):

- **Undershoot** (`n_ge(t) < K`): every element ≥ t is a confirmed top-K winner —
  save their indices. We still need `K − n_ge` more, from elements `< t`. Pick the
  next threshold `t₁` by secant interpolation between `(t, n_ge)` and
  `(min(preIdx-values), n_lo)`, and **scan only the not-yet-selected region** for
  `≥ t₁`. Repeat.
- **Overshoot** (`n_ge(t) = N_cand ≥ K`):
  - if `N_cand > C·K` → buffer can't hold them → run a **full-N secant `count_ge`**
    pass to get a higher threshold, repeat;
  - **converged** when `K ≤ N_cand ≤ C·K` → copy all `cand` to smem and do an
    exact size-`N_cand` top-K refine (= rank-scatter P4).

Goal: drive total HBM element scans toward **≈ 1**.

## 2. What this IS, mapped to GVR's existing phases

| Proposed step | GVR today |
|---|---|
| "converged when N_cand ≤ C·K → copy to smem → refine" | **P3 collect + P4 rank-scatter**, with `kCC = C·K`. op13 already made `kCC` tunable; `kCC=2×K` is exactly C=2. |
| "secant interpolation to pick next threshold" | **P2 `phase2_secant_search`** (identical secant on the (val_lo,cnt_lo)/(val_hi,cnt_hi) bracket). |
| "full-N `count_ge` when N_cand > C·K" | **`block_count_ge`** (the full-N HBM streaming counter). |
| preIdx-min as a bracket endpoint, preIdx-based init `t` | **P1 `phase1_preidx_stats`** (already computes pmin/pmax/pmean from preIdx values). |

⇒ The **refine-in-smem** and **secant** parts are already GVR. The two parts that
are **genuinely new** vs current GVR:

- **(N1) Cross-iteration candidate buffering + region restriction.** Today
  `block_count_ge` only *counts* ≥ t over all N and **re-reads all N** on the next
  eval. The proposal **compacts survivors to a scratch buffer** during the first
  pass and restricts later passes to that shrinking survivor set (classic
  radix/bucket compaction). This is what can cut HBM passes from ~2–3 → ~1.
- **(N2) The undershoot fast-exit** (save confirmed winners, only hunt the deficit
  in the residual region).

## 3. Honest analysis — where this can and cannot win

**Win regime = LARGE N (the P2-HBM-bound regime).** At large N each `count_ge` is
~8–12 µs of pure HBM read; op13 showed the secant runs ≥3 evals there and that is
the dominant cost (op13's narrow-kCC *lost* at N≥131072 because it *added* evals).
This algorithm attacks the **opposite lever op13/op12 flagged as bigger**: *reduce
the number of full-N passes*. If passes drop 3→1, P2 HBM traffic ≈ −2/3.

**Where it will NOT help (manage expectations):**
- **Small/mid N** (≤ 32K): GVR is launch/floor-bound there (op13: ~16 µs CUDA-graph
  floor; rank-scatter P4 has its own ~14–20 µs floor — op12). Fewer HBM passes won't
  show under the floor. So validate the *large-N* configs first, not small N.
- **rank-scatter P4 is barrier/floor-bound** (op12). This algorithm does NOT touch
  P4 cost; it only reduces P2 traffic. That's fine — at large N, P2 dominates and
  the P4 floor amortizes. But it caps the achievable small-N gain to ~0.

**Two correctness/feasibility subtleties to respect:**
- **Undershoot saves <K winners but the residual ≈ N** → no traffic saving in that
  branch *unless* the residual is compacted. The "≈1 scan" win comes from the
  **overshoot/compaction** side: first pass with a slightly-low threshold writes
  survivors (≤ a few ×K, ≫ smem but ≪ N) to **global scratch**, then all later
  evals + the P3 collect read that scratch (≪ N) instead of re-streaming HBM.
- **The compaction itself writes ≈ N_survivors to HBM once.** Net traffic ≈
  `N (first read) + N_surv (write) + Σ N_surv (re-reads)`. This beats current
  `≈ passes × N` only when `N_surv ≪ N` AND the preIdx-seeded first threshold lands
  survivors small. preIdx hit-rate (Flash ~0.46, Pro ~0.75) makes the seed good →
  plausible. **This is the empirical question to validate.**

**Verdict going in:** plausible net win at large N (≥131072), ~neutral at small N,
and it is the right direction (op12/op13 both concluded "reduce P2 passes" is the
larger unexploited lever). The risk is that the extra scratch read/write + control
complexity eats the saved reads when survivors aren't small enough.

## 4. Build / validation plan (validate-first on typical seq_len)

1. Baseline = `harness/gvr_cutedsl_rs_op.gvr_cutedsl_rs` (op#7), UNMODIFIED.
2. New op `src/gvr_1pass_op.py` wraps `src/gvr_topk_decode_1pass.py`, mirrors the
   rs-op compile/launch EXACTLY (local == integration).
3. Implement N1+N2 in the kernel via **/omni-kernel** (autonomous kernel loop),
   gated behind a flag so the baseline path is untouched.
4. **Data = identical to report.html**: `synth_data.get_bundle(K,dtype,N,
   cfg='beta_moderate', seed=42)` (the report's exact inputs). preIdx semantics per
   `report/PREIDX_SEMANTICS.md` (K512/1024 cr=4 vertical, offset 0).
5. **Typical-seqlen first** (where the win must appear): N ∈ {131072, 262144} (and
   a 65536 control), K ∈ {512, 1024}, dtype ∈ {fp32, bf16}. Exactness =
   value-equiv to torch.topk + no fallback. Perf = nsys pure-kernel cold-L2 (same
   protocol as `harness/sweep_nsys.py` / op13 `nsys_*_ab.py`, ×3-median).
6. Only if a typical-N win is real → extend to the full report grid + add as a
   report op column.

## 5. nsys protocol (must match report)
- 512 MB L2 evict OUTSIDE an NVTX range; eager launch + sync INSIDE; whole loop
  under cudaProfilerApi; `nvtx_kern_sum` Total/Inst, evict kernel filtered.
- nsys/event ≈ 0.88; **≥3-batch median** before any ship/no-ship call (op13 lesson:
  a single batch hid multi-µs regressions for K=1024).

## 6. Measured ceiling (host replay on report data: beta_moderate, seed 42)

Baseline full-N HBM reads at large N = **3** (2 P2 secant evals + 1 P3 collect),
for BOTH K=512/1024 and fp32/bf16 at N∈{65536,131072,262144}. Final candidate
count: K512 ~2.1–2.7k (≈4.3×K), K1024 ~3.6–4.9k (≈4.6×K) — i.e. ≪ N.

Cost model (from op13 phase_ab fractions): at N=262144, P2+P3 ≈ 80% of total.
3N→~1N traffic ⇒ P2+P3 ≈ −60% ⇒ total ≈ −40% to −50%. **40% avg at large N is
physically reachable**; small N (≤32K) stays ~neutral (launch + P4 floor).

## 7. Concrete kernel-level design (fast path + fallback)

Add a **global scratch** `(cand_val[cap] fp32, cand_idx[cap] int32)`, `cap = 16*K`
(op-wrapper-allocated, one row's worth for BS=1; for BS use per-row stride). Flag
`enable_1pass_compaction`.

- **P1 (unchanged):** preIdx stats → val_lo=pmin, val_hi=pmax, t0 = a CONSERVATIVE
  low threshold. Use **t0 = pmin** (smallest preIdx value): since preIdx are the
  prev-step top-K with high hit-rate, count_ge(pmin) ≈ a few ×K. (Guarantees the
  fast-path test below; if it undershoots we fall back.)
- **Pass-1 fused count+compact (replaces the FIRST block_count_ge):** stream all N
  once; for each v ≥ t0, `slot = atomicAdd(smem_ctr, 1)`; if slot < cap write
  `cand_val[slot]=v, cand_idx[slot]=orig_index`. Block-reduce total c0.
  - **Exactness gate:** valid fast path requires `K ≤ c0 ≤ cap`. Because c0 ≥ K ⇒
    t0 ≤ (K-th value) ⇒ every top-K element is ≥ t0 ⇒ present in scratch ⇒ refine is
    exact. c0 ≤ cap ⇒ no overflow truncation.
  - If `c0 < K` (t0 too high) OR `c0 > cap` (overflow): **FALLBACK** to the
    unmodified baseline path (re-run from P2 on full input). Rare; keeps exact.
- **P2 secant on scratch:** identical secant, but `block_count_ge` now counts over
  `cand_val[0:c0]` (≪ N) instead of full N. Converges to thr with cand∈[K,kCC].
- **P3 collect on scratch:** scatter `cand_val ≥ thr` from scratch to smem_keys
  (reads c0, not N).
- **P4 (unchanged):** rank-scatter exact over smem candidates.

Net HBM: `N (pass-1 read) + c0 (write) + evals×c0 (secant) + c0 (P3)` ≈ `N + ~4·c0`
≈ N (since c0 ≪ N). vs baseline `3N`. Expected large-N total ≈ −40%.

Exactness across dtype: scratch stores fp32 values (like smem_keys today) → no new
precision loss. bf16/fp16 inputs upcast on load exactly as baseline P3 does.

Fallback guarantees correctness for any distribution where pmin-seeded survivors
overflow cap (heavy upper tail) or undershoot (degenerate preIdx).
