# RESUME — op16_gvr_dualthresh (GVR top-K secant-framework optimization, B300)

> Paste into a fresh session on a B300 host mounting this NFS workspace to
> continue. Everything is on-disk under `indexer_topk_op_bench/op16_gvr_dualthresh/`.

## GOAL & CONSTRAINTS (from user)
- Base op: **GVR cuteDSL rank-scatter P4** (op#7) = `p4_recursive_digit/src/
  gvr_topk_decode_p4.py` → `op16_gvr_dualthresh/src/gvr_topk_decode_dt.py`.
- Optimize on the report's synth data so that >95% of report cases beat
  Radix-cuteDSL AND SGLang, +40% avg per seqlen. **[shown structurally unreachable
  in this regime — see below]**
- **HARD CONSTRAINT (user, latest):** stay in the **P2 secant** framework
  ("secant bracket-then-refine"); optimize the ITERATION/refine only. Do NOT
  replace secant (sampling was falsified). Two-threshold algo-A meaning
  (clarified): in the SAME secant run, also obtain threshold_1 — either
  (1) single-CTA reusing the iteration path's intermediate thresholds [BUILT],
  or (2) a 2nd CTA searching threshold_1 in parallel, write partial top-K, sync.
- B300 only for now (B200 deferred until B300 target met).
- **Perf claims MUST use the report's exact protocol:** nsys pure-kernel cold-L2
  (512MB evict + eager + NVTX-range sync + nvtx_kern_sum, ×3-median, nsys/event
  ≈0.88). Anchor-check by re-measuring gvr_cutedsl_rs vs report's
  gvr_cutedsl_rs_cold_us (validated 0.976–1.020 comparable).

## STATE (iters committed; full detail in ITERATIONS.md / LEARNINGS.md)
- **iter 0**: two-threshold *band-shrink* FALSIFIED (tax-bound): tightening the
  threshold to shrink the band explodes P2 1.3–2.5x ≥ P4 saving. (`scripts/p4_scaling.py`)
- **iter 1**: sampling-histogram P2 init — host predicted 1-pass + cand~1.1×K,
  9/9 exact. (`scripts/host_sample_p2.py`)
- **iter 2**: implemented `enable_sampled_init` in kernel; **432/432 exact**.
- **iter 3**: nsys A/B (report protocol) — **sampling FALSIFIED on HW**: NET
  SLOWER small/mid N (0.72–0.87×), +5–7% large N only. Root cause = op14 L2 trap
  (input ⊂ L2 ⇒ pass-count savings are L2-cheap; sampling is pure added cost).
  Data: `results/nsys_ab/ab_K512_fp32.jsonl` + `nsys_reps/ab_K512_fp32.nsys-rep`.
- **iter 4**: PIVOT to secant framework (user constraint). Host: **quad interp**
  (inverse-quadratic on last 3 pts) = best stable secant accel (2.602→2.509 evals,
  no regression, helps K1024 large N); **free threshold_1** from the secant path
  gives M≈0.7–0.9×K definite winners at zero extra pass, but band=M0−M is LARGE
  for K512/K1024 (wide window kCC=10K ⇒ M0 2–4K) and SMALL for K2048 (0.4–1K).
  (`scripts/secant_variants.py`, `scripts/host_thr1_free.py`)
- **iter 5 (WIP)**: Scheme X build. DONE: free threshold_1 recording in
  `phase2_secant_search` (gated `enable_dual_thresh`), smem s_thr[4]/s_iscalars[6].
  Baseline still exact (vdiff=0). **PENDING: P4 band-refine.**

## KEY QUANTITATIVE FINDINGS
- rank-scatter P4 is **count-sensitive**, collapse point cand≈1–2K (K1024:
  cand 4K→1K ⇒ P4 9963→271 cyc; K2048: 2K→0.5K ⇒ 10184→576). But collapsing via
  a tight accept window explodes P2 (tax). (`p4_scaling.py`)
- Scheme X (free peel, wide window) band vs P4 collapse point:
  K2048 band ~0.5K (past collapse ⇒ **~−18%**), K512 ~1.6K / K1024 ~3.4K
  (before collapse ⇒ ~−3–6%). Band is large because M0 is large (wide window);
  shrinking it = tax; 2-CTA hides threshold_1's search but NOT the main CTA tax.
- **Target verdict**: >95%/+40%-over-Radix is **structurally unreachable** here
  (N≤262144, input⊂L2): small-N Radix floor (6.5µs) < GVR floor (~9µs); large-N
  GVR P3 full-N collect alone (~17µs @262K) ≈ Radix flat total (~19µs). GVR beats
  BOTH radix+sglang in only 6/60 report cases today.

## DO NEXT (finish iter 5 = Scheme X P4 band-refine), then decide
1. Implement P4 dual mode (gated `enable_dual_thresh`). Two options:
   - (A) in-SMEM register-staged partition (no signature change, keeps
     local==integration parity): stage smem_keys[0:M0] to regs, partition into
     band (<thr1) compacted to smem_keys[0:band] + winners (>=thr1) indices →
     output[0:M] (M=s_iscalars[5], thr1=s_thr[3]); then rank-scatter band with
     target_k=K−M, out_offset=M. **Preferred.**
   - (B) global band scratch via a new kernel-signature tensor (simpler but
     changes the op signature → breaks integration parity; avoid unless needed).
   rank-scatter parameterization: `phase4_rank_scatter` uses const_expr(kK) in
   ~5 spots (cum>=kK bin search + final writeout) — add runtime target_k + out_off.
2. Add quad interpolation to phase2 (gated) — small K1024 large-N accel, no regress.
3. Exactness gate (`src/gvr_dt_op.py __main__`, extend to dual path) — need
   value-equiv topk + uniq==K across fp32/bf16/fp16 × K × N × 3 cfgs × seeds.
   Exactness proof: winners (>=thr1, M<K) are definite top-K; band selects K−M;
   M+(K−M)=K. Tie at thr1 → all >=thr1 counted in M (no dup). thr1=NEG_FLT_MAX
   (M=0) ⇒ falls back to plain rank-scatter (safe).
4. nsys A/B (report protocol): `scripts/run_nsys_ab.sh` already measures
   gvr_cutedsl_rs (anchor) + gvr_dt + radix_cutedsl + sglang. Add a dual variant
   or set gvr_dt sampled=False + dual=True in `scripts/nsys_ab.py build_call`.
   Expect K2048 ~−18%, K512/K1024 small/neutral (partition overhead may eat it).
5. HEAD management: leave at baseline (no ship) unless a config nets a real win.

## TOOLS (all working, B300)
- `scripts/p4_scaling.py` — P4-vs-cand clock64 (kC override subclass). floor-vs-count.
- `scripts/host_sample_p2.py` / `host_thr1_free.py` / `secant_variants.py` — host
  (searchsorted) validators for pass-count / free-M / secant-accel.
- `scripts/nsys_ab.py` + `run_nsys_ab.sh` + `parse_ab.py` — nsys report-protocol A/B.
- `src/gvr_dt_op.py` — op-wrapper (sampled / dual flags), exactness gate in __main__.

## GOTCHAS
- clock64 tot ≈ cold wall at single-CTA sizes → cycles OK for phase diagnosis;
  but ALWAYS nsys cold-L2 for the ship claim (op13/op14 lesson).
- The op14 L2 trap: "reduce #passes" is moot while input ⊂ L2 (all DSv4 decode N).
- Override kernel flags via ctor (enable_dual_thresh etc.); do NOT edit the
  vendored p4 kernel — op16 has its own copy `src/gvr_topk_decode_dt.py`.
- Keep the op-wrapper compile path identical to `harness/gvr_cutedsl_rs_op.py`.
- Memory notes: [[project_op16_dualthresh_falsified]], [[project_gvr_topk_falsification_history]],
  [[feedback_kernel_bench_l2_flush_nsys]], [[feedback_html_deliverable_css_only_toggle]].
