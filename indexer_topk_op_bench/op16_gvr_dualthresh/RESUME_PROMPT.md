# RESUME — op16_gvr_dualthresh (GVR top-K secant-framework optimization, B300)

> Paste the "PASTE-READY PROMPT" block below into a fresh Claude Code session on
> ANY B300 host that mounts this NFS workspace. All state is on-disk under
> `indexer_topk_op_bench/op16_gvr_dualthresh/` and committed to git branch
> `omni/gvr-dualthresh`.

---

## PASTE-READY PROMPT (copy from here) ============================================

Continue the op16 GVR top-K kernel optimization in
`/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op16_gvr_dualthresh/`
on this B300 host (sm_103). Use the /omni-kernel skill discipline. Answer in Chinese.

FIRST read, in order:
1. `op16_gvr_dualthresh/RESUME_PROMPT.md` (this file — full state below)
2. `op16_gvr_dualthresh/ITERATIONS.md` (iter 0–6 log) + `LEARNINGS.md`
3. `op16_gvr_dualthresh/REPORT.html` (bilingual report — regenerate at the end)
4. Memory: project_op16_dualthresh_falsified, project_gvr_topk_falsification_history,
   feedback_kernel_bench_l2_flush_nsys, feedback_html_deliverable_css_only_toggle

GOAL (user): starting from GVR-rs, on the report synth data, make >95% of report
cases beat Radix-cuteDSL AND SGLang with +40% avg per-seqlen speedup.
HARD CONSTRAINT (user): stay within the **P2 secant** method (or optimized forms)
— "secant bracket then refine"; do NOT replace secant (sampling was falsified).
B300 only for now. **Perf claims MUST use the report protocol**: nsys pure-kernel
cold-L2 (512MB evict + eager + in-range sync + nvtx_kern_sum, ×3-median); anchor-
check by re-measuring gvr_cutedsl_rs vs report seqlen_data.csv (validated 0.98–1.02).

STATUS 2026-07-01 (RESUMED & COMPLETED): op16 is CLOSED — NO-SHIP, HEAD at baseline.
  - K512 fp32 nsys A/B finished: X/rs 0.86–0.98× (net LOSS every N; anchor 0.94–1.02).
    K2048 was 0.95–1.02× (neutral). Data: results/nsys_ab/abX_K512_fp32.{jsonl,nsys-rep}.
  - REPORT.html / ITERATIONS.md / LEARNINGS.md finalized (iter 6 + §7d table + §8 conclusion).
  - Resume-hang root cause fixed: stale torch cpp_extension baton at
    `_build/radix_cuda/lock` (prior host died mid-JIT) hung imports ∞; removed → ~6s.
    See LEARNINGS "Resume gotcha".
  - Nothing left to do unless pursuing op13's cheaper-P2 lever (separate ticket).

--- Original immediate task (now DONE, kept for provenance) ---
IMMEDIATE TASK ON RESUME:
1. Finish/parse the K512 fp32 nsys A/B (a run may have been in flight when the
   prior host expired). Run:
     `bash op16_gvr_dualthresh/scripts/run_nsys_ab.sh 0` (full grid, ~30 min), OR
   per-batch: `nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi
   --capture-range-end=stop -o op16_gvr_dualthresh/results/nsys_ab/nsys_reps/abX_K<K>_<dt>
   -f true python3 op16_gvr_dualthresh/scripts/nsys_ab.py --K <K> --dtype <dt>
   --out op16_gvr_dualthresh/results/nsys_ab/abX_K<K>_<dt>.jsonl`
   Then parse with the inline parser (see below) or adapt `scripts/parse_ab.py`
   (note: files use the `abX_` prefix now; parse_ab globs `ab_K*` — update the glob).
   IMPORTANT: run ONE nsys at a time (concurrent nsys to the same -o path fails
   silently — that bit the prior session). Confirm GPU idle first (co-tenancy
   corrupts cold-L2): `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv`.
2. Based on results, decide: (a) accept Scheme X as net-neutral → HEAD at baseline,
   finalize report; or (b) try the ONE remaining lever to make Scheme X positive:
   replace phase4_partition's smem-atomic slot allocation (2 counters, ~M0 atomics,
   high contention) with a warp-aggregated / prefix-sum compaction (like P3's
   warp+block prefix). This is the suspected cause of the neutral result. Even if
   it works, expect K2048 large-N gain only; K512/K1024 stay small and the
   >95%/+40% target remains structurally unreachable (see verdict).
3. Regenerate `REPORT.html` (CSS-only bilingual toggle, 0 `<script>` — user viewer
   strips inline JS) and update `ITERATIONS.md`/`LEARNINGS.md`. HEAD at best iter.

## END PASTE-READY PROMPT ========================================================

---

## FULL STATE (iter 0–6; details in ITERATIONS.md)

**Base op** = GVR cuteDSL rank-scatter P4 (op#7), copied to
`src/gvr_topk_decode_dt.py`; op-wrapper `src/gvr_dt_op.py` (flags: `sampled`,
`dual`). Kernel flags: `enable_sampled_init` (sampling, FALSIFIED, off),
`enable_dual_thresh` (Scheme X, the current candidate).

- **iter 0**: two-threshold band-SHRINK falsified (tax-bound). `scripts/p4_scaling.py`.
- **iter 1–3**: sampling-histogram P2 init — host predicted −40–60% but nsys
  FALSIFIED (net slower small/mid N, +5–7% large N); root cause = op14 L2 trap
  (input ⊂ L2 ⇒ pass-cut is L2-cheap; sampling is pure added cost). Kept off.
- **iter 4**: PIVOT to secant framework (user constraint). Host: quad interp best
  stable secant accel (2.602→2.509 evals, no regress); FREE threshold_1 from the
  secant path gives M≈0.7–0.9×K definite winners at zero extra pass; band=M0−M is
  large for K512/K1024 (wide window), small for K2048. `scripts/secant_variants.py`,
  `scripts/host_thr1_free.py`.
- **iter 5**: BUILT Scheme X (`enable_dual_thresh`): P2 records free threshold_1
  (s_thr[3], M=s_iscalars[5]); `phase4_dual` dispatcher → `phase4_partition`
  (register-staged: winners ≥thr1 → output[K−M:K], band <thr1 → smem_keys[0:band])
  → `phase4_rank_scatter(target_k=K−M)` fills output[0:K−M]. All P4 methods are
  `@cute.jit` (required — see gotcha). **Exactness: baseline 3/3 + dual 81/81 OK**
  (fp32/bf16/fp16 × K512/1024/2048 × N × 3 cfgs). smem s_thr[4]/s_iscalars[6].
- **iter 6 (current)**: nsys A/B (report protocol). **K2048 fp32: Scheme X NET
  NEUTRAL** (X/rs 0.95–1.02×; anchor 0.99–1.02 comparable) — partition overhead
  (M0-wide 2-pass + smem-atomic contention) ≈ the P4-collapse saving. K512 fp32
  was re-running when the host expired (parse it on resume; expect similarly
  neutral or slightly worse — larger band).
  K2048 data: `results/nsys_ab/abX_K2048_fp32.jsonl` + `nsys_reps/abX_K2048_fp32.nsys-rep`.

## VERDICT SO FAR
- Target >95%/+40%-over-Radix is **structurally unreachable** (N≤262144, input⊂L2):
  small-N Radix floor (6.5µs) < GVR floor (~9µs); large-N GVR P3 full-N collect
  alone (~17µs @262K) ≈ Radix flat total (~19µs). GVR beats BOTH baselines in only
  6/60 report cases today.
- Every lever attacked (P4 two-threshold, P2 sampling, secant accel, free-peel
  Scheme X) yields net-neutral-to-small on B300: theoretical gains are eaten by the
  L2-fit reality + rank-scatter/launch floor + implementation overhead.
- Scheme X is EXACT and no-regression; the one untried refinement is a cheaper
  partition (warp-aggregated atomics). HEAD should stay at baseline (no ship)
  unless that flips a config to a real nsys win.

## INLINE nsys PARSER (files use abX_ prefix)
```python
import sys, json, csv; from pathlib import Path; from collections import defaultdict
sys.path.insert(0,"report"); from parse_nsys_full import parse_rep
R={}
for r in csv.DictReader(open("report/seqlen_data.csv")):
    if r["hw"]=="B300" and r["BS"]=="1":
        g=lambda k:(float(r[k]) if r[k] else None)
        R[(int(r["K"]),r["dtype"],int(r["N"]))]={"rs":g("gvr_cutedsl_rs_cold_us"),"radix":g("radix_cutedsl_cold_us"),"sg":g("sglang_streaming_cold_us")}
tag="abX_K512_fp32"  # edit
kern=parse_rep(Path(f"op16_gvr_dualthresh/results/nsys_ab/nsys_reps/{tag}.nsys-rep"))
cells=defaultdict(dict)
for line in Path(f"op16_gvr_dualthresh/results/nsys_ab/{tag}.jsonl").read_text().splitlines():
    if line.strip():
        rec=json.loads(line)
        if "error" not in rec: cells[(rec["K"],rec["dtype"],rec["N"])][rec["op"]]=kern.get(rec["range_cold"])
# op names: gvr_cutedsl_rs (anchor), gvr_dt (Scheme X), radix_cutedsl, sglang_streaming
```

## GOTCHAS
- **CuTe DSL decorators** (cost 5 turns iter 5): a helper with `range_constexpr`/
  runtime-if called from inside a `@cute.jit` method's runtime branch MUST also be
  `@cute.jit`; call the dispatcher from a `const_expr(...)` branch (NOT a runtime
  if → "@cute.jit must be innermost"). All P4 methods are `@cute.jit`. See
  `.perfbot/learnings/20260701T042539-agent.yaml`.
- ONE nsys at a time per -o path (concurrent → silent failure).
- clock64 tot ≈ cold wall at single-CTA sizes (phase diagnosis OK); ship claim = nsys.
- Override flags via ctor; never edit the vendored p4 kernel (op16 has its own copy).
- git: branch `omni/gvr-dualthresh`, all iters committed; baseline path byte-identical
  (flags default off), always vdiff=0.
