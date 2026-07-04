# RESUME PROMPT — op13_gvr_p2cand (GVR cuteDSL P2-secant / candidate-count opt)

> Paste the block below into a fresh Claude Code session (any B200/B300 host that
> mounts this NFS workspace) to continue. Everything needed is on-disk under
> `indexer_topk_op_bench/op13_gvr_p2cand/`.

---

## PASTE-READY PROMPT

Continue the op13 GVR cuteDSL P2-secant / candidate-count optimization in
`/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op13_gvr_p2cand/`.

FIRST read, in order:
1. `op13_gvr_p2cand/ITERATIONS.md`  (iter 0–2 done; iter-3 plan at bottom)
2. `op13_gvr_p2cand/LEARNINGS.md`
3. `indexer_topk_op_bench/op12_gvr_lp/LEARNINGS.md` (prior P2/P4 work — note op12
   measured the rank-scatter P4, which op13 iter-2 SHOWS does not transfer to snap)
4. Memory: project_op13_gvr_p2cand_resume, project_gvr_topk_falsification_history

GOAL (unchanged, from user): start from GVR (cuteDSL) = the plain **snap-P4**
kernel `ops/cute_vendored/blackwell/top_k/gvr_topk_decode.py`; reduce the Phase-4
candidate count to cut P3+P4 time, keeping Phase-2 secant iterations AS LOW AS
POSSIBLE (tune kFTarget / kCC / interpolation / init), staying EXACT with NO P2
fallback across dtype{fp32,bf16,fp16} × K{512,1024,2048} × seq-len × the 3 beta
data-distributions. Keep the torch wrapper identical to `harness/gvr_cutedsl_op.py`
so local single-op perf == tensorrt_llm integration perf. Canonical timing =
cold-L2 (512 MB evict) CUDA-graph + cudaEvent (= `report/sweep.py`); validate with
nsys pure-kernel (the report uses nsys/event ≈ 0.88).

STATE / KEY RESULT (iter 2-5 DONE; full detail in ITERATIONS.md):
- Premise VALIDATED but MODEST + regime-specific. With kCC≈2-3×K + EVAL-OPTIMAL
  kFTarget (host pre-pass picks min-eval kFTarget per kCC → only +1 eval, not +3),
  nsys pure-kernel A/B shows K=512 fp32: ~10% net win at N≤16K + 65K, EXACT, no
  fallback. bf16 only ~3% (ties collapse → P4 already cheap); K=1024 small/noisy;
  K=2048 baseline already lean. Large N≥131K always LOSES (P2-eval explosion) →
  N-dispatch mandatory (narrow kCC only N≤~65K).
- Win = (P4 fraction × cand-cut) − (P2-eval tax); P3 is full-N-READ-bound so never
  shrinks much → no "drastic" cut is physically possible.
- METHOD: event-timed wall is useless at small N (~16µs launch floor, 1.024µs
  quantization) — MUST use nsys pure-kernel. `phase_ab.py` ΔTOT is INVALID
  (production_wall × modified_fractions); use `kcc_walltime_ab.py` / nsys for
  absolute A/B. (This voided an earlier iter-2 "wash" reading.)
- Host pre-pass winners (kCC,kFTarget): K512 kc2x=(1024,1024) kc3x=(1536,1280);
  K1024 (2048,2048)/(3072,2560); K2048 (4096,3686)/(6144,3686).

CONFIRMED (iter 6, ×3-median nsys, all EXACT): K=512 small/mid-N win robust —
fp32 ~7–15% (kc3x best), fp16 ~5–8% (kc2x), bf16 ~3% — at N∈{4K,8K,16K,65K}; N=32K
neutral; N≥131K LOSES. SHIP DECISION RESOLVED: **production indexer logits are
fp32** (dsa.py:97 warmup fp32; decode logits from fp8_paged_mqa_logits = fp32
output, dsa.py:2381/2395/2433) ⇒ the fp32 ~10% regime IS production (K512=V4 Flash,
K1024=V4 Pro). **Ship-worthy** — the change is a pure (kCC,kFTarget)+N-dispatch
tweak to GvrParams, no algorithm change, exact, no fallback.

DO NEXT (iter 7 = BUILD), in order:
1. Copy `ops/cute_vendored/blackwell/top_k/gvr_topk_decode.py` →
   `op13_gvr_p2cand/src/gvr_topk_decode_p2c.py`. N-dispatched (kCC,kFT): N≤65536 →
   K512 (1536,1280) / K1024 (3072,2560) [fp32]; N≥131072 → baseline default. N is
   known at compile/launch → pick override in the op-wrapper compile path (like
   `scripts/kcc_walltime_ab.py::GvrOverride`), keyed by N. Confirm 65K–131K
   crossover under nsys first (parse `results/nsys/kcc_K512_fp32_b.nsys-rep`).
2. Op wrapper mirrors `harness/gvr_cutedsl_op.py` EXACTLY → local==integration.
3. Full-grid exactness: value-equiv torch.topk over 3 beta cfgs × seeds × all N,
   done==1 everywhere (reuse `kcc_walltime_ab.py::_exact`).
4. nsys cold-L2 A/B vs gvr_cutedsl on the report grid; report win + N-dispatch
   table. Optionally wire (kCC,kFT)+N-dispatch into production GvrParams.get.

TOOLS BUILT (all working, B200): `scripts/kcc_host_prepass.py` (eval-opt kFTarget
per kCC), `scripts/kcc_walltime_ab.py` (real modified-kernel cold-L2 wall A/B +
exactness), `scripts/nsys_kcc_ab.py` (nsys pure-kernel A/B; run under nsys then
`--parse <rep>`). nsys reps in `results/nsys/`. (`phase_ab.py` = fraction-split
only, do NOT read its ΔTOT.)

TOOLS ALREADY BUILT (all working, B200 verified):
- `src/p2_replay.py` — parameterized host secant replay (validated 720/720).
- `scripts/validate_replay.py` — replay==baseline check.
- `scripts/sweep_params.py`, `scripts/pareto.py` — host param sweeps (cand vs evals).
- `scripts/phase_ab.py` — REAL-kernel clock64 P1–P4 A/B (baseline vs kCC override).
  Run: `python3 scripts/phase_ab.py --K 512 --dt fp32` (~3–5 min, compiles per N).

GOTCHAS:
- Env: B200 sm_100 here; B300 sm_103 reproduces (report HW-invariant). Use
  `python3` directly; ignore pynvml/FutureWarning noise.
- Override kC/kFTarget via the `GvrTimedOverride` subclass pattern (post-ctor attr
  set, before `cute.compile`) — do NOT edit the vendored kernel for experiments.
- Exactness guard = `K ≤ count_ge(thr_final) ≤ kCC`; "no fallback" (done==1) is
  required (done==2 risks cap-truncation → inexact).
- Sweep ALL 3 beta cfgs (shallow/moderate/deep) + ≥4 seeds for the exactness claim;
  the perf A/B can use beta_moderate seed 0 as the representative cell.
- compile is per (dtype,N,K) and slow-ish; cache or limit the N grid when iterating.
