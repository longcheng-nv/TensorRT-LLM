# RESUME — op12_gvr_lp (optimize GVR cuteDSL top-K to beat SGLang StreamingTopK)

Paste the block below into a fresh session to continue. Everything needed is on disk;
this machine may be reclaimed at any time.

---

## PASTE-READY PROMPT

You are continuing an autonomous GPU-kernel optimization project in this repo:
`/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM`
Workspace: `indexer_topk_op_bench/op12_gvr_lp/`. Read these first, in order:
1. `op12_gvr_lp/ITERATIONS.md`  (full iter log 0–5, all measured tables)
2. `op12_gvr_lp/LEARNINGS.md`   (distilled facts + dead-ends + validated results)
3. Memory `project_op12_gvr_lp_p4_floor.md` (one-line index in MEMORY.md)

### Goal (original ask)
Starting from GVR (cuteDSL) — NEVER modify the in-tree originals; work only in the
copied kernel `op12_gvr_lp/src/gvr_topk_decode_lp.py` — build a NEW single-CTA op that
beats SGLang StreamingTopK across ALL report cells (fp32 input, K∈{512,1024},
N=4K..256K, BS=1..2048, beta_moderate synth, hit-rate 0.6), avg ≥50% faster, keeping
GVR's secant→refine outline. CTA size may change (512/1024).

### Status / verdict so far (B200 sm_100, matches report results_b200/)
- Comparison = 182 cells. Baseline best existing GVR (rank-scatter op#7) `sglang/gvr`
  median 0.868. Battleground = small N (4K–16K), where GVR loses (0.61–0.69×).
- **50%-everywhere is PROVEN physically infeasible at small N**: the P1+P2+P3 floor
  alone is ~1.2× SGLang (N=4K: ~10–11µs vs 12.7), and the shared ~4µs CUDA-graph launch
  + intrinsic secant cap it. Even a FREE P4 → ~1.2× at small N, not 1.5×.
- **P4 is barrier/latency-floor bound, NOT candidate-count bound** → opt-2 (kc_accept
  candidate-shrink) REJECTED by data (adds secant passes for ~0 P4 gain).
- **opt-1 implemented + validated (iter 5)**: EXACT mixed precision (`enable_lp_scan`:
  P1-P3 scan bf16/fp16 half-width, Phase-3 reloads orig fp32 for candidates →
  smem_keys[fp32] → P4 exact). EXACT confirmed (valdiff=0, all N incl 262144).
  Perf (cast included): **+15–23% over fp32 GVR at large N** (262K vs SGLang 1.36→1.59;
  `lp_fp16` best all-rounder, median vs SGLang 1.089); **neutral-to-worse at small N**
  (cast pre-pass dominates). Does NOT fix the small-N wall.
- Best shippable today = `gvr_lp(..., p4_mode="dispatch")` (rs_exact/512 for N<131072,
  snap/1024 for N≥131072). lp_fp16 should likely become the large-N arm.

### How to run (cwd = op12_gvr_lp/)
- Exactness smoke:  `python op_lp.py --p4 lp_fp16`  (modes: rs_exact|snap|rs|fine_hist|
  interp_seed|lp_bf16|lp_fp16|lp_bf16_snap|lp_fp16_snap|nop4(debug)|dispatch)
- A/B vs SGLang (cold-L2, exactness-gated):
  `python scripts/ab.py --cells "512,4096,1;512,65536,1;512,262144,1" --configs "rs_exact:512,lp_fp16:512"`
  config token = `mode:threads[:kc_accept[:input_dtype]]` (input_dtype default fp32;
  for lp_* modes the op casts fp32→bf16/fp16 internally, so use the fp32 default).
  `--cells battleground` | `full` | "K,N,BS;...". Summary prints sglang/new median/mean/min/win/fails.

### Pending tasks (pick up here)
1. **Fold lp_fp16 (or lp_fp16 + snap + 1024 threads) into the large-N dispatch arm**
   and validate the combined op on the FULL BS grid (not just BS=1). `_dispatch_config`
   is in op_lp.py.
2. **Best-case (cast-free) measurement**: time lp with the bf16 copy pre-built OUTSIDE
   the timed region (add a `scan=` passthrough to `gvr_lp`), to cleanly isolate the cast
   cost and show the small-N≈neutral / large-N≈better upper bound.
3. **The ONLY path to small-N wins** (if pursued): a STRUCTURAL rewrite collapsing GVR's
   4-phase / many-barrier pipeline toward SGLang's 2-pass + warp-parallel-threshold shape
   (fewer block barriers) — abandons part of the secant→refine outline. Scope/measure
   barrier count first (P4 has ~7–14 barriers; SGLang fewer effective syncs).
4. Optional: full 182-cell validation of the chosen final op for a published frontier.

### Gotchas (IMPORTANT)
- **GPU co-tenancy corrupts timing** (memory `env_sandbox_ps_nvidiasmi_blind`): same cell
  varied 14.8↔20.4µs across runs. ALWAYS A/B within ONE process; NEVER run two bench
  processes at once; trust within-process deltas + trends, not absolute per-cell µs.
  The `nop4` (early-return after P3) probe gives the cleanest phase-budget signal.
- Commit ONLY `indexer_topk_op_bench/op12_gvr_lp/` (repo has lots of unrelated user WIP).
  `git add indexer_topk_op_bench/op12_gvr_lp/ && git commit -m "[iter N] ..."`.
- Every change must pass the exactness gate (valdiff=0, uniq=K vs torch.topk fp32).
- SGLang is fp32-only, K∈{512,1024} → that's the whole comparison envelope.
- cuteDSL compiles are ~30–60s each (cached per (dtype,bs,n,K,cr,threads,p4mode,kc,...)).
- B200 here (sm_100); report's SGLang numbers are results_b200/, so direct A/B is valid.

### Key commits (branch feat/gvr-v4-dispatch-tuning)
- 2ef37b2885 iter1 copy + harness + config sweep
- 70a7358a35 iter3 P4-budget probe + opt-2 rejected + dispatch default
- df1b082508 iter4 opt-1 upper-bound (half-width input + P1-P3 decomp)
- 02cc87caab iter5 EXACT mixed-precision implemented + validated
