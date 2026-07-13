# op33 RESUME (refresh every commit)
1-min: optimizing HLS-op27 sandwich (op27_hls) to beat it avg +30% at BS=1 fp32 K512/1024/2048.
  Borrow op29-HBE (1-pass sample-col, issue-bound) + sglang-v2 (warp/register tie-select) — NO copy.
  Full context = SESSION_CONTEXT.md. Objective = PLAN.md. Ledger = FALSIFIED.md (+ op32/FALSIFIED.md).
Preflight: git HEAD; GPU healthy (nvidia-smi <50°C idle, GPU1); no co-resident driver;
  env OP21_FB_LOGFALSI=1 OP27_K2048_TAIL=1 for op27_hls.
Harness: `OP=gvr_ms_auto CUDA_VISIBLE_DEVICES=1 python3 scripts/harness.py` (K_LIST/N_LIST env).
  build_call("gvr_ms_auto",...) from harness/sweep_nsys; bundles = bundle_data_rr.get_bundle.
Status: iter0 SETUP done (bucket+context+harness validated, op27_hls ms_1cta exact). 
Next: nsys baseline (A/B floor) → D1 warp/register band tie-select probe.
Gotchas: L1 event NOISE at N<=16K BS=1 (nsys only); nsys sqlite token leak (env -u + gitignore);
  commit --no-verify scoped to bucket; N4096 missing for K2048 bundles (use N>=8192).
