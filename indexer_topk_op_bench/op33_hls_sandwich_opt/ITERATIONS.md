# op33 iterations — HLS-op27 sandwich optimization (target: beat op27_hls avg +30%, BS=1 fp32)
Env: B200 sm_100 GPU1, cold-L2 flush (256MB) + CUDA-graph (L1 screen); nsys = ship arbiter.
Incumbent = op27_hls (gvr_ms_auto @ op27 HEAD, OP21_FB_LOGFALSI=1 OP27_K2048_TAIL=1).

## iter0 — 2026-07-13 — SETUP: bucket + full-context save + harness validated
- Created op33 bucket; saved PLAN.md + SESSION_CONTEXT.md (full carryover) per user "保存所有上下文".
- Harness (scripts/harness.py) reuses harness/sweep_nsys.build_call("gvr_ms_auto",...) + cold-L2 +
  tie-aware exactness. VALIDATED: op27_hls K512 N8192/16384 BS=1 fp32 build+exact OK, ms_path=ms_1cta.
- L1 baseline (NOISY): K512 N8192 ~18µs / N16384 ~20µs. nsys baseline pending (the A/B floor).
- Next: (1) nsys baseline over K512/1024/2048 × N seqlen at BS=1 fp32 (the +30% floor);
  (2) D1 probe = warp/register band tie-select (sglang INSIGHTS-P3, P4 barrier-bound) — highest promise.
