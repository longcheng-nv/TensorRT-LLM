# op30 — GVR-base (cuteDSL) BEST/WORST synthetic-data extremes, 10-arm re-test

Date: 2026-07-13 · Node: umbriel-b200-047 (8× B200, all idle) · Branch: omni/op21-gvr-prod

## Motivation

op22 REPORT §1-2 uses the op24 *HLS-relative* poles (BEST = per-K cfg @ hr0.55,
WORST = beta_shallow @ hr0.05). For the original GVR (cuteDSL) single-CTA
baseline — and the seed-heuristic family (GVR multi-CTA, op26 R0) — the
direction is INVERTED by construction (REPORT §4 hr-sensitivity: GVR base
0.760×, i.e. faster on "WORST"; mechanism: mean(logits[preIdx]) seed lands
mid-top-K at hr0.55 → guaranteed undershoot → +1-3 full-row re-scans, while
hr0.05 boundary misses put the seed at the K-th value → ev1 accepted).

op30 re-derives the poles **for GVR (cuteDSL) base itself** (absolute cold-L2
kernel time, not radix-relative), then re-runs the full §1/§2 grids over 10
arms on the new BEST/WORST bundles.

Note the op24 pooled screen suggests GVR-base's true WORST may be hr≈0.75
(P2 evals 3.88 > 2.91 @ hr0.55) — the label swap is NOT symmetric, hence a
dedicated calibration phase instead of reusing op22rr poles.

## Phases

1. **Calibration** (`gen_calib_bundles_op30.py` → `drive_calib_op30.sh` →
   `pick_scen_op30.py`): sweep cfg∈{aggregate, beta_shallow, beta_moderate,
   beta_deep} × hr∈{0.05..0.90 (9 pts)} × N∈{16K, 64K, 256K}, BS=1 fp32,
   arms = gvr_cutedsl (object) + radix_cutedsl (data-insensitivity control),
   nsys cold-L2 canonical, seed = 42+crc32("{K}|{N}")%1e6 (op22 policy).
   Verdict = per-N-normalized geomean; BEST=argmin, WORST=argmax → scen_op30.json.
2. **Bundles** (`gen_bundles_op30.py`): 2 scenarios × 3 models × 3 dtypes ×
   N 4K..1M (N>2K) → bundles_op30/.
3. **Sweep** (`sweep_op30.py` under `drive_nsys_op30.sh`): 54 batches
   (2 scen × 3 sweeps × 3 K × 3 dtype), 10 arms:
   gvr_cutedsl (baseline), gvr_multicta_cutedsl, radix_cutedsl,
   radix_single_cuda, radix_multi_cuda, op25_hls (OP27_K2048_TAIL=0),
   op27_hls (=1), op26_r0auto, sglang_v2 (fp32-only), flashinfer_topk
   (fp32-only). Timing protocol byte-identical to sweep_op22rr.py
   (10 warmup, 50 warm-L2 "w|", 20 cold-L2 "c|" 512MB-evict, eager+sync,
   cudaProfilerApi window). Inline exactness at BS=1 per (arm,K,dt,N),
   sorted-value-multiset criterion. Claim-queue sharding over 8 GPUs
   (atomic mkdir per batch — no static shard collisions).
4. **Parse** (`parse_op30.py`): nvtx_kern_sum (+ nvtx_gpu_proj_sum span for
   sglang_v2 PDL overlap) with per-rep cache → results.jsonl → CSVs.
5. **REPORT.html** (`gen_report_op30.py`): op22-style, §0 calibration,
   §1 seqlen sweep BS=1, §2 BS scaling + hugeN, CSS-only toggles, no JS.

## Grid (identical to op22 §1-2)

- seqlen: BS=1, N ∈ {4K,8K,16K,32K,64K,128K,256K,512K,1M}, N>2K
- bs: N ∈ {4K..256K} × BS ∈ {1,2,4,8,16,32,64,128,256,512,1024,2048}
- bs_hugeN: N ∈ {512K,1M} × BS ∈ {2,4,8,16,32,64}
- K ∈ {512 (v4flash), 1024 (v4pro), 2048 (v32)} × dtype ∈ {fp32,bf16,fp16}

## Provenance / gotchas honored

- All measurements single-node (b200-047), no anchor transfer needed;
  absolute µs valid within this report only.
- nsys under `env -u GITHUB_TOKEN -u HF_TOKEN`; *.sqlite/*.nsys-rep
  gitignored BEFORE first commit.
- op25_hls vs op27_hls coexist in one process: OP27_K2048_TAIL is read per
  call inside `_qfracs_for` and qfracs is part of BOTH compile-cache keys
  (gvr_ms_op.py:2056, gvr_msc_op.py:1596).
- Long-running drivers under setsid; stop via pkill -f sweep_op30 三连 +
  respawn re-check (TaskStop does not kill process trees).
