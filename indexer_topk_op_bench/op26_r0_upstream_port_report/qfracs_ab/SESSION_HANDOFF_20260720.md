# SESSION HANDOFF — 2026-07-20 (rung/kb512 series → next session)

## Where things stand

**PR#16457** (`perf/gvr-topk-r0-histogram-ladder`, worktree `TensorRT-LLM-gvr-r0`,
fork = longcheng-nv): head **1128c0544f**, open, mergeable. This series added
2 pushed commits on top of the shipped vseed head eae374554c
(+ 2 interleaved commits from a concurrent session: ruff-format af7351cb74,
test-fix 7408f3bf58):
1. `0d6fc4f1f2` K2048 R0 low rung 0.85→0.6 (vseed-gated; pre-vseed fallback
   keeps (0.85,0.35); K512/K1024 untouched).
2. `1128c0544f` K2048 R0 P4 histogram diet kNumBins 2048→512 (enable_r0-gated,
   all 3 dtypes; base secant byte-identical; placed after kC-diet in ctor).
**CI not yet triggered** — next action on the PR: `/bot run`.

**Measured, stacked K2048 gains vs pre-series ship** (nsys cold-L2, paired
per-cell same-GPU, all cells exact): real V3.2 ≈ +8-9% fp32 / ~+13% bf16;
favorable synth +9-11%+; adverse wash. Rung swap alone: real v32 +2.2% (BS
1-1024 invariant, 8K rung +10-13% every BS); kb512 alone on new base: real
v32 +6.1% fp32 / +10.9% bf16 / +6.3% fp16, no losing cell.

## Verdicts locked this series (do not re-litigate)

- **admission ≠ latency, translation ≈0.1×** of naive memory-bound ceiling
  (measured: rung-study predicted F~3/P~13/V~29% → delivered ~0/~0/+2.2%).
- **V4 (K512/K1024) qfracs: DO NOT touch** — extra explicit rung column costs
  3-7%; column-preserving move is a wash; vseed single-rung design is ~optimal.
- **skip_h1 is DEAD on the new baseline** (ALL 1.000, real v32 −1.5%) despite
  its op35-era contribution; dropped.
- **Per-layer qneeds table: dead** (V4 net ≈0 over a retuned global pair).
- **kFTarget: production cr=4 values ALREADY = K** (upstream heuristic_topk.cuh
  aligned; my ftarget_replay independently re-derived target=K optimal — its
  "current is bad" row was the STALE cr=1 K512/K1024 bench-only values).
- GvrParams vs cpp/kernels/heuristic_topk.cuh: consistent everywhere except
  **cr=1 K512/K1024 kFTarget legacy (384/2560 vs cuh 512/1024)** — non-prod
  combos; optional 1-line hygiene fix (undecided, user hasn't approved).

## Next steps (priority order, from the approved plan)

1. (pending decision) cr=1 K512/K1024 kFTarget hygiene alignment → K.
2. **② real-axis residual vs op26 anchor** (Pro −2% / V3.2 −5%): diff bench
   dispatch_r0_op26 vs prod pick_config, item-wise A/B (qab protocol). ~1 day.
3. **⑥ op36 ship-scheme port** (sgl_bx + 3-arm dispatch + GVR flags; real-7b
   parity 1.017 vs sglang) — PROD_PORT_PLAN delivered, WAITING USER APPROVAL.
   Re-calibrate dispatch thresholds on the new head (rung+kb512 change GVR arm).
4. **④ N-gated merges** (op29 HBE 262K 1.62-1.75× + op27 HLS 1M 1.41): new
   campaign; apply the 0.1× admission calibration + re-validate on new head.
5. **⑤ 16-bit NCU probe** (16-bit trails best external 1.2-1.3×; kb512's
   outsized bf16 win says P4/hist share is bigger there).

## Infrastructure / how to re-run

- **A/B harness**: this dir (`qfracs_ab/`): ops_qab{,3}.py (3-arm qfracs,
  4-arm bundle), sweep_qab{,3}.py, drive_qab{_shard,2,3}.sh, aggregate_qab{,3}.py,
  batches_*.py, RESUME_QAB3.md (full /tmp rebuild recipe), qab{,3}.csv +
  qab3_ckpt/ (all jsonls committed; nsys reps stay on /tmp — env-token rule).
- **Machine**: umbriel-b200-027; /tmp/gvrqab = gvrpkg @ PR-head kernel file
  (currently the 1128c0544f content), cutlass450 symlink → /tmp/gvrlayers.
  GPU2-5 often occupied by others (invisible pids); use 0/6/7.
- **Analysis studies** (decode-capture, NOT git): `E2E_exp/indexer_decode_capture/`
  report `DSV3.2_and_DSV4_..._vs_ContextLength.html` §5c/§5c-CCDF/-b/§5d
  (stats↛hit, GVR seed validation, rung-table benefit, multi-step history);
  scripts src/{topk_logit_stats,topk_rho_resolved,topk_hist_model,preidx_ccdf,
  rung_table_eval,rung_table_real,ftarget_replay}.py; sole report writer =
  src/gen_report_interactive.py (esprima for JS checks; OMP_NUM_THREADS=8 for
  sklearn on this 224-core host).
- **op26 REPORT.html §9c** = the rung/kb512 evidence chain in the op-bench
  report; idempotent injector update_report_rungrecal.py (RUNGRECAL markers —
  survives concurrent-report-writer collisions, proven twice).

## Gotchas learned this series (also in memory)

- Re-sharding a running driver: pkill misses the ppid=1 nsys wrapper → sweep
  respawns; kill nsys pid explicitly, then delete the partial batch's
  jsonl+rep (cell-resume + rep-overwrite silently drops timings).
- Adverse-synthetic K2048 single cells swing 0.64-1.51 across identical runs
  (noise floor) — never tune against them; use 3-4 run gm.
- Cross-arm ctor flags via GvrTopKKernel.launch(**kernel_overrides) are
  cache-key-safe — zero-source-edit A/B pattern.
