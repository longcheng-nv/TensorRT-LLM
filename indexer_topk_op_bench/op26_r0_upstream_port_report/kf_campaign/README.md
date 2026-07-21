# KF campaign `gvr-topk-bs1-real` — code map & actual working path

Campaign `tfb91bvwm972kfyf1bc1trj5e0` (KernelFactory managed B200 pool) vs local
8×B200 verification arm. Baseline = GVR PR head @e6fdbfac3d (`gvrpkg_head/`).
Full narrative: `KF_PROCESS_LOG.html` (bilingual; Figs A–C = designed loop,
**Fig. D = the path that actually ran**; timeline table = every event).

## Outcome (2026-07-21, ~7.5 h wall)

**SHIP BAR MET** — `harvest/r2_c74f_sbx`: **geomean 1.6828× vs PR head, 865/865
exact, zero cold regressions** (865-cell real decode grid, BS=1, nsys cold-L2
paired). Rivals (PR-normalized): vs sglang_v2 ≈1.12 (win 569/865, first
in-tree-family win on the full real envelope), vs radix_cutedsl ≈1.62.
Composite = campaign round-2 winner `c74fb3c0` (a003) + 3-line engineer graft
(`topk_small<17><<<1,1024>>>` rung for 8448<n≤16896).

## Code map

| File | Role |
|---|---|
| `ws/` | Campaign workspace: `definition.json` (SOLBench problem), `workload.jsonl` (28 stratified cells), `baselines.jsonl`, `prompt.md`, `campaign.yaml` |
| `export_cells.py` | §4 real data → 28-cell campaign subset (safetensors + tie/mandatory sets) |
| `monitor_campaign.sh` | Poll loop; exits (→ notifies agent) on round advance / +0.02 speedup / terminal phase |
| `quick_ab.py` | Build candidate (TVM-FFI) + 28-cell CUDA-event smoke A/B |
| `nsys_ab.py` | House-protocol nsys worker: cold-L2 (512 MB evict) + warm, NVTX, paired arms same GPU; `--grid full` = 865 cells, `--cells` filter, `--shard i/m` |
| `run_nsys_ab.sh` | Single-GPU nsys wrapper (probe); `parse_ab.py` → `ab_<tag>.json` |
| `drive_grid_shards.sh` | 8-GPU sharded 865-cell grid → `grid_logs/`, `nsys_reps/` |
| `aggregate_grid.py` | Shard reps → `grid_<tag>.csv` + verdict summary |
| `compare_rivals.py` | Join grid vs REPORT rival sweep (sglang_v2/radix_cutedsl/flashinfer), per-cell PR-arm normalization |
| `gen_diagrams.py` / `add_fig_d.py` | Idempotent SVG diagram injectors (Figs A–C / Fig. D) |
| `EXPANDED_VS_PR.md` | Per-layer tables: champion vs PR head (25 model×ISL groups × all layers) |
| `harvest/` | One dir per harvested candidate (`kernel.cu`, `main.cpp`, `raw_source.txt`); `*_sbx` = engineer grafts |
| `grid_*.csv`, `ab_*.json`, `exact_*.json` | Verdict artifacts (tags: r1a/r1c/r2a2_fixed/r2c2g/c74fsbx = clean; r2a/r2b = INVALIDATED, kept for the contamination record) |

## Actual working path (what Fig. D draws)

1. **Setup**: 28-cell stratified subset from the 865-cell §4 grid → SOLBench bundle → `kf campaign init/prepare/start` (6 agents/round, cuda_cpp, judge bans CUDA-graph & framework imports).
2. **Round 1** (01:19–05:29): ramp 0.29→1.077 internal. Harvest `41a94aaa` → grid r1a: gm 1.316, 137 regs, ALL in N∈[16k,65k] → diagnosis: **dispatch-boundary artifact** (SMALL_N=16384 vs grid npad 16387). Local `sb17/sb17b` probes prove the 1024-thread single-CTA fix direction.
3. **Round 2**: insights cross-pollination (bottom3 complement trick, early-exit) → `0260cee7` 1.284 → `0197c2a1` 1.311 → `c74fb3c0` 1.339 internal.
4. **Contamination incident**: probes launched during sharded grids invalidated two verdicts (r2a "1.7758", r2b "1.7101") — caught by cross-run per-rung `pr_cold` comparison. **Standing protocol since**: serialize all GPU work; anchor-check every grid; quiet-GPU probe first.
5. **Clean verdicts**: `0260cee7` 1.6421/4 regs, `c74fb3c0` 1.6713/5 regs; `compare_rivals.py` shows c74fb3c0 already beats sglang_v2 overall (1.111) with mid-N strongholds left.
6. **Engineer graft**: `c74f_sbx` = c74fb3c0 + sb<17>@1024 rung (its `topk_small` was already `blockDim.x`-parameterized; the earlier 0260cee7 graft needed the NT-hardcode fix and taught the "dup" lesson). Boundary heals (L28 0.846→1.253) with no pro give-back.
7. **Ship verdict**: grid c74fsbx gm 1.6828, zero regs after 60-rep adjudication of two 0.99x borderline cells; anchors clean.
8. **Campaign left running** (round-2 plateau 1.3385): any later harvest must beat the composite to displace it.

## Measurement discipline (hard-won this run)

- NEVER run probes while a sharded grid is in flight (double-driver contamination inflates whole rungs 25–50%).
- Every grid: per-rung median `pr_cold` vs prior runs of the same PR arm must agree within ~3%; drifted rungs → re-measure on quiet GPUs and patch (`grid_r2a2_fixed.csv` pattern).
- Campaign-internal speedup understates local nsys ratio ~1.3–1.4× (its ~15 µs eval floor); never compare the two scales directly.
- Borderline (<1.02) cells: adjudicate with ≥60 cold reps before calling them regressions.
