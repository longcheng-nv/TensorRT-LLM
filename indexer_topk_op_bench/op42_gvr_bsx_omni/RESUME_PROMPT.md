# RESUME — op42 GVR BS=1-1024 omni-kernel campaign

## 1-minute context
Base = KF R4 champion 28dc11f6 (`champion_ref/`, immutable; working copy
`src/gvr_bsx.cu`). Incumbent = PR#16457 pinned head @04a0900ff7
(`../op26_r0_upstream_port_report/kf_campaign/gvrpkg_04a0`, native batched).
Bar: gm ≥1.40 over 865 real cells × BS{1..1024} (identical replicated rows),
per-case ≥0.95, exact (value multiset vs torch.topk), GVR skeleton mandatory,
nsys cold-L2 = only arbiter. Full objective: PLAN.md. Contract: AUTONOMY.md.

## Preflight checklist
- Node umbriel-b200-073, 4× B200; check `nvidia-smi` idle + <50 °C.
- `git log --oneline -3` must show latest `[op42]` commit.
- No co-resident driver: `ls results/nsys/*.log` mtimes stable for 2 min.
- Harness: `scripts/ab.py` (arms gvr_pr|bsx; --smoke = event axis, bsx-only
  validity — head event numbers are GARBAGE, see FALSIFIED.md), parse =
  `scripts/parse_ab.py` → `results/ab_data.csv`.
- Env knob: `GVR_BSX_DENSE_BS=<n>` dense-tier BS threshold (0=off, default 32).

## State (2026-07-24)
- iter0: node anchored; head event-axis artifact documented (nsys only).
- iter1 DONE (nsys, 12 cells × 11 BS, 264 measurements, all exact):
  grid.y row-batching. gm 0.95 overall; BS≤8 1.47-1.75; collapse ≥BS16 on
  cluster tiers (GPC cluster co-residency: CS16 ≈ 8-9 clusters resident).
  Data: results/ab_data.csv (tag iter1 reps in results/nsys/).
- iter2 RUNNING: dense tiers (minimal-CS TB1024 reg, GVR_BSX_DENSE_BS=8),
  7 cluster-tier cells × BS8-1024, tags iter2d8_*. DO NOT edit src/ while
  a sweep runs (per-cell JIT rebuild ⇒ mixed binaries).
- iter3 candidate (queued): route direct-tier (npad≤12288) to reg CS1 dense
  at large BS — direct path's full-row cand+2048-bin hist is ALU-bound at
  BS≥256 (cells dip to 0.86-1.05); hint-based count+collect is less work/row.

## Launch commands (byte-exact)
```bash
cd .../indexer_topk_op_bench/op42_gvr_bsx_omni
setsid ./scripts/run_screen.sh <tag> <gpu> <cell>... > results/nsys/<shard>.log 2>&1 &
# iter2 variant (BS8-1024, dense on): scripts/run_iter2.sh
python3 scripts/parse_ab.py --rep results/nsys/<tag>_*.nsys-rep
```

## Known gotchas
- Head arm host-issue latency 1.2ms at BS<128 mCTA variants (this node) —
  event-axis A/B banned; production hides it under CUDA graphs.
- Cluster co-residency is GPC-quantized: degradation starts BS8 (CS16),
  BS16-32 (CS8), BS32+ (CS4).
- results/exact_*.json per tag; done-markers results/nsys/<tag>_<cell>.done
  (delete marker to re-run).
- Commit after every iteration verdict (user backup demand).
