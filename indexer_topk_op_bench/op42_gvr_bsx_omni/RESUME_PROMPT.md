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

## State (2026-07-24, session resumed ~09:10 UTC)
- iter6 DONE: uniform sample stride fix — 9-cell nsys: BS256-1024 gm 1.20-1.23
  (min 0.89), BS128 1.01, BS32-64 0.80-0.87; pro_1024k 1.98, flash_1024k 1.38.
- iter7 DONE: tp CS in {1,2,4,8} clusters; dispatch (bs,npad) bands BAKED into
  launcher. Portfolio: direct(<=12288)->bs<256; latency bs<8; dense [8,16)
  big-npad / [64,128) small-npad; tp elsewhere. CS1 for bs>=128.
- M1 82-cell screen (tag m1, cells results/screen_cells.txt, full BS ladder,
  baked dispatch): first launch died with prior session @22/82; RELAUNCHED
  09:08 on this shell's node (hostname reads umbriel-b200-048 — anchor cell
  flash_128k_L02 re-run matched 073 data within 0.5% on head arm, mixing OK).
  4 shards x 15 cells, setsid, idempotent via .done markers.
- GOTCHA: parse_ab.py dedups by (cell,BS,arm) WITHOUT tag — never parse a
  probe/anchor rep before the canonical m1 rep, or the canonical rows get
  dropped. (Happened once; anchor rows stripped + m1 rep re-parsed, CSV clean.)
- After M1: (a) gm projection vs 1.40 bar; (b) weak-cell clustering -> iter8
  levers: BW efficiency (ncu), tail-trim K2048, tp fused U=8 batch loads +
  __ldcs, 3 CTA/SM via smaller smem at K<=1024.

## Older state
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
