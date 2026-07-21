# R3 campaign ledger — gvr-topk-r3 (beyond-champion)

Campaign id: `e5q1zgrfhs0z57dj6850kc444r` (KF managed B200, effort high,
6 agents/round: 2×fable-5(high) + 2×gpt-5.6-sol(high) + 2×n3-opus-4.8;
max_rounds 20, max_duration 8h, max_cost $800, stagnation 4).
Node: umbriel-b200-027 (8×B200). Started 2026-07-21 13:40Z.

## Decisions

- **D1 (baseline packaging workaround).** `kf campaign prepare` with
  `--baseline-solution` failed: the platform baseline evaluator does not stage
  campaign assets (0/28 workloads — safetensors missing). Workaround: champion
  per-workload platform timings extracted from campaign-1 trace of kernel
  `c74fb3c0` (28/28 PASSED, geomean 10.76 µs) → `gvr-topk-r3/baselines.jsonl`;
  champion full source inlined into prompt.md v2 instead. c74fb3c0 timings ==
  c74f_sbx timings on this subset (no cell in the sbx graft rung 8448<n≤16896).
  Residual risk: the sbx graft raised `topk_small` `__launch_bounds__` 768→1024,
  which could shift register allocation on small-n rungs; local ab_sbx showed
  no give-back, accepted.
- **D2 (PR head moved).** PR#16457 head advanced e6fdbfac3d → `b14ec40e1b`
  (182 commits; GVR deltas: P4 bracket-window histogram + multi-level
  refinement, lane-parallel bin-search, redundant-warp sync reduction, parity-
  buffered DSMEM count exchange, float-domain bin clamp; only
  `gvr_topk_decode.py` changed among gvrpkg files, 210833→247384 B).
  `gvrpkg_head/` refreshed in place; old arm kept at `gvrpkg_e6fd/`.
- **D3 (foreign GPU load).** Intermittent short-lived foreign job bursts
  observed (67-77 GB, up to 83% util), hopping GPUs (0-3 then 7). Full grids
  run on 7 shards (GPUs 0-6) when GPU7 busy; contamination watched via
  per-rung pr_cold anchors + per-batch p95.

## Timeline

| ts (UTC) | event |
|---|---|
| 13:33 | prepare attempt #1 with baseline_solution → platform asset-staging gap (D1) |
| 13:40 | campaign started (baselines.jsonl path), monitor armed |
| 13:47 | gvrpkg_head refreshed to b14ec40e1b, import OK |
| 13:52 | 28-cell probe champh2 (GPU6): champion vs NEW head cold gm **1.7193**, 0/28 reg, 28/28 exact |
| 13:56 | full 865-cell grid champh2 launched, 7 shards GPUs 0-6 |

## Round log

- **Round 1** (13:40–~17:45Z, 6 agents, 17 kernels): best internal 0.9956 =
  verbatim champion resubmission (`821e5e5f topk_champion_final`, diff=0 vs
  c74f_sbx) → direct calibration of platform eval noise: identical code scores
  −0.4%; an agent also logged the same solution.json timing 17/23/20 vs
  23/28/26 µs across runs. `0d057e1e` = trivial rebase (0.9899).
  Only genuine variant: **`5f3daaf8` (0.9926)** — warm-hint min-threshold
  filter in coop pass-0 + final collect, gated n≥512K, with a provable
  ≥k-admission superset argument (min over logits[pre_idx] ⇒ pool ≥ k ⇒
  exact regardless of hint quality). Harvested to `harvest/r3_5f3daaf8/`;
  local probe pending GPU quiescence.
  Insights (40): grid.sync barrier stall ≈48% of large-n runtime (NCU);
  falsified: hand-rolled atomic barrier, block-count sweeps, TMA/cp.async
  hist prefetch, warp-agg (__match_any) hist accumulation, 15-bit 2-pass smem
  hist, champion+hier hybrid, 3 pre_idx threshold grafts (unGated), tight
  T_seed, single-1024-block collapse. Round 2 launched ~17:45Z.

- **Local-timing pause**: from ~17:40Z an 8-GPU foreign job occupies all GPUs
  (~118 GB resident, 1–17% util bursts). All local probes/grids paused until
  quiescence (monitor armed) per no-probes discipline.

## Barrier-ordering study (engineer variants of 09d13c81)

Measured cost of FORMAL memory ordering in the hand-rolled grid barrier
(28-cell probes GPU1, clean anchors):

| variant | ordering | cold gm vs PR | note |
|---|---|---|---|
| 09d13c81 as-harvested | none (relaxed intrinsics) | ~1.72-1.75 | fastest |
| + __threadfence pair | full membar.gl | 1.5657 | −11% |
| acq_rel asm barrier | scoped acq/rel | 1.6142 | −8% |
| relaxed asm (clobber only) | none | ~1.65 (6-cell 1.55) | clobber alone −5-7% |
| surgical (relaxed spin + trailing acquire) | scoped | 1.6146 | ordering cost is intrinsic: release-add must wait for the block's L2-pending writes on the critical path |

Conclusion: 09d13c81's win comes precisely from omitting barrier ordering +
avoiding the cooperative-launch premium. Safety argument for shipping
fence-less ON THIS PATTERN: (1) merged-hist lines are never plain-read before
the barrier within a launch (first plain touch is post-barrier), (2) L1 is
invalidated at kernel-launch boundaries, (3) pre-barrier writes are L2 atomics
⇒ post-barrier plain loads must miss L1 and fetch fresh from L2. Constraint
documented: any future edit that plain-reads merged hist pre-barrier, or an
L1-persistent-across-launch arch, breaks this. Flag for production port review.

## Verdicts

| tag | arms | cells | cold gm | regs | exact | notes |
|---|---|---|---|---|---|---|
| champh2_probe | c74f_sbx vs PR@b14ec40e1b | 28 | 1.7193 | 0 | 28/28 | GPU6 probe |
| r3a_5f3d | 5f3daaf8 vs PR@b14ec40e1b | 28 | 1.7158 | 0 | 28/28 | GPU6; vs champion: ALL 0.9985, n≥512K activation zone 1.0001 → **WASH, no displacement** (hint filter doesn't pay on radix-scan skeleton) |
| **r3grid09d1** | 09d13c81 vs PR@b14ec40e1b | 865 | **1.7553** | **0** (min 1.009) | **865/865** | 3-shard GPUs0/1/5; anchors med 1.002 all rungs ≤1.045; vs champion gm 1.0428 (coop rungs +6-11%, small-n ≈1.00) → **new composite** |
| r3c_fence | 09d1+threadfence | 28 | 1.5657 | 0 | 28/28 | REJECTED −11% |
| r3d_relacq | 09d1 acq_rel asm | 28 | 1.6142 | 0 | 28/28 | REJECTED −8% |
| r3f_surg | 09d1 surgical rel/acq | 28 | 1.6146 | 0 | 28/28 | REJECTED — ordering cost intrinsic |
| r3g_30e7 | 30e79029 vs PR | 28 | 1.8150 | 0 | 28/28 | GPU1, anchors med 0.991; **vs champion +6.2%** (contiguous-slice scan on top of 09d1 barrier); full grid running |
| ~~r3b_09d1~~ | 09d13c81 vs PR | 28 | ~~2.3698~~ | — | 28/28 | **INVALIDATED** — foreign job at 100% util GPUs 1-7 during run (pr arm inflated 19→26 µs); my quiet-check echo was unconditional (scripting bug, fixed to gated form). Exactness (load-independent) retained: 28/28. Re-probe pending quiescence |

- **09d13c81** (r2, internal 1.0351): replaces `cudaLaunchCooperativeKernel`
  with regular launch + hand-rolled sense-reversing global barrier (generation
  token ⇒ no per-launch reset), grid sized to co-residency. CAUTION for ship:
  barrier uses relaxed atomics with no __threadfence — memory-ordering risk on
  paper even if exact in practice; if it wins, add fence + re-measure.
- 09d13c81 partial probes (r3b2 GPU5 full-28, r3b3 GPU5 flash-only), per-cell
  anchor-gated (±6% vs champh2 refs): clean-anchor cells show a REAL win vs
  champion — v32_32k 1.19-1.23, v32_256k 1.17-1.18, v32_128k 1.06-1.10,
  pro_128k 1.13-1.18, pro_512k 1.05-1.13, pro_1024k 1.06-1.07,
  flash_128k 1.08-1.11, flash_32k/4k ≈1.00, v32_4k ≈1.00 (single-CTA path
  untouched, as expected). flash_512k/1024k anchors dirty in BOTH GPU5 runs
  (foreign bursts; cand times ≈ champion 0.98-1.00 but unverified).
  → genuine displacement candidate; full-865 verdict queued for quiescence,
  batched with any further round-2 winners.
| **champh2** | c74f_sbx vs PR@b14ec40e1b | 865 | **1.6770** | **0** (min 1.018) | **865/865** | 7-shard GPUs0-6; Bar-1/2/3 denominators; worst cells all N=16387 (graft-rung boundary, 1.02-1.08) |

## Anchor checks

- champh2 vs c74fsbx (old-head grid): per-cell `pr_cold(old)/pr_cold(new)`
  overall median **1.005**, p95 1.058; rung medians 0.995–1.048 (worst
  pro_4k 1.048 — small-n launch noise). No drifted rung. Conclusion: PR head
  b14ec40e1b ≈ e6fdbfac3d on this 865-cell envelope (the 07-20/07-21 P4
  bracket/kb512 commits do not materially move these cells); champion start
  vs current head = 1.6770 (vs 1.6828 on old head), consistent with the
  1.005 anchor shift.
