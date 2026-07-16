# op35 — GVR round-2 optimization campaign (post-PR#16457)

Started 2026-07-16 on umbriel-b200-081 (8×B200 idle, all ≤33°C).

## Objective triple

```yaml
objective:
  incumbent: PR#16457 kernel @ worktree TensorRT-LLM-gvr-r0 HEAD eae374554c
             (R0 ladder + vseed + per-K qfracs + fb_fix log-falsi + rank-scatter P4
              exact-tail + pick_config/launch). Baseline snapshot: gvrpkg_head/.
  rivals: [op26_r0auto (op-bench anchor), sglang_v2 (reference only, excluded as
           donor per USER: "op29-HBE too sglang-like; we need GVR-based methods")]
  envelope:
    metric_cells: op26 REPORT §6 tables = synth 52 cells (scen{best,worst} ×
      K{512,1024,2048} × N{4k..1M}, BS=1 fp32) + real 25 cells (flash 9 ISL +
      pro 9 ISL + v32 7 ISL, fp32, BS=1). CSVs: ../op26_r0_upstream_port_report/
      {synth_3arm.csv, real_3arm.csv} (cell IDs; times re-anchored locally).
  verdict_axes: [worst, real, best]   # synth-best / synth-worst / real = the 3 data axes
  target: geomean t(PR)/t(op35) >= 1.40 on synth AND real (USER ask).
  ship_rule: "no cell regresses >5%; exactness 3-track green; new machinery
              compile-key gated; enable_r0=False byte-identity preserved;
              NOT merged into PR#16457 — separate follow-up PR after validation"
  hard_constraints: [GVR skeleton retained, no hit-rate/data-dependent dispatch,
                     CUDA-graph compatible, op29-HBE port EXCLUDED by USER]
```

## Target math (sobering, from local CSV analysis)

- All 77 cells have PR times 8–33µs; 35 cells N≤32K sit near the ~9-10µs
  launch/latency floor (op32/op34 double-locked: BS=1 small-N is structural).
- If N≤32K stays 1.0× → remaining 42 cells need geomean **1.85×**.
- If N≤64K stays 1.0× → remaining 33 cells need geomean **2.19×**.
- op23 deterministic bounds (vs Radix): UB=0.851/real=0.599/LB_eff=0.365 —
  headroom exists mostly in tail/fallback price + per-element instruction count.
- CONSEQUENCE: campaign harvests every live lever honestly; if +40% is not
  reachable, deliver the double-locked infeasibility bound (pre-authorized).

## Live levers (post gap-analysis, see ANALYSIS.md)

| # | Lever | Target cells | Prior | Rung |
|---|-------|-------------|-------|------|
| L1 | H3-tail: K2048 qfracs tail ladder (0.75,0.45,0.048) via ctor arg | synth-worst K2048 (op27: 1.15→1.44×) | op27 shipped in ms lineage | config-only → kernel gate |
| L2 | B1: P2-sideband per-tile rung-class → sparse P3 collect | all N≥65536 (P3 = full-N rescan today) | UNTESTED (≠ op16 Scheme X, ≠ ms fusion) | iter0 host replay first |
| L3 | H4: native 16-bit P2/P3 compares | 16-bit cells (NOT in §6 metric set — fp32 only!) → deprioritized for target, keep for follow-up PR | op21 iter9 shipped | microbench |
| L4 | launch: cs=4 nt=512 (+3.8% unconfirmed), cs8 boundary, mbpm | N≥65536 | OPT_CAMPAIGN iter1 | nsys confirm |
| L5 | kC tightening K1024/K2048 (P3 over-collect 3.96×K; kC-diet only done K512-cs1) | large-N + K1024/2048 | op26 kc=3072 shipped K512 | config sweep |
| L6 | fallback-path B2 skip + falsi iteration diet on worst axis | synth-worst large-N | untested | after L2 |
| L7 | intra-CTA warp pipelining / phase overlap ("deepest untouched") | small-N (bounded ≤10%) | never run | opportunistic |

NOTE: §6 metric set is fp32-only ⇒ H4 (16-bit) does NOT move the target metric.
It stays in scope only as follow-up-PR harvest. Priority: L2 > L1 > L4 > L5 > L6 > L7.

## Red lines (falsification ledger — do NOT re-tread)

- ms_auto single-pass fused count+collect: 1.47× SLOWER than PR (07-15 E2).
- Block-max as a separate prepass: op31 fixed-tax wall (must fuse into P2).
- P4-internal refine (fine-hist seed, interp): falsified.
- SMEM residency / traffic savers at N≤262K: L2-trap (op14/op15).
- Cluster DSM at high BS / bigger clusters: GPC-capped.
- P1 self-loop reseeding: falsified on 91k cells.
- P2 secant is NOT a bottleneck (1.46 iters); multi-threshold k=4 wash.
- Small-N single-CTA micro-opt: ≤4% (op8/op32/op34 walls).

## Measurement protocol

- L1 triage: CUDA-event cold-L2 paired same-process A/B (inflates both arms
  equally; ratio-fair). L2 verdict: nsys ×3-median, single-GPU paired,
  ≤2 concurrent nsys, `env -u GITHUB_TOKEN -u HF_TOKEN`.
- Anchor cell: synth best K1024 N131072 (re-run before quoting absolutes;
  >3% drift ⇒ re-baseline).
- Data: SYNTH_POSITIONAL=1 bundle_data_env.get_bundle; real_data_v4cap/v32.
- Exactness: tie-aware value-multiset vs torch.topk; 3 tracks (synth grid,
  real captures, adversarial: uniform/all-equal/near-tie/fp16-collision).
- Env per box: PYTHONNOUSERSITE=1, PYTHONPATH=/tmp/r0val/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/r0val/cutlass450:<op35>/gvrpkg_head
