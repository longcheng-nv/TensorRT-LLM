# op35_gvr_round2 — RESUME (updated 2026-07-16, mid-L2-verdict)

## 1-minute context
Campaign: 2nd-round GVR opt from PR#16457 HEAD eae374554c (worktree
TensorRT-LLM-gvr-r0). USER target: +40% avg vs PR on op26 REPORT §6 cells
(synth 52 + real 25, BS=1 fp32). op29-HBE excluded (too sglang-like).
NOT merged into PR#16457 — separate follow-up PR after validation.
Distinct from concurrent op35_apex_topk (new-algorithm campaign, other session).

## State
- Proposal gap-analysis: H1 (log-falsi) + H2 (cluster fallback) ALREADY in PR;
  H4 16-bit is fp32-metric-irrelevant. See ANALYSIS.md.
- iter0 nsys 4-arm oracle (results/nsys_oracle_decomp.csv): P4blk med ~37%
  dominant; UB(zero P4blk)=1.578, UB(zero P3+P4blk)=1.771. NCU: cs8 barrier
  stalls 61%; cs1 small-N icache 31%+barrier 26% (latency chain).
- FALSIFIED this campaign: H3-tail qfracs (vseed covers it); p4_fused_hist
  (P3 scan pollution); scatter-atomic hypothesis (wash); launch-cfg refinement
  (best-of ceiling 1.025); B1 whole-window skip (replay ceiling ~0 real).
- LIVE bundle iter3 (nsys ×3 verdict RUNNING, logs/nsysab_*.log):
  skip_h1 + p4_fuse_mmz + kNumBins_override_k2048=512.
  L1: K2048 cells +20-37% kernel-est; kb512 must be K2048-ONLY
  (K1024 pro_1024k regressed 0.84 with global kb512 — fixed per-K).
- B1 (warp,window) sideband ceiling exists (replay: 56-79% of P3 at N>=64K)
  but P3 share only 0-26% -> expected ~1.03-1.07 overall; not yet built.

## Preflight (per box)
1. cutlass 4.5.0 machine-local: pip install --target /tmp/r0val/cutlass450
   nvidia-cutlass-dsl==4.5.0; export PYTHONNOUSERSITE=1
   PYTHONPATH=/tmp/r0val/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/r0val/cutlass450:<op35>/gvrpkg_head:<op35>/variant
2. Baseline snapshot = gvrpkg_head/ (== PR HEAD eae366..eae374554c kernel).
   Variant = variant/gvrpkg35/ (all op35 flags, default-off = byte-identical).
3. nsys artifacts to /tmp/op35_nsys only; env -u GITHUB_TOKEN -u HF_TOKEN.
4. GPUs: b200-081 all 8 cool. Paired A/B same-GPU per cell.

## Harness map
- scripts/ab_op35.py       L1 event paired A/B (--var-flags JSON; *_k2048 suffix
                           applies a flag only at K=2048 cells)
- scripts/nsys_ab.py + drive_nsys_ab.sh   L2 x3-round nsys verdict
- scripts/parse_nsys_ab.py L2 parser -> results/nsys_ab_verdict.csv
- scripts/nsys_oracle.py + parse_oracle.py  4-arm phase attribution
- scripts/replay_b1.py     B1 host replay (rung 1)
- scripts/ncu_cell.py      NCU single-cell attribution

## Next steps
1. Parse L2 verdict (parse_nsys_ab.py) -> 3-axis + no-cell-lost check.
2. If green: iter3 = harvest candidate for follow-up PR (kernel diffs live in
   variant/gvrpkg35; port = 3 small blocks + GvrParams K2048 kNumBins 512).
3. Remaining unexplored: distP4 (cluster handoff2 + leader-P4 parallelization,
   biggest remaining block), P4 warp0-ized searches (~2-3 barriers), B1 sideband.
4. Honest bound: realistic stack ceiling ~1.15-1.25x avg vs the +40% ask —
   document double-lock unless distP4 surprises.
5. Final deliverable: bilingual HTML report + COST.md + memory + commits.
