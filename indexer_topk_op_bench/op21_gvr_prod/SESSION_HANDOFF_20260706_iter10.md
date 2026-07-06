# op21 session handoff (2026-07-06, post-iter9) — paste into a fresh Claude Code session on any B200 host

---- PASTE BELOW THIS LINE ----

Continue the op21 GVR production-kernel campaign at iter10, in
`indexer_topk_op_bench/op21_gvr_prod/` (TensorRT-LLM checkout on shared
NFS, branch `omni/op21-gvr-prod`, HEAD must be `ccb22734b0` `[op21
iter9]` or later). The previous session (umbriel-b200-047) landed
iter7+8+9 in one day; everything is committed, nothing is lost with the
node.

Read, in order:
1. `op21_gvr_prod/RESUME_PROMPT.md` — campaign state through iter9,
   node-recovery + RE-ANCHOR protocol (do the anchor cell FIRST on any
   new node: K512 fp32 262144 BS1 expected ~18.0±0.3us on the 047 axis;
   >3% off => re-run the full fp32 grid before judging anything).
2. `op21_gvr_prod/PLAN.md` — goal, red lines. NEVER retry red-lined or
   falsified levers (list at the bottom of RESUME_PROMPT).
3. `op21_gvr_prod/ITERATIONS.md` + `LEARNINGS.md` — canonical numbers
   and mechanisms.

## Where the campaign stands (all nsys cold-L2, B200-047 axis)
- fp32 P0: gm 1.249 vs per-cell best rival, **17/17 — goal MET**.
- bf16 P0: gm 1.091, 15/17 (iter9 native ladder). fp16: 1.055, 12/17.
- P1 grid (smallN highBS, fp32): 0.901, structural walls, deprioritized.
- Open holes: K2048 16-bit 131K/262K BS1 (0.88-0.96, K-proportional
  P3/P4 tail at cr=1, NOT the ladder); fp16 K1024 262K BS1/4
  (0.977/0.968); fp16 131K BS8 par.

## Agreed iter10 plan (user-approved priority, 2026-07-06)
1. **B300 cross-check runs IN PARALLEL on a B300 host** — a separate
   session was launched with `op21_gvr_prod/B300_PROMPT.md`. Do NOT
   duplicate it here. When its `B300_RESULTS.md` + `[op21 B300]` commit
   appear, fold the HW-invariance verdict into the ship review.
2. **THIS session's main task: ship review + upstream integration
   assessment** (~half day):
   a. No-regress ship table: one artifact consolidating fp32 P0 (msa
      grid) + bf16/fp16 P0 (msa 16-bit grids) + P1 canaries, with the
      dispatch distillation writeup — C rules are exactly 3
      (16bit && N>=65536 && N>=32768*BS -> C8; K>=2048 && fp32-huge-N
      && BS<=4 -> C8; N>=65536 && 4BS<=NUM_SMS -> C4; else single-CTA)
      + the fuse gate (bs<=NUM_SMS && 4K<=kC), all compile-time keys,
      CUDA-graph compatible. Env A/B knobs: OP21_P4_RS / OP21_P4_FAST /
      OP21_P3_PUSH / OP21_P2_NATIVE (all default ON).
   b. Upstream integration assessment: diff op21's gvr_ms/gvr_msc vs
      the production GVR operator in tensorrt_llm (the rank-scatter P4
      went upstream via PR #15709 — same route). Enumerate which levers
      port (P3 push, P4 small-bin fast paths, native-16bit ladder,
      C8-16bit rule), the code-surface delta, and the e2e validation
      plan (dsv4-pareto-bench GVR ON/OFF A/B). DELIVERABLE = a written
      plan, not the port itself.
3. **GPU-idle filler: K2048 16-bit BS1 tail ablation** (bounded probe,
   ~20 min): extend scripts/ablate_p3_split.py pattern to bf16 K2048
   262K C8 (full/noP4/noWG) to pin the tail; if it confirms the
   K-proportional P3/P4 structural tail, ACCEPT and document — do not
   burn iterations on the lowest-priority cell family.
4. Deferred (do not start): fp16 ~3% residual; P1 structural wall.

## Environment / protocol reminders
- New-node recovery + anchor protocol: RESUME_PROMPT.md §Environment.
  GPU preflight: idle >50C => don't trust that GPU (035-GPU0 cooling is
  broken; 047 was healthy).
- nsys always `env -u GITHUB_TOKEN -u HF_TOKEN`; *.nsys-rep/*.sqlite
  gitignored, NEVER commit. Archives: results/nsys/{iter6_msa,
  iter8_16bit_c4rule, iter8_16bit_c8rule, iter1_ms_p1}/; current msa_*
  fp32 = iter7 grid, msa_* {bf16,fp16} = iter9 native-ladder grid.
- Verdicts ONLY via scripts/nsys_verdict.py msa <dtype> [hw]; event
  screens are screening-only (three codegen-jitter lies on record —
  nsys arbitrates).
- Exactness gates per code change: scripts/smoke_exact.py +
  src/gvr_msc_op.py <C> + scripts/smoke_real_msc.py +
  scripts/smoke_real_16bit.py (16-bit real 360).
- Commit per unit `[op21 iter10]` / `[op21 ship]`, `git commit -s`,
  trailers `Made-with: Claude Code` + `Co-Authored-By: Claude Fable 5
  <noreply@anthropic.com>`; update RESUME_PROMPT.md + ITERATIONS.md +
  LEARNINGS.md in the same commit; update the op21 memory file + its
  MEMORY.md index line at session end.
