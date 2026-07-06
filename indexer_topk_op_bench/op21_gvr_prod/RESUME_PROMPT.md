# op21 RESUME PROMPT — paste this into a fresh Claude Code session

> Keep this file CURRENT: update the "State" and "Next" sections at every
> iteration commit. It is the disaster-recovery handoff for the campaign.

---- PASTE BELOW THIS LINE ----

Continue the op21 GVR production-kernel campaign in
`indexer_topk_op_bench/op21_gvr_prod/` (TensorRT-LLM checkout, branch
`omni/op21-gvr-prod`). Read, in order:
1. `op21_gvr_prod/PLAN.md` — goal, priority grids (P0/P1), physics, red
   lines, gates. NEVER retry red-lined levers.
2. `op21_gvr_prod/ITERATIONS.md` — per-iteration log incl. nsys verdict
   tables (canonical numbers live here).
3. `op21_gvr_prod/LEARNINGS.md` — falsified levers + mechanisms.

## State after iter7 (2026-07-06, commits 2f35c0d192..HEAD)
- iter7 SHIPPED P3 band remote-store push (src/gvr_msc_op.py): the slot
  walk writes band entries straight into the LEADER's smem at the
  pre-known global band prefix via new st.shared::cluster primitives —
  deletes the leader DSMEM gather pass + one cluster barrier pair +
  the count publish. OP21_P3_PUSH=0 restores the gather (A/B).
  **P0 nsys gm 1.249, win 17/17 — first clean P0 sweep; vs op8 gm
  1.139. ALL four 262K smallBS holes closed** (K1024 BS1/4 1.064/1.038,
  K512 BS1 1.064, K2048 BS1 1.115). Verdict grid measured on
  umbriel-b200-047 GPU0 (anchor 20.13us = iter6 axis +0.95%, transfers;
  iter6 grid archived results/nsys/iter6_msa/).
  KEY LESSON: the gather was INLINE in the kernel body => invisible to
  every prior no-op ablation ("scan floor" lied by 1.7-2.4us). All
  phases now behind overridable methods; split probe =
  scripts/ablate_p3_split.py (walk no-op must publish zero counts).
- iter6 SHIPPED small-bin P4 fast paths in phase4_band_rank_scatter
  (host probe: cnt(b*) p50=2 max=4 => path B = warp0 exact register
  ranking of <=32 b* members covers ~100%; path A = big-bin equality
  whole-bin emit; path C = fine recursion extracted to
  _p4_band_fine_scatter, fallback-only). OP21_P4_FAST=0 forces fine.
  P0 nsys gm 1.125, win 13/17; vs op8 gm 1.026. GOTCHA: event A/B
  showed a reproducible-but-FALSE 0.957 at K512 131K BS1 — nsys refuted
  it (codegen jitter); nsys arbitrates single-cell event verdicts.
- iter1 `src/gvr_ms_op.py`: single-CTA `gvr_ms`, mode-5 rank-quantile
  seeding (parallel suffix-scan P1b — NEVER reintroduce tid0 serial bin
  scans), phase1 smem stash, fuse gate `bs<=NUM_SMS && 4K<=kC`.
- iter2 `src/gvr_msc_op.py`: row-chunked C-CTA cluster `gvr_msc` +
  production entry `gvr_ms_auto` (dispatch: C=8 iff K>=2048 && N>=196608
  && BS<=4; C=4 iff N>=65536 && 4BS<=NUM_SMS; else single-CTA).
- iter3 FALSIFIED dist-P1; iter4 FALSIFIED dist-P4 (default-False flags,
  measured references). Generalized: distributing serial fixed parts
  loses to cluster-barrier cost; make the serial phase cheaper instead.
- iter5 SHIPPED phase4_band_rank_scatter (op8 exact rank-scatter P4 port,
  default ON, OP21_P4_RS=0 = legacy snap A/B): replaces the snap
  convergence loop with coarse hist -> fine 256-bin recursion -> one
  scatter pass. **P0 nsys cold-L2 verdict: gm 1.104 vs per-cell best
  rival, 13/17 wins; gm 1.007 vs best GVR-family (op8) — first iter at
  parity with op8.** iter5 also FALSIFIED P1b QBINS=64 at highBS (event
  gm 1.004 — P1b is not the highBS bottleneck; OP21_QBINS knob kept).
- Exactness green every iter: synth x3 seeds + REAL captures (pro 30L
  K1024 + flash 21L K512 + v32 9L K2048, via `harness/real_data_v2`) +
  adversarial preIdx (`scripts/smoke_real_msc.py` = 180 real x C + 36
  adversarial). all-invalid preIdx -> identity emit is the inherited
  vendored contract (matches single-CTA), NOT a bug.

## Environment / cross-machine recovery
- Everything lives on shared NFS (`/home/scratch.loncheng_gpu/...`):
  repo, branch, bucket, real captures, nsys archives. A node timeout
  loses NOTHING except /tmp. On a NEW B200 node:
  1. `cd` this checkout; `git log --oneline -1` should show the latest
     `[op21 iter N]` commit on `omni/op21-gvr-prod`.
  2. Env check: `python3 -c "import torch, cutlass"` (plain python3 has
     both on umbriel B200 nodes; if not, see trtllm-machine-local-install
     skill — but op21 bench needs only torch+cutlass, NOT trtllm).
  3. GPU preflight: `nvidia-smi --query-gpu=index,temperature.gpu
     --format=csv` — idle >50C => do not trust that GPU for timing.
  4. **RE-ANCHOR THE MEASUREMENT AXIS** before comparing to any table in
     ITERATIONS.md: run the anchor cell (K512 fp32 262144 BS1, e.g.
     `GPU=<healthy> nsys ... scripts/nsys_run_auto.py 512 fp32 262144 1
     60`) — expected ~18.0±0.3us on the iter7 axis (post-push HEAD;
     ~19.9us if OP21_P3_PUSH=0). Off by >3% => different-silicon axis:
     re-run the FULL 17-cell grid once (drive_nsys_iter2.sh) to
     establish the new baseline before judging any new lever; rival CSV
     bars (report/{bs,seqlen}_data.csv) were measured on yet another
     B200 — per-cell rival ratios remain the canonical metric, absolute
     us do not transfer across nodes. (Axis history: 035-GPU1 = iter5/6;
     047-GPU0 = iter7, +0.95% vs 035-GPU1 on the anchor.)
- Old host umbriel-b200-035: GPU0 cooling BROKEN as of 2026-07-06 (79C
  idle, +13% mid-sweep drift poisoned the first iter5 grid) — if back on
  035, canonical nsys goes GPU=1. Paired same-process event A/B is
  throttle-immune anywhere.
- Check co-tenancy by output-file growth, not nvidia-smi (sandbox is
  namespace-blind). Run bench/smoke from the bucket dirs.
- nsys MUST run `env -u GITHUB_TOKEN -u HF_TOKEN`; *.nsys-rep/*.sqlite are
  gitignored — NEVER commit them. nsys result archives (NFS, gitignored):
  results/nsys/{iter3_msa, iter5_msa, iter5_gpu0_thermal_poisoned}/ +
  current msa_* = iter6 verdict grid; probe/ = anchor cells.

## Canonical measurement protocol (do not deviate)
- Event screens (CUDA-graph cold-L2 medians) are SCREENING ONLY.
- Verdicts: nsys pure-kernel cold-L2 — `scripts/drive_nsys_iter2.sh`
  (gvr_ms_auto, P0 grid, resumable, per-cell reps `results/nsys/msa_*`)
  then `python3 scripts/nsys_verdict.py msa` (joins per-cell best rival
  from `report/{bs,seqlen}_data.csv` B200 fp32 cold).
- Exactness gate per iteration: `scripts/smoke_exact.py` (single-CTA),
  built-in `src/gvr_msc_op.py <C>` smoke, plus the real+adversarial gate
  (re-create from ITERATIONS if lost: all 60 real layers x C in {2,4,8},
  vd==0 && nneg==0 && uniq==K; adversarial random/half-invalid preIdx).
- Phase attribution: no-op @cute.jit subclass overrides (iter4 pattern).
- Commit per iteration `[op21 iter N]`, `git commit -s`, trailers:
  `Made-with: Claude Code` + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
  Update THIS FILE + ITERATIONS.md + LEARNINGS.md in the same commit.

## Next (iter8 leads, ranked) — P0 goal is MET; remaining = roadmap tail
(a) 16-bit ports (roadmap item, untouched; kNumBins differs — 512/2048 —
    and the small-bin CAP/fine semantics need real-capture re-validation;
    push port should be mechanical but needs the exactness gates).
(b) Dispatch distillation (rules already <=3) + no-regress FULL grid
    (largeN midBS/highBS + P1 canaries must not regress).
(c) B300 cross-check.
(d) Optional stretch: P1 highBS grid (SGLang 0.77-0.94, single-CTA
    scan/ladder structural — the known deprioritized wall).
DONE/FALSIFIED (do not retry): C=8 tier at 262K holes (+0.6-1.3% noise);
P1b QBINS=64 at highBS (gm 1.004); dist-P1/dist-P4; small-bin P4 fast
paths SHIPPED iter6; P3 push SHIPPED iter7 (gather A/B via
OP21_P3_PUSH=0).

## Open holes (nsys, after iter7)
- P0 grid: NONE — 17/17 wins, gm 1.249 (rival) / 1.139 (op8).
- P1 grid (N 4-16K, BS 64-1024): gm 0.816 (iter1 measurement; event
  screens suggest ~5-10% total improvement since; push is a no-op there
  — single-CTA), SGLang-dominated; BS64 smallN vs radix 0.60-0.79 is the
  known deprioritized structural wall. P1b (QBINS) ruled out as cause.
