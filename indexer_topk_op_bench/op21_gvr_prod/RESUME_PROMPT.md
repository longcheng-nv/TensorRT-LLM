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

## State after iter10 (2026-07-06, docs-only iteration)
- **SHIP_REVIEW.md** = the no-regress ship artifact (P0 17 cells × 3
  dtypes + P1 canaries + dispatch distillation + knob table). Regenerate
  any table: `OP21_NSYS_DIR=results/nsys/<archive> python3
  scripts/nsys_verdict.py msa <dtype> [hw]` (new env override).
- **UPSTREAM_ASSESSMENT.md** = the port plan (Strategy B kernel-variant PR
  chain). **P0 blocker discovered: P4 path C (_p4_band_fine_scatter)
  fixed-depth is NOT unconditionally exact** — same mode upstream reverted
  in ec04147502; must become an exact fallback before any default-ON port
  (LEARNINGS iter10). op21's own gates have a logits-collision blind spot;
  adopt upstream's adversarial multi-bucket cases.
- **B300 cross-check**: fp32 DONE on umb-b300-dp-185 — gm 1.268, 17/17
  (HW-invariant vs B200 1.249). 16-bit sweep died 11/34 (driver outlived
  its session; bf16 K1024 partial 11/11 wins gm 1.097). Completion recipe
  for ANY B300 host = **B300_RELAUNCH_PROMPT.md** (archive dp-185 partials
  to results/nsys/iter10_b300_dp185_partial/ first, then re-run all 51
  cells on one axis). B300_RESULTS.md still pending the 16-bit grids.
- K2048 16-bit BS1 tail: ABLATED and ACCEPTED as structural (ITERATIONS
  iter10 addendum, scripts/ablate_16bit_tail.py): P4 is normal (+0.5us vs
  green K1024 ref); the penalty is floor-resident K-proportional P1
  gather + P1b at cr=1 vs a K-flat radix 16-bit bar. v3.2 geometry only —
  DSv4 Flash/Pro (K512/K1024) unaffected. Do NOT reopen.

## State after iter9 (2026-07-06, commits 2f35c0d192..HEAD)
- iter9 SHIPPED the native 16-bit ladder (OP21_P2_NATIVE=0 = cvt A/B):
  P1b quantizes all threshold columns to the dtype grid at emit (the
  one-point consistency trick — 16-bit-domain compares become
  bit-equivalent to every fp32 phase), both fused ladders get
  set.ge.{bf16x2,f16x2}+add.rn packed counts (int32 flush every 16
  iters) + set.ge.u32 mask collect. Microbench probe/count16_native.cu
  (1.73x N262K single-CTA, 1.21x C8 slice) GO'd the design; real gate
  360/360 with native ON; fp32 binaries untouched (const_expr).
  **16-bit nsys: bf16 gm 1.091 win 15/17 (K1024 column fully green);
  fp16 gm 1.055 win 12/17.** Remaining 16-bit holes: K2048 131K/262K
  BS1 (0.95/0.88, K-proportional P3/P4 tail at cr=1, NOT the ladder)
  + fp16 262K BS1/4 near-par (0.97-0.98). nsys archives:
  iter8_16bit_c8rule/ = pre-native grid.

## State after iter8 (2026-07-06)
- iter8 landed the 16-bit tier + P1 refresh (all on 047 GPU0):
  * 16-bit exactness: real 360/360 (60L x ms/C4/C8 x bf16/fp16, NEW
    scripts/smoke_real_16bit.py) + synth spot checks — the kernels were
    already dtype-generic; no code port needed.
  * C8-at-16bit dispatch rule SHIPPED (`C=8 iff 16-bit && N>=65536 &&
    N>=32768*BS` in gvr_ms_auto): the fp32 C8 falsification does NOT
    transfer to 16-bit (halved scan re-weights the tail; event C4/C8
    1.08-1.14 at the win region, 262K BS16 still collapses 0.71).
  * 16-bit nsys P0 verdict: bf16 gm 1.028 (11/17), fp16 gm 1.043
    (11/17); vs best GVR-family 1.21/1.25. Holes = largeN smallBS
    (262K BS1/4 0.92; K2048 262K BS1 0.86): per-element cost bound
    (cvt+fp32 ladder) — lever = 16-bit native compares (PLAN #6).
  * P1 nsys refreshed on HEAD (iter1 reps archived iter1_ms_p1/):
    gm 0.816 -> 0.901 (5/24 wins); walls unchanged (SGLang midN-highBS,
    radix N4-8K BS64).
  * nsys archives: 16-bit C4-rule grid = iter8_16bit_c4rule/; current
    msa_*_{bf16,fp16}_* = C8-rule verdict grid.

## State after iter7 (2026-07-06)
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

## Next (post-iter10, ranked)
(a) Finish B300 16-bit cross-check: run B300_RELAUNCH_PROMPT.md Option A
    on any B300 host (archive dp-185 partials first); then verdicts
    (nsys_verdict.py msa bf16/fp16 B300) + B300_RESULTS.md +
    `[op21 B300]` commit. fp32 verdict already DONE (gm 1.268, 17/17).
(b) Upstream port PR-1 per UPSTREAM_ASSESSMENT.md Strategy B — FIRST
    code change = P4 path-C exact fallback (LEARNINGS iter10 P0
    blocker), then kernel-variant port + runner extension + tests.
(c) Deferred: fp16 262K BS1/4 residual (~3%); P1 highBS structural wall.
DONE/FALSIFIED (do not retry): C=8 at fp32 262K holes (noise; WINS at
16-bit — dtype-conditional); P1b QBINS=64 highBS; dist-P1/dist-P4;
small-bin P4 SHIPPED iter6; P3 push SHIPPED iter7 (OP21_P3_PUSH=0);
native 16-bit ladder SHIPPED iter9 (OP21_P2_NATIVE=0).

## Open holes (nsys, after iter9)
- fp32 P0 grid: NONE — 17/17 wins, gm 1.249 (rival) / 1.139 (op8).
- 16-bit P0: bf16 gm 1.091 (15/17), fp16 gm 1.055 (12/17); holes =
  K2048 131K/262K BS1 (0.95-0.96 / 0.88 both dtypes; K-tail) + fp16
  K1024 262K BS1/4 (0.977/0.968) + fp16 131K BS8 (0.996 par).
- P1 grid (N 4-16K, BS 64-1024, fp32): gm 0.901, win 5/24; SGLang
  midN-highBS 0.86-0.96 + radix N4-8K BS64 0.68-0.86 walls unchanged.
  Deprioritized.
