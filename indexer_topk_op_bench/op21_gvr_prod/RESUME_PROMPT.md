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

## State after iter5 (2026-07-06, commits 2f35c0d192..HEAD)
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

## Environment
- Host: umbriel-b200-035 (2x B200, usually idle; check co-tenancy by
  output-file growth, not nvidia-smi — sandbox is namespace-blind).
- **GPU0 cooling is BROKEN as of 2026-07-06 (79C idle, +13% mid-sweep
  drift poisoned the first iter5 grid): run ALL canonical nsys with
  GPU=1; paired same-process event A/B is throttle-immune.** GPU1 verified
  to reproduce the iter1-4 GPU0 baseline axis within 1.5%.
- Plain `python3` has torch+cutlass. Run bench/smoke from the bucket dirs.
- nsys MUST run `env -u GITHUB_TOKEN -u HF_TOKEN`; *.nsys-rep/*.sqlite are
  gitignored — NEVER commit them.

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

## Next (iter6 leads, ranked)
(a) C-scaling at the remaining 262K BS<=4 holes: try a C=8 tier for
    K1024/K512 at N262K BS<=4 — iter3 measured K1024 C8 as noise-level
    with the OLD expensive P4; the serial tail just shrank, so the
    oversubscription-vs-BW calculus may flip. Watch the BS16 collapse
    (launch geometry) before widening any gate.
(b) P3/P4 residual at BS1: leader DSMEM band gather + K output writes
    (~2-3us ablation); a distributed OUTPUT write is NOT the falsified
    dist-P4 (no cut-bin pick involved) but respect the iter3/4 red line:
    any new cluster barrier must buy more than ~0.5us.
(c) 16-bit ports (roadmap item, untouched).
Then: iter5-roadmap tail = dispatch distillation (rules already <=3),
no-regress full grid (largeN midBS/highBS must not regress), B300
cross-check.

## Open holes (nsys, after iter5)
- K1024 262K BS1/4: 0.950/0.936 vs radix_cutedsl (~20.1-20.4us bar;
  we are 21.1-21.8us — 1.1-1.4us over).
- K512 262K BS1: 0.971; K2048 262K BS1: 0.899 vs radix_cutedsl_multi
  (19.81us bar) [C8 tier].
- P1 grid (N 4-16K, BS 64-1024): gm 0.816 (iter1 measurement, largely
  unchanged; event screen says rank-scatter helped ~5-8% at some cells),
  SGLang-dominated; BS64 smallN vs radix 0.60-0.79 is the known
  deprioritized structural wall. P1b (QBINS) ruled out as the cause.
