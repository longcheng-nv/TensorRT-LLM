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

## State after iter4 (2026-07-05, commits 37fa9040e4..HEAD)
- iter1 `src/gvr_ms_op.py`: single-CTA `gvr_ms`, mode-5 rank-quantile
  seeding (parallel suffix-scan P1b — NEVER reintroduce tid0 serial bin
  scans), phase1 smem stash, fuse gate `bs<=NUM_SMS && 4K<=kC`.
- iter2 `src/gvr_msc_op.py`: row-chunked C-CTA cluster `gvr_msc` +
  production entry `gvr_ms_auto` (dispatch: C=8 iff K>=2048 && N>=196608
  && BS<=4; C=4 iff N>=65536 && 4BS<=NUM_SMS; else single-CTA).
  **P0 nsys cold-L2 verdict: gm 1.054 vs per-cell best rival, 12/17 wins.**
- iter3 FALSIFIED dist-P1; iter4 FALSIFIED dist-P4 (both kept behind
  default-False flags as measured references). Generalized lesson:
  distributing serial fixed parts loses to cluster-barrier cost on this
  family; make the serial phase cheaper instead.
- Exactness green every iter: synth x3 seeds + REAL captures (pro 30L
  K1024 + flash 21L K512 + v32 9L K2048, via `harness/real_data_v2`) +
  adversarial preIdx. all-invalid preIdx -> identity emit is the
  inherited vendored contract (matches single-CTA), NOT a bug.

## Environment
- Host: umbriel-b200-035 (2x B200, usually idle; check co-tenancy by
  output-file growth, not nvidia-smi — sandbox is namespace-blind).
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

## Next (iter5 leads, ranked)
(a) Cheaper P4: port op8's rank-scatter-exact
    (`op8b_gvr_b300/src/gvr_topk_decode_cluster_rs.py`,
    enable_p4_rank_scatter_exact) as the band refine, replacing
    phase4_band_snap. Ablation says P4 = 3.9us (K1024 C4) / 7.0us (K2048
    C8) vs remaining rival gaps of 1.7/4.1us — this is the whole margin.
    Benefits single-CTA gvr_ms too.
(b) P1-grid highBS SGLang gap (0.77-0.94): P1b per-row cost at BS>=256
    (hist zero + 16 scan barriers per row); try QBINS=64 rule for
    bs > NUM_SMS (production-legal BS rule).
(c) 16-bit ports (roadmap iter4 item, untouched).
Then: iter5 roadmap = dispatch distillation, no-regress full grid
(GVR-winning regimes largeN midBS/highBS must not regress), B300
cross-check.

## Open holes (nsys, after iter3)
- K1024 262K BS1/4: 0.92/0.91 vs radix_cutedsl (~20.1us bar).
- K2048 262K BS1: 0.829 vs radix_cutedsl_multi (19.81us bar) [C8 tier].
- P1 grid (N 4-16K, BS 64-1024): gm 0.816, SGLang-dominated; BS64 smallN
  vs radix 0.60-0.79 is the known deprioritized structural wall.
