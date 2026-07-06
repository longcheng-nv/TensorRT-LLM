# op21 B300 cross-check — paste this into a fresh Claude Code session ON A B300 HOST

Goal: HW-invariance cross-check of the op21 production GVR kernel
(`gvr_ms_auto`) after iter9 — fp32 P0 grid + bf16/fp16 P0 grids, nsys
pure-kernel cold-L2, verdict vs the per-cell best rival from the report
CSVs **B300** rows. This is a MEASUREMENT-ONLY session: no kernel edits,
no dispatch-rule edits (verdicts feed the next B200 session's decisions).

Read first (same NFS checkout, branch `omni/op21-gvr-prod`):
1. `indexer_topk_op_bench/op21_gvr_prod/RESUME_PROMPT.md` — campaign
   state (B200 standings: fp32 gm 1.249 17/17; bf16 1.091 15/17;
   fp16 1.055 12/17 — the 16-bit grids are the iter9 NATIVE-ladder ones).
2. `op21_gvr_prod/PLAN.md` §Gates — measurement protocol; nsys is the
   only verdict axis.

## Steps
1. `cd` this checkout; `git log --oneline -1` must show `[op21 iter9]`
   (ccb22734b0) or later. Env: `python3 -c "import torch, cutlass"`.
2. GPU preflight: `nvidia-smi --query-gpu=index,temperature.gpu
   --format=csv` — idle >50C ⇒ do not use that GPU (see
   env_b200_035 thermal lesson; applies to B300 nodes too).
3. Exactness sanity on this host (NOT a full gate re-run):
   `python3 src/gvr_msc_op.py 4` (from the op21_gvr_prod dir) — 9/9 OK
   expected; any FAIL ⇒ stop, report.
4. nsys grids (from `op21_gvr_prod/`, healthy GPU):
   - `GPU=<g> bash scripts/drive_nsys_iter2.sh` (fp32, 17 cells) — but
     FIRST move any existing `results/nsys/msa_k*_fp32_*` B200 reps to
     `results/nsys/iter7_msa_b200/` (they are the B200 verdict grid —
     do NOT overwrite; the driver skips existing files).
   - `GPU=<g> bash scripts/drive_nsys_16bit.sh` (bf16+fp16, 34 cells) —
     same: first archive `msa_k*_{bf16,fp16}_*` to
     `results/nsys/iter9_16bit_b200/` (iter9 native-ladder grid).
5. Verdicts (rival = B300 rows of report CSVs):
   - `python3 scripts/nsys_verdict.py msa fp32 B300`
   - `python3 scripts/nsys_verdict.py msa bf16 B300`
   - `python3 scripts/nsys_verdict.py msa fp16 B300`
6. Write results to `op21_gvr_prod/B300_RESULTS.md`: the three verdict
   tables + gm/win lines, plus a judgment vs the B200 standing
   (HW-invariant = per-cell ratio pattern similar; call out any cell
   that FLIPS win<->loss by >5%). Per op#9 lesson: judge HW-invariance
   on aggregate verdicts, not per-cell ties.
7. Commit `[op21 B300] cross-check results` (docs + B300_RESULTS.md
   only; *.nsys-rep/*.sqlite are gitignored — NEVER commit), `git
   commit -s`, trailers `Made-with: Claude Code` +
   `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

## Notes
- nsys MUST run `env -u GITHUB_TOKEN -u HF_TOKEN` (the drivers already
  do). Check co-tenancy by output-file growth, not nvidia-smi.
- B300 absolute µs will differ from every B200 table — only the
  per-cell rival ratios matter (rival bars come from B300 CSV rows).
- C8/C4 dispatch thresholds were tuned on B200 (NUM_SMS=148); B300
  NUM_SMS may differ (auto-read in the code via
  torch.cuda.get_device_properties). If 16-bit largeN smallBS ratios
  look far off the B200 pattern, note it as a possible dispatch-boundary
  shift — do NOT retune in this session.
