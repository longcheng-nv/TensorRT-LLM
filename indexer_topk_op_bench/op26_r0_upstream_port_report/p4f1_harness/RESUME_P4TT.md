# RESUME — p4tt tiny-tie fast path campaign (session handoff)

> Written 2026-07-20 ~04:5x UTC on umbriel-b200-027 for migration to another
> 8x B200 node. Read top-to-bottom; everything needed is on NFS in this dir.

## Mission (user-directed)

Implement + validate a tiny-tie COLLECT+SELECT fast path inside
`p4_exact_tail`'s fire branch of the shipped GVR top-K kernel, then commit
to the PR#16457 branch (`perf/gvr-topk-r0-histogram-ladder`, worktree
`../../../TensorRT-LLM-gvr-r0`, head @0d6fc4f1f2) and push to `fork`.
Why: PRO512K_ROOTCAUSE.md — a genuine 2-element boundary tie at real
pro/512k (K1024, N=131075) fires the exact-tail EVERY step; its 4
unconditional radix passes cost ~5.3µs (+45% on that cell). Fast path
(cnt_strad<=CAP=128: one collect pass + thread0 full-key select) is
expected to recover ~80%. Flag `p4_tail_fast` (default ON, False =
PTX-byte-identical to head — hard contract, verified per battery case A).

## State at handoff

- Implementation lives in `gvrpkgprod2/top_k/gvr_topk_decode.py`
  (standalone copy of PR head; `[p4tt]` markers; pristine copy for byte
  checks in `gvrpkgprod2_pristine/`). gvrpkgprod = @eae374554c,
  gvrpkgvseed = @88a563b145, gvrpkg_snapshot = @018251950f (bisect arms).
- Battery `battery_p4tt.py` runs 1-3 done (run3 150/150) BUT Gate C'
  exposed a config-dependent COMPILE bug: at N>=65538 launch configs
  (flash/256k, pro/256k first hit it) JIT fails "dynamic Boolean -> bool
  at compile time". A kernel-cute-specialist subagent (this session's,
  now dead with the session) was mid-fix at handoff: kernel md5 changed to
  712171691b934c5b07345ec60809361f and `battery_p4tt_run4_035.log` was in
  flight, INCLUDING a new mandated battery section: compile+exact smoke
  over ALL launch-contract configs of the 25 real bench cells.
- TAKEOVER CHECK: read the newest battery_p4tt_run*.log — if TOTAL all-pass
  and the 25-config smoke section exists+passes, the fix landed; verify
  `p4_tail_fast=False` PTX byte-identity case still passes. Else re-launch
  a kernel-cute-specialist with the repro (flash/256k L22 via
  real_data_v4cap, launch(..., p4_tail_fast=True)) and the instructions
  recorded in the session transcript (essence: fix the dynamic-if-in-
  const-context with the file's dynamic Int32 predicate idioms; battery
  must cover all 25 launch configs; report md5).

## Validation gates (user-approved protocol; per-axis = REPORT §4 real
## decode-capture, BS=1 fp32 ONLY for now — BS>1 explicitly deferred)

- A': `f1_gates_p4tt.py --fixtures` (env P4TT=on) — 9 fixture cells exact.
- D': `f1_gates_p4tt.py --grid --model {flash,pro,v32}` — 865/865 exact.
  (A'/D' passed on md5 c8cf671b... = pre-fix kernel; MUST RE-RUN on the
  fixed kernel.)
- C': `p4tt_nsys_ab.py` per model under nsys (see gatelogs/p4ttab_*.log for
  the driver invocation shape; x3 in-process rounds, off vs on paired),
  parse with `f1_ab_parse.py <outdir>` (BENCH_L map inside). PASS criteria:
  pro/512k ratio recovers substantially (expect off/on ~1.2-1.4 i.e.
  on/off ~0.75; at minimum on/off < 0.9 there), all non-firing bench cells
  on/off within [0.975, 1.025] x3-median, fixture cells improved vs the
  exact-tail baseline. Anchor context vs REPORT §4 pr column (cross-node
  med ~1.03 on 027).
- After all green: apply the [p4tt] diff to the worktree file
  `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode.py`
  (branch checkout at @0d6fc4f1f2), DCO commit
  `[None][perf] GVR top-K decode: tiny-tie fast path for p4_exact_tail`,
  push to `fork` (updates PR#16457). gh CLI is a browser-opener on these
  hosts — use git push / curl REST with $GITHUB_TOKEN.

## New-node environment recipe (REQUIRED before anything runs)

/tmp farm is NODE-LOCAL — recreate on the new node:
```
WD=/tmp/gvrlayers; mkdir -p $WD/cutlass450
ln -sfn /home/scratch.loncheng_gpu/python-userbase/lib/python3.12/site-packages/nvidia_cutlass_dsl $WD/cutlass450/nvidia_cutlass_dsl
ln -sfn /home/scratch.loncheng_gpu/python-userbase/lib/python3.12/site-packages/flashinfer $WD/cutlass450/flashinfer
export PYTHONNOUSERSITE=1
export PYTHONPATH=$WD/cutlass450/nvidia_cutlass_dsl/python_packages:$WD/cutlass450
# verify: python3 -c "import cutlass; print(cutlass.__version__)"  -> 4.5.0
#         python3 -c "import cutlass.cute as c; print(hasattr(c,'make_fragment'))" -> True
```
(The container .pth force-inserts cutlass 4.6 dsl_packages at sys.path[0];
the symlink farm makes the .pth resolve 4.5.0 instead — see memory
env_newer_container_nvshmem_cutlass §4.) All profiling with
`env -u GITHUB_TOKEN -u HF_TOKEN`; nsys reps stay in /tmp, never commit.
Run everything from THIS dir (op26_r0_upstream_port_report) — background
launches must use absolute cd (session cwd resets bite; two incidents).
Anchor gate on the new node: re-run `newpr_nsys_ab.py`-style old-vs-new or
compare a few `p4ttab` off-arm cells vs gatelogs/ values; cross-node
med <=1.05 acceptable for paired-ratio verdicts (ratios are within-run).

## Evidence & data map (all NFS, all committed except the in-flight kernel)

- PRO512K_ROOTCAUSE.md — root cause + fix proposal (bisect table inside).
- KERNEL_FIX_P4_FINEBIN.md — F1 campaign history (v2-v4, Gate C fails,
  RESOLVED-upstream discovery, F4 log rejection) — background reading.
- pro512k_bisect.py / prhead_rival_ab.py / prhead_rival_parse.py /
  newpr_nsys_ab.py / newpr_ab_parse.py — bisect + rival + old-vs-new tools.
- f1_gates*.py (fixtures/grid gates), p4tt_nsys_ab.py + f1_ab_parse.py (C').
- battery_p4tt.py + battery_p4tt_run*.log; battery_f1.py (F1-era, 164-case).
- gatelogs/ — archived /tmp verdict logs from b200-027.
- Real data loaders: ../harness/real_data_v4cap.py, real_data_v32.py
  (v32 slims already rebuilt all-58-layer on NFS).
- Uncommitted at handoff: gvrpkgprod2 kernel (agent-modified, md5 712171...),
  battery_p4tt.py latest, run4 log — COMMIT after takeover verification.

## Deferred / follow-ups

- BS>1 perf axis (user: wait until after BS=1 ships).
- op36 production port (PROD_PORT_PLAN.md) — separate track, awaiting user.
