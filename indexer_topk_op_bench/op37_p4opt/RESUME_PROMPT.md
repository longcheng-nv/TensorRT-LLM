# op37 P4-opt campaign — RESUME PROMPT (2026-07-22 EOD, umbriel-b200-027)

Self-contained handoff. Read this + `NOTES.md` (all results/gotchas) +
`../op26_r0_upstream_port_report/PROPOSAL_P4_OPT.html` (design rationale) +
op26 REPORT §9e/§9f (attribution data). Branch `omni/op21-gvr-prod`, all
work committed (last: phase differential PASS @faae5cc53a).

## Campaign state: SHIP CHECKLIST ALL GREEN — only PR packaging remains

1. **Variant** `variant/gvrpkg37` = PR#16457 head + 3 ctor-default-OFF flags:
   `p4_rs_rw_search` (D2a), `p4_fine_skip` (D2b-v2), `p4_peer_push` (D1a).
   Re-derive on a new head: copy head pkg → splice_d2a.py → splice_d2b.py →
   splice_d1a.py (exact anchors, die loudly).
2. **FULL 865-cell ship verdict PASS** (user envelope ruling 07-22: §7b real
   decode-capture only, BS=1 fp32, K512/1024/2048, ISL 4k-1M):
   gm **1.1284**, win 863/865, worst 0.9945 (pro_4k_L16 launch floor),
   865/865 ≥0.975, exactness green both arms. Per rung: cs1-small 1.1480 /
   cs1-mid 1.1010 / cs4 1.1103 / cs8 1.1366. Tail cell flash_128k_L42 1.514.
   Per (model,layer) table = ship/ship_cells.csv (harness ab37_ship.py +
   drive_ab37_ship.sh, 25 batches, paired same-GPU nsys cold-L2 GPUs 2/3).
3. **d1a DSMEM fixtures 144/144 OK** (validate_d1a_fixtures.py):
   hit1.0/miss/noise/short-row-degrade × cs{2,4,8,16} × 3K × {base,d1a,all},
   incl do_cluster_sync=False with p4_peer_push ON.
4. **Phase differential PASS** (measure_phases_37.py on gvrpkg37t ptime twin,
   splice_ptime_37.py 21 anchors clean): speedup isolated to P4, 25/26 clean,
   1 benign P3/P4 boundary shift at cs8 (d1a push, faster direction).

## NEXT STEP (needs user approval to open PR)

**Follow-up PR stacked on #16457** (op35 precedent). Contents:
- Port the 3 splices into the production package
  (`tensorrt_llm/_torch/.../gvr` upstream path used by PR#16457), ctor-gate
  style mirroring kb512; decide default-ON (evidence says ON for fp32 —
  bf16/fp16 not in verdict envelope per user ruling; keep flags OFF for
  non-fp32 or re-verdict them first).
- PR body: 865-cell verdict tables + fixtures + differential (draw from
  NOTES.md sections dated 07-22).
- GH: use curl+REST with $GITHUB_TOKEN (gh CLI is a browser-opener on these
  hosts); push branch to `fork`, PR on NVIDIA/TensorRT-LLM.

## Env recipe (b200-027)

- cutlass 4.5.0 overlay: `mkdir -p /tmp/gvrlayers/cutlass450 && cp -r
  /home/scratch.loncheng_gpu/python-userbase/lib/python3.12/site-packages/nvidia_cutlass_dsl
  /tmp/gvrlayers/cutlass450/`; then
  `PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450`.
- All nsys via `env -u GITHUB_TOKEN -u HF_TOKEN`; *.nsys-rep/*.sqlite
  gitignored. `find -newermt '-30 minutes'` syntax is bfs-specific on 027 —
  use `stat -c %Y` polls for co-tenancy checks.
- GPUs 0/1 hold a dormant KF-session's memory — use 2-5 for timing.
- Real-data loaders: `../harness/real_data_v4cap.py` / `real_data_v32.py`
  (set RV32.BENCH_LAYERS = LAYERS_ALL).
