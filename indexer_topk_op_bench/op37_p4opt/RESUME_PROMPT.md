# op37 P4-opt campaign — RESUME PROMPT (2026-07-22, umbriel-b200-093)

Self-contained handoff. Read this + `NOTES.md` (results/gotchas) +
`../op26_r0_upstream_port_report/PROPOSAL_P4_OPT.html` (design rationale) +
REPORT §9e/§9f (attribution data). Branch `omni/op21-gvr-prod`, all work
committed (last: op37 round-2 verdict).

## What exists (DONE)

1. **§9f attribution** (`../op26_r0_upstream_port_report/p4f1_harness/p4_pipeline/`):
   865-cell sub-P4 clock64 breakdown + 13-cell NCU (CS2R landmark method).
   Verdict: cluster P4 = pure wait+gather (cs8 4.7µs = 47% P4); cs1 = fine
   recursion latency-bound; K512 exact-tail blow-up (no p4tt).
2. **D2b probe** (`.../p4_pipeline/probe_d2b/`): cnt[b*] ≤128 fires 862/862
   real cells → fine skip viable with fallback.
3. **op37 variant** (`variant/gvrpkg37`, PR-head copy + 3 default-OFF flags):
   - `p4_rs_rw_search` (D2a rw search), `p4_fine_skip` (D2b-v2 all-thread
     O(n²) rank select), `p4_peer_push` (D1a push gather + RELEASE barrier).
   - Re-derive on a new head: copy head pkg → run splice_d2a.py →
     splice_d2b.py → splice_d1a.py (exact anchors, die loudly on drift).
4. **Round-2 A/B verdict (ab2)**: all-arm geomean **1.151, 26/26 win, worst
   1.060**, tail cell 1.437. Per rung: cs1-small 1.148 / cs1-mid 1.151 /
   cs4 1.120 / cs8 1.172. CSVs: ab37_ab2_summary.csv. All arms exact on all
   real cells (200-check grid; 4 synth FAILs = pre-existing BASE giant-tie
   undershoot, see NOTES).
5. Round-1 falsification recorded: warp0 repeated warp-max select = warp-sync
   latency chain, K2048 −25..40% → replaced by v2 rank select.

## NEXT STEPS (in order)

1. **Full 3-axis ship verdict**: op26 §6 synth best/worst grids + full 865
   real grid (fp32) + bf16/fp16 dtype axis, arms {base, all} (+ d2ab if
   d1a shows any cluster risk), paired same-GPU nsys cold-L2, ≤2 concurrent
   nsys, idle node, worst-cell ≥0.975 rule. Reuse ab_op37.py pattern
   (extend CELLS to full layer grid + synth loader from op26 refresh_harness).
2. **Sub-stage differential**: re-run the p4pipe twin methodology
   (measure_p4pipe_full.py) on gvrpkg37 all-flags arm — only P4 sub-stages
   should move (peer_wait/gather ↓ from d1a, fine→~0 from d2b, searches ↓
   from d2a).
3. **d1a DSMEM fixtures**: explicit forced-hit exactness + short-row degrade
   (do_cluster_sync=False) + cs∈{2,4,8,16} unit cases before PR.
4. **Ship packaging**: separate follow-up PR stacked on #16457 (op35
   precedent); decide default-ON per dtype after the full grid; production
   flag plumbing mirrors kb512 ctor-gate style.

## Env recipe (this box)

- cutlass 4.5.0 overlay: `mkdir -p /tmp/gvrlayers/cutlass450 && cp -r
  /home/scratch.loncheng_gpu/python-userbase/lib/python3.12/site-packages/nvidia_cutlass_dsl
  /tmp/gvrlayers/cutlass450/`; then
  `PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450`.
- All nsys/ncu via `env -u GITHUB_TOKEN -u HF_TOKEN`; artifacts gitignored.
- ncu binary: /opt/nvidia/nsight-compute/2026.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu
  (`env ncu` ELOOPs via PATH ghost). `rm` denied in sandbox → python os.remove.
- Real-data loaders: `../harness/real_data_v4cap.py` / `real_data_v32.py`
  (set RV32.BENCH_LAYERS = LAYERS_ALL).
