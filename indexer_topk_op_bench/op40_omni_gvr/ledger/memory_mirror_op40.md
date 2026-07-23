---
name: op40-omni-gvr-campaign
description: "op40 omni-kernel GVR campaign CONVERGED — v7 gm 1.1250/0 reg, 1.60 double-locked infeasible, 2 baseline defects fixed"
metadata: 
  node_type: memory
  type: project
  originSessionId: 3cc8d4b3-7bea-49a3-8d46-4820d0f55bc3
  modified: 2026-07-23T04:54:16.752Z
---

op40_omni_gvr (started 2026-07-23, umb-b200-239 8×B200): omni-kernel campaign
to beat PR#16457 pinned head @e612fc2f38 on the 865-cell §7b real envelope
(BS=1 fp32, K512/1024/2048, ISL 4k-1M). User rulings: stretch goal gm 1.60
(not hard bar), firewall = KF-material only (non-KF op-series ledgers OK),
regression band ratio≥0.97, budget uncapped. RESUME =
`indexer_topk_op_bench/op40_omni_gvr/RESUME_PROMPT.md` (read it first).

Key facts:
- Fresh baseline bl0: gm 14.071µs; anchors pro_64k_L30=11.78 etc.
- e612 head already absorbed p4tt/p4_warp_redundant/R0-ladder — much stronger
  than op26 REPORT-era snapshots; d2a/d2b/d1a (draft PR#16715) NOT in head.
- Scoreboard: v1 (d2a/d2b/d1a resplice) SHIP gm 1.1261/0 reg; iter2 ladder
  widening FALSIFIED for K512/1024 (column tax reproduces), K2048 4-rung
  harvested; iter3 multi-level distributed radix P4 FALSIFIED as perf path
  (gm 0.87 vs v1 — 4-level descent common case, barrier/scan economics);
  v4 = p2_radix_fallback GATE GREEN 138/138 — fixes TWO e612 baseline
  correctness defects (plateau duplicate-index + neartie 54%-of-draws, both
  rooted in P2 fail-soft under-fill, NOT P4; upstream-reportable).
- v5best = v1+K2048 ladder+fallback (ship candidate); v6 = +p3_hist_fuse
  (P3 builds P4 coarse hist, cs1).
- Gotchas: concurrent sessions commit to same repo (index race repaired
  once); gate seeds must be crc32 (hash() salted); probes never during grid;
  parse_ab40 needs bg (>2min).

CONVERGED 2026-07-23: 9 iters (1 SHIP + 6 FALSIFIED). Ship arm v7 =
gvrpkg40v3 + (p4_rs_rw_search, p4_fine_skip, p4_peer_push,
p2_radix_fallback): terminal full-865 gm 1.1250, 0 regressions, gate
138/138 (first arm green on plateau/neartie adversarials the baseline
fails). 1.60 stretch DOUBLE-LOCKED infeasible in-skeleton: phase-floor UB
~1.30-1.35x + icache/fetch wall (47.6% stall, NCU) + occupancy structural
(1-8 CTA/148 SM). Key transferable walls: mega-kernel icache wall (unroll
cuts falsified); content-dependent tails kill every static config lever
(K2048 ladder, p3_hist_fuse, T/cs tweaks) under zero-regression rules.
Baseline defects (upstream-reportable): P2 fail-soft under-fill causes
plateau duplicate-index + neartie 54%-draw failures (cs-independent);
enable_smem_cache inexact at cs8. Deliverable = REPORT.html in bucket.
Pending human: follow-up PR port decision + upstream defect report.