# op37 P4 optimization campaign (from §9e/§9f attribution + PROPOSAL_P4_OPT)

Host umbriel-b200-093. Variant `variant/gvrpkg37` = copy of the kf PR#16457
head (`op26_r0_upstream_port_report/kf_campaign/gvrpkg_head/gvrpkg`), all
flags ctor-default OFF (base arm byte-equivalent). Splice order:
splice_d2a.py → splice_d2b.py → splice_d1a.py.

## Levers implemented

- **D2a `p4_rs_rw_search`** — redundant-warp coarse/fine rank search inside
  phase4_rank_scatter (ports the proven `_kth_bin_search_rw` idiom from the
  snap path); 1 barrier replaces 3 (coarse) / 4 (fine); results in registers
  on every warp. Exact-path only.
- **D2b `p4_fine_skip`** — tiny-bin fine skip. Probe (probe_d2b, 865 real
  cells): cnt[b*] ≤ 128 fires 862/862 (med 23, p90 63). Runtime CTA-uniform
  gate replaces fine recursion + exact-tail/p4tt with one scatter+collect
  pass and an all-thread O(n²) rank select (v2). **Round-1 lesson: the
  warp0 repeated warp-max select serialized ~need redux+ballot latencies and
  LOST 25-40% at K2048** (v32 0.59-0.87×); v2 rank select (no atomics, no
  warp-sync chain) fixed it completely. Fallback cnt>128 = unchanged fine
  path (uniform dynamic branch, barriers legal).
- **D1a `p4_peer_push`** — gather inversion: peers push (st.shared::cluster,
  new prims) their chunks into the leader's SMEM at cluster-rank prefix
  offsets after handoff #2; one extra cluster arrive(RELEASE)+wait publishes;
  leader only sums counts. Prefix accounting mirrors the pull path.

## Validation

validate_op37.py: 200 checks = (13 real cells + 27 adversarial synth) × 5
arms. **All 65 real-cell checks exact for every arm/combination.** 20 FAILs
= 4 plateau/narrow synth cases where the BASE head also fails (known GVR
giant-tie undershoot contract boundary: plateau K512 returns 351/512 valid;
pre-existing, not introduced). d2b fallback exercised via plateau (cnt>128).

## A/B verdict (round 2 = ab2; paired same-GPU nsys cold-L2, 2 shards,
26 real cells = 25 (model×ISL) + tail cell flash_128k_L42; all arms exact)

geomean base/arm (worst cell):
| rung | d2a | d2b-v2 | d2a+d2b | d1a | all |
|---|---|---|---|---|---|
| cs1-small (10) | 1.074 | 1.075 | 1.149 | 1.000 | **1.148** (w 1.104) |
| cs1-mid (7) | 1.053 | 1.091 | 1.154 | 1.001 | **1.151** (w 1.060) |
| cs4 (3) | 1.035 | 1.066 | 1.097 | 1.009 | **1.120** (w 1.085) |
| cs8 (6) | 1.037 | 1.068 | 1.097 | 1.068 | **1.172** (w 1.121) |
| ALL (26) | 1.055 | 1.077 | 1.132 | 1.016 | **1.151** (w 1.060) |

- `all` wins 26/26 cells, range 1.060–1.437 (tail cell flash_128k_L42 1.437 —
  the K512 exact-tail blow-up is gone, subsumes proposal D3).
- d1a is compile-no-op at cs=1 (0.966-1.03 = launch-floor noise); its real
  domain cs8 = 1.068 gm, 6/6 ≥ 1.058.
- Landed mid-range of PROPOSAL_P4_OPT's +12-18% stack estimate on round 1
  of implementation.

## Remaining before ship (per discipline)

1. Full 3-axis verdict: op26 §6/§7b synth best/worst + full 865 real grid,
   bf16/fp16 dtypes, ≤2 concurrent nsys idle node, worst-cell ≥0.975 rule.
2. Sub-stage differential re-measure on the p4pipe twin (only touched stages
   should move).
3. DSMEM checklist for d1a: forced-hit exactness fixture + degrade-path
   (do_cluster_sync=False) coverage — validate covers real cs8 cells incl.
   degrade implicitly; add explicit forced cases before PR.
4. Destination = separate follow-up PR stacked on #16457 (op35 precedent);
   default-ON decision after the full grid.

## Gotchas

- Round-1 lesson above (warp-sync serial chains in `need`-длина loops).
- `env ncu`/PATH ghost + `rm` sandbox denial on this box; use python
  os.remove and absolute ncu path.
- plateau/narrow synth exactness failures are BASE behavior — do not burn
  time re-deriving (characterized: undershoot on giant tie class).

## Ship-verdict envelope ruling (user, 2026-07-22)

Perf verdict axis = **SS7b real decode-capture only, BS=1, fp32**,
K={512,1024,2048} (flash/pro/v32), ISL 4k-1M — the full 865-cell real grid.
Synth best/worst grids and the bf16/fp16 dtype axis are DROPPED from the ship
verdict (driver still supports them for ad-hoc probes). Harness:
ab37_ship.py + drive_ab37_ship.sh (25 real865 batches, arms {base,all},
paired same-GPU nsys cold-L2, <=2 concurrent, batch csv = resume marker);
parser parse_ab37_ship.py emits per-rung/per-(K,N) tables + worst-cell rule.
