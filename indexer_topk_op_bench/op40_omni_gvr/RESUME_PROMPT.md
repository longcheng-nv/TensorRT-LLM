# RESUME_PROMPT.md — op40_omni_gvr (refreshed 2026-07-23, post-iter2)

## 1-minute context
omni-kernel v2 campaign: optimize GVR top-K on B200 from PR#16457 head pinned
@ e612fc2f38 (vendored src/gvrpkg40b, byte-frozen). Envelope: 865-cell real
decode grid (§7b), BS=1 fp32, stretch goal gm 1.60x vs fresh same-node
baseline (bl0: gm 14.071us), zero-regression band ratio>=0.97, tie-aware
value-multiset exactness. KF firewall: never read kf_campaign/,
op37_bs_scaling/, op38_r3v11_bs/, op39_gvr_bsx/. Node umb-b200-239 (8x B200
healthy). Env: PYTHONNOUSERSITE=1 PYTHONPATH=/tmp/gvrlayers/cutlass450/...
(rebuild: cp -r NFS nvidia_cutlass_dsl -> /tmp/gvrlayers/cutlass450/).

## State (see ITERATIONS.md for full log)
- bl0 baseline: gm 14.071us, 865/865 exact, anchors pro_64k_L30=11.78us(cs1)
  pro_256k_L30=15.13(cs4) v32_128k_L14=18.31(cs8) flash_128k_L42=17.63(cs1).
- iter1 SHIP: arm v1 (d2a/d2b/d1a re-splice) gm 1.1261, 0 reg, 865/865 exact.
- iter2 FALSIFIED: ladder widening K512/K1024 (tax reproduces); K2048 4-rung
  ladder harvested (+1.8% on v32). p1b_cache WASH.
- iter3 IN FLIGHT: v3 = distributed MSB-radix-select P4 (phase4_radix_select
  in gvrpkg40v3), both cs paths, smoke 5/5 exact. v3k = v3 + K2048 ladder.
  Gate running; then 4-arm grid base,v1,v3,v3k (drive_ab40.sh).
- Baseline defects found (upstream-reportable): plateau duplicate-index
  (P2 fail-soft undershoot, no tie-fill) + neartie 54%-of-draws (P4 float-bin
  resolution); v3 radix P4 fixes the neartie class by design (bit-exact).
- iter5 design ready: p2_radix_fallback — full-row distributed radix select
  when admission fail-softs (closes plateau class exactly).
- Residual map post-v1 (results/phase_v1_40.log): cluster P4 gather 2.4-4.4us
  + leader compute 2.8-6.1us (v3 attacks); P2 multi-count latency-wall at
  large N (64 elem/thread dependent chain, T2 lever = ILP/threads/cs);
  P1 gather flat 1.7-2.2us.

## Preflight
- [ ] git log -1 shows latest op40 commit; CONCURRENT SESSIONS commit to this
      repo — on commit ref-lock failure, re-check git status before retrying
      (index race once swallowed op39 files; repair commit 817a943d32).
- [ ] no co-resident driver: poll results/ file growth, not ps.
- [ ] env -u GITHUB_TOKEN -u HF_TOKEN on every nsys/ncu run.
- [ ] harness: scripts/{arms40,ab40,gate40,parse_ab40,drive_ab40.sh}; grid:
      GPUS_LIST="0 1 2 3 4 5 6 7" bash scripts/drive_ab40.sh <arms> <tagdir>
- [ ] gate seeding is crc32 (hash() is salted — never use for seeds).

## Gotchas
- REPORT.html absolute numbers are stale-node; only paired same-GPU ratios.
- PR branch read-only on this machine. Baseline pkg byte-frozen.
- probes NEVER run while a timing grid is on the GPUs.
- parse_ab40 takes ~3-4 min per arm-pair (25 sqlite exports); run in bg.
