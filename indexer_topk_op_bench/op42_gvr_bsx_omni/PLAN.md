# op42 — GVR BS=1-1024 batched campaign (omni-kernel, R4-champion base)

Kickoff 2026-07-24, umbriel-b200-073 (4× B200 idle, all <40 °C — all 4 usable
for sweeps; ship verdicts single-GPU paired). Operator goal set by user
(loncheng) 2026-07-24.

## Objective triple

```yaml
objective:
  base:      KF R4 champion 28dc11f6 (branch kf/r4-champion-final-bs1 @e1049bca,
             local copy champion_ref/ — IMMUTABLE; working copy src/gvr_bsx.cu)
  incumbent: GVR PR#16457 pinned head @04a0900ff7 (gvrpkg_04a0 cuteDSL, native
             batched launch [BS, Npad]) — A/B is ALWAYS against this arm.
  rivals:    []   # per user constraint (5): no dispatch-to-other-operator arms
  envelope:
    data:  §7b real decode-capture cells (REPORT.html op26_r0): 865 cells =
           flash(K=512,21L,9 ISL) + pro(K=1024,30L,9 ISL) + v32(K=2048,58L,7 ISL),
           ISL 4k-1M, fp32. BS>1 = REPLICATE the same row (identical rows).
    BS:    [1,2,4,8,16,32,64,128,256,512,1024]
    dtype: fp32 only
  verdict_axes: [BS=1 slice, BS 2-32 latency band, BS 64-1024 throughput band]
                # all three reported; never headline one
  ship_rule: "geomean over ALL (cell × BS) cases >= 1.40 vs incumbent
              AND per-case >= 0.95 AND exactness green (tie-aware value
              multiset == torch.topk values) AND GVR skeleton preserved"
  hard_constraints:
    - GVR skeleton mandatory: P1 preIdx prior -> threshold guess; P2 secant +
      log-transform threshold solve; P3 candidate collect; P4 exact refine.
    - P1/P4 may be equivalently re-formed for HW efficiency (user grant).
    - Exact algorithm (no approximation).
    - No per-case dispatch to non-GVR top-K operators (radix MAY be absorbed
      into GVR subphases, e.g. P4).
    - Optimization lineage: start from 28dc11f6 ONLY. Do NOT borrow from
      c74f_sbx / compA / compB / R3-lineage mechanisms or prior GVR-abandonment
      analyses. Sanctioned materials: op26_r0 REPORT.html, PR#16457, and the
      op37 BS-decay analysis the user cited in the goal.
  measurement: B200, cold-L2 (512MB evict outside timed window), nsys
               pure-kernel time = only ship arbiter.
```

Envelope set by: user, 2026-07-24 (this file). Seq-len mapping: v32 N=ISL,
V4 N=ISL/4 (cr=4) — matches the real-capture loaders.

## Feasibility priors (Phase 1.4)

- **gm decomposition**: 9515 cases = 865 cells × 11 BS. BS=1 slice inherits
  ~1.65 (champion, must not regress >5% anywhere). To reach overall 1.40 the
  10 BS>1 points must carry gm ≈ (1.40^11 / 1.653)^(1/10) ≈ **1.38** vs the
  incumbent's native batched arm.
- **Math floor at BS=1024** (single-read batched kernel vs head, op37 anchors,
  this node re-anchored before use): flash_32k head 42.4µs vs 1-pass floor
  ~6µs (7×); flash_128k 67.7µs vs ~24µs (2.8×); flash_1024k 385µs vs ~180µs
  (2.1×); pro_1024k 465µs vs ~180µs (2.6×). Floor room ≥2× everywhere ⇒
  1.38 gm NOT a-priori infeasible at the throughput end — the burden is the
  BS 2-32 latency band where head is near-flat (~10-20µs) and amortized.
- **Occupancy structure**: champion is 1-cluster/launch (≤16 CTAs = ≤10.8% of
  148 SMs). Sequential BS launches are linear in BS (op37: µs/row@BS1024 ≈
  BS=1 time). Any win at BS>1 REQUIRES kernel-side row parallelism
  (grid.y row-per-cluster / row-teams) — reg/pipelining levers are void for
  this axis (structural).
- **L2 note**: BS=1 cells ≤1MB (small N) sit in L2; at BS≥64 total working set
  exceeds L2 (60-1024MB) — traffic levers that are idle at BS=1 become live at
  large BS. Identical-row replication does NOT collapse DRAM traffic: each row
  is a distinct allocation row (BS×Npad tensor), so reads are unique addresses.

## Bars (from ship_rule)

- Bar-1: gm(all 9515) ≥ 1.40 vs gvrpkg_04a0
- Bar-2: min(case) ≥ 0.95
- Bar-3: exactness = tie-aware value-multiset vs torch.topk, all cases
- Bar-4: skeleton audit — P1 prior / P2 secant+log / P3 collect / P4 refine
  present and live on the GVR paths

## Verdict grid & cost plan

- L1 screen grid: 27 cells (3 models × {4k,32k,128k,256k|1024k} × spread
  layers) × 11 BS, CUDA-event + cold-L2 — direction decisions.
- L2 ship grid: full 865 × 11 BS, nsys, sharded by cell over GPU0-3, both arms
  same GPU back-to-back per cell. Anchor cell: flash_128k_L36 BS=1 and BS=256.
- Anchor protocol: this is a NEW node (b200-073) — no absolute number from
  b200-026/027 may be quoted without same-node re-anchoring.
