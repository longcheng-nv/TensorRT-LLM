# HBE-C rung-3 — tier-5 kernel pilot: **CONDITIONAL WIN, NO-SHIP for the deployment envelope** (2026-07-13, node 072 GPU0/1/2)

Spec: DESIGN §6 rung 3. Kernel `src/gvr29/sgl_kernel/deepseek_v4/topk_hbec.cuh`
(@cf90e929f0). nsys 4-arm same-batch pilot: anchor `gvr_cutedsl` / rival
`sglang_v2` / `gvr29_hbec` (tier-5) / `gvr29_hbec_off` (fork parity). 3 scen ×
3 K × 4 N {131072..1048576} × 5 BS {1,16,64,256,512} = 240 cells/scen, cold-L2.
Data: `results/pilot_hbec/` (gitignored — nsys token leak). Ratios =
`scripts/parse_pilot_hbec.py` (rv/hbec = speedup vs rival).

## Safety gates (all PASS)
- **gate 720/720 exact** (gate_hbec.py, 3 scen × 3 K × 4 N × 5 BS × 4 hint
  tracks incl adversarial bottom-K + out-of-range garbage + use_hbe=off).
- **fork parity rv/off = 1.001 all 3 scenarios** — when GVR29_HBEC unset the
  dispatch is byte-identical; engaging it never touches non-cluster cells.

## Win/loss map (rv/hbec geomean per slice)
| region | geomean rv/hbec | verdict |
|---|---|---|
| **N ≥ 524288** (stress-probe) | 1.10-1.54 rising with N, BS | strong win |
| **deployment envelope N ≤ 262144** | **0.991** | net WASH-TO-LOSS |
| N=131072 BS≤16 (all K) | 0.88-0.98 | 2-12% LOSS |
| N=131072/262144 BS≥64 | ~1.00 | wash (cluster barely engaged) |
| worst K2048 N=524288 BS≥256 | 0.94 | 6% LOSS |

19 cells lose >5%; the worst is real K2048 N=131072 BS=1 at 0.879 (−12%).

## Mechanism (why the envelope doesn't pay)
- **Small N + BS≤16**: rung-2 predicted it. Cell times are 10-20µs; the
  hint-gather pass + the extra C2 sync are a fixed ~1-2µs tax, and the
  full-scan elimination saves only ~2-3µs at N=131072 → net wash-to-loss.
  BS=1 is worst (no batch amortization of the fixed tax).
- **worst K2048 N=524288 BS≥256**: NOT a fallback (rung-0: worst K2048
  cand p90 3.3-7.3×K < 8×K cap, no overflow). The HIT path's C1 does BOTH
  a mini-hist atomicAdd AND a candidate append for every element ≥ v_spec;
  at worst's loose effective boundary that's ~1-2K elems/CTA of doubled
  atomic pressure, heavier than stock's two clean passes. best/real (tighter
  boundary → fewer candidates) win the same cell.

## Verdict vs the deployment envelope
[[project_indexer_deployment_envelope]]: the main battleground is **N ≤
256K**; N ≥ 512K is a stress probe only. HBE-C's win region (N ≥ 524288)
sits **entirely outside** the envelope, and **inside** the envelope it is a
net wash-to-loss (geomean 0.991, only 10% of cells win >5%). Contrast op29
streaming HBE, which won 1.33-1.75× at exactly 131072-262144 — the core
envelope. **HBE-C is NO-SHIP as a production default.**

## Safe conditional guard (if the ≥512K regime ever matters)
`N ≥ 524288 && !(K==2048 && N<1048576)` → 75 cells, geomean **1.258**, min
**1.018**, **zero cells lose >5%** (excludes the worst-K2048-524288 pocket).
This is a clean, bounded win — but stress-probe-only. Recorded for the
production tier decision (USER, priority-queue item 5); not shipped.

## Ledger write-back
- DESIGN §3 "cluster domain prize 41% of grid, est 1.4-1.9×" — the 1.4-1.9×
  is REAL but only at N≥524288 high-BS, NOT at the report grid's envelope N.
  The realized envelope value is ~0.99. Prize over-stated by conflating the
  cluster domain's cell COUNT with its deployment WEIGHT.
- rung-2's "BS=1 small-N ≈ wash" — confirmed, actually a mild LOSS at 131072.
- The hint's structural asset (row-global, zero-reduce coordination) is real
  but its payoff is dominated by the fixed hint-pass tax below N≈512K.

## Next (recorded, not in scope)
P3 sub-65536 hint tier / P4 HLS Step-3 temporal h-hat — both need the same
envelope-value scrutiny before any build. Production integration = USER.
