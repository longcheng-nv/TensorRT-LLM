# p4tt tiny-tie fast path — BS=1 fp32 real-data verdict (2026-07-20, umbriel-b200-035)

**Question**: does the tiny-tie collect+select fast path (PRO512K_ROOTCAUSE.md
proposal) deliver its predicted gain on the REPORT §4 real-data cells, BS=1 fp32?
**Protocol**: nsys cold-L2, paired same-GPU same-process, x3 interleaved rounds,
launch contract, arms = gvrpkgprod2 `p4_tail_fast=True` (fast) vs `False` (slow
≡ PR#16457 head @1128c0544f — prod2/pristine upgraded this session to include
kb512; pristine `diff`-identical to the worktree head file; OFF arm PTX
byte-identical to pristine, battery caseA). Harness: `p4tt_real_ab.py` +
`p4tt_ab_parse.py`; raw jsonl in `p4tt_results/` (nsys reps stay on /tmp,
env-token rule). flash ran twice (GPU0 + GPU1) after a self-inflicted
dual-driver overlap invalidated the first attempt — both re-runs agree ≤0.4%.

## Correctness

- battery_p4tt **150/150** (this box, full order): OFF==pristine PTX byte-eq
  (3 K), random + planted-pair + 1-ULP-ladder + all-equal + cand==kK,
  CAP boundary 128/129, -FLT_MAX straddle, real pro/512k L30.
  - run1's 2 caseD "failures" were a TEST bug (index-set agreement on a
    massive-tie row where index sets legitimately differ); the 03:09 battery
    already checks value-multisets; kernel unchanged since 02:42.
- nsys sweep exactness: **25/25 cells exact on BOTH arms** (fast + slow).

## BS=1 fp32 §4 results (slow/fast ratio, >1 = fast wins)

| cell | slow µs | fast µs | ratio | x3 range |
|---|---|---|---|---|
| flash 4k–32k (N 1k–8k) | 7.5–9.1 | 7.5–9.1 | 0.984–0.998 | tight |
| **flash 64k (N16k)** | 11.95 | 12.78 | **0.934** | 0.930–0.938 |
| **flash 128k (N32k)** | 12.03 | 13.20 | **0.910** | 0.906–0.918 |
| flash 256k–1024k (cs8) | 13.6–20.5 | 14.2–20.7 | 0.962–0.992 | tight |
| pro 4k | 8.92 | 9.35 | 0.953 | 0.951–0.973 |
| pro 8k–256k | 8.5–14.5 | 8.5–14.8 | 0.982–1.002 | tight |
| **pro 512k (N131k)** | **23.03** | **17.17** | **1.340** | 1.338–1.341 |
| pro 1024k | 19.78 | 19.96 | 0.993 | |
| v32 4k–64k | 9.8–19.7 | 10.0–20.1 | 0.979–0.987 | |
| v32 128k / 256k | 18.5 / 19.3 | 18.3 / 18.9 | 1.013 / 1.019 | |

GPU1 flash confirmation: 64k **0.937**, 128k **0.909** — the tax is
cross-GPU/cross-process reproducible, NOT noise and NOT the GPU0 thermal history.

## Verdict

1. **The fire-path fix fully delivers**: pro/512k (the only §4 bench cell where
   p4_exact_tail fires — genuine 2-element boundary tie every step) goes
   23.03 → 17.17µs = **+34%**, landing inside the predicted 16.5–17µs window.
   Anchor: 027-node slow arm vs REPORT (094) pr = 1.468 on this cell (the known
   +45% regression), i.e. fast recovers most of the regression (17.17 vs
   pre-vseed snap 16.09 on the 07-20 bisect).
2. **Non-firing cells pay a codegen tax** (same class as the F1 campaign's
   Gate-C finding; fire never executes on these cells): mostly −1…−2%, but
   concentrated at **K512 cs1 mid-N: flash/64k −6.6%, flash/128k −9.1%**
   (reproduced on both GPUs). fp32 K512 short-row BS=1 is the op32-documented
   latency/icache-sensitive regime.
3. Ungated §4 geomean **0.9948** — ship rule (no cell <0.95) FAILS on flash.
4. **Recommended shape: gate `p4_tail_fast` to fp32 AND top_k ≥ 1024.**
   No known K512 firing cell exists (fire census: pro/512k bench + 9 fixture
   layer cells = 4×pro K1024 + 5×v32 K2048); the gate keeps the K512 kernel
   byte-identical (zero flash tax, exactness unchanged — radix backstop stays).
   K-gated §4 geomean **1.0053**, worst cell pro/4k 0.953 (N=1024, OUTSIDE the
   N≥32K deployment focus). Within N≥32K: worst −2.1% (v32/64k 0.979),
   wins v32/128k +1.3%, v32/256k +1.9%, **pro/512k +34%**.

## Open before any PR decision (not run yet)

- Implement the K-gate in the ctor default (1-line), re-run battery caseA to
  prove K512 OFF-shape byte-identity, spot-check K1024/K2048 exactness.
- BS-axis + synthetic-grid A/B (qab protocol) if promoting; per-layer 865-grid
  would weight the 9 firing fixture cells more favorably than the single-layer
  view above.
- Per user directive: **verify first, PR decision after** — nothing pushed.
