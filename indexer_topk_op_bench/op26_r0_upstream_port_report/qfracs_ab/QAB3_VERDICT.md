# qab3 VERDICT — op35 bundle-v2 on (PR#16457 head 0d6fc4f1f2 = shipped + K2048 rung swap)

**Protocol**: nsys cold-L2, 4 arms paired per cell same-GPU same-run
(ship / +skip_h1 / +kb512@K2048 / full), 77-cell fp32 grid + 46-cell 16-bit
K2048 spot (bf16/fp16 x best/worst/real-v32). b200-027, 2026-07-19.
All cells exact (fp32 77x4, 16-bit 46x3), 0 errors.

## Verdict: bundle-v2 collapses to kb512@K2048 ONLY; skip_h1 is DEAD on the new baseline

ship/arm geomeans (>1 = arm faster):

| axis                    | s/h1  | s/kb  | s/full |
|-------------------------|-------|-------|--------|
| real v32 (K2048, fp32)  | 0.985 | **1.061** | 1.045 |
| synth best K2048 fp32   | 1.006 | **1.052** | 1.071 |
| synth worst K2048 fp32  | 1.003 | **1.051** | 1.057 |
| ALL 77 fp32             | 1.000 | 1.018 | 1.017 |
| synth best/worst K2048 bf16 | — | **1.110 / 1.106** | 1.105/1.102 |
| real v32 bf16           | —     | **1.109** (min 1.066) | 1.112 |
| synth best/worst K2048 fp16 | — | **1.070 / 1.065** | 1.076/1.075 |
| real v32 fp16           | —     | **1.063** (min 0.999) | 1.066 |

- **skip_h1**: zero net (ALL 1.000), real v32 −1.5% -> DROP (simplicity rule;
  its op35-era contribution does not survive the new rung pair / current head).
- **kb512**: carries everything, ALL-POSITIVE on every axis (fp32 min cell
  0.994; bf16 min 1.035; fp16 min 0.999); K512/K1024 byte-identical (gate).
- **Interaction with rung swap**: op35's standalone +13.3% (K2048, old base)
  shrinks to +5.2-6.1% fp32 on the new baseline (partial overlap with the
  rung swap's +3.8%); combined vs original ship ~ +8-9% fp32 K2048;
  16-bit keeps the full ~+10-11% (bf16) / +6-7% (fp16) since the 16-bit rung
  effect was smaller.

## Ship recipe (separate follow-up PR, per prior user directive — NOT #16457)
GvrParams K2048 kNumBins 2048->512 for fp32+bf16+fp16, **gated on enable_r0**
(the table is shared with the base secant P4, whose kb512 behaviour was NOT
measured — apply as ctor override `if enable_r0 and top_k==2048: kNumBins=512`
to keep enable_r0=False byte-identical). Stack the branch on PR#16457 head.

Data: qab3.csv (fp32 77 cells), qab3_ckpt/ (all jsonls incl 16-bit), reps on
b200-027:/tmp/gvrqab/qab3_results (not committed - env-token hygiene).
