# op25 — HLS win-region expansion campaign

Goal: expand the HLS (op21 iter16) win region against radix-cuteDSL and
SGLang StreamingTopK on the op22rr loss regions, fp32, deployment envelope
N <= 262144. Remote branch: `op25-hls-topk` on longcheng-nv/TensorRT-LLM
(orphan; baseline commit = iter16 ship state; each shipped step lands a
commit).

## Loss-region -> bottleneck map (from op22rr REPORT + falsification history)

| region | mechanism | lever |
|---|---|---|
| R1 small N <=16K, all scen, BS-invariant | K-proportional serial phase floor (P1/P1b/P4 ~ 3/4 of row work at N=4K) on 1 CTA; op12: floor alone 1.2x SGLang | S3a kC diet, S3b warp pipeline, dispatch |
| R2 mid N 64-256K low BS vs radix | msc C=4 uses 4-8 SMs vs radix 4-32; + static ladder fast rate 25-35% on real | S1a/S1b (admission), S2 HLS-MC (bandwidth) |
| R3 worst (hr=.05) vs radix | all ladder cols count>kC (all_ge; window f in [h,2.5h]); bracket-hi unknown -> falsi can't help | S1a low tail col 0.048 |
| R4 high BS x 64-128K vs SGLang | non-fused ms 2-pass + fallback rows pay extra full passes; throughput domain = bytes | S1a admission (fallback rows -> 0) |

## Step 1 (S1a screening) — DONE 2026-07-08

`screen_qfracs.py` 3 rounds, 30290 rows x 17 arms
(op22rr fp32 grid 78 + op24 392-combo hr sweep + 29,820 REAL Pro
multi-turn transitions). Verdict:

- **wide4b_c8 = (0.92, 0.60, 0.25, 0.048), M_thr=5, slot_scale=2** for
  K512/K1024: Pro real fast 0.298 -> 0.958 (oracle 0.998); op22rr real
  0.667 -> 1.0; worst 0 -> 0.89/1.0; best stays 1.0; op24 samp 0.65->0.9+.
- K2048: deep cols REGRESS (v32 real 0.75 -> 0.375, band geometry);
  keep stock triple, slot_scale=2 only (real 0.75 -> 0.875 in replay).
- Structural finding: the fused-collect column is pinned to qfracs[0], so
  a static ladder must trade depth (h>0.75 pair01 cliff, 67% of real Pro
  steps) against mid-h collect overflow; slot_scale=2 (smem
  num_threads*slot_cap*8B, fused path only = bs <= NUM_SMS, no high-BS
  residency cost) buys back the overflow. 0.95 col overflows (deep3c),
  0.92 is the edge.

## Step 2 (S1a on silicon) — IN FLIGHT

- Kernel edits (src/gvr_ms_op.py + gvr_msc_op.py): `slot_scale` ctor param;
  per-K `_QFRACS_SHIP` table; env knobs `OP25_QFRACS` ("base" = stock) /
  `OP25_SLOTCAP`; both in the compile keys; M_thr = len(qfracs)+1.
- Gates: smoke_exact 54+60 GREEN; real_msc / adversarial_band / gate_op22
  456 / real_16bit RUNNING (GPU0, /tmp/op25_gates.log).
- nsys A/B: ab_qfracs.py 3 arms base|ship|radix, op22rr bundles, 24
  cells/scenario (envelope, BS=1 tail + BS spots 512/64/16), GPU1,
  results/nsys/ab_qfracs/. Parse: parse_ab_qfracs.py (flip counter).
- S4 (histogram fallback for residual all_ge) deferred until A/B shows the
  residual matters (iter14 lesson: code mass is currency).

## Step 3 (S1b rho-tracking) — planned

Host-replay E[T] on real Pro chains ON TOP of wide4b_c8 first; build the
kernel (side-tensor h-state, runtime fracs, low-side asymmetric band) only
if the marginal over S1a exceeds the silicon noise floor (~2%).
Screen said the S1a->oracle gap is ~4pts fast-rate on Pro real.

## Step 4 (S2 HLS-MC) — planned

First cheap probe: msc C=8 for fp32 K512/K1024 at BS<=8, N>=131072 (the
old fp32-C8 falsification predates iter16 dist fallback + S1a admission).
Then the >8-CTA gmem-atomic count-merge 2-kernel split, single point BS=1
N=262K vs radix; expand only if >= 1.0.

## Step 5 (S3a/S3b small-N) — planned

kC_override small-N table (host pre-screen via proto Row + GvrParams
override; silicon via gvr_sw kC=). S3b warp pipeline one strict ablation.

## Measurement discipline

nsys cold-L2 paired same-process A/B only (feedback_kernel_bench_l2_flush);
`env -u GITHUB_TOKEN -u HF_TOKEN`; never commit *.sqlite/*.nsys-rep;
b200-027 (this node) GPU0/GPU1 both healthy (33-38C idle).
