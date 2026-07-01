# op17 GVR threshold-portfolio — results summary (B200 sm_100)

Operator: `src/gvr_portfolio_cluster_op.py` (`gvr_portfolio_cluster(..., G="auto")`).
Baselines: single-CTA `gvr_cutedsl` (target) and PR#15198 multicta cluster. Synth = report
bundles (seed=42, preIdx hit-rate 0.6). All cells EXACT (vdiff=0, uniq=K).

## nsys pure-kernel (BS=1, fp32, cold-L2, 100 iters)  [canonical]
cell,base_us,port_us,nsys_speedup,event_speedup
K512_N16384,16.69,9.97,1.673,1.365
K512_N65536,19.24,14.71,1.309,1.174
K1024_N65536,24.27,17.91,1.355,1.250
K2048_N262144,52.44,33.31,1.574,1.475
K512_N262144,39.20,32.50,1.206,1.174

## ×3-median cold-L2 (BS=1), port/base = vs single-CTA, port/mc = vs multicta
dtype,K,N,port_over_base,port_over_mc
fp32,512,4096,1.166,1.159
fp32,512,16384,1.365,1.361
fp32,512,65536,1.174,1.017
fp32,512,262144,1.174,0.714
fp32,1024,4096,1.002,0.973
fp32,1024,16384,1.216,1.218
fp32,1024,65536,1.250,1.147
fp32,1024,262144,1.270,0.777
fp32,2048,16384,1.030,1.031
fp32,2048,65536,1.363,1.164
fp32,2048,262144,1.475,0.831
bf16,512,4096,1.156,1.171
bf16,512,16384,1.147,1.134
bf16,512,65536,1.146,1.108
bf16,512,262144,1.234,0.749
bf16,1024,4096,1.216,1.192
bf16,1024,16384,1.190,1.141
bf16,1024,65536,1.128,1.097
bf16,1024,262144,1.227,0.761
bf16,2048,16384,1.136,1.127
bf16,2048,65536,1.154,1.057
bf16,2048,262144,1.386,0.797
fp16,512,4096,1.137,1.137
fp16,512,16384,1.076,1.028
fp16,512,65536,1.223,1.177
fp16,512,262144,1.224,0.787
fp16,1024,4096,1.149,1.156
fp16,1024,16384,1.301,1.289
fp16,1024,65536,1.197,1.173
fp16,1024,262144,1.303,0.812
fp16,2048,16384,1.135,1.110
fp16,2048,65536,1.169,1.120
fp16,2048,262144,1.376,0.824

## Batch-size sweep (K512 fp32 N65536, auto-G), port/base
BS,G,speedup
1,16,1.273
4,16,1.203
8,8,1.173
16,4,1.230
32,1,1.020
64,1,0.986
128,1,1.000

## Summary
- vs single-CTA baseline: NO regression, all dtypes; min 1.00x, avg ~1.22x (event), max 1.48x.
- nsys BS=1: 1.21-1.67x (event is a conservative lower bound for the cluster kernel).
- vs PR#15198 multicta: wins N<=65K, loses N>=262K (crossover ~131K) -> per-(N,BS) dispatch.
- BS<=16 wins 1.17-1.27x; BS>=32 -> baseline fallback (no regression). G=2 never emitted.
