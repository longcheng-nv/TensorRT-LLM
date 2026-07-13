# op30 calibration — GVR-base (cuteDSL) favorability grid

Source: /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/results_b200_op30_calib (nsys cold-L2 canonical, fp32 BS=1, N∈[16384, 65536, 262144], seed=42+crc32(K|N)%1e6)

## v4flash (K=512)

- **BEST**  = `beta_shallow` hr=0.30 (score 1.014, realised hr 0.301)
- **WORST** = `beta_shallow` hr=0.90 (score 2.201, realised hr 0.900)
- WORST/BEST time ratio (geomean): 2.170x
- radix control max spread over cfg×hr: 1.036x

| cfg | hr | N=16384 | N=65536 | N=262144 | score |
|---|---|---|---|---|---|
| beta_shallow | 0.30 | 10.1 | 14.8 | 30.6 | 1.014 **B** |
| aggregate | 0.30 | 9.9 | 15.5 | 31.0 | 1.026 |
| beta_moderate | 0.05 | 10.8 | 15.7 | 30.3 | 1.050 |
| beta_shallow | 0.05 | 10.8 | 16.3 | 30.3 | 1.064 |
| beta_moderate | 0.15 | 11.9 | 14.8 | 30.1 | 1.065 |
| aggregate | 0.05 | 11.1 | 15.3 | 31.5 | 1.066 |
| beta_moderate | 0.30 | 11.7 | 15.3 | 30.3 | 1.071 |
| beta_shallow | 0.15 | 12.2 | 14.8 | 30.9 | 1.083 |
| aggregate | 0.15 | 12.3 | 14.8 | 31.2 | 1.088 |
| beta_deep | 0.05 | 12.3 | 15.3 | 30.3 | 1.090 |
| beta_deep | 0.15 | 12.7 | 15.0 | 30.4 | 1.096 |
| beta_deep | 0.30 | 14.6 | 14.8 | 37.6 | 1.226 |
| aggregate | 0.45 | 11.9 | 20.7 | 39.9 | 1.307 |
| beta_shallow | 0.45 | 11.9 | 19.7 | 44.4 | 1.333 |
| beta_moderate | 0.45 | 13.9 | 19.6 | 38.3 | 1.334 |
| beta_deep | 0.45 | 16.5 | 19.5 | 37.2 | 1.397 |
| beta_deep | 0.55 | 15.6 | 21.9 | 37.6 | 1.429 |
| beta_shallow | 0.55 | 13.1 | 19.7 | 51.2 | 1.442 |
| aggregate | 0.55 | 13.1 | 29.1 | 37.8 | 1.483 |
| aggregate | 0.65 | 13.8 | 32.2 | 37.9 | 1.562 |
| beta_deep | 0.90 | 12.4 | 36.9 | 37.8 | 1.578 |
| beta_deep | 0.75 | 14.0 | 34.4 | 37.5 | 1.599 |
| beta_deep | 0.65 | 16.2 | 31.4 | 38.0 | 1.638 |
| beta_moderate | 0.65 | 15.7 | 32.2 | 38.4 | 1.639 |
| aggregate | 0.75 | 14.8 | 33.8 | 38.9 | 1.639 |
| beta_moderate | 0.55 | 17.1 | 29.7 | 38.2 | 1.640 |
| beta_moderate | 0.90 | 13.8 | 37.8 | 39.1 | 1.665 |
| beta_moderate | 0.75 | 16.5 | 34.1 | 39.1 | 1.708 |
| beta_moderate | 0.85 | 16.4 | 34.6 | 39.2 | 1.716 |
| beta_deep | 0.85 | 16.8 | 36.7 | 37.7 | 1.741 |
| aggregate | 0.85 | 17.2 | 34.1 | 41.5 | 1.768 |
| aggregate | 0.90 | 17.7 | 37.2 | 39.2 | 1.802 |
| beta_shallow | 0.65 | 13.8 | 31.3 | 63.9 | 1.843 |
| beta_shallow | 0.75 | 14.6 | 34.3 | 71.7 | 2.012 |
| beta_shallow | 0.85 | 17.2 | 37.2 | 72.1 | 2.189 |
| beta_shallow | 0.90 | 17.8 | 37.1 | 71.2 | 2.201 **W** |
* per-N N=16384: argmin=aggregate/hr0.30 argmax=beta_shallow/hr0.90
* per-N N=65536: argmin=beta_shallow/hr0.30 argmax=beta_moderate/hr0.90
* per-N N=262144: argmin=beta_moderate/hr0.15 argmax=beta_shallow/hr0.85

## v4pro (K=1024)

- **BEST**  = `aggregate` hr=0.15 (score 1.082, realised hr 0.150)
- **WORST** = `beta_deep` hr=0.85 (score 1.892, realised hr 0.850)
- WORST/BEST time ratio (geomean): 1.749x
- radix control max spread over cfg×hr: 1.043x

| cfg | hr | N=16384 | N=65536 | N=262144 | score |
|---|---|---|---|---|---|
| aggregate | 0.15 | 12.2 | 15.1 | 30.2 | 1.082 **B** |
| beta_moderate | 0.15 | 10.0 | 17.1 | 32.8 | 1.086 |
| beta_deep | 0.15 | 9.8 | 17.5 | 34.4 | 1.105 |
| beta_shallow | 0.15 | 11.6 | 18.6 | 31.2 | 1.153 |
| beta_deep | 0.05 | 13.9 | 15.6 | 32.9 | 1.176 |
| beta_moderate | 0.05 | 13.5 | 18.2 | 29.6 | 1.183 |
| aggregate | 0.05 | 13.3 | 18.1 | 30.8 | 1.190 |
| beta_shallow | 0.05 | 13.4 | 19.4 | 30.3 | 1.216 |
| beta_moderate | 0.30 | 13.3 | 16.9 | 40.0 | 1.268 |
| aggregate | 0.90 | 14.3 | 19.4 | 38.2 | 1.340 |
| beta_shallow | 0.45 | 12.1 | 24.4 | 36.8 | 1.350 |
| beta_shallow | 0.30 | 17.8 | 16.6 | 37.0 | 1.354 |
| aggregate | 0.30 | 11.6 | 25.4 | 37.6 | 1.361 |
| beta_deep | 0.30 | 12.6 | 21.5 | 46.0 | 1.414 |
| beta_moderate | 0.45 | 14.2 | 23.7 | 38.8 | 1.436 |
| aggregate | 0.85 | 16.0 | 24.0 | 37.2 | 1.481 |
| beta_moderate | 0.85 | 17.5 | 22.9 | 41.1 | 1.554 |
| beta_moderate | 0.90 | 17.4 | 23.7 | 40.9 | 1.564 |
| beta_shallow | 0.85 | 13.2 | 35.9 | 37.6 | 1.596 |
| beta_deep | 0.90 | 19.1 | 21.7 | 44.1 | 1.607 |
| beta_shallow | 0.90 | 12.8 | 39.4 | 36.7 | 1.614 |
| aggregate | 0.65 | 13.2 | 37.4 | 38.8 | 1.631 |
| beta_deep | 0.45 | 16.0 | 18.5 | 65.5 | 1.640 |
| aggregate | 0.75 | 12.0 | 42.3 | 38.5 | 1.642 |
| beta_shallow | 0.65 | 16.7 | 31.1 | 39.0 | 1.666 |
| beta_shallow | 0.55 | 19.1 | 29.3 | 37.6 | 1.685 |
| aggregate | 0.45 | 16.2 | 34.8 | 38.2 | 1.697 |
| beta_moderate | 0.65 | 15.9 | 36.7 | 38.1 | 1.714 |
| beta_shallow | 0.75 | 16.5 | 35.4 | 38.1 | 1.718 |
| beta_moderate | 0.55 | 17.5 | 31.5 | 41.1 | 1.727 |
| aggregate | 0.55 | 17.8 | 36.4 | 38.3 | 1.781 |
| beta_deep | 0.65 | 14.5 | 22.0 | 80.3 | 1.797 |
| beta_deep | 0.55 | 16.0 | 22.5 | 72.9 | 1.815 |
| beta_moderate | 0.75 | 18.3 | 39.6 | 37.6 | 1.836 |
| beta_deep | 0.75 | 15.2 | 22.0 | 87.2 | 1.877 |
| beta_deep | 0.85 | 16.6 | 20.7 | 86.9 | 1.892 **W** |
* per-N N=16384: argmin=beta_deep/hr0.15 argmax=beta_deep/hr0.90
* per-N N=65536: argmin=aggregate/hr0.15 argmax=aggregate/hr0.75
* per-N N=262144: argmin=beta_moderate/hr0.05 argmax=beta_deep/hr0.75

## v32 (K=2048)

- **BEST**  = `beta_shallow` hr=0.15 (score 1.044, realised hr 0.150)
- **WORST** = `aggregate` hr=0.85 (score 1.611, realised hr 0.850)
- WORST/BEST time ratio (geomean): 1.543x
- radix control max spread over cfg×hr: 1.034x

| cfg | hr | N=16384 | N=65536 | N=262144 | score |
|---|---|---|---|---|---|
| beta_shallow | 0.15 | 13.1 | 18.3 | 33.0 | 1.044 **B** |
| aggregate | 0.05 | 13.1 | 17.5 | 35.1 | 1.052 |
| beta_shallow | 0.30 | 12.5 | 19.8 | 33.3 | 1.060 |
| aggregate | 0.30 | 12.3 | 20.2 | 33.6 | 1.065 |
| aggregate | 0.15 | 12.8 | 18.4 | 35.3 | 1.065 |
| beta_deep | 0.45 | 14.8 | 17.3 | 33.6 | 1.076 |
| aggregate | 0.45 | 13.5 | 19.6 | 33.3 | 1.084 |
| beta_shallow | 0.05 | 13.1 | 19.7 | 34.2 | 1.085 |
| beta_moderate | 0.15 | 13.1 | 19.9 | 34.8 | 1.094 |
| beta_moderate | 0.45 | 14.8 | 18.6 | 33.3 | 1.098 |
| beta_moderate | 0.30 | 13.5 | 21.2 | 32.5 | 1.102 |
| beta_moderate | 0.05 | 13.4 | 21.5 | 35.7 | 1.140 |
| beta_deep | 0.65 | 13.6 | 23.2 | 33.1 | 1.148 |
| beta_deep | 0.15 | 12.3 | 19.9 | 45.5 | 1.170 |
| beta_deep | 0.30 | 12.6 | 19.3 | 45.8 | 1.172 |
| beta_deep | 0.55 | 14.2 | 24.3 | 33.6 | 1.189 |
| beta_deep | 0.05 | 13.5 | 20.6 | 45.8 | 1.224 |
| beta_shallow | 0.45 | 13.7 | 25.7 | 45.5 | 1.322 |
| beta_moderate | 0.55 | 15.2 | 23.3 | 45.9 | 1.330 |
| beta_shallow | 0.55 | 15.8 | 23.0 | 45.2 | 1.333 |
| beta_deep | 0.85 | 14.0 | 27.5 | 44.0 | 1.348 |
| beta_deep | 0.90 | 14.0 | 27.0 | 45.7 | 1.357 |
| aggregate | 0.55 | 15.9 | 23.9 | 45.5 | 1.357 |
| beta_shallow | 0.75 | 14.2 | 27.0 | 45.6 | 1.360 |
| beta_shallow | 0.65 | 14.8 | 27.2 | 45.7 | 1.387 |
| beta_deep | 0.75 | 15.2 | 27.1 | 45.9 | 1.399 |
| aggregate | 0.65 | 14.6 | 25.8 | 51.9 | 1.411 |
| beta_shallow | 0.90 | 16.1 | 26.4 | 45.7 | 1.412 |
| beta_shallow | 0.85 | 15.8 | 27.1 | 46.0 | 1.416 |
| aggregate | 0.75 | 14.1 | 26.9 | 52.1 | 1.420 |
| beta_moderate | 0.65 | 16.3 | 23.6 | 51.8 | 1.423 |
| beta_moderate | 0.75 | 15.0 | 26.2 | 51.5 | 1.431 |
| beta_moderate | 0.85 | 13.3 | 27.0 | 57.9 | 1.441 |
| aggregate | 0.90 | 16.2 | 27.9 | 58.2 | 1.561 |
| beta_moderate | 0.90 | 15.9 | 29.9 | 57.5 | 1.581 |
| aggregate | 0.85 | 15.7 | 31.8 | 58.0 | 1.611 **W** |
* per-N N=16384: argmin=beta_deep/hr0.15 argmax=beta_moderate/hr0.65
* per-N N=65536: argmin=beta_deep/hr0.45 argmax=aggregate/hr0.85
* per-N N=262144: argmin=beta_moderate/hr0.30 argmax=aggregate/hr0.90

