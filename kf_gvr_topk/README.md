# `r3_v11` — exact fp32 top-K index selection for the DSv4/V3.2 indexer (B200)

KernelFactory R4 cold-start campaign result (campaign
`pra6srbd7h4pqecqbgxgm15rgg`, round-3 winner, kernel id `7d8272b7`,
internal 1.3556). MERGE lineage: register-resident round-2 base + barrier
folding + per-(tier, K) measured AR6/AR8 quantile ladders. Independent
evolution from the R3-campaign branches on this fork
(`kf/gvr-topk-c74fsbx`, `kf/gvr-topk-compB`) — different kernel family.

## Contract

```
run(logits: fp32[1, npad] (CUDA), pre_idx: i32[1, k] (CUDA), n_valid: i64, indices_out: i32[1, k])
```

Exact top-k **indices** of `logits[0, :npad]` (pad is lowest-float and can
never enter the top-k; any order; tie-robust set semantics). Unlike the R3
branches this kernel is a true GVR: `pre_idx` (previous-step top-k hint) is
consumed on the register-resident and streaming paths; the direct path
ignores it.

## Algorithm (top-level dispatch on npad only, plus K for ladder choice)

- `npad <= 12288` -> `launch_direct`: single CTA x 1024 threads, hint-free.
- `12288 < npad <= 262144` -> `launch_reg`: register-resident GVR.
  1/4/8/16-CTA cluster x 512 threads by tier; the whole row is read ONCE
  into registers, then secant-style multi-pass threshold refinement runs
  with zero global re-scans. Per-(tier, K) quantile ladders (AR6 shifted vs
  AR8 dense) are set from per-cell measurement (see dispatch comments at
  the bottom of `kernel.cu`).
- `npad > 262144` -> `launch_gvr`: streaming 16-CTA cluster.

## Measured (B200, nsys cold-L2, paired same-GPU, 865 real decode cells, BS=1)

- geomean **1.6315x** vs TensorRT-LLM GVR PR#16457 pinned head
  @04a0900ff7, **865/865 exact**, 1 real regression: pro_64k_L38 = 0.963
  cold by 60-rep adjudication (hit 0.269, lowest of the band; low hint ->
  more secant rounds, ~0.4 us).
- 28-cell probe cold gm 1.7835, 0 regressions.
- BS-scaling (op37, 9 cells x BS 1..1024, real stacked rows, 198/198
  exact): BS=1 wins everywhere (1.137-2.283x); sequential per-row launches
  lose to the head's batched arm from **BS=2** (pro_1024k from BS=4).
  **Production dispatch gate: BS == 1 only** — structural (each launch
  fills the GPU), not tunable.

Process log and verdicts:
`indexer_topk_op_bench/op26_r0_upstream_port_report/kf_campaign/`
(`R4_RUN_STATE.md`, `grid_r4r3bg.csv`) and
`indexer_topk_op_bench/op37_bs_scaling/RESULTS.md` (repo-local, not on
GitHub).
