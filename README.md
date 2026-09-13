# Kernel Factory winner — GVR V2 decode top-K, native bf16 arm from the fp32 production kernel

Campaign `gvr-v2-bf16-native` (Kernel Factory id `678x0j5q1d5kvdjjnw9k95j88m`, 2026-09-11/12, 10 rounds,
B200, pool 4x claude:fable-5 + 4x codex:gpt-5.6-sol + 2x n3:gpt-5.6-sol). Winning candidate
`ae73b0955e39a6c555e5bdcc074c9f671f614a94ecf01ce9cb69255b92e7c58c`
(`gvr-v2-bf16-r10-regclus-carrier-r8split`, round 10, codex agent a006). KF metric: 1.2151x vs the
unmodified fp32 GVR V2 (9.91 us vs 12.04 us, weighted geomean over 150 workloads: 96 real decode
captures + 54 random-row shape-coverage rows).

Base: TensorRT-LLM `main` 7351c9c882
(`tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode_self_sampling{,_host}.py`,
blobs 8bbd9030 / a88937fe), kept verbatim here under the same names.

## Files (first commit = the winner exactly as submitted to Kernel Factory)

| file | role |
|---|---|
| `kernel_entry_wide.py` | DPS entry `run(logits, kv_lens, compress_ratio, index_topk, indices)` for **bf16** logits; `run_fp32(...)` twin routes fp32 logits through the unmodified fp32 arm; geometry-only `(B, NPAD, K)` dispatch between the two bf16 hosts |
| `gvr_device_bf16_wide.py`, `gvr_host_bf16_wide.py` | "wide" bf16 module: streaming main / clus / reg / reg_clus families with packed `v2.b32` loads, fused 16-byte `v4.b32` sample loads, packed-u32 P3 carriers widened only at classify, bf16 candidate-volume retune, B=16 R=8 fuller-CTA and B=1 262K R=16/U=4 rungs |
| `gvr_bf16_device.py`, `gvr_bf16_host.py` | "compact / register-fringe" bf16 module: register-family tiles for small NPAD and B=1 rows, VPT=2/4 kernels that issue every 8-byte packet before widening |
| `gvr_topk_decode_self_sampling.py`, `gvr_topk_decode_self_sampling_host.py` | verbatim fp32 production kernel + host (byte-identical to main) |
| `solution.json`, `solution_metrics.json` | CudaGym solution bundle and KF score record |

Contract: `logits [B, NPAD]` bf16 row-major (NPAD % 64 == 0, 16-byte base, pad = `finfo(bf16).min`),
`kv_lens [B]` int32 in KV-token space (= N_valid * compress_ratio, consumed on device), `compress_ratio`
1 (DSv3.2) or 4 (DSv4), `index_topk` = K = `indices.shape[1]`; writes int32 indices in place; one kernel
launch per call; routing envelope `max_seq_len = NPAD * compress_ratio`. Exactness = tie-aware value
multiset vs `torch.topk` on the bf16 row.

## Verification (local, B200, cutlass DSL 4.6.1)

- fp32 arm: the two fp32 modules are byte-identical to main (sha256), so fp32 SASS is unchanged by
  construction; `run_fp32` parity 150/150 cases (25 buckets x 6 batch sizes, index set / value multiset /
  torch.topk).
- 886-cell x 6-BS real-capture grid (BS 1, 16, 128, 256, 512, 1024 = 5,316 cases, cold-L2 nsys,
  kernel-only, both arms under `NPAD*cr`): with the compile fix of the second commit, geomean **1.2429x**
  vs fp32 GVR V2 (median 1.234, p10 1.10, p90 1.43), 5,316/5,316 exact, 98.8 % of cases faster;
  BS 1/16/128/256/512/1024 = 1.186/1.196/1.273/1.234/1.275/1.298; flash 1.235 / pro 1.233 / v3.2 1.252.
- As submitted, the bf16 modules do not JIT under DSL 4.6.1 on 51 of the 150 grid shapes
  (`TYPE_UNSTABLE_JOIN`: `qt` in `gvr_device_bf16_wide.py` classify tail, `i` in the
  `gvr_bf16_device.py` short-row emit loop); the KF image accepted them. The second commit (`fix1`)
  pre-binds the names (22 lines, value-neutral) -> compiles on all 150 shapes.
- Known losses (exact but slower than fp32): 64 / 5,316 cases, the catastrophic ones being the
  row-layout knife-edge B=1 cells `v32_128k_L02` (0.22x), `v32_256k_L02` (0.24x), `v32_128k_L01`
  (0.63x) where the bf16 route lands on reg_clus while fp32 uses the streaming main; `v4_pro_512k_L28`
  at B 256-1024 (0.85-0.89x).
- Campaign targets (geomean >= 1.8x, per-case floor >= 1.2x) were **not** met; the earlier op54/op56
  lineage measures 1.31-1.32x on the same 5,316 cases.
