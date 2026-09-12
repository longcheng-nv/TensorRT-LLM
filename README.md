# Kernel Factory winner — GVR V2 decode top-K, native bf16 arm, BS=1 campaign (fp32 arm unchanged)

Campaign `gvr-v2-bs1-bf16-r2` (Kernel Factory id `y2tnvsfgvx1vkb1xpc1vp1bt3g`, fork of `zwzeecy7nn6h3b3rx9j0nkwwfw`
at round 2; 10 rounds, 2026-09-11/12, B200). Winning candidate
`c6af0bf6251e655133500582e832898faa92fff8e2fe841bf31bac846a1f4cbd` (`gvr_v2_bs1_bf16_unroll40_p9fold`, round 9,
author claude-fable-5 agent a003). KF metric: 1.392x vs the unmodified fp32 GVR V2 (6.66 us vs 9.27 us, weighted
geomean over 100 BS=1 real-capture workloads).

Lineage: TensorRT-LLM `main` 8a931f619682d5703bff8595e96e0f28f460ce26
(`tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode_self_sampling{,_host}.py`) -> KF campaign
`r3156qv9610mq59yqdv8q5ctfm` winner + 2-line compile fix (branch `kf/gvr-v2-bf16-winner-r3156qv`) = this campaign's seed
-> this winner. Modules renamed here to `gvr_ss.py` / `gvr_ss_host.py` / `gvr_ss_v4.py`, entry `main.py`.

## Files (this commit = the winner exactly as submitted to Kernel Factory)

| file | role |
|---|---|
| `main.py` | DPS entry `run(logits, kv_lens, compress_ratio, index_topk, indices)` for **bf16** logits; `run_fp32(...)` twin routes fp32 logits through the unmodified fp32 arm |
| `gvr_ss.py` | GVR V2 device module + bf16 arm (constexpr-dtype; 16-byte `ld.global.nc.v4.b32` = 8 bf16/vector in the streaming families, 8-byte in `reg`/`reg_clus`; packed `pk16`/`v16` tie keys; P0/P1 head-latency hoists; **mc>=40-gated 8-wide crossing-bin rank-walk unroll**; **P9 fold**: on VPT1 one-class `reg_clus` routes the P9 proof scan and its barrier are replaced by fire-and-forget `red.relaxed.cluster.shared::cluster` u32 min/max issued in P7) |
| `gvr_ss_v4.py` | 8-byte (4 bf16/vector) twin engine used when a deep-split chunk under-fills a 16-byte tile |
| `gvr_ss_host.py` | host: verbatim `route()/run_varlen()` + `route_bf16`, `route_streaming_bf16`, `_varlen_launcher_bf16`, `run_varlen_bf16` (bf16 route table: 10 reg / 13 reg_clus rungs, V32-256K CS10, Flash-1024K SPLIT-main, NB1024+NB512 cursor folds, QC192 small-CS scans) |
| `solution.json`, `solution_metrics.json` | CudaGym solution bundle and KF score record |

Contract: `logits [B, NPAD]` bf16 row-major (NPAD % 8 == 0, 16-byte base, pad = `finfo(bf16).min`),
`kv_lens [B]` int32 in KV-token space (= N_valid * compress_ratio, consumed on device), `compress_ratio`
1 (DSv3.2) or 4 (DSv4), `index_topk` = K = `indices.shape[1]`; writes int32 indices in place; one kernel
launch per call; routing envelope `max_seq_len = NPAD * compress_ratio`. Exactness = tie-aware value
multiset vs `torch.topk` on the bf16 row.

## Verification (local, B200, cutlass DSL 4.6.1, 2026-09-12)

- fp32 arm: parity 6/6 cases; fp32 compiled PTX/SASS identical to the verbatim kernel for reg (3,928 SASS lines),
  reg_clus (4,184) and main (7,528, identical after stripping the mangled entry symbol); fp32 route tables, kernel
  ABI and constants untouched.
- 886 real-capture cells at BS=1 (cold-L2 nsys, kernel-only, 20 reps, same-process paired arms): geomean
  **1.342x** vs fp32 GVR V2 (5.274 us vs 7.078 us), 886/886 exact; by fp32 family main 1.445 / reg_clus 1.281 /
  reg 1.250; by segment v3.2 K=2048 up to 1.72-1.74x at N 16-32K. Versus the local op56/op57 bf16 lineage
  (1.238x at BS=1): +8.4 % geomean, gains on 21 of 25 (segment, ISL) buckets far outside the A/A noise band
  (±2-3 % per bucket), regressions on flash/pro 8k (0.95x, the reg(1024,1,2) rung whose head-latency hoist this
  lineage lacks) and flash 4k/16k (0.98x).
- Known bf16 pathology shared with the seed (exact but slow): 4 v3.2 layers at BS=1 (v32_128k L01/L02, v32_64k
  L02/L08) 0.27-0.81x vs fp32 through the degenerate whole-row narrowing escape.
- The KF metric (1.392x) was measured on a 1500 MHz clock-locked evaluator against a 9.27 us fp32 baseline; the
  local numbers above are the reference.

## Run

```
export PYTHONPATH=<cutlass DSL 4.6.x + apache-tvm-ffi>:$PYTHONPATH
python3 -c "
import torch, main as kernel
B,NPAD,K,cr=1,16448,512,4; n=16387
lg=torch.randn(B,NPAD,device='cuda').to(torch.bfloat16); lg[:,n:]=torch.finfo(torch.bfloat16).min
kv=torch.full((B,),n*cr,dtype=torch.int32,device='cuda'); idx=torch.empty(B,K,dtype=torch.int32,device='cuda')
kernel.run(lg,kv,cr,K,idx)
ref=torch.topk(lg[:,:n].float(),K,dim=1).values.sort(1).values
got=lg.float().gather(1,idx.long()).sort(1).values
print('exact', torch.equal(ref,got))"
```
