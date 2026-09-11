# Kernel Factory winner — GVR V2 decode top-K with a native bf16 arm (fp32 arm unchanged)

Campaign `gvr-v2-bf16` (Kernel Factory id `r3156qv9610mq59yqdv8q5ctfm`, 2026-09-10/11, 10 rounds,
B200). Winning candidate `0d4ea7432a04c9b64ee3e3508d8944099f37c5c4bbc16949558a4ff903e61b7f`
(`gvr_v2_topk_bf16_r10_p0hoist`, round 10, author claude-fable-5 agent a000). KF metric: 1.268x vs
the unmodified fp32 GVR V2 (10.15 us vs 12.87 us, weighted geomean over 44 real-capture workloads).

Base: TensorRT-LLM `main` 8a931f619682d5703bff8595e96e0f28f460ce26
(`tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode_self_sampling{,_host}.py`,
blobs 8bbd9030 / a88937fe), renamed here to `gvr_topk_ss.py` / `gvr_topk_ss_host.py`.

## Files (this commit = the winner exactly as submitted to Kernel Factory)

| file | role |
|---|---|
| `kernel.py` | DPS entry `run(logits, kv_lens, compress_ratio, index_topk, indices)` for **bf16** logits; `run_fp32(...)` twin routes fp32 logits through the unmodified fp32 arm |
| `gvr_topk_ss.py` | GVR V2 device module + bf16 arm (16-byte `ld.global.nc.v4.b32` = 8 bf16/vector in the streaming families, 8-byte in `reg`, packed `pk16`/`v16` tie keys, P0/P1 head-latency hoisting), all inside `cutlass.const_expr(self.dtype == cutlass.Float32)` branches |
| `gvr_topk_ss_v4.py` | 8-byte (4 bf16/vector) twin engine used when a deep-split chunk under-fills a 16-byte tile |
| `gvr_topk_ss_host.py` | host: verbatim `route()/run_varlen()` + added `route_bf16`, `route_streaming_bf16`, `_varlen_launcher_bf16`, `run_varlen_bf16` |
| `solution.json`, `solution_metrics.json` | CudaGym solution bundle and KF score record |

Contract: `logits [B, NPAD]` bf16 row-major (NPAD % 4 == 0, 16-byte base, pad = `finfo(bf16).min`),
`kv_lens [B]` int32 in KV-token space (= N_valid * compress_ratio, consumed on device), `compress_ratio`
1 (DSv3.2) or 4 (DSv4), `index_topk` = K = `indices.shape[1]`; writes int32 indices in place; one kernel
launch per call; routing envelope `max_seq_len = NPAD * compress_ratio`. Exactness = tie-aware value
multiset vs `torch.topk` on the bf16 row.

## Verification (local, B200, cutlass DSL 4.6.1)

- fp32 arm: parity 22/22 cases, fp32 compiled PTX/SASS byte-identical to the verbatim kernel for
  reg / reg_clus / main; fp32 route tables, kernel ABI and constants untouched.
- 886-cell x 11-BS real-capture grid (9,746 cases, cold-L2 nsys, kernel-only): with the compile fix
  below, geomean **1.274x** vs fp32 GVR V2, 100 % exact (flash 1.26 / pro 1.24 / v3.2 1.30; BS1 1.22 ->
  BS1024 1.38). As submitted, the bf16 `reg` family with `brl=True` does not JIT under DSL 4.6.1
  (`TYPE_UNSTABLE_JOIN`, `gvr_topk_ss.py` ~L4461), affecting 31.7 % of grid cases; the KF image accepted it.
- Known bf16 pathology (routing, exact but slow): 26/9,746 cases 12-35x slower than fp32 (pro 1M BS1 on 3
  layers, v3.2 128k BS16/32 on 11 layers) via the degenerate whole-row radix-narrowing escape; the
  `r512` rung in `route_streaming_bf16` halves the constexpr degen gate (CMPB/SCPB) at BLK=512. Plus a
  uniform 0.54-0.70x loss of the `reg_clus(1024,2,16)` refit band at n >= 128K (61 cases).

The next commit on this branch applies the 2-line compile fix (`fix1`); everything else is unchanged.

## Run

```
export PYTHONPATH=<cutlass DSL 4.6.x + apache-tvm-ffi>:$PYTHONPATH
python3 -c "
import torch, kernel
B,NPAD,K,cr=4,16448,512,4; n=16387
lg=torch.randn(B,NPAD,device='cuda').to(torch.bfloat16); lg[:,n:]=torch.finfo(torch.bfloat16).min
kv=torch.full((B,),n*cr,dtype=torch.int32,device='cuda'); idx=torch.empty(B,K,dtype=torch.int32,device='cuda')
kernel.run(lg,kv,cr,K,idx)
ref=torch.topk(lg[:,:n].float(),K,dim=1).values.sort(1).values
got=lg.float().gather(1,idx.long()).sort(1).values
print('exact', torch.equal(ref,got))"
```
