---
name: automerge-trtllm-topk
description: Integrates a standalone single-CTA heuristic TopK micro-kernel into TensorRT-LLM's DSA indexer decode path, extending it to multi-row (multi-batch, next_n>1) with CUDA Graph compatibility. Use when the user provides a new or modified heuristic_topk.cuh and wants end-to-end integration and testing.
---

# Heuristic TopK Micro-Kernel → TensorRT-LLM Integration

Take a **single-CTA, single-batch, next_n=1** heuristic TopK micro-kernel and integrate it into TensorRT-LLM's indexer TopK decode path as a production-ready multi-row kernel.

## Inputs

The user provides:
1. **Micro-kernel**: `torchCPP_heuristictopk/heuristic_topk.cuh` (one-CTA, single-row)
2. **Standalone test**: `torchCPP_heuristictopk/predicted_topK_filtering_perf_simulator.py`
3. **TensorRT-LLM source** at workspace root
4. **GPU environment** with nsys, ncu, and TRT-LLM Docker

## Integration Pipeline

### Phase 1: Codebase & Interface Analysis

1. **Read the micro-kernel** — extract the core device function signature, shared memory struct, constants (TOP_K, BLOCK_SIZE, etc.), and understand its single-row algorithm.
2. **Analyze TRT-LLM's existing indexer TopK** — find the decode entry point (`invokeIndexerTopKDecode` or equivalent), understand existing sort paths, and identify how `pre_idx` (heuristic hint indices) is plumbed from Python through thop binding to CUDA kernel.
3. **Analyze the DSA sparse attention Python code** — understand how `indexer_topk_decode` is called in `sparse_attn_indexer`, how metadata buffers are managed, and how CUDA Graph capture works.
4. **Identify interface gaps** between micro-kernel (single-row) and TRT-LLM's needs:
   - Multi-row (multi-batch) support
   - MTP support (next_n > 1): each request produces next_n rows, preIdx/seqLens are per-request
   - CUDA Graph compatibility for pre_idx management
   - Build system integration (CMake, separate compilation)

**Gate**: Interface gap report produced.

### Phase 2: Baseline Performance

#### 2a. Functional smoke test (no profiling)

```bash
cd torchCPP_heuristictopk
# Random data: batch=1, topK=2048, N=65536, next_n=1, warmup=4, use_real_data=0
python predicted_topK_filtering_perf_simulator.py decode 1 2048 65536 1 4 0
# Real data: layer=20, topK=2048, N=70690, row=2023, warmup=4, use_real_data=1
python predicted_topK_filtering_perf_simulator.py decode 20 2048 70690 2023 4 1
```

#### 2b. nsys timeline profiling (kernel latency)

nsys captures the GPU timeline to measure kernel duration, grid/block config, and call frequency.

```bash
# Random data
nsys profile -o baseline_random \
  -t cuda,nvtx \
  --cuda-graph-trace node \
  --force-overwrite true \
  python predicted_topK_filtering_perf_simulator.py decode 1 2048 65536 1 4 0

# Real data
nsys profile -o baseline_real \
  -t cuda,nvtx \
  --cuda-graph-trace node \
  --force-overwrite true \
  python predicted_topK_filtering_perf_simulator.py decode 20 2048 70690 2023 4 1
```

Extract kernel stats from the `.nsys-rep` file:
```bash
# List all kernel launches with duration
nsys stats --report cuda_gpu_kern_sum baseline_random.nsys-rep
# Or open in Nsight Systems GUI for timeline view
```

Record for each kernel: **name**, **avg duration (us)**, **grid dim**, **block dim**, **shared memory (bytes)**.

#### 2c. ncu detailed kernel analysis (optional, for optimization)

ncu provides per-kernel metrics: occupancy, memory throughput, compute throughput, warp stall reasons.

```bash
# Profile a single kernel invocation with full metrics
ncu --target-processes all \
    --set full \
    --kernel-name "heuristicTopK" \
    --launch-skip 4 --launch-count 1 \
    -o baseline_ncu \
    python predicted_topK_filtering_perf_simulator.py decode 1 2048 65536 1 4 0
```

Key ncu metrics to record:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — compute utilization
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — memory bandwidth utilization
- `sm__warps_active.avg.pct_of_peak_sustained_active` — occupancy
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` — L2 cache hit rate indicator

**Gate**: Baseline latencies for both random and real data documented, with nsys kernel stats.

### Phase 3: Integration Design

Based on Phase 1 analysis, design the integration. Consider these dimensions:

**Kernel wrapping strategy** — how to extend single-row to multi-row:
- Separate compilation unit with thin multi-row wrapper calling the micro-kernel as `__noinline__` device function (preferred for build isolation and ptxas optimization)
- Inline into existing indexer file (simpler but risks optimizer interference and compile time)
- Header-only template (forces recompilation of dependents)

**Gating mechanism** — how the heuristic path is selected at runtime:
- Conditions based on `preIdx != nullptr`, stride, topK value, preIdxCount, sequence length
- Fallback to existing sort paths when conditions not met

**MTP (next_n > 1) handling**:
- preIdx is per-request, not per-token: row indexing must use `rowIdx / next_n`
- Valid range per row: `rowEnd = seq_len - next_n + (rowIdx % next_n) + 1`
- Only the last MTP position's TopK should be saved for next-step hints

**Python-side pre_idx management** — must be CUDA-Graph safe:
- Pre-allocated metadata buffers per layer (indexed by local layer index)
- Feedback loop: each step's write becomes next step's read
- +1 offset to preserve RoPE relative distances
- Only `.copy_()`, in-place ops, and tensor views inside captured forward
- NO `.item()`, `torch.tensor()`, or Python dict lookups on GPU data in the forward path

**Configuration** — user-facing toggle:
- Config field on `DeepSeekSparseAttentionConfig` (default disabled)
- Must be propagated through model config builder (check for config reconstruction that may drop new fields)

**Memory budget**:
- Per-layer persistent buffer: `num_layers x max_batch x topK x 4B`
- Shared staging buffer: `max_batch x topK x 4B`
- When disabled: zero extra memory

Document the chosen approach and produce a file modification plan (which files to create, which to modify).

**Gate**: Design document with file plan.

### Phase 4: Implementation

Execute the file plan from Phase 3. General ordering:

1. **Kernel layer** (C++/CUDA) — create/modify kernel files, wrapper, entry point, thop binding. These require wheel rebuild.
2. **Python layer** — config field, config propagation, metadata buffers, pre_idx load/save logic. These do NOT require rebuild.
3. Run `pre-commit run` on all changed files. Re-stage any auto-formatted files.
4. Rebuild wheel if C++ files changed:
   ```bash
   python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --use_ccache --cuda_architectures "100-real"
   pip install build/tensorrt_llm-*.whl --force-reinstall
   ```

**Gate**: All files created/modified, pre-commit passes, wheel rebuilt.

### Phase 5: Functional Correctness

#### 5a. Existing tests (regression check)

Run the existing indexer TopK unit tests to ensure non-heuristic paths are unbroken:
```bash
pytest tests/unittest/_torch/thop/parallel/test_indexer_topk.py -v
```
These tests call `indexer_topk_decode` **without** `pre_idx`, so they only exercise the radix-sort/insertion-sort fallback. They must still pass.

#### 5b. Synthesize heuristic-specific tests

The existing unit tests do NOT cover the heuristic path (they never pass `pre_idx`). Create new test cases that:

1. **Generate `pre_idx`** simulating the decode feedback loop: run `torch.topk` on logits as a reference, then use `(reference_topk_indices + 1) % seq_len` as `pre_idx` (mimicking the previous-step hint with +1 offset).
2. **Call `indexer_topk_decode` with `pre_idx`** and verify the heuristic path activates (the gating conditions must be met: `topK == 2048`, `pre_idx.size(1) == 2048`, contiguous logits).
3. **Test shapes** covering:
   - **Batch sizes**: 1 (single-row, same as micro-kernel), 4, 8, 32 (multi-row)
   - **next_n values**: 1 (standard decode), 2, 4 (MTP) — with per-request `pre_idx` shape `[batch, 2048]`
   - **Sequence lengths**: 4096, 16384, 65536, 131072 (variable valid ranges)
4. **Correctness criterion**: for each row, the **set** of TopK indices from the heuristic path must match the set from `torch.topk` (order may differ). Use value-based comparison: gather logit values at both index sets, sort descending, assert `torch.allclose`.
5. **Edge cases**:
   - `pre_idx` contains indices beyond current valid range (simulating stale hints from cold start) — kernel must ignore them safely
   - Rows with `seq_len < topK` — output should have valid indices plus -1 padding
   - Mixed short/long sequences in the same batch

Reference for pre_idx generation pattern (from standalone test `predicted_topK_filtering_perf_simulator.py`):
```python
# Simulate previous step's TopK as hint for current step
ref_topk_indices = logits.topk(index_topk, dim=-1)[1]
pre_idx = (ref_topk_indices + 1) % seq_len_per_row  # +1 RoPE shift
```

#### 5c. Verify heuristic path activation

Use nsys on a test case to confirm the kernel name is the heuristic variant (not `topKPerRowDecode`):
```bash
nsys profile -o test_heuristic -t cuda \
  pytest tests/unittest/_torch/thop/parallel/test_indexer_topk.py::test_heuristic_topk_decode -v -s
nsys stats --report cuda_gpu_kern_sum test_heuristic.nsys-rep | grep -i "topk\|heuristic"
```

**If tests fail**, diagnose:
- Index out of range → check rowEnd computation
- Shape mismatch → check `pre_idx.size(0)` vs `numRows / next_n`
- Heuristic path not taken (wrong kernel in nsys) → check gating conditions
- CUDA Graph crash → check for `.item()`, `torch.tensor()`, or dynamic allocation in forward

Fix and re-test until all pass.

**Gate**: All existing + new heuristic tests pass; nsys confirms heuristic kernel activation.

### Phase 6: Performance Validation

#### 6a. Standalone kernel comparison

Re-run Phase 2 workloads against the **integrated** kernel (after wheel rebuild):

```bash
cd torchCPP_heuristictopk
# Same commands as Phase 2b — but now the micro-kernel is the TRT-LLM integrated version
nsys profile -o integrated_random \
  -t cuda,nvtx --cuda-graph-trace node --force-overwrite true \
  python predicted_topK_filtering_perf_simulator.py decode 1 2048 65536 1 4 0

nsys profile -o integrated_real \
  -t cuda,nvtx --cuda-graph-trace node --force-overwrite true \
  python predicted_topK_filtering_perf_simulator.py decode 20 2048 70690 2023 4 1

# Compare kernel stats side-by-side
nsys stats --report cuda_gpu_kern_sum integrated_random.nsys-rep
nsys stats --report cuda_gpu_kern_sum integrated_real.nsys-rep
```

Target: integrated kernel duration < 105% of Phase 2 baseline.

#### 6b. Performance gap diagnosis (if > 5% overhead)

Use ncu to compare integrated vs standalone:

```bash
ncu --target-processes all \
    --set full \
    --kernel-name "heuristicTopK" \
    --launch-skip 4 --launch-count 1 \
    -o integrated_ncu \
    python predicted_topK_filtering_perf_simulator.py decode 1 2048 65536 1 4 0
```

Common causes and fixes:
- **Higher register count** → check `__launch_bounds__`, ensure it matches standalone kernel's block size
- **Lower occupancy** → compare `sm__warps_active` between baseline and integrated ncu reports
- **Extra memory ops** → check for alignment padding copies (`stride0 % 4 != 0` triggers `cudaMemcpy2DAsync`)
- **Scratch allocation overhead** → `cudaMallocAsync` for scratch values adds latency; consider pre-allocation
- **Different SASS quality** → verify `__noinline__` is on the device function to ensure ptxas optimizes independently

#### 6c. End-to-end TRT-LLM inference profiling

Test with DeepSeek V3.2 model, using nsys to confirm the heuristic kernel runs in production:

Use `scripts/run_e2e_bench.sh` which generates the YAML config, dataset, and runs `trtllm-bench`:
```bash
# Baseline (no heuristic)
./scripts/run_e2e_bench.sh

# With heuristic enabled
./scripts/run_e2e_bench.sh --heuristic

# With heuristic + nsys profiling
./scripts/run_e2e_bench.sh --heuristic --profile
```
Override defaults via environment: `MAX_BATCH=8 ISL=2048 EP=8 ./scripts/run_e2e_bench.sh --heuristic`

Verify in nsys timeline:
- Kernel name contains `heuristicTopK` (not `topKPerRowDecode`) in decode iterations
- No unexpected `cudaMemcpy` or sync operations around the kernel
- Kernel duration is consistent across decode steps (no outliers from cold start)

```bash
# Extract kernel stats from e2e profile
nsys stats --report cuda_gpu_kern_sum e2e_heuristic.nsys-rep | grep -i "topk\|heuristic"
```

#### 6d. A/B comparison (heuristic vs baseline)

Run the same benchmark twice — once with heuristic enabled, once disabled — to measure end-to-end throughput impact:

```bash
# Baseline (heuristic off)
# sparse_attention_config:
#     algorithm: dsa
#     enable_heuristic_topk: false

# Heuristic (heuristic on)
# sparse_attention_config:
#     algorithm: dsa
#     enable_heuristic_topk: true
```

Compare: tokens/sec, time-to-first-token (TTFT), inter-token latency (ITL).

**Gate**: Integrated kernel latency within 5% of standalone baseline; end-to-end heuristic kernel confirmed in nsys.

## Critical Design Constraints

Refer to [pitfalls.md](pitfalls.md) for detailed lessons learned from prior integration attempts.

Key points:
- **Per-layer isolation**: Each attention layer needs independent prev_topk storage to avoid cross-layer data corruption
- **CUDA Graph feedback loop**: The graph captures read→compute→write on the same pre-allocated buffer; each replay's write = next replay's read
- **Config propagation trap**: Model config builders may reconstruct sparse attention config objects, dropping new fields unless explicitly forwarded
- **Cold start**: First decode step after prefill has stale hint data — kernel must validate indices safely (`idx >= 0 && idx < N`)
- **Build boundary**: C++/CUDA changes require wheel rebuild; Python-only changes do not
