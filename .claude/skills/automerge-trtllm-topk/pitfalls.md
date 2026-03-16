# Pitfalls & Lessons Learned

Hard-won knowledge from prior integration attempts.

## 1. Config Not Propagating

`model_config.py` reconstructs `DeepSeekSparseAttentionConfig` from pretrained fields. Any new field (like `enable_heuristic_topk`) must be explicitly forwarded in the reconstruction block at `tensorrt_llm/_torch/model_config.py` around line 509-534. Without this, the YAML config has no effect.

**Symptom**: nsys shows `topKPerRowDecode` instead of `heuristicTopKMultiRowKernel` despite YAML setting `enable_heuristic_topk: true`.

## 2. CUDA Graph Capture Crash

Any GPU→CPU sync (`.item()`, `torch.tensor(list, device=cuda)`) inside the model forward crashes during CUDA Graph capture with `cudaErrorStreamCaptureUnsupported`.

**Solution**: All heuristic logic must use only:
- `.copy_()` to pre-allocated buffers
- In-place ops (`+= 1`)
- Tensor slicing (views, no allocation)

Never use Python dicts or validity checks that touch GPU data inside the forward path.

## 3. Multi-Layer Buffer Sharing

61 Indexer instances share one metadata object. If a single `prev_decode_topk` buffer is on metadata, all layers overwrite each other — Layer 0 reads Layer 60's data.

**Solution**: 3D buffer `[num_local_layers, max_seqs, topk]` indexed by `layer_offsets[layer_idx]`.

## 4. pre_idx +1 Offset

The saved TopK comes from a query at position P. The next step's query is at P+1 (due to autoregressive generation). RoPE makes attention position-dependent. Shifting all preIdx by +1 preserves relative distances.

For MTP: last MTP position query is at `kv_len - 1`, next step's first query is at `kv_len`. Offset is always +1 regardless of next_n.

## 5. canUseHeuristic Conditions

All must be true simultaneously:
- `preIdx != nullptr`
- `stride1 == 1` (logits contiguous in last dim)
- `topK == 2048` (kHeuristicTopK)
- `preIdxCount == 2048` (kHeuristicSize)
- `preIdxStride >= 2048`
- `numColumns < 200000` (splitWorkThreshold)

If any fails, kernel silently falls back to radix sort. Check nsys kernel names to verify.

## 6. thop pre_idx Size Check

Original check: `pre_idx.size(0) == numRows`. With per-request preIdx and MTP:
`pre_idx.size(0) * next_n == numRows || pre_idx.size(0) == numRows`.

## 7. MTP preIdx Row Indexing

In the kernel: `rowPreIdx = preIdx + (rowIdx / next_n) * preIdxStride` (not `rowIdx * preIdxStride`). This matches `seqLens[rowIdx / next_n]`.

## 8. Build Requirements

Only C++/CUDA file changes require wheel rebuild:
```bash
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --use_ccache --job_count 72 --cuda_architectures "100-real"
pip install build/tensorrt_llm-*.whl --force-reinstall
```
Pure Python changes (`dsa.py`, `llm_args.py`, `model_config.py`) do NOT require rebuild.

## 9. Pre-commit Auto-Formatting

`pre-commit run` may modify files (yapf, clang-format). Always re-stage after:
```bash
pre-commit run --files <files>
git add <files>  # re-stage formatted files
```

## 10. heuristicTopKDecode Scratch Buffer

`launchHeuristicTopKDecode` uses `cudaMallocAsync` for `scratchValues` (numRows * topK * sizeof(float)). This is stream-ordered and freed via `cudaFreeAsync`. For performance-critical paths, consider pre-allocating this buffer.
