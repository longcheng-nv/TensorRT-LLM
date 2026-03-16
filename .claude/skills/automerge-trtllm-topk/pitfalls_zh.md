# 踩坑手册

从多轮集成尝试中提炼的关键教训。

## 1. 配置不生效

`model_config.py` 会重建 `DeepSeekSparseAttentionConfig`，只提取预定义字段。新增的 `enable_heuristic_topk` 字段如果没有在重建逻辑中显式透传，YAML 中的设置将被丢弃，字段始终为默认值 `False`。

**现象**: nsys 中只看到 `topKPerRowDecode`，不见 `heuristicTopKMultiRowKernel`，尽管 YAML 已设置 `enable_heuristic_topk: true`。

**修复位置**: `tensorrt_llm/_torch/model_config.py`，约第 509-534 行的 `DeepSeekSparseAttentionConfig` 重建代码块。

## 2. CUDA Graph 捕获崩溃

在模型 forward 内部执行任何 GPU→CPU 同步操作（`.item()`、`torch.tensor(list, device=cuda)`），都会在 CUDA Graph 捕获阶段触发 `cudaErrorStreamCaptureUnsupported` 崩溃。

**根本原因**: CUDA Graph capture 期间，CUDA 流处于录制状态，禁止同步操作和动态内存分配。

**解决方案**: heuristic 相关的所有逻辑只能使用以下操作：
- `.copy_()` 到预分配缓冲区（地址固定）
- 原地算术运算（`+= 1`）
- 张量切片/视图（不分配新内存）

绝对不能在 forward 内部使用 Python dict 查 GPU 数据、`.item()` 取值、`torch.tensor()` 创建新张量。

## 3. 多层缓冲区覆盖

DeepSeek V3.2 有 61 层 Indexer，共享同一个 metadata 对象。如果只在 metadata 上放一个 `[max_batch, topk]` 的缓冲区，61 层会轮流覆盖同一块内存 — Layer 0 在下一步读到的是 Layer 60 的数据，而非自己上一步的结果。

**解决方案**: 使用 3D 缓冲区 `[num_local_layers, max_seqs, topk]`，按 `layer_offsets[layer_idx]` 索引每层独立区域。

**内存开销**: batch=8 时 61 x 8 x 2048 x 4B = 3.9 MB，可接受。

## 4. pre_idx 需要 +1 偏移

保存的 TopK 索引来自上一步 query 位置 P（最后一个 MTP position），当前步的第一个 query 位于 P+1（自回归生成）。RoPE 使注意力依赖相对位置距离，因此对所有 preIdx 加 1 可保持与 query 的相对距离一致，提供更精确的初始阈值。

**数学推导**:
- 上一步 query 在位置 `kv_len - 1`，选出的 TopK 索引 = `{i1, i2, ...}`
- 当前步 query 在位置 `kv_len`
- 要保持相同相对距离：`kv_len - i'j = (kv_len - 1) - ij` → `i'j = ij + 1`
- 无论 next_n 为何值（1、2、4），偏移始终为 +1

**安全性**: kernel 内部校验 `idx >= 0 && idx < N`，越界索引被安全忽略。

## 5. canUseHeuristic 条件

6 个条件必须同时满足，缺一则静默回退到 radix sort：

| 条件 | 含义 |
|------|------|
| `preIdx != nullptr` | Python 侧传入了 pre_idx |
| `stride1 == 1` | logits 在最后一维连续 |
| `topK == 2048` | TopK 值为硬编码的 kHeuristicTopK |
| `preIdxCount == 2048` | pre_idx.size(1) 为 kHeuristicSize |
| `preIdxStride >= 2048` | pre_idx 行间距足够 |
| `numColumns < 200000` | logits 列数小于 splitWorkThreshold |

**调试方法**: 用 nsys 检查 kernel 名称 — 如果是 `topKPerRowDecode` 而非 `heuristicTopKMultiRowKernel`，说明某个条件不满足。

## 6. thop 绑定的 pre_idx 维度检查

原始代码要求 `pre_idx.size(0) == numRows`（按 token）。引入 per-request preIdx 和 MTP 后需放宽为：

```cpp
pre_idx.size(0) * next_n == numRows || pre_idx.size(0) == numRows
```

## 7. MTP 下的 preIdx 行索引

在 kernel 中，preIdx 的行寻址必须使用 `rowIdx / next_n`（按请求），而非 `rowIdx`（按 token）：

```cpp
rowPreIdx = preIdx + (rowIdx / next_n) * preIdxStride;
```

这与 `seqLens[rowIdx / next_n]` 的索引方式一致。

## 8. 构建需求

| 修改文件类型 | 是否需要重新构建 wheel |
|---|---|
| C++/CUDA (`.cu`, `.cpp`, `.cuh`) | 是 |
| Python (`.py`) | 否 |

构建命令：
```bash
python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --use_ccache --job_count 72 --cuda_architectures "100-real"
pip install build/tensorrt_llm-*.whl --force-reinstall
```

纯 Python 改动（`dsa.py`、`llm_args.py`、`model_config.py`）不需要重新构建。如使用 `pip install -e .`（开发模式），Python 改动即时生效。

## 9. pre-commit 自动格式化

`pre-commit run` 可能修改文件（yapf 格式化 Python、clang-format 格式化 C++）。修改后文件变为未暂存状态，需要重新 add：

```bash
pre-commit run --files <files>
git add <files>  # 重新暂存被格式化的文件
```

## 10. heuristicTopKDecode 的 scratch 缓冲区

`launchHeuristicTopKDecode` 内部使用 `cudaMallocAsync` 分配 `scratchValues`（大小 = numRows x topK x sizeof(float)），通过 `cudaFreeAsync` 释放。这是流序分配，开销较小但非零。

对于性能极致优化场景，可考虑将此缓冲区预分配为固定大小（需外部传入指针）。

## 11. CUDA Graph 反馈环路机制

CUDA Graph 捕获的操作序列（per layer）：

```
① 读取: staging[:B] = prev_topk[layer, :B]     ← 上一步结果
② 偏移: staging[:B] += 1                        ← RoPE 补偿
③ TopK:  indexer_topk_decode(pre_idx=staging)    ← 用作 hint
④ 写回: prev_topk[layer, :B] = new_topk         ← 本步结果
```

每次 replay 的第④步写入，自动成为下次 replay 第①步的读入。这个"反馈环路"是 CUDA Graph 兼容设计的核心。

**冷启动**: 新请求占据的 batch 位置可能包含上一个请求遗留的陈旧数据。kernel 的 `idx >= 0 && idx < N` 校验保证安全，仅第一个 decode 步需多几次收敛迭代，之后即进入稳态。
