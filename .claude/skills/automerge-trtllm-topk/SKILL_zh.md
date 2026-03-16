# Heuristic TopK 微内核 → TensorRT-LLM 集成指南

将**单 CTA、单 batch、next_n=1** 的 heuristic TopK 微内核集成到 TensorRT-LLM 的 indexer TopK 解码路径，扩展为生产级多行 kernel。

## 输入

用户提供：
1. **微内核**: `torchCPP_heuristictopk/heuristic_topk.cuh`（单 CTA、单行）
2. **独立测试**: `torchCPP_heuristictopk/predicted_topK_filtering_perf_simulator.py`
3. **TensorRT-LLM 源码**（工作区根目录）
4. **GPU 环境**（nsys、ncu、TRT-LLM Docker）

## 集成流水线

### 阶段 1：代码库与接口分析

1. **读取微内核** — 提取核心设备函数签名、共享内存结构体、常量（TOP_K、BLOCK_SIZE 等），理解单行算法。
2. **分析 TRT-LLM 现有 indexer TopK** — 找到 decode 入口函数（`invokeIndexerTopKDecode` 或等价），理解现有排序路径，理清 `pre_idx`（启发式提示索引）从 Python 经 thop 绑定到 CUDA kernel 的完整数据流。
3. **分析 DSA 稀疏注意力 Python 代码** — 理解 `sparse_attn_indexer` 中如何调用 `indexer_topk_decode`，metadata 缓冲区如何管理，CUDA Graph capture 如何工作。
4. **识别接口差异**：
   - 多行（多 batch）支持
   - MTP 支持（next_n > 1）：每个请求产生 next_n 行，preIdx/seqLens 按请求索引
   - pre_idx 管理的 CUDA Graph 兼容性
   - 构建系统集成（CMake、独立编译）

**门控**: 输出接口差异报告。

### 阶段 2：基线性能标定

#### 2a. 功能冒烟测试（无 profiling）

```bash
cd torchCPP_heuristictopk
# 随机数据: batch=1, topK=2048, N=65536, next_n=1, warmup=4, use_real_data=0
python predicted_topK_filtering_perf_simulator.py decode 1 2048 65536 1 4 0
# 真实数据: layer=20, topK=2048, N=70690, row=2023, warmup=4, use_real_data=1
python predicted_topK_filtering_perf_simulator.py decode 20 2048 70690 2023 4 1
```

#### 2b. nsys 时间线采集（kernel 延迟）

nsys 采集 GPU 时间线，测量 kernel 执行时间、grid/block 配置和调用频率。

```bash
# 随机数据
nsys profile -o baseline_random \
  -t cuda,nvtx \
  --cuda-graph-trace node \
  --force-overwrite true \
  python predicted_topK_filtering_perf_simulator.py decode 1 2048 65536 1 4 0

# 真实数据
nsys profile -o baseline_real \
  -t cuda,nvtx \
  --cuda-graph-trace node \
  --force-overwrite true \
  python predicted_topK_filtering_perf_simulator.py decode 20 2048 70690 2023 4 1
```

从 `.nsys-rep` 文件提取 kernel 统计：
```bash
nsys stats --report cuda_gpu_kern_sum baseline_random.nsys-rep
```

记录每个 kernel 的：**名称**、**平均执行时间 (us)**、**grid 维度**、**block 维度**、**共享内存 (bytes)**。

#### 2c. ncu 详细 kernel 分析（可选，用于优化）

ncu 提供单 kernel 级指标：占用率、显存吞吐、计算吞吐、warp 阻塞原因。

```bash
ncu --target-processes all \
    --set full \
    --kernel-name "heuristicTopK" \
    --launch-skip 4 --launch-count 1 \
    -o baseline_ncu \
    python predicted_topK_filtering_perf_simulator.py decode 1 2048 65536 1 4 0
```

关键 ncu 指标：
- `sm__throughput` — 计算利用率
- `dram__throughput` — 显存带宽利用率
- `sm__warps_active` — 占用率
- L2 cache 命中率相关指标

**门控**: 随机数据和真实数据的基线延迟均已记录，附 nsys kernel 统计。

### 阶段 3：集成方案设计

基于阶段 1 的分析，设计集成方案。需考虑以下维度：

**Kernel 包装策略** — 如何将单行扩展为多行：
- 独立编译单元 + 薄多行包装器调用微内核的 `__noinline__` 设备函数（推荐：构建隔离 + ptxas 独立优化）
- 内联到现有 indexer 文件（简单但有优化器干扰和编译时间风险）
- 纯头文件模板（强制重编译依赖文件）

**门控机制** — 运行时如何选择 heuristic 路径：
- 基于 `preIdx != nullptr`、stride、topK 值、preIdxCount、序列长度等条件
- 条件不满足时回退到现有排序路径

**MTP（next_n > 1）处理**：
- preIdx 按请求索引而非按 token：行寻址使用 `rowIdx / next_n`
- 每行有效范围：`rowEnd = seq_len - next_n + (rowIdx % next_n) + 1`
- 仅保存最后一个 MTP position 的 TopK 用于下一步提示

**Python 侧 pre_idx 管理** — 必须 CUDA Graph 安全：
- 按层预分配 metadata 缓冲区（按本地层索引）
- 反馈环路：每步写入 = 下步读取
- +1 偏移以保持 RoPE 相对距离
- 捕获的 forward 内仅允许 `.copy_()`、原地运算、张量视图
- 禁止在 forward 路径中使用 `.item()`、`torch.tensor()`、Python dict 查 GPU 数据

**配置管理** — 用户可见的开关：
- 在 `DeepSeekSparseAttentionConfig` 上添加配置字段（默认关闭）
- 必须通过模型配置构建器透传（检查是否有重建 config 对象时丢弃新字段的逻辑）

**显存预算**：
- 每层持久化缓冲区：`num_layers x max_batch x topK x 4B`
- 共享暂存缓冲区：`max_batch x topK x 4B`
- 关闭时：零额外显存

记录所选方案，产出文件修改计划（哪些文件需新建、哪些需修改）。

**门控**: 设计文档及文件修改计划。

### 阶段 4：代码实现

执行阶段 3 的文件计划。一般顺序：

1. **Kernel 层**（C++/CUDA）— 创建/修改 kernel 文件、包装器、入口函数、thop 绑定。需重新构建 wheel。
2. **Python 层** — 配置字段、配置透传、metadata 缓冲区、pre_idx 加载/保存逻辑。无需重新构建。
3. 对所有改动文件运行 `pre-commit run`，重新暂存被格式化的文件。
4. 如有 C++ 改动，重新构建 wheel：
   ```bash
   python3 ./scripts/build_wheel.py --trt_root /usr/local/tensorrt --use_ccache --cuda_architectures "100-real"
   pip install build/tensorrt_llm-*.whl --force-reinstall
   ```

**门控**: 所有文件创建/修改完成，pre-commit 通过，wheel 已重建。

### 阶段 5：功能正确性验证

#### 5a. 现有测试（回归检查）

运行现有 indexer TopK 单元测试，确保非 heuristic 路径不被破坏：
```bash
pytest tests/unittest/_torch/thop/parallel/test_indexer_topk.py -v
```
这些测试调用 `indexer_topk_decode` 时**不传 `pre_idx`**，仅验证 radix-sort/insertion-sort 回退路径。

#### 5b. 合成 heuristic 专用测试

现有单元测试**不覆盖** heuristic 路径（从不传 `pre_idx`）。需新建测试用例：

1. **生成 `pre_idx`** 模拟 decode 反馈环路：先用 `torch.topk` 作为参考，再用 `(参考 TopK 索引 + 1) % seq_len` 作为 `pre_idx`（模拟上一步 hint + 1 偏移）。
2. **调用 `indexer_topk_decode` 并传入 `pre_idx`**，验证 heuristic 路径被激活（门控条件：`topK == 2048`、`pre_idx.size(1) == 2048`、logits 连续）。
3. **测试形状覆盖**：
   - Batch 大小：1（单行，与微内核一致）、4、8、32（多行）
   - next_n 值：1（标准 decode）、2、4（MTP）— `pre_idx` 形状为 `[batch, 2048]`（按请求）
   - 序列长度：4096、16384、65536、131072（不同有效范围）
4. **正确性判据**：每行的 TopK 索引**集合**必须与 `torch.topk` 一致（顺序可不同）。使用值比较：在两组索引处取 logit 值，降序排序后 `torch.allclose`。
5. **边界场景**：
   - `pre_idx` 含超出当前有效范围的索引（模拟冷启动陈旧 hint）— kernel 应安全忽略
   - `seq_len < topK` 的行 — 输出应有有效索引 + `-1` 填充
   - 同一 batch 中混合长短序列

`pre_idx` 生成参考模式（来自独立测试 `predicted_topK_filtering_perf_simulator.py`）：
```python
ref_topk_indices = logits.topk(index_topk, dim=-1)[1]
pre_idx = (ref_topk_indices + 1) % seq_len_per_row  # +1 RoPE 偏移
```

#### 5c. 验证 heuristic 路径激活

用 nsys 对测试用例 profile，确认 kernel 名称为 heuristic 变体（而非 `topKPerRowDecode`）：
```bash
nsys profile -o test_heuristic -t cuda \
  pytest tests/unittest/_torch/thop/parallel/test_indexer_topk.py::test_heuristic_topk_decode -v -s
nsys stats --report cuda_gpu_kern_sum test_heuristic.nsys-rep | grep -i "topk\|heuristic"
```

**失败排查**：
- 索引越界 → 检查 rowEnd 计算
- 形状不匹配 → 检查 `pre_idx.size(0)` vs `numRows / next_n`
- heuristic 路径未触发（nsys 中 kernel 名称错误）→ 检查门控条件
- CUDA Graph 崩溃 → 检查 forward 中是否有 `.item()`、`torch.tensor()` 或动态分配

修复后重新测试直到全部通过。

**门控**: 现有测试 + 新 heuristic 测试全部通过；nsys 确认 heuristic kernel 被激活。

### 阶段 6：性能验证

#### 6a. 单算子对比

用阶段 2 相同工作负载 profile **集成后**的 kernel（wheel 重建后）：

```bash
cd torchCPP_heuristictopk
# 与阶段 2b 相同命令 — 但此时微内核已是 TRT-LLM 集成版本
nsys profile -o integrated_random \
  -t cuda,nvtx --cuda-graph-trace node --force-overwrite true \
  python predicted_topK_filtering_perf_simulator.py decode 1 2048 65536 1 4 0

nsys profile -o integrated_real \
  -t cuda,nvtx --cuda-graph-trace node --force-overwrite true \
  python predicted_topK_filtering_perf_simulator.py decode 20 2048 70690 2023 4 1

# 提取 kernel 统计并对比
nsys stats --report cuda_gpu_kern_sum integrated_random.nsys-rep
nsys stats --report cuda_gpu_kern_sum integrated_real.nsys-rep
```

目标：集成后 kernel 延迟 < 基线的 105%。

#### 6b. 性能差距排查（如 > 5% 开销）

用 ncu 对比集成版 vs 独立版：

```bash
ncu --target-processes all \
    --set full \
    --kernel-name "heuristicTopK" \
    --launch-skip 4 --launch-count 1 \
    -o integrated_ncu \
    python predicted_topK_filtering_perf_simulator.py decode 1 2048 65536 1 4 0
```

常见原因及修复：
- **寄存器数增加** → 检查 `__launch_bounds__`，确保与独立 kernel 的 block size 一致
- **占用率下降** → 对比 baseline 和 integrated 的 `sm__warps_active` ncu 报告
- **额外内存操作** → 检查对齐填充拷贝（`stride0 % 4 != 0` 触发 `cudaMemcpy2DAsync`）
- **Scratch 分配开销** → `cudaMallocAsync` 有延迟；考虑预分配
- **SASS 质量不同** → 验证设备函数标记了 `__noinline__`，确保 ptxas 独立优化

#### 6c. 端到端 TRT-LLM 推理 profiling

使用 `scripts/run_e2e_bench.sh` 自动生成 YAML 配置、数据集并运行 `trtllm-bench`：
```bash
# 基线（无 heuristic）
./scripts/run_e2e_bench.sh

# 开启 heuristic
./scripts/run_e2e_bench.sh --heuristic

# 开启 heuristic + nsys profiling
./scripts/run_e2e_bench.sh --heuristic --profile
```
通过环境变量覆盖默认值：`MAX_BATCH=8 ISL=2048 EP=8 ./scripts/run_e2e_bench.sh --heuristic`

在 nsys 时间线中验证：
- Kernel 名称包含 `heuristicTopK`（而非 `topKPerRowDecode`）
- kernel 周围无意外的 `cudaMemcpy` 或同步操作
- kernel 延迟跨 decode 步一致（无冷启动异常值）

```bash
nsys stats --report cuda_gpu_kern_sum e2e_heuristic.nsys-rep | grep -i "topk\|heuristic"
```

#### 6d. A/B 对比（heuristic vs 基线）

同一 benchmark 运行两次 — 一次开启 heuristic，一次关闭 — 测量端到端吞吐影响：

```bash
# 基线（关闭 heuristic）
# enable_heuristic_topk: false

# 实验组（开启 heuristic）
# enable_heuristic_topk: true
```

对比指标：tokens/sec、首 token 延迟 (TTFT)、token 间延迟 (ITL)。

**门控**: 集成后 kernel 延迟在独立基线 5% 以内；nsys 中确认端到端使用了 heuristic kernel。

## 关键设计约束

详见 [pitfalls_zh.md](pitfalls_zh.md)。

要点：
- **每层隔离**: 每个注意力层需独立的 prev_topk 存储，避免跨层数据污染
- **CUDA Graph 反馈环路**: Graph 捕获对同一预分配缓冲区的读→计算→写；每次 replay 的写入 = 下次 replay 的读入
- **配置透传陷阱**: 模型配置构建器可能重建 sparse attention config 对象，丢弃未显式转发的新字段
- **冷启动**: prefill 后第一个 decode 步的提示数据为陈旧值 — kernel 必须安全校验索引（`idx >= 0 && idx < N`）
- **构建边界**: C++/CUDA 改动需重建 wheel；纯 Python 改动无需重建
