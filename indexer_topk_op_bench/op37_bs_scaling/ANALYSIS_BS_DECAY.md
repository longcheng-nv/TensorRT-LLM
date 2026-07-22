# op37 附录: r3_v11 BS>1 线性衰减的机理拆解

2026-07-22, umbriel-b200-026 GPU0。探针 = `ms_probe.py`(多 stream 循环) +
`graph_probe.py`(CUDA-graph fork-join, 剥离 host 发射成本)。CUDA-event 粗筛
量级判断(graph replay host 开销极小, 数字可信;裸 Python 多 stream 的 head
参照值被 host 开销污染, 以 bs_data.csv 的 nsys 数为准)。

## 三层机理(逐层实证)

### 1. "每行一个 cluster" + 同 stream 顺序发射 = 严格串行(主因, 结构性)

- `kernel.cu` 的 launch 配置: **grid = 恰好 1 个 cluster/行** —— direct 档
  1 CTA(148 SM 占 0.7%), 寄存器驻留档 1/4/8/16 CTA(≤10.8%), 无全局 scratch。
- 同 stream 上 BS 次 launch 首尾相接: nsys 实测 BS=1024 时 µs/行 ≈ BS=1
  单行时间(7.3–12.9 vs 7.5–12.4), 零摊销、零重叠 —— GPU ≥89% 的 SM 全程空转。
- host 间隙只占 3.3%(flash_128k BS1024: NVTX 墙钟 8612µs vs kernel 时间和
  8328µs) —— 衰减不是 host 循环慢, 而是 kernel 串行本身。

### 2. 反直觉: 裸 Python 多 stream 循环救不了(host 发射率瓶颈)

`ms_probe.py`: 行轮转到 2/4/8/16 个 stream, 结果**更慢**(vs 单 stream
0.53–0.97×, 全部 exact)。原因: 每行 kernel 仅 8–12µs, 与每次 host 发射成本
(pybind + `torch.cuda.stream` 上下文 + cudaLaunchKernelEx)同量级 ——
发射间隔 ≥ kernel 时长, 任意时刻仍只有 ~1 个 kernel 在飞, 多 stream 只
增加切换开销。即: **朴素循环的病根有两条 —— GPU 串行 + host 发射率,
只改 stream 派发治不了第二条。**

### 3. CUDA-graph fork-join: 行间本可完全并发(硬件无冲突)

`graph_probe.py` 把整个 BS 行 DAG 捕获成单图回放(host 成本≈0):

| cell | BS | 顺序(图内 1 stream) | 图内 16 stream | 恢复 | µs/行 | head nsys 参照 |
|---|---|---|---|---|---|---|
| flash_128k | 64 | 495.8µs | **58.7µs** | 8.4× | 0.92 | 17.0µs |
| flash_32k | 64 | 453.3µs | **56.3µs** | 8.0× | 0.88 | 11.1µs |
| pro_1024k | 64 | 694.1µs | **91.4µs** (S16) | 7.6× | 1.43 | 50.2µs |
| flash_128k | 8 | 67.6µs | **21.2µs** (S8) | 3.2× | 2.65 | 13.1µs |
| pro_1024k | 8 | 86.4µs | **24.9µs** (S8) | 3.5× | 3.11 | 28.2µs |

- 全部配置 exact —— kernel 无隐藏的跨 launch 依赖, 行间并发安全。
- 并发度天花板 ≈ 148/CS 个 cluster 共驻(S16→S32 无增益): 8-CTA 档 ~18 行,
  16-CTA 档 ~9 行, 之后按波次线性。
- **pro_1024k BS=8 图回放 24.9µs 已略胜 head 28.2µs** —— 高价 head 格上
  fork-join 图能把胜区从 BS=1 推到 BS≈8; 但 flash 格 BS≥8 仍输 head
  1.6–3.5×(head 批量臂把行摊进同一次扫描, 摊销结构性更优)。

## 结论

1. 用户判断正确并可加细: 线性衰减 = "naive 循环"的两条腿 —— **同 stream
   GPU 串行**(主) + **host 逐行发射率**(隐藏, 使多 stream 循环也无效)。
2. kernel 本身不是障碍: 无全局 scratch、行间硬件可并发, 8× 并发余量实测存在。
3. 要吃到 BS>1, 按代价递增的三档:
   (a) **CUDA-graph fork-join**(零 kernel 改动): 胜区扩到 BS≈8(仅
       高价 head 格, 如 K=1024/N=262K); 需按 (BS, 形状, 地址) 建图, 生产
       接入面大, 收益窄;
   (b) **C++ 侧批量发射器**(单 stream→多 stream in C++, 免 Python 税):
       上限同 (a);
   (c) **kernel 侧批处理**(grid.y 批量 / row-team 共驻, compB
       `kf_bs_scaling/ext/` 已验证的方向): 唯一能追平 head 批量臂
       摊销结构的路径, 属新工程。
4. **生产 dispatch 门 BS==1 的判决不变**(RESULTS.md); (a)/(c) 是若要
   扩大胜区时的候选路线。
