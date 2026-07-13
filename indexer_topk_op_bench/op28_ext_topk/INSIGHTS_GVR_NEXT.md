# op28 → GVR 家族下一步优化启发(对照 6 月证伪史过滤)

数据基础:op28 全网格(906 cells × 2 外部臂,B200-027,cold+warm L2,锚漂移 med 1.0013);
源码基础:vendored `ops/sglang_v2/sgl_kernel/deepseek_v4/topk_impl.cuh` + flashinfer
`fast_topk_clusters_exact.cuh`。证伪史 = memory `project_gvr_topk_falsification_history`。

## 1. 两个外部算子的算法解剖

### SGLang v2(4 路径按 max_seq_len/batch 分发,全部单遍框架)
统一骨架 = **"粗定界 + 边界精确"**:12-bit fp16 单调键直方图(cluster 路径 10-bit)一遍定出阈值 bin;
`> bin` 的元素在 collect 遍直接 atomicAdd-cursor 写最终输出(**above_count < K 由 bin 不变量保证,无 P4**);
bin 内候选(≤kMaxNumTie=2048)进 smem tie buffer,由**按候选数分层的 tie-select** 精确解:
≤32 → 单 warp ballot 排名(零 block barrier);≤64/≤128 → 2/4-warp 寄存器排名;
≤2048 → 4 轮 8-bit 精确 radix(全 fp32 位 + index tie-break)。
- **register 路径(N≤8K/16K)**:行一次性读进**寄存器**(每线程 kItems 条 32B 向量),直方图后直接从寄存器 scatter——零复读、零第二 kernel、单 CTA 1024 线程。
- **streaming 路径(≤cluster_floor)**:同框架,两遍向量化全局读。
- **cluster 路径(长行)**:8-CTA cluster 分块;直方图用**分区一次性 DSMEM all-reduce**(每 rank 拥有 1/8 bins,按 lane 收 8 个 peer 值 warp-sum 回写,无 leader 汇聚);collect 后**非主 rank 只做 2 个 atomicAdd 拿跨 rank 前缀 + 把 tie 候选 peer-PUSH 进 rank0 buffer + 各自 emit 自己 chunk 的胜者**;rank0 只做 ≤2048 的 tie-select。
- **调度**:`topk_plan`(1-block,~7µs/step,61 层摊销)从 seq_len 分布选 cluster 阈值+编长行工作表;31<BS≤512 时 = **30 个持久 cluster 池啃长行 + main kernel 做短行和 PDL 衔接的 epilogue**(双 kernel 拼接);occupancy 2(`__launch_bounds__(1024,2)` + `enable_smem_spilling` pragma 把溢出寄存器放进闲置 smem)。

### FlashInfer fast_topk_clusters_exact
多轮 8-bit(256-bin)radix,cluster 版首扫(NClusters 按 BS:≤32→8,<128→4,<256→2,否则 1;N<8192→1);
核心特色 = **候选回收压缩**:每轮只把阈值 bin 内候选压进 smem 双缓冲 cache
(`num_cached` 由 occupancy 公式从 smem 预算推导),溢出走全局 overflow buffer——
候选集逐轮几何收缩,后续轮只扫候选不扫全行。公共 API 带 values+int64(税 2-5%)。

## 2. 优势区间(t(GVR臂)/t(外部),>1=外部快;pooled 全场景×K geomean)

### vs op27_hls(生产 HLS)
| N | BS 1-15 | 16-30 | 31-512 | 513-2048 |
|---|---|---|---|---|
| 4096-16384 | sgl **1.9-2.2** / fi 1.1-1.6 | 1.9-2.2 / 0.8-1.5 | 1.8-2.1 / 0.9-1.4 | 1.7-1.9 / 1.0-1.1 |
| 32768-65536 | 1.4 / 1.2-1.4 | 0.8-1.4 / 0.9-1.1 | 1.2-1.4 / 0.9-1.0 | 1.4 / 0.9 |
| 131072-262144 | 1.1-1.3 / 0.9-1.0 | 1.1 / 0.7-0.8 | **0.7-0.9** / 0.7-0.8 | 1.2 / 0.8 |
| 524288-1048576 (BS=1) | **0.83-0.99** / 0.6-0.8 | | | |

### vs op26_mc(PR#15198 multi-CTA)
sglang_v2 全面 1.2-2.3×,唯一接近的格 = **N≥262K & BS 31-512(0.86)**——恰是 op26 iter7 定位的 leader 尾段主导区。

要点:
- sglang_v2 的碾压区 = **短行(≤16K)全 BS**(HLS/GVR 缺 register 路径)和**长行小 BS**(cluster 行内并行);
- **GVR 家族守住的口袋**:HLS 在 hugeN BS=1(0.83-0.99)和 N≥262K 中 BS 带(0.72-0.9,部分是 sglang 双 kernel sum 口径高估);op26_mc 在同带 0.86;
- **warm-L2 下差距仍在**(HLS/sgl_v2 = 1.48×,base = 1.84×)→ 优势是**结构性的(遍数+barrier+occupancy),不是冷 DRAM 流量**——与 op15 的 warm-L2 决断法一致。

## 3. 启发(已对照证伪史;每条注明目标臂与量级依据)

**P0 — 短行 register-resident 快路径**(目标:gvr base + op21 ms_auto 新增分发支)
N≤16K 一遍读进寄存器 → 粗直方图 → 从寄存器 scatter。**与被证伪的 op15 smem 驻留本质不同**:
op15 只省 DRAM 复读(warm-L2 证明白省);register 路径消灭的是**遍数与 barrier**
(sglang warm-L2 仍快 1.8× 是直接证据)。ms dispatch 已有按 N 分支的骨架,加一支即可。
量级依据:4-16K 带 gap 1.7-2.3×,且部署包络内 4-64K 是高频段。

**P1 — op26 mc leader 尾段重构 = sglang cluster 收尾配方**(目标:op26_r0mc/r0auto)
直接采纳:① 分区一次性 DSMEM hist all-reduce(每 rank 1/8 bins,无 leader 汇聚);
② 跨 rank 前缀 = 非主 rank 2 个 atomicAdd 到 rank0 计数器(替代 leader peer 拉取);
③ **各 rank 自己 emit 自己 chunk 的胜者**(leader 不再串行写全量);
④ rank0 只解边界 bin(≤2048 候选,分层 tie-select),不跑全量 P4。
这与 iter7 排队的 D1 peer-push 同向,且把"leader 只剩多小的活"给出了参考答案。
量级依据:op26_mc 唯一输 1.2-1.6× 的带正是 leader 尾段 57-61% 的带。

**P2 — "边界精确"替代 P2 秒切 + P4 snap**(目标:gvr base 结构改造,保留 hint 资产)
GVR 的 ~2.5 遍(P1 hint 直方图 + ~1.46 次全 N secant count + collect)可压成
sglang 形态的 2 遍(1 次全 N 直方图 + 1 次 collect),exactness 只花在 ≤2048 边界候选上。
**GVR 独有的升级**:hint 可预测阈值 bin——hr 高时**跳过直方图遍**(hint 的 K 分位直接给 bin,
collect 遍顺带验证 count 不变量,失败回退)→ 比 hint-blind 的 sglang 还少一遍。
证伪史核对:Opt-L(P3 融进 P2)死于 ballot 链上的在线 slot-reserve——sglang 的 collect 用的是
朴素 smem atomicAdd cursor(GVR P3 已同款),融合的是**消灭 P2 迭代本身**,不是 Opt-L 复刻;
P4-内部 reseed 已死,而这里是**让 P4 只剩 ≤2048 候选**(正对"P3 over-collect 3.96×K"这个未跑的活杠杆)。

**P3 — 分层 tie-select 作为 P4 小候选快路径**(目标:base/op26 的 P4)
cand_count ≤32/64/128 时用 warp 寄存器排名(零 block barrier)。与 rank-scatter 生产教训一致
("P4 是 barrier-bound"):rank-scatter 整体 ~0.92× 的根因是 barrier,warp 层级直接归零。
flashinfer 的**候选回收压缩**(smem 双缓冲 + 全局 overflow)是 cand_count 逐轮收缩的
有界内存实现,可作 K=2048 大候选场景的补充。

**P4 — 持久 cluster 池 + 设备端 plan + PDL 拼接**(目标:op26 mc 的 BS 31-512 长行带 + 生产 ragged batch)
每行一个 cluster 的 launch/wave 量化税 → 30 个持久池轮询 + plan kernel 按 seq_len 分布路由;
epilogue 放 main kernel 用 PDL 衔接。生产 ragged batch(本 bench 未覆盖)收益应大于等长网格。

**P5 — 工程三件套**(目标:全臂,~10-20%)
`__launch_bounds__(1024,2)` + CUDA13 `enable_smem_spilling` pragma(高 BS 带 2048 线程/SM);
Blackwell 32B(256-bit)向量化读;runtime-K 单 kernel 减少模板分发面。
(NCU 旧发现"4 个多 K spec reg 超标压 occupancy"与此同一杠杆。)

**不做**(证伪史红线):smem 行驻留(op15)、高 BS cluster DSM(Opt-B)、P2 多阈值(Opt-F)、
P4 内部 reseed/fine-hist、P1 模型化 seed。sglang/flashinfer 的设计与这些红线全部相容
(它们同样不在高 BS 用 cluster、不做 P4 内 reseed)。

## 4. 战略判断
sglang v2 证明:**hint-blind 的"粗定界+边界精确"框架在我们全部三个场景都压过 hint-seeded 的
P2 精确阈值搜索**——hint 的正确用法不是省 P2 迭代(≈1.46 已到底),而是省掉定界遍本身(P2 启发)。
守住的口袋(hugeN BS=1、262K 中 BS)+ hint 资产 + 上述 P0-P4 是下一战役(op29)的构成。
