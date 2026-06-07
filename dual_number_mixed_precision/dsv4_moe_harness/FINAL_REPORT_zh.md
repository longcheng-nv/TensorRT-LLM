# 最终报告 — DSV4 MoE GEMM Dual-Number 混合精度 Harness

**日期:** 2026-06-07 · **硬件:** 本地 8× NVIDIA B200 (sm_100) · **软件栈:** TensorRT-LLM
1.3.0rc15(DSV4 分支)、Torch 2.11、torchao 0.15、NumPy 2.3 · **信任模型:** harness 拥有每一个数字,
LLM 只提议 artifact(twin / dual 规则 / 精度策略)并读取实测结果。

> 配套文档:`HARNESS_API_DESIGN.md`(Phase-0 API)、`SCOPE_DSV4_MOE_BS1-512.md`(已验证架构 + 范围)、
> `PROGRAM.md`(循环 steering)、`RESEARCH_LOG.md`(逐迭代日志)。英文版:`FINAL_REPORT.md`。

---

## 1. 这是什么

一个 **Phase-0 统一 harness**:把 DeepSeek-V4 MoE grouped GEMM 的 dual-number 数值误差追踪,变成一个
确定性、可被 agent 反复调用的单一接口;外加在真实 B200 上跑的 **8 轮 autoresearch 迭代**——它们
(a) 在 DSV4 MoE 算子上验证 dual-number 误差模型,(b) 为其非线性 SwiGLU epilogue 建立所需的升级阶梯,
(c) 跨 batch size 1–512 **映射并最终解决**生产 fp4 加速问题。

设计遵循 proposal 的 LLM-native 原则:模型提议 twin / dual 规则 / 精度策略;确定性的
`harness.measure(...)` 拥有每一个实测数字;generate-and-verify 边界把模型输出挡在数值信任路径之外。

## 2. 目标算子(从活跃 editable install 验证)

`…/perf/workloads/DSV4/TensorRT-LLM`,模型 `DeepseekV4Config`:

| MoE 配置 | 值 | | 内核 | 值 |
|---|---|---|---|---|
| hidden_size | 4096 | | 类 | `Sm100BlockScaledContiguousGroupedGemmKernel` |
| moe_intermediate_size | 2048 | | MMA | tcgen05 block-scaled UMMA (SM100) |
| n_routed_experts | 256 | | 累积 | Float32 |
| num_experts_per_tok | 6 | | 格式 | MXF8 (sv32)、MXF4 (sv32)、**NVF4** (sv16) |
| n_shared_experts | 1 | | 融合变体 | `…gather_grouped_gemm_act_fusion`(gather+GEMM+SwiGLU+fp4 输出) |
| MoE 层数 | 40(43 − `first_k_dense_replace`=3) | | 激活 | SwiGLU(`silu`,gate+up 融合),`swiglu_limit`=10 |

每个 expert 两个 GEMM:**FC1** K=4096 → N=2·2048=4096(gate_up 融合,SwiGLU 后 → 2048);
**FC2** K=2048 → N=4096。BS=1…512 映射为 `M_total = BS·top_k` 个路由 token-expert 对。

## 3. Harness(Phase 0)

单一入口 `measure(MeasureRequest) -> MeasureResult`,给定 `(request, seed)` 完全确定:

- **输入:** shape(FC1/FC2,M/K/N,组数)、精度策略(ab_format ∈ {bf16, mxf8, nvf4, mxf4}、
  out_dtype、逐算子量化开关)、分布、参考精度、升级模式。
- **输出(JSON):** `measured_rel`、`predicted_rel`、**`rho`**(信任门)、Higham `μ_F`、逐源 budget
  + 对 leave-one-out 的 cosine、`twin_fidelity`、`flip_risk`、latency/SOL、regime、接受判定。
- **Twin:** primitive 对齐的 FC1(SwiGLU)/FC2 twin,携带源标记 dual 通道(`A_input_round`、
  `B_input_round`、`mma_accum`、`D_store_round`,以及 `cross_AB`、`swiglu_2nd`)。
- **确定性:** fp64 参考、**TF32 关闭**、固定种子、残差 `δz` 在真实 block scale 下计算。

## 4. 八轮迭代(所有数字由 harness 实测)

| # | Gap | 位置 | 判定 | 关键实测结果 |
|---|---|---|---|---|
| 1 | API + twin | host | **KEPT** | dual vs fp64 一阶 = **2.54e-8**;一次 budget vs leave-one-out cosine = **1.000000** |
| 2 | BS=1–512 归因图 | host | **KEPT** | FC2 cross-term → 2.5e-8(所有格式);point-ρ 随精度变粗递增(nvf4 6.7e-2 < mxf8 1.1e-1 < mxf4 1.4e-1);outlier **不**抬高 FC2 ρ |
| 3 | silicon fp8 oracle | B200 | **KEPT** | twin_fidelity **1.66e-3**;fp8 加速 0.77×(BS1) → **1.44×**(BS512);仅 compute-bound 才赢 |
| 4 | FC1 matmul cross-term | host | **DISCARDED** | cross-term 仅 0.160→0.146;单边 FC1 ρ=0.103 ⇒ 误差是 **SwiGLU 曲率**,非 matmul |
| 5 | FC1 SwiGLU 二阶 | host | **KEPT** | 加二阶 epilogue 通道:FC1 nvf4 ρ **0.160 → 0.046**;FC2 不受影响 |
| 6 | 生产 NVF4 fp4 silicon | B200 | **KEPT** | twin_fid **1.67e-3**、meas_rel **0.134**;fp4 GEMM 0.9×(BS1)→**2.10×**(BS512);**未融合 eager 量化把解码压到 0.05–0.15×** |
| 7 | 框架融合(CUDA-graph) | B200 | **DISCARDED** | CUDA-graph 融合仅 0.054×→**0.27×**(仍比 bf16 慢 3.7×) |
| 8 | **真实融合 act-fusion 内核** | B200 | **KEPT** | 生产融合内核 **1.82×(BS1) / 1.90×(BS8) / 1.75–1.78×(BS32–512)** vs bf16 —— fp4 在解码也赢 |

## 5. 四条结论

### 5.1 dual 误差模型在线性路径上精确,在非线性路径上一阶
**FC2(线性)** 的一阶 dual 预测与 fp64 一阶吻合到 **2.5e-8**,一次性逐源 budget 等于精确 leave-one-out
归因(**cosine 1.0**)。唯一的一阶缺失是双线性 `δA·δB` 交叉项,一个 `cross_AB` 通道即可**精确**补上
(6.7e-2 → 2.5e-8)。**outlier 不破坏线性归因**——channel-outlier 的 ρ(0.046)并不比 benign(0.067)差。

### 5.2 非线性 SwiGLU epilogue 需要二阶通道,而非交叉项
**FC1(SwiGLU)** 的主导一阶缺失是 **epilogue 曲率**,由单边 FC1(δA≡0,matmul 交叉项恒为零)已经有
ρ=0.103 证明。matmul 交叉项几乎没用(0.160→0.146,iter4 DISCARDED)。加上解析的 SwiGLU 二阶 Taylor
通道 `0.5·silu''(g)·δg²·u + silu'(g)·δg·δu`,FC1 nvf4 ρ 降到 **0.046**(iter5)。**升级阶梯:**
point dual → `+cross_AB`(matmul 双线性)→ `+swiglu_2nd`(epilogue 曲率)。

### 5.3 twin 忠实代表 silicon —— 前提是配方匹配
在真实 B200 上,`twin_fidelity` 为 **1.66e-3(fp8)** 与 **1.67e-3(fp4)**——即真实的 fp8/fp4
MMA 累积地板——但需要两处修正:twin 必须用与 silicon **相同的缩放配方**(per-tensor vs per-block),
且必须**复用 silicon 消费的同一份量化算子**(软件 fp8/fp4 模拟与硬件舍入逐元素不同)。fp4 实测误差
0.134 与之前 NVF4 研究(1.32e-1)一致。

### 5.4 fp4 解码加速存在 —— 但仅当激活量化在内核 epilogue 融合
这是本报告的头条结论,跨 iter 6→7→8 解决:

| 路径 | 解码(BS≤8)fp4 FC1 vs bf16 |
|---|---|
| 仅 fp4 GEMM 内核(iter6) | 大 BS 时最高 2.10×,但激活量化是独立的一遍 |
| **未融合** eager `nvfp4_quantize` + GEMM(iter6) | **0.05–0.15×**(灾难性) |
| **框架** 融合 —— CUDA-graph / `torch.compile`(iter7) | **0.27×**(仍慢 3.7×) |
| **内核 epilogue** 融合 —— 真实生产 `…act_fusion` 内核(iter8) | **1.8–1.9×** ✅ |

瓶颈是激活的**量化遍**(对整张张量在 HBM 上做多遍 amax/scale/cast/pack),而非 GEMM。框架融合去掉了
launch 开销,但去不掉这份 HBM 流量。生产融合内核在一次 launch 内就把 FC1 激活**直接产出为 fp4**,
独立的量化遍消失,fp4 在解码也胜过 bf16。

## 6. 对 DSV4 MoE 混合精度的可执行答案

- **用什么精度、用在哪:** NVF4(fp4 e2m1 + e4m3 block scale,sv=16)是正确的 MoE 权重格式——fp4 GEMM
  随 BS 0.9×→2.1×,误差 0.134(已知 NVF4 水平)。对本算子,MXF8/MXF4 在精度与速度两轴都更差。
- **误差由哪条路径主导:** FC2 误差完全可由一阶解释且对 outlier 鲁棒;FC1 误差需要 SwiGLU 二阶项——
  即精度紧张时应**保护/校验 FC1 SwiGLU epilogue**,而非 matmul。
- **加速何时兑现:** fp4 在 **compute-bound(prefill / 大 BS)** 无条件取胜;在**解码仅当激活量化被融合
  进产出内核**——生产 `…gather_grouped_gemm_act_fusion` 内核正是如此(实测解码 1.8–1.9×)。
- **40%+ 目标**因此在**解码与 prefill 两个区间**都可通过生产融合内核达成;独立量化遍的开销曾是全部障碍,
  而它在生产路径中已被解决。

## 7. 诚实的局限

- dual 模型是**一阶线性化**:在注入 + 线性路径上精确,在光滑非线性路径上一阶(ρ 可测),SwiGLU clamp
  是被守卫的非光滑节点(`flip_risk`)。
- host 度量 sweep(iter2)在**缩减维度**下计算 ρ/budget(比值与维度无关;已显式记录,非静默);真实维度
  驱动 silicon 迭代。
- iter8 的 bf16 基线是**粗糙的逐 local-expert mm 代理**,故 1.8× 比值为近似——但方向(融合 fp4 ≫ bf16
  解码,框架 fp4 ≪ bf16)是明确的。
- 结果基于 FC1/FC2 grouped-GEMM 算子,而非完整 DSV4 engine;auto-twin 合成(GA8)是 PARKED 的 Phase-2。

## 8. 产物

```
dsv4_moe_harness/
├── HARNESS_API_DESIGN.md        Phase-0 API 设计(信任边界、JSON 契约、升级阶梯)
├── SCOPE_DSV4_MOE_BS1-512.md    已验证 DSV4 架构 + BS 范围 + 精度矩阵
├── PROGRAM.md                   循环 steering:gap board、acceptance gate(全部 CLOSED/PARKED)
├── RESEARCH_LOG.md              8 轮迭代,逐行(KEPT/DISCARDED,负结果保留)
├── harness.py                   harness:API + FC1/FC2 twin + 指标 + 升级
├── iter2_bs_sweep.py            host BS=1..512 归因/ρ/flip/regime 图
├── iter3_silicon.py             B200 fp8 oracle(twin fidelity + latency/SOL)
├── iter6_silicon_nvf4.py        B200 NVF4 fp4 _scaled_mm(GEMM-only vs eager-quant)
├── iter7_fused_quant.py         B200 CUDA-graph 框架融合测试
├── iter8_fused_kernel.py        B200 真实生产融合 act-fusion 内核 driver
└── results/                     逐迭代 CSV/JSON(重新生成,勿手改)
```

## 9. 下一步

1. **GA8(Phase 2):** 为 cute_dsl grouped GEMM 做 auto-twin 合成(AST / 算子重载)——从内核自动再生
   twin,使 twin 维护变成"再生并复验"。
2. **把 iter 6–8**(fp4 融合故事)折进主 proposal 的实验证据章节。
3. **完整 engine 集成:** 在真实 DSV4 MoE 层内驱动融合内核,带实测的 per-shape/per-phase 上下文。
