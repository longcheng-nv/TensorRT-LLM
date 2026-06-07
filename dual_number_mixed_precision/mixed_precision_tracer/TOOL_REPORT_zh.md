# 工具报告 — `mptracer`:混合精度误差追踪的模块化 agent/harness

**日期:** 2026-06-07 · **状态:** 已构建并验证(回归测试在本地 B200 host 上 PASS) ·
**来源:** 由 DSV4 MoE dual-number harness(8 轮迭代,`../dsv4_moe_harness/`)产品化而来。
**英文版:** `TOOL_REPORT.md`。

---

## 1. 做了什么、为什么

DSV4 MoE 的工作在一个真实算子上验证了一套方法论(前向 dual-number 误差归因 + ρ-gated 升级 +
roofline/融合门控的精度搜索)。本工具按 Anthropic agent 工程分析,把这套方法论**模块化**成可复用、
具一定通用性的包:**算子无关的 harness 内核**(确定性环境)、**可插拔 twins/backends**、**5 个 Skill**
(固化的程序性知识)、以及一个**专科 agent**。

架构基石是**信任边界**——现在由类型结构强制,而非靠纪律:每一个数字都活在 `MeasureResult` 里;
proposer 的数值依据只能通过 id **引用**某个结果(`PolicyProposal.result_ref`)。这与 Claude Code 自身
的 propose-verify 模式同构:模型提议,确定性环境裁决。

## 2. 架构(使其通用的"接缝")

```
            ┌──────────── 信任边界(类型强制) ────────────┐
 agent /    │ 提议:twin 代码 · dual 规则 · 精度策略 · 下一探点   │
 5 skills   │ 读取:MeasureResult(rho/budget/fidelity/latency/接受) │
            └────────────────────────┬───────────────────────────────┘
                                      ▼
   ┌──────────────── mptracer(算子无关内核)────────────────┐
   │ core.measure() · types(JSON) · metrics · roofline · escalation · policy_search │
   └──────────────┬────────────────────────────────────┬─────────────────────────────┘
                  ▼(插件)                                ▼(插件)
            twins/(逐算子 dual twin)            backends/(host fake-quant │ silicon)
            moe_gemm: FC1(SwiGLU)/FC2 ……          torch _scaled_mm fp8/fp4 │ 融合 cute_dsl op
```

| 层 | 通用性 | 位置 |
|---|---|---|
| Harness 内核(measure / metrics / escalation / roofline / policy search) | **算子无关** | `mptracer/*.py` |
| Twin 插件(primitive 序列 + 源标记 dual 通道) | 逐算子(GEMM/attention/MLP 同族共享模板) | `mptracer/twins/` |
| Silicon backend(量化 recipe + 内核调用) | DSL/版本相关(已隔离、已锁版) | `mptracer/backends/`、`silicon-precision-oracle` skill |
| 方法论(升级阶梯、fidelity 规则、融合门控) | **完全通用** | 5 个 Skill |

## 3. 五个 Skill(模块化的方法论)

1. **`mixed-precision-error-tracing`** —— 顶层 playbook:归因 → ρ 门控 → roofline 门控 → 搜索 →
   silicon 验证。编码升级阶梯与"单边 ρ"诊断法。
2. **`dual-twin-authoring`** —— 如何加新算子:primitive 传播契约、升级通道、3 项强制自校验。
   **这是通用性的杠杆。**
3. **`precision-policy-search`** —— 归因引导的 greedy / LLM-in-loop 策略搜索;proposer 只排序旋钮,
   实测误差才接受。
4. **`silicon-precision-oracle`** —— 版本脆弱的 fp8/fp4 `_scaled_mm` + 融合内核 recipe、正确的
   twin-fidelity、GEMM-only-vs-量化 分开计时、以及解码融合门控。
5. **`autoresearch-loop`** —— gap-board / acceptance-gate / `/loop` 协议,用于自主、自记录的多步运行。

外加 **`mixed-precision-specialist`** agent,在五者之间路由并强制信任边界——设计为与 PerfBot 的
triton/cudeepy/tileir specialist 并列接入。

## 4. 验证(harness 拥有数字)

`tests/test_regression.py` 在重构后复现了已验证的迭代数字:

| 检查 | 结果 | 来源迭代 |
|---|---|---|
| FC2 单边 nvf4 dual == fp64 一阶 | `rho = 2.54e-8` ✅ | iter 1 |
| 一次 budget vs leave-one-out | `cosine = 1.000000` ✅ | iter 1 |
| FC2 双边 + cross_term 补上双线性缺口 | `rho = 2.55e-8` ✅ | iter 2/6 |
| FC1(K=512)+ cross + SwiGLU 二阶 | `rho = 0.046` ✅ | iter 5 |

重构保持正确性。回归测试还**暴露了一条新的诚实警示**:SwiGLU 二阶残差**并非维度无关**——在真实
DSV4 K=4096 下是 `rho = 0.147`,而非 0.046,因为 gate 量级落在 silu 曲率更深的区域。故在非线性路径上,
应在真实收缩维验证,并在生产 K 下预期需要 ρ-gated ablation 兜底。

## 5. 通用性边界(诚实)

- **可干净泛化:** 方法论(信任边界、ρ 门控、升级阶梯、roofline/融合门控、loop 协议)与算子无关;
  twin 模板覆盖"线性/双线性 matmul + 光滑非线性 epilogue"族(GEMM、attention、MLP、MoE)——
  即 TRT-LLM 大多数热点。
- **会失效:** 任意控制流 / 奇异 fused epilogue / 重非光滑路由(top-k、稀疏)超出一阶 + 二阶 Taylor;
  SwiGLU 残差的 K 依赖说明非线性路径在真实维度需谨慎。
- **运营风险(设计上已隔离):** twin 维护(auto-twin 合成是 PARKED 的 Phase-2 杠杆)在 `twins/`;
  DSL/recipe 脆弱性(fp4 `_scaled_mm` recipe 绑定 torch/torchao 版本)在 `backends/` +
  `silicon-precision-oracle` 的版本锁。

## 6. 如何使用

- **库:** `import mptracer; mptracer.measure(...)` / `greedy_policy_search(...)`。
- **作为 agent:** 委派给 `mixed-precision-specialist`,它路由五个 skill 并守住信任边界。
- **自主运行:** 用 `autoresearch-loop` 的 `PROGRAM.md` gap board 配 `/loop`。
- **加新算子:** 按 `dual-twin-authoring`,在 `twins/<op>.py` drop 一个并加回归行。

## 7. 成熟度与下一步

已完成:算子无关内核、MoE twin 插件、升级阶梯、策略搜索、roofline/融合门控、5 个 skill、专科 agent、
通过的回归测试、信任边界作为类型不变量。

要达到完全生产级通用性,剩余:
1. **Auto-twin 合成**(AST / 算子重载)→ twin 从内核自动再生;消除手写瓶颈(决定性的通用性一步)。
2. **第二个内置 twin**(attention / FlashAttention),证明模板可迁移出 MoE。
3. **Silicon backend 模块**:把 iter3/6/8 driver 统一封进 `backends/` API,带 CI/headless 的
   `silicon-pending` 降级。
4. **PerfBot 集成:** 把 specialist 注册进 performance-optimization 路由表。

## 8. 评估

方法论本已正确且在真实 silicon 上验证;本工具使其**可复用、半通用、且构造上信任安全**。它是一个站得住
脚的"GEMM/attention/MoE 族的 primitive 级误差归因 + 精度搜索专科 agent"——不是万能内核银弹,且它如实
声明了这一点。两个真实风险(twin drift、DSL 脆弱性)被关进两个插件层,这正是 agent/harness 工程师希望
它们待的地方。
