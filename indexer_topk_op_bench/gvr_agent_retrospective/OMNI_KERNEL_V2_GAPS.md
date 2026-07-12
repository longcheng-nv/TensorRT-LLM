# omni-kernel v2 方法论缺口分析 — production-level 通用优化 harness 的距离

> 2026-07-12。前提:v2 @27fe8ed 已含 GVR 复盘的全部四层机制。
> 本文回答:作为**通用**(跨算子类别)production harness,还缺什么;
> 哪些先验可判,哪些必须实践反馈。

## 一、先验可见的明显缺口(不必等实践)

### G1 单案例过拟合 — n=1 的方法论
v2 的全部实证权重来自**一个 kernel 家族**:memory-bound、selection/search、µs 级、
单算子、B200/cuteDSL。多个"公理"其实是该家族的校准值:
- cold-L2 canonical(25-35%)对 compute-bound GEMM 类权重很低;
- "primitive 重组 > 发明新算法"在 autotune/config-search 主导的 GEMM/attention
  空间可能不成立(那里 CudaCoder 式并行 config 锦标赛更高产);
- 全部数值常数(锚容差 ±3%、×3 批、512MB evict、≤3 dispatch 规则、50°C)是
  GVR 校准,不是普适值。
**补法**:把这些降级为 per-class profile(见 G2),常数进配置而非正文。

### G2 缺"问题类别 → 判据集"路由层
v1 有语言路由矩阵,v2 有测量阶梯,但没有第三张路由表:
| 类别 | 等价判据 | 主目标 | 主搜索维 |
|---|---|---|---|
| dense 数值 | atol/rtol | 时延/SOL 均可参考 | tiling/pipeline autotune |
| selection/search | tie-aware multiset | 对手相对 | primitive 重组 |
| **随机算子(sampling/dropout)** | **分布等价(KS/χ²)— v2 完全没有** | — | — |
| 归约+原子(float atomicAdd) | 统计容差 + 决定论开关 | — | — |
verify_exact.py 只有 dense/select 两模式;随机与非结合性浮点归约类是判据真空。

### G3 假设生成未被方法论化 — autonomous 的真正瓶颈
v2 把"如何**杀死**假设"做到了工业级,但"好假设从哪来"仍是 LLM 直觉 +
primitive 库存 + exemplar。复盘 §6.1 的结论就是:GVR 最高杠杆的方向
(portfolio、sandwich、"P4 是 cand-线性")全部来自人(T4)。v2 没有:
- profile→假设的系统枚举器(NCU stall 分类 → 候选杠杆表的自动映射);
- 对搜索空间的显式建模(哪些维度已扫、哪些未触);
- 廉价的"假设多样性"机制(如同题多 agent 独立提案再合并去重)。
**这是"人类触点降到 1-2 次"预测最可能失败的地方。**

### G4 无 SHIP 前独立复核(single-agent self-deception)
GVR 记录里最有效的一次防线是 orchestrator 对 op14 的**独立重 parse + 独立 ncu**。
v2 的 anti-patterns 靠同一个 agent 自觉执行——测量假象恰恰骗的是"当事 agent"。
production 级应硬编码:SHIP verdict 前由第二个 agent(干净上下文)独立重跑
nsys_verdict + 重 parse + 检查 gate 日志,签"复核通过"才允许 SHIP。成本极低。

### G5 standalone→production 集成断层
F006 的教训写进了正文,但 scripts 全是 standalone 接口(kernel_fn/get_inputs)。
缺一个 **Integration Phase 协议**:perf-parity 门(standalone vs 生产调用面 ≤3%)
的可执行脚本、CUDA-graph/多 stream/autotuner 交互检查、以及 e2e 精度门
(GVR 有 GSM8K gate 先例)——从"擂台赢了"到"生产真的变快了"之间目前是空白。

### G6 冷启动:无 incumbent 的算子
目标三件套要求人给 incumbent/envelope。对全新算子(没有生产默认、没有 rival)
v2 会退化到无锚状态。需定义 no-incumbent 模式:reference+roofline 做临时对手、
envelope 用部署假设占位并标注"待裁决"。

### G7 数据分布获取没有协议
Era-0 的核心教训是"每次算法危机来自数据分布",GVR 靠 real-capture harness +
validated synth skill 这两个**重资产**才闭环。v2 的 C2/C3 假设它们存在。对新算子,
"如何低成本建立 faithful 输入分布"(capture 点位选择→分布拟合→质量门→synth 生成器)
应当是 harness 的一等协议,而不是外部前提。

### G8 知识库规模化
FALSIFIED/WALLS 靠 markdown + grep 关键词。跨战役/跨家族/跨 arch 积累到几百条后:
- 关键词漏检(“smem-resident" vs "shared-memory staging"是同一个死路);
- 条目过期无触发器(arch/driver 换代后 wall 需重验——WALLS 的 one-line test
  是好设计,但没有"何时重跑"的钩子)。
补法:结构化 schema(id/keywords/domain/test/expiry-trigger)+ 语义检索,
arch 变更时自动列出待重验清单。

### G9 无停止/分配的决策理论
COST.md 是记账不是决策。op23(UB/LB)其实给出了"剩余理论空间"的度量,但 v2
没有把 `剩余空间 × ship 概率 vs 边际成本` 做成战役停止规则,也没有多假设间的
预算分配策略(autoresearch 的固定 time-budget 是一个可借的简化)。

### G10 ship 后无监护
驱动/编译器/arch 升级后的周期性 re-verdict(CI 化的 anchor+gate 重跑)没有协议。
production harness 的生命周期不该终止在 SHIP。

### G11 一刀切的仪式重量
v2 协议对"帮我把这个 kernel 快 2 倍"级别的小任务过重(每 iter 三轨门 + ledger +
commit + RESUME 刷新)。需要显式双模式:**quick mode**(单文件、L1+L2 即可)与
**campaign mode**(全协议),并写清晰的升级触发条件。

## 二、必须实践检验才能裁决的

| # | 问题 | 为什么先验判不了 | 建议的检验 |
|---|---|---|---|
| P1 | **协议遵守率**:agent 会不会跳门、偷懒、绕过 ledger | 文字协议对 LLM 的约束力只有实测才知道;哪些环节要硬化成 hook(pre-commit 检查 ITERATIONS 条目、SHIP 前强制复核)取决于实际违规模式 | 试跑战役 + 违规审计,再决定 hook 位置 |
| P2 | **跨类别迁移度**:阶梯/判据在 GEMM、attention、elementwise 上的适配 | G1 的具体形态未知 | 三类各跑一个小战役(有成熟 incumbent 的算子如 RMSNorm/GEMM-epilogue),记录哪些条款被架空/误导 |
| P3 | **自主假设质量**:无人 pivot 时能否找到 portfolio 级方向 | 这是能力问题不是协议问题 | 同一算子:v2-autonomous vs v2+人类 pivot 对照,量化 gap |
| P4 | **scripts 鲁棒性**:nsys csv 解析、graph 捕获兼容、多 kernel 的 regex 过滤、B300/H100 差异 | 工程细节只能撞 | 试跑中收集失败模式 |
| P5 | **成本曲线**:全协议的 token/GPU-h 开销 vs 收益,quick/campaign 分界线画在哪 | 依赖任务分布 | 每战役 COST.md 汇总回归 |
| P6 | **并行编排收益**:多假设并行(每假设一个 worktree/GPU)+ ledger 并发写的正确姿势 | 收益与冲突模式依赖实际战役形态 | 先单线跑通,再挑一个假设密集的战役试并行 |

## 三、结论与优先级

v2 解决的是 GVR 复盘里"人类被迫反复纠错"的那一半(测量、证伪、接力、卫生)——
这部分已经达到 production 纪律。**尚未解决的一半集中在三处**:

1. **通用性**(G1/G2/G7):方法论目前是"selection kernel 优化 harness",不是
   "通用算子优化 harness"。路由层 + 判据扩展 + 数据分布协议是升级为通用的必要件。
2. **自主性上限**(G3/G4):证伪机器强、生成机器弱,且缺独立复核。这两个决定
   "autonomous" 这个词的成色。
3. **生命周期**(G5/G9/G10):standalone 胜利 ≠ 生产胜利 ≠ 持续胜利。

**建议顺序**:先做零成本高确定性的 G4(SHIP 前独立复核)与 G11(双模式)与 G6
(冷启动条款)——纯文本改动;然后用 P2 的三类小战役同时收 P1/P4/P5 的实践数据;
拿到反馈后再动 G2(路由表)、G3(假设枚举器)、G8(知识库 schema)这些需要
设计权衡的大件。G3 是唯一可能需要架构级创新的项,其余都是工程化。
