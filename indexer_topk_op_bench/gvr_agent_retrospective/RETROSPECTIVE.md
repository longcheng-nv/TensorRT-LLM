# GVR 阈值类 Top-K 算子优化历程复盘 — Agent 专家视角

> 状态: COMPLETE(2026-07-12,5 个考古 agent 素材已全部回填)。锚点报告 = `op22_temporal_fixed_hr_bench/REPORT.html`。
> 配套: `OMNI_KERNEL_UPGRADE.md`(升级提案)、`trajectory.png`(轨迹可视化, autoresearch progress-plot 范式, 生成脚本 `gen_trajectory.py`)。
> 目标: 从"人类专家 + Agent/harness + LLM"合作产出生产级 kernel 的成功案例中,
> 提炼可自动化的方法论,升级 `/omni-kernel` skill,逼近 autonomous kernel optimization。

## 0. 三个时代总览(时间线骨架,已确证)

| 时代 | 时间 | 工作区 | 形态 | 关键产出 |
|---|---|---|---|---|
| Era-0 诞生 | 2026-01 ~ 2026-03 | `tllm_toolbox/indexer_topK_perf/custom_indexer_topK/` | 目录级 _bak/_latest/_v2 手工版本管理, gemini/claude 双 LLM 产物对照 | GVR(guess-verify-refine)阈值法原型, preidx temporal hint, newton 法, radix 对照, UT |
| Era-1 消融 | 2026-04 ~ 2026-06 | `CUDAProgram/auto_optimization_v1/ablation_study/gvr_phase_timing/` | 编号子目录 01-19 + 00_docs 状态账本 + SESSION_RESUME 接力 | P1-P4 分相计时, OPT1-6 上线, 12 条负结果账本, CUDA→cuteDSL 移植(17_), synth/real 校准 |
| Era-2 战役 | 2026-06-11 ~ 2026-07-12 | `TensorRT-LLM/indexer_topk_op_bench/` | git-native op 战役(每 op 一桶: ITERATIONS/REPORT/RESUME), 148 commits/15 天 | op7 rank-scatter → op8/10 infeasible 定界 → op17/18/19 组合探索 → op21 HLS ship → op22 总擂台 → op25/26/27 收口 |

### Era-2 迭代节奏(git 实测)
- 148 个 op-bench commit / 15 个活跃日;峰值日 25-28 commits(07-01 op16-17, 07-02 op18-19)。
- 单轮迭代(假设→实现→exactness→nsys→verdict→commit)最快 **10-30 分钟**(如 op16 iter0-iter2: 03:07→03:17→03:31)。
- 单 op 战役全周期:最快半天(op15: 4 commits 收口),最长 ~4 天(op21: iter0.5→iter16)。
- 成本实测(op26 COST.md): 一场战役 ≈ 15 GPU-h + ~$108 Claude tokens。

## 1. Era-0: GVR 的诞生(2026-01 ~ 2026-03, `custom_indexer_topK/`)

### 1.1 时间线要点
- **01-25**:起点 = 精确分层归约 top-K(CUB BlockRadixSort);同日即迭代为 warp-level Top-L + block merge(smem 爆炸驱动)。动机:B200 上 device radix sort 在 K=2048、N<32768 输给 on-chip 选择。
- **01-26/27**:**GVR 胚胎 + preidx 首次出现**(`onchip_topk_preidx_bak/USAGE.md`):阈值 T=min(dedup preidx) → 扫描收集 ≥T → merge,溢出 full-scan 兜底,附精确性证明。preidx 复用(temporal hint)这一核心 premise 是**人的假设**(simulator 名字 `predicted_topK_filtering_perf_simulator.py` 即整个前提)。
- **02-02**:**双 LLM 同题竞赛**——`gemini_topK/`(visited-bitmap + bitonic 流式 merge,sort-based)1 天出局无后续;Claude 的 sort-free 阈值系获全部后续投资。
- **02-09**:**GVR 完整形态首现**(Claude 产物,`onchip_topk_preidx_claude_bak/heuristic_topk.cuh`):P1 preidx min/max → P2 插值搜索 T 使 count∈[K, MAX_CAND] → P3 单 pass collect → P4 selection。此后所有 op7~op27 工作的直系祖先。
- **02-12**:**V1 插值法被集中分布(1+0.1·N[0,1])灾难性证伪**(CDF 近垂直、反复 overshoot、最多 6 次全局扫描)→ 分出三条 refine 支线并行:secant+damping(v2 主线)/ 两级 256-bin 直方图(`_radix`,注意不是 radix-sort 对照)/ FD-Newton 求根(`onchip_newton_topK`)。
- **02-13**:真实数据首次进场(DSv3.2 SWE-64K Layer 20/40)。此前全靠 distype 0-5 合成分布 + `topK(logits+Am·σ·noise)` preidx 模型。
- **03-02~03-15**:v2 主线定型 V2d-final(secant 阻尼 + bisection 兜底 + ballot-free collect + 256-bin hist + snap≤3);`ALGORITHM.md` 充分性定理 = 今日 GVR 正确性论证原型。
- **03-10→03-12**:进 TRT-LLM 生产(`torch.ops.trtllm.indexer_topk_decode`,preidx 已入生产签名);性能测试(35μs 目标)与**边界正确性 UT 重写**(-1 padding/变长/next_n>1/非对齐 stride)相隔一个月——正确性契约是集成时才补的账。
- 同期旁支:`dynamic-kernel-generator/` = NVIDIA 内部 DKG/TileIR MLIR 编译器栈 checkout(潜在替代工具链探索,未产出 topk 结果)。

### 1.2 人机分工(Era-0)
- **人**:问题设定(exact/unordered/K=2048/单CTA)、preidx premise、基线选择(拷 TRT-LLM 生产 radix kernel 做参照)、双 LLM A/B 的开设与裁决、合成分布矩阵与 Am-noise 扫描设计、02-13 引真实数据的转向、三条 refine 支线的开设与裁决、生产集成。
- **LLM**:全部 kernel 代码与逐 phase 微优化(V2a→V2d)、ALGORITHM.md 数学文档。
- 物证:`topk_ext.cu` 注释 "Optimization Constants requested by user";`_bak/_latest/_v2` 手工快照(无 git);03-05 `CLAUDE.md` + 03-12 用户第一个自写 `SKILL.md` = 从"用 LLM 写 kernel"到"给 agent 建基础设施"的萌芽。

### 1.3 Era-0 教训
1. "阈值求根"核心抽象第 2 周即定型,此后 5 个月都在优化求根算子与数据模型而非骨架——**早期方向选择(人)的杠杆远大于 kernel 微优化(LLM)**。
2. 双 LLM 同题竞赛成本极低、裁决极快——对照组的价值在快速证伪,不在公平比较。
3. **每一次算法危机都来自数据分布,不来自硬件**(V1 被集中分布击穿、Am-noise 被真实 hit-rate 分布击穿)——测试分布覆盖应先于 kernel 迭代扩张。
4. 正确性契约在原型期缺位、集成期一次性补账——后世 exactness-gate-first 是直接矫正。
5. 无版本控制的快照式演化使 provenance 只能靠 mtime 考古——后世 per-iter commit 纪律的反面教材。

## 2. Era-1: 消融研究与 12 条负结果账本(2026-04-06 ~ 06-09, `gvr_phase_timing/`)

定位:V2e GVR kernel 在 B200 vs Radix 的系统消融("大 BS 死亡谷"主线),产出 Scheme X 生产 dispatcher、12+2 条负结果清单、cuteDSL 移植、以及 op-bench 全部方法论的雏形。权威账本 = `00_docs/STATUS.md`(1108 行)+ `KERNEL_OPTIMIZATIONS.md` + `EXPERIMENT_INDEX.md`("14 次独立证伪")。

### 2.1 方法论演进的四个转折点
1. **04-06 clock64 相位计时上线**(Q1,复用 write-only scratch 零新增 HBM;首次落地花 15 turns 调试);数值立即用纯 Python CCDF 模拟交叉验证——"插桩数值必须有独立通道复核"从此建立。
2. **04-23 cudaEvent vs nsys 分歧定量化**(方法论转折点):host overhead 对小 kernel 是常量(+6-8µs),BS=1 加速比 1.65×(event) vs 1.93×(nsys),数学自洽验证。此后 nsys NVTX 成为 canonical。
3. **05-08 F006 硬规则诞生**:同一 kernel SASS,standalone JIT(preIdxOffset=0)比生产 op 慢 20.9-34.4%(P1 采样元素不同→secant seed 质量不同)——headline 数字只准走 `torch.ops.trtllm.indexer_topk_decode` 生产路径。同日 **printf 污染事故**:multi-CTA 全部 baseline 用了带 printf 的插桩 kernel(+25-45µs/launch),干净重测后"1.65-6.48× 胜"翻转为"全 cell 慢 1.5-3×",整份报告作废改名 `REPORT_pre_correction.md`。
4. **05-11 方法论合同成文**(`HOW_TO_BENCHMARK.md §5` 七条:128MiB L2 flush 每 launch 前、NVTX 格式、只报 median、preIdx[0]=行 argmax 等)。

三种计时工具最终分工:clock64=相内归因(禁做 baseline)/ nsys NVTX=一切 headline 仲裁 / cudaEvent=快速相对 Δ(且明确记录 launch floor ~12µs 占 70% wall,会误判 multi-CTA 类优化上限)。

### 2.2 Shipped(OPT4-7 + Q1b + Scheme X v1.0→1.2)与 12+2 条负结果
Shipped 验证纪律:JIT snapshot 快测(~30s 循环)→ 正收益才增量 rebuild(双 .so 同步)→ 生产 harness 复测 → 负收益 revert 但 raw csv 入档。
负结果按根因分三类:**结构墙**(FP16 scan cache/单扫融合/cp.async/L2-persistence/cluster DSM 4小时决判/P1 self-loop 91,080 真实 cells/γ exp-bisect)、**测量假象**(Oracle +51% 被误读后显式修正、printf 污染、AIR 移植静态分析架构错配)、**复杂度反噬**(cub sort 原型 reg spill +25-37%、coarse-fine 两级直方图 +14-24%)。总结论写入 `STATUS.md §已排除路径`:"Scheme X 是 B200+DSv3.2 的 Pareto 终点"——即 falsification-history 记忆的直接前身。

### 2.3 CUDA→cuteDSL 移植(Q16, 05-10~05-11, 计划 15-17 天实际 2 天)
- 诱因:cuTe-DSL radix 在 2/9 cell 胜 GVR;用户下逐字 mandate(算法等价+性能差≤5%+全量正确性)。中途一次关键人为纠错:agent 引 Q9d-04c 论证"infeasible"被用户指出是 **category error**(算法差+语言差叠加≠语言差),README 留整段 RECONSIDERED。
- 决定性影响:①交付物成为 op-bench 一切的基线(gvr_cutedsl);②cuTe DSL 陷阱表 10 条成为 op1-op27 快速迭代的工程底座;③**tie-aware exactness 判据**定型(半精度 set-equality 假阴性→"无低于 K-th 值元素+无重复+计数=K");④JIT compile-cache 把迭代周期从小时级 rebuild 降到分钟级。

### 2.4 合成数据三级演进(校准问题的暴露)
Q18 uniform(baseline 被扭曲 2.2×)→ Q19 分布拟合(Radix 恢复不变性但 GVR 仍系统性低——preIdx 时间结构没建模)→ Q19 Design C temporal-coherence(`prev_topk=topK(row+c·σ·noise)`,二分 c 逼命中率 0.50)→ 确立 **"synth = 真实加速比的上界估计"** 教义。真实 per-layer 捕获(Q9j/Q9k, B300 生产路径)发现 synth 永远发现不了的东西(L36 outlier、kFTarget 384→512 单常数修复)。Era-1 写下防御姿态但没修复分布本身(单 Beta 无法匹配真实尾部——留给 op-bench 时代的 unified temporal-synth skill)。

### 2.5 人机分工(Era-1)
- 人写规则(三层 CLAUDE.md:提案前必查已排除路径、F006、双 .so 同步、改 dispatcher 必重跑 4 份验证报告);人拍板方向发起/目标数值/scope 裁剪/认识论纠错/land 与否。
- **显式预授权 self-decide 模式**(SESSION_RESUME §用户指令逐字保留):"每个优化完成后…负收益则丢弃,继续下一个…自行给出最优解决方案后执行,不要停下来等待人为确认";自主决策点单列存档。
- 接力:顶层 SESSION_RESUME.md 每里程碑刷新 + 桶内 SESSION_LOG.md;`.perfbot/learnings/` 80+ yaml 自动沉淀踩坑。

### 2.6 Era-1 教训(浓缩)
计时器即世界观;cold-L2 canonical;插桩 kernel 永远不能当 baseline;harness 路径纪律(F006);tie-aware exactness;synth 是上界、真实数据才是判决轴;**负结果注册表是最高 ROI 的文档**(设"提案前必查"关卡);测量环境卫生(共享服务器 Pareto 数据整批作废→独占/同节点锚点要求由此而来)。

## 3. Era-2 前期: op-bench 基建与 op7-op16(2026-06-10 ~ 07-04)

### 3.1 Harness 基建的六个决定性设计(`PLAN.md`/`harness/`/`report/`)
1. **解耦 + perf-parity 门**:每个 standalone op 先对 in-tree `torch.ops.trtllm.indexer_topk_decode` 打 ~3% parity,保证结论可迁移回生产。
2. **cold-L2 + CUDA-graph 计时收敛到单函数** `harness/sweep.py::_time_both`:512MB `_EVICT.uniform_()` 冲 L2 + graph replay 单发中位;所有后续 op 只能 import 或逐字复刻——**纪律靠代码路径唯一性维持,不靠规程文字**。
3. **nsys 仲裁层** `sweep_nsys.py`:event 计时带 0.76-0.95× graph-launch 偏差;ship 裁决一律 nsys pure-kernel,后升级为 ×3 独立批取中位 + **anchor 校验**(重测锚 op 对照历史 csv,0.94-1.03 才认可本批可比)。
4. **exactness 门 = vdiff=0(sorted value-multiset)+ uniq==K**,数据必须 `synth_data.get_bundle`,禁 torch.randn(bf16 塌缩 ~256 值破坏平票语义——op7 教训)。
5. **可恢复性**:jsonl 逐 cell append + done-skip;报告 = `gen_report.py` 单一 last-writer;新 op 以 anchored ratio-transfer 叠加。
6. **测量阶梯固化**:host replay/clock64 相位(便宜选参)→ cold-L2 event(全网格)→ nsys ×3 中位(ship)→ NCU(物理归因:op8 占用率、op14 dram bytes)。升级触发规则:小 N launch 地板、event 出可疑 win、任何 ship claim。

### 3.2 逐 op 复盘表(前期)
| op | 假设 | 结局 | 决定性证据 | 耗时 |
|---|---|---|---|---|
| harness Phase0-8 | 解耦可 3% 复现 | ✅ 3900 cells | nsys spot-check 校准 event 偏差 | ~2天 |
| op7 rank-scatter P4 | 2级直方图 reseed 收敛 P4 | ✅ ship(PR#15709);但产线 sweep 揭示 clock64 插桩伪造过 "P4 win" | 产线 sweep 证伪插桩数据;F006 硬规则由此立 | ~3天+ |
| op8 GVR-turbo 1.5× | 95% cells 快 radix 1.5× | ❌ INFEASIBLE | NCU:BS=1 占用 24.3% 结构性(1/148 SM);字节模型 ~2.5 pass vs 1 | ~2天 |
| op8b B300 | 叠加正交杠杆 | ⚙️ real V3.2 K2048 首次打平 radix | nsys on real data 63/63 | 1天 |
| op9 LB dispatch | 按 seq_lens 分布 dispatch | ⚙️ 结论=不需要,always-single 最优 | B300 live probe e=0.25≠传说 0.62;并发 session 数据集标签碰撞事故 | ~5天 |
| op10 GVR-2x | 单 CTA 2× | ❌ INFEASIBLE | 数学地板(≥2 pass×54GB/s≈38µs)+ 放开约束对照(cluster 峰值 1.79×<2×)双重锁死 | **单日** |
| op12 lp vs SGLang | 低精度 P1-P3 快 50% | ❌;残值=regime dispatch | iter2 nop4 分解:P1-P3 地板已 1.2×;iter6 自纠"A/B 必须对真 incumbent" | 2天 |
| op13 P2/cand | 收紧候选砍 P3/P4 | ✅ 两次 ship(narrow table + log-count secant) | clock64 纠正 op12 结论(相位结论不跨 kernel 族);b200-019 坏散热事故→driver 加 util 门 | ~6天 |
| op14 1-pass compaction | 3 HBM pass→1 | ❌ 前提为假 | **NCU 一行证伪**:`dram__bytes_read`=1.11MB,输入≤1MB≪126.5MB L2,基线本就 ≈1 HBM pass | **单日** |
| op15 SMEM-resident | 整行进 SMEM 省 pass | ❌ | **warm-L2 A/B 终审隔离器**:warm 也慢 ⇒ 目标流量本来免费 | **单日** |
| op16 dual-threshold | 双阈值 +40% over radix | ❌ NO-SHIP | iter0 先测机制上限(tax-bound 预判)→两次用户 pivot→全 nsys 网格 0.845-1.022 | **单日** |

### 3.3 迭代循环解剖
标准节奏(op13/op16 最典型):**iter0 先测不建**(clock64/host replay 给机制定 ceiling,"before building, measure whether the mechanism can pay")→ host 原型参数搜索(host 必须先对拍真核 720/720 才有预测资格,且 host 预测被 nsys 打脸是常态)→ kernel 实现遵守"不改 vendored 文件"(subclass override / gated flag,基线永远可回归)→ exactness 门 → cold-L2 pilot → nsys ×3+anchor ship 裁决。判决词汇固定:REJECT/FALSIFIED/WASH/SHIP,每轮尾部 Next action;被否方向进 LEARNINGS "do NOT re-propose" 并回写 memory 证伪台账(每个新 op 的 PLAN 强制前置阅读)。一轮迭代典型数小时,单日关掉一个方向是常态(op10/14/15/16)。

### 3.4 人机分工(前期,文档考据)
- RESUME/HANDOFF 几乎全是 **agent 写给下一个 agent 的 paste-block**(人类只做转贴),含环境 gotcha 与"pick per user"决策菜单——agent 把决策点显式留给人。
- 人类拍板的物证:op8 原始任务 = 逐字用户中文 prompt(目标 1.5×、P2 冻结、cold-L2+nsys 纪律全是人定);op9 人类**预授权负面结论**("如 always-single 仍赢就直说"+禁擅改 dispatch.py);op10 agent 宣布收敛后由**用户追问**触发 iter5/6(放开约束再逼一步);op16 两次 pivot 都是用户指令(git message 留有 "(user constraint)")。
- 分工模式:人=出题(目标+硬约束)、选 pivot、预授权负面结论、纠测量纪律、定 ship;agent=假设生成、实现、全部测量与证伪、写接力文档、维护证伪台账。

### 3.5 失败战役沉淀的"结构墙"(支撑后期选路)
① 单CTA 占用墙(grid-limited 24.3%,寄存器类杠杆全无效→后期转 cluster);② ~2.5 global passes 定数;③ **L2 陷阱一行式判据**(N≤262144 ⇒ 输入≤1MB≪L2,任何"减 pass"杠杆空转;对偶判据:warm-L2 也慢⇒直接拒);④ P2 已 Pareto("iteration count is NOT a target");⑤ 相位结论不跨 kernel 族迁移(snap-P4 cand-bound vs rank-scatter barrier-bound 都对);⑥ **dispatch 是可 ship 答案的形态**(算法撞墙时,按 N/K/dtype 分区间取最优+大N回基线零回归);⑦ 测量伪影名录(clock64 膨胀、cold event 假 win、伪 ΔTOT、标签碰撞、坏散热漂移)→ 每次事故固化成 driver 硬门。

## 4. Era-2 中期: op17→op21, HLS 的演化与 ship(2026-07-01 ~ 07-08)

### 4.1 HLS 是什么
**HLS = h-tracked Ladder-Secant**(`op21_gvr_prod/MATH_THRESHOLD_ESTIMATION.html` §10):把阈值搜索形式化——最优阈值 = preIdx gather 样本的 plug-in 次序统计量 θ̂≈G_S⁻¹(h·K),**唯一未知量是一维标量 h(有效保留率)**;静态多阈值 ladder 本质是对未知 h 的批量网格搜索,两个性能悬崖(fallback 240µs、slot 溢出)都是"h 落在静态括号之外"。完整 HLS = M-ary ladder 单遍计数 + log-count regula-falsi fallback(Step1)+ cluster 并行 fallback 重计数(Step2)+ ĥ 自适应列放置(Step3, 推迟)。

### 4.2 演化链 op17→op18→op19→op20→op21
- **op17 portfolio(07-01, 11 iter 单日)**:用户提议 148 CTA 冗余多阈值扫描;iter0 crux 证实 G=148 冗余扫≈1 次扫(L2 复用);用户纠偏"真正杠杆=P4 随候选数线性";两次实现证伪(单CTA M=16 ALU 税、双 kernel +1 pass)推导出唯一可行形态:**单 cooperative kernel,赢家 CTA 的 smem 计数在 thr\* 处就绪,P3 零重计数**(G=16, avg 1.21×,nsys 1.21-1.67×)。
- **op18 单CTA M-ary(07-02)**:`block_count_ge_multi<M>` + **CDF-aware 离线 frac 表**(决定性杠杆,大 N 0.70-0.99×→1.06-1.35×);修正 op17 模型:P4 snap 对阈值放置敏感,非纯 cand-线性。
- **op19 sandwich(07-02, 17 iter 单日)**:用户命题双阈值三明治(thr0 保证 top-K 直写 + band 精化);发现 **ACTIVE-SET L2 规则**(起作用的是 ~400 并发活跃行是否溢出 L2);工程遗产 straddle-fracs/defer-direct/occupancy 双向性教训(省 8KB smem 反而 0.918→0.745);720 cells gm 1.122,**≥1.5× avg 证明结构性不可达**。
- **op20 extreme(07-03, 单日)**:价值=把损失面**逐块归因为结构**(小N相位链延迟墙、262K BS≤4 threshold-并行 O(N)/CTA 墙→指向 data-并行、16-bit 大N残差);反面教材=240-key dispatch 表(op21 PLAN 第一句否定)。探索→生产化的切换信号:**残余损失全部归因为结构**。
- **op21 prod(07-05~07-08, iter0.5→16)**:开局改写目标函数——生产约束入目标(≤3 条 dispatch 规则、graph 兼容、real captures exact、fail-soft),权重由用户给定。路线:host 原型 GO → row-chunked C-CTA(op20 结论)→ ablation 钉 P4 → 移植 rank-scatter(op8 primitive)→ P3 remote-store push(1.249 首次 17/17)→ 16-bit ladder → B300 HW-不变性 → **iter11 对抗 gate 修复前 0/72** → iter12.9 HLS 数学验证 → iter13 log-falsi ship(K2048 1M 2.105×)→ iter14 分布式 fallback(worst 2.0×)→ iter15 三代×三场景×三 dtype 大对测(real 轴 HLS>legacy>orig 全 dtype)→ iter16 fallback 代码瘦身(4% 税→≤1%,门槛降 n≥65536,由用户部署包络裁决触发)。

### 4.3 数学验证与硅上验证的闭环
三份文档:MATH(理论)→ HLS_VALIDATION(host 原型 78 bundle)→ ITERATIONS iter13-16(硅判决)。要点:模型成本常数全部来自已实测战役(唯一插值常数 M3 τ=1.2 被显式标记并在 iter13 实测 1.219);秩空间 bridge 被数据证伪、log-falsi 全胜;**"硅验证义务"制度**(报告 §5 列义务清单,iter13 逐项履行);代码质量税判据=跨独立二进制的**符号一致性**(区分系统税与 codegen 彩票)。核心关系:**数学不产生数字,产生结构**——证明快路径已在单遍信息论下界,把火力全部导向 fallback 价格;从 1.25×(均值)到 2×+(尾部)的转折点。

### 4.4 op17→op21 对自动化 harness 的启示(agent D 提炼)
1. 廉价探针强制阶梯:crux→host-replay→microbench→kernel("20 分钟 host replay 杀掉数周错形 kernel 工作")。
2. 证伪要带作用域:(结论, 条件域, 证据强度) 三元组——C8 在 fp32 是噪声、16-bit 是 1.14× 赢。
3. **"组合已验证 primitive"比"发明新 kernel"更高产**:HLS = op17 投机并行 + op18 ladder + op19 sandwich + op20 fused collect + op8 rank-scatter + iter7 push 的合成;op21 16 个 iter 无一全新算法。
4. 测量轴纪律:event 轴三次产生可复现谎言;配对同进程 A/B、锚点 cell 重锚、符号一致性、run 内配比才是判决。
5. 探索→生产化切换信号 = 残余损失全归因为结构;dispatch 复杂度要计入目标函数而非事后补救。
6. 经验平台期后安排一次数学形式化(不抬 geomean,但统一解释证伪史+指出剩余自由度)。
7. exactness gate 需要对抗样本+真实数据双轨(iter11 0/72、pair=(0,1) 只有 real capture 触发)。
8. 代码质量与编译键是真实优化货币(从不执行的 fallback 代码 = 系统性 4% 税;门控编译 > 运行时分支);可恢复性是 harness 原生能力(~1/3 handoff 文档是节点流失保险费,全部兑付)。

## 5. Era-2 收口: op22 总擂台与 op23-op27(2026-07-07 ~ 07-12)

### 5.1 op22 总擂台设计(锚点 `op22_temporal_fixed_hr_bench/REPORT.html`, 4.15MB 双语)
- **12 臂**同台:GVR 基线/op21 legacy/op21 HLS/GVR-mCTA/HLS-op25/op26×2/HLS-op27/Radix×3/SGLang,每个回填臂标注日期+anchor-transfer 来源节点。
- **81 批网格**:3 场景(BEST/WORST/REAL,由 op24 重定义两极)× 3 sweep 子网格 × K × dtype;`.done_*` marker 批级幂等。
- **数据可比性**:所有臂逐字节同 bundle(cell_seed=42+crc32(K|N),防恒定 seed 陷阱);数据由已验证的 `indexer-topk-temporal-synth` skill 生成(5 个 KS/边界质量门内嵌报告)。
- **锚机制**(跨节点扩擂台的唯一正解):规范指标=同 cell 同节点同批的对手比值;回填臂在异地与同址锚臂同批跑,按 `us_adj = us_arm·us_base(orig)/us_base(local)` 换算,**锚漂移 med/p10/p90 作为 QA 门**(op25 回填 med 1.0001、op27 1.0022)。
- **代价与判据**:跨节点 CSV 逐 cell 异常默认是 transfer 噪声,**同节点复测才升级为 regression**(op27 两个"回退"一真一假)。
- **事故史**:双 driver ×2(TaskStop 杀不死子树→pkill 三连+setsid 纪律)、漏传 OP22RR_ARMS 浪费 4 GPU-h(→env 整段照抄+发车后 grep `arms=` 核验)、跨节点锚噪声两次伪造 regression。

### 5.2 元分析 op(op23 UB/LB + op24 顺逆风)
- **op23**:零随机构造确定性上下界(只改 preIdx)。vs Radix UB 0.851/real 0.599/LB_eff 0.365(必须报 LB_eff:19/78 cell REAL 比 LB 构造还慢)。作用=**划定理论天花板**,终结"再调 hint 利用率"方向。
- **op24**:两阶段(host 筛 392 bundle→硅上配对 A/B)定出顺风 hr≈0.55(fast-path 窗 f∈[h,2.5h])/逆风 hr=0.05;发现 hr→速度**非单调**,直接重定义 op22 两极并触发 §1-2 重测;部署包络裁决(用户:N≤256K 主战场,1M 只是探针)改变结论口径。
- 教训:**元分析 op 花费极小,却给出剩余理论空间、不值得追的 cell、和后续所有战役的场景定义**——应在收口早期做。

### 5.3 op25/op27/op26 收口战役
- **op25(6 iter,SHIPPED)**:host 筛 30k 行→首版硅上暴露"梯列也是货币"(M=5 税 +7-19%)→ ship w3a 梯+slot_scale=2 门控+fp32 C8;途中 Amdahl 单点判据不建即关 HLS-MC;P0 门抓住 C8 规则外推区→"dispatch 规则止步于数据止步处"。
- **op27(SHIPPED)**:初始 M-probe 假设 **iter0 host 重放即证伪**→硅上四臂分解 A/B 把损失重定位为 K2048 尾列几何→ship K2048 尾梯,worst 1.15→1.44×,real/best 逐 bit 不变。用户约束:禁数据依赖 dispatch、已胜 cell 零回退。
- **op26(iter5 收口,iter6 在途)**:证伪链最密集——ROOTCAUSE_P2.md("更准反而更糟":log 插值准星对着接受带边缘,vendored 表为线性 overshoot 偏置调的);iter5 硅上证伪 secant2 默认开启(**"趟数不是硅上时延的完备代理"**);最终 gm 1.065/fp32 1.127 胜率 90%。COST.md 按 phase 记账(15 GPU-h + ~$108)。

### 5.4 工程纪律 ←→ 逼出它的事故(全部由事故驱动)
| 纪律 | 事故 |
|---|---|
| setsid 独立单条发车,禁 shell 组合技 | op26 iter5d 两次发车事故 |
| pkill 三连 + respawn 复查 | 双 driver ×2 |
| marker 粒度幂等 | 节点回收接力(028→036→038) |
| env 整段照抄 + 头行核验 | 漏 OP22RR_ARMS 4 GPU-h |
| exactness 前置 + sorted value-multiset | atomicAdd 乱序 |
| 锚漂移分位数 QA 门 | 跨节点 µs 不可迁移 |
| 同节点复测才认 regression | op27 假 regression |
| REPORT last-writer 自含契约 | 多 updater 互相覆盖风险 |

**last-writer 链**:每个 updater 全量重导所有已有臂再加自己的臂,docstring 写明后继义务,幂等 mark,机检不变量(script 数/`const D=[`×1/行数),改前 .bak。绝不做增量 patch updater。

### 5.5 多节点接力 prompt 模式(NODEB/NODEC)
自含首条消息固定五段:1 分钟背景 / 预检清单(HEAD、env、GPU 温度黑名单、无共驻、marker 计数)/ 不相交拆分表 / 精确到字节的 setsid 启动命令 / 已知 gotcha。**角色分离**:shard 节点跑完即停,parse/update/commit 由协调会话统一执行——防 last-writer 竞争的组织级手段。

## 6. 人机分工解剖(综合)

### 6.1 人类专家干预的完整分类(按不可替代性排序)
| 干预类型 | 实例 | 自动化前景 |
|---|---|---|
| **T1 出题与硬约束** | op8"95% cell 快 radix 50%、P2 冻结"、op10 2×、op16 +40%、op21 生产约束权重矩阵 | 难:需求来自部署语境。但可模板化(目标+禁区+判据三元组 prompt 表单) |
| **T2 部署包络/口径裁决** | "无 ISL≥1M, N≤256K 主战场"(改变 op24/26 结论口径)、F006 生产路径纪律、op7 基线口径纠偏 | 半自动:包络可作为战役输入声明;口径纠偏可规则化(A/B 必须对真 incumbent) |
| **T3 测量纪律纠错** | cold-L2 两次纠正、cudaEvent→nsys、printf 污染追责 | **可完全自动化**:已全部固化为可执行判据(见 §7) |
| **T4 方向注入/pivot** | op16 两次 pivot、op10 收敛后"再逼一步"、op17 P4-cand-线性洞察、op19 sandwich 命题 | 部分:portfolio 并行试探 + 元分析 op 可替代一部分;深度洞察仍稀缺 |
| **T5 预授权与自治边界** | Era-1"负收益丢弃继续,不要停下等确认";op9 预授权负面结论;PR-1"PAUSED, user decision" | **可完全模板化**:自治合同(self-decide 域 + 必停点清单) |
| **T6 ship 判据与暂停权** | worst/real/best 三轴、零回退要求、禁数据依赖 dispatch | 半自动:判据可写成机器可查的 gate;最终 ship 仍宜人签 |
| **T7 认识论纠错** | Q9d-04c category error、Oracle +51% 误读修正、"趟数≠硅上时延" | 部分:多视角对抗验证可拦截一部分;根本上依赖领域直觉 |

### 6.2 关键观察
1. **人类干预密度随时代递减但杠杆递增**:Era-0 人选算法方向(日级);Era-1 人写规则+预授权(周级);Era-2 人只在 pivot/ship/包络三类节点出手(战役级)。方法论固化(写进 harness 代码与 CLAUDE.md)是干预密度下降的直接原因。
2. **T3(测量纪律)是历史上人类纠错次数最多、但最可自动化的一类**——每条纠错最终都变成了一行可执行判据。这是 omni-kernel 升级的最大低垂果实。
3. **agent 的自主域**在 Era-2 已经覆盖:假设生成、host 原型、kernel 实现、全部测量与证伪、报告维护、接力文档、成本记账。**未覆盖**:出题、包络、pivot 洞察、ship 签字。

## 7. 经验教训总表(供 harness 设计消费)

### 7.1 测量学(全部可执行化)
- M1 计时阶梯:host-replay/clock64(诊断)→ cold-L2 graph event(sweep)→ nsys ×3 中位(ship)→ NCU(物理归因);升级触发 = 小N launch 地板/可疑 win/ship claim。
- M2 cold-L2 canonical(warm 低估 25-35%);warm-L2 A/B 是"省流量类优化"的一票否决器。
- M3 插桩 kernel 永不做 baseline;插桩数据只用于相位占比。
- M4 headline 数字只走生产调用面(F006);A/B 只对真 incumbent。
- M5 锚协议:跨节点/跨会话引用绝对数前必跑锚 cell(期望±3%);回填臂报锚漂移分位数;跨节点异常默认 transfer 噪声,同节点复测才认 regression。
- M6 环境卫生:GPU idle >50C 不计时、共驻检测用输出文件增长、配对同进程 A/B、符号一致性判据区分系统税与 codegen 彩票。
- M7 L2 陷阱一行式判据:`ncu dram__bytes_read` vs 输入字节;输入≪L2 时一切"减 pass"杠杆先验空转。

### 7.2 正确性
- C1 exactness = tie-aware value-multiset(vdiff=0 + uniq==K),前置门不是事后检查。
- C2 双轨:synth gate + real-capture gate + 对抗样本 gate(op21 iter11 0/72 只有对抗样本能抓)。
- C3 禁 torch.randn(bf16 塌缩);数据生成走已验证 synth skill;synth 结论 = 上界估计,真实数据才是判决轴。

### 7.3 搜索策略
- S1 iter0 先测机制上限再写 kernel(crux→host→microbench→kernel 强制阶梯)。
- S2 证伪账本一等公民:提案前必查;条目 = (结论, 条件域, 证据强度);证伪可 dtype/N-conditional。
- S3 组合已验证 primitive > 发明新算法(HLS 六源合成;op21 16 iter 零新算法)。
- S4 infeasible 判定 = 数学地板 + 放开约束对照双重锁死。
- S5 元分析 op(UB/LB 定界 + 顺逆风参数定界)早做、便宜、改变全局口径。
- S6 经验平台期后做一次数学形式化(产生结构不产生数字,重导火力)。
- S7 dispatch 是可 ship 答案的常见形态,但 dispatch 复杂度必须计入目标函数(240-key 表反例)。
- S8 探索→生产化切换信号 = 残余损失全部归因为结构墙。

### 7.4 工程与组织
- E1 纪律写进代码路径(单一 `_time_both`/`sweep_nsys.py`),不写进规程文字。
- E2 每轮迭代 = 假设→实现→gate→硅→verdict(REJECT/FALSIFIED/WASH/SHIP)→commit;基线永远可回归(subclass/flag,不改 vendored)。
- E3 幂等粒度设计期敲定(marker/jsonl/rep 缓存);发车 setsid 单条、停车 pkill 三连、env 整段照抄+头行核验。
- E4 RESUME/HANDOFF 是资产:自含 paste-block 五段结构;shard 跑完即停、协调会话统一收口(防 last-writer 竞争)。
- E5 报告 updater = 自含 last-writer + 显式契约(全量重导所有臂 + 机检不变量 + .bak)。
- E6 安全:nsys 产物泄露 env token(`env -u` + 先 gitignore);成本记账(COST.md per-phase)。
- E7 分析工作随时 checkpoint 落盘 + commit(heredoc→.py、结论→md)。

## 8. 对 /omni-kernel 的升级建议
见同目录 `OMNI_KERNEL_UPGRADE.md`(基于本复盘的逐条差距分析与改造方案)。
可视化轨迹见 `trajectory.png` / `gen_trajectory.py`(autoresearch progress-plot 范式)。
