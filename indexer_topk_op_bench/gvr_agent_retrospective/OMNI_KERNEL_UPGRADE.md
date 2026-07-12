# /omni-kernel 升级提案 — 从 GVR 成功案例到 autonomous kernel optimization

> 依据:同目录 `RETROSPECTIVE.md`(Era-0/1/2 全史考古)。
> 现状:`omni-distill/omni-kernel/SKILL.md` 是 AKO4ALL/CudaCoder/PerfBot/autoresearch 的通用蒸馏
> (多假设生成 + SOL% 目标 + 锦标赛 + per-iter commit),`LEARNINGS.md` 基本为空。
> 结论先行:**omni-kernel 缺的不是循环骨架,而是让循环"判决可信、死路不复活、跨会话不失忆、
> 人只在高杠杆点出手"的四层机制。** GVR 战役已把这四层全部打磨成可执行形态,照抄即可。

## 0. 差距诊断表(现有 SKILL.md ↔ GVR 实战)

| 维度 | omni-kernel 现状 | GVR 实战已验证的形态 | 差距等级 |
|---|---|---|---|
| 目标函数 | 固定 SOL% ≥80% | **对手相对判决**(vs incumbent/rival, nsys gm + worst/real/best 三轴)+ 部署包络 + 零回归约束 + dispatch 复杂度入目标 | 🔴 根本性 |
| 计时 | benchmark.py(event 紧循环)+ ncu | 四级阶梯:host-replay/clock64 → cold-L2 graph event → **nsys ×3 中位(唯一 ship 仲裁)** → NCU 归因;warm-L2 一票否决器;锚协议 | 🔴 根本性 |
| 正确性 | verify.py atol/rtol | **tie-aware value-multiset 精确门** + synth/real/对抗样本三轨 + 禁 randn | 🔴 根本性 |
| 失败记忆 | LEARNINGS.md "Ineffective Directions" 自由文本 | **证伪账本 = (结论, 条件域, 证据强度) 三元组 + 提案前强制查阅关卡** + 根因分类(结构墙/测量假象/复杂度反噬) | 🟠 结构缺失 |
| 假设成本控制 | 直接写 3 个实现 | **iter0 先测机制上限**:crux→host-replay→microbench→kernel 强制阶梯,"20 分钟 host replay 杀掉数周错形 kernel" | 🟠 结构缺失 |
| 搜索空间 | 3-strategy(语言维度) | 语言维度只解决一次(cuteDSL 定型后不再切换);高产维度 = **组合已验证 primitive**(HLS=六源合成)+ portfolio 参数投机 | 🟠 结构缺失 |
| 收敛判据 | SOL%≥目标 / 3 次无改进 | **残余损失全部归因为结构墙** → 切生产化;infeasible = 数学地板+放开约束对照双锁 | 🟠 结构缺失 |
| 跨会话 | git commit 历史 | RESUME/HANDOFF paste-block 五段结构、marker 级幂等、锚点重锚、多节点接力+协调会话收口 | 🟠 结构缺失 |
| 人类接口 | 无(NEVER STOP) | **自治合同**:预授权 self-decide 域 + 必停点清单(ship 签字/pivot/包络)+ 预授权负面结论 | 🟡 待补 |
| 长期擂台 | 无 | 多臂 arena + anchor-transfer + last-writer 自含契约 | 🟡 待补(进阶) |
| 运维卫生 | 无 | setsid/pkill 三连/env 核验/温度黑名单/token 卫生(nsys sqlite 泄露)/COST 记账 | 🟡 待补 |

## 1. 目标函数改造(最高优先)

**废除"SOL% 达标"作为主目标**。GVR 案例证明它对选择/搜索类 kernel 是错误坐标系:BS=1 单 CTA 占用率 24.3% 是**结构性**(grid-limited),SOL% 永远难看但 kernel 已是 Pareto 最优;反之 SOL% 高也可能输给对手(radix 1-pass vs GVR 2.5-pass 是算法层差距)。SOL% 降级为诊断信号。

新目标函数三件套(战役开题时由人填,agent 不得自行放宽):
```yaml
objective:
  incumbent: <当前生产默认实现>          # A/B 永远对着它(op12 iter6 教训)
  rivals: [<其他候选>]                  # 多臂对照
  envelope: {N: ..., K: ..., dtype: ..., BS: ...}   # 部署包络(op24 教训:包络外 cell 只是应力探针)
  verdict_axes: [worst, real, best]     # ship 三轴
  ship_rule: "worst 提升 AND real/best 零回退 AND exactness 全绿 AND dispatch 规则 ≤3 条"
  hard_constraints: [<冻结的算法相位>, <graph 兼容>, <fail-soft>]
```

## 2. 测量学改造(P0,全部可执行化)

替换 Phase 5 为**四级测量阶梯**,并把纪律写进唯一代码路径(scripts/ 单入口,禁止每个 iter 自写计时):

1. `scripts/host_replay.py` / clock64 相位分数 — 只用于机制 ceiling 与参数筛选,**禁止**作为 op-vs-op 结论(F006);插桩 kernel 永不做 baseline(printf 事故)。
2. `scripts/bench_cold.py` — cold-L2(>L2 容量 buffer `.uniform_()` 逐 launch 前)+ CUDA-graph 中位;warm-L2 单列报告。
3. `scripts/nsys_verdict.py` — ship 唯一仲裁:eager+NVTX、×3 独立批中位、**锚 cell 校验**(引用绝对数前重测锚点,偏 >3% 全网格重锚)。
4. `scripts/ncu_attrib.sh` — 仅物理归因(占用率是否结构性、`dram__bytes_read` L2-陷阱一行检验)。

内置**三个一票否决器**(在提出优化方向时先跑,不是实现后):
- L2-陷阱检验:输入字节 ≪ L2 ⇒ 一切"减 pass/省流量"方向先验空转(op14/15/16 三连杀的教训)。
- warm-L2 A/B:若 warm 下也不赢,方向直接毙(op15)。
- 数学地板:bytes/BW 下界 + 放开约束对照,双锁才允许写 "INFEASIBLE"(op10)。

升级触发规则写死:小 N launch 地板 / event 出现可疑 win / 任何 ship claim ⇒ 必升 nsys。

## 3. 正确性门改造(P0)

- 判据 = **tie-aware value-multiset**:sorted-value 等价 + uniq==K(或算子等价物),不是 atol/rtol(选择类 kernel)也不是 index-equality(平票非确定)。
- 三轨数据:①已验证 synth 生成器(声明"synth=上界估计");②real capture(synth exact ≠ real exact,op21 红牌);③**对抗样本**(近平票簇、边界 padding、变长——iter11 修复前 0/72 只有对抗轨能抓)。
- 禁 `torch.randn`(bf16 塌缩 ~256 值)。get_inputs 必须走数据生成器并记录 seed 策略(cell_seed=f(K,N),防全 cell 同分布)。

## 4. 证伪账本改造(P0)

`LEARNINGS.md` 拆成两个一等公民文件:
- `FALSIFIED.md`:条目 = `(假设, 结论, 条件域[K/N/dtype/BS/arch], 证据强度[host/event/nsys/NCU], 根因类型[结构墙|测量假象|复杂度反噬], 复活条件)`。证伪可以是条件性的(C8 在 fp32 是噪声、16-bit 是 1.14× 赢)。
- `WALLS.md`:结构墙清单(占用墙/pass 定数/相位链延迟墙…),每条附一行式检验命令。

**流程关卡**:每轮假设生成后、实现前,强制 grep 两文件;命中即要求引用"复活条件"或放弃。这是 GVR 六个月里 ROI 最高的单一机制(Era-1 12+2 条 → op-bench 28+ 条,多次实际拦截重复开采)。

## 5. 迭代协议改造(P1)

```
iter N = 假设(引用账本) → iter0-探针(crux/host, 给出 ceiling 与 GO/NO-GO)
       → 实现(gated flag / subclass, 基线字节不变)
       → exactness 三轨门 → cold pilot → [nsys verdict if ship-candidate]
       → 判决 ∈ {SHIP, FALSIFIED(+条件域), WASH, PIVOT}
       → ITERATIONS.md 追加 + FALSIFIED/WALLS 回写 + git commit -s
```
- 保留现有 per-iter commit 与 HEAD-at-best,加两条:**基线不可变**(不改 vendored 文件)与**判决词汇表**。
- 平台期动作升级:现有"换语言"降级为末位;新增 ①元分析 op(UB/LB 定界 + 参数顺逆风定界,便宜且改变全局口径)、②**数学形式化**(把经验结果统一成模型,识别剩余自由度——HLS 从 1.25× 均值到 2× 尾部的转折点)、③primitive 重组搜索(跨 iter/跨战役组合已验证组件)。

## 6. 自治合同与人类接口(P1)

把 "NEVER STOP" 改成**合同式自治**(Era-1 原文范式:"负收益则丢弃,继续下一个…自行给出最优解决方案后执行,不要停下来等待人为确认"):
- `AUTONOMY.md`(战役开题时人签):self-decide 域(参数、实验设计、负收益丢弃、checkpoint)/ 必停点(ship 进生产、改基线语义、超成本预算、包络修改)/ **预授权负面结论**("如 X 仍最优就如实负面报告"——op9 教训,防 agent 硬凑 win)。
- 人类干预模板化(对应 RETROSPECTIVE §6.1 T1-T7):开题表单(目标三件套)、pivot 请求(agent 给出证据+选项,人只选方向)、ship 签字。T3(测量纪律)不再需要人——已全部判据化。

## 7. 跨会话与运维(P1/P2)

- `RESUME_PROMPT.md` 成为强制交付物,五段结构:背景 1 分钟 / 预检清单(HEAD、env、GPU 温度黑名单、无共驻、进度 marker 计数)/ 拆分表 / 精确 setsid 启动命令 / gotcha 清单。
- 幂等粒度设计期敲定(marker/jsonl/缓存 parse);发车 setsid 单条、停车 pkill 三连 + respawn 复查、env 整段照抄 + 头行核验。
- 安全与成本:nsys 前 `env -u GITHUB_TOKEN -u HF_TOKEN`;`*.sqlite/*.nsys-rep` 先 gitignore;`COST.md` per-phase 记 GPU-h 与 token 花费(op26 实测一场战役 ≈15 GPU-h + $108,给预算控制提供锚)。
- (进阶)长期多臂 arena:锚臂协议 + anchor-transfer + 锚漂移分位数 QA 门 + last-writer 自含 updater 契约——当战役产出 >3 个候选实现时启用。

## 8. SKILL.md 落地改造清单

| 动作 | 位置 | 内容 |
|---|---|---|
| 改写 | Phase 0.2 | Resolved Plan → 目标三件套 yaml + AUTONOMY 合同 |
| 改写 | Phase 1.4 | Roofline 保留,SOL% 降级为诊断;新增 L2-陷阱/数学地板先验检查 |
| 新增 | Phase 2.5 | 证伪账本查阅关卡(FALSIFIED/WALLS grep) |
| 改写 | Phase 3 | 3-strategy 语言投机 → "iter0 探针阶梯 + primitive 重组"双轨;语言选择只做一次 |
| 改写 | Phase 4 | atol/rtol → tie-aware 三轨 exactness 门 |
| 改写 | Phase 5 | 四级测量阶梯 + 三个一票否决器 + 锚协议 |
| 改写 | Stall Handling | 换语言降末位;新增元分析 op / 数学形式化 / 重组搜索 |
| 改写 | Anti-Patterns | 并入测量伪影名录(clock64 膨胀、event 假 win、randn、插桩 baseline、双 driver…) |
| 新增 | 文件 | `FALSIFIED.md`、`WALLS.md`、`AUTONOMY.md`、`RESUME_PROMPT.md` 模板、`COST.md` 模板 |
| 新增 | scripts/ | host_replay / bench_cold / nsys_verdict(含锚校验)/ ncu_attrib 单入口四件套 |
| 回填 | LEARNINGS.md | 以 GVR 战役的 M1-M7/C1-C3/S1-S8/E1-E7(RETROSPECTIVE §7)作为种子知识 |

## 9. 降低人类参与度的现实路径(结论)

GVR 案例中人类干预按可自动化性排序(RETROSPECTIVE §6.1):
- **立即可移除**(本提案 P0):测量纪律纠错(T3)、口径纠偏(T2 的规则部分)——历史上人被迫纠错最多的一类,全部已判据化。
- **可大幅稀释**(P1):方向注入(T4)的一部分被"元分析 op + 探针阶梯 + 重组搜索"替代;自治边界(T5)模板化后 agent 可长跑数天不请示。
- **保留给人**(设计为显式接口):出题与硬约束(T1)、部署包络(T2 裁决部分)、ship 签字(T6)、深度洞察型 pivot(T4 残余,如"P4 是 cand-线性"“sandwich 命题")。
预期效果:人类触点从 GVR 实战的 ~每战役 3-5 次(pivot/纠偏/签字)收敛到 ~1-2 次(开题+签字),其余由 harness 承接。
