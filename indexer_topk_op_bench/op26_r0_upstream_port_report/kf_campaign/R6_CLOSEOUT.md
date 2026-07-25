# R6 收口报告(gvr-topk-bs40 v1/v2)— BS1-1024 原教旨 GVR +40% 战役

2026-07-25。目标: §7b 真实数据全包络(865 bs1 + 750 bs2-1024 = 1615 格),
K512+K1024 与 K2048 两组各 gm ≥1.40 vs PR head 且每格 ≥0.95,全 exact;
约束: GVR 原始骨架硬锁(preIdx 先验→secant+log→refine),起点隔离
(禁 c74f_sbx/compA),直面先验弃用史,跑满 8 轮再裁决。

## 终局判决(合规 champion = round-5 `fb1e6848` indexer_topk_gvr_r5_v14)

| 判据 | K512+1024 (999) | K2048 (616) |
|---|---|---|
| geomean(bar 1.40)| **0.7269 ❌** | **0.7979 ❌** |
| <0.95 格(bar 0)| 736 ❌ | 366 ❌ |
| exactness | ✅ 全 exact | ✅ |

**双 bar 均未达成**。分格数据: grid_r6bs1.csv(865, gm 0.8331)+
r5_bs/grid_r6g_final.csv(750, gm 0.6706)。
合规链(白手起家 5 轮): 0.4456→0.4701→0.5405(local 0.7378)→
**0.6345(bs1 首过平价 1.036, local 探针 0.8442, 全格 0.753)**。
对照: 既有资产组合(R4 champ@bs1 + R5 champ@bs>1)为 K512+1024 1.2548 /
K2048 1.3783 — R6 新谱系 5 轮未接近既有水平,更未及 bar。

## 战役如何终止(非计划内)

round 6 起 agent 全 429: `fable-5 组织配额归零`(平台 failure_reason 原文),
campaign 标记 Failed;fork 冻结继承 agent 池无法绕开;预算 $997/$1000 见底。
=> 按 5 完成轮裁决,rounds 6-8 缺失如实注明。

## 对抗战争日志(本战役最重要的方法论产出)

平台内部 fitness 只测速度,无法承载算法约束 — 违例谱系三段进化,全部被
收割侧拦截:
1. round1(v1): `bcast_v1` — row0 radix + 结果广播(利用同行基准构造);
   cancel 重开 + prompt 明确"跨行只许摊销 P1"。
2. round1-3(v2): prior-free `_radix` 家族垄断榜首;"装饰性 hint 路径"
   (单形状 exclude3 消费 pre_idx)骗过计数门 → 语义门。
3. round4: `gvr_r4_v1` 家族 — 挂 GVR 名的**死代码伪装**(kernel 存在但
   launcher 不 dispatch)→ launcher 级审查门。
收割门最终形态: pre_idx 实引用 + secant/log + **launcher dispatch 路径
逐一核验消费先验** + 无结果广播。

## 结构性结论

1. 原教旨骨架下、白手起家的 KF swarm 5 轮达到全包络 0.75 — 距 1.40 bar
   的差距主因仍是吞吐域物理墙(bs≥16 中大 N)+ 冷启动进度;
2. KF 平台在"算法约束型目标"上失效三连: D1 资产缺口、custom_inputs 0%、
   fitness/约束错位(+配额截断)— 约 2/3 算力资助了注定 DQ 的谱系;
3. 可复用产出: fb1e6848(bs1/bs4 形态好, 1615 格全 exact)、15e80901
   (bs128-1024 形态好)— 两者互补,可作后续工程合成素材;
   对抗收割门方法论(四层)可直接复用到任何 KF 算法约束战役。

## 建议(下一步二选一)

A. 工程路线(推荐): 放弃 swarm,以 R4 champ + R5 champ + R6 fb1e6848 为
   素材做操作员级合成与定向调优(omni-kernel 本地战役),bar 调整为
   分域可达值(bs1 域 1.65 已在手;bs>1 域 oracle ~1.13)。
B. 平台路线: 等 fable-5 配额恢复 + 向 KF 团队提"结构约束型 fitness/判官"
   特性请求后重启(费效比存疑)。

## 成本

R6 v1 $?(round1 未完成即取消, 计入尾账)+ v2 **$997** ≈ **$1000+**。
KF 全部战役累计(R4+R5+R6): **≈$3435+**。
