# op26 iterations

## iter0 — 设计 + 首版实现 (2026-07-09, b200-038)

- PLAN.md 定型:优化点1 = op13 GvrP2CLog 复用(单 CTA)+ cluster 首次移植;
  优化点2 = op#7 exact rank-scatter 条件调度 + **fb_fix**(op21 FIX 移植,
  真正的红牌修复)。
- 实现 = `src/gvr_op26_op.py`:GvrOp26Kernel(⊂ p4 内核)、
  GvrOp26ClusterKernel(⊂ PR#15198 cluster)、两个 wrapper + 调度表。
  全部 subclass override,vendored 零编辑。

## iter1 — fb_fix 首版被 smoke 证伪 → 根因 → 重设计 (同日)

- **smoke 失败**:fp32 K512 N16384 valdiff=2.8。消融矩阵定位:窄窗/rs 全无辜,
  **fb_fix 首版是元凶**(base+fbfix-on 错,off 精确)。
- **host 重放揭示双重根因**:
  1. 该行(preIdx=exact topk,hr=1.0)上 **vendored P2 本身 15 轮不收敛**
     (count 蠕行 399→502<512):P1 种子 `cnt_lo=1.25K@pmin` **从未实测**
     (真值 count(pmin)=512),毒化线性插值 → undershoot creep。这正是
     op22 MECH_FINDINGS "hr .90 毒化 pmean init" 的机理,在 hr=1.0 达到极值。
  2. fb_fix 首版信任了种子 clo 做 falsi、且从不评估 lo 端本身 → 30 轮耗尽
     落 count<K 的 hi 侧 → −1 槽(与红牌同型!修复代码复刻了被修复的 bug,
     教训:**任何来自 P2 的 bracket 计数都可能是未实测种子**)。
- **重设计**(iter1 版,现行):fb 入口两端计数一律置 -1(unknown);
  第一步 = 在 P2 强制 thr 处重数(vendored 兼容,creep 行一步接受);
  未知端优先实测端点本身;两端都实测后才 log-falsi 内点瞄准
  (m*=K·(kCC/K)^0.2);两侧均可扩张(hi 超收缩 ×8 向上,lo 欠收缩 ×8 向下);
  30 轮耗尽 → 实测 undershoot 侧 fail-soft。P3 写入自带 `wc<kCC` clamp,
  全路径无 smem 溢出。
- **smoke 27/27 全绿**(所有调度路径:lin-narrow+rs / log-narrow / stock /
  cluster fp32-log / cluster 16bit-base)。

## iter2 — 门禁 + dry-run(进行中)

- gate_op26.py:A = op22rr bundles 3 场景×3K×3dt×4N×3BS×4 臂;
  B = hr=0/hr=1 对抗(专打 undershoot);C = 16-bit tie 平台。
- dry-run:drive_nsys_op22rr.sh 单批 ×2 臂对,验证 harness→nsys→jsonl。

## iter3 — 机器中断恢复 + C-tie K2048 失败鉴别 (2026-07-09, b200-036)

- 中断前:gate_mc 跑完 287 ok / 4 fail(全部 C:tie|K2048|bf16/fp16,
  row0 1024 out-of-range/-1);gate_1cta 刚进 Suite A 即被打断。
- **鉴别(diag_tie_anchor.py):四臂全败,含两个生产锚**(gvr_cutedsl、
  gvr_multicta_cutedsl PR#15198)——失败形态逐 bit 相同(1024=K/2 个 -1)。
  根因 = 设计包络:16-bit K2048 kC=5120,而 5*K=10240 个 boundary tie 使
  count 从 1024 直接跳 10240,不存在 count∈[kK,kC] 的阈值 → 候选缓冲类
  GVR 变体必然截断(§5 红牌同族,锚继承,非 op26 回归)。fb_fix 的
  fail-soft 语义在此正确生效:落在实测 undershoot 侧的有界截断。
- **处置**:Suite C 平台 cap 到 min(5K, 5120)=kC —— K2048 时平台恰压
  kC 边界,仍是包络内最强 tie 应力;包络外行为以 diag_tie_anchor.py 记档。
  gate_op26.py 新增 OP26_GATE_SUITES 选择器。
- **b200-036 GPU1 确诊散热坏**(idle 82C、slowdown 余量 6C、SW thermal
  slowdown 累计 8.7h)→ 只跑正确性、禁计时;nsys 战役改 GPU0 串行双臂对。

## iter4 — nsys 战役收口 + 报告上线 (2026-07-10, b200-069 收尾)

- **战役节点接力**:op26b 全 81 批 = b200-027(GPU3-5,07-09 06:54-07:23Z,
  MC 臂批速快);op26a 81 批 = 036 遗留 4 批 + 027 跑 60 批(06:53-08:53Z,
  机器级中断三 driver 同刻冻结)+ 069 补 worst 17 批(07-10,8 卡
  SWEEPS×KS×DTYPES 细分片,关键路径 6 批→2-3 批)。
- **069 补齐两次事故与教训**:
  1. 3 卡 dtype 分片发车 4.5min 后为用满 8 卡主动重切(pkill 三连 + 显存
     归零 + 日志冻结复核后重发,损 3 个在飞批 ~4min)。
  2. **重发车漏传 `OP22RR_ARMS`** → 17 批跑成默认 5 臂(无 op26_1cta),
     被 parse ranges 翻倍(672 vs 336)暴露 → 删 17 marker 用正确臂对重跑
     (~9min,页缓存全热)。教训:**单分片重发的 env 必须整段照抄 TAKEOVER
     Step 1(OUT/GPU/切分 + OP22RR_ARMS 一个都不能少),发车后立即 grep
     `arms=` 头行核验**。
- **QA 门全过**:exactness 4 臂各 414/414;锚漂移 op26a-038(direct)
  med 1.0012 p10/p90 0.989/1.032、op26b-038-chained med 1.0024
  0.989/1.022;REPORT.html script=2、`const D=[`×1;3 CSV 落盘
  (op22rr_op26_raw.csv 5432 行)。
- **headline(nsys cold 同卡配对 gm)**:
  - `op26_1cta` vs gvr_cutedsl(2714 格):总体 **1.032**;**fp32 1.100
    (胜率 80%)** = real **1.129** / worst 1.102 / best 1.069;16-bit
    wash(bf16 0.994 / fp16 1.005)。最佳格 K512/K1024 fp32 N=8-16K
    达 1.86-2.12×。
  - `op26_mc` vs gvr_multicta_cutedsl(2718 格):总体 **0.985**
    (fp32 0.995 / bf16 0.976 / fp16 0.985,K2048 0.953)——wash 偏负,
    PLAN §3 预言落在证伪侧:**cluster 的 P2 eval 已被切片并行摊薄,
    log 插值移植不 ship**。
  - **红旗(两臂共有)**:K1024 fp32 N=131072 BS≥256 掉到 0.56-0.60×
    (op13 ship 表在该格的 logn 窗口在 op22rr 数据轴上是倒退)——
    后续应把 K1024@131K 从 dispatch 表摘掉再收窄结论。
- **花费记录**(本战役全周期):GPU ≈ **15 GPU-h**
  (036 门禁+4 批 ~1;027 op26a 3卡×2h + op26b 3卡×0.5h ≈ 7.5;
  069 三轮 r0 0.4 + r1 错臂 4.0 + r2 1.2 ≈ 5.6,其中错臂返工浪费 ~4 GPU-h);
  069 收尾 session 墙钟 ~1.3h(01:47-03:05Z)。
- **Claude token 花费**(从 session transcript usage 实测,Fable 5 定价
  $10/$50 in/out、$12.5 cache-write(全部 5-min TTL)、$1 cache-read /MTok):
  - 确定归属 op26 的 session:设计+实现+门禁(f72976b5,038/036)$23 +
    027 战役三 session(327729b8/10234912/840e34e8)$29 + 069 收尾
    (79dce2d2,含 8 卡重切与错臂返工)$56 ≈ **$108**。
  - 口径说明:07-09 凌晨另有 6 个 session 共 ~$317(含 op22rr/op25 等
    并行战役,无法按臂拆分)+ 07-10 并行 session da924869 $70,归属不明,
    未计入。成本大头是 cache-read(op26 相关 ~50M tok cache-read/session
    级),output 仅 ~0.5M tok。
