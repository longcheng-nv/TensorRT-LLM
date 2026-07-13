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

## iter5 — V3 上硅 → 硅上消融证伪 secant2 → iter5b 剪枝收口 (2026-07-10, umbriel-b200-037)

- **V3 上硅** @0087ff74f7(前一 session):log-secant 过最近两实测点
  (p2_secant2,thread-uniform 寄存器跨迭代)+ 几何中心瞄准 √(kK·kCC);
  16-bit K1024/K2048 全 N 开 log+secant。门禁 582/582、smoke 27/27。
- **单格 nsys A/B(real/bs,GPU0/1,arms 核验过)**:
  - K1024 fp32 @131K BS≥256:0.56-0.62 → **0.83**(未达 0.85 门槛);
  - K2048 fp16 @65536:iter4 0.877 → **0.71-0.74(恶化)**;32K 低 BS 同恶化;
  - K1024 fp32 @8192 win 保持(1.68-1.86);
  - **意外**:K2048 fp16 @262144 iter4 0.996 → **1.11-1.21 翻盘**,且
    V3 效果随 N 单调改善(65K 0.71 → 131K 0.97 → 262K 1.15)。
- **硅上四变体消融(diag_iter5_silicon.py,events+L2-flush)**:
  | 格 | anchor-lin | iter4 | V1-center | V3-secant |
  |---|---|---|---|---|
  | K2048 fp16 65536 BS=1 | 24.58 | 28.67 | 28.67 | 32.90 |
  | K2048 fp16 65536 BS=256 | 40.96 | 47.10 | 47.10 | 49.15 |
  | K1024 fp32 131072 BS=1 | 28.67 | 38.98 | **32.77** | 32.77 |
  | K1024 fp32 131072 BS=256 | 94.27 | 155.65 | **112.64** | 114.69 |
  判决:**修复收益 100% 来自中心瞄准(V1);secant2 硅上默认证伪**
  (host 重放趟数节省真实,但硅上循环开销 > 节省;K2048 fp16 65536 +15%)。
  与重放的分歧教训:**趟数不是硅上时延的完备代理**。
- **iter5b 剪枝**(全部有 iter4-real + iter5-A/B 双证据):secant2 仅存
  K2048 16-bit n≥262144;中心瞄准保留全部 log 格;摘除 K1024@131K
  (全 dtype,0.69-0.91 恒输)、K1024 16-bit 收窄至 [8K,64K]、
  K2048 16-bit 16K-131K(0.84-0.94 损失带)。
- 消融口径(secant2/kFT_center 分离):kFT_center 单独 = V1 列(即 ship 组
  分);secant2 单独增量 = V3−V1 列(除 K2048 16-bit 大 N 外恒 ≤0)。
- iter5b 验证:gate(双臂并行 GPU2/3)+ 6 卡 real/bs 全 (K,dtype) A/B
  (K1024/K2048 × fp32/bf16/fp16,GPU0/1/4-7)—— 结果见下方追记。

### iter5b 判读(6 卡 real/bs 全 (K,dtype) A/B,同节点 037)+ iter5c 微修

- gate 双臂 291/291;6 批 arms 核验全过。
- **深坑全平**:K1024@131K 全 dtype 0.69-0.72 → 1.01-1.04(剪回 stock 后
  rank-scatter 余量还小赢);K2048 16-bit 16K-131K 损失带 0.84-0.94 →
  0.99-1.03;**K2048@262144 secant2 在 bf16 上外推确认**(bf16 1.143 /
  fp16 1.147)。16-bit 整体转正:bf16 0.952→1.040、fp16 0.956→1.033;
  fp32 1.121→1.142;六区 overall gm 1.071。
- **中心瞄准的两处系统性回退**(iter5b vs iter4,>>噪声):
  1. K2048 fp32:edge(2048)全 N ≥ center(2896)——32K 1.033 vs 0.934、
     65K 1.095 vs 0.950、≥131K 平手 → iter5c 整段回 edge;
  2. K1024 16-bit @8192:edge 1.26/1.46 vs center 1.09/1.12 → 回 edge
     (stock);16384-65536 center 更好(bf16 16384 0.92→0.98、65536
     1.15→1.23)保留。
  机理注:最优瞄准点依 (K,dtype,N) 翻转 = log-CCDF 体/尾插值偏置方向翻转;
  edge 瞄准是在借线性尾 overshoot 偏置把落点推回带中。
- iter5c = 仅上述两处 kFT 回退;re-verify 3 段(K2048 fp32、K1024 bf16/fp16,
  OUT=results_b200_op26_iter5c)+ 1cta gate 重跑。

### iter5d — 全网格首轮判读 + 两处尾部剪枝 (同日, 037)

- 全网格 81 批(8 卡 dtype×K 分片,~70min)首轮:**overall gm 1.0602 /
  胜率 68%**(iter4 战役 1.032);fp32 1.1272 胜率 90%(1.100/80%);
  16-bit 转正 bf16 1.0237 / fp16 1.0329(0.994/1.005);
  real 1.0721 / best 1.0573 / worst 1.0515。
- 两处残余损失 → iter5d 剪回 stock:
  1. **K2048 16-bit @524288 = 0.878/0.880**:262144 的 secant2 win 不向上
     外推(iter4 stock-aim log 在 512K 是 ~0.96-1.01)→ secant2 门改
     n==262144 精确匹配;
  2. (1024,bf16,16384) 0.928 混合场景持续损失(两种瞄准都输)→ 16-bit
     center 带收窄为 [32768,65536]。
- 重跑受影响 24 批(K2048 16-bit seqlen/bs_hugeN ×3 scen、K1024 16-bit
  bs/seqlen ×3 scen)。发车两次事故均为 shell `&&…&`/循环后台化优先级坑:
  一次 GPU1-7 OUT 为空(driver failsafe 拒跑),一次 for-loop 变串行 +
  前条命令超时连坐杀了 GPU0 driver(seqlen K2048 bf16 real/best 已完,
  worst 由 marker 幂等补发)。教训:**多卡发车 = 每分片独立一条
  `setsid bash -c "cd …; env … ./drive.sh" > log &`,不用变量/循环/前置 cd**。

### iter5 收口 — 049 接管补尾批 + 最终判决 (2026-07-12, umbriel-b200-049)

- 037 被回收时 gpu6 分片(worst/bs K1024 fp16)死在 52/84,marker 80/81;
  049 接管后 marker 幂等补发单批(GPU0,13min,arms 头行核验过)+
  iter5d 调度码 gate 重跑 **291/291**(此前 gate 只盖到 iter5c)。
- **最终全网格(2718 配对格,nsys cold 同卡配对 gm)**:
  overall **1.0648 / 胜率 69%**(iter4 战役 1.032/—);
  fp32 **1.1272 / 90%**、bf16 1.0334 / 67%、fp16 1.0365 / 50%;
  real 1.0759 / best 1.0674 / worst 1.0513;
  K512 1.0465 / K1024 1.0941 / K2048 1.0530。
- **iter5d 剪枝区全部复位**:K2048 16-bit @524288 0.878/0.880 → 1.006;
  K1024 bf16 @16384 0.928 → 1.011;secant2 存留格 @262144 gm 1.047;
  K1024 fp32 @131K(iter5b 剪回 stock)1.038。
- **仅存损失簇 = K2048 fp32 @N=1048576 gm 0.942**(21 格)——部署包络外
  (应力探针,主战场 N≤256K),记档不再追。
- 报告:`update_report_op26_iter5.py`(op27 last-writer 之上的薄 wrapper,
  已扩展为同时把 `op22rr_op26_raw.csv` 重指 iter5 根)全绿:
  exactness 全臂 414/414;锚漂移 op26a(iter5 根)med **1.0006**
  p10/p90 0.987/1.015;REPORT.html D=30528、script=2、`const D=[`×1、
  iter5 双语注记入卡(幂等 mark)。判读工具固化 = `analyze_iter5_grid.py`。

## iter6 — R0 h-空间梯准入(HLS w3a → 经典 GVR 统一移植)(2026-07-12, umbriel-b200-049)

- **目标重申(用户)**:GVR cuteDSL base + PR#15198 MC 对 radix_cutedsl 与
  SGLang 在 N≥8K 全网格胜 10%+;禁 dispatch 到 radix;GVR 阈值法打到 SOL。
  Scoreboard(iter5 态):op26_1cta vs radix gm **0.919**(win10 1020/2718)、
  op26_mc **1.005**;vs sglang 1cta 1.237 但 fp32 8K-262K 成片 0.75-1.03。
  塌方区 = N≥262K 低 BS(0.27-0.45,单 CTA 带宽墙)+ K2048 16-bit 262K
  0.45 + mc@8192 0.52-0.64。计划 = `PLAN_ITER6.md`。
- **host 筛选**(`screen_r0_qfracs.py`,216 格 × 梯候选,经典窗 [K,kC]):
  基线 pmean seed 首趟准入 real 3/72、best 0/72、worst 72/72(worst 的
  seed 恰落 K 名值 → ev1,与报告注记一致);**uh4 (0.90,0.65,0.40,0.15)
  M=4 三场景 216/216 静态准入**(w3a 0.894——ms 窄窗刀锋在宽窗下非最优,
  均匀覆盖胜出);m3_a (0.90,0.55,0.20) M=3 97.7%(miss 全为实测括号型)。
  接受 cand med 1229 / p90 2597。
- **kC / f_target 再分析**(用户追加):R0 路径下 f_target 退役(最紧可接受
  rung 自瞄准);kC 敏感度 = K512@1536 69/72、K1024@3072 72/72、
  K2048 需 5120 → kC-diet(smem 省 28KB→occupancy)列为二阶消融,
  v0 保持 stock kC 使 A/B 纯算法差。
- **v0 实现** `src/gvr_op26_r0_op.py`(GvrOp26R0Kernel ⊂ GvrOp26Kernel):
  P1b = 256-bin smem hist over prev-topK 值(复用 smem_hist 前 256 bin,
  K 次 gather L2 热)→ tid0 一次 256-bin 降序游走取 M rung;R0 =
  op18 block_count_ge_multi **原样拷入**(教训:p4 谱系的 GvrTopKKernel
  是 vendored 全拷贝而非子类,与 op18 vendored 谱系菱形混入不可行,
  MT.super 直落 vendored——首次 smoke 证伪);最紧可接受 rung 缓存列
  直接喂 P3 零重扫;miss → 实测括号进 fb_fix。P4 = rank-scatter 照 op26 门。
- **smoke 108/108 exact**(3 dtype × 6 (K,N) × hr∈{1,~0.5,0} × BS16)。
- 门禁 + 8 卡 nsys A/B(gvr_cutedsl + op26_r0 + radix_cutedsl 同批三臂,
  代表格 = 缺口区 + win 保持区)进行中 —— 结果见下方追记。

### iter6 v0 首硅判决 (8 批 nsys, 049) — 准入机理成立,但 tid0 串行游走是致命税

- 门禁 291/291 绿(gate_iter6_r0.log)。8 批 A/B(real/best/worst × K × dt,
  三/四臂同批)判决呈**清晰的二分**:
  - **带宽主导域大赢**(K1024 fp32 real):65536 BS2048 **1.649×** vs anchor、
    131072 BS256/2048 1.01/1.10、262144 BS256/2048 **1.15/1.23**;
    vs radix @256K 1.045-1.08 —— 趟数账(R0 一趟 ×1.25-1.4 替代 2-2.4 趟)
    在聚合带宽域如模型成立。
  - **latency 域塌方**(BS=1-16 全 N、小 N 全 BS):r0/anchor 0.37-0.72,
    **绝对差 +10-19µs 且 BS 平坦** → per-CTA 固定串行链。
- **根因 = P1b 的 tid0 256-bin 降序游走**:深 rung q=0.90 需走完 ~全部 bin,
  256 次 smem 依赖链 ≈ 10-15µs(op20 相位链墙的重演)。v0.1 修复 =
  warp-0 并行提取(lane 分段 8 bin + 5 步 shfl_up 前缀 + 段内 crossing
  定位,串行 256→8 迭代)。
- 结构判断:即使修复,BS=1 中 N 对 radix 的差距属单 CTA 占用墙
  (op8 ROOT,anchor 第二趟 L2 热使趟数账在 BS=1 打折——op18 L2 trap),
  1cta 臂的可赢集 = 高 BS 全 N + 大 N 高 BS;低 BS 大 N 需 iter6b
  (R0 → PR#15198 cluster)接棒。

### iter6 v0.1/v0.2 硅上判决 (8 批 ×2 轮, 049) — real 轴对 radix 转正

- **v0.1(warp 并行提取)**:latency 域塌方修复大半;real K512 fp32
  vs radix 1.197(64-128K 1.68)、best K1024 1.096;但 **worst 轴自家
  回归 0.901**——worst 的 pmean seed 本来首趟即中(基线筛 72/72),
  M=4 趟税(×1.25-1.4)纯亏。
- **v0.2 = M2D (0.85,0.35) M=2 梯(税 ×1.02)+ R1 inline 双实测点
  log-falsi 一发**(筛选:96.8% 准入,miss 全括号型;期望趟数
  real ≈1.1 / worst ≈1.02)。门禁 291/291。8 批判决(ALL gm):
  | 批 | vs anchor | vs radix | vs sglang | vs iter5-1cta |
  |---|---|---|---|---|
  | real K512 fp32 | **1.342** | **1.295** | **1.155** | 1.157 |
  | real K1024 fp32 | **1.214** | **1.087** | **1.073** | 1.022 |
  | real K1024 bf16 | 1.153 | 1.041 | — | 1.078 |
  | real K2048 fp32 | 1.118 | 1.088 | — | 0.962 |
  | real K2048 fp16 | 1.097 | 0.907 | — | 1.057 |
  | best K1024 fp32 | 1.192 | 1.096 | 0.997 | 1.004 |
  | worst K1024 fp32 | 0.938 | **1.169** | 1.049 | 0.792 |
  64K-256K 带对两个对手全面 1.1-1.8×(real K512 64-128K vs radix 1.843)。
- **残余缺口三类**:(a) worst 对自家 anchor 0.938 = P1b 固定开销 ~6%
  (P1b 并入 P1 gather 是消融项;pmean 混合梯已筛证伪——worst 上
  pmean rung cand 反而更大 1730 vs 1353);(b) 8K-32K 相位链带
  (radix 0.76-1.11);(c) **低 BS ≥131K + 16-bit 大 N = 行内多 CTA
  结构墙**(radix 4-32 SM/行;K2048 fp16 262K 低 BS ~0.5)→ iter6b
  mc 港主战。
- 全网格 81 批(4 臂,OUT=results_b200_op26_iter6grid,8 卡 dtype×K
  分片)在飞 —— 完整 scoreboard 见追记。

### iter6b mc 臂首硅 (074 接管) — smoke hr0 修正 + 包络判定

- 049 回收后 074 接管;grid 61/81 起步幂等续跑(8 分片原样)。
- **op26_r0mc 首跑 smoke 在 hr0 失败(K512 N32768 fp32 cs=1,
  valdiff 8.56e-01, uniq 512/512)——判定为 out-of-envelope,非移植 bug**:
  smoke 的 hr0 抄了 1cta R0 smoke 的 bottom-K preIdx(topk(-row)),
  P1 括号 [pmin,pmax] 整体低于真第 K 值,任何括号内 bisect 都不可达;
  1cta 靠 fb_fix 修括号才过。三臂复现(debug_r0mc_hr0.py):
  **vendored cluster 锚同 case 同样截断,valdiff 完全一致 8.56e-01**;
  op26_r0 (fb_fix) 0.0。=> mc 港与锚异常包络一致,符合设计
  (RESUME_ITER6B_PROMPT §1.3: fb_fix 不移植,op26_mc 先例 smoke 只测 hr1)。
- 修正:smoke hr0 改 gate Suite B 同款随机不相交 preIdx(包络内,
  必须精确);bottom-K 记录为已知包络边界(锚平价)。
- **第二硅上 bug 推翻"fb_fix 不移植"设计决定**:K2048(cr=1,+1 offset
  → gathered 近随机)N=262144 全 hr 档重复索引失败,**cs=1 也失败**
  (iter5 op26_mc / 锚 / 1cta 同数据全过 → iter6b 新 glue)。printf 实锤
  (debug_r0mc_k2048_dbg.py):梯子两级全 overshoot(89k/222k ≫ kC)→
  chi=-1 跳过 R1 → status2 交给 vendored retry-shrink 的括号宽达 6 个
  count 十倍程;**vendored shrink 只修 overshoot(`while count>kCC`),
  一步 bisect 跨过 [kK,kCC] 窗即 undershoot 退出**(cand 1654<2048,
  P4 Branch C 补 -1)。N=131072 过 = mid 恰落窗内的运气。锚不踩坑 =
  它的 P2 secant 交接前已有 ~10 次实测迭代;1cta 不踩 = fb_fix。
  **根修 = fb_fix 整体移植进 mc phase3 override(cluster 聚合计数,
  决策输入全 CTA 一致故轨迹一致;fb_alpha=0.2 同参)**,仅 status!=1
  跑,热路径零税。修后 K2048 复现 8/8 过(cs=1/2/4 × hr1/hr~/hr0),
  且 bottom-K hr0 也进包络(fb_fix expand-upward 守卫)→ smoke 恢复
  bottom-K(hr0bk)+ 保留随机不相交 hr0,包络 = 1cta 平价 > 锚。

### iter6 全网格 81 批终判 (074, 2718 cells) — M3 部分达标

- **核心域 8K-262K(部署主战场)**:op26_r0 vs radix gm real 1.068 /
  best 1.065 / worst 1.096(win10 ≈40-41%);vs anchor real 1.200 /
  best 1.180 / **worst 0.928(P1b 固定税洞,比 8 批时的 0.938 实测更宽:
  worst 各 K/dtype 带 0.87-0.94,14 个 <0.9 (scen,K,dtype,band) 洞里
  13 个在 worst 轴)**。
- **全网格 gm vs radix 0.936 < 1.0 ✗** —— 被 hugeN 512K-1M 结构墙拖累
  (该带 gm 0.521,radix 行内 4-32 SM vs 单 CTA;iter6c 预研标的实证)。
- win10 率 8K-262K:40.6% vs iter5-1cta(锚转移隐含)42.0% —— **未抬升**
  (gm 1.055→1.076 抬了,但赢格分布迁移:real/best 大N 更深,worst 轴
  8-32K 回吐)。
- M3 判据逐条:核心域 radix gm≥1.0 ✓(三场景);全网格 ≥1.0 ✗;
  win10 显著抬升 ✗;anchor 无 <0.9 新洞 ✗(worst 带加宽 + best K1024
  fp32 @4K 0.848,后者在 8K 界外)。
- **方向判决:1cta R0 v0.2 的可赢集 = real/best 全域 + worst vs radix;
  worst-vs-anchor 洞与 hugeN 墙 都指向已排program 的后续**(P1b 并入 P1
  gather 消融 → worst 税;mc 港/iter6c → hugeN)。工具 = m3_verdict.py。

### iter6b mc 臂 A/B 终判 (074, mcab 18 批: K1024 fp32 + K2048 fp16 × 3 场景 × 3 sweep, 4 臂批内配对) — M4 大体达标

- **缺口格(N 131K-262K, BS≤16)**:op26_r0mc vs 1cta-r0 **1.47-1.54×**,
  hugeN(≥512K)**1.93-2.04×** —— 行内多 CTA 墙如设计被击穿;
  vs mc-anchor real/best 1.09-1.14(R0 梯子在 cluster 港成立)。
- **vs radix**:K1024 fp32 全域转正(核心 1.24-1.34,hugeN 1.28-1.34);
  K2048 fp16 缺口带 0.78-0.82(radix 仍胜,但 1cta 的 ~0.5 差距近腰斩),
  worst K2048 fp16 核心域 1.014 首次持平。
- **洞**:worst 轴 vs mc-anchor 0.87-0.93(P1b 税家族,继承 1cta);
  real K2048 fp16 @1M 0.749(单深洞);小 N(4-8K)0.65-0.93
  (dispatch 本就该路由 1cta/单 CTA,非 mc 部署域)。
- **dispatch 蓝图(数据已齐)**:N<~64K → op26_r0(或按 worst 用
  stock);N≥131K 低 BS → op26_r0mc;worst 轴税待 P1b 并入 P1 gather
  消融。工具 = m4_verdict.py;结果根 = results_b200_op26_iter6b_mcab
  (不入库)。

### op26_r0f(P1b 并入 P1 gather)消融判决 (074, 24 批 fp32+bf16) — 主假设证伪,16-bit 门控小赚

- **"二次 gather 是 P1b 税主体"证伪**:r0f/r0 整体 gm 仅 1.0045 (worst) /
  1.0046 (real);worst-vs-anchor 洞主体不动(K2048 bf16 8-32K 0.879→0.903)。
  税的主体 = hist 原子加 + warp0 提取 + 额外 barrier + R0 count 趟本身。
- **dtype 结构干净**:bf16 全 K 双轴一致 +1.5~2.8%(16-bit 随机 gather 更贵,
  缓存有效);fp32 K2048 回归 -3.3~-4.4%(+top_k*4B smem 在 kC=6144 处
  掉 occupancy,8-32K 带最痛);fp32 K512/K1024 持平。
- **裁决:不做统一默认;p1b_cache 按 dtype 门控(16-bit 开 / fp32 关)**,
  fp16 确认批在跑(同根 results_b200_op26_r0f_ab)。确认后把默认写进
  gvr_r0_op26 dispatch(r0f 臂保留为显式对照)。
- **fp16 确认 + 门控落地**:fp16 全 K 双轴 +0.8~2.3%(ALL +1.45%,532 格)
  复现 bf16 方向 → `dispatch_p1bc_op26(dt) = dt != fp32` 写入 gvr_r0_op26
  默认(op26_r0f 臂保留为 force-True 对照);dispatch 默认 smoke 0 FAIL、
  gate 291/291 重验通过。16-bit 白捡 ~1.5-2.8%,fp32 维持原路径。

### op26_r0auto 生产臂落地 (074) — 臂间 dispatch

- mcab 网格 (N×BS) 边界(r0mc/r0 gm,scenario×K×dtype 汇总):N≤8192 mc
  微亏(BS≥128 亏 4-9%);16-32K 持平;65536 起 BS≤64 全胜(1.07 →
  131K 1.14-1.38 → 262K 1.24-1.68 → 1M 1.4-2.6);BS≥128 大 N 持平。
- **dispatch_r0_arm_op26: mc iff (N≥65536 且 BS≤64),否则 1cta**(1cta
  同时保留 op#7 rank-scatter P4)。gvr_r0_auto_op26 wrapper + 臂
  `op26_r0auto` 四处注册;gate 291/291。
- 报告回填时 headline 臂 = op26_r0auto(sweep_nsys extra 记 r0_arm)。

### 报告回填数据采集 — 074 死亡 + 069 marker 接力 (2026-07-12, 069)

- 074 发起的 iter6final 回填 sweep(OUT=results_b200_op26_iter6final,
  臂 = gvr_cutedsl + **op26_r0auto** + radix_cutedsl + sglang;3 场景 ×
  3 sweep × 3K × 3dtype = 81 批)在 13:20:25 随节点回收整体死亡
  (8 分片日志同秒冻结,17/81 marker)。069 按交接配方 marker 幂等重发
  (smoke 先过、无陈锁、25min 零增长确认无双 driver;日志 fin_gpu*b.log);
  driver 对无 marker 批 `rm -f` 后全新重测,半批残留无害。
- **update_report_op26_iter6.py 已就绪**(iter5 wrapper 之上的
  last-writer):u5.main() 全量重导后追加 op26_r0auto —— exactness 门、
  逐格锚迁移(borig 刻度)、D blob/COL/SHORT/复选框/双语 note 卡/
  方法学表行/两 csv 列扩展/含 r0_arm 的 op22rr_op26r_raw.csv。
  note 卡同机 gm 运行时实算并**重跑刷新**(refresh-if-present,防
  partial 数据数字固化)。partial dry-run(26/81)全链路通过:
  exactness 199/199,锚漂移 med 1.0006,real vs radix core 1.23
  (含 mc 路由抬升,收齐后重derive)。

### 报告回填终判 — op26_r0auto 臂入库 (2026-07-12 069, 81/81)

- **同机 vs radix_cutedsl(cold gm)**:核心域 8K-262K real **1.169** /
  best **1.165** / worst **1.206** 三场景全正(1cta 单臂 m3 时代
  1.065-1.096 → dispatch 后大幅抬升);hugeN ≥512K real 1.042 /
  best 1.052 / worst 1.089 —— **hugeN 对 radix 的结构墙(1cta gm
  0.52)被 mc 路由击穿并转正**。
- **同机 vs gvr_cutedsl 锚**:core real 1.313 / best 1.287 / worst
  1.016(worst 轴聚合首次转正;P1b 税洞仍在逐格 0.87-0.94 带);
  hugeN real 2.277 / best 2.389 / worst 1.825。
- QA:exactness 414/414 全 FAIL=0;锚漂移 med 1.0014 (p10 0.992 /
  p90 1.023);r0_arm 分布 mc 1026 / 1cta 1692 格。
- REPORT.html 补丁:D +2718 行、图例/复选框/双语 note 卡(数字运行时
  实算,refresh-if-present)/方法学表 ×2、两 csv +3 列、
  op22rr_op26r_raw.csv(含逐格 r0_arm)。**last-writer 现为
  update_report_op26_iter6.py**(其内部先跑 iter5 wrapper 全量重导)。

### fin 数据负格地图 + iter6c 可行性判决 (2026-07-12 069, 工具 analyze_fin_negatives.py)

- **聚合 gm 掩盖强双峰**:47.8% 格对 radix 仍负。负格几乎全部集中在
  **低 BS**:core BS=1 gm 0.844(胜率 30%)→ BS=2048 gm 2.107(胜率
  91%);hugeN BS1-4 全败(0.51-0.67)、BS16-64 全胜(1.56-2.92);
  smallN(<8K)BS≤512 基本全败(dispatch 域外,记档)。
- **机理(scaling 分析)**:低 BS 两边都是 latency-bound(op26r 时间
  BS 1→16 平坦),缺口 = 每行关键路径延迟比 —— radix 行内 CTA 随 N
  扩,每行延迟几乎不随 N 涨(65K→1M 仅 13→17µs);GVR cs 封顶 4,
  每行延迟 15.6→18.9→38.6µs(65K/262K/1M)。缺口 1.20×/1.34×/2.3×。
  饱和端反之:GVR 每行 0.07µs vs radix 0.16µs(2.3× 更省),故高 BS
  全胜。
- **iter6c(行内多 CTA / GMEM-atomic 两段 kernel)可行性判决**:
  - hugeN(≥512K)BS≤8:缺口 2.3×、内核时长 38µs+,3-5 次全局同步
    (每次 ~2-3µs)可摊 → **可行,但按部署包络记忆 N≤256K 才是主战场,
    hugeN 是应力探针 → 不值得单独立项实现**。
  - core(65K-262K)BS≤8:缺口仅 1.20-1.34×、内核仅 15-19µs,全局
    同步开销即吃掉大半收益 → **多 CTA 路线判不可行**;该带的正确杠杆
    是**每行延迟微优化**(P1b 税 ~µs 级、scan 循环 ILP/向量宽度、
    R0 趟内存级并行),或接受为结构税。
- 下一步优先序更新:① mc 臂 P1b cache 移植(16-bit,直接削低 BS 带的
  P1b 延迟税,与上面判决同向);② core 低 BS 每行延迟微优化预研;
  ③ iter6c 多 CTA 不立项(判决记档)。

### mc 港 p1b_cache 移植 + A/B 判决 (2026-07-12 069, 臂 op26_r0mcc)

- 移植 = 1cta r0f 的 cached P1/P1b verbatim 进 GvrOp26R0ClusterKernel
  (每 cluster CTA +top_k*4B SMEM;`_fmin_f32_inline` 改从 vendored
  cluster 文件 import);wrapper `p1b_cache` 入编译 key;消融臂
  `op26_r0mcc` 四处注册。smoke 首跑全绿(含 hr0/bottom-K),gate 291/291。
- **A/B(54 批,1812 配对格,worst+real × 3 sweep × 9 K/dtype)**:
  mc 调度域(N≥65536 & BS≤64)**全 dtype 全 K 转正** —— gm 按 K 递增
  1.003 (K512) → 1.010-1.017 (K1024) → 1.020-1.034 (K2048),
  **损失格 <0.98 = 0**;worst 轴同构;全格上下文 gm 1.014 无系统性
  损失带。**1cta 的 fp32-K2048 occupancy 回归在 cluster 港不复现**
  (SMEM 预算不同 + mc 域 latency-bound,occupancy 非瓶颈)。
- **落默认:`dispatch_p1bc_mc_op26 = True(无条件,全 dtype)`**,
  与 1cta 的 dtype 门控(fp32 关)形成对照;op26_r0mcc 保留为
  force-True 对照臂。默认 smoke OK,gate 582/582(r0mc+r0auto)。
- 注:REPORT 里 op26_r0auto 的 mc 域行采于本默认之前(少 ~1-3.4%),
  下次 backfill 自动收编,不为此重跑 81 批。
- 工具 = analyze_r0mcc_ab.py;根 = results_b200_op26_r0mcc_ab(不入库)。

## iter7 — core 低 BS leader 尾段攻坚 (2026-07-13, umbriel-b200-092)

### 预研判决 (ncu full-set, 3 负格代表 + radix 对照) — 假设"scan ILP/向量宽度"证伪, 真主税 = leader 串行尾段

- 工具链:prof_lowbs_cell.py(复用 sweep_op22rr.build_call,BS expand/
  radix_aux/cr 与 nsys 战役逐字节同约定)→ analyze_iter7_ncu.py(SOL/
  stall 汇总)→ analyze_iter7_segments.py(warp-stall 采样按**执行过的**
  UCGABAR_WAIT 站点分段;采样对全体常驻 warp 时间均匀 ⇒ 段占比 ≈ 墙钟
  相位分解)。根 = results_iter7_prof(不入库);ncu 锁基频 1.15GHz,
  绝对时长勿与 nsys boost 比。
- **三负格一致(K1024 fp32 131072 BS1 / K2048 fp16 131072 BS1 / K1024
  fp32 65536 BS8)**:barrier-stall 54-58% 居首,其中 UCGABAR_WAIT
  (cluster CGA barrier)独占 40-46% 全部采样;DRAM 吞吐 0.25-0.5%
  (完全非带宽);BS=8 与 BS=1 时长同(38.3 vs 37.9µs)再证
  latency-bound。radix 同格对照 58% barrier-stall 但无单一热点。
- **快路恰 4 次 cluster 同步,段分解**:seg0 P1+P1b+R0 扫描+归并 24-27%;
  seg1 R0 判决/R1 区 0-3.5%;seg2 P3 各 CTA 并行 collect 11-14%;
  **seg3 leader 串行尾段(3×peer DSMEM 拉取 + 单 CTA vendored snap P4)
  57-61%,其中 ~70% 采样是 3 个非 leader CTA 堵在最终 UCGABAR** —— 墙钟
  杠杆 = 缩短/并行化 leader 尾段,非"减同步"本身。
- 设计序:D3 p4_rs(1cta 已验证 rank-scatter P4 港 leader)→ D1
  peer-push gather → D2 cluster-cooperative P4(备选)。PLAN_ITER7.md。

### D3 p4_rs 臂 op26_r0mcr + A/B 判决 (092, 54 批 1812 配对格) — mc 域正向, 默认按 (dtype,K) 门控落地

- 移植 = op#7 exact rank-scatter P4(307 行 verbatim,fixed 256-bin fine
  level,vdiff=0 语义)进 GvrOp26R0ClusterKernel leader 段,const_expr
  调度 vs vendored snap;`p4_rs` 入编译 key;臂 op26_r0mcr 四处注册。
  首硅 smoke 全绿(全 dtype/K 含 hr0/bottom-K),gate 291/291。
- **A/B(worst+real × 3 sweep × 9 K/dtype,us_cold(r0mc)/us_cold(r0mcr))**:
  mc 调度域 ALL gm **1.038**;fp32 K1024/K2048 **1.093/1.093**(max
  1.21/1.23),fp16 全 K 1.016-1.052,bf16 K1024/K2048 1.024/1.031;
  **唯一负带 = (bf16, K512) gm 0.992,18 个 <0.98 损失格全在此**;
  worst 轴同构(ALL 1.035)。全格上下文 gm 1.096(域外 1cta 区更大,
  不参与 ship 判据)。
- **落默认:`dispatch_p4rs_mc_op26(dt, K) = not (bf16 ∧ K512)`**。与
  1cta 的 dispatch_rs_op26(fp32 ∪ BS≥256)再次分化——mc 域
  latency-bound,16-bit rank-scatter 也赢(第三个 port-must-rejudge
  实例,前两个 = fb_fix、p1b_cache)。op26_r0mcr 保留 force-True 对照。
- 工具 analyze_r0mcr_ab.py;根 = results_b200_op26_r0mcr_ab(不入库)。
  注:REPORT 的 op26_r0auto mc 域行采于本默认之前,下次 backfill 收编
  (同 p1bc 先例,不单独重跑 81 批)。

### D2 p4_coop(cluster 协作 P4)臂 op26_r0mcp — 上硅证伪,不 ship (092, 54 批 1812 配对格)

- 设计:去掉 leader gather,各 CTA 就地 rank-scatter 自己的 P3 候选;
  粗/细直方图驻 leader smem 由全 cluster `red.shared::cluster` 共建,
  三个 rank 计数器 `atom.shared::cluster` 取号;6 次均衡 cluster 同步。
- **首硅 bug(新 gotcha,已入库记忆)**:`cluster_arrive_relaxed()` 无
  release 语义 —— 同步后 rank-3 CTA 经 DSMEM 读 leader **刚写**的广播
  标量读到过期值 → 整 slice 候选被误判跳过(错/漏选 index 按 CTA slice
  聚类,debug_r0mcp.py 定位法)。修复 = coop 内 6 同步改
  `cluster_arrive()`(release)。修复后 smoke 全绿 + gate 291/291。
- **A/B 判决(vs op26_r0mc @p4_rs 默认)**:mc 调度域 ALL gm **0.914**,
  全 dtype 全 K 一致负(0.903-0.924),worst 轴同构,679 损失格无正带;
  全格上下文也 0.967。**同步税 + DSMEM 原子延迟 > 分布式收益**,
  iter6c 的全局同步税警告在 intra-cluster 尺度同样成立。
- **裁决:p4_coop 默认 OFF 不变,op26_r0mcp 保留证伪对照;D1 peer-push
  连带不立项**(只省 leader 串行拉取 ~2-3µs 却加 1 次同步,按本数据
  上限即 wash)。
- 工具 analyze_r0mcp_ab.py;根 = results_b200_op26_r0mcp_ab(不入库)。

### iter7 收口判决 (2026-07-13, 092)

- **Ship = D3 p4_rs**(dispatch_p4rs_mc_op26 = not(bf16∧K512))@de9d33b7aa:
  mc 域 gm 1.038(fp32 大 K 1.093,max 1.23)。
- **负格带现状(3 臂 real/bs 探针,当前默认 vs radix)**:K1024 fp32
  65K-262K BS1-8 **全翻正 1.06-1.41**;残余 = 16-bit 262K 低 BS
  (fp16 K2048 0.69-0.71 / bf16 K512 0.78)与 fp16 K2048 65-131K
  0.84-0.90 —— ncu 证 seg3(尾段)仍 51%,但 D2 证伪后 cluster 内
  再无可拿的并行化路径(D1/D2 均判死),**判结构税记档**(radix 行内
  CTA 随 N 扩 + 2-byte 带宽优势的组合)。
- 顺带:gate_op26.py 增 `OP26_GATE_DTYPES` 分片选择器(3 卡并行
  gate,分片按 dtype 编译无重复 JIT)。
- 报告回填:p4_rs 默认(mc 域 +1.5~9%)与 p1bc 默认一起待下次统一
  backfill 收编,不单独重跑 81 批。

### 小 N R0 门 A/B + dispatch_r0_smalln_op26 落地 (2026-07-13, 042→039 迁移接力)

- **动机**:iter6/iter7 判决中 r0auto 的 N≤16K 段输 iter5 op26_1cta
  (fin 全网格锚转移 gm:4096 0.971 / 8192 0.877 / 16384 0.984,
  fp32 段 0.79-0.86)。机理 = 小 N 相位链/latency 主导,R0 梯省趟
  收益≈0,P1b 固定税纯亏。本 A/B 只钉 N∈{16384,32768} 模糊带。
- **A/B(27 批 324 配对格,同批三臂 byte-identical:gvr_cutedsl 锚 +
  op26_1cta + op26_r0;N∈{16K,32K} × BS 1-1024 × 3K × 3 场景;
  042 跑 11 批 → 039 8 卡接力 16 批;metric us_cold(r0)/us_cold(1cta),
  >1 = R0 梯净税)**:
  - fp32:16K gm **1.138** / 32K gm **1.096**,两档 plain 全胜
    (worst 轴最痛 1.154/1.253)→ **OFF 区 = N<65536**。残余带
    (K512,32K) gm 0.979(R0 +1.021,恰在 ≥1.02 红线上且场景分裂:
    real 高 BS R0 +1.26-1.29,worst plain +1.25)——(dt,n) 粒度
    无法切,判不可行动,记档。
  - bf16:16K gm **0.976** / 32K gm **0.932**,R0 系统性赢带
    (512,32K) +16% / (512,16K) +10%(real 轴 K512 全 BS +1.29-1.47)
    → 非 wash,**R0 从 16384 开**。
  - fp16:16K gm **0.953** / 32K gm **0.965**,赢带 (1024,16K) +13%
    → 同 bf16,**R0 从 16384 开**。
  - "16-bit wash 则并 fp32 简单优先"预案未触发——赢带远超 1.02。
- **落默认:`dispatch_r0_smalln_op26(dt,n) = n < (65536 if fp32 else
  16384)`**;gvr_r0_auto_op26 mc 判定后小 N 直路由 gvr_cutedsl_op26
  (qfracs 强制 = 消融调用不改道);op26_r0/op26_r0mc 臂不动(纯对照);
  harness _build_op26_r0auto_call 小 N 记 r0_arm="plain"。
  证据口径注:16K/32K 判决 = 本 A/B 同批数据;4096/8192 方向 = fin
  全网格锚转移历史(op22rr_op26{,r}_raw.csv),两者分开引。
- **验证**:smoke 全绿;r0auto 三点路径(8192 plain / 32768 分 dtype /
  131072 BS4 mc)3 dtype × 3 点全 exact;gate 582/582。
- 工具 analyze_smalln_ab.py;根 = results_b200_op26_smalln_ab(不入库);
  harness sweep_op22rr.py 新增 OP22RR_NS/OP22RR_BS opt-in 网格过滤器。
- **过程事故记档(双 driver 禁令第 3 例,模式新)**:8 卡发车布局中
  g7 兜底分片(real fp16 → best fp16 串行)因 real fp16 三批全 marker
  秒跳,提前撞进 g5 正在重测的半批 best K2048 fp16(两个 nsys 写同一
  输出文件)。处置 = 精确 PID 击杀两棵树(不用 pkill -f)→ 显存归零
  复核 → 作废半批单 driver 重发。教训:**兜底分片不能与主分片有
  未完成批次交集**——串行兜底段只该在全部主分片收尾后手动补发。

### 积压① qfracs UH4 上硅对照 — 证伪 (2026-07-13, umbriel-b200-027)

- **动机**:host 筛选(screen_r0_qfracs.py)uh4 (0.90,0.65,0.40,0.15)
  静态准入 216/216 > M2D;iter5 教训"趟数≠时延"要求上硅。
- **A/B(sweep_qfracs.py,批内配对,臂 = 同 wrapper qfracs 参数切换,
  编译 key 含 qfracs;代表格 K1024 fp32 131072 BS1-16 / K2048 fp16
  65536-262144 BS1-8 + win 保持格 8-32K + 1cta 16-bit 小 N 带;
  real+worst,4 批 4 卡,66 配对格,metric us_cold(m2d)/us_cold(uh4),
  >1 = uh4 快)**:
  - mc ship-gate 域(N≥65536)gm **0.9565**(uh4 慢 4-5%):real fp16
    K2048 0.9905 / real fp32 K1024 0.9330 / worst fp16 0.9447 /
    worst fp32 0.9289;
  - win 保持带(8-32K)gm 0.9583、1cta 16-bit 带 gm 0.9645 —— 全域负;
  - 损失格 <0.98 共 28+ 个(最深 0.879),正带仅孤立 max 1.138。
- **判决:FALSIFIED,M2D 保持默认**。M=4 的静态准入优势完全没有
  转化为时延——多 2 列 smem_ptcnt_multi + tid0 4-rung 游走 + 更深
  rung 的更大 cand 数扫描成本 > 省下的 falsi 趟。"趟数≠时延" 第 2 例。
- 工具:sweep_qfracs.py / drive_nsys_qfracs.sh / analyze_qfracs_ab.py;
  根 = results_b200_op26_qfracs_ab(不入库)。uh4/M3A 候选就此关闭,
  qfracs 杠杆除非出现新机理证据不再重开。

### 积压② K2048 fp32 edge-aim R1 对照 — CLOSED 无收益 (2026-07-13, 027)

- **动机**:iter5d 1cta aim 表按 (K,dtype,N) 翻转,mc 港 R1 inline shot
  只用过几何中心 √(kK·kCC);K2048 fp32 是唯一未对照的重 K。
- **实现**:GvrOp26R0ClusterKernel 加 r1aim∈{center,edge} ctor 参数
  (edge = log2(kK)),wrapper gvr_r0_mc_op26 透传并入编译 key(批内
  同进程双臂配对);默认保持 center,post-edit gate 582/582。
- **A/B(sweep_r1aim.py,K2048 fp32 131072/262144 BS1-16 real+worst,
  20 cold reps)**:整体 gm 1.0021,BS≥2 全平;唯一疑似正带 = BS1
  4 格 gm≈1.016。**确认批(60 cold reps,BS∈{1,2})**:BS1 带缩回
  噪声(real gm 1.0059 / worst gm 0.9992,worst 262K BS1 翻 0.992),
  ALL gm 1.0026。
- **判决:CLOSED,center 保持默认**。机理符合预期:real 轴 R0 静态
  命中 ~0.96,R1 触发率低 → 收益上限天然小;worst 轴也无肉。
  低 reps 的 BS1 单点正带是重测教训第 N 例——判带先看能否复现。
- 工具:sweep_r1aim.py / drive_nsys_r1aim.sh / analyze_r1aim_ab.py;
  根 = results_b200_op26_r1aim_ab + _confirm(不入库)。
