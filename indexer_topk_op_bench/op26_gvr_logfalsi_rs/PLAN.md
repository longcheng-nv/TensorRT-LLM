# op26 — 把 HLS-op25 的两个思想移植回 原始 GVR(cuteDSL) 与 GVR multi-CTA(PR#15198)

2026-07-09 · umbriel-b200-038 · branch omni/op21-gvr-prod
目标报告:`op22_temporal_fixed_hr_bench/REPORT.html` §1/§2(op22rr bundles,同数据同条件)

## 0 · 两个优化点 vs 证伪史(动手前的判决依据)

### 优化点 1 — P2 secant 换 log-count 插值(+ kCC/kFTarget 窗口收窄)

用户直觉:indexer logits 在 top-K 阈值附近的 CCDF 近似指数 ⇒ log(count) 对阈值近线性
⇒ 秒差法在 log 域插值收敛更快。

**证伪史裁定:方向正确,且单 CTA 形态已经存在。**
- Q5e(gvr_phase_timing)证伪的是"合成数据上 CCDF 形状拖慢 P2"(1.46 iter/100% 收敛)
  —— 但 op22 机理结论(MECH_FINDINGS)明确:**真实/固定-hr 数据上 hr≈0.9 毒化 pmean
  init → undershoot 全行重扫才是 GVR 家族输给 radix 的真因**。P2 在 op22rr 数据轴上
  就是靶子,与 Q5e 不矛盾(数据轴不同)。
- **op13 iter8 已实现并 nsys-ship 单 CTA 形态** `GvrP2CLog`(`op13_gvr_p2cand/src/
  gvr_p2clog_op.py`,@390c99c3e4):log-count 插值 `f=log2(clo/kFT)/log2(clo/chi)`
  + 窗口收窄。ship 表 `dispatch_p2c_v2`(fp32-only):K512 → 线性窄窗 (1536,1280)
  N≤65536(log 被证伪);K1024 → logn(2048,1024) @ N≤32K∪131K(−32%@8K/−22%@131K);
  K2048 → logn(4096,2048) @ N≥8192(−12%@262K)。16-bit 无证据 → 基线。
- **本战役的增量**:①ship 表从未在 op22rr bundles(op24 radix-相对场景 + 真实 hr 分布
  temporal synth)上验证;②cluster(PR#15198)从未移植——而证伪史指出 MC 域 N≥64K
  恰是 P2 主导相位;③undershoot 修复(见下)从未进这两个内核。

### 优化点 2 — exact rank-scatter P4

用户直觉:HLS-op25 的 rank-scatter 是精确的,可以搬回原始 GVR 消除历史上 top-K
非精确的隐患。

**证伪史裁定:两件事必须拆开。**
- **非精确的根因不在 P4**:report.html §5 红牌(rank-scatter 臂 pro fp32 L10、op18/19
  多层,−1 空槽)的根因 = vendored P3 retry-shrink 只处理 overshoot(`while count>kCC`),
  undershoot 直接漏出且 bracket 高端无 count<kK 保证(gvr_ms_op.py:1237 op21 FIX 注释)。
  **真正的精确性修复 = 把 op21 FIX(双向 bounded refine + log-falsi 瞄准 + hi 端扩张)
  移植回来**,与 P4 选型无关。
- **rank-scatter P4 本身已存在且已有生产判决**:op#7 `gvr_cutedsl_rs`
  (`p4_recursive_digit/src/gvr_topk_decode_p4.py`,enable_p4_rank_scatter_exact,
  vdiff=0 全矩阵)。生产 sweep 校正:**中位 0.92×(偏慢),只在 fp32(中位 1.01×,
  60% 胜)∪ BS≥256(1.01×,59%)有净赢** ⇒ 只能条件调度,不能无脑替换。
- **本战役姿态**:rank-scatter 以 op#7 判决域条件调度(fp32 全域 + 16-bit BS≥256),
  与优化点 1 组合后窗口收窄使 cand 减少(P4 代价 ∝ cand),经济性可能改变——
  由 op22rr A/B 数据裁定,不预设。

## 1 · 臂定义

| 臂 | 内核 | P2 | P3 fallback | P4 |
|---|---|---|---|---|
| `op26_1cta` | GvrOp26Kernel ⊂ gvr_topk_decode_p4.GvrTopKKernel | fp32: op13 v2 表(lin-narrow/logn/基线 分 K 分 N);16-bit: 基线 | **fb_fix 常开**(op21 FIX 移植:只接受 [kK,kCC],hi 无保证则扩张,log-falsi 瞄准,30 iter 耗尽落 undershoot 侧) | exact rank-scatter @ fp32 ∪ BS≥256;其余 snap |
| `op26_mc` | GvrOp26ClusterKernel ⊂ GvrTopKClusterKernel | fp32: log-count 插值(基线窗口,不收窄);16-bit: 基线 | 暂不动(cluster P3 为 per-slice 决策 + leader handoff,改动风险大;记为后续) | 原 snap(leader-only;op#7 判决域不含 MC) |
| 锚 | gvr_cutedsl / gvr_multicta_cutedsl | 原样 | 原样 | 原样 |

设计原则(证伪史教训):不编辑 vendored 文件,全部 subclass override(op13 GvrP2C
ship 模式);每个改动独立门控旗标,便于分解消融。

## 2 · 验证

1. **精确性门禁**(先于任何 nsys):op22rr bundles 三场景 × K × dtype × N @BS=1,
   排序值集合 vs torch.topk;undershoot 对抗压力(hr≈0 preIdx + 近 tie 平台,
   专打 fb_fix);16-bit tie 压力。红线 = 任何非精确即停。
2. **nsys 全网格**(方法论同 op22rr/op25 backfill):cold-L2 纯核、同进程配对、
   done-marker 幂等、`env -u GITHUB_TOKEN -u HF_TOKEN`、*.sqlite 永不入库。
   - GPU0:`OP22RR_ARMS="gvr_cutedsl,op26_1cta"` OUT=results_b200_op26a
   - GPU1:`OP22RR_ARMS="gvr_multicta_cutedsl,op26_mc"` OUT=results_b200_op26b
   - 81 批/卡,估 ~6-7h 墙钟(seqlen ~2min/批,bs ~8.5min/批)。
3. **报告更新**:update_report_op26.py 以 update_report_radix.py 为底
   (自包含 last-writer,重导 mc+op25+radix+op26 全套,避免互抹 gotcha);
   QA 门 = script 数不变(2)、锚漂移 vs 已有 gvr_cutedsl/gvr_multicta 行、
   exactness 全过。

## 3 · 预期与可证伪预言

- real 场景(hr 高,pmean init 中毒域):op26_1cta 对锚 fp32 K1024/K2048 应复现
  op13 量级(大 N 5-20%);K512 窄窗小赢;16-bit ≈1.0(P2 未动)。
- worst 场景(hr=0.05,undershoot 域):fb_fix + log 插值应给出最大相对增益
  (锚在此域重扫最多);若 wash 则说明 worst 的重扫瓶颈在 P3 collect 而非 P2 eval 数。
- op26_mc:MC 域(N≥64K)P2 主导 ⇒ log 插值应有可测收益;若 wash,
  说明 cluster 的 P2 eval 已被切片并行摊薄。
- rank-scatter 组件:fp32 小幅正贡献(op#7 判决),若与窄窗组合后仍 ≤1.0,
  记录并在最终臂中关闭(分解消融定位)。
