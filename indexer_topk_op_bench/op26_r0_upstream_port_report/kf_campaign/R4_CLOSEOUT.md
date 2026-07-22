# R4 冷启动战役收口报告(gvr-topk-cold60)

2026-07-22, umbriel-b200-027。第二 lineage 冷启动:KF 从 PR#16457 pinned head
(04a0900ff7)起步,骨架硬锁,禁注入第一 lineage 解法。

## 终局判决(champion = round-3 `28dc11f6` gvr_topk_r3_perK_dispatch)

- **geomean 1.6531× vs pinned head**,865/865 tie-robust exact,**0 真实回退**
  (唯一 <1.0 格 pro_64k_L38 grid 0.991 → 60-rep 裁定 1.0129 = 噪声)。
- **三条验收 Bar 全达成**(Bar-1 ≥1.60 / Bar-2 0 回退 / Bar-3 exact)。
- p5 1.221 / median 1.593 / max 3.516;≥1.5× 528 格,≥2× 146 格。

| | 4k | 8k | 16k | 32k | 64k | 128k | 256k | 512k | 1M |
|---|---|---|---|---|---|---|---|---|---|
| **Flash K=512** | 2.069 | 1.901 | 1.628 | 1.320 | 1.285 | 1.586 | 1.444 | 1.735 | 1.846 |
| **Pro K=1024** | 2.653 | 2.060 | 1.700 | 1.357 | 1.279 | 1.568 | 1.476 | 1.808 | 1.974 |
| **V3.2 K=2048** | 1.935 | 1.465 | 1.292 | 1.474 | 1.530 | 1.794 | 1.982 | — | — |

## 横向对照(PR-arm 归一)

- vs sglang_v2: gm 1.099,win 567/865(弱区 N=8195 gm 0.80)
- vs radix_cutedsl: gm 1.593,win 845/865
- vs 第一 lineage compB(参考,分母时点略异): 1.6531 < 1.8267 —
  冷启动+硬锁 3 轮未及第一期积累值,符合 handoff 期望管理。

## 算法要点(冷启动 lineage 自行走出)

P1 hint-CCDF 两级直方图 8 分位阈值梯 → P2 单 pass 8 阈值计数 + log-secant
括号 9×/pass 收缩 + plateau 精确回退 → P3 DSMEM 收集(奇偶双 bank)→
P4 CTA0 4×8bit radix + tie-ticket。dispatch(仅 npad/K):≤12288 direct
1×1024;≤262144 寄存器驻留 GVR 1/4/8/16-CTA(整行进寄存器,多 pass 零重扫,
per-(tier,K) AR6/AR8 实测梯);>262144 流式 16-CTA。多项第一期结论被独立
重发现(1024 线程小 N 档、8-16 CTA 大 N、prior-free 死路)。

## 合规附注(收口披露)

npad≤12288 的 direct 路径不消费 pre_idx(阈值求解在 kC≥npad 下的解析退化
极限)。操作员裁定 = 合规(与生产 kernel 的平凡路径同物、非 per-case dispatch
到无关算子);该裁定为骨架锁死条款的边界情形,在此明示。

## 成本与过程

- 平台花费 **$1110.62**(r1 $352 / r2 $373 / r3 $386;预算检查粒度导致
  超 $800 上限后仍跑完 r3 才截断);对照第一期 $690→1.68×(旧 head 分母)。
- 本地判决:10 次全格/裁定 nsys(GPU 集 1-3/4-7),全部 865/865 exact;
  2 起污染事件(r4pr 自双 driver、GPU 跳卡外来负载)均隔离重测。
- 判决链:v5 1.295(78reg)→ v14 1.343(24)→ r2_wd 1.441(19)→ v25 1.521(2)
  → v27 1.582(2)→ r3_a003 1.618(1)→ **28dc11f6 1.6531(0)**。
- 交付:fork 分支 `kf/r4-champion-final-bs1`(@e1049bca)与
  `kf/r4-champion-r3v11-bs1`(@4c82d9a8);台账 R4_RUN_STATE.md。
- 残余弱点(后续杠杆):N=8195-16399 带 vs sglang_v2 仍落后(gm 0.80-0.99);
  L38 类低 hit 格的 secant 税已被寄存器驻留压到噪声级。
