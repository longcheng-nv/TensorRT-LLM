# R5 BS 战役收口报告(gvr-topk-bs2x v1/v2/v3)

2026-07-23, umbriel-b200-027。目标: §7b 真实数据 75 cell × BS2-1024(750 格)
avg ≥2.0× vs PR#16457 pinned head 原生批 + 全格 ≥1.0 + BS=1 守 R4 champion 水位。

## 终局判决(champion = v3 round-5 `gvr-topk-r5-combined`, 156ab438)

| 判据 | 结果 | 结论 |
|---|---|---|
| 逐行 exactness | 750 格 + 865 格全 exact(盲区 cs8 竞态末轮修复) | ✅ |
| BS>1 avg ≥2.0× | **geomean 0.9862**(min 0.374, max 4.79) | ❌ |
| BS>1 全格 ≥1.0 | 359/750 达标(391 回退) | ❌ |
| BS=1 守 R4 水位 | 1.2233 vs head,但 = R4 champ 的 0.743×(仅 40/865 更快) | ❌ |

**目标 +100% 未达成**;三个 campaign(v1 平台故障/v2 lineage DQ/v3 fork)
$1325 与 5 轮演进后,吞吐域(BS≥16 中大 N)对 head 原生批的摊销优势未被撼动。

## 分域 geomean(model × BS,vs head 原生批)

胜域: BS 2-8(pro 1.12-1.25×)与零星 512-1024(pro 1.13-1.19×);
败域: BS 16-256 全线(最深 bs128 0.76-0.88), flash 除 bs2-4 外全败。
逐格 ≥2× 的 41 例集中在 4k×高BS(direct 批量化,最高 4.79×)。

## 结构性结论(如实)

1. head(cuteDSL)原生批在吞吐域本就接近带宽极限——BS≥32 时 750 格中
   head 的行摊销让"每行一份 GVR 仪式"的候选难以为继;agents 的
   rows-per-CTA/流式收缩只追平到 ~0.95-0.99。
2. +100% 的验收 bar 在该分母下超出 GVR 谱系当前可达范围(R4 的 BS=1
   1.65× 是 latency 域;吞吐域头部空间实测 ~1.0-1.25×)。
3. exactness 战役有效: 外部全格审计 → fork steering → 自测清单,
   8 例平台盲区竞态三轮内清零 — 方法论沉淀可复用。

## 部署建议(工程组合,非 KF 产物)

按 (b, npad, K) 静态分派(全部推理时已知,无 hit-rate):
- b == 1 → **R4 champion 28dc11f6**(865 格 1.6531×, 0 回退);
- b 2-8(K≥1024)或 4k 小 N 任意 b → **R5 champion 156ab438**
  (该域 1.04-1.25×, 4k 高 BS 至 4.8×);
- 其余(b≥16 中大 N)→ **保留 head 原生批**。
组合期望: 全 (865+750) 格无回退,加权收益集中在 decode 主域 BS=1。

## 成本(KF 最终账)

R4 $1110.62 + R5 v1 $9.38 + v2 $719.02 + v3 $596.91 = **$2435.93**
(+ 本地 Claude 会话,见 COST_LEDGER.md 回填行)。

## 交付

- fork 分支: kf/r5-champion-bs-combined(R5 终局 champion, code-only);
- 判决数据: r5_bs/grid_r5g_final.csv(750)/ grid_r5bs1guard.csv(865)/
  grid_r5pr.csv(head 分母);
- 过程台账: R4_RUN_STATE.md(R5 各节)+ 本文件。
