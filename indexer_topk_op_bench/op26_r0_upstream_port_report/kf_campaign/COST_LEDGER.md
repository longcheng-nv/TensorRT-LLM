# KF GVR 战役成本台账(R4 + R5,2026-07-22/23 会话)

更新 2026-07-23 08:1x UTC。KernelFactory 数字 = `kf campaign cost` 实测
(计费有滞后,v3 仍在跑会继续增长;其余三个 campaign 已终态、数字最终)。

## KernelFactory 平台侧

| Campaign | ID | 状态 | 结果 | 花费 (USD) |
|---|---|---|---|---|
| R4 gvr-topk-cold60 (BS=1 冷启动) | pra6srbd7h4… | Cancelled(预算截断) | ✅ champion 28dc11f6, gm 1.6531, 三 Bar 全达成 | **$1110.62** |
| R5 gvr-topk-bs2x (v1) | rngnxv95cx5… | Cancelled(平台 custom_inputs 0% 故障,止损) | 无产出(平台缺口复证) | **$9.38** |
| R5 gvr-topk-bs2x-v2 | vk9m3tetqh1… | Cancelled(lineage exactness DQ → fork) | r1/r2: 内部 0.98;全格 0.93-0.94 + 8 例盲区 inexact | **$719.02** |
| R5 gvr-topk-bs2x-v3 (fork, rounds 3-5) | befh5fh2595… | Cancelled(rounds 3-5 完成) | champion 156ab438: 全 exact;750 格 gm 0.986(目标未达) | **$596.91**(终) |
| **KF 小计(本会话)** | | | | **$2435.93(终)** |

参考(既往 lineage,非本会话花费): 第一期 gvr-topk-bs1-real ≈$690(→1.68×);
R3 gvr-topk-r3 $761(→compB 1.8267)。

## 本地 Claude Code 会话侧

- 本会话 token 费用无法从会话内部读取 —— 请在 CLI 运行 `/cost` 查看,
  并把数字回填到下行:
  - **本地 Claude 会话花费: $______(/cost 实测,待回填)**
- 本地 GPU(umbriel-b200-027 8×B200)为自有资源,不计费。本会话 GPU 消耗:
  R4 侧 10 次全格/裁定 nsys + 3 次 parity/探针;R5 侧 1 次 750 分母 +
  3 次 750 全格 + 5 次探针/exactness 扫描。

## 产出对照(平台花费 → 实测收益)

- **R4($1110.62)**: BS=1 865 格 geomean **1.6531× vs PR#16457 head**,
  865/865 exact,0 真实回退;交付 fork 分支 kf/r4-champion-final-bs1。
- **R5(至今 $753.61)**: BS=2-1024 尚未达标(目标 2.0×;现全格 ~0.94,
  内部 1.006 爬升中);已产出: 750 格 head 分母、平台两个缺口的复证与
  规避方案、lineage exactness bug 定位 + graft 保底、弱区图谱 steering。
