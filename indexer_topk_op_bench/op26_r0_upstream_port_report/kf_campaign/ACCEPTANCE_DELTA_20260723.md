# 验收 bar 调整记录 — 2026-07-23 (campaign gvr-topk-pr16457-fresh 运行中)

用户指令（本 session）：目标中的"0 回退"放宽为 **"最大性能回退的 case 回退不超过 5%"**（即 per-case speedup ≥ 0.95× vs PR#16457 锚）。

生效范围：
1. **外部终判**（865 格 + BS 网格 nsys cold-L2 harvest 流程）：判卷规则改为
   `gm 提升 ≥ 1.60×（BS1-1024 全网格几何平均）且 min per-case ≥ 0.95`。
   正确性 bar 不变（tie-robust 值多重集 exact，全格全行）。
2. **潜在 fork steering**：若本战役收敛于 bar 之下，fork 时通过
   `--append-prompt` 注入放宽后的 bar —— 允许 agent 用 ≤5% 的窄域牺牲
   换取整体 gm（现行 prompt 的 0-回退约束会让 agent 过度保护最差格）。

运行中战役 prompt 冻结，本调整不影响其进行；round 内 agent 仍按 0 回退
优化（更严子集，不会产生违反新 bar 的候选）。
