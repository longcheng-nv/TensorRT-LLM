# SESSION HANDOFF — 2026-07-13 (umbriel-b200-027, branch omni/op21-gvr-prod)

本 session 完成的三块工作(全部已 commit,HEAD @d321f64323):

## A. op28 — 外部最新 top-K 基准(DONE)
sglang@main v2 + flashinfer 0.6.11 vendored 为单算子,27 批 nsys 全网格
(GPU2-7),906 cells,gate 459/459。判决:**sglang_v2 胜当时所有 in-tree 臂
(vs op21_hls 1.38-1.77×)**,flashinfer ≈ 持平。已并入 op22 REPORT.html
(update_report_op28.py = last-writer)。启发提炼 = op28_ext_topk/INSIGHTS_GVR_NEXT.md。

## B. op27 vs op25 分析(DONE)
唯一 ship 改动 = K2048 尾梯 (0.75,0.45,0.048);K2048-worst 1.146→1.437×
对基线,其余全网格 ±2% 噪声。遗留 = worst-bf16 K2048 bs 口袋(OP27_R2 旋钮
备好未启用)。

## C. op29 — GVR-HBE 战役(IN PROGRESS,iter10 收口)★ 主恢复目标
- **恢复文档 = `op29_gvr_hbe/RESUME_PROMPT.md`**(1 分钟上下文、已证明判决、
  证伪清单、6 步优先队列、字节级启动命令、gotchas)。
- 状态:iter9 突破(engaged 域 K≤1024 & N≥131K & BS>512 对 sglang_v2
  **1.06-1.50× 三场景全胜含 worst**,守卫外零回归,gate 216/216);
  iter10 扩围证伪已回撤。
- 工件:kernel = src/gvr29(fork + topk_hbe.cuh);脚本 = scripts/
  {gvr29_op,gate_op29,pilot_op29,parse_pilot,crux_*}.py;
  pilot 历史 = results/pilot/iter{3..9}/;账本 = ITERATIONS/FALSIFIED/WALLS.md。
- 下一步(优先序,详见 RESUME):① NCU 归因 K2048 (+188µs);② 131K 固定相
  瘦身;③ cluster 路径 HBE;④ 全网格 sweep + REPORT 臂;⑤ 生产集成(需用户)。

## 新 session 恢复方法
1. cd 到 TensorRT-LLM 仓库根(本文件所在 checkout)。
2. 粘贴提示词:
   「继续 op29 GVR-HBE 战役:先读 indexer_topk_op_bench/op29_gvr_hbe/
   {RESUME_PROMPT.md, ITERATIONS.md, FALSIFIED.md},按 RESUME 的优先队列
   从第 1 步(NCU 归因 K2048)开始,遵循 omni-kernel 协议。」
3. memory 会自动带入 [op29 GVR-HBE 战役] 索引行;preflight 按 RESUME
   检查 git HEAD ≥ d321f64323、GPU 占用黑名单、nsys 需 env -u *_TOKEN。
