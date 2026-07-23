# gvr-topk-pr16457-fresh — 战役状态与接续指南

- **Campaign ID**: `6em6mf55g11g767p5wcepgy07w` (display: gvr-topk-pr16457-fresh, artifact indexer-topk-decode-fresh-full)
- **启动**: 2026-07-23 08:17 UTC, b200, cuda_cpp_only, effort high (2×fable-5 + 2×gpt-5.6-sol + 2×n3-opus-4.8), max_duration 28800s, max_cost 800 USD, eval_timeout 1800s, stagnation 3
- **目标 (外部验收 bar)**: BS=1 865 格 gm ≥1.60× vs PR#16457;BS 2-1024 全网格 gm ≥2.0×。优先级 Pro > Flash≈V3.2, n≥32K 优先。
- **回退 bar (2026-07-23 用户放宽)**: 原 "0 回退" 调整为 **单格最大回退 ≤5%** (即每格 speedup vs PR#16457 ≥0.95),BS=1 与 BS>1 网格同标准;外部 nsys cold-L2 终判时执行。
- **平台缺陷 (2026-07-23 排查)**: agent pool 中 2×claude fable-5 槽位每轮即死 — KF org 对 `claude-fable-5` 限额 0 req/min (429),模型未开通;pool 运行中/fork 均冻结,本战役有效 agent = 4/轮 (2×gpt-5.6-sol + 2×opus-4.8)。后续新战役 pool 应把 fable-5 换成 opus-4.8 或加码 codex。r1-a004 为 14700s 硬超时正常损耗。
- **约束**: GVR 骨架强制 (preIdx 先验 + secant/log 阈值 + exact refine);禁 prior-free 整体替换/跨算子 dispatcher;子阶段可吸收 radix/histogram 成熟原语;exact (tie-robust set 判卷);冷启动 — prompt 未含任何旧 KF champion 或历史弃用分析,仅 op26 REPORT 事实 (P4 主导 827/865 median 37%、undershoot 偏置) + PR#16457 源码摘录附录。
- **Workload**: 58 = 28×BS1 层格 (cell_*.safetensors, 865 格分层子集) + 30×BS 扩展格 (wl_*_bs{1,4,32,128,256,1024})。baseline = PR#16457 头原生批量 nsys cold-L2 外测中位数 (gm 15.53µs)。
- **接续**: `kf campaign show 6em6mf55g11g767p5wcepgy07w`;收割 `kf campaign kernel list 6em6mf55g11g767p5wcepgy07w --top 10`;外部终判走本目录既有 harvest 流程 (drive_grid_* + nsys_ab.py, 865 格 + BS 网格, per-batch p95 锚检查, 禁并发探针)。
- **同名不同物**: 平台上另有 Running 的 gvr-topk-bs2x-v3 (befh5fh…, R5 旧战役) — 勿混淆。
- 花费快照: COSTS_20260723.md
