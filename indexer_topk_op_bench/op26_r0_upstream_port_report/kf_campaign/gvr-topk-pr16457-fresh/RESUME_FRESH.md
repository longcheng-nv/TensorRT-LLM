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
- **终局 (07-23 20:49 UTC Completed, 3 轮停滞, $636.43)**: champion `gvr_topk_r3_dense1024` (82e2b292, r3-a004)。本地 nsys 终判探针 (b200-027 GPU0, 6 代表格): **cold gm 0.952, 4/6 回退, 最好 pro_1024k 1.019** — 真实 kernel 时间贴平略输 PR 头, 距 gm1.6 bar 无望; 维持并行 session 判读: 不建议 fork (pool 冻结 + 差距结构性)。骨架合规 (多阈值单遍探测替代 secant + SMEM radix refine, 均在允许范围)。
- **测量陷阱 (本次实锤, 复测 KF 候选必读)**: (1) quick_ab CUDA-event 的 5.7 gm 是假象 — PR 头 cuteDSL 大 n (post-cr n≥128K) 路径有 ~1.1ms/call **host 税** (event 计时含 host gap, nsys GPU-projected 剔除后 21-29µs); 小 n 的 1.1-1.6× 同样被 host 开销不对称污染 (nsys 实际 0.87-0.95)。kernel 判决只认 nsys (再次验证)。(2) 候选 torch-ext 构建必须 `TORCH_CUDA_ARCH_LIST=10.0a` 强制覆盖 (环境预设多架构列表使 cluster_group 代码在 sm_75/8x pass 下解体) — 已固化进 quick_ab.py。(3) b200-027 overlay 根分区 100% 满 → nsys Bus error; 绕过 = `TMPDIR=/dev/shm/loncheng_tmp`。

## FORK (2026-07-24, 本 session): gvr-topk-fresh-banded
- **ID**: `xp49vmaw193c90sthdzyk0av9c`, forked from round 4, rounds 4-8, stagnation 4.
- **分带 bar (用户重定标)**: Band A = ISL 32K-1M: gm ≥1.60× 且 per-case ≥0.95;
  Band B = ISL 4K-32K: gm ≥1.00× 且 per-case ≥0.95。steering = fork_steering_banded.md
  (勿碰 4k 地板格 / 砍行流量是唯一 1.6× 杠杆 / r1-3 GO+dead-end 清单)。
- 已知随行缺陷: pool 冻结 → 2 个 fable-5 死槽 (429) 继续空转, 有效 4 agent/轮。
- 终判: Band A/B 分带判卷, 外部 nsys cold-L2 (ACCEPTANCE_DELTA_20260723.md 放宽版)。
