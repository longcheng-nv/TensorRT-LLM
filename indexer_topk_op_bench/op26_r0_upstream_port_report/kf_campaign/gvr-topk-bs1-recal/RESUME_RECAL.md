# gvr-topk-bs1-recal — 战役 A (bar 重定标 + R4 种子) 状态与接续指南

- **Campaign ID**: `rd06zd9zf55jdfxfr5z077t6wg` (display: gvr-topk-bs1-recal, artifact indexer-topk-decode-bs1-real)
- **启动**: 2026-07-24 06:07 UTC, b200, cuda_cpp_only, effort high, pool = 2×claude opus-4.8 (high) + 2×codex gpt-5.6-sol (high) + 2×n3 opus-4.8 (**无 fable-5** — org 限额 0 rpm 的死槽教训), max_rounds 10, max_duration 28800, stagnation 5, eval_timeout 1800, max_cost $1200。
- **目标 (外部验收 bar, 07-24 重定标)**: BS=1 865 格 gm ≥1.60× vs PR#16457, **单格回退 ≤5% (per-case ≥0.95×)**, 865/865 tie-robust exact。战役 B (BS 2-1024, bar gm ≥1.25, oracle≈1.44 之下) 待 A 收口后启动。
- **种子 (用户 07-24 批准放开)**: R4 champion `28dc11f6` (gvr_topk_r3_perK_dispatch, 外判 gm 1.6531/865 exact/0 回退) 以 `--asset seed_r4_champion.cu + seed_r4_main.cpp` 附入, prompt 指示先交种子立地板再攻弱区。
- **弱区攻击序**: (1) 32K-64K 全模型 (1.28-1.36); (2) V3.2 K2048 @8K-16K; (3) 128K-256K。禁回吐 4k-8k ≥1.9 / 512K-1M ≥1.7 强区。
- **Prompt 结构**: cold60 基底 (含 harness ~15µs 地板校准段 + PR#16457 源码摘要附录) + bar 放宽补丁 + 种子章节 (判决表 + 算法梗概 + fresh 战役 nsys 验证过的 6 条技术事实; 源码走 asset, 32KB prompt 上限所迫)。
- **Workload/baseline**: 复用 cold60 的 28 格 BS1 分层子集 + baselines.jsonl (gm 14.30µs, PR 头 nsys cold-L2 外测)。
- **收割纪律**: 每轮 champion 先 `quick_ab.py` smoke (已修: 强制 TORCH_CUDA_ARCH_LIST=10.0a) 再 `run_nsys_ab.sh` 探针 (本机 overlay 满 → 必须 `TMPDIR=/dev/shm/loncheng_tmp`); 终判 = drive_grid_shards.sh 865 格 + per-rung pr_cold 锚检查 + 禁并发探针; CUDA-event 大 n 数字不可信 (PR 头 cuteDSL host 税 ~1.1ms/call, 只认 nsys)。
- **接续**: `kf campaign show rd06zd9zf55jdfxfr5z077t6wg`; 收割 `kf campaign kernel list rd06zd9zf55jdfxfr5z077t6wg --top 10`; 不达标 fork `--append-prompt` 纠偏 (pool 健康, fork 有意义)。
- **同库其他战役**: gvr-topk-pr16457-fresh (6em6mf55…, 已收口 CLOSED, 判决见其 RESUME_FRESH.md); gvr-topk-bs2x-v3 (befh5fh…, R5 旧战役)。

## 终局 (2026-07-25 01:47 UTC Completed, 5 轮, $1083.68)

- champion `gvr_topk_cap4096_pmin_cs1_direct_hist` (6d9fe9e4, r5-a003), 内部 1.1502。
- **本地 nsys 双探针判决 (8 格, GPU0 背靠背, 锚差 <3%)**: champion cold gm **1.3951** (0 回退, 8/8 exact) vs 种子 r3_28dc11f6 cold gm **1.6867** — **种子 8/8 全胜, champion 无合并价值 → 不采纳**。
- **战役 A 验收结论**: bar (gm≥1.60 + 回退≤5% + exact) 由**种子本身**满足 (R4 全网格判决 gm 1.6531, 865/865 exact, 0 真实回退, grid_r4r3cg.csv) — A 的交付物 = R4 champion 原样;KF 增量为负。
- **模式实锤 (两次战役重复)**: KF swarm 无法超越强种子 — fresh 冷启动 5 轮 0.95×, recal 种子加持 5 轮 1.15 内部 (nsys 1.40) 仍距种子 -17%。内部计分尺跨战役漂移 (种子在 R4 harness 读 1.37, 在 recal 世代读数未复现) 疑似 agents 未成功以种子立板 — 但外判已足以裁决, 不再深挖。
- 探针数据: `ab_recal6d9f.json` / `ab_seedr4chk.json`; champion 源码 `harvest/recal_6d9fe9e4/`。
- **战役 B 决策待用户**: B 原案 (BS2-1024 gm≥1.25, R4 种子) 的前提 "种子加持的 swarm 能加值" 已两连败;且 BS>1 域 R5/R6 histor gm 0.99/0.67。启动前建议重议。
