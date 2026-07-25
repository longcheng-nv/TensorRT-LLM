# gvr-topk-weakrung — 战役 B(方向 iii:复合包络弱 rung 专啃)状态与接续指南

- **Campaign ID**: `gkreb2rzas2gh99hx1xczd4gym` (display: gvr-topk-weakrung, artifact indexer-topk-decode-weakrung)
- **启动**: 2026-07-25 06:55 UTC, b200, cuda_cpp_only, effort high, pool = 2×claude opus-4.8 (high) + 2×codex gpt-5.6-sol (high) + 2×n3 opus-4.8 (无 fable-5), max_rounds 8, stagnation 3, eval_timeout 1800, **max_cost $400 (止损盘)**。
- **定位**: 方向 (iii) 的 KF 臂 — 工程复合包络 (COMPOSITE_ENVELOPE_20260725.md, 全 bar PASS) 已立;本战役只啃其 **28 个 PR 回退 rung**(BS 256-1024 × 16k-256k 吞吐墙 + pro_512k@128 / pro_1024k@16)。任何 rung 上 exact 且 >1.0× 的胜利都直接抬升包络(分派按 rung 换臂)。
- **Workload**: 14 格 = 10 个弱 rung 家族 × lo-hit 层, 真实捕获数据复制行物化 (r5_bs/assets_weak, 294MB)。**平台单资产上限 ~64MiB** — 4 个大格 (flash/pro_256k、pro_512k、v32_64k) 用 BS 128/64 代理 (proxy 组合均有 grid_r5pr 实测基线);BS 512/1024 大 npad 角落只能本地终判。
- **Bar (内部)**: 14 格 gm >1.0 且单格 ≥0.95; stretch gm ≥1.15。外部验收 = 弱 rung 上 nsys cold-L2 逐格 beat PR。
- **种子**: v3mt (op41 @bce921d0b1) + e6 (op39 arm_v2) 全源码经 --asset 附入;prompt 含反 broadcast 合规线 (R6 战争教训)、跨行摊销限定 (仅 P1 先验共享)、op39 已证伪清单 (ILP/cp.async/CDP2)。
- **结构判断**: 该域 op39 双锁 (基线原生批摊销 scan pass, 种子付 per-row 税) — 战役目标就是啃这个结构缝, $400 止损防翻车。
- **收割纪律**: 同 RESUME_RECAL.md (quick_ab TORCH_CUDA_ARCH_LIST=10.0a / nsys TMPDIR=/dev/shm / 只认 nsys / 锚检查)。
- **接续**: `kf campaign show gkreb2rzas2gh99hx1xczd4gym`; 收割 → 本地 nsys 探针 vs PR + vs 种子 → 胜 rung 并入 composite_dispatch_routing.json。
- **启动踩坑记录**: (1) 平台单资产 >64MiB 上传 500 (wl_flash_256k_bs256 67.7MB) — 降 BS 重物化; (2) 物化工具 = r5_bs/export_cells_weak.py (幂等)。
