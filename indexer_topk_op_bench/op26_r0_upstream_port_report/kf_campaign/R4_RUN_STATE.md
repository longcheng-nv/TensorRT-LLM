# R4 冷启动战役 — 运行状态 checkpoint

Updated 2026-07-22 (会话暂停：用户要求迁移到另一台机器)。
接力指南 = 本文件 + `R4_COLDSTART_HANDOFF_PROMPT.md`（§A 贴入目标机器新会话）。

## 已完成（迁移后不必重做）

- **分母 pin: PR#16457 head = `04a0900ff7c233a03e95dc8c35321c37c256d627`**
  （2026-07-22 fetch 自 `refs/pull/16457/head`；比 07-21 台账的 @6140078816 又前进了
  3 个 commit）。一期一 pin，目标机器上直接用该 SHA，不要重新追新。
- 增量盘点（旧打包基点 e6fdbfac3d → 04a0900ff7，GVR 相关）：
  - `04a0900ff7` [test] r0-equivalence tie-aware fp32 index-set comparison（纯测试）
  - `a707cfe41c` [fix] equalize cs=1/cs>1 SMEM layouts for the **LB hybrid kernel**
  - `b14ec40e1b` [chore] ruff-format（无语义）
  - `6140078816` **Merge PR #16424 (perf/gvr-kernel-opts)** ← 主要增量来源
  - 文件级 diff（需 re-port 的实质范围）：
    `gvr_topk_decode.py` +1208/-...（大改）、`gvr_topk_decode_load_balance.py` +36、
    `run_gvr_topk.py` +81、tests +141。
  - ⇒ re-port 工作量集中在 gvr_topk_decode.py 的 #16424 kernel-opts + LB hybrid fix；
    kb512@K2048 (07-20) 已含在其中。

## 未完成 / 打开的问题（目标机器上做）

1. **gvrpkg 底座目录未定位**：README.md 说 baseline 打包在 `gvrpkg_head/`，
   但 kf_campaign/ 下 `ls gvrpkg/`、`ls gvrpkg_head/` 均不存在（find 输出可疑）。
   到目标机器先 `find … -maxdepth 2 -name 'gvrpkg*'` 或查 `harvest/r0*`；
   若确实丢失，从 ws/definition.json 的 baseline solution 记录或第一期
   campaign 的 baseline_solution.json 恢复。
2. Task 2: re-port 增量 → `pr_head_solution.json`。
3. Task 3: parity gate（28-cell nsys 配对 vs in-tree pinned head，|gm 漂移|≤2%）。
4. Task 4: `kf auth status` → lifecycle 探测 → ws/prompt.md 换成
   R4_COLDSTART_HANDOFF_PROMPT.md §B → `kf campaign init gvr-topk-cold60
   --definition ws/definition.json --workloads ws/workload.jsonl
   --baseline-solution pr_head_solution.json --gpu-spec b200
   --language cuda_cpp_only --effort high --prompt-file ws/prompt.md`
   → prepare → start。**尚未 init，平台侧零花费。**
5. Task 5: 本地 865 格 pinned-head 分母（drive_grid_shards.sh 8-shard）。
6. Task 6: monitor + 收割循环。

## 环境备忘（原机器 umbriel-b200-048，已弃用）

- 8×B200 全空闲、温度正常；kf CLI 1.0.0 在 ~/.local/bin/kf（用户级，NFS home，
  目标机器大概率同样可见——先 `which kf && kf auth status` 核查）。
- 本仓库在共享 NFS，git FETCH_HEAD 状态不随机器走：目标机器重新
  `git fetch origin refs/pull/16457/head` 并核对 == 04a0900ff7。
- 机器局部件需重建：/tmp/gvrlayers overlay（cutlass make_fragment 报错时按
  README.md 一行命令重建）、PYTHONNOUSERSITE=1 + machine-local cutlass 4.5.0
  （新容器）、venv（若目标机器无既有 gvr-prefill venv）。
- GPU 健康备忘（若迁去这些机器）：b200-019 GPU0 坏散热 pin GPU1；
  b200-035 GPU0 热节流 nsys 用 GPU1；b200-036 GPU1 坏散热 timing 只用 GPU0。
