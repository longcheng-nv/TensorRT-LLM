# R4 handoff — GVR Top-K KernelFactory 冷启动战役（beat-the-PR-head，第二 lineage）

Prepared 2026-07-22 by the analysis session on branch `omni/op21-gvr-prod`.
两级 prompt：§A = operator /goal（贴进判决机器的 Claude Code 会话），
§B = campaign `prompt.md` v3-coldstart（KF agent 看到的题面，替换 `ws/prompt.md`）。

**与 R3 的本质区别**（用户指定）：本战役是**第二 lineage 冷启动** —
KF 的 baseline-solution = PR#16457 **当前 head** 的 GVR kernel 本身，
**不以**第一期 champion `c74f_sbx` 或 R3 composite `compA` 为起点，
**不注入**它们的任何正向解法（all-coop 多 CTA、bottom-k complement、
per-pass early-exit、composite dispatch 等），让 agent 依靠自行分析寻找优化方向。
**允许注入**的历史材料仅限三类（折中注入，用户 2026-07-22 拍板）：
(a) 实测死路清单（"别走哪条路"的事实，非解法）；
(b) 正确性陷阱（tie 边界、undershoot 偏置、cluster arrive 内存序）；
(c) `op26_r0_upstream_port_report/REPORT.html` 内已发布的事实
    （含 §9e phase breakdown：P4 块主导）与 PR#16457 页面上的历史讨论。
除此之外的本地分析材料（op35 oracle、R3 台账、harvest 源码）一律不进题面。

---

## §A Operator /goal prompt（贴入判决机器的新会话）

```
/goal — GVR Top-K KernelFactory 冷启动战役：以 PR#16457 当前 head 为唯一起点，
KF 自行分析推高，验收目标 geomean +60% vs 当前 head

## 0. 定位与已有资产（先读，不要重复花钱）
- 优化对象: TensorRT-LLM PR#16457 的 GVR (guess-verify-refine) top-K kernel，
  DeepSeek 稀疏注意力 indexer decode 场景，BS=1，fp32 logits，
  三类模型 K=512 (V4-Flash, ISL 4K–1M) / 1024 (V4-Pro, ISL 4K–1M) /
  2048 (V3.2, ISL 4K–256K)，全 layers，共 865 个真实采集 cell
  （cell 集合与测量口径 = op26_r0_upstream_port_report/REPORT.html §7b，
  由 kf_campaign/export_cells.py 从 §4 real data 生成，不多不少）。
- 本战役是第二 lineage 冷启动: KF baseline-solution = PR#16457 当前 head 的
  GVR kernel（独立打包），起点 speedup = 1.0×。禁止把第一期 champion
  c74f_sbx / R3 compA 及其正向解法作为起点或注入 steering；
  它们只在收口后做事后对照（横向比较两条 lineage，不进 campaign）。
- 注入边界（折中版，已拍板）: 题面允许含 (a) 实测死路清单 (b) 正确性陷阱
  (c) REPORT.html 已发布事实（含 §9e phase breakdown）；禁止含 champion/compA
  的解法方向、op35 oracle 结论、R3 harvest 源码。§B 的 prompt.md v3 已按
  此边界写好，替换 ws/prompt.md 后再 init，不要往里加料。
- 工装全部可复用（工装≠解法，不受冷启动限制），位于
  /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/
  indexer_topk_op_bench/op26_r0_upstream_port_report/kf_campaign/：
  ws/{definition.json,workload.jsonl,campaign.yaml}、gvrpkg_head/（旧 head
  独立打包，作 re-port 底座）、export_cells.py、monitor_campaign.sh、
  quick_ab.py、nsys_ab.py、run_nsys_ab.sh、drive_grid_shards.sh、
  aggregate_grid.py、compare_rivals.py。先通读 kf_campaign/README.md。
  注意 ws/baselines.jsonl 是旧 head/旧节点数据，本期作废，由 prepare 现测覆盖。

## 1. 分母 pin 与验收标准（三条全要）
- 分母 pin: 开跑前 `git fetch` PR#16457 当前 head（fetch
  refs/pull/16457/head），记录 commit SHA 入台账；本期所有 Bar 的基线
  = 该 SHA 的 GVR kernel，在判决节点同 session 现测。
  禁止引用 REPORT.html 冻结数字做分母（跨节点/跨时间无效）；
  REPORT §7b 只定义 cell 集合与测量口径。head 若在战役中途再动，
  分母不追新 — 一期一 pin。
- Bar-1（geomean +60%）: 865 个真实 cell 上，优化后 kernel vs pinned head
  的 nsys 纯 kernel 时间 geomean ≥ 1.60×。
  期望管理（如实入台账，不改 bar）: 第一期从旧 head 花 $690/2 轮达 1.68×，
  且当前 head 已含 07-20 kb512@K2048 等改进，1.60× vs 当前 head 可能贴近
  结构墙；战役以尽力推高为实操目标，验收报告如实给出达成与否。
- Bar-2（0 回退）: 同一 865 格中任何单个 cell，优化后 vs pinned head
  （同 session 同 GPU 配对测量）不得性能回退。
  边界 cell（±2% 以内）允许 60-reps 配对复测裁定为噪声；裁定记录入台账。
- Bar-3（exact）: 865/865 cell 输出满足 tie-robust index-set 语义 —
  值严格大于第 k 名的索引必须全部出现，第 k 名边界并列值可任取，
  索引不要求保序（ws/definition.json 的 check_topk 已实现，直接复用，
  不要放宽也不要收紧）。注意 kernel 输出是 indices，"value 完全相同"
  的口语表述以 check_topk 为准。

## 2. 测量纪律（硬约束，违反即判决作废）
- B200；L2-flushed cold data（512MB evict，nsys_ab.py 内置）；
  最终验收一律 nsys 纯 kernel 时间；CUDA-event 计时只做轮内粗筛。
- 平台内测试数据 = 28-cell 分层子采样（--asset 总量 ≤500MiB，865 全格
  放不进平台）；865 全格判决只在本地做。KF 平台内部 speedup 有 ~15µs
  eval 地板，系统性低估本地 nsys 比值 ~1.3-1.4×（第一期实测: 平台 1.34
  时本地已 1.67）。平台指标只用于轮内相对排序，一切收割/ship 判断
  以本地 nsys 为准。
- 全格 grid 运行期间禁止任何探针作业（no-probes-during-grid — 第一期
  两次 verdict 因 double-driver 污染作废）；每 rung 做 pr_cold 锚漂移检查；
  ≤2 并发 nsys；长跑一律 setsid；判决前确认 GPU 空闲。
- 8 卡机器上全格判决用 drive_grid_shards.sh 8-shard（~17 min/格）+
  aggregate_grid.py；单卡 28-cell 探针用 run_nsys_ab.sh。
- 环境: 若 harness import 报 cutlass make_fragment 错 → /tmp/gvrlayers
  overlay 被清，按 kf_campaign/README.md 的一行命令从 NFS userbase 重建；
  新容器需 PYTHONNOUSERSITE=1 + machine-local cutlass 4.5.0。
- nsys sqlite 会内嵌环境 token: profiling 一律 `env -u GITHUB_TOKEN -u HF_TOKEN`，
  *.sqlite/*.nsys-rep 永不入 git。

## 3. 算法约束（骨架锁死 — 本期强化条款）
- 必须保留 GVR 骨架三要素: (a) 上一步 top-K (preIdx) 作阈值先验；
  (b) secant + log 变换求解最佳阈值（或等价的阈值求解结构）；
  (c) 对剩余 candidates refine 出 exact top-K。
- **骨架锁死（用户 2026-07-22 强化）**: 目标是优化 GVR 本身，不是换算法。
  任何放弃 preIdx 先验、或把阈值先验结构整体替换为 prior-free 全刷算法
  （纯 radix-select、全排序、采样式 selection 等）的方案，**即使更快也
  判不合规**，收割时直接拒绝；§B 题面与 compliance 判官口径同此。
- 各子阶段（尤其 P1/P4）允许等价重构，前提是 exactness 与骨架三要素不破。
  子阶段可吸收成熟原语（histogram ladder / radix digit pass 等），
  但禁止 per-case dispatch 到其他完整 top-K 算子。
- 禁止依据 hit-rate 做 dispatch（推理时不可知）；允许 kernel 内 admission
  逃逸或滞后命中反馈。
- CUDA graphs / replay 摊销不算 kernel 收益；framework kernel import
  （flashinfer/trtllm 等）不算 —— 两者都是平台判官默认 ban，
  不要传 --allow-*。

## 4. case 重要性（用于子采样加权与 steering；验收仍是 865 格等权 geomean）
- DSV4 Pro (K=1024) > DSV4 Flash (K=512) ≈ V3.2 (K=2048)；ISL 32K–1M > 4K–32K。
- 现有 28-cell 子采样已含 512K/1M rungs 且按 (model×ISL×hint 高/低) 分层，
  结构直接复用；如加权，优先加 Pro 大 ISL cell，总 asset ≤500MiB。

## 5. 执行流程
0) 分母 pin + baseline 打包 + parity gate:
   a. fetch PR#16457 当前 head，记录 SHA；
   b. 以 gvrpkg_head/ 为底座 re-port 旧 head → 当前 head 的增量
      （至少含 07-20 kb512@K2048；逐 commit 对照 PR 页面）
      → pr_head_solution.json；
   c. parity gate: 28-cell nsys 探针，独立打包 vs in-tree head 同 GPU 配对，
      |geomean 漂移| ≤2% 才准用作 KF baseline-solution 与本地分母，
      否则先修打包再开跑。
1) `kf auth status` / `kf status` 核查 → 先看 `kf campaign init --help` 与
   `kf campaign --help` 探测 lifecycle（prepare 型 vs legacy
   --reference-benchmark 型，两套不能混）→
   `kf campaign init gvr-topk-cold60 --definition ws/definition.json
   --workloads ws/workload.jsonl --baseline-solution pr_head_solution.json
   --gpu-spec b200 --language cuda_cpp_only --effort high
   --prompt-file ws/prompt.md`（prompt.md 先换成 §B v3-coldstart）
   → `prepare`（现测 pinned-head 基线）→ `start`。
2) 同时在判决节点用 drive_grid_shards.sh 现测 pinned head 的 865 格
   本地基线（Bar-1/Bar-2 的分母）。
3) monitor_campaign.sh 轮次监控；每轮收割: kernel list → harvest 源码 →
   合规初筛（骨架三要素在不在，见 §3）→ 本地 28-cell nsys 探针 →
   过线者全 865 格判决 → 判决结论喂回 campaign steering
   （insights / fork --append-prompt）。
4) kill 线: 连续 2 轮无候选超过现任最佳，或平台花费超 $800 → 收口。
   收口后（campaign 外）与第一期 champion/compA 做一次横向对照入台账。
5) 交付: 最佳 kernel 代码推 fork code-only 分支（只 kernel.cu/main.cpp/
   README，报告不上 GitHub）；EXPANDED_VS_PR 式 per-layer 报告；
   过程台账（KF_PROCESS_LOG 新 R4 章节）；成本核算。
6) 分析工作随时 checkpoint 落盘（脚本固化 .py、结论写 md、及时 commit），
   不等任务完成。
```

---

## §B Campaign `prompt.md` v3-coldstart（替换 ws/prompt.md；英文，面向 KF agent）

````markdown
# DeepSeek-V4 Indexer Top-K Decode (BS=1, fp32, B200) — Optimize the Production GVR Kernel

## Problem

Sparse-attention indexer top-K selection at decode time. One row of real
captured indexer logits (`logits[1, npad]`, fp32, valid length `n_valid`,
tail padded so pad never enters the top-k). Return the `int32` indices of
the `k` largest values, any order; ties at the k-th value boundary may be
resolved either way (the correctness checker is index-SET based and
tie-robust). Exactness is non-negotiable: every index whose value is
strictly greater than the k-th value must appear, on every run.

`pre_idx[1, k]` is the PREVIOUS decode step's top-k (temporal warm hint).
Overlap with the true top-k ranges 0.02–1.0 across workloads (typically
>0.5). Exploiting it is REQUIRED (see skeleton below), but correctness and
the no-regression bar must hold even at 0.02 overlap. You may NOT branch
on any estimate of hint quality computed outside the kernel (hit-rate is
unknowable at inference); in-kernel admission escape / lagged feedback is
fine.

Workloads are REAL production captures from three models, n up to ~1.05M:
- V4-Flash: k=512,  n rungs 4K / 32K / 128K / 512K / 1M
- V4-Pro:   k=1024, n rungs 4K / 32K / 128K / 512K / 1M  (highest priority)
- V3.2:     k=2048, n rungs 4K / 32K / 128K / 256K

Two workloads per (model, rung): a low-hint-overlap layer and a
high-overlap layer. The logits distribution is NOT random — heavy-tailed
real indexer scores (near-exponential CCDF); algorithms that look good on
`randn` behave differently here. Priority for effort allocation:
V4-Pro > others; n ≥ 32K > small n.

## Baseline

The baseline solution you are given is the CURRENT PRODUCTION kernel: the
guess-verify-refine (GVR) top-K from TensorRT-LLM PR#16457 (latest head,
including its K=2048 tail-ladder tuning). Its structure: seed a threshold
guess from `pre_idx`, verify/refine the threshold with a secant solve in
log space, then exactly collect the surviving candidates. Read its full
source first — your job is to make THIS kernel faster, not to replace it.
Incremental surgery is the expected strategy; you must find the profitable
directions yourself by profiling and analysis.

One measured fact about where its time goes (from the external report on
the full 865-cell grid): the FINAL-COLLECT block (threshold handoff +
refine + writeback, "P4") dominates — it is the largest phase on 827 of
865 cells, median ~37% of kernel time (range 23–58%). The mid scan/count
passes are second. How to attack that is yours to work out.

## Target

- **Required:** geomean speedup > 1.0× over the given baseline with NO
  workload slower (no-regression is a hard acceptance bar — a kernel that
  wins big on average but loses any cell will be rejected downstream).
  The external acceptance goal is +60% geomean on the full 865-cell grid;
  every incremental win counts toward it.
- Final acceptance re-measures externally with nsys cold-L2 on all 865
  real cells (the cells here are a stratified subset). Platform timings
  have a ~15µs floor that compresses your true speedup roughly 1.3–1.4× —
  do not tune to the harness floor; win in kernel time. Do not overfit to
  these exact n values: `n` is dynamic (up to ~1.05M), `k ∈ {512, 1024,
  2048}` at runtime, hint quality is dynamic.

## Required algorithmic skeleton — HARD compliance rule

Keep the GVR skeleton: (a) `pre_idx` as the threshold prior, (b) a
secant+log-transform style exact threshold solve (or an equivalent
threshold-refinement structure), (c) an exact refine of the surviving
candidates. Any per-stage restructuring that preserves exactness is
allowed. Mature primitives (histogram ladders, radix digit passes) may be
absorbed INTO stages.

**Non-negotiable:** a submission that abandons the `pre_idx` prior, or
replaces the threshold-prior structure wholesale with a prior-free
selection algorithm (plain radix-select, full sort, sampling-based
selection), is NON-COMPLIANT and will be rejected even if it is faster.
Likewise, do not build a per-case dispatcher across unrelated top-k
operators. The goal of this campaign is a better GVR, not a different
algorithm.

## Dead ends — measured net-negative on THIS workload/hardware; do not re-discover

1. BS=1 is latency-bound, not bandwidth-bound (24% occupancy, <1% DRAM at
   small n). Bandwidth-oriented rewrites miss the bottleneck.
2. `pre_idx` warm-hint grafted onto radix-select: no win (hint only helps
   threshold-style skeletons) — and prior-free pivots are banned anyway.
3. Private per-warp histograms to avoid smem atomics: loses (SM100
   pipelines same-address atomics fine).
4. Multi-CTA for SMALL n (< ~8K): launch/sync overhead dominates;
   single-CTA wins there.
5. More than 8–32 CTAs at large n: merge cost eats the scan win.
6. Per-element ballot/popc slot-reservation to fuse count+collect into
   one pass: coordination ≈ a full extra pass.
7. Staging the row into shared memory first: row re-reads are cheap L2
   hits.
8. Extra secant/interpolation refinement rounds: each is a
   barrier-separated pass; keep passes ≤2.
9. CUB DeviceRadixSort / full sort: ~10× too slow at these sizes (and
   banned as a wholesale replacement).
10. Fusing the final-collect histogram into the P3 scan loop: pollutes
    the scan inner loop, −15%.
11. Shrinking histogram bins below 512 for k=2048: exact-tail scratch
    overflows (silent UB); kNumBins=512 at k=2048 is already in the
    baseline.
12. Launch-config-only retuning: ceiling measured at ~1.025×.
13. CUDA graphs / replay amortization are banned by the compliance judge —
    win inside the kernel.

## Correctness traps

- The k-th-value tie boundary: the checker requires ALL indices with
  value strictly greater than the k-th value, plus any tie subset to fill
  the remainder. Arrival-order races on the boundary bin under concurrent
  compaction are the classic silent bug — never drop a strictly-greater
  element.
- Real data is UNDERSHOOT-biased for hint-seeded thresholds (the seeded
  count almost always comes in below k, not above): guards that only fire
  on overshoot are dead code here.
- On cluster launches, `cluster.arrive_relaxed()` has no release
  semantics: a DSMEM read of a just-written scalar can observe stale
  data. Use `cluster_arrive()` (release) or an acq_rel cluster fence on
  the write side. Symptom: wrong indices clustered by CTA slice.

## Requirements

- CUDA C++ (sm_100a Blackwell). fp32 in, int32 indices out.
- Exact per the tie-robust set semantics above — no approximation.
- Dynamic `n` (up to ~1.05M, padded width `npad = ceil(n/64)*64`), dynamic
  hint quality, `k ∈ {512, 1024, 2048}` at runtime.
- Deterministic output not required (any tie resolution accepted), but
  the index set must be exactly right on every run.
- One kernel launch preferred (or 2 with programmatic dependent launch);
  launch overhead is material at 3–29 µs.
````
