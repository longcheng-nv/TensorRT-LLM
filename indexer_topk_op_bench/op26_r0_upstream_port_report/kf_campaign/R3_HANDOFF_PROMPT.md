# R3 handoff — GVR Top-K KernelFactory 第二期战役（beyond-champion）

Prepared 2026-07-21 by the analysis session on branch `omni/op21-gvr-prod`.
两级 prompt：§A = operator /goal（贴进新 8×B200 机器的 Claude Code 会话），
§B = campaign `prompt.md` v2（KF agent 看到的，替换 `ws/prompt.md`）。
背景判定见本目录 `KF_PROCESS_LOG.html` 与 `README.md`。

---

## §A Operator /goal prompt（贴入新会话）

```
/goal — GVR Top-K KernelFactory 第二期战役：在既有 champion 之上继续推高，并完成 +60% 正式验收

## 0. 背景与已有资产（先读，不要重复花钱）
- 优化对象: TensorRT-LLM PR#16457 的 GVR (guess-verify-refine) top-K kernel，
  DeepSeek 稀疏注意力 indexer decode 场景，BS=1，fp32 logits，
  三类模型 K=512 (V4-Flash) / 1024 (V4-Pro) / 2048 (V3.2)，ISL 4K–1M，全 layers，
  共 865 个真实采集 cell（口径 = op26_r0_upstream_port_report/REPORT.html §7b）。
- 第一期 KF 战役 `gvr-topk-bs1-real` (tfb91bvwm972kfyf1bc1trj5e0) 已于 2026-07-21 收口
  （$690.81，2 rounds / 13 agents，round-2 内部 plateau 后取消）。
  champion = round-2 冠军 c74fb3c0 (a003: all-coop ≥8448 / per-pass early-exit /
  bottom3 complement kernel) + 工程师 graft (topk_small<17><<<1,1024>>> rung,
  8448<n≤16896)。本地 nsys 全格判决: geomean 1.6828× vs 当时 PR head
  (e6fdbfac3d)，865/865 exact，0 cold regression。
  代码在 fork 分支 `kf/gvr-topk-c74fsbx` @193156c8；
  本地快照 = kf_campaign/harvest/r2_c74f_sbx/。
- 全部 campaign 输入与判决 harness 可直接复用，位于
  /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/
  indexer_topk_op_bench/op26_r0_upstream_port_report/kf_campaign/：
  ws/{definition.json,workload.jsonl,baselines.jsonl,prompt.md,campaign.yaml}、
  export_cells.py（865 cell → per-cell safetensors + 28-cell 分层子采样）、
  monitor_campaign.sh、quick_ab.py、nsys_ab.py、run_nsys_ab.sh、
  drive_grid_shards.sh、aggregate_grid.py、compare_rivals.py。
  先通读 kf_campaign/README.md（code map + 实际工作路径 + 测量纪律）。
- 本期任务 = 以 champion 为 KF baseline-solution 再开一轮战役继续推高；
  无论新轮是否有收获，最终验收对象 = 届时最佳复合 kernel（champion 或其后继）。

## 1. 验收标准（双基线，三条全要）
- Bar-1（geomean +60%）: 865 个真实 cell 上，优化后 kernel vs 基线 GVR 的
  nsys 纯 kernel 时间 geomean ≥ 1.60×。
  基线 = PR#16457 当前 head 的 GVR kernel（含 07-20 已落地的 kb512@K2048），
  在判决节点同 session 现测。禁止引用 REPORT.html 的冻结数字做分母
  （跨节点/跨时间比较无效）；REPORT §7b 只定义 cell 集合与测量口径。
- Bar-2（0 回退）: 同一 865 格中任何单个 cell，优化后 vs 基线 PR#16457
  （当前 head，同 session 同 GPU 配对测量）不得性能回退。
  边界 cell（±2% 以内）允许 60-reps 配对复测裁定为噪声；裁定记录入台账。
- Bar-3（exact）: 865/865 cell 输出与 torch.topk 值语义完全一致 —
  值严格大于第 k 名的索引必须全部出现，第 k 名边界的并列值可任取，
  索引不要求保序（ws/definition.json 的 check_topk 已实现该 tie-robust set 语义，
  直接复用，不要放宽也不要收紧）。

## 2. 测量纪律（硬约束，违反即判决作废）
- B200；L2-flushed cold data（512MB evict，nsys_ab.py 内置）；
  最终验收一律 nsys 纯 kernel 时间；CUDA-event 计时只做轮内粗筛。
- KF 平台内部 speedup 有 ~15µs eval 地板，系统性低估本地 nsys 比值 ~1.3-1.4×
  （第一期实测: 平台 plateau 1.34 时本地已 1.67）。平台指标只用于轮内相对排序，
  一切收割/ship 判断以本地 nsys 为准。
- 全格 grid 运行期间禁止任何探针作业（no-probes-during-grid — 第一期两次
  verdict 因 double-driver 污染作废）；每 rung 做 pr_cold 锚漂移检查；
  ≤2 并发 nsys；长跑一律 setsid；判决前确认 GPU 空闲。
- 8 卡机器上全格判决用 drive_grid_shards.sh 8-shard（~17 min/格）+
  aggregate_grid.py；单卡 28-cell 探针用 run_nsys_ab.sh。
- 新机器环境: 若 harness import 报 cutlass make_fragment 错 → /tmp/gvrlayers
  overlay 被清，按 kf_campaign/README.md 的一行命令从 NFS userbase 重建；
  新容器需 PYTHONNOUSERSITE=1 + machine-local cutlass 4.5.0。
- nsys sqlite 会内嵌环境 token: profiling 一律 `env -u GITHUB_TOKEN -u HF_TOKEN`，
  *.sqlite/*.nsys-rep 永不入 git。

## 3. 算法约束
- 保留 GVR 骨架三要素: (a) 上一步 top-K (preIdx) 作阈值先验；
  (b) secant + log 变换求解最佳阈值（或等价的阈值求解结构）；
  (c) 对剩余 candidates refine 出 exact top-K。
- 各子阶段（尤其 P1/P4）允许等价重构。已裁定合规的先例: all-coop 多 CTA 协作、
  per-pass early-exit、bottom-k complement kernel、按 n 分档的 launch-config rung。
- 禁止 per-case dispatch 到其他完整 top-K 算子（radix-select 全刷等）；
  但子阶段可吸收成熟原语（histogram ladder / radix digit pass 等）。
- 禁止依据 hit-rate 做 dispatch（推理时不可知）；允许 kernel 内 admission
  逃逸或滞后命中反馈。
- CUDA graphs / replay 摊销不算 kernel 收益（合规判定会拒绝）。

## 4. case 重要性（用于子采样加权与 steering；验收仍是全格等权 geomean）
- DSV4 Pro (K=1024) > DSV4 Flash (K=512) ≈ V3.2 (K=2048)；ISL 32K–1M > 4K–32K。
- 现有 28-cell 子采样已含 512K/1M rungs 且按 (model×ISL×hint 高/低) 分层，
  结构可直接复用；如加权，优先加 Pro 大 ISL cell，总 asset ≤500MiB。

## 5. 执行流程
1) `kf auth status` / `kf status` 核查 → 将 champion 打包为 SOLBench
   solution.json（源 = harvest/r2_c74f_sbx 或 fork 分支 kf/gvr-topk-c74fsbx）→
   `kf campaign init gvr-topk-r3 --definition ... --workloads ...
   --baseline-solution champion_solution.json --gpu-spec b200
   --language cuda_cpp --effort high` → `prepare`（现测 champion 基线）→ `start`。
   注意先按 skill 指引探测 CLI 的 lifecycle（prepare 是否存在）。
2) 同时在判决节点现测 PR#16457 当前 head 的 865 格基线（Bar-1/Bar-2 的分母），
   并复测 champion 全格（确认新节点/新 head 下的起点 geomean）。
3) monitor_campaign.sh 轮次监控；每轮收割: kernel list → harvest 源码 →
   本地 28-cell nsys 探针 → 过线者全 865 格判决 → 判决结论喂回
   campaign steering（insights / prompt 更新）。
4) kill 线: 连续一整轮无候选超过现任 composite，或平台花费超 $800 → 收口。
5) 交付: champion 代码推 fork code-only 分支（只 kernel.cu/main.cpp/README，
   报告不上 GitHub）；EXPANDED_VS_PR 式 per-layer 报告；过程台账
   （KF_PROCESS_LOG 续写或新 R3 章节）；成本核算。
6) 分析工作随时 checkpoint 落盘（脚本固化 .py、结论写 md、及时 commit），
   不等任务完成。

## 6. 已知矛头与死路（喂给 agent steering，防止重走）
- 剩余矛头（op35 nsys/NCU oracle）: distP4 — P4blk (handoff2+P4+writeback)
  中位 ~37% (23-58%) 是全场最大块，kill handoff2 value-ship + leader P4
  跨 cluster CTA 并行；warp0-ized P4 搜索（省 2-3 个 barrier）；
  (warp,window) B1 sideband（仅 512K-1M）。
- 结构参照: UB(zero P3+P4blk) = 1.771 geomean（op35 松弛控制实验）。
  现任 1.68 离墙 ~5%；对本期的合理期望 = 再 +5-15%，不是再 +60%。
  Bar-1 的 1.60 主要靠 champion 已达成，本期价值在 Bar-2 复核 + 增量推高。
- 死路清单: ws/prompt.md §"Hard-won structural knowledge" 已含 June 史 +
  第一期 round-1 重走的共识死路（pre_idx warm-hint on radix、private
  histograms vs smem atomics、multi-CTA small-n、>8-32 CTAs large-n、
  ballot/popc 单遍融合、smem staging、多轮 secant pass、CUB full sort）。
  用 §B 的 prompt.md v2（本文件）替换后再 init。
```

---

## §B Campaign `prompt.md` v2（替换 ws/prompt.md；英文，面向 KF agent）

````markdown
# DeepSeek-V4 Indexer Top-K Decode (BS=1, fp32, B200) — Beat the Round-2 Champion

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
>0.5). Exploit it (e.g. threshold seeding), but correctness and the
no-regression bar must hold even at 0.02 overlap. You may NOT branch on
any estimate of hint quality computed outside the kernel (hit-rate is
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

## Baseline — this is the hard part

The baseline you must beat is NOT the original production kernel. It is
the ROUND-2 CHAMPION of the previous campaign on this exact problem: a
guess-verify-refine (GVR) design with all-CTA cooperative scanning for
n ≥ 8448, per-pass early-exit, a bottom-k complement kernel for high-hint
rows, a dedicated `<<<1,1024>>>` single-CTA rung for 8448 < n ≤ 16896, and
secant+log threshold refinement seeded from `pre_idx`. It already runs
1.68× faster (external nsys, cold-L2) than the production kernel it
replaced, is exact on all 865 external cells, and regresses none of them.
Its full source is provided as the baseline solution — read it first;
incremental surgery on it is a legitimate strategy.

## Target

- **Required:** geomean speedup > 1.0× over the given champion baselines
  with NO workload slower (no-regression is a hard acceptance bar — a
  kernel that wins big on average but loses any cell will be rejected
  downstream). Meaningful wins are +5–15% geomean; single-cell heroics
  that lose elsewhere are worthless.
- Final acceptance re-measures externally with nsys cold-L2 on all 865
  real cells (the cells here are a stratified subset). Platform timings
  have a ~15µs floor that compresses your true speedup — do not tune to
  the harness floor; win in kernel time. Do not overfit to these exact n
  values: `n` is dynamic (up to ~1.05M), `k ∈ {512, 1024, 2048}` at
  runtime, hint quality is dynamic.

## Required algorithmic skeleton

Keep the GVR skeleton: (a) `pre_idx` as the threshold prior, (b) a
secant+log-transform style exact threshold solve (or an equivalent
threshold-refinement structure), (c) an exact refine of the surviving
candidates. Any per-stage restructuring that preserves exactness is
allowed (the champion's all-coop scan, early-exit, and complement kernel
are precedents). Mature primitives (histogram ladders, radix digit
passes) may be absorbed INTO stages, but do not replace the whole kernel
with a generic radix-select / full-sort top-k and do not build a
per-case dispatcher across unrelated top-k operators.

## Where the remaining time is (measured, nsys/NCU — start here)

1. **P4 block (final collect: handoff + refine + writeback) is the
   dominant cost: median ~37% of kernel time (23–58%) across the grid.**
   Known-untried levers: eliminate the leader-CTA value-handoff and
   parallelize the final collect across cluster CTAs ("distP4");
   confine the P4 threshold search to warp0 to remove 2–3 block-wide
   barriers.
2. Mid passes (scan + count + falsification) are 17–48%; per-pass
   early-exit already harvests much of this in the champion.
3. For n ≥ 512K only: a (warp, window) sideband that lets the scan skip
   provably sub-threshold windows may pay; it measured ~0 replay benefit
   at smaller n.

## Dead ends — measured net-negative on THIS workload/hardware; do not re-discover

1. BS=1 is latency-bound, not bandwidth-bound (24% occupancy, <1% DRAM at
   small n). The multi-CTA cooperation lever is ALREADY in the champion —
   re-deriving it is not progress.
2. `pre_idx` warm-hint grafted onto radix-select: no win (hint only helps
   threshold-style skeletons).
3. Private per-warp histograms to avoid smem atomics: loses (SM100
   pipelines same-address atomics fine).
4. Multi-CTA for SMALL n (< ~8K): launch/sync overhead dominates; the
   champion's single-CTA rungs win there.
5. More than 8–32 CTAs at large n: merge cost eats the scan win.
6. Per-element ballot/popc slot-reservation to fuse count+collect into
   one pass: coordination ≈ a full extra pass.
7. Staging the row into shared memory first: row re-reads are cheap L2
   hits.
8. Extra secant/interpolation refinement rounds: each is a
   barrier-separated pass; keep passes ≤2.
9. CUB DeviceRadixSort / full sort: ~10× too slow at these sizes.
10. Fusing the final-collect histogram into the P3 scan loop: pollutes
    the scan inner loop, −15%.
11. Shrinking histogram bins below 512 for k=2048: exact-tail scratch
    overflows (silent UB); kNumBins=512 at k=2048 is already in the
    production baseline.
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
