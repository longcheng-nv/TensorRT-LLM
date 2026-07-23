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

---

## 2026-07-22 T2 进展(执行机器 = umbriel-b200-027, 8×B200 空闲)

- 疑难#1 解决: `gvrpkg_head/`、`gvrpkg_e6fd/` 在 NFS 上均存在(上一台机器 find 误报)。
  gvrpkg_head 实为 in-tree @6140078816 + ruff 格式化(残差仅格式);距 pinned head
  只差 a707cfe41c(LB-hybrid SMEM 均衡修复, kc_diet 参数)。
- **T0 DONE**: `gvrpkg_04a0/` = pinned head 04a0900ff7 in-tree GVR 文件逐字节原样。
  `build_pr_head_solution.py` → `pr_head_solution.json`(cute_dsl 语言, main.py::run
  DPS 入口, cr = 1 if k==2048 else 4, seq_lens = n*cr, 与 quick_ab.pr_call 同义)。
  冒烟 exact: flash_4k_L24 / v32_256k_L12 / pro_1024k_L40 / flash_1024k_L36 全 True。
- harness 包路径参数化: nsys_ab.py / quick_ab.py 读 `GVRPKG_DIR`(默认改为
  **gvrpkg_04a0** — R4 起 gvr_pr 臂 = pinned head;旧 tag 已归档不受影响)。
- ws/prompt.md 已替换为 §B v3-coldstart(旧版备份 ws/prompt_r3.md.bak)。
- T1 parity gate IN-FLIGHT: GPU6 背靠背双跑 tag=parity04a0/parityhead(arms=gvr_pr,
  28 cells)→ 每 cell cold 中位配对, |gm 漂移|≤2% 过线。两包同名不可同进程,
  故取同 GPU 连续两次 nsys 配对(机器空闲)。日志 parity_r4.log。
- **prepare 失败→D1 复现**: 平台 baseline evaluator 不 stage 资产(0/28
  safetensors missing,与 R3 D1 同缺口)。变通(campaign-1 先例):
  `gen_r4_baselines.py` → baselines.jsonl = parity04a0 冷中位(28 格,
  pinned head 本地 nsys 纯 kernel 时间,平台 geomean 口径 14.30µs);
  campaign.yaml 切 baselines 路径,baseline_solution: null。
- prompt 上限 32768B → `gen_prompt_appendix.py` 生成基线源码 digest
  (GvrParams+pick_config 全文 + 全 phase 签名/docstring,19.9KB),
  §Baseline 段加计时尺度 steering(外部纯 kernel 分母 → 平台平价读数
  ~0.5-0.9x,勿灰心);最终 prompt 27.7KB。
- **T2 DONE — campaign STARTED**: `pra6srbd7h4pqecqbgxgm15rgg`
  (gvr-topk-cold60, 2026-07-22T02:48Z, effort high, max $800/8h)。
- **T3 IN-FLIGHT**: 865 格 pinned-head 分母 grid tag=r4pr;外来作业占
  GPU0-3(62-82% util,19-75GB,R3 D3 同款)→ 新驱动
  drive_grid_pr_gpulist.sh 只用安静 GPU4-7,4 shards。
- 监控:monitor_campaign.sh 后台轮询(round advance / +0.02 / terminal 退出)。
- **事故记录(r4pr 作废)**: 第一次 8-shard 启动被误判为失败(tail 时序竞态,
  grid_logs 实为 R3 遗留已存在,setsid 作业实际已跑起来占满 GPU0-7);
  其 GPU0-3 负载被误读为"外来作业",随即又叠发 4-shard gpulist 同 tag 同
  rep 文件 → 双 driver + last-writer 混写,r4pr 全部作废并隔离至
  nsys_reps/INVALID_r4pr/。教训:relaunch 前必须 pgrep+GPU util 双确认
  上一发真死;同 tag 严禁二次发射。另确认 GPU0/2/3 确有真实外来负载
  (清场后仍 66-85% util / 56-74GB 常驻)→ 分母 grid 限 GPU4-7。
- r4pr2 重跑 IN-FLIGHT: 4 shards,GPU4-7,launch 后 pgrep 核对 = 恰好
  4 nsys + 4 python,单 driver 洁净。
- **T3 DONE (r4pr2)**: 865 格 pinned-head 分母 grid,GPU4-7 单 driver,
  **865/865 exact**;grid_r4pr2.csv(pr_cold/pr_warm)。同节点锚:与
  parity04a0 28 格重叠 gm=1.0079(0.8%,±尾 0.881/1.092 为小 N 冷噪声);
  跨节点 R3 锚不可用(既有教训)。此 grid 即本节点后续所有 rung 锚基准。
- **T4 首收割 r1_a275c747**(round1 内部 0.9188,claude-fable agent,
  `gvr_topk_cuda_v3`): 骨架合规 PASS(P1 hint-CCDF rung ladder / P2 8-rung
  多阈值+log-secant / P4 candidates 内 radix 精修+tie-ticket,dispatch 仅按
  npad 1/4/8 CTA)。28-cell 探针(GPU6): **cold gm 1.1867, 28/28 exact,
  6/28 回退** — 回退全在 N≈16K-33K 中段带(flash/pro_128k 0.83-0.95,
  32k 部分 0.93-0.98),与 CTA 档位切换带重合(第一期 r1 同型缺陷);
  4K 大胜 1.15-1.86,≥512K 稳胜 1.08-1.45。未过 Bar-2,不上全格;
  等 round 演进。注:平台 insights 只读,mid-campaign 无 operator steering
  通道,判决材料留作收口后 fork --append-prompt。
- **r1_35721475 收割**(`gvr_topk_cuda_v5_cs16`, 内部 1.0333): 同骨架,
  dispatch 重调(1CTA<16K / 4<32K / 8<131K / 16≥131K)。28-cell 探针:
  **cold gm 1.3490, 28/28 exact, 1 边界回退**(flash_32k_L16 0.977,
  待 60-rep 裁定)。过线 → 865 全格判决。
- 全格首发 r4v5g 作废重发: 启动后发现外来 171GB 常驻已迁至 GPU4(0-3 反而
  清空)→ pkill 三连清场 + 部分 rep 隔离 INVALID_r4v5g/,新 tag r4v5g2
  在 GPU5,6,7 3-shard 重发(launch 后核对恰 3 nsys+3 python)。
  本机外来负载会跳卡,每次 launch 前后必须双查。
- **v5_cs16 全格 865 判决(r4v5g2, GPU5-7)**: **geomean 1.2954,865/865
  exact,78 回退(min 0.851, p5 0.968)**。回退带 = N≈8K-16K(v32_8k 簇
  0.889-0.913、N=8195 的 flash/pro 0.915-0.930、N=16387 0.908-0.922)——
  v3 的 32K 弱带被 v5 dispatch 修复后弱带下移到 8-16K 单/4-CTA 过渡区。
  锚检查: 25 rung 中 4 个 med 漂移 3.4-7.3%(跨 GPU 集合可解释;配对比值
  不受影响;ship 级判决需漂移补测)。vs Bar: Bar-3 PASS / Bar-1 1.295<1.60 /
  Bar-2 FAIL(78)。round 1 仍在演进,继续监控。
- **r1_bbe9b903 收割**(`gvr_topk_cuda_v10_tbtier`, 内部 1.0727): TB 模板化 +
  npad≤kC 加 1024 线程单 CTA 档(独立重发现第一期 sb17 方向)。28-cell 探针:
  **cold gm 1.3771, 28/28 exact, 1 边界**(flash_32k_L16 0.985)。v32_4k 大幅
  改善(1.151→1.463)。8-16K 弱带 dispatch 未动,预计仍在。
- 环境: 外来 ~168GB 常驻显存已扩散至 GPU4/6/7(util 0%,纯 parked);
  GPU5 唯一全净。v10 探针在 GPU6 带 parked 内存下跑,pr 臂读数与干净跑
  一致(噪声内),结果采信但注记。
- 策略: round 内每个新 best 仅探针;全格判决留 round 收口/平台期
  (全格 ~30min,round 内 best 每 ~30min 在涨,逐一全格浪费)。
- **r1_a4e07868 收割**(`gvr_topk_cuda_v14`, 内部 1.0873): kC 6144→8192
  (接受窗更宽→pass 更少)+ count 归约改 warp-per-rung(去 atomic)。
  GPU5 首探针作废(外来 67GB→95%util 跳卡入 5,pr 臂膨胀 25-35%,
  产物隔离 INVALID_r4v5g/)。GPU1 净卡复测: **cold gm 1.3871, 28/28
  exact, 0 回退** — 首个零回退候选。全格 r4v14g 已发 GPU1,2,3。
- 环境更新: 外来作业现 95-100% util 压满 GPU0/4/5/6/7,净卡仅 1/2/3。
- **v14 全格 865 判决(r4v14g, GPU1-3)**: **geomean 1.3428, 865/865 exact,
  24 回退(v5 是 78;min 0.834, p5 1.021)**。弱带仍 = N=8195/16387-16399
  簇(hit 低格更痛,pro_64k_L38 hit0.27 → 0.834)+ 孤立 flash_4k_L06 0.914。
  锚: 整体 pr_cold gm 0.9864(GPU1-3 vs 分母集 4-7 的集合差),5 rung
  3-5% 漂移,配对比值有效,中期判决采信。vs Bar: 1.343/1.60, 24 regs。

## Round 1 收口(06:31-07:0x UTC 前后,~4h)

- 终局: 内部 best 1.0873 (`v14`, a4e07868);6 agents 全结束;累计花费 **$352/800**。
- 本地判决链: v3 1.187(6reg) → v5 1.349(1bl)/全格 1.2954(78reg) →
  v10 1.377(1bl) → **v14 1.387(0reg)/全格 1.3428(24reg, 865 exact)**。
- Round-1 insights(24 条)要点: 集群 8-16 CTA 在 n≥32K 有效;warp 层级
  scan 减 barrier;k-th bin 搜索并行化(串行 thread-0 是延迟地板);
  count_ge(max_hint) 在首 pass 免费捎带;低 overlap 时 escalate 到 max
  hinted key;cp.async 流水在此负载亏(barrier>隐藏);全行 radix 直方图
  输给 secant 计数(prior-free 死路自证);弱 ld.global.cg 自旋会读stale。
- 判官注意: 有 agent 记录 "compliance judge 按源码 hash 缓存,可用微小
  编辑绕过 buggy 判决" — 收割时人工复核合规不可省。

## Round 2

- **r2_be2d8f30 收割**(`gvr_topk_cuda_r2_wider_direct`, 内部 1.1513, r002-a002):
  KCMAX 8192→8448(盖住 N=8195→npad 8256 的整个弱带 rung)→ 8K 带走整行
  直通道;kC 分层 (K≥2048→8192, 否则 6144)。28-cell 探针(GPU1):
  **cold gm 1.5106, 0 回退, 28/28 exact**(flash_32k 0.98→1.30,
  pro_4k 2.20, v32_4k 2.14)。全格 r4r2wdg 已发 GPU1-3。
- **r2_wider_direct 全格 865(r4r2wdg, GPU1-3)**: **geomean 1.4410,
  865/865 exact, 19 回退(min 0.828, p5 1.043)**。N=8195(npad 8256)带
  完全愈合;残余回退 = N=16387-16399 rung(直通窗外第一档,1CTA@512
  多 pass)12 格 + 低 hit v32_32k 数格 + pro_64k_L38(hit0.27)0.828。
  锚: 整体 0.9850,4 rung 3-5% 漂移(与 r4v14g 同型,GPU 集合差)。
  vs Bar: **1.441/1.60**, 19 regs。快照: 探针 1.511 → 全格 1.441。
- **r2_ca4486d8 收割**(`gvr_topk_cuda_v25`, 内部 1.2069): npad≤12288 独立
  direct_topk_kernel(fused level-0 直方图 + 11/11/10 radix 提前退出 +
  边界 bin 压缩 + warp 聚合发射)。**合规裁定 = 合规-附注**: 该路径为
  阈值求解在 kC≥npad 下的解析退化极限(与 v14/r2_wd 已接受的平凡路径
  同物,仅独立成 kernel + 扩到 12288),非 dispatch 到无关算子;
  因骨架锁死条款边界性,收口时向用户明示。28-cell 探针(GPU1):
  **cold gm 1.6114, 0 回退, 28/28 exact** — 子集首破 1.60。
  全格 r4v25g 已发 GPU1-3。
- **v25 全格 865(r4v25g, GPU1-3)**: **geomean 1.5208, 865/865 exact,
  回退仅 2**(pro_64k_L38 hit0.27 → 0.912; v32_16k_L30 hit0.50 → 0.948;
  p5 1.104, max 3.177)。vs Bar: **1.521/1.60**, 2 regs。
- **r2_ce05e032 收割**(`gvr_topk_cuda_v27`, 内部 1.2669): 寄存器驻留 GVR
  (npad≤262144 整行一次读进寄存器,count/secant/collect 全程零重扫,
  与 P1 hint gather 重叠)+ 大 N 流式 + 直通道;骨架保留。28-cell 探针
  (GPU1): **cold gm 1.6669, 0 回退, 28/28 exact**。全格 r4v27g 已发。
  注: op32 曾判 register-resident 在小 N latency 地板 WASH — 此处机制
  不同(中段消重扫),且实测大幅有效。
- **v27 全格 865(r4v27g, GPU1-3)**: **geomean 1.5824, 865/865 exact,
  回退 2**(pro_64k_L38 0.916 真实; v32_16k_L30 0.978 边界可 60-rep 裁)。
  p5 1.160, max 3.472。vs Bar: **1.582/1.60**, ~1-2 regs。现任本地 champion。
- r2_52f7d1a2(`r2_v9`, 内部 1.2887): v27 基础上 K=1024 reg-kernel RUNGS=6
  单格特调。28-cell 探针 1.6591(0 reg, 28/28 exact)— 低于 v27 的 1.6669,
  子集 wash → 不上全格,v27 保持本地 champion。

## Round 2 收口 → Round 3

- Round 2 终局: 内部 best 1.2887(52f7d1a2 r2_v9);50 kernels scored;
  花费 ~$493。本地链 round-2 增量: r2_wd 1.441 → v25 1.521 → **v27 1.5824**
  (865 exact, 2 regs)。Round 3 启动。

## Round 3

- **r3_489c71dc 收割**(`gvr_topk_cuda_r3_a003_v1`, 内部 1.3209): v27
  寄存器驻留 + fused count publication barrier + P4 直方图清零重叠。
  28-cell 探针(GPU1): **cold gm 1.7257, 0 回退, 28/28 exact**。
  全格 r4r3ag 已发 GPU1-3。花费已 $725/800,round 3 大概率末轮。
- **★ Bar-1 首达: r3_a003_v1 全格 865(r4r3ag)= geomean 1.6184 ≥ 1.60,
  865/865 exact,回退 1**(pro_64k_L38)。60-rep 裁定: L38 = 0.970 cold
  (warm 0.858)— 3% 真实回退,窗外,Bar-2 差此一格。该格特征 =
  hit 0.269 全带最低(同 N/K 的 L52 hit0.37 已 ≥1.0),修复通道只能是
  kernel 内 admission 逃逸(hit-rate dispatch 违禁)。
  锚: overall 0.9865,5 rung 3-9% 漂移(pro_4k 1.087 最大;配对比值有效,
  ship 判决时若定此候选需在单一 GPU 集上复测定稿)。
- 状态: Bar-1 ✅ 1.6184 / Bar-2 差 1 格(-3%) / Bar-3 ✅。round 3 继续
  ($725/800,可能被预算截断),等它清最后一格或收口。
- **r3_7d8272b7 收割**(`r3_v11` MERGE: barrier 折叠 + per-(tier,K)
  AR6/AR8 实测梯子, 内部 1.3556): 28-cell 探针 **cold gm 1.7835,
  0 回退, 28/28 exact** — 新高。全格 r4r3bg 已发 GPU1-3。
- **r3_v11 全格 865(r4r3bg, GPU1-3)**: **geomean 1.6315(新高), 865/865
  exact, 回退 1**(pro_64k_L38)。60-rep 裁定 L38 = 0.963 cold — ~4% 真实,
  窗外。**本地 champion = r3_v11**: Bar-1 ✅ 1.6315 / Bar-2 差 L38 一格 /
  Bar-3 ✅。L38 = hit 0.269 全带最低,寄存器驻留把它从 v14 时代 -17% 压到
  -4%,秒差 ~0.4µs,机理 = 低 hint 下 secant 轮数多(合规修复只剩 kernel 内
  admission 逃逸;或收口后 fork steering)。

## R5 BS 战役(gvr-topk-bs2x)启动

- **R5 campaign STARTED**: `rngnxv95cx5qfdmte69vz0b0n8`(2026-07-22T14:55Z,
  effort high, $800/8h)。目标: §7b 750 真实格(75 cell × BS2-1024)avg ≥2.0×
  vs pinned head 原生批 + 全格 ≥1.0 + 抬最低;BS=1 守 r3_v11 水位。
- 工装: r5_bs/{definition_bs(全 custom 输入, cell 为 axis, gen_inputs 从资产
  物化同行×BS), workload_bs 47 条, nsys_bs.py 本地判决, drive_grid_bs.sh}。
  平台校验坑: custom 与非 custom 输入不可混 → n_valid 也走 custom。
- **R5 本地分母 DONE (r5pr)**: head 原生批 750/750 exact,grid_r5pr.csv;
  baselines_bs.jsonl 47 条(BS=1 守门取自 R4 grid_r4pr2)。
- r3_v11 已推 fork 留档: 分支 kf/r4-champion-r3v11-bs1 @4c82d9a8。
- R4 末段收割 r3_28dc11f6(per-K AR dispatch, 内部 1.3701): 探针 1.7847
  (0 reg)与 r3_v11 1.7835 打平;全格 r4r3cg 判决中(与 R5 无 GPU 冲突,
  R5 平台跑,本地 GPU1-3 归 R4 判决)。
- **★★ R4 三 Bar 全达成 — 新 champion r3_28dc11f6**(`gvr_topk_r3_perK_dispatch`,
  内部 1.3701): 全格 865 = **geomean 1.6531, 865/865 exact, 0 真实回退**
  (唯一 <1.0 格 pro_64k_L38 grid 0.991@15rep → 60-rep 裁定 **1.0129** = 噪声;
  warm 0.908 记录在案,验收口径 = nsys cold)。min(裁定后)>0.99, p5 1.221,
  max 3.516。Bar-1 ✅ 1.6531/1.60 · Bar-2 ✅ · Bar-3 ✅。
- **横向对照(PR-arm 归一, r4r3cg)**: champion 28dc11f6 vs sglang_v2
  gm **1.099**(win 567/865, 弱区 N=8195 gm 0.80)· vs radix_cutedsl
  gm **1.593**(win 845/865)。与第一 lineage 对照: compB 1.8267 vs
  (R3 时点 head)>本期 1.6531 vs 04a0 —— 冷启动+骨架硬锁下第二 lineage
  3 轮到 1.65,未及第一 lineage 积累值(与 handoff 期望管理一致,如实入账)。
- R4 campaign 平台侧仍 Running(round 3 残余 agent,$966>800,等预算截断);
  champion 已定稿 28dc11f6,后续平台残余候选须超 1.3701 才重开判决。
- **R5 v1 平台故障 → 止损重开 v2**: submit 评测路径对 custom_inputs 全判
  0.0% correctness(agents 4 个正确 kernel 复证,本地 cudagym evaluate 全过;
  bugs 26583/84/97/26600/26602;R4 safetensors 路径无此问题 ⇒ 缺口 =
  custom_inputs × submit 评测端)。v1 campaign rngnxv95… Cancelled(账面 $9,
  round1 全废/n3 超时)。**v2 = gvr-topk-bs2x-v2 @vk9m3tetqh165a5y01j7nnrns0**
  (21:06Z STARTED): 物化 [BS,npad] safetensors 30 workloads/210MB,
  BS∈{1,4,32,128,256,1024},极端角点(高BS×大N)平台外、留本地 750 格判决;
  prompt 已注明。教训: custom_inputs 与非 custom 不可混 + submit 路径不可信,
  平台 workload 一律走 safetensors。
- **R5 v2 首收割 r5r1_39789012**(`gvr-topk-batched-v2`, 内部 0.794):
  36 案例探针(9 cell × BS{4,32,256,1024}, GPU2 净, 锚 gm 1.0002):
  **总 gm 0.834, 36/36 exact**。两极: 4k 全段强(bs1024 达 1.98-3.37×);
  中/大 N × BS≥32 重亏(pro_1024k bs1024 0.25, 64k 带 0.36-0.59,
  flash_512k 0.67-0.71)— 吞吐域仍输 head 原生批。round 1 早期符合预期。
  harness 修复: v2 契约 4 参(cell_id 已删),cand_call 已同步;
  首次探针(GPU1, 带外来 parked)因签名错未测成 → 作废,r5p1b 为准。
- R5 收割 r5_31279803(`batched-v7`, 内部 0.862): 探针 gm **0.9584**
  (36/36 exact, 锚 1.0014)。弱区(中/大N × BS≥32)0.25-0.59 → 0.46-0.57,
  小 N bs1024 2.2-3.2×。主战场未破,趋势正确。
- R5 round 1 收口(01:13Z): 终局内部 best 0.887(batched-v7 系)。
  本地判决链: v2 0.834 → v7 0.9584(36 案例, 全 exact)。
  主战场(中/大N × BS≥32)round 2 待破。
- R5 r2 收割 r5_7bea12c6(`r2-v23-exact-reg-prior`, 内部 0.9513): 探针
  **gm 1.0750**(36/36 exact, 锚 0.9968)— 首次本地过平价。
  链: 0.834 → 0.958 → 1.075。残余弱区: pro_1024k BS≥32 0.42-0.68、
  64k 带 BS≥32 0.63-0.71;小 N bs1024 2.2-3.5×。
- **R5 round-2 终局 v39 全格判决 = DQ**: 750 格 gm 0.9426, 458 回退,
  且 **8 例 kf_cand inexact**(pro_512k bs2-16、flash/pro_128k bs64;
  vdiff 最大 1.8 — 丢 mandatory)。失败带全在平台盲区(子集只有
  BS{1,4,32,128,256,1024}×9 cell,无 512k×小BS / bs64)⇒ 平台内部分数
  看不见这些错;36 案例探针(v23 1.075)亦高估全格。
  处置: (a) v23 全格复核中(是否同带 bug);(b) round 3 照跑;
  (c) 收口后 fork --append-prompt 注入盲区 exactness/弱区图谱。
- **v23 全格复核 = 同 8 例 inexact(同 cell 集, 全 row0)** → 整个 r1/r2
  lineage 正确性 DQ;全格 gm 0.9348/462 回退;bs128×16k 档洞 0.40-0.47。
  bug 特征: 仅 row0 丢 mandatory(疑跨行竞态/row-base 错),平台盲区不可见。
- **决策: cancel v2($480 止)→ fork 注入 steering**。
  fork = `befh5fh2595es8ztpcg0nmq6q8`(gvr-topk-bs2x-v3, rounds 3-5,
  max $400, 继承 44 candidates/42 insights + 2.7KB append-prompt:
  exactness bug 图谱 + 自测清单(bs{2..96}×npad 扫描)+ bs128 洞 +
  K1024 大N 吞吐方向(rows-per-CTA/persistent CTA)+ 保强区提醒)。
- **R5 v3 首收割 bc0de1e3(r3-a003-v5, 内部 1.0013 — fork 后首过 1.0)**:
  盲区 exactness 复测 = **同 8 例仍 inexact**(vdiff 数值与 v23/v39 逐位同,
  且 bs2/bs16 间会互换 → 竞态)。cand 单臂复现,排除双臂交互;
  exclude_bottom3 非元凶(仅 n-K==3 触发)。**定位: 失败形状全落
  cs>1 多 CTA cluster 寄存器驻留档 × b>1**(131072→cs8, 32832×bs64→cs4)
  — cluster DSMEM 交换的行基址/rank 在 batch 网格下混用。
  fork 种子共享代码 ⇒ bug 全 lineage 遗传;agents 无失败行资产,不可复现。
  **保底方案(收口时 engineer graft)**: dispatch 把 (b>1 && cs>1) 形状
  路由到 cs=1 档,外科修复后本地全格复验。
- **R5 v3 round-3 终局 v5-covfix(a1c460de)全格**: gm **0.9572**
  (0.935→0.957 缓升), 452/750 回退, inexact 8→**4**(仅剩 pro_512k
  npad131072 cs8 档 × bs2-16)。covfix 修好 cs4 带,但 65k×bs16-32 出现
  新弱带 0.38-0.50(cs4→streaming 的代价),pro_1024k 高 BS 0.45-0.49 未动。
  评估: 距 2.0× 目标仍远,head 原生批的吞吐摊销是结构性强项;
  rounds 4-5 后如实收口(达成与否按台账)。
- R5 v3 round 4 收口(终局内部 1.0344, 近平轮 +2.9%)→ round 5 = fork 末轮。
  round-4 增量小,跳过全格;终局判决与收口在 campaign terminal 后执行:
  最终 best 全格 + exactness → 若 cs8@512k bug 仍在则 engineer graft
  (b>1&&cs>1→安全档)→ graft 后全格复验 + BS=1 865 守门 → 报告/交付。
