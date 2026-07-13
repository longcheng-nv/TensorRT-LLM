# op26 小 N R0 门 A/B — 8 卡接管 TAKEOVER(2026-07-13,自 umbriel-b200-042 迁出)

> 任意 8 卡 B200 节点 `cd` 进本目录即接管,零拷贝(全部状态在 NFS)。
> **042 已完成清场**(精确 PID 击杀,双卡显存 0 MiB,07:40Z 复核)——
> 新机发车前无需再管旧节点,但**发车前必须确认本节点无同 OUT 根 driver**
> (op22 双 driver 事故 ×2)。

## 0 · 任务背景(为什么做)

- **目标**:修 op26_r0auto 的小 N 回退——N≤16K 段 r0auto(1cta 分支 =
  op26_r0 R0 梯)输给 iter5 的 op26_1cta(锚转移 gm:4096 0.971 /
  8192 **0.877** / 16384 0.984;fp32 段 0.79-0.86,worst 轴全 dtype 负)。
  机理 = 小 N 相位链/latency 主导,R0 梯省趟收益≈0,P1b 固定税
  (hist 原子加+warp0 提取+barrier+R0 count 趟)纯亏;fp32 更痛因
  dispatch_p1bc_op26 关 cache。判决记录见 ITERATIONS.md iter6/iter7 段。
- **修法**:gvr_r0_auto_op26 加第三路——小 N 直接路由 plain 1cta
  (gvr_cutedsl_op26,即报告臂 op26_1cta),R0 梯只在 N ≥ N_R0_MIN(dt)
  开。边界候选 {16384, 32768};N=4096/8192 方向历史数据已定
  (信号远超锚噪声 ±3%),**本 A/B 只钉 N∈{16384,32768} 模糊带**。
- **判据**(先例 dispatch_p4rs_mc_op26 风格):每 dtype 取最小的 N 使
  plain/r0 gm ≤1.0 且更大 N 均 ≤1.0;OFF 区内不得有 R0 系统性赢
  ≥1.02 的 (K,N) 带。

## 1 · 状态快照(不要重做)

- **A/B 战役**:OUT = `../results_b200_op26_smalln_ab`(bench 根下,
  不入库)。臂 = gvr_cutedsl(锚)+ op26_1cta + op26_r0,同批三臂
  byte-identical 输入。网格 = bs sweep,N∈{16384,32768},
  BS∈{1,4,16,64,256,1024},3 场景 × 3K × 3dtype = 27 批,每批 12 格。
- **marker 11/27 已完成**(042,07:17-07:36Z):real fp32×3 + fp16×3;
  best fp32×3 + fp16 K512/K1024。剩 16:best K2048_fp16、worst
  fp32/fp16 全 6、bf16 全 9。半批残留(best K2048_fp16 / worst
  K512_fp32)无 marker,driver 自动 rm -f 全新重测。
- **harness 已改**(未提交时看 git log,checkpoint commit 应含):
  `../op22_temporal_fixed_hr_bench/sweep_op22rr.py` 新增 opt-in
  `OP22RR_NS` / `OP22RR_BS` 过滤器(照 OP22RR_ARMS 模式)。
- **判读工具**:`analyze_smalln_ab.py`(本目录,us_cold(r0)/us_cold(1cta)
  同批配对,三视图 + OFF 候选区回归格)。
- **节奏参考**:~15s/格(JIT 热后),~3.7min/批;单卡串行剩 16 批
  ≈1h,8 卡 ≈ 10-15min。

## 2 · 新机预检(~3 min)

```bash
cd <repo>/indexer_topk_op_bench/op26_gvr_logfalsi_rs
git log --oneline -3        # 应含小 N A/B checkpoint + op28 INSIGHTS
nvidia-smi --query-gpu=index,temperature.gpu,memory.used --format=csv
# idle >50C 的卡不用(memory: 019 GPU0 / 035 GPU0 / 036 GPU1 散热坏先例)
pgrep -af "drive_nsys_op22rr|sweep_op22rr"   # 必须为空!(双 driver 禁令)
```

## 3 · 8 卡发车(剩 16 批;marker 幂等,已完成批秒跳)

按 (dtype, scenario) 9 分片布 8 卡(GPU7 串两片)。**env 整段照抄,
一个都不能少**(iter4 漏传 OP22RR_ARMS 事故 = 17 批全部返工):

```bash
cd <repo>/indexer_topk_op_bench/op22_temporal_fixed_hr_bench
ENVS='OUT=results_b200_op26_smalln_ab SWEEPS=bs OP22RR_ARMS=gvr_cutedsl,op26_1cta,op26_r0 OP22RR_NS=16384,32768 OP22RR_BS=1,4,16,64,256,1024'
LOGD=../op26_gvr_logfalsi_rs
# 每分片独立一条 setsid,不用循环/变量展开发车(iter5d shell 教训);
# ENVS 仅为文档紧凑,实操请手工展开粘贴:
setsid bash -c "cd $PWD; env $ENVS GPU=0 SCENARIOS=worst DTYPES=fp32 ./drive_nsys_op22rr.sh" > $LOGD/smalln8_g0.log 2>&1 &
setsid bash -c "cd $PWD; env $ENVS GPU=1 SCENARIOS=worst DTYPES=fp16 ./drive_nsys_op22rr.sh" > $LOGD/smalln8_g1.log 2>&1 &
setsid bash -c "cd $PWD; env $ENVS GPU=2 SCENARIOS=worst DTYPES=bf16 ./drive_nsys_op22rr.sh" > $LOGD/smalln8_g2.log 2>&1 &
setsid bash -c "cd $PWD; env $ENVS GPU=3 SCENARIOS=real  DTYPES=bf16 ./drive_nsys_op22rr.sh" > $LOGD/smalln8_g3.log 2>&1 &
setsid bash -c "cd $PWD; env $ENVS GPU=4 SCENARIOS=best  DTYPES=bf16 ./drive_nsys_op22rr.sh" > $LOGD/smalln8_g4.log 2>&1 &
setsid bash -c "cd $PWD; env $ENVS GPU=5 SCENARIOS=best  DTYPES=fp16 ./drive_nsys_op22rr.sh" > $LOGD/smalln8_g5.log 2>&1 &
# 补漏分片(大多秒跳,兜底半批/漏批):
setsid bash -c "cd $PWD; env $ENVS GPU=6 SCENARIOS='real best' DTYPES=fp32 ./drive_nsys_op22rr.sh" > $LOGD/smalln8_g6.log 2>&1 &
setsid bash -c "cd $PWD; env $ENVS GPU=7 SCENARIOS=real DTYPES=fp16 ./drive_nsys_op22rr.sh; cd $PWD; env $ENVS GPU=7 SCENARIOS=best DTYPES=fp16 ./drive_nsys_op22rr.sh" > $LOGD/smalln8_g7.log 2>&1 &
```

发车后**立即核验每个 log**:`grep -m1 "cells=" smalln8_g*.log` —— 必须
`cells=12 arms=['gvr_cutedsl', 'op26_1cta', 'op26_r0']`;错任何一个 =
立刻杀掉该分片修 env 重发。

完成判据:`find ../results_b200_op26_smalln_ab -name ".done_*" | wc -l`
= **27**;各 log 末行 `ALL OP22RR NSYS BATCHES DONE`。

## 4 · A/B 完成后执行序

1. **parse**:`python3 parse_op22.py ../results_b200_op26_smalln_ab`
   (op22 目录下跑;写 results.jsonl 带 us_cold)。
2. **判读**:`python3 ../op26_gvr_logfalsi_rs/analyze_smalln_ab.py`
   → 按判据定 N_R0_MIN(dt)∈{16384,32768,65536}(65536 = 该 dtype
   小 N 全关 R0,mc 门以下全走 plain)。
3. **src 改动**(此时 A/B 已收,src 解冻;⚠️ sweep 每批 re-import 源码,
   任何 A/B 在飞时禁改 src):`src/gvr_op26_r0mc_op.py`:
   - 新增 `dispatch_r0_smalln_op26(dt, n)`(docstring 记 A/B 证据数字);
   - `gvr_r0_auto_op26`:mc 判定后加
     `if qfracs is None and dispatch_r0_smalln_op26(dt, n): return
     gvr_cutedsl_op26(...)`(qfracs 强制 = 消融调用,不改道);
     顶部 lazy import `from gvr_op26_op import gvr_cutedsl_op26`。
   - `op26_r0` / `op26_r0mc` 臂**不动**(保持纯对照)。
   - harness `sweep_op26.py::_build_op26_r0auto_call` 的 r0_arm 记录
     同步:小 N 路由时记 `"plain"`(报告 backfill 逐格审计用)。
4. **smoke**:`python3 src/gvr_op26_r0mc_op.py` 末行 OK;再快验 r0auto
   路径三点 (N=8192 plain / N=32768 视判决 / N=131072 BS=4 mc)。
5. **gate 582/582**:
   `CUDA_VISIBLE_DEVICES=g OP26_GATE_ARMS=op26_r0mc,op26_r0auto
   OP26_GATE_DTYPES=<fp32|bf16|fp16> python3 gate_op26.py`(3 卡分片,
   每分片独立 setsid)。
6. **收口**:判决 + 边界值 + 证据写 ITERATIONS.md(iter7 追记或 iter8 段);
   RESUME_POST_ITER7.md 积压清单更新(本项完成;REPORT backfill 仍走
   统一积压⑤,把本默认与 p1bc_mc/p4_rs 一起收编);COST.md 记账
   (GPU-h 从 log 时间戳,token 从 session usage);`git commit -s`。

## 5 · Gotchas(本战役实测)

- **pkill -f 会误中自己的包装 shell**(exit 144 半途而废)→ 停 sweep
  先 `pgrep -af` 列 PID,再 `kill -9 <pids>`,最后 `pgrep` 复核 +
  nvidia-smi 显存归零。
- nsys/ncu 一律经 driver(内置 `env -u GITHUB_TOKEN -u HF_TOKEN`);
  results_*/、*.sqlite、*.log 永不 git add。
- 历史证据口径:N=4096/8192 的 OFF 判决引 fin 全网格锚转移数据
  (op22rr_op26{,r}_raw.csv,gm 0.971/0.877),本 A/B 同批数据只用于
  16K/32K;ITERATIONS.md 里要分开注明。
- 判读若 16-bit 在 16K/32K 是 wash(历史迹象:real/best 小赢 worst 小输),
  按"简单优先"取与 fp32 相同边界,除非有 ≥1.02 的系统性带。
