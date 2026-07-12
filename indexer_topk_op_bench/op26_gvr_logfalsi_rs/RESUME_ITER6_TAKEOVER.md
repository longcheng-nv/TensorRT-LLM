# op26 iter6 接管 prompt(任意 B200,2026-07-12 自 umbriel-b200-049 紧急交接)

> 049 即将回收。工作区在共享 NFS,新机 `cd` 进来即接管,零拷贝。
> **所有代码/文档/判读工具已 commit**;唯一在飞 = op26_r0 全网格 81 批
> nsys 战役(交接时 ~32-50/81 marker 完成,driver 会随机器死掉,
> **marker 幂等重发即可续跑**)。

## 0 · 状态快照(不要重做)

- **iter6 v0.2(1cta 臂 `op26_r0`)已收敛可测**:src/gvr_op26_r0_op.py
  @HEAD;门禁 291/291;8 批代表格判决(见 ITERATIONS.md iter6 追记):
  real fp32 vs radix K512 1.295 / K1024 1.087,64K-256K 带 1.1-1.8×;
  残余 = worst 轴 anchor 0.938、8-32K 带、低 BS 大 N 多 CTA 墙。
- **iter6b(cluster 臂 `op26_r0mc`)已实现+四处注册,UNTESTED**:
  src/gvr_op26_r0mc_op.py;设计文档 = RESUME_ITER6B_PROMPT.md(必读)。
- 工具:analyze_iter6_ab.py(判读)、screen_r0_qfracs.py(铺法筛选)、
  PLAN_ITER6.md(战役计划+里程碑判据 M1-M4)。
- 结果根:results_b200_op26_iter6 (v0.1) / iter6b (v0.1) / iter6c (v0.2
  8 批) / **iter6grid(全网格,部分完成,待续跑)**——全部不入库。

## 1 · 新机预检

```bash
cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM
git rev-parse --abbrev-ref HEAD   # omni/op21-gvr-prod
git log --oneline -8              # 应见 [op26] iter6b GvrOp26R0ClusterKernel... 等
nvidia-smi --query-gpu=index,temperature.gpu,memory.used --format=csv  # 病卡排查
cd indexer_topk_op_bench/op26_gvr_logfalsi_rs
python3 src/gvr_op26_r0_op.py     # smoke,末行 "op26_r0 smoke OK"
# 已完成 marker 数(交接时 32+,后续只重发缺的)
find ../results_b200_op26_iter6grid -name ".done_*" | wc -l
```

## 2 · Step 1 — 续跑全网格(marker 幂等,原样 8 分片)

每分片独立一条(禁 `&&…&`/for-loop;env 整段照抄;发车后 grep `arms=`):

```bash
D=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op22_temporal_fixed_hr_bench
A="gvr_cutedsl,op26_r0,radix_cutedsl,sglang_streaming"
setsid bash -c "cd $D; env OUT=results_b200_op26_iter6grid GPU=0 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=512  DTYPES=fp32 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $D/grid6r_gpu0.log 2>&1 &
setsid bash -c "cd $D; env OUT=results_b200_op26_iter6grid GPU=1 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=1024 DTYPES=fp32 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $D/grid6r_gpu1.log 2>&1 &
setsid bash -c "cd $D; env OUT=results_b200_op26_iter6grid GPU=2 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=2048 DTYPES=fp32 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $D/grid6r_gpu2.log 2>&1 &
setsid bash -c "cd $D; env OUT=results_b200_op26_iter6grid GPU=3 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=512  DTYPES=bf16 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $D/grid6r_gpu3.log 2>&1 &
setsid bash -c "cd $D; env OUT=results_b200_op26_iter6grid GPU=4 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=1024 DTYPES=bf16 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $D/grid6r_gpu4.log 2>&1 &
setsid bash -c "cd $D; env OUT=results_b200_op26_iter6grid GPU=5 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=2048 DTYPES=bf16 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $D/grid6r_gpu5.log 2>&1 &
setsid bash -c "cd $D; env OUT=results_b200_op26_iter6grid GPU=6 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=512  DTYPES=fp16 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh; env OUT=results_b200_op26_iter6grid GPU=6 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=2048 DTYPES=fp16 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $D/grid6r_gpu6.log 2>&1 &
setsid bash -c "cd $D; env OUT=results_b200_op26_iter6grid GPU=7 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=1024 DTYPES=fp16 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $D/grid6r_gpu7.log 2>&1 &
```

跨节点 OK:每批 3-4 臂同卡同批配对,规范指标是批内比值(iter4/iter5
多节点接力先例);别忘了 GPU 健康预检。

## 3 · Step 2 — 全网格判读

```bash
cd $D && python3 parse_op22_cached.py ../results_b200_op26_iter6grid
cd ../op26_gvr_logfalsi_rs && python3 analyze_iter6_ab.py ../results_b200_op26_iter6grid
```
判据(PLAN_ITER6 M3):op26_r0 vs radix_cutedsl 全网格 gm ≥1.0 且
8K-262K win10 率显著抬升;vs anchor 无 <0.9 新洞(worst 轴 0.94 已知)。

## 4 · Step 3 — iter6b mc 臂上硅(照 RESUME_ITER6B_PROMPT.md §2)

1. `python3 src/gvr_op26_r0mc_op.py`(smoke,含 cs=4 格;**首跑,可能
   需要 debug**——cluster entry 是 vendored 拷贝+R0 段替换,重点检查
   R1 分支的 cluster 同步与 s_iscalars[5] 契约);
2. `OP26_GATE_ARMS=op26_r0mc python3 gate_op26.py` → 291/291;
3. 缺口格 nsys:K2048 fp16 262144 / K1024 fp32 131072-262144 BS1-16 /
   hugeN;臂对 = gvr_multicta_cutedsl + op26_r0mc + radix_cutedsl。

## 5 · 待办清单(优先级序)

1. 续跑 grid + 判读(M3 判据);
2. mc 臂 smoke/gate/A/B(M4);
3. 消融积压:P1b 并入 P1 gather(worst 0.94 残余)、kC-diet
   (K512@1536 省 28KB smem)、K2048 fp32 edge-aim R1、qfracs=UH4/M3A 对照;
4. iter6c hugeN 预研(RESUME_ITER6B_PROMPT.md §4);
5. 报告回填:扩 update_report_op26_iter5.py 系(last-writer 纪律)。
6. **花费记账(用户要求)**:战役收口时把你这段的 GPU-h(日志时间戳)
   + Claude token(解析你自己 session 的 transcript usage,定价见
   COST.md §2)追加到 **COST.md §5** 并更新顶部汇总——049 段已记
   14 GPU-h / ~$215,总花费在你处合计。

## 6 · gotcha(本战役实测)

- `cmd1 && cmd2 &` 把整串后台化(又踩);多卡发车 = 每分片独立 setsid 一条。
- gate 臂名全名 `op26_r0`/`op26_r0mc`;sglang 由 sweep 自过滤可全分片统一传。
- 趟数/准入率是 host 代理,上硅消融才算数(tid0 游走 10-15µs 教训)。
- nsys 计时批跑时不要在同 GPU 跑别的(smoke 等排后面)。
- results_* / *.sqlite / *.nsys-rep 永不 git add。
