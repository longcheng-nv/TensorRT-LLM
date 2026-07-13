# op26 fin2 收尾接管 prompt(任意 8 卡 B200,2026-07-13 自 umbriel-b200-027 交接)

> 工作区在共享 NFS,新机 `cd` 进来即接管,零拷贝。**积压①②③ 已全部
> 收口并 commit**(本文档 §1);唯一未完 = 积压⑤ fin2 全网格 81 批
> nsys 战役,交接时 **36/81 marker(全部干净,≤13:55:27)**。
> **027 已于 14:2X 全部清场**(外部 8 卡任务两度上节点,污染批全杀
> 且无污染 marker 落盘)——新机走 §2 判活(应为死)→ §3 直接发车。
> 本文档取代 RESUME_POST_ITER7.md 成为 op26 唯一恢复入口。

## 0 · 背景(30 秒)

- 分支 omni/op21-gvr-prod;`git log --oneline -5` 应见:
  `16daaf5478 [op26] backlog-3 SHIP: K512 1cta kC-diet 3072 ...` 及其后
  的 checkpoint commit。
- 积压判决(全部已 commit,详见 ITERATIONS.md 末三段):
  - **① qfracs UH4 证伪**(mc 域 gm 0.956,M2D 保持);
  - **② edge-aim R1 关闭**(60-rep 确认批 BS1 带未复现,center 保持);
  - **③ kC-diet SHIP kc=3072**(仅 K512 1cta 港,16-bit 16-32K 带
    gm fp16 1.0245 / bf16 1.0167,省 16KB smem;kc=1536 被 gate
    Suite C tie 平台证伪——kC 是 16-bit tie 正确性契约 ≥5K=2560;
    K1024/K2048 无 diet 空间,永久关闭);gate 582/582 全绿。
- **fin2 = 积压⑤ 统一 backfill**:81 批(3 场景 × 3 sweep × 3K ×
  3dtype),臂 = gvr_cutedsl + op26_r0auto + radix_cutedsl +
  sglang_streaming(fp32 K≤1024 自过滤),OUT=
  `../results_b200_op26_fin2`,K512 臂按 ship 后默认(kc=3072)测。
- 027 交接时:36/81 done(全部完成于 13:55:27 前,干净);当时一个
  外部 8 卡任务上过节点,污染批已全杀未打 marker,重跑即干净;
  **13:45-13:55 的最后几个 marker 在报告阶段用锚漂移复核**。

## 1 · 新机预检(~2 min)

```bash
cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op26_gvr_logfalsi_rs
git log --oneline -5                  # 见 §0
nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv
#   任何卡 >20GB = 有外部任务,nsys 批禁发(等清净;同卡共跑数据全废)
python3 src/gvr_op26_r0mc_op.py       # 末行 "op26_r0mc smoke OK"(可跳)
```

## 2 · 判活:027 的 driver 还在跑吗(必做,双 driver 禁令)

```bash
# NFS 上看 marker 是否还在推进(027 的 setsid driver 独立于 session 存活)
find ../results_b200_op26_fin2 -name ".done_*" | wc -l   # 交接时 36
find ../results_b200_op26_fin2 -name ".done_*" -newermt "-30 minutes" | wc -l
```
- **>0(30 分钟内有新 marker)= 027 还活着 → 跳到 §5 只做监控,
  禁止发任何 driver**(跨机双 driver 写同一 NFS 输出 = 事故重演)。
- **=0 且总数 <81 = 027 已死 → §3 重发**。
- **总数 =81 → 直接 §4 收尾**。

## 3 · 重发(仅 027 判死后;marker 幂等,9 组不相交拆分)

每组 5 批,每分片独立一条 setsid(禁 `&&…&` 把整串后台化):

```bash
T=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op22_temporal_fixed_hr_bench
A="gvr_cutedsl,op26_r0auto,radix_cutedsl,sglang_streaming"
setsid bash -c "cd $T; env OUT=results_b200_op26_fin2 GPU=0 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=512  DTYPES=fp32 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $T/fin2t_gpu0.log 2>&1 &
setsid bash -c "cd $T; env OUT=results_b200_op26_fin2 GPU=1 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=512  DTYPES=bf16 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $T/fin2t_gpu1.log 2>&1 &
setsid bash -c "cd $T; env OUT=results_b200_op26_fin2 GPU=2 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=512  DTYPES=fp16 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $T/fin2t_gpu2.log 2>&1 &
setsid bash -c "cd $T; env OUT=results_b200_op26_fin2 GPU=3 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=1024 DTYPES=fp32 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $T/fin2t_gpu3.log 2>&1 &
setsid bash -c "cd $T; env OUT=results_b200_op26_fin2 GPU=4 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=1024 DTYPES=bf16 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $T/fin2t_gpu4.log 2>&1 &
setsid bash -c "cd $T; env OUT=results_b200_op26_fin2 GPU=5 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=1024 DTYPES=fp16 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $T/fin2t_gpu5.log 2>&1 &
setsid bash -c "cd $T; env OUT=results_b200_op26_fin2 GPU=6 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=2048 DTYPES=fp32 OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $T/fin2t_gpu6.log 2>&1 &
setsid bash -c "cd $T; env OUT=results_b200_op26_fin2 GPU=7 SCENARIOS='real best worst' SWEEPS='seqlen bs bs_hugeN' KS=2048 DTYPES='bf16 fp16' OP22RR_ARMS='$A' ./drive_nsys_op22rr.sh" > $T/fin2t_gpu7.log 2>&1 &
```
- GPU7 串 2 组(bf16→fp16 K2048)与其他分片零交集,安全;若想更快,
  fp16 K2048 一组可拆给最先跑完的卡**手动补发**(先确认那张卡的
  driver 已打出 "ALL OP22RR NSYS BATCHES DONE")。
- 发车后 `grep arms= $T/fin2t_gpu*.log` 应见 4 臂(16-bit 无 sglang)。
- **027 若还有半批残留 jsonl/rep**:driver 每批开头 `rm -f` 重建,无需手清。
- 跨节点 OK:规范指标是批内比值(iter4/iter5 多节点接力先例);
  收尾锚漂移检查会兜底。

## 4 · 81/81 后收尾(~30 min)

```bash
cd $T
env -u GITHUB_TOKEN -u HF_TOKEN python3 parse_op22_cached.py ../results_b200_op26_fin2
# ^ 增量,可在跑批途中反复执行摊薄尾巴
```
1. **重靶 last-writer**:update_report_op26_iter6.py 两处——
   `FIN_ROOT = HERE.parents[0] / "results_b200_op26_fin2"`;
   `FIN_NODE = u27._detect_nodes("fin2*_gpu*.log", "<实际节点串>")`。
2. `env -u GITHUB_TOKEN -u HF_TOKEN python3 update_report_op26_iter6.py`
   (内部先跑 iter5 wrapper 全量重导 + gate_exactness;报错即 exactness
   不齐,先查缺批)。
3. **判据**:exactness 414/414;锚漂移(gvr_cutedsl vs ORIG_ROOT)
   med ~1.00×;**特查 13:45-13:55 尾部 marker 的批**(027 外部任务
   上线前最后几批),漂移 >1.05 的批删 marker 重跑。
4. report/report.html 打开肉眼过一遍 op26_r0auto 曲线(K512 16-bit
   16-32K 段应比 iter6final 快 ~2%,kc-diet 效果)。
5. **记账(COST.md §7-8 追加 + 顶部汇总)**:GPU-h 从 fin2*/logs_resume027
   日志时间戳实算;Claude token 从 session transcript usage。
6. RESUME_POST_ITER7.md 刷新(积压①②③⑤ 全收口,指向本文档存档)
   + memory(project_op26_gvr_logfalsi_rs.md)更新。
7. commit(**只 commit 代码/文档;results_* / *.sqlite / *.nsys-rep
   永不 git add**)。

## 5 · 只监控模式(027 活着时)

```bash
watch -n 120 'find ../results_b200_op26_fin2 -name ".done_*" | wc -l'
# 到 81 → §4;停滞 >30min → 按 §2 重判(可能 027 被回收或撞外部任务)
# 注意:027 的第 9 组(K2048 fp16)未发车——如 80/81 长期停滞,缺的
# 就是它,按 §3 GPU7 的 env(只留 DTYPES=fp16 KS=2048)单独补发即可
```

## 6 · gotcha 全集(本 session 新增 3 条 + 存量)

- **外部任务同节点 = nsys 数据全废**:发车前和监控中都要看
  nvidia-smi;>20GB/卡即停手等清净(027 已实测踩过,污染批全杀重跑)。
- **kC 是 16-bit tie 正确性契约(≥5K)**,不只是性能包络——任何动
  candidate 窗的优化必须过全 gate(Suite C),smoke 随机数据抓不到。
- **低 reps 单点正带先复现再判**(edge-aim BS1 1.9-2.4% 在 60-rep
  下缩回噪声)。
- sweep 每批 re-import 源码,**A/B 在飞时禁改 src**;
- 停 driver 用精确 PID 树杀(`kill -TERM -- -PGID`),不用 pkill -f;
- nsys/ncu 一律 `env -u GITHUB_TOKEN -u HF_TOKEN`;
- `cmd1 && cmd2 &` 会把整串后台化,多卡发车每分片独立 setsid 一条。
