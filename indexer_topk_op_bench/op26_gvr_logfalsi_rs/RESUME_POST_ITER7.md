# op26 post-iter7 恢复 checkpoint(2026-07-13;092 iter7 收口 → 039 小 N 门收口 @edb9da095d)

> 任意 B200 节点 `cd` 进来即接管,零拷贝。**iter7 主线 + 小 N R0 门
> A/B 均已收口并提交,无在飞任务、无脏 src**;本文档是 op26 唯一恢复
> 入口,只为二阶积压与统一 backfill 提供续跑配方。
> (TAKEOVER_SMALLN_8GPU.md 已执行完毕仅存档;039 节点已清场。)

## 0 · 状态快照(不要重做)

- **HEAD 应含**(`git log --oneline -4`):iter7 收口(D2 证伪 + gate 分片)
  → D2 实现+内存序修复 → D3 p4_rs 默认落地 @de9d33b7aa → D3 A/B 臂。
- **现役默认**(src/gvr_op26_r0mc_op.py):
  `dispatch_p1bc_mc_op26 = True(无条件)`;
  `dispatch_p4rs_mc_op26(dt,K) = not(bf16 ∧ K512)`;
  `dispatch_p4co_mc_op26 = False(D2 证伪,勿翻)`;
  `dispatch_r0_smalln_op26(dt,n) = n < (65536 if fp32 else 16384)`
  (小 N 门,07-13 smalln A/B,r0auto 小 N 直路由 op26_1cta)。
  消融臂:op26_r0mcc(p1bc 强制)/ op26_r0mcr(p4_rs 强制)/
  op26_r0mcp(p4_coop 强制,证伪对照)——四处已注册;
  op26_r0/op26_r0mc 保持纯 R0 对照(不吃小 N 门)。
- **验证态**:smoke 全绿;gate 582/582(r0mc+r0auto,p4_rs + 小 N 门
  默认下);gate 291/291(r0mcp);r0auto 三点路径 exact。
- **判决记录**:ITERATIONS.md iter7 段(预研/D3 ship/D2 证伪/结构税
  记档)+ 小 N R0 门段;PLAN_ITER7.md;COST.md §7-8。
- **小 N 门 A/B 已收口**(TAKEOVER_SMALLN_8GPU.md 已执行完毕,042 11 批
  + 039 16 批 = 27/27):其 REPORT backfill 并入下方积压⑤统一收编。
- **gotcha 全集**:ITERATIONS.md + memory;最要命的三条:
  ① sweep 每批 re-import 源码,**A/B 在飞时禁改 src**;
  ② `cluster_arrive_relaxed()` 无 release,DSMEM 读刚写数据必须
  `cluster_arrive()`(见 debug_r0mcp.py 定位法);
  ③ nsys/ncu 一律 `env -u GITHUB_TOKEN -u HF_TOKEN`,results_*/
  *.sqlite/*.ncu-rep 永不 git add。

## 1 · 新机预检(~5 min + 编译)

```bash
cd <repo>/indexer_topk_op_bench/op26_gvr_logfalsi_rs
git log --oneline -4          # 见 §0
nvidia-smi --query-gpu=index,temperature.gpu,memory.used --format=csv
python3 src/gvr_op26_r0mc_op.py   # 末行 "op26_r0mc smoke OK"
# gate 快验(3 卡 dtype 分片,每分片独立 setsid 一条):
# CUDA_VISIBLE_DEVICES=g OP26_GATE_ARMS=op26_r0mc,op26_r0auto \
#   OP26_GATE_DTYPES=<fp32|bf16|fp16> python3 gate_op26.py
```

## 2 · 积压 ①:qfracs UH4/M3A 上硅对照(~1h,零内核改动)

- 背景:host 筛选(screen_r0_qfracs.py,ITERATIONS.md L196-204)显示
  uh4 (0.90,0.65,0.40,0.15) 有更高静态接受率;现役 = M2D (0.85,0.35)。
  趟数≠时延(iter5 教训),必须上硅。
- 做法:wrapper 已带 `qfracs=` 参数,无需新臂——写小驱动对代表格
  (K1024 fp32 131072 BS1-16 / K2048 fp16 65536-262144 BS1-8 +
  win 保持格 8-32K)跑 nsys 批内配对:臂对 = op26_r0mc(M2D) vs
  op26_r0mc(qfracs=uh4)。注意 M=4 会多 2 列 smem_ptcnt_multi
  (M*num_threads*4B),K2048 kC=6144 时留意 smem 预算。
- 判据:mc 域正带 ≥1.01 且无 <0.98 损失带才谈换默认。

## 3 · 积压 ②:K2048 fp32 edge-aim R1 对照(~1h)

- 背景:iter5d 的 aim 表(edge vs center 按 (K,dtype,N) 翻转)只在
  1cta 裁过;mc 港 R1 inline shot 现用几何中心 √(kK·kCC)
  (self.log2_r1aim)。K2048 fp32 是唯一未对照过的重 K。
- 做法:加 wrapper 参数或环境开关切换 r1aim = edge(=log2(kK)),
  单格 nsys A/B(K2048 fp32 131072/262144 BS1-16,real+worst)。
  R1 只在 R0 miss 时走——real 轴静态命中 ~0.96,收益上限小,
  worst 轴才可能有肉。预期是快速 close。

## 4 · 积压 ③:kC-diet K512@1536(~1.5h,动包络)

- 背景:K512 kC=5120 → smem_keys/vals 各 5120 槽;收到 1536 省 ~28KB
  → occupancy(1cta 帮助最大;mc 域 latency-bound 可能无感)。
- 做法:GvrOp26R0Kernel/_resolve 有 kC_override(1cta 有,mc 需
  检查);gate **必须全 582 重验**(窗变窄 → miss 率升,fb_fix 兜底
  路径压力变大);单格 nsys A/B 看 8-32K 带(occupancy 敏感区)。
- 风险:窗 [K,kC] 收窄 2.3× 会抬 R0 miss 率,host 先用
  screen_r0_qfracs.py 估接受率再决定是否上硅。

## 5 · 统一报告 backfill(~2h,可与未来战役合并)

- 内容:把 p1bc_mc + p4_rs 两个默认的 mc 域增量(+1.5~9%)**加上
  07-13 小 N 门默认(N≤32K 段 r0auto 回到 op26_1cta 水位)**收进
  op22rr REPORT——81 批 fin sweep 重跑(臂 = gvr_cutedsl + op26_r0auto
  + radix_cutedsl + sglang)+ **last-writer = update_report_op26_iter6.py**
  (其内部先跑 iter5 wrapper 全量重导;marker 幂等,断点续跑配方照
  RESUME_ITER6_TAKEOVER.md §2,OUT 换新根)。
- 判据:exactness 414/414、锚漂移 med ~1.00×;收齐后 note 卡数字
  runtime 实算自动刷新。

## 6 · 记账义务

战役收口时:GPU-h(日志时间戳)+ Claude token(session transcript
usage)追加 COST.md(§7 的 092 段 token 仍待回填)+ 更新顶部汇总。
