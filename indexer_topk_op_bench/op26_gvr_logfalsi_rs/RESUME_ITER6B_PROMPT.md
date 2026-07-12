# op26 iter6b 接管 prompt — R0 h-空间梯移植进 PR#15198 cluster 内核(mc 港)

> 2026-07-12 umbriel-b200-049 落笔。iter6 1cta 臂(op26_r0)已 v0.2 收敛:
> real fp32 K512 vs radix 1.295 / K1024 1.087,64K-256K 带 1.1-1.8×,
> 门禁 291/291;全网格 81 批(OUT=results_b200_op26_iter6grid,4 臂)
> 判决见 ITERATIONS.md iter6 追记。**剩余结构缺口 = 低 BS ≥131K +
> 16-bit 大 N(radix 行内 4-32 SM vs 单 CTA)→ 本文档的 mc 港任务。**

## 0 · 状态快照

- 1cta 臂:`src/gvr_op26_r0_op.py` @HEAD(GvrOp26R0Kernel ⊂ GvrOp26Kernel,
  P1b 256-bin hist → M2D (0.85,0.35) rung → R0 一趟 M=2 count(税 ×1.02)
  → 最紧可接受 rung 缓存列零重扫喂 P3 → miss 时 R1 inline 双实测点
  log-falsi 一发 → 再 miss 落 fb_fix)。臂名 `op26_r0` 已注册:
  gate_op26.py / harness/sweep_op26.py / harness/sweep_nsys.py /
  op22_temporal_fixed_hr_bench/sweep_op22rr.py。
- 筛选工具:`screen_r0_qfracs.py`(可改 LADDERS 复筛);判读:
  `analyze_iter6_ab.py <root>`。
- 三条已上硅教训:①tid0 256-bin 串行游走 = 10-15µs/CTA 致命税
  (v0.1 改 warp-0 分段 shfl 前缀);②M=4 趟税在 worst(基线首趟即中)
  纯亏 → M=2;③p4 谱系 GvrTopKKernel 是 vendored 全拷贝,不能与 op18
  谱系菱形混入 —— cluster 港同理:**直接 subclass
  GvrTopKClusterKernel(vendored),方法拷入,不做跨谱系 mixin**。

## 1 · iter6b 设计(GvrOp26R0ClusterKernel ⊂ GvrTopKClusterKernel)

1. **P1b**:cluster P1(vendored phase1_preidx_stats,cluster 版在
   gvr_topk_decode_cluster.py:526)每 CTA 冗余算全 preIdx 统计 →
   P1b hist 同样每 CTA 冗余建(K 次 gather,确定性)→ **rung 每 CTA
   一致,零 DSMEM 协调**。warp-0 并行提取照抄 1cta 版。
2. **R0 multi-count**:拷 cluster block_count_ge(:706,slice 版,含
   smem_input 分支)改 M 计数器(照 1cta 的 block_count_ge_multi 改法);
   per-CTA M 个 slice 总数写 `s_cluster_partial[0..M-1]`(entry 里把该
   smem 张量长度从 cluster_size?×1 扩到 M 槽——看 vendored entry 的
   实际 layout),**一次** cluster_arrive_relaxed/wait,tid0 对每列
   mapa+ld_shared_cluster 求和 → s_mt_cnt[m] 全行计数 → 每 CTA 独立
   同判(决定性,无需再广播)。
3. **准入**:count∈[kK,kCC] 最紧 rung;缓存列喂 cluster P3(per-slice
   collect 语义:缓存列 = 本 CTA slice 的 per-thread counts,与 vendored
   contract 一致)。miss → R1 inline 一发(cluster block_count_ge 单阈值,
   含 DSMEM 归并)→ 再 miss 落 **vendored** cluster fallback
   (op26_mc 先例:per-slice retry + leader handoff 不动,fb_fix 不移植
   ——异常包络与锚一致)。
4. **wrapper** `gvr_r0_mc_op26`:镜像 gvr_multicta_op26 的
   _resolve_config_mc;臂名建议 `op26_r0mc`,同四处注册。
5. **消融旗标**:qfracs(M2D 默认)、R1 开关。cluster_size 调度不动
   (PR#15198 host auto-dispatch 原样)。

## 2 · 验证流水线

1. smoke(gvr_op26_r0mc_op.py __main__,含 cs>1 格:BS≤16 & N≥65536);
2. gate_op26.py 加臂 `op26_r0mc` → OP26_GATE_ARMS=op26_r0mc,期望 291/291;
3. 单格 nsys:缺口格 = K2048 fp16 262144 低 BS、K1024 fp32 131072/262144
   BS1-16、hugeN 524288/1M;臂对 = gvr_multicta_cutedsl(锚)+ op26_r0mc
   + radix_cutedsl;
4. 赢了 → 全网格(注意 REPORT last-writer 仍是 update_report_op26_iter5.py
   系;新臂回填需扩 updater,勿跑旧 update_report_op26.py)。

## 3 · 已知 gotcha(继承 + 本战役新增)

- 多卡发车 = 每分片独立一条 `setsid bash -c "cd …; env … ./drive.sh" > log &`;
  **`cmd1 && cmd2 &` 会把整串后台化**(本 session 又踩一次)。
- OP22RR_ARMS 整段照抄 + 发车后 grep `arms=` 核验;sglang 臂由
  sweep_op22rr 内部自动过滤(fp32 & K≤1024),全分片可统一传。
- gate 臂名是全名(`op26_r0`,传 `r0` 会 ValueError);pkill -f 会误中
  自己的包装 shell;sandbox 对他 namespace 进程不可见,用 nvidia-smi
  pid + 日志 mtime 判活。
- nsys 由 driver 内部 `env -u GITHUB_TOKEN -u HF_TOKEN`;
  results_* / *.sqlite / *.nsys-rep 永不 git add。
- 趟数/准入率是 host 代理,**上硅消融才算数**(v0 的 tid0 游走教训)。

## 4 · iter6c(预研,mc 之后)

hugeN(≥512K)低 BS 即使 cs=4 也只有 4 SM/行 vs radix 32:评估
radix 式行内多 CTA count(GVR 阈值逻辑,GMEM atomic 归并 + 两段
kernel 或 cooperative launch;cluster>4 已被 GPC wave_cap 证伪)。
先出可行性判决,再决定是否实现。
