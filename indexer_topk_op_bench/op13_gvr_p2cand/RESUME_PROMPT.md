# RESUME PROMPT — op13_gvr_p2cand(campaign CLOSED @390c99c3e4;本文件 = 续作 handoff)

> op13 cheaper-P2 已全部收口(iter0–8,log-count secant SHIPPED)。把下面
> PASTE-READY PROMPT 整段贴进任意挂载本 NFS 的机器上的 Claude Code 新 session
> (工作目录 `/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM`)
> 即可继续 post-campaign 工作。分支 `omni/op20-gvr-extreme`。

---

## PASTE-READY PROMPT

继续 op13 GVR cheaper-P2 的 post-campaign 工作,目录
`indexer_topk_op_bench/op13_gvr_p2cand/`。

FIRST read,按序:
1. `op13_gvr_p2cand/ITERATIONS.md`(iter0–8 全记录;iter8c 条目 = 最终 ship)
2. `op13_gvr_p2cand/LEARNINGS.md`(含 019 GPU0 散热事故 + 计时环境守卫规程)
3. `op13_gvr_p2cand/REPORT.html`(双语终版报告,§5 = ship 表)
4. Memory: project_op13_gvr_p2cand_resume, env-b200-019-gpu0-broken-cooling

CAMPAIGN 状态(勿重做):iter8 CLOSED @390c99c3e4。
Ship = `src/gvr_p2clog_op.py::dispatch_p2c_v2()` / 入口 `gvr_cutedsl_p2c_v2()`,
fp32-only:
- K512: iter7 lin-narrow(1536,1280)@N≤65536 不变(log 变体被 nsys 证伪);
- K1024(V4 Pro,首次 ship): logn(2048,1024)@ N≤32768 ∪ N==131072
  (8K −32.1%、131K −22.0%;65K/262K 真实回归留基线);
- K2048: logn(4096,2048)@ 全 N≥8192(worst +0.6% tie,262K −12.2%);
- 16-bit: 基线(无 nsys 证据)。
正确性:396/396 全网格 + 路由 9/9 分支 exact。核心洞察:log-interp 的赢集中在
linear 基线有 P2-eval 尖峰的 cell,host replay 忠实预测 eval 数但不预测 µs。

待开展方向(按用户指示选择,无指示则按 1→2→3 顺序执行):

1. **接入生产 GvrParams**(主推,代码量小、收益已验):
   把 dispatch_p2c_v2 的 N-keyed (kCC,kFTarget,log_interp) override 移植进
   `tensorrt_llm` 生产 GVR 路径(vendored `gvr_topk_decode.py` 的 GvrParams.get
   调用点,dsa.py 侧)。注意:(a) log-interp 需要 phase2_secant_search 公式改动
   ——生产侧没有子类注入点的话要改 vendored 文件本体(参照 op-bench 侧
   GvrP2CLog 的 diff,cute.math.log2 fastmath + 退化分母回退);(b) N 在
   compile/launch 时已知,dispatch 挂在编译 key 上;(c) 走
   trtllm-code-contribution 流程(DCO、pre-commit、PR 标题规范),PR 前先在
   op-bench 侧用 gvr_cutedsl_p2c_v2 vs 生产内核 A/B 验证 local==integration;
   (d) 先查 PR #15709(op7 rank-scatter)与现 HEAD 的 GvrParams 形状再动手。

2. **16-bit 扩展 ship 表**(需 GPU,~1-2h):
   host 赢家与 fp32 相同(rootfinder_sweep_{bf16,fp16}.log),但缺 nsys 证据且
   op20 先例 16-bit 带边界会移位。步骤:(a) 扩展
   `scripts/validate_log_exactness.py` VARIANTS 加 16-bit 键,跑 exactness;
   (b) 仿 `scripts/nsys_p2clog_ab.py` 加 --dt bf16/fp16,用
   `scripts/run_iter8c_batches.sh`(v2 驱动,已含空闲门+sanity 门+隔离重跑)
   跑 ×3-median;(c) 仅对无回归带扩 dispatch_p2c_v2;全程 ITERATIONS.md 记录。

3. **op-bench 主报告加 p2c_v2 列**:
   `indexer_topk_op_bench/report/` 的 gen_report 体系加一列(参考 op#7 加列
   先例);需全 report-grid sweep,工作量大,仅在用户明确要时做。

GOTCHAS(血泪,必读):
- 计时环境:冷 L2 nsys 前必须 (a) mem<30GB 且 util≤5% ×3 连续采样;
  (b) `nvidia-smi -q -d PERFORMANCE` 查寿命 thermal-slowdown 计数器 + 跨 GPU
  空闲温度不对称(b200-019 GPU0 散热坏:空闲 70°C,预检不可见!钉
  CUDA_VISIBLE_DEVICES 到干净卡);(c) 每个 rep 过 base 区间 sanity 门
  (fp32: base@minN<20µs, base@maxN<65µs),漂移的 BASELINE = 环境问题。
  v2 驱动已内置全部守卫,直接复用。
- nsys 产物内嵌进程 env token:跑 nsys 必须 `env -u GITHUB_TOKEN -u HF_TOKEN`;
  *.sqlite/*.nsys-rep 永不入库(op13 目录 .gitignore 已挡)。
- 计时只认 nsys pure-kernel ×3-median;event 墙钟小 N 有 ~16µs 启动地板;
  单 batch 方差 ≥0.5µs。
- 双会话协作:同 NFS 工作树共享 git index,双方 add/commit 会互相吸收
  staged 内容——多机并行时提交前先 `git log -1` 确认没被对方抢跑。
- bash 驱动运行中不可原地覆写(增量读文件);先停再改。
- 提交:`git commit -s` + `Made-with: Claude Code` +
  `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>` trailers。

KEY FILES:
- `src/gvr_p2clog_op.py` — GvrP2CLog(log-interp 覆写)+ dispatch_p2c_v2 ship 路由
- `src/gvr_p2c_op.py` — iter7 lin-narrow(K512 现役)
- `src/p2_replay.py` — host replay(interp_mode 旋钮;linear 720/720 校验)
- `scripts/{rootfinder_sweep,validate_log_exactness,nsys_p2clog_ab,run_iter8c_batches}`
- `results/nsys_p2clog_ab_medians.txt` — 终版 ×3 中位数;`REPORT.html` — 终版报告
