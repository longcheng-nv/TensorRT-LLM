# op20 tier3 交接 prompt(粘贴到新机器 Claude Code 新 session 的首条消息)

> 用法:在任意 **B200 节点**(2×B200,GPU0 可独占)启动 Claude Code,工作目录
> 设为 `/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM`,
> 把本文件 TASK 一节整体粘贴。上一台节点(umbriel-b200-039)在 tier3 基线
> 跑到 ~80/240 cells 时到期中断。

---

## TASK(从这里开始粘贴)

继续 op20 GVR 极致优化 campaign 的**最后阶段:tier3(16-bit)+ 最终总结**。
先读 `indexer_topk_op_bench/op20_gvr_extreme/{PLAN,ITERATIONS,LEARNINGS,RESUME_PROMPT}.md`。

### 0. 已完成状态(branch `omni/op20-gvr-extreme`)
- **tier1 CLOSED** @01aa989b4f:65/84 fastest,rival/x gm 1.345,84/84 exact;
  小 N 墙(N4-8K)判定结构性(相位链地板,smem-resident/mc-cache/P1 子采样
  三路证伪)。iter4+5 @ff2e86216a:fused P2+P3(f/nf 后缀)+ fusP4T4 路由。
- **tier2 CLOSED** @bd98ac2135:24/36 fastest,rival/x gm 1.251,36/36 exact;
  K2048 大 N 用 fusP4T4/mc 路由,N8192 墙接受。
- **tier3 进行中**:基线 `results/tier3_iter0.jsonl` 已有 ~80/240 条
  (bf16/fp16 × K512/1024/2048 × N × BS{1,4,16,64,256,1024})。
  `scripts/tier_bench.py` 已加断点续跑(按 out JSONL 逐 key 跳过)。

### 1. 预检
```bash
cd indexer_topk_op_bench/op20_gvr_extreme
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu --format=csv
# 只能用 memory.used=0 的 GPU;本环境 ps 对其他 namespace 不可见,
# 中途要复查 memory.used 防 co-tenancy 污染 cold-L2 计时。
# 防双写:确认上一节点确实死了(连续两次 stat 间隔 60s,mtime 不变)
stat -c '%y %s' results/tier3_iter0.jsonl; sleep 60; stat -c '%y %s' results/tier3_iter0.jsonl
```

### 2. 续跑 tier3 基线(~45 分钟剩余)
```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/tier_bench.py --tier 3 --out results/tier3_iter0.jsonl >> results/tier3_iter0.log 2>&1 &
```

### 3. 洞分析 + 按桶探针(协议红线:未在该 (dtype,K,N,BS) 桶探针过的
配置不得写进 dispatch 表)
- 从 `tier3_iter0.jsonl` 列出 losses(rival_us < x_us),按 (dtype,K,N-区间) 聚类。
- 预期形态(fp32 经验平移):大 N(131K/262K)低 BS 若挂 cluster16 → 探
  fusP4T4(cr:K512/1024=4,K2048=1)与 mc-auto/mcC8;BS16 探 mc-auto
  (fusion 在 BS16 会 bs×P 过展崩溃,fp32 两次验证,不要路由);
  M*R1p4 keys 探 {M2,M4,M6}×{f,nf}(fused 使最优 M 下移)。
- 探针脚本参考:`scripts/iter5_probe.py`(改 K/cr/dtype)与
  `scripts/iter4b_retune.py`(改 K 循环与表文件名)。
- 16-bit 特有注意:dispatch 表是 `results/dispatch_table_{bf16,fp16}.json`
  (op19 时代,无 iter4/5 杠杆);16-bit tie-stepped CCDF 使 count>kC 更常见,
  任何新路由必须过 exactness(vdiff=0 + uniq=K);fusP4T4 的 D0 修复只在
  op17 v2 算子里,确认 import 的是 `op17_gvr_portfolio/v2/gvr_portfolio_fusion_op`。
- 小 N 墙(N4-8K)预计同样结构性:探一轮变体,无 ≥3% 改进就按 tier1/2
  先例接受,不要重复 iter6 的三条已证伪路线。

### 4. 改表(先备份 `.pre_tier3`)→ tier3 验收
```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/tier_bench.py --tier 3 --out results/tier3_iter1.jsonl > results/tier3_iter1.log 2>&1 &
```
验收标准:240/240 exact;改动 key 全部改善;未动 key gm 无回退(>0.97)。

### 5. 提交 + 最终总结
- ITERATIONS.md 写 tier3 条目 → `git add`(jsonl/表/备份/脚本;.log 被
  gitignore)→ `git commit -s`,message 末尾加:
  `Made-with: Claude Code` 与 `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`
- 更新 RESUME_PROMPT.md 为 campaign-CLOSED 状态。
- **输出 op20 最终总结**(用户已明确要求):iter0→tier3 全程记分板
  (tier1 65/84 gm1.345 / tier2 24/36 gm1.251 / tier3 结果)、采纳杠杆
  (fused P2+P3 slot-collect + f/nf、fusP4T4 路由、mc/mcC8 路由、dispatch
  重调 M4→M2)、证伪清单(iter1 level-2、小 N 三路手术、fusP8T4、BS16
  fusion)、结构墙结论(小 N 相位链地板 ~2µs)、交付物索引(src/表/结果/
  文档)。更新持久记忆 `project_op20_gvr_extreme_resume.md` + MEMORY.md 行。

### 协议(硬规则,全程有效)
- 取舍只认 GPU0 独占全量 tier_bench;探针必须在准确的 (dtype,K,N,BS) 桶。
- 每步落盘即 checkpoint;commit 用 `-s` + 上述 trailers;不 push(用户没要求)。
- exactness 是绝对红线:任何 inexact 的路由立刻回退。

## TASK 结束
