# NODEC_OP25_SPLIT_PROMPT — op25_hls backfill 三卡拆分:新增两张 B200 的接手指令

把下面整段作为首条消息粘贴给新节点的 Claude Code 即可。

---

接手 op22rr 报告 **op25_hls 臂 backfill** 的拆分 shard(2026-07-09 02:15Z 拆分)。
全局由 umbriel-b200-036 会话协调:036 GPU0 正在跑 `fp32 全部 + bf16 real/best`
shard 并负责最终收尾(parse/update_report/commit)。**本节点只跑自己的 shard,
跑完即停,不做收尾。**

## 拆分表(互不相交,marker 粒度幂等)

| shard | env | 内容 | 估时 |
|---|---|---|---|
| 036 GPU0(勿动) | — | fp32 全剩余 + bf16 real/best | ~25 min |
| **卡A** | `DTYPES="fp16" SCENARIOS="real best"` | fp16 real+best 18 批 | ~75 min |
| **卡B** | `DTYPES="bf16 fp16" SCENARIOS="worst"` | bf16/worst + fp16/worst 18 批 | ~75 min |

两张卡在同一节点 → 本 session 同时拉起 A、B 两条链;
两张卡在不同节点 → 每个 session 各取一行。

## 步骤

1. 预检(缺一不跑):
   ```bash
   cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op22_temporal_fixed_hr_bench
   git log --oneline -1            # 应见 460ae8211b "[op22 op25] op25_hls arm harness" 或更新
   python3 -c "import torch, cutlass"
   nvidia-smi --query-gpu=index,temperature.gpu --format=csv
   #   idle >50C 的卡禁用;若只剩一张好卡,把 A、B 两条命令用 && 串在好卡上(makespan ~150 min)
   pgrep -af drive_nsys_op22rr | grep -v pgrep   # 本机必须无 driver
   ```
2. 共驻检查(与 NODEB prompt 不同!):results 目录**会**增长——那是 036 在跑
   自己的 shard,属正常。只需确认增长不落在本 shard 内:
   ```bash
   tail -3 op25hls028_gpu0_node036.log   # 当前批头应是 fp32 或 bf16(036 shard)
   ls ../results_b200_op25hls028/*/.done_*fp16* 2>/dev/null   # fp16 marker 若已出现,说明 A/B shard 已有人跑,先联系协调会话
   ls ../results_b200_op25hls028/worst/.done_*bf16* 2>/dev/null # 同上,B shard 检查
   ```
3. 启动(setsid 必须;日志名必须匹配 `op25hls028_gpu*.log` 且各不相同):
   ```bash
   # 卡A(把 GPU=0 换成实际好卡号)
   setsid env OUT=results_b200_op25hls028 GPU=0 OP22RR_ARMS="gvr_cutedsl,op25_hls" \
     DTYPES="fp16" SCENARIOS="real best" ./drive_nsys_op22rr.sh \
     > op25hls028_gpuA_split.log 2>&1 < /dev/null &
   # 卡B
   setsid env OUT=results_b200_op25hls028 GPU=1 OP22RR_ARMS="gvr_cutedsl,op25_hls" \
     DTYPES="bf16 fp16" SCENARIOS="worst" ./drive_nsys_op22rr.sh \
     > op25hls028_gpuB_split.log 2>&1 < /dev/null &
   ```
   启动 ~60s 后 `tail op25hls028_gpu[AB]_split.log` 确认批头出现且 cell 在推进。
4. 值守到本 shard 全部 done(卡A 18 个 fp16 real/best marker、卡B 18 个 worst
   bf16/fp16 marker),日志尾无 FAILED 即完成。**到此为止**——不跑
   parse_op22_cached / update_report_op25 / git commit,收尾由 036 会话在
   81/81 后统一执行。

## 已知 gotcha(继承自 NODEB prompt)

- nsys 由 driver 内部 `env -u GITHUB_TOKEN -u HF_TOKEN` 处理,勿绕过 driver 手跑。
- 停 sweep 不要用 TaskStop:`pkill -f drive_nsys_op22rr; pkill -f sweep_op22rr;
  pkill -f "nsys profile"`,再查 respawn。
- `nsys -c cudaProfilerApi` 正常退出码可能是 143,driver 已处理。
- 单批 ~2 min(seqlen/hugeN)/ ~8.5 min(bs);批失败会打印 FAILED 且不打 marker,
  重跑同命令自动续传。
- **绝不**提交/推送 *.sqlite/*.nsys-rep(内嵌 env token)。
