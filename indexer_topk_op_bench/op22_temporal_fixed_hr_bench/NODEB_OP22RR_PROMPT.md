# op22rr Node-B 分担跑批 — 粘贴给另一台 B200 上的 Claude Code

> 生成于 2026-07-08 09:00 UTC(主控 session 在 umbriel-b200-037)。
> 本文件是唯一事实来源;若与口头描述冲突,以本文件为准。

## 任务

op22rr 五臂 nsys 全网格(81 批)的 Node-B 分片。主节点 umbriel-b200-037 正在跑
fp32/bf16 的 real+best 与 fp16 K512/1024 的 real;**本机只跑下面两个分片**,与主
节点按 (SCENARIOS × DTYPES × KS) 完全不相交,共享 NFS 结果目录,batch 级
`.done_*` 标记天然幂等。

工作目录(NFS,所有机器同一份):

```
/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op22_temporal_fixed_hr_bench
```

## 预检(必做,~1 分钟)

1. `nvidia-smi --query-gpu=index,temperature.gpu,memory.used,utilization.gpu --format=csv`
   —— 两卡应为 0 MiB / 0%;**任一卡空闲温度 >50°C 则该卡不可信**(经验规则),
   改用健康卡把两个分片串行跑。
2. `hostname` 记录下来——最终报告 notice 需要写明本机主机名。
3. bundle 数据已由主控 session 全量预检通过(best/worst 各 234 个 .pt),无需重查。

## 拉起(两条命令,各占一张卡,后台运行)

```bash
cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op22_temporal_fixed_hr_bench

# GPU0 分片: fp32 worst(9 批) → fp16 K2048 全三场景(9 批), 约 2h
LOG0=../results_b200_op22rr/driver_gpu0_$(hostname -s).log
nohup bash -c 'OUT=results_b200_op22rr GPU=0 SCENARIOS="worst" DTYPES="fp32" ./drive_nsys_op22rr.sh && OUT=results_b200_op22rr GPU=0 SCENARIOS="real best worst" DTYPES="fp16" KS="2048" ./drive_nsys_op22rr.sh' >> "$LOG0" 2>&1 &

# GPU1 分片: bf16 worst(9 批) → fp16 K512/1024 best+worst(12 批), 约 2.3h
LOG1=../results_b200_op22rr/driver_gpu1_$(hostname -s).log
nohup bash -c 'OUT=results_b200_op22rr GPU=1 SCENARIOS="worst" DTYPES="bf16" ./drive_nsys_op22rr.sh && OUT=results_b200_op22rr GPU=1 SCENARIOS="best worst" DTYPES="fp16" KS="512 1024" ./drive_nsys_op22rr.sh' >> "$LOG1" 2>&1 &
```

(在 Claude Code 里可用 run_in_background 的 Bash 代替 nohup;日志路径保持不变。)

## 监控

- 进度:`grep "=== nsys batch" ../results_b200_op22rr/driver_gpu*_$(hostname -s).log | tail`
  批完成即在 `../results_b200_op22rr/<scen>/` 落 `.done_<sweep>_K<K>_<dt>` 标记。
- 失败信号:`grep -E "FAILED|Traceback|OOM" 日志`;单批失败不阻塞后续批
  (不落 done 标记,可重跑),但要记录并汇报。
- 节奏参考(037 实测):seqlen 批 ~3.7 min,bs 批 ~11–12 min,bs_hugeN ~5 min。
  首批若含冷编译可能多几分钟;若首批 >20 min 无输出,检查 harness 编译锁
  (曾有 stale `_build/**/lock` 挂死 import 的先例)——确认无进程持有后删锁重试。

## 红线

- **不要**改动或删除任何 `.done_*` 标记(含其他分片的);不要动 `bundles*/`。
- **不要**跑本文件分片之外的 (scenario, dtype, K) 组合——会和主节点互踩。
- **不要** `git add` 任何 results/*.sqlite/*.nsys-rep(nsys 工件内嵌 env token;
  driver 已 `env -u GITHUB_TOKEN -u HF_TOKEN`,但文件仍不入库)。
- 全部批完成(两条命令都打出 `ALL OP22RR NSYS BATCHES DONE`)后**停在那里**:
  解析与 REPORT.html 更新由主控 session 统一做,本机不要跑 parse_op22.py /
  update_report_rr.py。

## 完成判据

`ls ../results_b200_op22rr/{worst,real,best}/.done_* | wc -l` 中由本机负责的 30 个
标记齐全:worst 的 fp32×9 + bf16×9、fp16 K2048 三场景×3、fp16 K512/1024 的
best+worst×12(注意 real/best 里另有主节点落的标记,勿混淆)。汇报完成即可。
