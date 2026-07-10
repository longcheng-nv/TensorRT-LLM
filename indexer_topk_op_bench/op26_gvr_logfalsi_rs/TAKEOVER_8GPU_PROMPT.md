# op26 8 卡 B200 接管 prompt(nsys 战役 → parse → 报告)

> 交接自 umbriel-b200-036,2026-07-09 06:5xZ。036 已 pkill 清场并用日志冻结法
> 复核(**严禁双 driver**:同一 OUT 目录两台机器同时跑 = op22 战役两次事故重演)。
> 工作区在共享 NFS,新机直接续传,无需拷贝任何东西。

## 背景(30 秒版)

op26 = 把 op13 的 log-count P2 插值 + op#7 的 exact rank-scatter P4 移植回
经典 GVR(cuteDSL)与 GVR multi-CTA(PR#15198)。实现/注册/门禁**全部完成**:

- 实现:`op26_gvr_logfalsi_rs/src/gvr_op26_op.py`(op26_1cta / op26_mc,
  纯 subclass,vendored 零编辑);臂注册已在 `sweep_op22rr.py` ARMS_EXTRA。
- 门禁 DONE:op26_1cta 291/291,op26_mc 291/291(A+B+C 三套件 0 fail 0 err)。
  Suite C 的 K2048 16-bit 宽平台案例是**锚继承的设计包络限制**(四臂含两生产锚
  逐 bit 同败,证据 `diag_tie_anchor.py`,记档 ITERATIONS.md iter3),已 cap 到
  包络内最强应力。
- 剩余 = 本文档的 nsys 战役 + parse + 报告更新。

## Step 0 — 预检(必须先做)

```bash
cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM
git rev-parse --abbrev-ref HEAD   # 必须 omni/op21-gvr-prod;不对就停下来问
# GPU 健康:idle 温度 + slowdown 余量。>50C idle 或余量 <15C 的卡禁用计时
# (前科:019 GPU0 / 035 GPU0 / 036 GPU1 三张病卡)
nvidia-smi --query-gpu=index,temperature.gpu,utilization.gpu,memory.used --format=csv
for i in 0 1 2 3 4 5 6 7; do echo "GPU$i:"; nvidia-smi -q -i $i | grep "T.Limit Temp" | head -1; done
# 环境 smoke(JIT 编译走 NFS 缓存,应秒过)
cd indexer_topk_op_bench/op26_gvr_logfalsi_rs && python3 -c "
import sys; sys.path.insert(0,'src'); import torch
from gvr_op26_op import gvr_cutedsl_op26
x=torch.randn(1,8192,device='cuda'); p=torch.topk(x[0],512).indices.int().view(1,512).contiguous()
s=torch.full((1,),8192,dtype=torch.int32,device='cuda')
o=gvr_cutedsl_op26(x,p,s,512,compress_ratio=4); torch.cuda.synchronize(); print('smoke ok',o.shape)"
```

若健康卡不足 6 张,用可用卡数重排下面的 dtype 分片(driver 支持任意
SCENARIOS/SWEEPS/KS/DTYPES 组合切分)。

## Step 1 — nsys 全网格(6 卡 dtype 分片,~2.5h 墙钟)

81 批/OUT × 2 OUT。**marker 幂等**(`$OUT/<scen>/.done_<sweep>_K<K>_<dt>`),
重跑同命令自动跳过已完成批;op26a 已有 4/81(036 遗留,直接续)。

```bash
cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op22_temporal_fixed_hr_bench
for spec in "0 op26a fp32" "1 op26a bf16" "2 op26a fp16" "3 op26b fp32" "4 op26b bf16" "5 op26b fp16"; do
  set -- $spec; g=$1; tag=$2; dt=$3
  if [ "$tag" = op26a ]; then arms="gvr_cutedsl,op26_1cta"; else arms="gvr_multicta_cutedsl,op26_mc"; fi
  setsid env OUT=results_b200_$tag GPU=$g DTYPES=$dt OP22RR_ARMS="$arms" \
    ./drive_nsys_op22rr.sh > ${tag}_${dt}_gpu${g}.log 2>&1 < /dev/null &
done
```

- 完成判据:`ls ../results_b200_op26a/*/.done_* | wc -l` == 81,op26b 同理。
- 监控:轮询 marker 数 + 日志 grep FAILED。**停 sweep 用 pkill 三连**
  (drive_nsys_op22rr.sh / sweep_op22rr.py / "nsys profile"),勿用 TaskStop;
  pkill -f 会误中自带匹配串的包装 shell(exit 144 属正常),按 PID 补刀后用
  **日志冻结法**复核(本 sandbox 对他人 namespace 的 ps/nvidia-smi 是瞎的)。
- nsys 由 driver 内部 `env -u GITHUB_TOKEN -u HF_TOKEN`;
  ***.sqlite/*.nsys-rep 永不 git add**(内嵌 env token,泄过一次)。

## Step 2 — parse

```bash
python3 parse_op22_cached.py ../results_b200_op26a
python3 parse_op22_cached.py ../results_b200_op26b
```

## Step 3 — 报告更新(update_report_op26.py)

目标报告 = `op22_temporal_fixed_hr_bench/REPORT.html` sections 1-2 + CSVs。
**必须以 `update_report_radix.py` 为底扩展**(自包含 last-writer:重导
mc + op25 + radix + op26 全部臂;任何旧 updater 在它之后跑会抹掉别的臂——
同理 update_report_op26.py 写好后就成为新的 last-writer,radix updater 不得再跑)。

- 锚迁移:op26_1cta 用本批内 gvr_cutedsl 锚;**op26_mc 链式锚迁移** =
  本批内 gvr_multicta_cutedsl × mc_adj(074→orig 刻度,系数在
  update_report_radix.py 的 mc 处理里)。
- QA 门(全过才算完):script 数=2;exactness 全 ok;**两条锚链漂移 med≈1.0**
  (各自报出);nsys_verdict 类工具若用到,必须传 `msa`。

## Step 4 — 收尾

- ITERATIONS.md 加 iter4(战役节点、批数、锚漂移、headline 数字)。
- 提交:代码 + REPORT.html + CSVs + md(参照 e8c783a62a 风格,
  `[op26] ...` 标题,`git commit -s`,加 Claude Code attribution trailers;
  永不 add sqlite/nsys-rep/大 pt)。
- 更新 memory(op26 campaign 状态)。

## 时间预算

6 卡分片:27 批/卡 × ~4.7 min ≈ 2.1h + parse ~10min + 报告 ~1h ≈ **3.5h 全收尾**。
(单 dtype 批时长方差大,bs_hugeN 慢批集中在大 N;±40min 正常。)
