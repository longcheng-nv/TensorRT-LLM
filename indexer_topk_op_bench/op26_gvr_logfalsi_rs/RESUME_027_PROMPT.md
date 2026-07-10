# op26 nsys 战役续传 prompt(b200-027,session 迁移交接)

> **[2026-07-10 CLOSED]** 战役已全部收口于 b200-069:81+81、QA 门全过、
> REPORT.html/CSV 已上线、headline 见 ITERATIONS.md iter4。本文档仅存档。

> 交接自 umbriel-b200-027 上的前一 Claude session,2026-07-09 08:32Z。
> **三个 op26a sweep driver 仍在本机 GPU0-2 上活着**(setsid 脱管,不随
> session 死亡)。新 session 的首要职责 = 监控收口,**严禁重新发车**。

## 已完成(不要重做)

- Step 0 预检:分支 `omni/op21-gvr-prod` @72a1d256c6 ✓;8 卡全健康
  (idle 33-38C,T.Limit 余量 50-55C,无病卡,GPU0-7 全可用)✓;smoke ✓。
- 036 双 driver 冻结复核 ✓(06:53Z,md5 快照 2.5min 零变化后才发车)。
- **op26b:81/81 全部完成**,且 parse 已跑完
  (`op22_temporal_fixed_hr_bench/parse_op26b_pass1.log`,所有
  results.jsonl 已写)。GPU3-5 已空闲。
- op26a:08:32Z 时 45/81,GPU0=fp32(best bs_hugeN K1024)、
  GPU1=bf16(best bs_hugeN K512)、GPU2=fp16(best bs K2048 慢批)。
  预计 ~10:00-10:40Z 收口。0 FAILED。
- parse pass1 于 08:36Z 完整跑完(op26b 全量 + op26a 当时已完成的
  ~50 批预热,.kern.json 缓存已落盘)——最终 parse 只需补 op26a 尾批,
  几分钟级。
- `update_report_op26.py`(Step 3 的 updater,036 写好)已做 3 处
  provenance 修正:节点检测 glob `op26*_gpu*.log`(旧 glob 匹配不到新
  日志名会误报 038)、CSV 改名 `op22rr_op26_raw.csv`(原 raw038 误导)、
  docstring 对齐实际节点(036+027;锚点按批同卡配对,节点无关)。
- QA 基线已核:REPORT.html `<script>`=2、单 `const D=[` blob。

## 新 session 立即执行

```bash
cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op22_temporal_fixed_hr_bench
# 1. 状态核对(driver 是否还活着:看 marker 数在涨 / 日志 mtime 新鲜)
ls ../results_b200_op26a/*/.done_* | wc -l   # 目标 81
ls ../results_b200_op26b/*/.done_* | wc -l   # 应已 =81
ls -la op26a_*_gpu?.log                       # mtime 若 >25min 不动 → 见下"卡死处置"
grep -l FAILED op26*_gpu*.log || echo no-fail
```

- **若 op26a<81 且日志仍在更新**:挂监控循环轮询(每 5min,marker 数 +
  grep FAILED),等 81/81。**不要再跑 Step 1 的发车命令**——三个 driver
  还活着,重发 = 双 driver 事故(op22 战役两次前科)。
- **若 81/81**:直接进入下面 Step 2。
- **卡死处置**(仅当某分片日志 >25min 零增长且对应 GPU 显存归零):按
  TAKEOVER_8GPU_PROMPT.md 的 pkill 三连清场(drive_nsys_op22rr.sh /
  sweep_op22rr.py / "nsys profile"),日志冻结法复核后,只对缺失
  dtype 重发单分片(OUT/GPU/DTYPES/OP22RR_ARMS 同 TAKEOVER 文档 Step 1)。

## Step 2 — 最终 parse(op26a 81/81 后)

```bash
python3 parse_op22_cached.py ../results_b200_op26a   # 增量,只导新 rep
# op26b 已 parse 过;若不放心可重跑一遍,全命中缓存秒过
```

## Step 3 — 报告更新

```bash
python3 update_report_op26.py   # 自包含 last-writer,重导 mc+op25+radix+op26 全臂
```

QA 门(全过才算完):
- 输出里 `exactness[op26_1cta]` / `[op26_mc]` / radix 两臂 全 ok、0 FAIL;
- **两条锚链漂移**:`anchor drift ... [op26a-038]`(direct)与
  `[op26b-038-chained]`(链式)各自 median≈1.0(op22 前例 med 1.0001-1.005);
- `grep -c "<script" REPORT.html` == 2;`grep -c 'const D=\[' REPORT.html` == 1;
- update_report_op26.py 之后**不得再跑任何旧 updater**(radix/mc/op25/rr
  会抹掉 op26 臂——last-writer 规则)。

## Step 4 — 收尾

- `op26_gvr_logfalsi_rs/ITERATIONS.md` 加 iter4:战役节点
  (036 4批 + 027 77批,6 卡 dtype 分片)、81+81、双锚链漂移 med、
  headline(op26_1cta vs gvr_cutedsl、op26_mc vs mc 的 real/best/worst
  几何均值,从新 CSV 提取)。
- 提交(参照 e8c783a62a 风格):`[op26] ...` 标题,`git commit -s` +
  Claude Code attribution trailers;add 范围 = update_report_op26.py、
  REPORT.html、op22rr_{seqlen,bs}_data.csv、op22rr_op26_raw.csv、
  ITERATIONS.md、本 md;**永不 add *.sqlite/*.nsys-rep/大 pt**(token 泄露前科)。
- 更新 memory `project_op26_gvr_logfalsi_rs.md`(campaign DONE + headline)。

## 已知 gotcha(继承自 TAKEOVER_8GPU_PROMPT.md,原则不变)

- 停 sweep 用 pkill 三连,勿用 TaskStop;pkill -f 会误中包装 shell
  (exit 144 正常),按 PID 补刀后日志冻结法复核(sandbox 对他人
  namespace 的 ps/nvidia-smi 是瞎的,但 nvidia-smi 显存/利用率是
  设备级可信的)。
- nsys_verdict 类工具若用到,必须传 `msa`。
- update_report_radix.py 的 `_urx` 导入路径被 update_report_op26.py
  依赖,勿改名/移动。
