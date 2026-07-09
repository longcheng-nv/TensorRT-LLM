# NODEB_OP25_PROMPT — op25-HLS 臂 backfill 接手(粘贴给新 B200 节点的 Claude Code)

把下面整段作为首条消息粘贴即可:

---

接手 op22rr 报告的 **op25_hls 臂 backfill** 剩余批次(上一节点 umbriel-b200-028
中断,已按批次粒度干净暂停;断点/结果全在 NFS,靠 `.done_*` marker 自动续传)。

## 背景(1 分钟)
- 目标:把 op25 优化后的 HLS 算子(`gvr_ms_auto` @ HEAD ship 默认 = w3a 梯
  (0.92,0.45,0.048) K512/K1024 + slot×2 N<65536 门控 + fp32 C=8 调度 bs≤8)
  加进 `op22_temporal_fixed_hr_bench/REPORT.html` §1(seqlen BS=1)/§2(BS 扩展)
  交互图,与 rr 重测**逐字节相同 bundle**、同 nsys cold-L2 协议 A/B。
- 每批 2 臂同进程:`op25_hls` + `gvr_cutedsl`(同机锚臂,跨节点按 cell 锚点迁移,
  所以多节点接力不损可比性;mc backfill 同配方,漂移中位 1.0053)。
- 全网格 = 3 场景 × {seqlen,bs,bs_hugeN} × K{512,1024,2048} × {fp32,bf16,fp16}
  = 81 批。harness 改动已提交(sweep_op22rr.py 的 `op25_hls` 臂、
  harness/sweep_op21.py 的 ms_path C8 复刻、update_report_op25.py 收尾脚本)。

## 步骤
1. 预检(缺一不跑):
   ```bash
   cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op22_temporal_fixed_hr_bench
   git log --oneline -3        # 应见 "[op22 op25] op25_hls arm harness" 或更新
   python3 -c "import torch, cutlass"
   nvidia-smi --query-gpu=index,temperature.gpu --format=csv   # idle >50C 的 GPU 不用(019/035 GPU0 坏散热)
   ls ../results_b200_op25hls028/*/.done_* | wc -l             # 已完成批数 X/81
   pgrep -af drive_nsys_op22rr                                  # 必须为空(防双 driver 共驻)
   ```
   共驻检测:沙箱 ps/nvidia-smi 跨 namespace 不可见,启动前 `watch` 一下
   `../results_b200_op25hls028` 下 jsonl 是否仍在增长(增长=别处还有 driver,禁止启动)。
2. 启动(setsid 必须;TaskStop 杀不死子树,清场用 pkill 三连:
   `pkill -f drive_nsys_op22rr; pkill -f sweep_op22rr; pkill -f "nsys profile"`,
   再查 respawn):
   ```bash
   setsid env OUT=results_b200_op25hls028 GPU=0 OP22RR_ARMS="gvr_cutedsl,op25_hls" \
     bash -c 'DTYPES="fp32" ./drive_nsys_op22rr.sh && DTYPES="fp16" SCENARIOS="worst" ./drive_nsys_op22rr.sh' \
     > op25hls028_gpu0_nodeB.log 2>&1 < /dev/null &
   setsid env OUT=results_b200_op25hls028 GPU=1 OP22RR_ARMS="gvr_cutedsl,op25_hls" \
     bash -c 'DTYPES="bf16" ./drive_nsys_op22rr.sh && DTYPES="fp16" SCENARIOS="real best" ./drive_nsys_op22rr.sh' \
     > op25hls028_gpu1_nodeB.log 2>&1 < /dev/null &
   ```
   日志名必须匹配 `op25hls028_gpu*.log`(update_report_op25.py 靠它检测节点名)。
   已 done 的批会打印 `SKIP done:` 直接跳过;两个 driver 的分片有重叠也无害
   (marker 粒度幂等)。单批 ~2 min(seqlen/hugeN)/ ~8.5 min(bs)。
3. 81/81 后收尾:
   ```bash
   python3 parse_op22_cached.py ../results_b200_op25hls028   # rep 级缓存,只导出新 rep
   python3 update_report_op25.py                              # 门禁+锚迁移+patch REPORT/CSV
   ```
   QA 门:脚本自带 exactness 断言(任何 FAIL 即停);打印的 anchor drift
   (每节点一行)中位应在 ~0.97-1.03;`REPORT.html` 内 `<script` 标签数应保持 2
   (`grep -c '<script' REPORT.html`)。
4. 提交(遵守 repo 惯例:`git commit -s`,加 Claude attribution trailers;
   **只**提交 REPORT.html、op22rr_{seqlen,bs}_data.csv、op22rr_op25_raw028.csv、
   本目录 md/py;**绝不**提交 *.sqlite/*.nsys-rep(内嵌 env token)):
5. 收尾后在 RESUME_PROMPT.md 追加 ADDENDUM(参考 mc backfill 的写法),
   并把 headline gm(t(base)/t(op25) seqlen 与 bs 网格、以及 op25/op21_hls 直比)
   写进总结。

## 已知 gotcha
- nsys 由 driver 内部 `env -u GITHUB_TOKEN -u HF_TOKEN` 处理,勿绕过 driver 手跑。
- `.done` marker mtime = 最后 touch 者,不能用于判断首完成;判有效性看日志批头。
- `nsys -c cudaProfilerApi` 正常退出码可能是 143,driver 已处理(无 set -e)。
- 若某批解析 ranges=0:parse_op22_cached 对空解析不缓存,重跑 parse 即可。
