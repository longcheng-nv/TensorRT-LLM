# ITER8C HANDOFF — umbriel-b200-019 (or any idle B200 mounting this NFS)

> 把下面 PASTE-READY PROMPT 整段贴进新服务器上的 Claude Code(工作目录
> `/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM`)。
> 原节点 umbriel-b200-027 的驱动已停止(GPU 被共租,从未开跑)——无双跑竞态。
> 所有代码/结果都在共享 NFS,分支 `omni/op20-gvr-extreme` @ 1c05c45a8c。

---

## PASTE-READY PROMPT

继续 op13 GVR cheaper-P2 campaign 的 **iter8c**(nsys ×3-median 冷 L2 A/B + ship
决定),目录 `indexer_topk_op_bench/op13_gvr_p2cand/`。

CONTEXT(已完成,勿重做):
- iter8a:host replay 证明 **log-count 插值** `f=log2(clo/kFT)/log2(clo/chi)`
  消除 P2 大 N eval 膨胀(K512 窄窗 262K:3.75→3.00 evals;K1024/K2048 基线窗大
  N 免费省 0.58–0.75 个 full-N 扫描且 cand 同降);Illinois 证伪(≡linear);
  赢家跨 dtype 不变。`results/rootfinder_sweep_{fp32,bf16,fp16}.{log,json}`。
- iter8b:`src/gvr_p2clog_op.py`(GvrP2CLog 子类覆写 @cute.jit
  phase2_secant_search,零 vendored 修改)exactness **396/396 PASS**
  (`results/log_exactness_fp32.log`)。
- 全程记录:`ITERATIONS.md` iter8a/8b 条目;提交 @1c05c45a8c。

DO(iter8c),顺序执行:

1. **Preflight**(必须全过再计时):
   - `nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader`
     → 所有 GPU mem < 30 GB 且 util ~0%(冷 L2 计时在共租下无效;本机 shell 可能
     看不到其它 namespace 的进程,以 mem/util 为准)。
   - `which nsys`(需要 2024+ 版本,027 上是 2026.1.1)。
   - smoke:`cd indexer_topk_op_bench/op13_gvr_p2cand && python3 src/gvr_p2clog_op.py`
     → 应打印 5 行 valdiff=0.00e+00 + "GVR p2clog smoke OK"(顺带完成 JIT 预热)。

2. **跑 9 个 nsys batch**(可断点续跑,`.nsys-rep` 已存在自动 skip):
   ```bash
   cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op13_gvr_p2cand
   bash scripts/run_iter8c_batches.sh   # 后台跑;rep 外层交错 r1→r2→r3 × K512/1024/2048
   ```
   驱动自带:逐 batch 前 wait_free 重查共租;nsys 进程 `env -u GITHUB_TOKEN -u
   HF_TOKEN`(sqlite 内嵌进程 env,永不提交 *.nsys-rep/*.sqlite——已有
   .gitignore)。日志:`results/p2clog_ab_run.log`。每 batch 约 5–10 min
   (含每 (N,variant) 的 cute.compile)。若日志出现 "EXACTNESS FAIL" 立即停止排查。

3. **解析 ×3 中位数**:
   ```bash
   python3 scripts/nsys_p2clog_ab.py --K 512  --parse-multi results/nsys/p2clog_K512_fp32_r{1,2,3}.nsys-rep
   python3 scripts/nsys_p2clog_ab.py --K 1024 --parse-multi results/nsys/p2clog_K1024_fp32_r{1,2,3}.nsys-rep
   python3 scripts/nsys_p2clog_ab.py --K 2048 --parse-multi results/nsys/p2clog_K2048_fp32_r{1,2,3}.nsys-rep
   ```
   列:base(gvr_cutedsl 基线)/ p2c(iter7 shipped,仅 K512)/ log 变体。
   把三张表存入 `results/nsys_p2clog_ab_medians.txt`。

4. **Ship 判据**(逐项核对,±0.2µs 内记 ~tie;有真实回归的配置一律不 ship
   ——iter7 K1024 先例):
   - K512 `logn(kCC=1024,kFT=614)`:小/中 N 应 ≥ p2c 的赢幅(−11~15%);
     **131K/262K 是否翻正**(host 预测:税 +1.0 eval ≈ +8.4µs vs P3+P4 节省
     ~11µs @262K)。若大 N 也赢/平 → ship logn 全 N,取消 N-dispatch;
     若大 N 输 → logn 仅 N≤65536,大 N 保持基线。
   - K1024 `logn(2048,1024)` vs `logb(base,1024)`:iter7 因 4K +15.8% / 65K
     +12.4% 回归被拒;log 版若全 N 无回归 → K1024 首次可 ship。
   - K2048 `logb(base,2048)`:131K/262K 应 −10~15%(省 0.58–0.75 evals);
     8192 处 cand 2.74×K 可能伤 P4——若 8K 回归,用 `logn(4096,2048)` 兜 8K。
   - 全部输/平 → 维持 iter7 现状,写证伪记录收口。

5. **落地 + 收口**:
   - 把胜出配置烧进 `src/gvr_p2clog_op.py` 的 ship 表(模式:dispatch_params
     风格,per (dtype,K,N-band) → (kcc,kft);16-bit 配置与 fp32 相同,见
     rootfinder_sweep_{bf16,fp16}.log)。若 ship 表含 16-bit,先补
     `validate_log_exactness.py --dts bf16,fp16`(VARIANTS 需加 16-bit 键)。
   - 更新 `ITERATIONS.md`(iter8c 条目:三张中位数表 + 决定)与
     `LEARNINGS.md`(log-interp 条目已有 host 数据,补 nsys 结论)。
   - `git add op13_gvr_p2cand && git commit -s`,提交信息带
     `Made-with: Claude Code` + `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`;
     确认无 *.sqlite/*.nsys-rep 入库(`git diff --cached --name-only | grep -E 'sqlite|nsys-rep'` 应为空)。

GOTCHAS:
- 计时结论只认 nsys pure-kernel;CUDA-event 墙钟在小 N 有 ~16µs 启动地板
  (LEARNINGS.md "Measurement methodology")。
- `results/nsys/` 下旧 kcc_*.nsys-rep 是 iter4-6 数据,勿删勿覆盖。
- 单 batch nsys 方差 ≥0.5µs → 必须 ×3 中位数才可下 ship 结论。
- 若 019 也被共租(mem>30GB),驱动会一直等——先 nvidia-smi 确认再挂。
