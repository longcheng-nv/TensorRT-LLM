# op26 iter5 接管 prompt(任意 B200 机器,2026-07-10 自 b200-069 交接)

> b200-069 到期回收。**无在飞进程、无未提交的 op26 内容**——工作区在共享
> NFS,新机 `cd` 进来即接管,零拷贝。交接时 HEAD 上方有并行 op27 session
> 的提交(见 §5 协同注意)。

## 0 · 状态快照(不要重做)

- **战役 DONE** @206168c08e:162 批 nsys、QA 全过、REPORT.html §1-2 上线;
  判决 = op26_1cta fp32 gm 1.100 ship 候选、op26_mc 证伪不 ship;
  花费档案 COST.md @17b8140ddb。
- **回退根因 DONE** @ad032c1a73:`ROOTCAUSE_P2.md`(必读,含全部证据)。
  一句话:R1 = kFTarget=kK 瞄准接受带边缘 + falsi 端点冻结(K1024@131K
  5趟 vs 2趟);R2 = chi=1 假种子 + log 插值跨分布体部(K2048 16-bit
  几何爬行)。fb_fix 无辜;尾部 CCDF 确实指数(R²≥0.99)。
- **修复变体已在 host 重放量化**(`diag_p2_variants.py`):
  V3 = log-secant(最近两实测点)+ 几何中心瞄准 √(kK·kCC) + 保留窄窗
  → 平均趟数 4.11→3.07(anchor 2.37),cand 压缩 3.6× 保留。
- **剩余任务 = iter5**:V3(+B)上硅 → 门禁 → 单格 nsys A/B → 全网格。

## 1 · 新机预检(必须先做)

```bash
cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM
git rev-parse --abbrev-ref HEAD    # 必须 omni/op21-gvr-prod
git log --oneline -6               # 确认 ad032c1a73 (op26 rootcause) 在史内
# GPU 健康:idle >50C 或 T.Limit 余量 <15C 的卡禁用计时(病卡前科:
# 019-GPU0 / 035-GPU0 / 036-GPU1)
nvidia-smi --query-gpu=index,temperature.gpu,memory.used --format=csv
for i in 0 1 2 3; do nvidia-smi -q -i $i | grep -m1 "T.Limit"; done
# smoke(JIT 走 NFS 缓存,秒过)
cd indexer_topk_op_bench/op26_gvr_logfalsi_rs
python3 src/gvr_op26_op.py   # 内置 smoke,末行应 "op26 smoke OK"
# 重放脚本可跑性(读 bundles,~1min)
python3 diag_p2_variants.py | tail -3
```

## 2 · iter5 实现要点(V3 + B)

改 `src/gvr_op26_op.py` 的 `GvrOp26Kernel.phase2_secant_search`(mc 臂不动):

1. **log-secant 过最近两个实测点**:插值不再用括号端点 (vlo,clo)/(vhi,chi),
   改用最近两次 count_ge 的 (v,c)。实现:interp 全在 `tidx==0` 分支内,
   直接用 thread-0 的**寄存器局部变量** `v_prev/c_prev`(iter0 用
   thr_init/c0 初始化)跨迭代保存,无需加 smem 槽。算出的 nv 仍做 vendored
   括号 clamp(5% 边距 + 中点保底)→ 精确性防线不变(接受判据
   count∈[kK,kCC] 原样)。两点退化(c 相等/v 重合)→ 回落括号 log-falsi。
2. **瞄准点改几何中心**:`kFTarget → int(sqrt(kK*kCC))`,按生效窗算:
   fp32 K512 窄窗(1536)→ 887(线性窄窗格是否同改:先不改,K512 无回退);
   K1024 窄窗(2048)→ 1448;K2048 窄窗(4096)→ 2896;
   16-bit stock:K1024(5120)→ 2289、K2048(5120)→ 3238。
   用新 `__init__` 旗标(如 `p2_secant2=True, kFT_center=True`)门控,
   便于消融;dispatch_p2_op26 返回值加一位。
3. **B(可选,若 P1 结构允许)**:P1 求 pmin/pmax 时顺带实测两端计数替换
   1.25K/1 假种子。翻 `gvr_topk_decode_p4.py` phase1(vendored 行 306-),
   若需改 vendored 则放弃 B(零编辑铁律),只做 1+2。

## 3 · 验证流水线(顺序执行)

```bash
# (a) 291 门禁(全过才继续;Suite C 已 cap 到包络内,见 ITERATIONS iter3)
python3 gate_op26.py 2>&1 | tail -5     # 期望 ok=291 fails=0

# (b) 单格 nsys A/B —— 三个代表格,先证修复、再证不伤既有 win
cd ../op22_temporal_fixed_hr_bench
# 回退格1:K1024 fp32 131072(目标:0.56-0.62 → ≥0.85)
env OUT=results_b200_op26_iter5 GPU=0 SCENARIOS=real SWEEPS=bs KS=1024 \
    DTYPES=fp32 OP22RR_ARMS="gvr_cutedsl,op26_1cta" ./drive_nsys_op22rr.sh
# 回退格2:K2048 fp16 65536(16-bit 爬行修复)
env OUT=results_b200_op26_iter5 GPU=1 SCENARIOS=real SWEEPS=bs KS=2048 \
    DTYPES=fp16 OP22RR_ARMS="gvr_cutedsl,op26_1cta" ./drive_nsys_op22rr.sh
# win 保持格:K1024 fp32 8192 在上面 KS=1024 批内一并出数
python3 parse_op22_cached.py ../results_b200_op26_iter5
# 判读:N=131072 行 BS≥256 的 anchor/op26 比值

# (c) 赢了 → 全网格 81 批重测:照 TAKEOVER_8GPU_PROMPT.md Step 1,
#     OUT=results_b200_op26a_iter5;⚠️ env 必须整段带 OP22RR_ARMS
#     (069 战役漏传返工 4 GPU-h 的教训),发车后立即 grep 'arms=' 核验。
```

## 4 · 已知 gotcha(继承)

- 停 sweep 用 pkill 三连 + 日志冻结法复核;长跑一律 setsid;勿用 TaskStop。
- nsys 由 driver 内部 `env -u GITHUB_TOKEN -u HF_TOKEN`;
  *.sqlite / *.nsys-rep / results_* 目录**永不 git add**。
- vendored 文件零编辑;全部 subclass override + 独立旗标。
- 每个有产出的节点立即落盘 + commit(checkpoint 习惯,用户明确要求)。

## 5 · 与并行 op27 session 的协同(重要)

- 同分支上有活跃的 **op27 战役**(HLS 鲁棒性,session da924869,commits
  6dde8b8454/83a231f3b7/...)。**REPORT.html 的 last-writer 已易主**:
  op27 的 updater(见 6dde8b8454)重导 mc/op25/radix/op26/op27 全臂——
  **iter5 全网格重测后不要再跑 update_report_op26.py**,改用/扩展 op27
  的 updater(或以它为底做 update_report_op26_iter5),否则会互抹。
- 工作区里 op27 的未提交 CSV/results 目录不要动、不要顺手 commit。
- 若在 027 上跑:先按日志冻结法确认 op27 的 driver 不占目标 GPU。

## 6 · 完成后

- ITERATIONS.md 加 iter5(消融:secant2/kFT_center 分开与合并的数字);
- ROOTCAUSE_P2.md 顶部补一行上硅判决;COST.md 追加本轮花费;
- commit(-s + Claude Code attribution trailers);memory 更新
  project_op26_gvr_logfalsi_rs。
