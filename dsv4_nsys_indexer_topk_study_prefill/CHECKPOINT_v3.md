# Session Checkpoint v3 — 2026-06-01T09:48Z (umb-b300-023, Claude session restarting)

---

## ⚠️ LIVE STATUS UPDATE @ 2026-06-01T09:56Z (T+17min into Run #1 v2)

**Run #1 v2 is past weight load but **silent for >16 min** after that.** Triage facts:

- `nvidia-smi`: GPUs 0-3 carrying 42-45 GB each (model loaded, TP=4 ranks 0-3 ✓)
- `trtllm-bench` PID 6044: State=S, wchan=`do_poll.constprop.0` (waiting)
- orted child PID 6426: State=S, wchan=`ep_pol` (OpenMPI singleton, healthy)
- `run.log` ends at `Loading weights: 100% 1884/1884` — no autotuner / cuda-graph / bench messages
- No `[Autotuner]`, no `Running benchmark`, no iter lines

**Hypothesis** (in order of likelihood):
1. CUDA graph capture per-config stuck (cuda_graph_config has batch_sizes=[1..8] → 8 captures)
2. Autotuner tactic search cold cache (first-time on this host)
3. NCCL init / collective handshake stuck
4. Pathological scheduler behavior with 80-req dataset (BS=1, chunked-prefill)

**Action for the next session if the situation hasn't changed by your arrival:**

```bash
# 1. Check if it's STILL silent:
tail -3 /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/dsv4_nsys_indexer_topk_study_prefill/ds4_Flash_ISL68656_OSL2048_BS1_MTP0_TEP_GVRfalse_20260601T093908Z/run.log

# 2. If still no iter lines after T+25min from 09:39Z (i.e. > 10:04Z), KILL and bisect:
kill -9 5906 5912 6017 6044 6426 2>/dev/null
sleep 3
pkill -9 -f trtllm-bench

# 3. Re-launch with smaller NUM_PROMPTS to isolate:
#    - If NUM_PROMPTS=20 works → 80 was pathological; back off
#    - If NUM_PROMPTS=20 also hangs → not request count; investigate cuda_graph
sed -i 's/PROMPTS_REPLICATE=80/PROMPTS_REPLICATE=20/; s/NUM_PROMPTS=80/NUM_PROMPTS=20/' \
    /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/dsv4_nsys_indexer_topk_study_prefill/launch_gvr0_multireq.sh
# Re-run as before with launch_gvr0_multireq.sh
```

**Don't kill before T+25min from 09:39Z** — autotuner / cuda graph capture CAN legitimately take 15+ min on cold cache, esp. with 8 batch-size variants × 4 ranks. If you see `[Autotuner] Autotuning process starts` even at T+22min, let it run.

---

> **新会话开局先 cat 这个文件 + CHECKPOINT.md。** v3 是当前(09:48Z)状态;CHECKPOINT.md
> 是上一会话(08:22Z umb-b300-022 切机前)的状态。RESUME_PROMPT.md 是更早的恢复指南,**部分已过时**(详见 §6)。

---

## 1. 任务目标 (recap)

V4 Flash + TEP-4 + BS=1 + ISL=68656 OSL=2048 MTP=0,prefill+decode 双阶段 nsys,**GVR off vs on** 对比。最终 `REPORT_PREFILL.md`(骨架已有)填 4 个 headline Δ%:
- prefill TTFT
- decode TPOT (51-iter window iter 500-550)
- prefill indexer ms (4-GPU sum)
- decode indexer ms (window sum)

v3 decode-only baseline 在 `dsv4_nsys_indexer_topk_study/REPORT.md`(2026-05-29),目标复现 indexer Δ% ≈ −66.6% ±2pp。

---

## 2. 本次会话(B300-023 上)新解锁的 2 个关键 unblocking

### 2.1 不需要再 stage Flash —— `/dev/shm/DeepSeek-V4-Flash/` 已完整

**`/dev/shm` 在本机上已有完整 46/46 shards 的 V4 Flash**(149 GB, RAM-backed,**比 /raid 还快**)。本次会话开始时把 /raid stage 任务也跑了 15 min(~80 GB partial),user 提示 "为什么不检查 /raid 本地是否有 V4 Flash" 后,kill 了 /raid stage 改指 /dev/shm。

```bash
# 直接用,不需要重新 stage:
export MODEL_PATH=/dev/shm/DeepSeek-V4-Flash
# 验证(必须存在,否则 host 重启过):
ls /dev/shm/DeepSeek-V4-Flash/*.safetensors | wc -l   # 期望 46
```

⚠️ **/dev/shm 在主机重启会清空**。如果新 session 时 /dev/shm 没了,fallback 到 `/raid/data/loncheng-stage/DeepSeek-V4-Flash/`(/raid stage 已有 23 partial shards,要么续传要么重 stage)。

参见:`.perfbot/learnings/20260601T091630-agent.yaml` F001(skill 缺 Phase 0 scan)。

### 2.2 Venv 在 `/workspace/trtllm-venv-gvr-prefill/` 已建好,但需要 **manual .pth + 自制 trtllm-bench wrapper**

`trtllm-machine-local-install` skill 的 Phase 5 (Scenario B) 在本机踩到 G8 NFS cold-cache hang(`pip install -e` 子进程 D-state 5+ min),换成 **manual .pth 方案**。但 manual .pth **不会创建 trtllm-bench console script** — fallback 到 NFS userbase 的 trtllm-bench 时,`#!/usr/bin/python3` shebang + venv unset PYTHONUSERBASE → ModuleNotFoundError 秒退。

**修复方法(已应用)**:venv bin 里写一个用 venv-python 的 trtllm-bench wrapper:

```bash
VENV=/workspace/trtllm-venv-gvr-prefill
cat > $VENV/bin/trtllm-bench <<EOF
#!$VENV/bin/python3
import sys
from tensorrt_llm.commands.bench import main
if __name__ == '__main__':
    sys.argv[0] = sys.argv[0].removesuffix('.exe')
    sys.exit(main())
EOF
chmod +x $VENV/bin/trtllm-bench
```

参见:
- `.perfbot/learnings/20260601T090829-agent.yaml` F001(G8 cold-cache hang → manual .pth)
- `.perfbot/learnings/20260601T093950-agent.yaml` F001(manual .pth **不够**,要 + venv wrapper)

**新机若 host 重启 venv 丢失**,**完整重建步骤**:

```bash
# 1. Create venv (10s)
WORK=/workspace/trtllm-venv-gvr-prefill
WT=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/wt_gvr_prefill
/usr/bin/python3.12 -m venv --system-site-packages $WORK

# 2. Patch activate
cat >> $WORK/bin/activate <<'PATCH'

_OLD_VIRTUAL_PYTHONUSERBASE="${PYTHONUSERBASE:-}"
unset PYTHONUSERBASE
PATCH

# 3. Two .pth files
SP=$WORK/lib/python3.12/site-packages
echo "import sys; sys.path.append('/home/scratch.loncheng_gpu/python-userbase/lib/python3.12/site-packages')" > $SP/_nfs_user_site_readonly.pth
echo "import sys; sys.path.insert(2, '$WT')" > $SP/zz_wt_gvr_prefill_branch.pth

# 4. trtllm-bench wrapper (CRITICAL — without this, bench fails ModuleNotFoundError)
cat > $WORK/bin/trtllm-bench <<EOF
#!$WORK/bin/python3
import sys
from tensorrt_llm.commands.bench import main
if __name__ == '__main__':
    sys.argv[0] = sys.argv[0].removesuffix('.exe')
    sys.exit(main())
EOF
chmod +x $WORK/bin/trtllm-bench

# 5. Verify (do NOT `import tensorrt_llm` — MPI hang 60-180s; use find_spec)
cd /tmp
source $WORK/bin/activate
python3 -c "
import importlib.util, shutil
print('tensorrt_llm:', importlib.util.find_spec('tensorrt_llm').origin)
print('trtllm-bench:', shutil.which('trtllm-bench'))
# Must show wt_gvr_prefill + $WORK/bin/trtllm-bench
"
```

---

## 3. 当前在跑的实验:Run #1 v2(GVR=0 multi-request)

| 字段 | 值 |
|---|---|
| 启动时刻 | 2026-06-01T09:39:08Z |
| 状态 (~09:48Z) | T+9 min,仍在 bench startup/autotuning(import OK,MPI init OK,model load 中) |
| Launcher PID | 5906 (SID 5906,setsid OK,SSH-survivable) |
| 子进程 | bench 5912 → nsys 6017 → trtllm-bench 6044 (State R) |
| RUN_DIR | `dsv4_nsys_indexer_topk_study_prefill/ds4_Flash_ISL68656_OSL2048_BS1_MTP0_TEP_GVRfalse_20260601T093908Z/` |
| Console log | `dsv4_nsys_indexer_topk_study_prefill/run_gvr0_multireq.console.log` |
| Bench log | `<RUN_DIR>/run.log` |
| Launcher 脚本 | `dsv4_nsys_indexer_topk_study_prefill/launch_gvr0_multireq.sh` |
| Multi-request 配置 | NUM_PROMPTS=80, PROMPTS_REPLICATE=80, PROMPTS_INPUT=v3_dataset_seed.jsonl(68656 tokens, ISL同 v3) |
| 预期 wall | ~10-15 min from 09:39(model load + 80 req × ~10 iters) |

**新机 resume 时第一件事:验证 Run #1 v2 是否真的跑完**:

```bash
RD=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/dsv4_nsys_indexer_topk_study_prefill/ds4_Flash_ISL68656_OSL2048_BS1_MTP0_TEP_GVRfalse_20260601T093908Z
ls -la $RD/*.nsys-rep $RD/*.sqlite 2>&1
ps -p 5906 -o pid,etime,stat 2>&1 | tail -2   # 还活着说明在跑;消失说明已结束(看 console.log)

# 关键判定:sqlite iter 500-550 命中数
python3 - <<PY
import sqlite3, re, os
SQ = "$RD/decode_trace_iter0_12_500_550.sqlite"
if not os.path.exists(SQ): print("SQLITE NOT YET FLUSHED"); raise SystemExit
db = sqlite3.connect(SQ)
rows = db.execute("""SELECT s.value FROM NVTX_EVENTS ne JOIN StringIds s ON ne.textId=s.id
    WHERE s.value LIKE '[Executor] _forward_step %'""").fetchall()
fr = re.compile(r'_forward_step (\d+):')
iters = sorted({int(fr.search(v[0]).group(1)) for v in rows if fr.search(v[0])})
in_window = [i for i in iters if 500 <= i <= 550]
print(f"iters: min={iters[0] if iters else 0} max={iters[-1] if iters else 0} count={len(iters)}")
print(f"in [500-550]: {len(in_window)} -- STATUS:", "COMPLETE" if len(in_window) >= 40 else "INCOMPLETE")
PY
```

---

## 4. Run #2 (GVR=1) — 等 Run #1 完成后串行起

Launcher 已 stage:`dsv4_nsys_indexer_topk_study_prefill/launch_gvr1_multireq.sh`(就是 Run #1 launcher GVR=0→1)

```bash
# Run #1 verify COMPLETE 后直接起:
OUT=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/dsv4_nsys_indexer_topk_study_prefill
LAUNCH=$OUT/launch_gvr1_multireq.sh
LOG=$OUT/run_gvr1_multireq.console.log
PID_F=$OUT/run_gvr1_multireq.pid
setsid nohup bash $LAUNCH $LOG </dev/null >/dev/null 2>&1 &
echo $! > $PID_F
disown $!
```

**不要并发** Run #1 + Run #2(4 GPU 共享)。

---

## 5. Multi-request 模式的关键设计依据

Bench-side regression(v3 2026-05-29 → 现在 2026-06-01)使 single-request mode 在 iter ~13 EOS-truncate。我们这次用 multi-request **绕开** per-request truncation:

- 80 requests × ~15 iters/req(prefill chunks + EOS-早退 decode)= ~1200 iters
- PROFILE_ITER_RANGE=0-12,500-550 是 **跨请求的累计 iter 计数器**
- 窗口 [500-550] 落在第 ~35 个请求附近,steady state OK

**前提**:trtllm-bench 多请求模式在 EOS 后**继续下一个请求**,不会全局退出。这是 multi-request mode 的标准行为,验证由 Run #1 v2 完成时给出。

参见 `RESUME_PROMPT.md` §2.1 / `CHECKPOINT.md` §6 / `.perfbot/learnings/20260601T082216-agent.yaml`(根因排查 bench regression)。

---

## 6. **已过时的 RESUME_PROMPT.md 内容**(新会话避免被误导)

| RESUME_PROMPT.md 章节 | 状态 |
|---|---|
| §4.2 Stage Flash 到 /raid | ❌ **跳过** — `/dev/shm/DeepSeek-V4-Flash` 已完整(见 §2.1) |
| §4.3 Build venv(Phase 5 skill) | ⚠️ **G8 hang** — 用 manual .pth + wrapper 路径(见 §2.2) |
| §5 pre-seed v3 dataset 单请求 | ❌ **不行** — multi-request 才行(见 §3、§5) |
| §6.3 Launcher (NUM_PROMPTS=1) | ❌ **过时** — 用 launch_gvr0_multireq.sh / launch_gvr1_multireq.sh |
| §7 Run #2 launcher | ✅ 概念同,launch_gvr1_multireq.sh 已 ready |
| §8 Post-analysis | ✅ 不变 |
| §9 REPORT 填数字 | ✅ 不变 |
| §10 避坑(NFS 冷读/ignore_eos no-op 等) | ✅ 仍有效 |

---

## 7. 本会话产出的 3 个新 learnings(已 git commit)

| YAML | 内容摘要 |
|---|---|
| `.perfbot/learnings/20260601T090829-agent.yaml` | F001 — Phase 5 G8 NFS-stale-handle hang **CAN** still occur on fresh-NFS-client machine(skill 说不会但实测会),manual .pth workaround |
| `.perfbot/learnings/20260601T091630-agent.yaml` | F001 — computelab-hf-stage skill 缺 Phase 0 scan-before-stage,missed `/dev/shm/Flash` 完整副本造成 15 min 浪费 |
| `.perfbot/learnings/20260601T093950-agent.yaml` | F001 — manual .pth alone **不够** — 还要在 venv bin 里写 trtllm-bench wrapper(否则 fallback 到 NFS bench `#!/usr/bin/python3` + PYTHONUSERBASE-unset = ModuleNotFoundError) |

---

## 8. 物理资产清单(NFS, 跨机可见)

| 路径 | 内容 |
|---|---|
| `/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/` | 本 checkout(branch=feat/gvr-v4-dispatch-tuning,HEAD=a2d0a71b34) |
| `/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/wt_gvr_prefill/` | 源码树 feat/gvr-prefill-topk @ 4956083d72(含 prebuilt .so) |
| `/home/scratch.loncheng_gpu/python-userbase/lib/python3.12/site-packages/` | 728 个 NFS deps |
| `dsv4_nsys_indexer_topk_study_prefill/v3_dataset_seed.jsonl` | v3 (May29) tokenize 的 68656-token prompt(本会话保存供 multi-req replicate) |
| `dsv4_nsys_indexer_topk_study/...20260529T052829Z/dataset.jsonl` | v3 原始 dataset(同上,源) |
| `dsv4_nsys_indexer_topk_study/REPORT.md` | v3 decode-only baseline(GVR off vs on Δ% = −66.6%) |
| `dsv4_nsys_indexer_topk_study_prefill/REPORT_PREFILL.md` | 本研究 report 骨架(等填数字) |
| `dsv4_nsys_indexer_topk_study_prefill/analysis/extract_metrics.py` | 派生指标提取脚本(209 行) |

## 9. 物理资产清单(host-local on umb-b300-023)

| 路径 | 内容 | 重启丢? |
|---|---|---|
| `/dev/shm/DeepSeek-V4-Flash/` | 46/46 shards (149 GB) | ✅ 是 |
| `/raid/data/loncheng-stage/DeepSeek-V4-Flash/` | 23/46 partial shards (79 GB, killed) | ❌ 不丢(可续传) |
| `/raid/data/loncheng-stage/DeepSeek-V4-Pro/` | 55 shards(无关本实验) | ❌ 不丢 |
| `/workspace/trtllm-venv-gvr-prefill/` | venv + 2 .pth + trtllm-bench wrapper | ✅ 是(see §2.2 rebuild) |

---

## 10. 新会话恢复步骤(checklist,3 min 内完成)

1. ✅ `hostname` → 期望 `umb-b300-023`(若不同 host,执行 §2.2 重建 venv + §2.1 注意 `/dev/shm/Flash` 在新机可能没)
2. ✅ `ls /dev/shm/DeepSeek-V4-Flash/*.safetensors | wc -l` → 期望 46
3. ✅ `ls /workspace/trtllm-venv-gvr-prefill/bin/trtllm-bench` → 期望存在,shebang `#!/workspace/trtllm-venv-gvr-prefill/bin/python3`
4. ✅ `ps -p 5906` → Run #1 v2 还活着 / 或已退出(看 console.log + RUN_DIR)
5. ✅ **判定 Run #1 v2 结果**(§3 sqlite verify 命令)
6. 走 §4 起 Run #2(GVR=1)
7. 走 RESUME_PROMPT §8 extract metrics,§9 填 REPORT_PREFILL.md

---

## 11. 关键 pid 当前快照

```
SSH-survive launcher (Run #1 v2): PID=5906 SID=5906 etime=~9min
run_nsys.sh                     : PID=5912 (child of 5906)
nsys profile                    : PID=6017 (child of 5912)
trtllm-bench (venv python)      : PID=6044 (state R, doing work)
```

如果新会话发现这些 PID 都消失了,说明 Run #1 v2 已结束 —— 看
`run_gvr0_multireq.console.log` 找 `[launcher] ... exit=` 和 `[nsys] done` 行,
然后用 §3 sqlite verify 判断窗口命中。
