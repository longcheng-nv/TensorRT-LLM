# SESSION_HANDOFF — op40_omni_gvr (2026-07-23, from umb-b200-239)

跨机迁移自包含交接。原会话在 umb-b200-239 收敛收口；本文件 + RESUME_PROMPT.md
+ REPORT.html + git 历史 = 完整状态，无需原会话任何内存。

## 0. 战役状态：CONVERGED（无在飞任务）
- 所有 GPU 任务已完成，无后台进程需要接管；无半成品网格（全部批级 marker 完整）。
- 终判：ship arm v7 = gm 1.1250 vs base @e612fc2f38，0 回退，865/865 exact，
  gate 138/138。1.60 冲刺双锁不可达（REPORT.html §2）。
- 全部结论/数据/脚本已 commit 到分支 `omni/op21-gvr-prod`
  （最后 op40 提交：598f730a1f）。

## 1. 新机器上恢复工作的步骤
1) 登录任一可见同一 NFS 的机器（B200 节点若需复测）：
   `cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM`
   `git log --oneline -3 -- indexer_topk_op_bench/op40_omni_gvr`  # 应见 598f730a1f
2) 启动 claude code，首条 prompt 直接给：
   「读 indexer_topk_op_bench/op40_omni_gvr/SESSION_HANDOFF.md 和
   RESUME_PROMPT.md，接管 op40 战役的后续工作」
3) 环境重建（/tmp overlay 是机器本地的，每台新机器必做一次）：
   `mkdir -p /tmp/gvrlayers/cutlass450 && cp -r \
    /home/scratch.loncheng_gpu/python-userbase/lib/python3.12/site-packages/nvidia_cutlass_dsl \
    /tmp/gvrlayers/cutlass450/`
   所有 python 调用前缀：`PYTHONNOUSERSITE=1 \
    PYTHONPATH=/tmp/gvrlayers/cutlass450/nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450`
4) 冒烟验证（~5 分钟，一张空闲 GPU）：
   `CUDA_VISIBLE_DEVICES=<g> python3 scripts/gate40.py --arms v7 | tail -3`
   预期 GATE GREEN 69/69。
5) 若要复测性能：新节点必须整格重测（锚协议——绝对 µs 不跨节点），
   `GPUS_LIST="..." bash scripts/drive_ab40.sh base,v7 <tagdir>` 后
   `python3 scripts/parse_ab40.py <tagdir> --ref base --var v7`（后台跑，>2min）。

## 2. 待人类裁定（原会话结束时的两个开放决策）
a) 是否把 v7 四 flag 作为 follow-up PR（stacked on #16457）移植上游
   （含 p2_radix_fallback 缺陷修复 + plateau/neartie fixtures）。
b) 是否单独向上游报告基线缺陷（复现脚本 scripts/repro_plateau.py、
   probe_neartie_flake.py；三类缺陷见 REPORT.html §3）。

## 3. 不在 git 里的东西（如需要去哪找）
- nsys/ncu 原始 rep/sqlite：gitignored，仅存 umb-b200-239 本地
  results/ 下（cells.csv 等解析产物已提交，通常无需原始 rep）。
- /tmp/gvrlayers overlay：机器本地，按 §1.3 重建。
- claude 持久记忆（~/.claude/projects/.../memory/）：机器本地；其 op40 条目
  内容已镜像到本文件与 RESUME_PROMPT.md（自包含）。若新机器共享同一
  /home/loncheng 则记忆自动可见。

## 4. 关键纪律（新会话必读，血泪账本浓缩）
- 本仓有并发会话提交：commit 撞 ref-lock 后必须重新 git status 核对再 add
  （曾发生 index 竞态互吞，修复 commit 817a943d32）。
- 计时网格运行期间禁止任何 GPU 探针（双 driver 污染）。
- 种子一律 zlib.crc32（python hash() 跨进程盐化，曾伪造"非确定性"）。
- nsys/ncu 一律 `env -u GITHUB_TOKEN -u HF_TOKEN`；*.sqlite/*.nsys-rep 永不提交。
- KF 防火墙（对照实验约束）：不读 kf_campaign/、op37_bs_scaling/、
  op38_r3v11_bs/、op39_gvr_bsx/。
- ncu 直连二进制：
  /opt/nvidia/nsight-compute/2026.1.1/target/linux-desktop-glibc_2_11_3-x64/ncu
  （/usr/local/bin/ncu 符号链接损坏），加 `-k "regex:gvr_topk"` 过滤。
