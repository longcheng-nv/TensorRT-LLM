# op26 RESUME — GVR(cuteDSL)/GVR-mCTA(PR#15198) 上移植 log-falsi + rank-scatter

> **2026-07-09 迁移**:门禁两臂全绿(291+291);nsys 战役从 b200-036(GPU1 病卡,
> 只跑了 op26a 4/81)迁往 8 卡 B200 机器 —— 接管指南 = `TAKEOVER_8GPU_PROMPT.md`,
> 036 已清场,勿在 036 重启 driver。

接手时先读 `PLAN.md` + `ITERATIONS.md`。分支 omni/op21-gvr-prod,主战场
umbriel-b200-038(双 B200,均健康)。

## 战役状态检查点

1. **实现 DONE**:`src/gvr_op26_op.py`(op26_1cta / op26_mc 两臂)+
   harness 注册(`harness/sweep_op26.py`、sweep_nsys.py build_call、
   sweep_op22rr.py ARMS_EXTRA)。smoke 27/27。
2. **门禁**:`python3 gate_op26.py`(GPU 上 ~30min)必须 0 fail 0 err
   才可进 nsys。
3. **nsys 全网格**(81 批/卡,~6h,双卡并行):
   ```bash
   cd ../op22_temporal_fixed_hr_bench
   setsid env OUT=results_b200_op26a GPU=0 \
     OP22RR_ARMS="gvr_cutedsl,op26_1cta" ./drive_nsys_op22rr.sh \
     > op26a_gpu0.log 2>&1 < /dev/null &
   setsid env OUT=results_b200_op26b GPU=1 \
     OP22RR_ARMS="gvr_multicta_cutedsl,op26_mc" ./drive_nsys_op22rr.sh \
     > op26b_gpu1.log 2>&1 < /dev/null &
   ```
   marker 粒度幂等,重跑同命令自动续传。完成判据:各 27 个 .done marker
   (3 scen × 3 sweeps × 3 K × 3 dt / 每卡全 dtype)→ 81 批/卡。
4. **parse**:`python3 parse_op22_cached.py ../results_b200_op26a`(+ op26b)。
5. **报告更新**:写/跑 `update_report_op26.py` —— 必须以
   update_report_radix.py 为底扩展(自包含 last-writer,重导
   mc+op25+radix+op26 全部;见其 GOTCHA:任何旧 updater 后跑会抹掉别的臂)。
   op26_1cta 锚迁移 via 本地 gvr_cutedsl;**op26_mc 链式锚迁移** via
   本地 gvr_multicta_cutedsl × mc_adj(074→orig 刻度)。
   QA 门:script 数=2、exactness 全 ok、锚漂移 med≈1.0(两条链都要报)。

## 已知 gotcha(继承 + 新增)

- nsys 由 driver 内部 `env -u GITHUB_TOKEN -u HF_TOKEN`;*.sqlite/*.nsys-rep
  永不 add/commit。
- 停 sweep:pkill 三连(drive_nsys_op22rr / sweep_op22rr / "nsys profile"),
  勿用 TaskStop。
- **fb_fix 教训**(iter1):P2 bracket 的 cnt_lo 可能是未实测的 P1 种子
  (1.25K),任何 falsi/插值都不得信任未实测计数 —— 详见 ITERATIONS.md。
- op13 调度表出处 = op13_gvr_p2cand/src/gvr_p2clog_op.py dispatch_p2c_v2;
  K512 log 已证伪,勿"顺手"打开。
- 16-bit P2 保持基线(op13 无证据规则);rank-scatter 16-bit 只在 BS≥256。
