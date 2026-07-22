# GVR top-K 硬件级仿真（PerfSim/SMART）台账

Started 2026-07-22, session on umbriel-b200-048. 目标 = 算子硬件级仿真拿流水图。
工具链 = compute-infra-agent-skills 仓 (`perf-simulation-service` + `cuda-apic` +
`pic-perfsim`)，服务 = https://sim-service.nvidia.com/api。

## 资产（本目录）

| 文件 | 说明 |
|---|---|
| `apic_gvr_driver.py` | 最小 workload：pinned head (PR#16457 @04a0900ff7, gvrpkg_04a0) GVR kernel × 真实 pro_512k_L30 cell (K=1024, N=131075, cr=4, hit=0.23)，8 warmup + 5 稳态 launch，exact=True。依赖 env: PYTHONNOUSERSITE=1 + /tmp/gvrlayers cutlass450 overlay + CUDA_VISIBLE_DEVICES |
| `gvr_topk_pro512kL30_k11.cuda.tgz` | APIC trace：第 11 次 launch（稳态），grid 8×1×1 block 512×1×1，B200 silicon 抓取。**服务端已存为 `/trace/1748`**（复用免上传） |
| `cuda_apic_captured_traces.csv` | 抓取验证：status=PASSED |

## 抓取要点（可复制的方法论）

1. kernel listing 空捕获（`--capture_only --knob DumpControl="(func=__APIC_NO_MATCH__ & frange=0:0)"`）
   → GVR = 单 kernel，符号 `kernel_cutlass_gvr_topk_kernel_…`（cuteDSL JIT 名）。
2. 正式抓取：`--knob DumpControl="(func=kernel_cutlass_gvr_topk & frange=10:10)"`（第 11 次 launch）。
3. **GOTCHA：GVR 输出 tie/arrival-order 非确定 → post-process refcheck 逐字节比对必 FAIL；
   必须加 `--pp_args="-NoRefCheck"`**（对性能仿真无影响）。加了以后 PASSED。
4. APIC_BIN=/home/scratch.svc_compute_arch/release/cuda_apic/linux64/release/latest。

## 服务端提交记录

| Job | Flow | 结果 |
|---|---|---|
| 986099603 | smart | **FAIL**：`cudaReplayer -ReportInfo` 后静默 exit 1（4s，stdout/stderr 无错误） |
| 986109781 | smart (原样重试) | **FAIL**：同签名 → 确定性 |
| 986111834 | **perfsim (对照)** | **跑通**：APIC_Capture PASS → Trace3D_Gen …；dashboard = https://compute-nexus.nvidia.com/workflows/runs/139023 |

**结论：trace 无罪；smart flow 服务端 bug**（`-enableMorph`/`-pic` 均为 flow.smart
合法 flag；June-4 版 cudaReplayer 本地对本 trace ReportInfo exit 0，apicInfo.yml 完好）。
待办：把 smart 静默失败报给服务 owner（skill 作者 bshan@nvidia.com）；修复后同
`/trace/1748` 一键重投 smart 拿 SM 级流水。

## 流水图产出位置（perfsim 完成后）

- PIC Inspector 链接：`/flow/status` 里 PIC_Analysis section 的 `links`（主查看器）
- NVPDM 链接：PerfSim section `links`
- PFM 文件：`GET /executions/986111834/pfm`
- 后续定量拆解：pic-perfsim 套件（tma-latency-breakdown / classify-warps / tma-barrier-analysis）
