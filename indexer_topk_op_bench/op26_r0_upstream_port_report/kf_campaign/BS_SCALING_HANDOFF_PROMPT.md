# BS-scaling 实测接力任务(贴入另一台服务器的新 Claude Code 会话)

Prepared 2026-07-22 by the R4 campaign session on umbriel-b200-027.
**隔离要求:目标机器 ≠ umbriel-b200-027**(本机正在跑 R4 判决,严禁占用);
工作目录/产物一律新建 `indexer_topk_op_bench/op37_bs_scaling/`,
**只读引用** `op26_r0_upstream_port_report/kf_campaign/`,不得写入或改动其中任何文件。

---

```
/goal — KF champion GVR 算子 BS-scaling(BS=1→1024)性能实测 vs 生产 GVR PR head

## 0. 背景与对象(先读,勿重测已有结论)

- 对象 A(champion): KernelFactory R4 冷启动战役 round-3 胜者 `r3_v11`
  (kernel id 7d8272b7…),CUDA C++ torch-extension。源码(NFS 共享,直接可读):
  indexer_topk_op_bench/op26_r0_upstream_port_report/kf_campaign/harvest/r3_7d8272b7/
  ├── gvr.cu     — 三路 dispatch: npad≤12288 direct 单 CTA×1024;
  │                12K<npad≤262144 寄存器驻留 GVR(1/4/8/16-CTA cluster×512,
  │                整行一次读进寄存器,secant 多 pass 零重扫);
  │                >262144 流式 16-CTA cluster
  └── main.cpp   — 入口 run(logits[1,npad] f32, pre_idx[1,k] i32, n_valid int,
                   indices[1,k] i32 out),编译方式见 quick_ab.py::build_candidate
                   (torch.utils.cpp_extension, -O3, sm_100a)
- 对象 B(baseline): PR#16457 pinned head @04a0900ff7 的生产 GVR(cuteDSL),
  独立打包 kf_campaign/gvrpkg_04a0/(sys.path 加该目录后
  `from gvrpkg.top_k.gvr_topk_decode import GvrTopKKernel`;
  launch(logits, pre_idx, seq_lens, out, K, compress_ratio=cr),
  **原生支持 num_rows=BS>1**,cr: K=2048→1, K=512/1024→4, seq_lens=n*cr)。
- 已知 BS=1 结论(不要重测): 865 真实格 nsys cold-L2 配对,champion vs head
  geomean 1.6315×,865/865 exact,唯一回退 pro_64k_L38(hit0.27, 0.963)。
  分组数据见 kf_campaign/grid_r4r3bg.csv(uuid/model/isl/layer/N/K/hit/
  pr_cold/cand_cold/speedup_cold 列)。
- **核心问题**: champion 是 BS=1 特化 kernel(单行、单 launch、16-CTA 顶配),
  生产 decode 会以 BS 行批量调用。BS 增大后 champion 的"每行一次 launch"
  模式 vs 生产 kernel 的原生批处理,交叉点在哪里?一句话:**champion 的
  BS-scaling 曲线什么时候输给 head?**

## 1. 测量矩阵

- BS ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024}
- (K, N) 代表点(覆盖三模型 × 小/中/大 N,共 ~9 格):
  K=512:  N ∈ {8195, 32771, 262127}   (V4-Flash)
  K=1024: N ∈ {8195, 32771, 262127}   (V4-Pro, 最高优先级)
  K=2048: N ∈ {16399, 65551, 163775}  (V3.2)
- 两臂调用约定:
  A champion: 对 BS 行在**同一 stream 上顺序 launch BS 次**(行 i 传
    logits[i:i+1] 等切片)— 这就是它上生产的真实形态,如实测量,
    不允许给它加批处理改造(那是另一个工程,不在本任务内)。
  B head:    GvrTopKKernel.launch 一次批量调用 [BS, npad]。
  两臂均整批计时(BS 行全部完成 = 一个样本)。
- 数据: BS>1 用**真实行堆叠**——从 §4 real data 同 (model,N) 的不同 layer
  行循环堆满 BS(loader: indexer_topk_op_bench/harness/real_data_v4cap.py /
  real_data_v32.py, get_bundle(model, isl, layer, "fp32"));不足处循环重复
  但每行 preIdx 用其对应行的,禁止全批同一行(L2 会作弊)。
  若 loader 在目标机器不可用,退回 indexer-topk-temporal-synth skill 合成
  (标注 synth,与 real 分开报告)。

## 2. 测量纪律(硬约束,违反即作废)

- B200/B300 均可,但全程单机单卡种族背靠背配对(A/B 同 GPU 交替);
  nsys 纯 kernel 时间(BS 行总 GPU 时间/样本),cold-L2(512MB evict,参考
  kf_campaign/nsys_ab.py 的 measure_cell 协议),CUDA-event 只做粗筛。
- launch 前后 `pgrep` + `nvidia-smi` 双查空闲;外来负载会跳卡(本战役实测),
  每个 BS 批次前复查;长跑 setsid;tag 不复用;
  profiling 一律 `env -u GITHUB_TOKEN -u HF_TOKEN`;*.sqlite/*.nsys-rep 永不入 git。
- exactness 每格每 BS 全行校验(tie-robust index-set,参考 quick_ab.py::exact),
  任何 inexact 即该臂该格 FAIL 并如实报告。
- 环境: PYTHONNOUSERSITE=1 + PYTHONPATH=/tmp/gvrlayers/cutlass450/
  nvidia_cutlass_dsl/python_packages:/tmp/gvrlayers/cutlass450;
  /tmp/gvrlayers 不存在则按 kf_campaign/README.md 一行命令从 NFS userbase 重建;
  cutlass 必须 4.5.0(4.6 破 make_fragment)。champion 编译需 sm_100a。

## 3. 交付

- op37_bs_scaling/RESULTS.md: 每 (K,N) 的 speedup-vs-BS 曲线表 +
  交叉点(champion 落后于 head 的最小 BS)+ per-BS exactness 记录;
- op37_bs_scaling/bs_data.csv: 原始 (K, N, BS, arm, cold_us, warm_us, exact);
- 结论段: champion 若要上生产,BS 门限建议(dispatch 到 head 的 BS 阈值);
- 分析工作随时 checkpoint 落盘(脚本固化 .py、结论写 md、及时 commit;
  commit 只加 op37_bs_scaling/ 下文件,不碰 kf_campaign/)。

## 4. 已知陷阱速查

- b200-019 GPU0 坏散热 / b200-035 GPU0 热节流 / b200-036 GPU1 坏散热(避开);
- 本 sandbox ps/nvidia-smi 看不到他 namespace 进程,但显存/util 读数真实;
- champion 的 direct 路径 npad≤12288 不读 pre_idx —— BS 堆叠时照常传;
- head 臂 BS>1 时 seq_lens tensor 形状 [BS](每行 n*cr);
- cuteDSL 首次 launch JIT ~30-60s/变体,计时前充分 warmup(≥10 次)。
```
