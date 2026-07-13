# op28 CHECKPOINT — 2026-07-13 (umbriel-b200-027)

## 任务
拉取最新 SGLang v2 top-K (sglang@main 2026-07-13) + FlashInfer top_k (0.6.11==main)
为单算子可测状态,用与 op22 REPORT 完全相同的输入数据 (bundles_rr 逐字节) 和
条件 (nsys 纯 kernel NVTX 投影, 512MB cold-L2 evict, 20 cold/50 warm, B200) 测性能。

## 状态 (按序恢复)
1. DONE — vendor sglang_v2: `../ops/sglang_v2/` (kernels verbatim; tvm-ffi host
   → torch ext; PDL+cluster launch 保真; wrapper `../harness/sglang_v2_op.py`)。
   smoke ALL EXACT (K 512/1024/2048, 含 2-kernel 持久 cluster 路径)。
2. DONE — flashinfer: 本机 0.6.11 topk.py 与 main 逐字节一致; B200 clusters
   快路径 kernel 亦一致 (topk.cuh 的 154 行 diff 只在 radix 回退路径)。
   源码快照 `../ops/flashinfer_topk/`。运行走已装包 JIT (~/.cache 已编译)。
3. DONE — harness: ops_ext.py / sweep_op28.py / drive_nsys_op28.sh /
   gate_op28.py / parse_op28.py / gen_results_op28.py / measure_plan_cost.py。
   gate 459/459 绿 (gate_op28.log)。commit @3a35b83301。
4. DONE — 27/27 nsys 批次 (GPU2-7; GPU0/1 当时被占):
   `../results_b200_op28/{real,best,worst}/` 每批 .done_* 标记, jsonl 零 error。
   驱动日志 op28_gpu{2..7}.log。
5. DONE — parse + gen 完成 (906 cells, 0 error, 锚漂移 med 0.9984); 判决见 RESULTS_SUMMARY.md。
   results.jsonl, 然后 gen_results_op28.py → op28_{bs,seqlen,bs_hugeN}_data.csv
   + RESULTS_SUMMARY.md。若中断: 直接重跑
   `python3 parse_op28.py ../results_b200_op28 && python3 gen_results_op28.py ../results_b200_op28`
   (幂等; parse 需 env -u GITHUB_TOKEN -u HF_TOKEN 因 nsys 会导出 sqlite)。
   已知待验证: bs_hugeN 的早期 parse 出现 ranges=0 (rep 未写完时解析所致),
   完整重跑后必须确认 3 个 hugeN rep ranges>0。
6. DONE — RESULTS_SUMMARY.md 含 vs 生产臂 (op21_hls/op26_r0auto/op25_hls) 锚换算表 + caveats。
   REPORT html。锚换算: gvr_cutedsl 对 op22rr CSV (`../op22_temporal_fixed_hr_bench/
   op22rr_{bs,seqlen}_data.csv`), us_adj = us_rr / (gvr_rr/gvr_op28)。

## 关键口径
- sglang_v2 timed call = transform (1-2 kernel); plan UNTIMED (单测 ~7µs,
  摊 61 层 ≈0.11µs/层, measure_plan_cost.py)。
- 双 kernel cell (N>64K & 30<BS≤512): `us`=kernel 时间和 (与历史可比,
  重叠时高估), `us_cold_span`=NVTX GPU 投影跨度 (诚实墙钟)。早期数据
  span>sum ~5µs = 两 kernel 间隙 (PDL 下 main kernel 尾段等待)。
- 早期抽查 (real K=512): sglang_v2 大幅最快 (64K BS1: 9.0µs vs gvr 32.8 /
  radix 15.7 / 旧 sgl 25.3 / fi 12.5); fi_i32 略快于 fi 公共 API。

## 快速恢复命令
```bash
cd indexer_topk_op_bench/op28_ext_topk
ls ../results_b200_op28/*/.done_* | wc -l   # 应为 27
env -u GITHUB_TOKEN -u HF_TOKEN python3 parse_op28.py ../results_b200_op28
python3 gen_results_op28.py ../results_b200_op28
```
7. DONE — REPORT.html 并入 sglang_v2 + flashinfer_topk 两臂 @92bca18ffd;
   update_report_op28.py 为新 last-writer (append-on-top, 幂等)。
