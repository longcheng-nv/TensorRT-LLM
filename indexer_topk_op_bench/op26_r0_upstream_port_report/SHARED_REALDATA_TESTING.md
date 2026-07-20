# Shared real-data operator testing — REPORT §4 dataset handoff

Purpose: let a third party (same NFS filesystem) benchmark ANY top-K
operator on exactly the REPORT §4 "Real-data results" inputs, for
multi-party cross-validation. Everything below is directly usable in a
Claude Code session.

## 1. The dataset (canonical access = the loaders, not raw files)

```python
import sys
sys.path.insert(0, "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/harness")
import real_data_v4cap as RV4      # V4 Flash (K=512, cr=4) + V4 Pro (K=1024, cr=4)
import real_data_v32 as RV32       # V3.2 (K=2048, cr=1)

bd = RV4.get_bundle("flash", "64k", 22, "fp32")   # (model, ISL, layer, dtype)
# bd keys:
#   logits   [1, Npad]  device tensor; ONLY [:, :bd["N"]] is valid — the pad
#            tail is dtype-min/garbage and MUST NOT be scanned/selected
#   preIdx   [1, K] int32 — previous decode step's top-K (temporal warm-start
#            hint; cr=4 models use it directly, no offset)
#   N        valid indexer length (= ISL/cr for V4, ≈ISL for V3.2, capped by
#            physical kv_len at v32/256k -> 163775)
#   K, cr, hit_rate, ref (reference top-K indices)
```

§4 headline cells (BS=1, fp32, bench layer per model):

| model | layer | ISL rungs | K | cr | N per rung |
|---|---|---|---|---|---|
| flash | L22 | 4k,8k,...,1024k (9) | 512 | 4 | 1027, 2051, 4099, 8195, 16387, 32771, 65538, 131075, 262127 |
| pro | L30 | 4k,...,1024k (9) | 1024 | 4 | 1027, 2051, 4099, 8195, 16387, 32771, 65539, 131075, 262127 |
| v32 | L34 | 4k,...,256k (7) | 2048 | 1 | 4111, 8207, 16399, 32783, 65551, 131087, 163775 |

Per-layer extension (all captured GVR-active layers: flash 21 / pro 30 /
v32 58) is available through the same `get_bundle(..., layer, ...)`; the
865-cell inventory is `op26_r0_upstream_port_report/real_3arm_layers_full.csv`.
For v32 layers beyond {14,34,54}: set `RV32.BENCH_LAYERS = list(RV32.LAYERS_ALL)`
before first use (slims are already rebuilt all-layer on NFS).

Raw captures (for independent re-derivation; loaders are preferred):
- V4: /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/E2E_exp/indexer_decode_capture/data/{flash,pro}/ISL_*/layer_*/
- V3.2: .../indexer_decode_capture/data/v32/ISL_*/layer_*/
- slim caches: indexer_topk_op_bench/op22_temporal_fixed_hr_bench/data_{v4cap,v32}/

## 2. Correctness contract (what "exact" means here)

Output = K unique indices into [0, N); gathered VALUE multiset must equal
`torch.topk(logits[0,:N].float(), K).values` (sort both, torch.equal).
Tie handling: any index choice among bit-equal values is exact. fp32 is the
canonical dtype for cross-operator comparison (16-bit inputs exist but ties
and value-set semantics get murkier — declare dtype explicitly).

## 3. Timing protocol to be comparable with REPORT numbers

- nsys, cold-L2 canonical: evict a 512MB buffer OUTSIDE the timed NVTX
  range before each cold rep; 20 cold + 50 warm reps per cell.
- `us` = per-NVTX-range kernel-time sum (cold median across reps);
  multi-kernel / PDL-overlapped ops must ALSO report the projected NVTX
  span (`nvtx_gpu_proj`) — REPORT uses span for sglang_v2.
- Verdict-grade runs at <=2 concurrent nsys per node (8-way is screening
  only; saturated-node nsys fabricates +-15% outliers both ways).
- Ready-made harness (recommended — gives byte-compatible jsonl):
  `indexer_topk_op_bench/harness/sweep_nsys.py::measure_cell` for NVTX
  ranges, `indexer_topk_op_bench/report/parse_nsys_full.py::parse_rep` for
  extraction. A minimal end-to-end example of a same-process A/B sweep with
  this protocol: `op26_r0_upstream_port_report/p4f1_harness/prhead_rival_ab.py`
  (+ its parser `prhead_rival_parse.py`).
- CUDA-event timing is NOT comparable (warm-L2 understates 25-35%).

## 4. Reference numbers to compare against (same cells)

- `op26_r0_upstream_port_report/real_3arm.csv` — base/pr/op26 µs (§4 table).
- `op26_r0_upstream_port_report/rival_long.csv` — sglang_v2 (us_span),
  flashinfer, radix rows at the same cells.
- Cross-node caution: those were measured on b200-044/094; include a
  common anchor op (e.g. run one shared arm) or compare RATIOS not
  absolute µs; observed cross-node drift med ~1.01-1.03 on B200 fleet.

## 5. Paste-ready prompt for the other party's Claude Code session

---
在 NFS 数据集上为算子 <YOUR_OP> 做 REPORT-§4 同数据性能测试:
读 /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op26_r0_upstream_port_report/SHARED_REALDATA_TESTING.md,
按其 §1 用 loader 加载 25 个 bench cell(flash L22 / pro L30 / v32 L34,
BS=1 fp32),对 <YOUR_OP> 逐 cell 按 §3 协议(nsys cold-L2,20+50 reps,
NVTX,≤2 并发判决)测时并按 §2 契约折叠精确性检查,输出 jsonl + 对照 §4
参考表(real_3arm.csv / rival_long.csv)给出逐 cell 与 geomean 对比。
注意:logits 只有前 N 个有效(pad 尾是垃圾,严禁扫描);多 kernel/PDL 算子
必须同时报 NVTX 投影 span。
---
