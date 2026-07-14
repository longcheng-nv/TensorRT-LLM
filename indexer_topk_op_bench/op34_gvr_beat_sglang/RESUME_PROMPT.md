# op34 RESUME — GVR-skeleton top-K to beat sglang_v2 by 30% (real v4cap, BS=1)
Node-agnostic (NFS-shared). Workspace: indexer_topk_op_bench/op34_gvr_beat_sglang/

## 1-min context
Goal: GRAND BS=1 fp32 cold-L2 geomean(new GVR) <= sglang/1.30 = 6.06us (need 2.03x over
op26_r0auto 12.31us). Start kernel op26_r0auto; keep GVR threshold skeleton. sglang is the rival.
Phase-1 done (analysis/): gap map + data props + iter1 crux GO. Prior walls in FALSIFIED/WALLS.md.

## preflight (30s)
cd indexer_topk_op_bench/op34_gvr_beat_sglang
git rev-parse --abbrev-ref HEAD   # omni/op21-gvr-prod
python3 -c "import torch,cutlass,flashinfer;print(flashinfer.__version__)"  # 0.6.11
python3 -c "import sys;sys.path.insert(0,'../harness');import sglang_v2_op,real_data_v4cap"
nvidia-smi --query-gpu=index,temperature.gpu,memory.used --format=csv,noheader  # idle<50C, mem~0
# kernels JIT-rebuild per machine on first run.

## where we are
- iter0 CHARACTERIZATION done. iter1 (user idea: single-scan bracket + fast-write certain-winners)
  rung-0 CRUX = GO (fast-write 0.89, band/K 0.29, happy 100% except degenerate pro/4k).
- NEXT: implement iter1 kernel (rung 3): GvrOp34SingleScanKernel subclass of GvrOp26R0Kernel,
  fold P3 collect + certain-winner fast-write into R0 multi-count pass; P4 over contested band only;
  fallback to op26_r0 path when no happy rung (always correct — dispatch on MEASURED counts).
  Behind a flag (default byte-identical). Gate (tie-aware) -> nsys A/B vs sglang + op26_r0auto.

## measurement (to build in scripts/)
BS=1 real-data cold-L2 nsys pure-kernel A/B, arms {op26_r0auto[base], sglang_v2[rival], op34_new}
per (model,ISL,layer) fp32. IDENTICAL protocol to sweep_op22_v4cap.py (measure_cell, cold-L2 evict,
cudaProfilerApi window). Report [worst,real,best] never one axis; single-GPU paired A/B for ship.

## gotchas
- NEVER commit *.nsys-rep/*.sqlite (env tokens). setsid long runs; pkill-triple to stop.
- op26_r0 kC = 3072@K512 / 5120@K1024 (kC>=5K = 16-bit tie-safety contract, do not shrink kC itself;
  the contested BAND is separate from kC).
- exactness = tie-aware value-multiset vs same-dtype torch.topk (real_data_v4cap.value_metrics).
