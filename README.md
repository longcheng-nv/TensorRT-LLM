# KF R4 coldstart champion `r3_v11` (BS=1)

GVR (guess-verify-refine) indexer top-K decode kernel, CUDA C++ (sm_100a B200).
Second-lineage KernelFactory campaign `gvr-topk-cold60` (pra6srbd7h4pqecqbgxgm15rgg),
round-3 winner, kernel id 7d8272b7…

- Baseline / denominator: TensorRT-LLM PR#16457 pinned head
  04a0900ff7c233a03e95dc8c35321c37c256d627 (in-tree cuteDSL GVR).
- Verdict (local umbriel-b200-027, nsys cold-L2 paired, 865 real decode cells,
  BS=1 fp32): geomean 1.6315x, 865/865 tie-robust exact, 1 regression
  (pro_64k_L38, hit 0.269, 0.963 by 60-rep adjudication).
- Entry: main.cpp::run(logits[1,npad] f32, pre_idx[1,k] i32, n_valid, indices[1,k] i32 out),
  torch extension, -O3, sm_100a.
- Structure: npad<=12288 direct single-CTA exact path; <=262144 register-resident
  GVR (1/4/8/16-CTA cluster x 512thr, row read once into registers); >262144
  streaming 16-CTA cluster. GVR skeleton: pre_idx hint CCDF rung ladder ->
  8-threshold/pass log-secant bracket shrink -> DSMEM collect -> exact radix
  refine + tie-ticket writeback.
