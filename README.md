# KF R5 champion `gvr-topk-r5-combined` (batched GVR, BS 1-1024)

Batched GVR indexer top-K (CUDA C++, sm_100a B200), final winner of campaign
gvr-topk-bs2x v3 (befh5fh2595es8ztpcg0nmq6q8, round 5).
Verdict (nsys cold-L2 paired, real §7b data, vs PR#16457 head @04a0900ff7
NATIVE BATCH): 750-case BS2-1024 geomean 0.986 (win 359/750; up to 4.79x at
small-N high-BS; BS2-8 pro 1.12-1.25x), BS=1 865-case 1.2233. Exact everywhere
(1500/1500 + 865). Campaign target (+100% avg) NOT met — see R5_CLOSEOUT.md.
Deployment: use only in its winning regimes (b<=8 K>=1024, small-N any b);
b=1 should use the R4 champion; elsewhere keep the in-tree kernel.
Entry: main.cpp::run(logits[b,npad] f32, pre_idx[b,k] i32, n_valid, indices[b,k] out).
