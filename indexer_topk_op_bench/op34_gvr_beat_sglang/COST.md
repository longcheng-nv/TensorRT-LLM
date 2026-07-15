# op34 COST (per-phase GPU-h + notes)
| phase | GPU-h | notes |
|---|---|---|
| iter0 characterization | ~0.05 | host-only analysis, reused §10 parsed data (no re-sweep) |
| iter1 crux (host) | ~0.0 | host Python on cached bundles |
| iter2b re-anchor + NCU CRUX (048) | ~0.15 | anchor + ncu attribution (op26_r0/sglang) |
| iter3 CRUX-A/C (Triton + NCU) | ~0.25 | MLP scan-scaling + collect proxy, NCU replays |
| iter4 build + nsys A/B (harvest+decomp) | ~0.3 | multi-CTA kernel + 4 nsys batches |
| grid regime map (18 nsys batches) | ~0.3 | full 9-ISL×2-model, 3 arms, single GPU ~10min |
| **total (this session)** | **~1.05 GPU-h** | 1 GPU, node 048; converged to double-locked infeasible |
