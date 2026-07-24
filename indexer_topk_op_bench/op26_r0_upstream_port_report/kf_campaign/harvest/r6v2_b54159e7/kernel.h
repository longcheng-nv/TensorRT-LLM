#pragma once

#include <cuda_runtime.h>

void gvr_topk_launcher(
    const float* logits,
    const int* pre_idx,
    int n_valid,
    int* indices,
    int batch,
    int npad,
    int k,
    cudaStream_t stream);
