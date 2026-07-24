#pragma once
#include <cuda_runtime.h>

void gvr_topk_launch(const float* logits, const int* pre_idx, int* indices,
                     int batch, int stride, int n, int k,
                     cudaStream_t stream);


