#pragma once
#include <cuda_runtime.h>
void topk_launcher(const float* logits, const int* pre_idx, int* indices, int b, int n_valid, int npad, int k, cudaStream_t stream);
