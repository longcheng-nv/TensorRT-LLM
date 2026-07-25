#pragma once
#include <cuda_runtime.h>

void champion_launcher(
    const float* logits, const int* pre_idx, int* indices,
    int npad, int n_valid, int k, cudaStream_t stream);

void histogram_launcher(
    const float* logits, const int* pre_idx, int n_valid, int k,
    int* indices, cudaStream_t stream);
