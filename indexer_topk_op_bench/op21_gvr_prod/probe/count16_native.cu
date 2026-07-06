// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// op21 iter9 microbench: native 16-bit compares in the M=4 fused count
// ladder (GVR P2 primitive) vs the current cvt->fp32 path.
//
// Variants (M=4, 256-bit loads = 16 elems, 4-way unrolled, matching the
// production ladder's structure):
//   a  : load 16-bit, convert each to fp32, 4x (FSETP + pred IADD)  [current]
//   p1 : PTX set.ge.u32.{f16x2|bf16x2} -> per-half 0xFFFF mask, packed
//        u32 accumulate (c += m & 0x00010001), split at the end
//   p2 : __hge2 (1.0/0.0 per half) + __hadd2 16-bit accumulate; exact
//        while per-thread-per-half counts <= 256 (bf16) / 2048 (fp16) —
//        holds for N/threads <= 512 (our slices: N<=262144, T=1024)
//
// Thresholds are QUANTIZED TO THE DTYPE GRID first (the production plan:
// P1b emits thr_q = fp32(dtype(thr)) so 16-bit-domain compares are bit-
// equivalent to the fp32 compares every other phase performs). Variant
// counts are asserted equal to variant a — validating that equivalence.
//
// argv: <fp16|bf16> <N> <BS> <a|p1|p2>   (T=1024 fixed, M=4 fixed)
// Prints median cold-L2 (512MB evict) CUDA-event us over 40 reps.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

constexpr int M = 4, TT = 1024, VECW = 16;

// ---- 16-bit pair typed helpers -------------------------------------------
template<typename T> struct Pair;
template<> struct Pair<__half>       { using t2 = __half2; };
template<> struct Pair<__nv_bfloat16>{ using t2 = __nv_bfloat162; };

__device__ __forceinline__ unsigned set_ge_mask(__half2 a, __half2 b){
  unsigned d, ua = *reinterpret_cast<unsigned*>(&a), ub = *reinterpret_cast<unsigned*>(&b);
  asm("set.ge.u32.f16x2 %0,%1,%2;" : "=r"(d) : "r"(ua), "r"(ub));
  return d;
}
__device__ __forceinline__ unsigned set_ge_mask(__nv_bfloat162 a, __nv_bfloat162 b){
  unsigned d, ua = *reinterpret_cast<unsigned*>(&a), ub = *reinterpret_cast<unsigned*>(&b);
  asm("set.ge.u32.bf16x2 %0,%1,%2;" : "=r"(d) : "r"(ua), "r"(ub));
  return d;
}
__device__ __forceinline__ float to_f(__half v){ return __half2float(v); }
__device__ __forceinline__ float to_f(__nv_bfloat16 v){ return __bfloat162float(v); }
__device__ __forceinline__ float2 pair_to_f2(__half2 v){ return __half22float2(v); }
__device__ __forceinline__ float2 pair_to_f2(__nv_bfloat162 v){ return __bfloat1622float2(v); }

// ---- variant a: cvt -> fp32 compares (current production semantics) ------
template<typename T>
__global__ void k_cvt(const T* __restrict__ row, int N,
                      const float* __restrict__ thr, int* __restrict__ out){
  int tid = threadIdx.x;
  float t[M]; int c[M];
  #pragma unroll
  for(int m=0;m<M;m++){ t[m]=thr[m]; c[m]=0; }
  int step = TT*VECW, i = tid*VECW;
  for(; i+3*step+VECW<=N; i+=4*step){
    #pragma unroll
    for(int u=0;u<4;u++){
      int4 v0 = *reinterpret_cast<const int4*>(row+i+u*step);
      int4 v1 = *reinterpret_cast<const int4*>(row+i+u*step+8);
      const T* h0 = reinterpret_cast<const T*>(&v0);
      const T* h1 = reinterpret_cast<const T*>(&v1);
      #pragma unroll
      for(int j=0;j<8;j++){
        float f0 = to_f(h0[j]), f1 = to_f(h1[j]);
        #pragma unroll
        for(int m=0;m<M;m++){ c[m]+= (f0>=t[m]); c[m]+= (f1>=t[m]); }
      }
    }
  }
  for(; i+VECW<=N; i+=step){
    int4 v0 = *reinterpret_cast<const int4*>(row+i);
    int4 v1 = *reinterpret_cast<const int4*>(row+i+8);
    const T* h0 = reinterpret_cast<const T*>(&v0);
    const T* h1 = reinterpret_cast<const T*>(&v1);
    #pragma unroll
    for(int j=0;j<8;j++){
      float f0 = to_f(h0[j]), f1 = to_f(h1[j]);
      #pragma unroll
      for(int m=0;m<M;m++){ c[m]+= (f0>=t[m]); c[m]+= (f1>=t[m]); }
    }
  }
  for(int it=(N/VECW)*VECW+tid; it<N; it+=TT){
    float v = to_f(row[it]);
    #pragma unroll
    for(int m=0;m<M;m++) c[m]+= (v>=t[m]);
  }
  // block reduce
  #pragma unroll
  for(int m=0;m<M;m++)
    #pragma unroll
    for(int o=16;o>0;o>>=1) c[m]+=__shfl_down_sync(~0u,c[m],o);
  __shared__ int s[32*M];
  int warp=tid>>5, lane=tid&31;
  if(lane==0){ for(int m=0;m<M;m++) s[warp*M+m]=c[m]; }
  __syncthreads();
  if(warp==0){
    #pragma unroll
    for(int m=0;m<M;m++){
      int v=(lane<TT/32)? s[lane*M+m]:0;
      #pragma unroll
      for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(~0u,v,o);
      if(lane==0) out[blockIdx.x*M+m]=v;
    }
  }
}

// ---- variant p1: set.ge mask + packed u32 accumulate ----------------------
template<typename T>
__global__ void k_pk_mask(const T* __restrict__ row, int N,
                          const float* __restrict__ thr, int* __restrict__ out){
  using T2 = typename Pair<T>::t2;
  int tid = threadIdx.x;
  T2 t2[M]; unsigned c[M];
  #pragma unroll
  for(int m=0;m<M;m++){
    T tq = (T)thr[m];
    t2[m].x = tq; t2[m].y = tq; c[m]=0u;
  }
  int step = TT*VECW, i = tid*VECW;
  for(; i+3*step+VECW<=N; i+=4*step){
    #pragma unroll
    for(int u=0;u<4;u++){
      int4 v0 = *reinterpret_cast<const int4*>(row+i+u*step);
      int4 v1 = *reinterpret_cast<const int4*>(row+i+u*step+8);
      const T2* p0 = reinterpret_cast<const T2*>(&v0);
      const T2* p1 = reinterpret_cast<const T2*>(&v1);
      #pragma unroll
      for(int j=0;j<4;j++){
        #pragma unroll
        for(int m=0;m<M;m++){
          c[m] += set_ge_mask(p0[j], t2[m]) & 0x00010001u;
          c[m] += set_ge_mask(p1[j], t2[m]) & 0x00010001u;
        }
      }
    }
  }
  for(; i+VECW<=N; i+=step){
    int4 v0 = *reinterpret_cast<const int4*>(row+i);
    int4 v1 = *reinterpret_cast<const int4*>(row+i+8);
    const T2* p0 = reinterpret_cast<const T2*>(&v0);
    const T2* p1 = reinterpret_cast<const T2*>(&v1);
    #pragma unroll
    for(int j=0;j<4;j++){
      #pragma unroll
      for(int m=0;m<M;m++){
        c[m] += set_ge_mask(p0[j], t2[m]) & 0x00010001u;
        c[m] += set_ge_mask(p1[j], t2[m]) & 0x00010001u;
      }
    }
  }
  int cc[M];
  #pragma unroll
  for(int m=0;m<M;m++) cc[m] = (int)(c[m]&0xFFFFu) + (int)(c[m]>>16);
  for(int it=(N/VECW)*VECW+tid; it<N; it+=TT){
    float v = to_f(row[it]);
    #pragma unroll
    for(int m=0;m<M;m++) cc[m]+= (v>=thr[m]);
  }
  #pragma unroll
  for(int m=0;m<M;m++)
    #pragma unroll
    for(int o=16;o>0;o>>=1) cc[m]+=__shfl_down_sync(~0u,cc[m],o);
  __shared__ int s[32*M];
  int warp=tid>>5, lane=tid&31;
  if(lane==0){ for(int m=0;m<M;m++) s[warp*M+m]=cc[m]; }
  __syncthreads();
  if(warp==0){
    #pragma unroll
    for(int m=0;m<M;m++){
      int v=(lane<TT/32)? s[lane*M+m]:0;
      #pragma unroll
      for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(~0u,v,o);
      if(lane==0) out[blockIdx.x*M+m]=v;
    }
  }
}

// ---- variant p2: __hge2 (1.0/0.0) + 16-bit __hadd2 accumulate -------------
// Exact while per-thread-per-half count <= 256 (bf16 integer grid); our
// slices give N/TT/2 <= 128 per half.
template<typename T>
__global__ void k_pk_hadd(const T* __restrict__ row, int N,
                          const float* __restrict__ thr, int* __restrict__ out){
  using T2 = typename Pair<T>::t2;
  int tid = threadIdx.x;
  T2 t2[M], acc[M];
  #pragma unroll
  for(int m=0;m<M;m++){
    T tq = (T)thr[m];
    t2[m].x = tq; t2[m].y = tq;
    acc[m].x = (T)0.f; acc[m].y = (T)0.f;
  }
  int step = TT*VECW, i = tid*VECW;
  for(; i+3*step+VECW<=N; i+=4*step){
    #pragma unroll
    for(int u=0;u<4;u++){
      int4 v0 = *reinterpret_cast<const int4*>(row+i+u*step);
      int4 v1 = *reinterpret_cast<const int4*>(row+i+u*step+8);
      const T2* p0 = reinterpret_cast<const T2*>(&v0);
      const T2* p1 = reinterpret_cast<const T2*>(&v1);
      #pragma unroll
      for(int j=0;j<4;j++){
        #pragma unroll
        for(int m=0;m<M;m++){
          acc[m] = __hadd2(acc[m], __hge2(p0[j], t2[m]));
          acc[m] = __hadd2(acc[m], __hge2(p1[j], t2[m]));
        }
      }
    }
  }
  for(; i+VECW<=N; i+=step){
    int4 v0 = *reinterpret_cast<const int4*>(row+i);
    int4 v1 = *reinterpret_cast<const int4*>(row+i+8);
    const T2* p0 = reinterpret_cast<const T2*>(&v0);
    const T2* p1 = reinterpret_cast<const T2*>(&v1);
    #pragma unroll
    for(int j=0;j<4;j++){
      #pragma unroll
      for(int m=0;m<M;m++){
        acc[m] = __hadd2(acc[m], __hge2(p0[j], t2[m]));
        acc[m] = __hadd2(acc[m], __hge2(p1[j], t2[m]));
      }
    }
  }
  int cc[M];
  #pragma unroll
  for(int m=0;m<M;m++){
    float2 f = pair_to_f2(acc[m]);
    cc[m] = (int)f.x + (int)f.y;
  }
  for(int it=(N/VECW)*VECW+tid; it<N; it+=TT){
    float v = to_f(row[it]);
    #pragma unroll
    for(int m=0;m<M;m++) cc[m]+= (v>=thr[m]);
  }
  #pragma unroll
  for(int m=0;m<M;m++)
    #pragma unroll
    for(int o=16;o>0;o>>=1) cc[m]+=__shfl_down_sync(~0u,cc[m],o);
  __shared__ int s[32*M];
  int warp=tid>>5, lane=tid&31;
  if(lane==0){ for(int m=0;m<M;m++) s[warp*M+m]=cc[m]; }
  __syncthreads();
  if(warp==0){
    #pragma unroll
    for(int m=0;m<M;m++){
      int v=(lane<TT/32)? s[lane*M+m]:0;
      #pragma unroll
      for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(~0u,v,o);
      if(lane==0) out[blockIdx.x*M+m]=v;
    }
  }
}

template<typename T>
void run_all(int N, int BS, const std::string& var){
  // host data: uniform [-1,1), thresholds at quantiles, dtype-quantized
  std::vector<float> h(N);
  srand(1234);
  for(int i=0;i<N;i++) h[i] = 2.f*rand()/RAND_MAX - 1.f;
  std::vector<T> hd(N);
  for(int i=0;i<N;i++) hd[i] = (T)h[i];
  std::vector<float> sorted;
  for(int i=0;i<N;i++) sorted.push_back((float)(float)hd[i]);
  std::sort(sorted.begin(), sorted.end(), std::greater<float>());
  float thr[M];
  int K = std::min(1024, N/4);
  float fr[M] = {0.25f, 0.5f, 0.75f, 1.0f};
  for(int m=0;m<M;m++){
    float t = sorted[std::max(0,(int)(fr[m]*K)-1)];
    thr[m] = (float)(T)t;   // dtype-grid quantization
  }
  T* d_row; CK(cudaMalloc(&d_row,(size_t)N*sizeof(T)));
  CK(cudaMemcpy(d_row,hd.data(),(size_t)N*sizeof(T),cudaMemcpyHostToDevice));
  float* d_thr; CK(cudaMalloc(&d_thr,M*4));
  CK(cudaMemcpy(d_thr,thr,M*4,cudaMemcpyHostToDevice));
  int* d_out; CK(cudaMalloc(&d_out,(size_t)BS*M*4));
  size_t EB=512ull*1024*1024; char* d_eb; CK(cudaMalloc(&d_eb,EB));

  auto launch1=[&](){
    if(var=="a")       k_cvt<T><<<BS,TT>>>(d_row,N,d_thr,d_out);
    else if(var=="p1") k_pk_mask<T><<<BS,TT>>>(d_row,N,d_thr,d_out);
    else               k_pk_hadd<T><<<BS,TT>>>(d_row,N,d_thr,d_out);
  };
  // count validation vs variant a
  std::vector<int> ca(M), cv(M);
  k_cvt<T><<<BS,TT>>>(d_row,N,d_thr,d_out); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(ca.data(),d_out,M*4,cudaMemcpyDeviceToHost));
  launch1(); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(cv.data(),d_out,M*4,cudaMemcpyDeviceToHost));
  bool okc = true;
  for(int m=0;m<M;m++) okc = okc && (ca[m]==cv[m]);

  const int WARMUP=10, REPS=40;
  for(int i=0;i<WARMUP;i++){ CK(cudaMemset(d_eb,0,EB)); launch1(); }
  CK(cudaDeviceSynchronize());
  std::vector<float> ts;
  cudaEvent_t e0,e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  for(int i=0;i<REPS;i++){
    CK(cudaMemset(d_eb,0,EB)); CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0)); launch1(); CK(cudaEventRecord(e1));
    CK(cudaDeviceSynchronize());
    float ms; CK(cudaEventElapsedTime(&ms,e0,e1)); ts.push_back(ms*1000.f);
  }
  std::sort(ts.begin(), ts.end());
  printf("%s N=%d BS=%d med_us=%.2f counts_match=%s counts=", var.c_str(), N, BS,
         ts[REPS/2], okc?"YES":"**NO**");
  for(int m=0;m<M;m++) printf("%d ", cv[m]);
  printf("\n");
}

int main(int argc,char**argv){
  if(argc<5){printf("usage: %s <fp16|bf16> N BS <a|p1|p2>\n",argv[0]);return 1;}
  std::string dt=argv[1]; int N=atoi(argv[2]), BS=atoi(argv[3]); std::string var=argv[4];
  if(dt=="fp16") run_all<__half>(N,BS,var);
  else           run_all<__nv_bfloat16>(N,BS,var);
  return 0;
}
