// Micro-bench for a multi-threshold block_count_ge (GVR Phase-2/3 primitive).
// Faithful to tensorrt_llm/.../gvr_topk_decode.py:block_count_ge:
//   - one CTA per row, blockDim threads
//   - vectorized 128/256-bit loads, 4-way unrolled (LSU ILP)
//   - M static per-thread GE counters (register), M unrolled predicated adds
//   - M-wide warp-shuffle + block reduce
// Times the cold-L2 kernel under a cudaProfilerApi range (nsys captures it).
//
// argv: <dtype fp32|fp16> <N> <M> <BS>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_profiler_api.h>

#define CK(x) do{ cudaError_t e=(x); if(e){printf("CUDA err %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

template<typename T,int VECW> __device__ __forceinline__ void load_vec(const T* p, float fv[VECW]);
template<> __device__ __forceinline__ void load_vec<float,4>(const float* p, float fv[4]){
  float4 v=*reinterpret_cast<const float4*>(p); fv[0]=v.x;fv[1]=v.y;fv[2]=v.z;fv[3]=v.w; }
template<> __device__ __forceinline__ void load_vec<float,8>(const float* p, float fv[8]){
  float4 a=*reinterpret_cast<const float4*>(p); float4 b=*reinterpret_cast<const float4*>(p+4);
  fv[0]=a.x;fv[1]=a.y;fv[2]=a.z;fv[3]=a.w;fv[4]=b.x;fv[5]=b.y;fv[6]=b.z;fv[7]=b.w; }
template<> __device__ __forceinline__ void load_vec<__half,8>(const __half* p, float fv[8]){
  int4 v=*reinterpret_cast<const int4*>(p); const __half* h=reinterpret_cast<const __half*>(&v);
  #pragma unroll
  for(int j=0;j<8;j++) fv[j]=__half2float(h[j]); }
template<> __device__ __forceinline__ void load_vec<__half,16>(const __half* p, float fv[16]){
  int4 v0=*reinterpret_cast<const int4*>(p); int4 v1=*reinterpret_cast<const int4*>(p+8);
  const __half* h0=reinterpret_cast<const __half*>(&v0); const __half* h1=reinterpret_cast<const __half*>(&v1);
  #pragma unroll
  for(int j=0;j<8;j++){ fv[j]=__half2float(h0[j]); fv[j+8]=__half2float(h1[j]); } }

template<typename T,int M,int VECW>
__global__ void count_ge_multi(const T* __restrict__ row, int N,
                               const float* __restrict__ thr, int* __restrict__ out){
  int tid=threadIdx.x, nthreads=blockDim.x;
  const T* r=row;  // BS: all blocks read the same row (report "copy same data across BS")
  float t[M];
  #pragma unroll
  for(int m=0;m<M;m++) t[m]=thr[m];
  int c[M];
  #pragma unroll
  for(int m=0;m<M;m++) c[m]=0;

  int step=nthreads*VECW;
  int i=tid*VECW;
  // 4-way unrolled main vec loop
  for(; i+3*step+VECW<=N; i+=4*step){
    #pragma unroll
    for(int u=0;u<4;u++){
      float fv[VECW]; load_vec<T,VECW>(r+i+u*step, fv);
      #pragma unroll
      for(int j=0;j<VECW;j++){
        #pragma unroll
        for(int m=0;m<M;m++) c[m]+=(fv[j]>=t[m]);
      }
    }
  }
  // 1-way tail vec loop
  for(; i+VECW<=N; i+=step){
    float fv[VECW]; load_vec<T,VECW>(r+i, fv);
    #pragma unroll
    for(int j=0;j<VECW;j++){
      #pragma unroll
      for(int m=0;m<M;m++) c[m]+=(fv[j]>=t[m]);
    }
  }
  // scalar tail
  int nal=(N/VECW)*VECW;
  for(int it=nal+tid; it<N; it+=nthreads){
    float v=(float)r[it];
    #pragma unroll
    for(int m=0;m<M;m++) c[m]+=(v>=t[m]);
  }
  // warp reduce M counters
  #pragma unroll
  for(int m=0;m<M;m++){
    #pragma unroll
    for(int o=16;o>0;o>>=1) c[m]+=__shfl_down_sync(0xffffffffu,c[m],o);
  }
  __shared__ int s[32*M];
  int warp=tid>>5, lane=tid&31, nwarps=(nthreads+31)>>5;
  if(lane==0){
    #pragma unroll
    for(int m=0;m<M;m++) s[warp*M+m]=c[m];
  }
  __syncthreads();
  if(warp==0){
    #pragma unroll
    for(int m=0;m<M;m++){
      int v=(lane<nwarps)? s[lane*M+m]:0;
      #pragma unroll
      for(int o=16;o>0;o>>=1) v+=__shfl_down_sync(0xffffffffu,v,o);
      if(lane==0) out[blockIdx.x*M+m]=v;
    }
  }
}

template<typename T,int M,int VECW>
void launch(const T* d_row,int N,const float* d_thr,int* d_out,int BS,int T_THREADS){
  count_ge_multi<T,M,VECW><<<BS,T_THREADS>>>(d_row,N,d_thr,d_out);
}

template<typename T,int VECW>
void dispatchM(int M,const T* d_row,int N,const float* d_thr,int* d_out,int BS,int TT){
  switch(M){
    case 1: launch<T,1,VECW>(d_row,N,d_thr,d_out,BS,TT); break;
    case 2: launch<T,2,VECW>(d_row,N,d_thr,d_out,BS,TT); break;
    case 4: launch<T,4,VECW>(d_row,N,d_thr,d_out,BS,TT); break;
    case 6: launch<T,6,VECW>(d_row,N,d_thr,d_out,BS,TT); break;
    case 8: launch<T,8,VECW>(d_row,N,d_thr,d_out,BS,TT); break;
    default: printf("bad M %d\n",M); exit(1);
  }
}

int main(int argc,char**argv){
  if(argc<5){printf("usage: %s <fp32|fp16> N M BS\n",argv[0]);return 1;}
  std::string dt=argv[1]; int N=atoi(argv[2]), M=atoi(argv[3]), BS=atoi(argv[4]);
  bool fp16 = (dt=="fp16");
  // report tuning: T=1024 if (BS<=SMs && N>=65536) else 512; fp32 256-bit if N>=16384.
  int SMs; CK(cudaDeviceGetAttribute(&SMs,cudaDevAttrMultiProcessorCount,0));
  int TT = (BS<=SMs && N>=65536)?1024:512;
  int VECW = fp16 ? 8 : (N>=16384?8:4);

  char path[512]; snprintf(path,sizeof(path),"data/logits_%s_N%d.bin",dt.c_str(),N);
  FILE*f=fopen(path,"rb"); if(!f){printf("no %s\n",path);return 1;}
  size_t esz = fp16?2:4;
  std::vector<char> host(N*esz); fread(host.data(),esz,N,f); fclose(f);
  // thresholds
  char tp[512]; snprintf(tp,sizeof(tp),"data/thr_N%d.txt",N); FILE*tf=fopen(tp,"r");
  float thr[8]; for(int i=0;i<8;i++){ if(fscanf(tf,"%f",&thr[i])!=1) thr[i]=0.f; } fclose(tf);

  void* d_row; CK(cudaMalloc(&d_row,N*esz)); CK(cudaMemcpy(d_row,host.data(),N*esz,cudaMemcpyHostToDevice));
  float* d_thr; CK(cudaMalloc(&d_thr,8*4)); CK(cudaMemcpy(d_thr,thr,8*4,cudaMemcpyHostToDevice));
  int* d_out; CK(cudaMalloc(&d_out,(size_t)BS*8*4));
  // 512MB L2-evict buffer (matches report _EVICT)
  size_t EB=512ull*1024*1024; char* d_eb; CK(cudaMalloc(&d_eb,EB));

  auto run=[&](){
    if(fp16){ if(VECW==8) dispatchM<__half,8>(M,(const __half*)d_row,N,d_thr,d_out,BS,TT);
              else        dispatchM<__half,16>(M,(const __half*)d_row,N,d_thr,d_out,BS,TT); }
    else    { if(VECW==8) dispatchM<float,8>(M,(const float*)d_row,N,d_thr,d_out,BS,TT);
              else        dispatchM<float,4>(M,(const float*)d_row,N,d_thr,d_out,BS,TT); }
  };
  const int WARMUP=20, REPS=60;
  for(int i=0;i<WARMUP;i++){ CK(cudaMemset(d_eb,0,EB)); run(); }
  CK(cudaDeviceSynchronize());
  cudaProfilerStart();
  for(int i=0;i<REPS;i++){ CK(cudaMemset(d_eb,0,EB)); run(); }
  CK(cudaDeviceSynchronize());
  cudaProfilerStop();
  // sanity: print out[0..M-1] (counts must be monotone non-increasing in threshold)
  std::vector<int> ho(M); CK(cudaMemcpy(ho.data(),d_out,M*4,cudaMemcpyDeviceToHost));
  printf("done %s N=%d M=%d BS=%d TT=%d VECW=%d counts=",dt.c_str(),N,M,BS,TT,VECW);
  for(int m=0;m<M;m++) printf("%d ",ho[m]); printf("\n");
  return 0;
}
