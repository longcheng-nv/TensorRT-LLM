#include "kernel.h"
#include <cstdint>

__device__ __forceinline__ uint32_t f2key(float f){
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Parallel suffix-sum bin-select over 256 bins in smem `suf`.
// After call, for each tid<256: suf[tid] = sum_{e>=tid} hist[e]. Returns nothing;
// caller decides chosen digit = smallest d with suf[d]>=krem.
template<int BLK>
__device__ __forceinline__ void suffix256(int* suf){
    #pragma unroll
    for(int off=1; off<256; off<<=1){
        int v = (threadIdx.x<256 && threadIdx.x+off<256) ? suf[threadIdx.x+off] : 0;
        __syncthreads();
        if(threadIdx.x<256) suf[threadIdx.x]+=v;
        __syncthreads();
    }
}

// ============ FUSED: one CTA per row ============
template<int BLK>
__global__ void __launch_bounds__(BLK) topk_fused(
        const float* __restrict__ logits, int* __restrict__ indices,
        int n_valid, int npad, int k){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp = tid>>5, lane = tid&31;
    constexpr int NW = BLK/32;
    const float* __restrict__ rowp = logits + (size_t)row * npad;
    int* __restrict__ outp = indices + (size_t)row * k;
    int nvec = n_valid >> 2;
    int base = nvec << 2;
    const uint4* __restrict__ v4 = reinterpret_cast<const uint4*>(rowp);

    __shared__ int wh[NW][256];
    __shared__ int suf[256];
    __shared__ unsigned int prefix_s;
    __shared__ int krem_s;
    __shared__ int chosen_sh;
    if(tid==0){ prefix_s=0u; krem_s=k; }
    __syncthreads();

    #pragma unroll
    for(int pass=0; pass<4; ++pass){
        int shift = 24 - pass*8;
        unsigned int mask = (shift+8>=32)?0u:(0xFFFFFFFFu<<(shift+8));
        for(int i=tid;i<NW*256;i+=BLK) ((int*)wh)[i]=0;
        __syncthreads();
        unsigned int prefix = prefix_s;
        if(tid==0) chosen_sh=0;
        int* myh = wh[warp];
        for(int i=tid;i<nvec;i+=BLK){
            uint4 raw=v4[i];
            uint32_t ks[4]={f2key(__uint_as_float(raw.x)),f2key(__uint_as_float(raw.y)),
                            f2key(__uint_as_float(raw.z)),f2key(__uint_as_float(raw.w))};
            #pragma unroll
            for(int j=0;j<4;++j) if((ks[j]&mask)==(prefix&mask)) atomicAdd(&myh[(ks[j]>>shift)&0xff],1);
        }
        for(int i=base+tid;i<n_valid;i+=BLK){
            uint32_t key=f2key(rowp[i]);
            if((key&mask)==(prefix&mask)) atomicAdd(&myh[(key>>shift)&0xff],1);
        }
        __syncthreads();
        for(int d=tid; d<256; d+=BLK){
            int s=0;
            #pragma unroll
            for(int w=0; w<NW; ++w) s += wh[w][d];
            suf[d]=s;
        }
        __syncthreads();
        // parallel suffix-sum: suf[d] = sum_{e>=d} hist[e]
        suffix256<BLK>(suf);
        int krem=krem_s;
        if(tid<256){
            int sd=suf[tid], sd1=(tid<255)?suf[tid+1]:0;
            if(sd>=krem && sd1<krem) chosen_sh=tid;   // unique bin
        }
        __syncthreads();
        if(tid==0){
            int ch=chosen_sh;
            int sd1=(ch<255)?suf[ch+1]:0;
            prefix_s = prefix | ((unsigned int)ch<<shift);
            krem_s = krem - sd1;
        }
        __syncthreads();
    }

    __shared__ int cgt, ceq;
    if(tid==0){ cgt=0; ceq=0; }
    __syncthreads();
    unsigned int T = prefix_s;
    int need_eq = krem_s;
    int gt = k - need_eq;
    for(int i=tid;i<nvec;i+=BLK){
        uint4 raw=v4[i];
        uint32_t ks[4]={f2key(__uint_as_float(raw.x)),f2key(__uint_as_float(raw.y)),
                        f2key(__uint_as_float(raw.z)),f2key(__uint_as_float(raw.w))};
        #pragma unroll
        for(int j=0;j<4;++j){
            uint32_t key=ks[j]; int idx=(i<<2)+j;
            bool isgt=key>T;
            unsigned int gm=__ballot_sync(0xffffffffu,isgt);
            int rank=__popc(gm&((1u<<lane)-1)); int gc=__popc(gm); int gb=0;
            if(gc){ int ld=__ffs(gm)-1; if(lane==ld) gb=atomicAdd(&cgt,gc); gb=__shfl_sync(0xffffffffu,gb,ld); }
            if(isgt) outp[gb+rank]=idx;
            bool iseq=key==T;
            unsigned int em=__ballot_sync(0xffffffffu,iseq);
            int er=__popc(em&((1u<<lane)-1)); int ec=__popc(em); int eb=0;
            if(ec){ int ld=__ffs(em)-1; if(lane==ld) eb=atomicAdd(&ceq,ec); eb=__shfl_sync(0xffffffffu,eb,ld); }
            if(iseq){ int s=eb+er; if(s<need_eq) outp[gt+s]=idx; }
        }
    }
    for(int i=base+tid;i<n_valid;i+=BLK){
        uint32_t key=f2key(rowp[i]); int idx=i;
        if(key>T){ int s=atomicAdd(&cgt,1); outp[s]=idx; }
        else if(key==T){ int s=atomicAdd(&ceq,1); if(s<need_eq) outp[gt+s]=idx; }
    }
}

// ============ 2-PHASE: multi-launch parallel threshold + multi-CTA/row collect ====
__global__ void init_state(int* __restrict__ state, int* __restrict__ hist_g, int k){
    int t=threadIdx.x;
    if(t==0){ state[0]=0; state[1]=k; }
    for(int d=t; d<256; d+=blockDim.x) hist_g[d]=0;
}

template<int BLK>
__global__ void __launch_bounds__(BLK) hist_pass(
        const float* __restrict__ logits, int n_valid, int pass,
        int* __restrict__ hist_g, const int* __restrict__ state){
    int tid = threadIdx.x;
    int gtid = blockIdx.x*BLK + tid;
    int gstride = gridDim.x*BLK;
    int shift = 24 - pass*8;
    unsigned int mask = (shift+8>=32)?0u:(0xFFFFFFFFu<<(shift+8));
    unsigned int prefix = (unsigned int)state[0];
    const float* __restrict__ rowp = logits;
    int nvec = n_valid >> 2;
    int base = nvec << 2;
    const uint4* __restrict__ v4 = reinterpret_cast<const uint4*>(rowp);
    __shared__ int sh[256];
    for(int i=tid;i<256;i+=BLK) sh[i]=0;
    __syncthreads();
    for(int i=gtid;i<nvec;i+=gstride){
        uint4 raw=v4[i];
        uint32_t ks[4]={f2key(__uint_as_float(raw.x)),f2key(__uint_as_float(raw.y)),
                        f2key(__uint_as_float(raw.z)),f2key(__uint_as_float(raw.w))};
        #pragma unroll
        for(int j=0;j<4;++j) if((ks[j]&mask)==(prefix&mask)) atomicAdd(&sh[(ks[j]>>shift)&0xff],1);
    }
    for(int i=base+gtid;i<n_valid;i+=gstride){
        uint32_t key=f2key(rowp[i]);
        if((key&mask)==(prefix&mask)) atomicAdd(&sh[(key>>shift)&0xff],1);
    }
    __syncthreads();
    for(int i=tid;i<256;i+=BLK) if(sh[i]) atomicAdd(&hist_g[i], sh[i]);
}

__global__ void pick_pass(int pass, int* __restrict__ hist_g, int* __restrict__ state,
                          int* __restrict__ meta, int k){
    int tid = threadIdx.x;
    __shared__ int suf[256];
    suf[tid]=hist_g[tid];
    __syncthreads();
    #pragma unroll
    for(int off=1; off<256; off<<=1){
        int v=(tid+off<256)?suf[tid+off]:0;
        __syncthreads();
        suf[tid]+=v;
        __syncthreads();
    }
    int krem=state[1];
    int sd=suf[tid], sd1=(tid<255)?suf[tid+1]:0;
    __shared__ int chosen_s, krnew_s;
    if(sd>=krem && sd1<krem){ chosen_s=tid; krnew_s=krem-sd1; }
    __syncthreads();
    if(tid==0){
        int shift=24-pass*8;
        unsigned int prefix=(unsigned int)state[0];
        state[0]=(int)(prefix | ((unsigned int)chosen_s<<shift));
        state[1]=krnew_s;
        if(pass==3){ meta[0]=state[0]; meta[1]=k-krnew_s; meta[2]=krnew_s; }
    }
    __syncthreads();
    hist_g[tid]=0;
}

template<int BLK>
__global__ void __launch_bounds__(BLK) collect_kernel(
        const float* __restrict__ logits, int* __restrict__ indices,
        int n_valid, int npad, int k, const int* __restrict__ meta,
        int* __restrict__ cnt, int nCTA){
    int row = blockIdx.y;
    int c = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    unsigned int T = (unsigned int)meta[0];
    int gt = meta[1];
    int need_eq = meta[2];
    const float* __restrict__ rowp = logits + (size_t)row * npad;
    int* __restrict__ outp = indices + (size_t)row * k;
    int* __restrict__ cg = &cnt[2*row];
    int* __restrict__ ce = &cnt[2*row+1];
    long chunk = ((long)n_valid + nCTA - 1) / nCTA;
    long start = (long)c * chunk;
    long end = start + chunk; if(end > n_valid) end = n_valid;
    long lim = ((end - start + BLK - 1)/BLK)*BLK + start;
    for(long p = start + tid; p < lim; p += BLK){
        bool valid = p < end;
        uint32_t key = valid ? f2key(rowp[p]) : 0u;
        bool isgt = valid && (key > T);
        unsigned int gm=__ballot_sync(0xffffffffu,isgt);
        int rank=__popc(gm&((1u<<lane)-1)); int gc=__popc(gm); int gb=0;
        if(gc){ int ld=__ffs(gm)-1; if(lane==ld) gb=atomicAdd(cg,gc); gb=__shfl_sync(0xffffffffu,gb,ld); }
        if(isgt) outp[gb+rank]=(int)p;
        bool iseq = valid && (key==T);
        unsigned int em=__ballot_sync(0xffffffffu,iseq);
        int er=__popc(em&((1u<<lane)-1)); int ec=__popc(em); int eb=0;
        if(ec){ int ld=__ffs(em)-1; if(lane==ld) eb=atomicAdd(ce,ec); eb=__shfl_sync(0xffffffffu,eb,ld); }
        if(iseq){ int s=eb+er; if(s<need_eq) outp[gt+s]=(int)p; }
    }
}

void topk_launcher(const float* logits, int* indices, int b, int n_valid, int npad, int k, cudaStream_t stream){
    // Fused path: fills the machine (b>=132) OR n small enough that 5 reads are cheap.
    if(b >= 132 || npad <= 16384){
        constexpr int BLK = 256;
        topk_fused<BLK><<<b, BLK, 0, stream>>>(logits, indices, n_valid, npad, k);
        return;
    }
    // 2-phase for low-b large-n.
    constexpr int TBLK = 256;
    static int* meta=nullptr; static int* cnt=nullptr; static int* hist_g=nullptr; static int* state=nullptr;
    static int cap_b=0;
    if(meta==nullptr){ cudaMalloc(&meta,3*sizeof(int)); cudaMalloc(&hist_g,256*sizeof(int)); cudaMalloc(&state,2*sizeof(int)); }
    if(b>cap_b){ if(cnt) cudaFree(cnt); cudaMalloc(&cnt,2*(size_t)b*sizeof(int)); cap_b=b; }
    cudaMemsetAsync(cnt,0,2*(size_t)b*sizeof(int),stream);
    init_state<<<1,256,0,stream>>>(state,hist_g,k);
    int vgroups=(n_valid+TBLK-1)/TBLK; if(vgroups<1) vgroups=1;
    int G = vgroups<456?vgroups:456;
    #pragma unroll
    for(int pass=0;pass<4;++pass){
        hist_pass<TBLK><<<G,TBLK,0,stream>>>(logits,n_valid,pass,hist_g,state);
        pick_pass<<<1,256,0,stream>>>(pass,hist_g,state,meta,k);
    }
    int perCTA_min=256*4;
    int maxCTA=(n_valid+perCTA_min-1)/perCTA_min; if(maxCTA<1) maxCTA=1;
    int want=(592+b-1)/b; if(want<1) want=1;
    int nCTA=want<maxCTA?want:maxCTA;
    dim3 grid(nCTA,b);
    collect_kernel<256><<<grid,256,0,stream>>>(logits,indices,n_valid,npad,k,meta,cnt,nCTA);
}
