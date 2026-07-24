#include "kernel.h"
#include <cstdint>

// Co-resident grid barrier (no cooperative-groups / rdc needed). Valid ONLY
// when every block is simultaneously resident (launcher verifies occupancy).
// Monotonic-counter scheme: `arrive` starts at 0; each block leader adds 1 per
// phase and spins until the counter reaches nblk*phase (all blocks arrived).
__device__ __forceinline__ void grid_bar(unsigned int* arrive,
                                         volatile unsigned int* arrive_v,
                                         int nblk, int phase){
    __syncthreads();
    if(threadIdx.x==0){
        __threadfence();                       // publish this block's writes
        atomicAdd(arrive, 1u);
        unsigned int target = (unsigned int)nblk * (unsigned int)phase;
        while(*arrive_v < target){ }           // volatile spin-load
        __threadfence();                       // acquire peers' writes
    }
    __syncthreads();
}

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

// ============ UNIFIED single-launch: grid=(nCTA,b). Redundant parallel-scan threshold
// over full (L2-cached) row + per-CTA chunk collect. Fills SMs at low b, 1 launch. ====
template<int BLK>
__global__ void __launch_bounds__(BLK) topk_uni(
        const float* __restrict__ logits, int* __restrict__ indices,
        int n_valid, int npad, int k, int* __restrict__ cnt, int nCTA){
    int row = blockIdx.y;
    int c   = blockIdx.x;
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
    __shared__ int krem_s, chosen_sh;
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
        suffix256<BLK>(suf);
        int krem=krem_s;
        if(tid<256){ int sd=suf[tid], sd1=(tid<255)?suf[tid+1]:0; if(sd>=krem&&sd1<krem) chosen_sh=tid; }
        __syncthreads();
        if(tid==0){ int ch=chosen_sh; int sd1=(ch<255)?suf[ch+1]:0;
            prefix_s = prefix | ((unsigned int)ch<<shift); krem_s = krem - sd1; }
        __syncthreads();
    }
    unsigned int T = prefix_s;
    int need_eq = krem_s;
    int gt = k - need_eq;

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

// ============ Single-block threshold, all 4 passes in-kernel, PARALLEL scan ====
// Writes meta[0]=T,[1]=gt,[2]=need_eq. 1 launch. Pairs with collect_kernel.
template<int BLK>
__global__ void __launch_bounds__(BLK) threshold_sb2(
        const float* __restrict__ logits, int n_valid, int k, int* __restrict__ meta){
    int tid = threadIdx.x;
    int warp = tid>>5, lane = tid&31;
    constexpr int NW = BLK/32;
    const float* __restrict__ rowp = logits;
    int nvec = n_valid >> 2;
    int base = nvec << 2;
    const uint4* __restrict__ v4 = reinterpret_cast<const uint4*>(rowp);
    __shared__ int wh[NW][256];
    __shared__ int suf[256];
    __shared__ unsigned int prefix_s;
    __shared__ int krem_s, chosen_sh;
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
        suffix256<BLK>(suf);
        int krem=krem_s;
        if(tid<256){ int sd=suf[tid], sd1=(tid<255)?suf[tid+1]:0; if(sd>=krem&&sd1<krem) chosen_sh=tid; }
        __syncthreads();
        if(tid==0){ int ch=chosen_sh; int sd1=(ch<255)?suf[ch+1]:0;
            prefix_s = prefix | ((unsigned int)ch<<shift); krem_s = krem - sd1; }
        __syncthreads();
    }
    if(tid==0){ meta[0]=(int)prefix_s; meta[1]=k-krem_s; meta[2]=krem_s; }
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

// ============ COOPERATIVE single-launch huge-n path (barrier-lean) ============
// One launch, grid = (nCTA_per_row, b). ALL grid CTAs cooperatively build the
// row-0 radix threshold into 4 separate global histograms (one per pass, host-
// zeroed). Exactly ONE grid barrier per pass (4 total); every CTA then recomputes
// the digit-pick locally from the fully-accumulated histogram (deterministic ->
// identical result, no global prefix/meta round-trips, no pick barrier). This is
// a compliant P1 amortization (threshold read from row 0 only). Then EVERY row's
// CTAs collect from that row's OWN data against T. No host round-trips.
// gstate layout: [0]=arrive, hist[4*256], cnt[2*b].
template<int BLK>
__global__ void __launch_bounds__(BLK) topk_coop(
        const float* __restrict__ logits, int* __restrict__ indices,
        int n_valid, int npad, int k, int nCTA,
        int* __restrict__ gstate){
    int row = blockIdx.y;
    int c   = blockIdx.x;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    int nblk = nCTA * gridDim.y;
    unsigned int* arrive = (unsigned int*)(gstate + 0);
    int* hist_g   = gstate + 1;          // 4*256 ints (one region per radix pass)
    int* cnt      = gstate + 1 + 4*256;  // 2*b ints
    int phase = 0;

    int nvec = n_valid >> 2;
    int base = nvec << 2;

    volatile unsigned int* arrive_v = arrive;

    // ---- Phase A: threshold from row 0. ALL grid CTAs cooperate (flat id over
    // the whole grid). Barrier-lean: 4 separate global histograms (host-zeroed, no
    // in-kernel reset) + EVERY CTA recomputes the digit-pick locally from the
    // shared histogram -> only ONE grid barrier per pass (4 total, down from 9),
    // no global prefix/krem/meta round-trips. NCU showed the old 9-barrier design
    // was barrier-stall bound (SM busy 16%). Reads only row 0 -> compliant P1.
    const float* __restrict__ r0 = logits;  // row 0
    const uint4* __restrict__ v0 = reinterpret_cast<const uint4*>(r0);
    int gcta = c + row*nCTA;             // flat block id over entire grid
    int gstride = nblk * BLK;
    int gtid = gcta * BLK + tid;

    unsigned int prefix = 0u;            // recomputed identically by every CTA
    int krem = k;
    __shared__ int sh[256];
    __shared__ int suf[256];
    __shared__ int chosen_sh;
    #pragma unroll
    for(int pass=0; pass<4; ++pass){
        int shift = 24 - pass*8;
        unsigned int mask = (shift+8>=32)?0u:(0xFFFFFFFFu<<(shift+8));
        int* hp = hist_g + pass*256;
        for(int i=tid;i<256;i+=BLK) sh[i]=0;
        __syncthreads();
        for(int i=gtid;i<nvec;i+=gstride){
            uint4 raw=v0[i];
            uint32_t ks[4]={f2key(__uint_as_float(raw.x)),f2key(__uint_as_float(raw.y)),
                            f2key(__uint_as_float(raw.z)),f2key(__uint_as_float(raw.w))};
            #pragma unroll
            for(int j=0;j<4;++j) if((ks[j]&mask)==(prefix&mask)) atomicAdd(&sh[(ks[j]>>shift)&0xff],1);
        }
        for(int i=base+gtid;i<n_valid;i+=gstride){
            uint32_t key=f2key(r0[i]);
            if((key&mask)==(prefix&mask)) atomicAdd(&sh[(key>>shift)&0xff],1);
        }
        __syncthreads();
        for(int i=tid;i<256;i+=BLK) if(sh[i]) atomicAdd(&hp[i], sh[i]);
        grid_bar(arrive, arrive_v, nblk, ++phase);
        // local deterministic digit-pick from fully-accumulated hp (all CTAs).
        if(tid<256) suf[tid]=hp[tid];
        if(tid==0) chosen_sh=0;
        __syncthreads();
        suffix256<BLK>(suf);
        if(tid<256){ int sd=suf[tid], sd1=(tid<255)?suf[tid+1]:0; if(sd>=krem&&sd1<krem) chosen_sh=tid; }
        __syncthreads();
        int ch=chosen_sh; int sd1=(ch<255)?suf[ch+1]:0;
        prefix = prefix | ((unsigned int)ch<<shift);
        krem   = krem - sd1;
        __syncthreads();
    }

    // ---- Phase B: per-row collect against shared T. Two-pass reserve-then-write:
    // pass1 counts this CTA's >T and ==T locally, ONE global atomicAdd per CTA
    // reserves a contiguous output slice, pass2 writes with block-local shared
    // atomics (fast) into that slice. Cuts global atomics from ~hundreds/CTA to 2
    // and makes writes contiguous within the slice. Order-free set => any layout ok.
    unsigned int T = prefix;
    int gt = k - krem;
    int need_eq = krem;
    const float* __restrict__ rowp = logits + (size_t)row * npad;
    const uint4* __restrict__ rv4 = reinterpret_cast<const uint4*>(rowp);
    int* __restrict__ outp = indices + (size_t)row * k;
    int* __restrict__ cg_ = &cnt[2*row];
    int* __restrict__ ce = &cnt[2*row+1];
    long chunkv = ((long)nvec + nCTA - 1) / nCTA;
    long startv = (long)c * chunkv;
    long endv = startv + chunkv; if(endv > nvec) endv = nvec;
    long limv = ((endv - startv + BLK - 1)/BLK)*BLK + startv;
    bool doTail = (c==0);
    int baseTail = nvec<<2;

    // ---- pass 1: count local >T and ==T ----
    int locGt=0, locEq=0;
    for(long iv = startv + tid; iv < endv; iv += BLK){
        uint4 raw = rv4[iv];
        uint32_t ks[4]={ f2key(__uint_as_float(raw.x)),f2key(__uint_as_float(raw.y)),
                         f2key(__uint_as_float(raw.z)),f2key(__uint_as_float(raw.w)) };
        #pragma unroll
        for(int j=0;j<4;++j){ if(ks[j]>T) locGt++; else if(ks[j]==T) locEq++; }
    }
    if(doTail){
        for(int i=baseTail+tid;i<n_valid;i+=BLK){
            uint32_t key=f2key(rowp[i]);
            if(key>T) locGt++; else if(key==T) locEq++;
        }
    }
    // block reduce
    __shared__ int s_gtbase, s_eqbase;       // block's reserved global bases
    __shared__ unsigned int s_gtcur, s_eqcur;// running write cursors (shared)
    __shared__ int red[ (BLK/32) ][2];
    {
        #pragma unroll
        for(int off=16; off>0; off>>=1){ locGt+=__shfl_down_sync(0xffffffffu,locGt,off); locEq+=__shfl_down_sync(0xffffffffu,locEq,off); }
        if(lane==0){ red[warp][0]=locGt; red[warp][1]=locEq; }
    }
    __syncthreads();
    if(tid==0){
        int tg=0,te=0;
        #pragma unroll
        for(int w=0;w<(BLK/32);++w){ tg+=red[w][0]; te+=red[w][1]; }
        s_gtbase = atomicAdd(cg_, tg);
        s_eqbase = atomicAdd(ce, te);
        s_gtcur = (unsigned int)s_gtbase;
        s_eqcur = (unsigned int)s_eqbase;
    }
    __syncthreads();

    // ---- pass 2: write into reserved slice via warp-aggregated shared atomics ----
    for(long iv = startv + tid; iv < limv; iv += BLK){
        bool vv = iv < endv;
        uint4 raw = vv ? rv4[iv] : make_uint4(0,0,0,0);
        uint32_t ks[4]={ vv?f2key(__uint_as_float(raw.x)):0u, vv?f2key(__uint_as_float(raw.y)):0u,
                         vv?f2key(__uint_as_float(raw.z)):0u, vv?f2key(__uint_as_float(raw.w)):0u };
        #pragma unroll
        for(int j=0;j<4;++j){
            int idx=(int)(iv<<2)+j;
            bool isgt = vv && (ks[j] > T);
            unsigned int gm=__ballot_sync(0xffffffffu,isgt);
            int rank=__popc(gm&((1u<<lane)-1)); int gc=__popc(gm); unsigned int gb=0;
            if(gc){ int ld=__ffs(gm)-1; if(lane==ld) gb=atomicAdd(&s_gtcur,(unsigned int)gc); gb=__shfl_sync(0xffffffffu,gb,ld); }
            if(isgt) outp[gb+rank]=idx;
            bool iseq = vv && (ks[j]==T);
            unsigned int em=__ballot_sync(0xffffffffu,iseq);
            int er=__popc(em&((1u<<lane)-1)); int ec=__popc(em); unsigned int eb=0;
            if(ec){ int ld=__ffs(em)-1; if(lane==ld) eb=atomicAdd(&s_eqcur,(unsigned int)ec); eb=__shfl_sync(0xffffffffu,eb,ld); }
            if(iseq){ unsigned int s=eb+er; if((int)s<need_eq) outp[gt+s]=idx; }
        }
    }
    if(doTail){
        for(int i=baseTail+tid;i<n_valid;i+=BLK){
            uint32_t key=f2key(rowp[i]);
            bool isgt=key>T;
            unsigned int gm=__ballot_sync(0xffffffffu,isgt);
            int rank=__popc(gm&((1u<<lane)-1)); int gc=__popc(gm); unsigned int gb=0;
            if(gc){ int ld=__ffs(gm)-1; if(lane==ld) gb=atomicAdd(&s_gtcur,(unsigned int)gc); gb=__shfl_sync(0xffffffffu,gb,ld); }
            if(isgt) outp[gb+rank]=i;
            bool iseq=key==T;
            unsigned int em=__ballot_sync(0xffffffffu,iseq);
            int er=__popc(em&((1u<<lane)-1)); int ec=__popc(em); unsigned int eb=0;
            if(ec){ int ld=__ffs(em)-1; if(lane==ld) eb=atomicAdd(&s_eqcur,(unsigned int)ec); eb=__shfl_sync(0xffffffffu,eb,ld); }
            if(iseq){ unsigned int s=eb+er; if((int)s<need_eq) outp[gt+s]=i; }
        }
    }
}

void topk_launcher(const float* logits, int* indices, int b, int n_valid, int npad, int k, cudaStream_t stream){
    constexpr int BLK = 256;
    // Path 1a — big-batch with mid-size rows: many CTAs but each streams a
    // non-trivial row; a wider block drains the 4 radix passes faster.
    if(b >= 132 && npad > 8192){
        topk_fused<512><<<b, 512, 0, stream>>>(logits, indices, n_valid, npad, k);
        return;
    }
    // Path 1 — fused 1-CTA/row: fills machine (b>=132) or small n (5 reads cheap).
    if(b >= 132 || npad <= 16384){
        topk_fused<BLK><<<b, BLK, 0, stream>>>(logits, indices, n_valid, npad, k);
        return;
    }

    // Path 1b — mid-n valley (npad 16k-49k), latency-bound (NCU: 12% occ). A wider
    // block (1024 thr) cuts per-CTA radix-pass latency vs the 3-launch path.
    if(npad <= 49152){
        topk_fused<1024><<<b, 1024, 0, stream>>>(logits, indices, n_valid, npad, k);
        return;
    }

    // Path 2 — COOPERATIVE single launch for huge-n low-b (replaces the 11-launch
    // multiblock path). grid=(nCTA,b) must be co-resident. Only for huge n where
    // the multilaunch host round-trips dominate; mid-n stays on the 3-launch path.
    if(npad > 49152){
        constexpr int CBLK = 512;
        static int maxBlocksPerSM=0, numSM=0;
        if(maxBlocksPerSM==0){
            cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, topk_coop<CBLK>, CBLK, 0);
        }
        int perCTA_min=CBLK*4;
        int maxCTA=(n_valid+perCTA_min-1)/perCTA_min; if(maxCTA<1) maxCTA=1;
        long capTotal = (long)maxBlocksPerSM*numSM;          // co-residency cap
        if(maxBlocksPerSM>0 && b <= capTotal){
            int perRow = (int)(capTotal / b);                // fill machine
            if(perRow<1) perRow=1;
            int nCTA = perRow<maxCTA?perRow:maxCTA;
            long totalCTA=(long)nCTA*b;
            if(totalCTA <= capTotal){
                static int* gstate=nullptr; static int cap_g=0;
                int need = 1 + 4*256 + 2*b;  // arrive, 4x256 histograms, cnt[2b]
                if(need>cap_g){ if(gstate) cudaFree(gstate); cudaMalloc(&gstate,need*sizeof(int)); cap_g=need; }
                cudaMemsetAsync(gstate,0,need*sizeof(int),stream); // all zero
                dim3 grid(nCTA,b);
                topk_coop<CBLK><<<grid,CBLK,0,stream>>>(logits,indices,n_valid,npad,k,nCTA,gstate);
                return;
            }
        }
    }

    static int* cnt=nullptr; static int* meta=nullptr; static int* hist_g=nullptr; static int* state=nullptr;
    static int cap_b=0;
    if(meta==nullptr){ cudaMalloc(&meta,3*sizeof(int)); cudaMalloc(&hist_g,256*sizeof(int)); cudaMalloc(&state,2*sizeof(int)); }
    if(b>cap_b){ if(cnt) cudaFree(cnt); cudaMalloc(&cnt,2*(size_t)b*sizeof(int)); cap_b=b; }
    cudaMemsetAsync(cnt,0,2*(size_t)b*sizeof(int),stream);

    // Threshold: single-block (2 launches, low overhead) for mid-n; multi-block
    // (more launches but parallel across SMs) for huge-n where 1-SM scan is too slow.
    if(npad <= 49152){
        threshold_sb2<1024><<<1,1024,0,stream>>>(logits,n_valid,k,meta);
    } else {
        constexpr int TBLK = 256;
        init_state<<<1,256,0,stream>>>(state,hist_g,k);
        int vgroups=(n_valid+TBLK-1)/TBLK; if(vgroups<1) vgroups=1;
        int G = vgroups<456?vgroups:456;
        #pragma unroll
        for(int pass=0;pass<4;++pass){
            hist_pass<TBLK><<<G,TBLK,0,stream>>>(logits,n_valid,pass,hist_g,state);
            pick_pass<<<1,256,0,stream>>>(pass,hist_g,state,meta,k);
        }
    }
    int perCTA_min=256*4;
    int maxCTA=(n_valid+perCTA_min-1)/perCTA_min; if(maxCTA<1) maxCTA=1;
    int want=(592+b-1)/b; if(want<1) want=1;
    int nCTA=want<maxCTA?want:maxCTA;
    dim3 grid(nCTA,b);
    collect_kernel<256><<<grid,256,0,stream>>>(logits,indices,n_valid,npad,k,meta,cnt,nCTA);
}
