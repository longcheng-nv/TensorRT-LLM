#include "kernel.h"
#include "gvr_kernel.h"
#include <cstdint>
#include <math.h>

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

// GVR byte refinement.  Warp 0 first locates the winning 8-bin chunk, then
// uses a log-CCDF secant pivot to decide which half to search first.  The
// histogram counts verify the digit exactly; the pivot only shortens the
// serial walk and can never change the answer on plateaus or ties.
template<int BLK, bool CLEAR_HIST>
__device__ __forceinline__ void gvr_select256(
        int* hist, int krem, unsigned int prefix, int shift,
        unsigned int* prefix_s, int* krem_s, int* pivot_s){
    int tid=threadIdx.x, lane=tid&31;
    if(tid<32){
        int hi=255-lane*8, lo=hi-7;
        int sum=0;
        #pragma unroll
        for(int d=lo;d<=hi;++d) sum += hist[d];
        int inclusive=sum;
        #pragma unroll
        for(int off=1;off<32;off<<=1){
            int x=__shfl_up_sync(0xffffffffu,inclusive,off);
            if(lane>=off) inclusive+=x;
        }
        unsigned int owners=__ballot_sync(0xffffffffu,inclusive>=krem);
        int owner=__ffs(owners)-1;
        int higher=owner ? __shfl_sync(0xffffffffu,inclusive,owner-1) : 0;
        if(lane==owner){
            int p=*pivot_s;
            p=p<lo?lo:(p>hi?hi:p);
            int above_p=0;
            #pragma unroll
            for(int d=lo;d<=hi;++d) if(d>p) above_p+=hist[d];
            int chosen=lo;
            if(higher+above_p>=krem){
                for(int d=hi;d>p;--d){
                    int c=hist[d];
                    if(higher+c>=krem){ chosen=d; break; }
                    higher+=c;
                }
            }else{
                higher+=above_p;
                for(int d=p;d>=lo;--d){
                    int c=hist[d];
                    if(higher+c>=krem){ chosen=d; break; }
                    higher+=c;
                }
            }
            int rank=krem-higher;
            int pop=hist[chosen];
            float y=(pop<=1)?0.0f:
                (__log2f((float)rank+1.0f)/__log2f((float)pop+1.0f));
            int next=255-__float2int_rn(y*255.0f);
            *prefix_s=prefix|((unsigned int)chosen<<shift);
            *krem_s=rank;
            *pivot_s=next<0?0:(next>255?255:next);
        }
    }
    __syncthreads();
    if constexpr(CLEAR_HIST){
        if(tid<256) hist[tid]=0;
        __syncthreads();
    }
}

// ============ FUSED: one CTA per row ============
template<int BLK>
__global__ void __launch_bounds__(BLK) gvr_fused(
        const float* __restrict__ logits, const int* __restrict__ pre_idx,
        int* __restrict__ indices,
        int n_valid, int npad, int k){
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp = tid>>5, lane = tid&31;
    constexpr int NW = BLK/32;
    const float* __restrict__ rowp = logits + (size_t)row * npad;
    const int* __restrict__ hints = pre_idx + (size_t)row * k;
    int* __restrict__ outp = indices + (size_t)row * k;
    int nvec = n_valid >> 2;
    int base = nvec << 2;
    const uint4* __restrict__ v4 = reinterpret_cast<const uint4*>(rowp);

    // Single shared 256-bin histogram: on SM100 same-address shared atomics are
    // fast (insight: per-warp private histograms are a dead end here). Avoids the
    // NW*256 clear and the NW-way per-bin reduction the per-warp scheme needs.
    __shared__ int sh[256];
    __shared__ unsigned int prior_s;
    __shared__ unsigned int prefix_s;
    __shared__ int krem_s;
    __shared__ int pivot_s;
    if(tid==0) prior_s=0xffffffffu;
    __syncthreads();
    unsigned int prior=0xffffffffu;
    for(int j=tid;j<k;j+=BLK){
        int idx=hints[j];
        unsigned int key=(unsigned int)idx<(unsigned int)n_valid?f2key(rowp[idx]):0u;
        prior=key<prior?key:prior;
    }
    #pragma unroll
    for(int off=16;off;off>>=1){
        unsigned int x=__shfl_down_sync(0xffffffffu,prior,off);
        prior=x<prior?x:prior;
    }
    if(lane==0) atomicMin(&prior_s,prior);
    __syncthreads();
    if(tid==0){ prefix_s=0u; krem_s=k; pivot_s=(int)(prior_s>>24); }
    for(int i=tid;i<256;i+=BLK) sh[i]=0;   // clear once; select256_upd re-clears each pass
    __syncthreads();

    #pragma unroll
    for(int pass=0; pass<4; ++pass){
        int shift = 24 - pass*8;
        unsigned int mask = (shift+8>=32)?0u:(0xFFFFFFFFu<<(shift+8));
        unsigned int prefix = prefix_s;
        for(int i=tid;i<nvec;i+=BLK){
            uint4 raw=v4[i];
            uint32_t ks[4]={f2key(__uint_as_float(raw.x)),f2key(__uint_as_float(raw.y)),
                            f2key(__uint_as_float(raw.z)),f2key(__uint_as_float(raw.w))};
            #pragma unroll
            for(int j=0;j<4;++j){
                bool member=(pass==0)?(ks[j]>=prior_s):((ks[j]&mask)==(prefix&mask));
                if(member) atomicAdd(&sh[(ks[j]>>shift)&0xff],1);
            }
        }
        for(int i=base+tid;i<n_valid;i+=BLK){
            uint32_t key=f2key(rowp[i]);
            bool member=(pass==0)?(key>=prior_s):((key&mask)==(prefix&mask));
            if(member) atomicAdd(&sh[(key>>shift)&0xff],1);
        }
        __syncthreads();
        // suffix-sum digit-select on the single shared histogram, writing the
        // updated prefix/krem directly AND re-clearing sh for the next pass
        // (no separate clear+barrier per pass).
        int krem=krem_s;
        gvr_select256<BLK,true>(sh,krem,prefix,shift,&prefix_s,&krem_s,&pivot_s);
    }

    __shared__ int cgt, ceq;
    if(tid==0){ cgt=0; ceq=0; }
    __syncthreads();
    unsigned int T = prefix_s;
    int need_eq = krem_s;
    int gt = k - need_eq;
    int limvec=((nvec+BLK-1)/BLK)*BLK;
    for(int i=tid;i<limvec;i+=BLK){
        bool valid=i<nvec;
        uint4 raw=valid?v4[i]:make_uint4(0,0,0,0);
        uint32_t ks[4]={valid?f2key(__uint_as_float(raw.x)):0u,valid?f2key(__uint_as_float(raw.y)):0u,
                        valid?f2key(__uint_as_float(raw.z)):0u,valid?f2key(__uint_as_float(raw.w)):0u};
        #pragma unroll
        for(int j=0;j<4;++j){
            uint32_t key=ks[j]; int idx=(i<<2)+j;
            bool isgt=valid && key>T;
            unsigned int gm=__ballot_sync(0xffffffffu,isgt);
            int rank=__popc(gm&((1u<<lane)-1)); int gc=__popc(gm); int gb=0;
            if(gc){ int ld=__ffs(gm)-1; if(lane==ld) gb=atomicAdd(&cgt,gc); gb=__shfl_sync(0xffffffffu,gb,ld); }
            if(isgt) outp[gb+rank]=idx;
            bool iseq=valid && key==T;
            unsigned int em=__ballot_sync(0xffffffffu,iseq);
            int er=__popc(em&((1u<<lane)-1)); int ec=__popc(em); int eb=0;
            if(ec){ int ld=__ffs(em)-1; if(lane==ld) eb=atomicAdd(&ceq,ec); eb=__shfl_sync(0xffffffffu,eb,ld); }
            if(iseq){ int s=eb+er; if(s<need_eq) outp[gt+s]=idx; }
        }
    }
    int tail_idx=base+tid;
    bool tail_valid=tail_idx<n_valid;
    uint32_t tail_key=tail_valid?f2key(rowp[tail_idx]):0u;
    bool tail_gt=tail_valid && tail_key>T;
    unsigned int tgm=__ballot_sync(0xffffffffu,tail_gt);
    int tgr=__popc(tgm&((1u<<lane)-1)), tgc=__popc(tgm), tgb=0;
    if(tgc){ int ld=__ffs(tgm)-1; if(lane==ld) tgb=atomicAdd(&cgt,tgc); tgb=__shfl_sync(0xffffffffu,tgb,ld); }
    if(tail_gt) outp[tgb+tgr]=tail_idx;
    bool tail_eq=tail_valid && tail_key==T;
    unsigned int tem=__ballot_sync(0xffffffffu,tail_eq);
    int ter=__popc(tem&((1u<<lane)-1)), tec=__popc(tem), teb=0;
    if(tec){ int ld=__ffs(tem)-1; if(lane==ld) teb=atomicAdd(&ceq,tec); teb=__shfl_sync(0xffffffffu,teb,ld); }
    if(tail_eq){ int s=teb+ter; if(s<need_eq) outp[gt+s]=tail_idx; }
}

// ============ COOPERATIVE single-launch huge-n path (barrier-lean) ============
// Per-row P1 prior + four exact byte rungs.  All CTAs for a row contribute to
// that row's histogram; no solved threshold or output is shared across rows.
template<int BLK, bool ONE_CTA=false>
__global__ void __launch_bounds__(BLK) gvr_coop(
        const float* __restrict__ logits, const int* __restrict__ pre_idx,
        int* __restrict__ indices, int n_valid, int npad, int k, int nCTA,
        int* __restrict__ gstate){
    int row=blockIdx.y, c=blockIdx.x, tid=threadIdx.x;
    int lane=tid&31, warp=tid>>5;
    int nblk=nCTA*gridDim.y;
    unsigned int* arrive=(unsigned int*)gstate;
    volatile unsigned int* arrive_v=arrive;
    int* hist_g=gstate+1;
    int* cnt=gstate+1+gridDim.y*4*256;
    int phase=0;
    const float* __restrict__ rowp=logits+(size_t)row*npad;
    const int* __restrict__ hints=pre_idx+(size_t)row*k;
    const uint4* __restrict__ rv4=reinterpret_cast<const uint4*>(rowp);
    int nvec=n_valid>>2, base=nvec<<2;
    int gstride=nCTA*BLK, gtid=c*BLK+tid;

    __shared__ int sh[256];
    __shared__ unsigned int prior_s, prefix_s;
    __shared__ int krem_s, pivot_s;
    if(tid==0) prior_s=0xffffffffu;
    __syncthreads();
    unsigned int prior=0xffffffffu;
    for(int j=tid;j<k;j+=BLK){
        int idx=hints[j];
        unsigned int key=(unsigned int)idx<(unsigned int)n_valid?f2key(rowp[idx]):0u;
        prior=key<prior?key:prior;
    }
    #pragma unroll
    for(int off=16;off;off>>=1){
        unsigned int x=__shfl_down_sync(0xffffffffu,prior,off);
        prior=x<prior?x:prior;
    }
    if(lane==0) atomicMin(&prior_s,prior);
    __syncthreads();
    if(tid==0){ prefix_s=0u; krem_s=k; pivot_s=(int)(prior_s>>24); }
    __syncthreads();

    #pragma unroll
    for(int pass=0;pass<4;++pass){
        int shift=24-pass*8;
        unsigned int mask=(shift+8>=32)?0u:(0xffffffffu<<(shift+8));
        unsigned int prefix=prefix_s;
        int krem=krem_s;
        int* hp=hist_g+(row*4+pass)*256;
        for(int i=tid;i<256;i+=BLK) sh[i]=0;
        __syncthreads();
        for(int i=gtid;i<nvec;i+=gstride){
            uint4 raw=rv4[i];
            uint32_t ks[4]={f2key(__uint_as_float(raw.x)),f2key(__uint_as_float(raw.y)),
                            f2key(__uint_as_float(raw.z)),f2key(__uint_as_float(raw.w))};
            #pragma unroll
            for(int j=0;j<4;++j){
                bool member=(pass==0)?(ks[j]>=prior_s):((ks[j]&mask)==(prefix&mask));
                if(member) atomicAdd(&sh[(ks[j]>>shift)&255u],1);
            }
        }
        for(int i=base+gtid;i<n_valid;i+=gstride){
            uint32_t key=f2key(rowp[i]);
            bool member=(pass==0)?(key>=prior_s):((key&mask)==(prefix&mask));
            if(member) atomicAdd(&sh[(key>>shift)&255u],1);
        }
        __syncthreads();
        for(int i=tid;i<256;i+=BLK) if(sh[i]) atomicAdd(&hp[i],sh[i]);
        grid_bar(arrive,arrive_v,nblk,++phase);
        gvr_select256<BLK,false>(hp,krem,prefix,shift,&prefix_s,&krem_s,&pivot_s);
    }

    unsigned int T=prefix_s;
    int gt=k-krem_s, need_eq=krem_s;
    int* __restrict__ outp=indices+(size_t)row*k;
    int* __restrict__ cg_=&cnt[2*row];
    int* __restrict__ ce=&cnt[2*row+1];
    long chunkv = ((long)nvec + nCTA - 1) / nCTA;
    long startv = (long)c * chunkv;
    long endv = startv + chunkv; if(endv > nvec) endv = nvec;
    long limv = ((endv - startv + BLK - 1)/BLK)*BLK + startv;
    bool doTail = (c==0);
    int baseTail = nvec<<2;

    if constexpr (ONE_CTA) {
      // Wide (mid-b) path: exactly ONE CTA owns this row's entire output, so it
      // can collect in a SINGLE row read (no count pass) using fast block-local
      // SHARED-atomic cursors (no global-atomic contention, no reserve). This
      // avoids both v17's doubled global atomics and the two-pass double read.
      __shared__ unsigned int one_gt, one_eq;
      if(tid==0){ one_gt=0u; one_eq=0u; }
      __syncthreads();
      long one_lim=((long)nvec+BLK-1)/BLK*BLK;
      for(long iv = tid; iv < one_lim; iv += BLK){
        bool vv=iv<nvec;
        uint4 raw = vv?rv4[iv]:make_uint4(0,0,0,0);
        uint32_t ks[4]={ vv?f2key(__uint_as_float(raw.x)):0u,vv?f2key(__uint_as_float(raw.y)):0u,
                         vv?f2key(__uint_as_float(raw.z)):0u,vv?f2key(__uint_as_float(raw.w)):0u };
        #pragma unroll
        for(int j=0;j<4;++j){
            int idx=(int)(iv<<2)+j;
            bool isgt = vv && ks[j] > T;
            unsigned int gm=__ballot_sync(0xffffffffu,isgt);
            int rk=__popc(gm&((1u<<lane)-1)); int gc=__popc(gm); unsigned int gb=0;
            if(gc){ int ld=__ffs(gm)-1; if(lane==ld) gb=atomicAdd(&one_gt,(unsigned int)gc); gb=__shfl_sync(0xffffffffu,gb,ld); }
            if(isgt) outp[gb+rk]=idx;
            bool iseq = vv && ks[j]==T;
            unsigned int em=__ballot_sync(0xffffffffu,iseq);
            int er=__popc(em&((1u<<lane)-1)); int ec=__popc(em); unsigned int eb=0;
            if(ec){ int ld=__ffs(em)-1; if(lane==ld) eb=atomicAdd(&one_eq,(unsigned int)ec); eb=__shfl_sync(0xffffffffu,eb,ld); }
            if(iseq){ unsigned int s=eb+er; if((int)s<need_eq) outp[gt+s]=idx; }
        }
      }
      int i=baseTail+tid;
        bool valid=i<n_valid;
        uint32_t key=valid?f2key(rowp[i]):0u;
        bool isgt=valid && key>T;
        unsigned int gm=__ballot_sync(0xffffffffu,isgt);
        int rk=__popc(gm&((1u<<lane)-1)); int gc=__popc(gm); unsigned int gb=0;
        if(gc){ int ld=__ffs(gm)-1; if(lane==ld) gb=atomicAdd(&one_gt,(unsigned int)gc); gb=__shfl_sync(0xffffffffu,gb,ld); }
        if(isgt) outp[gb+rk]=i;
        bool iseq=valid && key==T;
        unsigned int em=__ballot_sync(0xffffffffu,iseq);
        int er=__popc(em&((1u<<lane)-1)); int ec=__popc(em); unsigned int eb=0;
        if(ec){ int ld=__ffs(em)-1; if(lane==ld) eb=atomicAdd(&one_eq,(unsigned int)ec); eb=__shfl_sync(0xffffffffu,eb,ld); }
        if(iseq){ unsigned int s=eb+er; if((int)s<need_eq) outp[gt+s]=i; }
      return;
    }

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
            int i=baseTail+tid;
            bool valid=i<n_valid;
            uint32_t key=valid?f2key(rowp[i]):0u;
            bool isgt=valid && key>T;
            unsigned int gm=__ballot_sync(0xffffffffu,isgt);
            int rank=__popc(gm&((1u<<lane)-1)); int gc=__popc(gm); unsigned int gb=0;
            if(gc){ int ld=__ffs(gm)-1; if(lane==ld) gb=atomicAdd(&s_gtcur,(unsigned int)gc); gb=__shfl_sync(0xffffffffu,gb,ld); }
            if(isgt) outp[gb+rank]=i;
            bool iseq=valid && key==T;
            unsigned int em=__ballot_sync(0xffffffffu,iseq);
            int er=__popc(em&((1u<<lane)-1)); int ec=__popc(em); unsigned int eb=0;
            if(ec){ int ld=__ffs(em)-1; if(lane==ld) eb=atomicAdd(&s_eqcur,(unsigned int)ec); eb=__shfl_sync(0xffffffffu,eb,ld); }
            if(iseq){ unsigned int s=eb+er; if((int)s<need_eq) outp[gt+s]=i; }
    }
}

void topk_launcher(const float* logits, const int* pre_idx, int* indices, int b, int n_valid, int npad, int k, cudaStream_t stream){
    constexpr int BLK = 256;
    // Path 1a — big-batch with mid-size rows: many CTAs but each streams a
    // non-trivial row; a wider block drains the 4 radix passes faster.
    if(b >= 132 && npad > 8192){
        gvr_fused<512><<<b, 512, 0, stream>>>(logits, pre_idx, indices, n_valid, npad, k);
        return;
    }
    // Path 1 — fused 1-CTA/row: fills machine (b>=132) or small n (5 reads cheap).
    if(b >= 132 || npad <= 16384){
        gvr_fused<BLK><<<b, BLK, 0, stream>>>(logits, pre_idx, indices, n_valid, npad, k);
        return;
    }

    // Path 1b — mid-n valley (npad 16k-49k), latency-bound (NCU: 18% SM). A wider
    // block (1024 thr) cuts per-CTA radix-pass latency vs the 3-launch path.
    if(npad <= 49152){
        gvr_fused<1024><<<b, 1024, 0, stream>>>(logits, pre_idx, indices, n_valid, npad, k);
        return;
    }

    // At one resident CTA per row, the cooperative histogram has no same-row
    // peer to reduce; skip its four whole-grid publication barriers entirely.
    if(b>=111){
        gvr_fused<1024><<<b,1024,0,stream>>>(logits,pre_idx,indices,n_valid,npad,k);
        return;
    }

    // Path 2 — COOPERATIVE single launch for huge-n low-b. grid=(nCTA,b)
    // co-resident. Each row owns a prior-seeded threshold and exact collect.
    // Block size is b-dependent: at mid-b (b>=32) a wider block (1024) lets each
    // row use fewer CTAs (smaller grid-barrier participant count nCTA*b) while
    // still parallelizing each row's solve -> measured +2-6% on mid-b huge-n.
    // Low-b (b<32) keeps 512 (wider block regressed the b=1/b=4 latency regime).
    if(npad > 49152){
        static int maxBlk512=0, maxBlk1024=0, numSM=0;
        if(numSM==0){
            cudaDeviceGetAttribute(&numSM, cudaDevAttrMultiProcessorCount, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlk512, gvr_coop<512>, 512, 0);
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlk1024, gvr_coop<1024>, 1024, 0);
        }
        bool wide = (b >= 32);
        int CBLK = wide ? 1024 : 512;
        int maxBlocksPerSM = wide ? maxBlk1024 : maxBlk512;
        int perCTA_min=CBLK*4;
        int maxCTA=(n_valid+perCTA_min-1)/perCTA_min; if(maxCTA<1) maxCTA=1;
        long capTotal = (long)maxBlocksPerSM*numSM;          // co-residency cap
        if(maxBlocksPerSM>0 && b <= capTotal){
            int perRow = (int)(capTotal / b);                // fill machine
            if(perRow<1) perRow=1;
            int nCTA = perRow<maxCTA?perRow:maxCTA;
            // Wide (mid-b) path: NCU showed these cells are wait/collect-bound,
            // NOT barrier-bound. Use ONE CTA per row (nCTA=1) with a single-pass
            // shared-atomic collect: reads the row once (vs 2x for the reserve
            // scheme) and avoids global-atomic contention. Only when b>=~SMs
            // so the machine stays full.
            bool oneCta = wide && (b >= (3*numSM)/4);
            if(oneCta) nCTA = 1;
            long totalCTA=(long)nCTA*b;
            if(totalCTA <= capTotal){
                static int* gstate=nullptr; static int cap_g=0;
                int need = 1 + b*4*256 + 2*b; // arrive, per-row byte histograms, gt/eq counters
                if(need>cap_g){ if(gstate) cudaFree(gstate); cudaMalloc(&gstate,need*sizeof(int)); cap_g=need; }
                cudaMemsetAsync(gstate,0,need*sizeof(int),stream); // all zero
                dim3 grid(nCTA,b);
                if(oneCta)    gvr_coop<1024,true><<<grid,1024,0,stream>>>(logits,pre_idx,indices,n_valid,npad,k,nCTA,gstate);
                else if(wide) gvr_coop<1024><<<grid,1024,0,stream>>>(logits,pre_idx,indices,n_valid,npad,k,nCTA,gstate);
                else          gvr_coop<512><<<grid,512,0,stream>>>(logits,pre_idx,indices,n_valid,npad,k,nCTA,gstate);
                return;
            }
        }
    }

    // Rare occupancy-query escape: retain the same per-row prior-seeded GVR.
    gvr_topk_launcher(logits,pre_idx,n_valid,indices,b,npad,k,stream);
}
