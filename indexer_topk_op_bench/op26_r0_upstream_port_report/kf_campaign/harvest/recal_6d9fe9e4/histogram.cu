#include "launchers.h"
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <float.h>
namespace cg = cooperative_groups;

// ---------------------------------------------------------------------------
// GVR top-K, cluster-capable (1..8 CTAs), fp32, BS=1.
//   P1: gather pre_idx values -> threshold prior (pmin/pmax).
//   P2/P3 (fused): fine histogram over [lo,hi] AND collect candidates (>=lo)
//        in ONE row scan (elements < lo are skipped -> cheap). Cluster-aggregate
//        the histogram; if count(>=lo) in [k,CAP] the buffer is complete in one
//        pass. Otherwise refine lo/hi (bounded) and re-collect.
//   P4: rank-0 exact radix-select of k-th value among candidates + tie fill.
// ---------------------------------------------------------------------------

#define BLOCK 1024
#define NWARP (BLOCK / 32)
#define CAP 4096            // candidate buffer capacity (rank-0 gather target)
#define NBINS 64            // fine histogram bins for threshold solve

__device__ __forceinline__ unsigned f2key(float x) {
    unsigned f = __float_as_uint(x);
    return f ^ (unsigned)(((int)f >> 31) | 0x80000000);
}
__device__ __forceinline__ float warpMin(float v){
    for(int o=16;o>0;o>>=1) v=fminf(v,__shfl_xor_sync(0xffffffff,v,o));
    return v;
}
__device__ __forceinline__ float warpMax(float v){
    for(int o=16;o>0;o>>=1) v=fmaxf(v,__shfl_xor_sync(0xffffffff,v,o));
    return v;
}

// static shared scratch
__shared__ float s_red[NWARP];
__shared__ int   s_redi[NWARP];
__shared__ int   s_lhist[2][NBINS]; // parity banks keep peer DSMEM reads alive
__shared__ int   s_ahist[NBINS];    // cluster-aggregated histogram
__shared__ int   s_hist[256];       // P4 radix histogram
__shared__ int   s_lcount;          // local candidate count
__shared__ int   s_off;             // offset of this CTA in rank-0 buffer
__shared__ int   s_total;           // total candidates across cluster
__shared__ float s_bf[4];           // broadcast: [lo, hi, spare, spare]
__shared__ int   s_bi[2];           // broadcast: [done, spare]
__shared__ int   s_g, s_posgt, s_poseq;
extern __shared__ char s_dyn[];

template <int CS>
__global__ void __launch_bounds__(BLOCK,1)
topk_kernel(const float* __restrict__ logits,
            const int* __restrict__ pre_idx,
            int n, int k, int* __restrict__ out_indices){
    cg::cluster_group cluster = cg::this_cluster();
    const unsigned brank = cluster.block_rank();
    constexpr unsigned nblk = CS;
    const int tid = threadIdx.x;
    const int warp = tid>>5;
    const int lane = tid&31;
    unsigned* s_ckey = reinterpret_cast<unsigned*>(s_dyn);
    int*      s_cidx = reinterpret_cast<int*>(s_dyn + CAP*sizeof(unsigned));

    long long sa = (long long)n * brank / nblk;
    long long sb = (long long)n * (brank+1) / nblk;
    int slice_a = (int)sa, slice_b = (int)sb;

    // ---------- P1: pre_idx stats ----------
    float pmn = FLT_MAX, pmx = -FLT_MAX;
    for(int j=tid;j<k;j+=BLOCK){
        int idx = pre_idx[j];
        if(idx>=0 && idx<n){ float v = logits[idx]; pmn=fminf(pmn,v); pmx=fmaxf(pmx,v); }
    }
    pmn = warpMin(pmn); pmx = warpMax(pmx);
    if(lane==0){ s_red[warp]=pmn; s_redi[warp]=__float_as_int(pmx); }
    __syncthreads();
    // Every warp redundantly reduces the 32 warp partials.  The extra
    // shuffles are cheaper than publishing through a second CTA barrier.
    pmn = warpMin(s_red[lane]);
    pmx = warpMax(__int_as_float(s_redi[lane]));
    pmn = __shfl_sync(0xffffffffu, pmn, 0);
    pmx = __shfl_sync(0xffffffffu, pmx, 0);

    // ---------- P2: histogram-only threshold solve (no collect) ----------
    float scale = fmaxf(pmx - pmn, fmaxf(fabsf(pmn)*1e-3f, 1e-6f));
    // k distinct previous top-k indices have values >= pmn, therefore
    // count(>=pmn) >= k exactly.  This tighter lower bound concentrates the
    // 64-bin solve on the only relevant interval and cuts refinement syncs.
    float lo = pmn;
    float hi = pmx + 0.125f*scale;
    float tau = lo;

    for(int refine=0; refine<6; refine++){
        float span = hi - lo;
        float inv = (float)NBINS / span;
        int* lhist = s_lhist[refine & 1];
        for(int j=tid;j<NBINS;j+=BLOCK) lhist[j]=0;
        __syncthreads();

        // n is the 64-element padded width; power-of-two cluster slicing
        // therefore gives every CTA a float4-aligned, float4-sized segment.
        const float4* v4 = reinterpret_cast<const float4*>(logits);
        int c0 = slice_a >> 2;
        int c1 = slice_b >> 2;
        for(int i = c0 + tid; i < c1; i += BLOCK){
            float4 f = v4[i];
            float fv[4]={f.x,f.y,f.z,f.w};
            #pragma unroll
            for(int t=0;t<4;t++){
                float v=fv[t];
                    if(v >= lo){ int b=(int)((v-lo)*inv); if(b>=NBINS)b=NBINS-1; atomicAdd(&lhist[b],1); }
            }
        }
        int* ahist;
        if(nblk>1){
            // cluster.sync is also a full CTA rendezvous and release/acquire
            // fence for this block's shared histogram.
            cluster.sync();
            for(int j=tid;j<NBINS;j+=BLOCK){
                int s=0;
                #pragma unroll
                for(unsigned r=0;r<nblk;r++) s += cluster.map_shared_rank(lhist,r)[j];
                s_ahist[j]=s;
            }
            __syncthreads();
            ahist = s_ahist;
        } else {
            __syncthreads();
            ahist = lhist;
        }

        // Redundant warp-parallel suffix scan over the 64 bins.  Each lane
        // owns two adjacent bins; every warp derives the same exact crossing
        // bin and keeps the next bracket in registers, avoiding a serial
        // thread-0 scan and a CTA publication barrier.
        int h0 = ahist[(lane << 1)];
        int h1 = ahist[(lane << 1) + 1];
        int lane_total = h0 + h1;
        int suffix = lane_total;
        #pragma unroll
        for(int d=1; d<32; d<<=1){
            int v = __shfl_down_sync(0xffffffffu, suffix, d);
            if(lane + d < 32) suffix += v;
        }
        int above_lane = suffix - lane_total;
        bool cross1 = above_lane < k && above_lane + h1 >= k;
        bool cross0 = above_lane + h1 < k && above_lane + h1 + h0 >= k;
        unsigned cross_mask = __ballot_sync(0xffffffffu, cross1 || cross0);
        int owner = __ffs(cross_mask) - 1;
        int local_bk = cross1 ? ((lane << 1) + 1) : (lane << 1);
        int local_ck = cross1 ? (above_lane + h1) : (above_lane + h1 + h0);
        int bk = owner >= 0 ? __shfl_sync(0xffffffffu, local_bk, owner) : 0;
        int ckcnt = owner >= 0 ? __shfl_sync(0xffffffffu, local_ck, owner) : 0;
        int total = __shfl_sync(0xffffffffu, suffix, 0);

        float binw = span/(float)NBINS;
        if(total < k){                           // lo too high -> widen down
            hi = lo; lo = lo - 4.0f*scale; tau = lo;
        } else if(ckcnt <= CAP){                 // selected lower edge is valid
            tau = lo + (float)bk*binw;
            break;
        } else {                                 // dense crossing bin: refine it
            float old_lo = lo;
            lo = old_lo + (float)bk*binw;
            hi = old_lo + (float)(bk+1)*binw;
            tau = lo;
        }
    }

    // ---------- P3: single collect pass with tau ----------
    // Track the TRUE max ordered-key of kept candidates (s_bf[3] is only a
    // histogram estimate that can underestimate when values exceed hi, which
    // would make the P4 prefix-skip unsafe).
    if(tid==0){ s_lcount=0; }
    __syncthreads();
    unsigned tmaxk = 0u;
    {
        const float4* v4 = reinterpret_cast<const float4*>(logits);
        int c0 = slice_a >> 2;
        int c1 = slice_b >> 2;
        for(int i = c0 + tid; i < c1; i += BLOCK){
            float4 f = v4[i];
            float fv[4]={f.x,f.y,f.z,f.w};
            #pragma unroll
            for(int t=0;t<4;t++){
                if(fv[t] >= tau){
                    unsigned key = f2key(fv[t]);
                    tmaxk = max(tmaxk, key);
                    int p=atomicAdd(&s_lcount,1);
                    if(p<CAP){s_ckey[p]=key; s_cidx[p]=(i<<2)+t;}
                }
            }
        }
    }
    // reduce tmaxk across block into s_redi[0] (reused; safe: rewritten in P4)
    #pragma unroll
    for(int d=16;d>0;d>>=1) tmaxk=max(tmaxk,__shfl_down_sync(0xffffffffu,tmaxk,d));
    if(lane==0) s_hist[warp]=(int)tmaxk;      // s_hist free until P4
    __syncthreads();
    if(warp==0){
        unsigned v=(lane<NWARP)?(unsigned)s_hist[lane]:0u;
        #pragma unroll
        for(int d=16;d>0;d>>=1) v=max(v,__shfl_down_sync(0xffffffffu,v,d));
        if(lane==0) s_bi[1]=(int)v;           // block-wide max key
    }
    unsigned blk_maxk = 0;
    int m = 0;

    // ---------- gather local buffers into rank-0 (2 cluster syncs) ----------
    if(nblk == 1){
        __syncthreads();
        blk_maxk = (unsigned)s_bi[1];
        m = s_lcount; if(m>CAP) m=CAP;
    } else {
        int lc = s_lcount; if(lc>CAP) lc=CAP;
        if(brank==0 && tid==0){
            // Keep rank 0's existing buffer fixed at offset zero.  This both
            // removes its self-copy and prevents peers from overwriting its
            // source before it has moved.
            s_total = lc;
            s_g = (int)(((unsigned)s_bi[1]) ^ 0x80000000u);
        }
        // Also publishes s_bi[1], replacing its former CTA-only barrier.
        cluster.sync();
        blk_maxk = (unsigned)s_bi[1];
        int* r0cnt = cluster.map_shared_rank(&s_total, 0);
        unsigned* r0key = cluster.map_shared_rank(s_ckey, 0);
        int*      r0idx = cluster.map_shared_rank(s_cidx, 0);
        // publish this CTA's max candidate key into rank-0 (while cluster synced),
        // so P4 reads only rank-0-local memory and peers can exit freely.
        // atomicMax in signed space: XOR the top bit so unsigned key order
        // matches signed int order (keys can exceed 2^31).
        __shared__ int s_base;
        if(tid==0){
            if(brank==0) s_base = 0;
            else {
                atomicMax(cluster.map_shared_rank(&s_g, 0), (int)(blk_maxk ^ 0x80000000u));
                s_base = atomicAdd(r0cnt, lc);
            }
        }
        __syncthreads();
        int base = s_base;
        if(brank != 0){
            for(int i=tid; i<lc; i+=BLOCK){
                int d = base + i;
                if(d < CAP){ r0key[d]=s_ckey[i]; r0idx[d]=s_cidx[i]; }
            }
        }
        cluster.sync();
        m = *cluster.map_shared_rank(&s_total, 0);
        if(m>CAP) m=CAP;
    }

    // ---------- P4: rank-0 exact k-th among candidates ----------
    if(brank==0){
        // Exact-count fast path: P3 retained every value >= tau, so when the
        // survivor count is already k the candidate set is the answer.
        if(m == k){
            for(int i=tid; i<m; i+=BLOCK) out_indices[i] = s_cidx[i];
            return;
        }
        // Prefix-window skip (exact): candidate ordered-keys lie in
        // [f2key(tau), kmax] where kmax is the true cluster-wide max candidate
        // key (tracked in P3). Radix passes whose top byte is common across
        // that range put every candidate in one bin, leaving krem unchanged and
        // found == the common bytes. Start at the first differing byte.
        unsigned kmin = f2key(tau);
        unsigned kmax = (nblk > 1) ? ((unsigned)s_g ^ 0x80000000u) : blk_maxk;
        unsigned diffb = kmin ^ kmax;
        int lead = diffb ? (__clz((int)diffb) >> 3) : 3;   // common leading bytes
        unsigned found = (lead==0) ? 0u : (kmin & (0xFFFFFFFFu << (32 - 8*lead)));
        int krem = k;
        for(int pass=lead; pass<4; pass++){
            int shift = 24 - pass*8;
            unsigned pmask = (shift+8 >= 32) ? 0u : (0xFFFFFFFFu << (shift+8));
            unsigned pref = found & pmask;
            for(int i=tid;i<256;i+=BLOCK) s_hist[i]=0;
            __syncthreads();
            for(int i=tid;i<m;i+=BLOCK){
                unsigned key = s_ckey[i];
                if((key & pmask)==pref){ int d=(key>>shift)&0xFF; atomicAdd(&s_hist[d],1); }
            }
            __syncthreads();
            // k-th-bin selection by a single warp with __shfl (no serial loop).
            if(tid < 32){
                int laneTotal = 0;
                #pragma unroll
                for(int r=0;r<8;r++) laneTotal += s_hist[lane*8 + r];
                int inc = laneTotal;
                #pragma unroll
                for(int d=1; d<32; d<<=1){
                    int t = __shfl_down_sync(0xffffffffu, inc, d);
                    if(lane + d < 32) inc += t;
                }
                int aboveLane = inc - laneTotal;   // sum over lanes > lane
                if(aboveLane < krem && aboveLane + laneTotal >= krem){
                    int prevcum = 0;
                    #pragma unroll
                    for(int r=7;r>=0;r--){
                        int b = lane*8 + r;
                        int hb = s_hist[b];
                        int aboveThis = aboveLane + prevcum;
                        if(aboveThis + hb >= krem && aboveThis < krem){
                            s_redi[0] = b; s_redi[1] = krem - aboveThis;
                        }
                        prevcum += hb;
                    }
                }
            }
            __syncthreads();
            int sel=s_redi[0]; krem=s_redi[1];
            found = (found & pmask) | ((unsigned)sel<<shift);
        }
        unsigned fkey = found;
        // radix invariant: krem == k - #{key > fkey} == ties needed; g == k-krem
        int g = k - krem; int need = krem;
        if(tid==0){ s_posgt=0; s_poseq=0; }
        __syncthreads();
        for(int i=tid;i<m;i+=BLOCK){
            unsigned key=s_ckey[i]; int idx=s_cidx[i];
            if(key>fkey){ int p=atomicAdd(&s_posgt,1); if(p<g) out_indices[p]=idx; }
            else if(key==fkey){ int p=atomicAdd(&s_poseq,1); if(p<need) out_indices[g+p]=idx; }
        }
    }
    // P4 reads only rank-0's own buffer -> peers may exit freely (no final sync).
}

template <int CS>
void launch_histogram(const float* logits, const int* pre_idx, int n, int k,
                      int* out_indices, cudaStream_t stream){
    static bool inited = false;
    size_t smem = (size_t)CAP*sizeof(float) + (size_t)CAP*sizeof(int);
    auto kernel = topk_kernel<CS>;
    if(!inited){
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        cudaFuncSetAttribute(kernel, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
        inited = true;
    }

    cudaLaunchConfig_t config = {0};
    config.gridDim = CS;
    config.blockDim = BLOCK;
    config.dynamicSmemBytes = smem;
    config.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CS;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    config.attrs = attr;
    config.numAttrs = 1;
    cudaLaunchKernelEx(&config, kernel, logits, pre_idx, n, k, out_indices);
}

void histogram_launcher(const float* logits, const int* pre_idx, int n, int k,
                   int* out_indices, cudaStream_t stream){
    if(n >= 131072) launch_histogram<16>(logits, pre_idx, n, k, out_indices, stream);
    else if(n >= 32768) launch_histogram<16>(logits, pre_idx, n, k, out_indices, stream);
    else if(n >= 16384) launch_histogram<4>(logits, pre_idx, n, k, out_indices, stream);
    else launch_histogram<1>(logits, pre_idx, n, k, out_indices, stream);
}
