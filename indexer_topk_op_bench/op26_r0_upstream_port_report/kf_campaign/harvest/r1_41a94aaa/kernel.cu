// Single-launch top-k index selection for one row of fp32 logits (B200).
//
// The eval flushes L2 before every timed iteration, so cost is dominated by
// cold-start effects (instruction fetch, serialized cold misses) and
// same-address shared-memory atomic serialization on hot histogram bins —
// not data volume. Two lean kernels:
//  - topk_one: single CTA for npad <= 8256. Row copied raw into shared
//    memory (all loads issued upfront), transformed in place with the first
//    radix round fused (1024 bins x 2 warp-parity replicas + consecutive-
//    pair add merging to cut hot-bin serialization); all rounds + emission
//    run from smem. No global scratch, no fences, no barriers.
//  - topk_multi: one CTA per 2048-element chunk staged in smem. Up to three
//    grid-wide radix rounds (2048-bin global histogram, epoch grid barrier
//    with acq_rel/release atomics, redundant per-CTA suffix scan), then a
//    distributed two-pass emission: each CTA reserves output slots with one
//    atomic for mandatory winners and one for tie ranks; tie slot indices
//    need no atomic at all. Cost is independent of tie mass (real indexer
//    logits carry huge tie plateaus that k=1024/2048 reaches into).
//
// Keys are the monotone map u = f ^ ((f>>31)|0x80000000); padding/invalid
// lanes are stored as u=0 and skipped everywhere (the k-th value is
// guaranteed > pad, so the skip can never drop a selectable element).

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kRadixBits = 11;
constexpr int kRadix = 1 << kRadixBits;

__device__ __align__(128) unsigned g_hist[3][kRadix];
__device__ __align__(128) unsigned g_out_count;
__device__ __align__(128) unsigned g_tie_count;
__device__ __align__(128) unsigned g_done;
__device__ __align__(128) unsigned g_arrive[4];
__device__ __align__(128) unsigned g_epoch;  // monotonic; never reset

__device__ __forceinline__ unsigned f2u(unsigned f) {
  return f ^ (0x80000000u | (unsigned)(-(int)(f >> 31)));
}

__device__ __forceinline__ unsigned atom_add_acq_rel(unsigned* p, unsigned v) {
  unsigned old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0,[%1],%2;"
               : "=r"(old) : "l"(p), "r"(v) : "memory");
  return old;
}
__device__ __forceinline__ void red_add_release(unsigned* p, unsigned v) {
  asm volatile("red.release.gpu.global.add.u32 [%0],%1;" ::"l"(p), "r"(v)
               : "memory");
}
__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0,[%1];" : "=r"(v) : "l"(p)
               : "memory");
  return v;
}

__device__ __forceinline__ unsigned warp_agg_inc(unsigned* ctr) {
  unsigned mask = __activemask();
  int lane = threadIdx.x & 31;
  int leader = __ffs(mask) - 1;
  unsigned base = 0;
  if (lane == leader) base = atomicAdd(ctr, (unsigned)__popc(mask));
  base = __shfl_sync(mask, base, leader);
  return base + (unsigned)__popc(mask & ((1u << lane) - 1u));
}

struct Sel {
  unsigned digit, c_above, m;
};

// Block-wide suffix scan over hist[0..R); locates the bin where the
// cumulative count from the top crosses `need`. hist may be smem or global.
// Rounds with fewer digits than log2(R) leave the upper bins zeroed.
template <int NT, int R>
__device__ __forceinline__ Sel select_body(const unsigned* hist, unsigned need,
                                           unsigned* s_scan,
                                           unsigned* s_bcast) {
  const int tid = threadIdx.x;
  constexpr int bpt = R / NT;
  const int base = tid * bpt;
  unsigned loc[bpt];
  unsigned lsum = 0;
#pragma unroll
  for (int j = 0; j < bpt; ++j) {
    loc[j] = hist[base + j];
    lsum += loc[j];
  }
  unsigned s = lsum;
  const int lane = tid & 31;
#pragma unroll
  for (int off = 1; off < 32; off <<= 1) {
    unsigned v = __shfl_down_sync(0xffffffffu, s, off);
    if (lane + off < 32) s += v;
  }
  const int warp = tid >> 5;
  if (lane == 0) s_scan[warp] = s;
  __syncthreads();
  unsigned wexcl = 0;
  for (int w = warp + 1; w < NT / 32; ++w) wexcl += s_scan[w];
  const unsigned incl = s + wexcl;
  const unsigned excl = incl - lsum;
  if (excl < need && need <= incl) {
    unsigned acc = excl;
#pragma unroll
    for (int j = bpt - 1; j >= 0; --j) {
      unsigned h = loc[j];
      if (need <= acc + h) {
        s_bcast[0] = (unsigned)(base + j);
        s_bcast[1] = acc;
        s_bcast[2] = h;
        break;
      }
      acc += h;
    }
  }
  __syncthreads();
  Sel r;
  r.digit = s_bcast[0];
  r.c_above = s_bcast[1];
  r.m = s_bcast[2];
  return r;
}

// ------------------------- single-CTA kernel -------------------------
template <int NT>
__global__ void __launch_bounds__(NT, 1)
topk_one(const float* __restrict__ logits, int* __restrict__ out, int n4,
         int npad, int n, unsigned k, int E) {
  const int tid = threadIdx.x;
  extern __shared__ __align__(16) unsigned su[];  // npad keys
  __shared__ unsigned s_hist[kRadix];  // 2 x 1024 replicas in round 0
  __shared__ unsigned s_scan[NT / 32];
  __shared__ unsigned s_bcast[3];
  __shared__ unsigned s_cnt[2];  // out cursor, tie count

  const uint4* __restrict__ in4 = reinterpret_cast<const uint4*>(logits);
  uint4* su4 = reinterpret_cast<uint4*>(su);
  uint4 r[5];  // npad <= 8256 -> at most 5 loads/thread, all in flight
  bool rv[5];
#pragma unroll
  for (int j = 0; j < 5; ++j) {
    const int i = tid + j * NT;
    rv[j] = (j < E) && (i < n4);
    if (rv[j]) r[j] = in4[i];
  }
  // warm the output lines while data loads are in flight
  if (tid < (int)(k >> 5)) (void)*(volatile int*)(out + tid * 32);
#pragma unroll
  for (int j = 0; j < 5; ++j) {
    if (rv[j]) su4[tid + j * NT] = r[j];
  }
  for (int b = tid; b < kRadix; b += NT) s_hist[b] = 0;
  __syncthreads();

  // transform in place; first round (10 bits, 1024 bins) fused. Two
  // warp-parity histogram replicas + same-bin pairing halve hot-bin
  // serialization. Threads own consecutive word pairs (uint2).
  {
    const unsigned rep = (tid & 32) ? 1024u : 0u;
    uint2* su2 = reinterpret_cast<uint2*>(su);
    const int np2 = npad >> 1;
    for (int p = tid; p < np2; p += NT) {
      uint2 v = su2[p];
      const int w = 2 * p;
      const unsigned u0 = (w < n) ? f2u(v.x) : 0u;
      const unsigned u1 = (w + 1 < n) ? f2u(v.y) : 0u;
      su2[p] = make_uint2(u0, u1);
      const unsigned b0 = u0 >> 22, b1 = u1 >> 22;
      if (u0 && u1 && b0 == b1) {
        atomicAdd(&s_hist[rep + b0], 2u);
      } else {
        if (u0) atomicAdd(&s_hist[rep + b0], 1u);
        if (u1) atomicAdd(&s_hist[rep + b1], 1u);
      }
    }
  }
  __syncthreads();
  for (int b = tid; b < 1024; b += NT) s_hist[b] += s_hist[b + 1024];
  __syncthreads();

  Sel sel = select_body<NT, 1024>(s_hist, k, s_scan, s_bcast);
  unsigned prefix = sel.digit;
  unsigned need = k - sel.c_above;
  int consumed = 10;
  while (consumed < 32 && sel.m != need) {
    const int D = (32 - consumed < kRadixBits) ? 32 - consumed : kRadixBits;
    const int shift = 32 - consumed - D;
    const unsigned mask = (1u << D) - 1u;
    for (int b = tid; b < kRadix; b += NT) s_hist[b] = 0;
    __syncthreads();
    for (int w = tid; w < npad; w += NT) {
      const unsigned u = su[w];
      if (!u || (u >> (32 - consumed)) != prefix) continue;
      atomicAdd(&s_hist[(u >> shift) & mask], 1u);
    }
    __syncthreads();
    sel = select_body<NT, kRadix>(s_hist, need, s_scan, s_bcast);
    prefix = (prefix << D) | sel.digit;
    consumed += D;
    need -= sel.c_above;
  }

  const int shift = 32 - consumed;
  if (tid == 0) {
    s_cnt[0] = 0;
    s_cnt[1] = 0;
  }
  __syncthreads();
  for (int w = tid; w < npad; w += NT) {
    const unsigned u = su[w];
    if (!u) continue;
    const unsigned hi = shift ? (u >> shift) : u;
    if (hi > prefix) {
      out[warp_agg_inc(&s_cnt[0])] = w;
    } else if (hi == prefix) {
      const unsigned t = warp_agg_inc(&s_cnt[1]);
      if (t < need) out[(k - need) + t] = w;
    }
  }
}

// ------------------------- multi-CTA kernel -------------------------
__global__ void __launch_bounds__(256, 1)
topk_multi(const float* __restrict__ logits, int* __restrict__ out, int n4,
           int n, unsigned k, int E) {
  constexpr int NT = 256;
  const int tid = threadIdx.x;
  const int nblk = gridDim.x;
  __shared__ __align__(16) unsigned su[8 * 4 * NT];  // this CTA's chunk
  __shared__ unsigned s_hist[kRadix];
  __shared__ unsigned s_scan[NT / 32];
  __shared__ unsigned s_bcast[3];
  __shared__ unsigned s_cnt[2];
  __shared__ unsigned s_base[2];

  unsigned epoch0 = 0;
  if (tid == 0) epoch0 = *(volatile unsigned*)&g_epoch;

  const int i0 = blockIdx.x * (NT * E);  // uint4 units
  const int e0 = 4 * i0;
  const uint4* __restrict__ in4 = reinterpret_cast<const uint4*>(logits);
  uint4* su4 = reinterpret_cast<uint4*>(su);

  // issue chunk copies; loads pipeline freely
  if (E == 2) {
    const int ia = i0 + tid, ib = i0 + tid + NT;
    if (ia < n4) su4[tid] = in4[ia];
    if (ib < n4) su4[tid + NT] = in4[ib];
  } else {  // general fallback (npad > 262144); not hit by real workloads
    for (int j = 0; j < E; ++j) {
      const int i = i0 + tid + j * NT;
      if (i < n4) su4[tid + j * NT] = in4[i];
    }
  }
  // warm scratch + output cache lines while the data loads are in flight
  if (tid < 192) (void)*(volatile unsigned*)&g_hist[0][tid * 32];
  if (tid < (int)(k >> 5)) (void)*(volatile int*)(out + tid * 32);
  if (tid == 0) {
    (void)*(volatile unsigned*)&g_out_count;
    (void)*(volatile unsigned*)&g_tie_count;
    (void)*(volatile unsigned*)&g_done;
    (void)*(volatile unsigned*)&g_arrive[0];
  }
  for (int b = tid; b < kRadix; b += NT) s_hist[b] = 0;
  __syncthreads();

  // transform in place, first radix round fused
  const int nw = 4 * NT * E;  // chunk words
  const int nl = n - e0;      // valid words in this chunk (may exceed nw)
  for (int w = tid; w < nw; w += NT) {
    const unsigned u = (w < nl) ? f2u(su[w]) : 0u;
    su[w] = u;
    if (u) atomicAdd(&s_hist[u >> 21], 1u);
  }
  __syncthreads();

  unsigned prefix = 0, need = k;
  int consumed = 0, round = 0;
  Sel sel;
  for (;;) {
    const int D = (32 - consumed < kRadixBits) ? 32 - consumed : kRadixBits;
    const int shift = 32 - consumed - D;
    if (round > 0) {  // round 0 histogram was fused with the transform
      const unsigned mask = (1u << D) - 1u;
      for (int b = tid; b < kRadix; b += NT) s_hist[b] = 0;
      __syncthreads();
      for (int w = tid; w < nw; w += NT) {
        const unsigned u = su[w];
        if (!u || (u >> (32 - consumed)) != prefix) continue;
        atomicAdd(&s_hist[(u >> shift) & mask], 1u);
      }
      __syncthreads();
    }
    for (int b = tid; b < kRadix; b += NT) {
      const unsigned c = s_hist[b];
      if (c) atomicAdd(&g_hist[round][b], c);
    }
    __syncthreads();
    if (tid == 0) {
      const unsigned r = atom_add_acq_rel(&g_arrive[round], 1u);
      if (r == (unsigned)(nblk - 1)) {
        g_arrive[round] = 0;
        red_add_release(&g_epoch, 1u);
      } else {
        const unsigned tgt = (unsigned)(round + 1);
        while (ld_acquire(&g_epoch) - epoch0 < tgt) __nanosleep(64);
      }
    }
    __syncthreads();
    sel = select_body<NT, kRadix>(g_hist[round], need, s_scan, s_bcast);
    prefix = (prefix << D) | sel.digit;
    consumed += D;
    need -= sel.c_above;
    round++;
    if (consumed >= 32 || sel.m == need) break;
  }

  // distributed emission: pass A counts, one slot reservation per CTA,
  // pass B writes. Mandatory slots [0, k-need); tie slot = k-need + rank.
  const int shift = 32 - consumed;
  if (tid == 0) {
    s_cnt[0] = 0;
    s_cnt[1] = 0;
  }
  __syncthreads();
  for (int w = tid; w < nw; w += NT) {
    const unsigned u = su[w];
    if (!u) continue;
    const unsigned hi = shift ? (u >> shift) : u;
    if (hi > prefix) {
      (void)warp_agg_inc(&s_cnt[0]);
    } else if (hi == prefix) {
      (void)warp_agg_inc(&s_cnt[1]);
    }
  }
  __syncthreads();
  if (tid == 0) {
    s_base[0] = s_cnt[0] ? atomicAdd(&g_out_count, s_cnt[0]) : 0u;
    s_base[1] = s_cnt[1] ? atomicAdd(&g_tie_count, s_cnt[1]) : 0u;
    s_cnt[0] = 0;
    s_cnt[1] = 0;
  }
  __syncthreads();
  const unsigned mbase = s_base[0], tbase = s_base[1];
  for (int w = tid; w < nw; w += NT) {
    const unsigned u = su[w];
    if (!u) continue;
    const unsigned hi = shift ? (u >> shift) : u;
    if (hi > prefix) {
      out[mbase + warp_agg_inc(&s_cnt[0])] = e0 + w;
    } else if (hi == prefix) {
      const unsigned t = tbase + warp_agg_inc(&s_cnt[1]);
      if (t < need) out[(k - need) + t] = e0 + w;
    }
  }
  __syncthreads();
  if (tid == 0) s_base[0] = atom_add_acq_rel(&g_done, 1u);
  __syncthreads();
  if (s_base[0] != (unsigned)(nblk - 1)) return;
  // last CTA resets scratch for the next launch (posted stores)
  for (int b = tid; b < round * kRadix; b += NT) (&g_hist[0][0])[b] = 0;
  if (tid == 0) {
    g_out_count = 0;
    g_tie_count = 0;
    g_done = 0;
  }
}

}  // namespace

extern "C" void topk_launch(const void* logits, void* out, int npad, int n,
                            int k, cudaStream_t stream) {
  const int n4 = npad >> 2;
  const float* lp = reinterpret_cast<const float*>(logits);
  int* op = reinterpret_cast<int*>(out);
  if (npad <= 2048) {
    const int E = (n4 + 511) >> 9;  // ceil(n4/512)
    topk_one<512><<<1, 512, (size_t)npad * 4, stream>>>(lp, op, n4, npad, n,
                                                        (unsigned)k, E);
  } else if (npad <= 8256) {
    const int E = (n4 + 1023) >> 10;  // ceil(n4/1024)
    topk_one<1024><<<1, 1024, (size_t)npad * 4, stream>>>(lp, op, n4, npad, n,
                                                          (unsigned)k, E);
  } else {
    int nblk = (n4 + 511) >> 9;  // one CTA per 2048 elements
    if (nblk > 128) nblk = 128;
    const int E = (n4 + 256 * nblk - 1) / (256 * nblk);
    topk_multi<<<nblk, 256, 0, stream>>>(lp, op, n4, n, (unsigned)k, E);
  }
}
