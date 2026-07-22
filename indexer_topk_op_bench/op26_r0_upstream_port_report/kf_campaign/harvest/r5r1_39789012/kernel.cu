// Batched GVR (guess-verify-refine) top-k for DeepSeek indexer decode, fp32, B200 (sm_100a).
//
// Per row (grid.y = b):
//   P1: gather logits[pre_idx] (temporal warm hint), build rung ladder of candidate
//       thresholds from the hint value CCDF (+ downward extrapolation, exp-CCDF model).
//   P2: one multi-threshold count pass measures the global CCDF at all rungs at once;
//       if no rung lands in [K, kC], iterate a log-space secant solve.
//   P3: collect all (value, index) pairs >= threshold into CTA0 shared memory.
//   P4: exact refine: bitwise radix select of the K-th key among candidates, then a
//       strict-greater + tie-ticket writeback (tie-robust exactness).
// Value-plateau fallback (max-below descent + direct global emit) keeps the kernel
// exact when no threshold lands a count inside [K, kC].
// Direct exact path (per-row full radix select) for npad <= 12288.
//
// Batch scaling: CTAs-per-row (cluster size CS) shrinks as b grows; register-resident
// rows for latency tiers, streaming re-reads with a reduced 4-rung ladder at high b.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cfloat>
#include <cmath>

namespace cg = cooperative_groups;

constexpr int MAXPASS = 8;
// All thresholds are clamped above the -FLT_MAX row padding so pads never count.
constexpr float TMIN = -3.2e38f;

__device__ __forceinline__ unsigned f2u(float f) {
  unsigned u = __float_as_uint(f);
  return u ^ ((u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}

// Warp-aggregated ticket counter. All 32 lanes of the warp must be converged.
__device__ __forceinline__ int atomic_agg(int* ctr, bool active) {
  unsigned mask = __ballot_sync(0xFFFFFFFFu, active);
  if (!active) return -1;
  int lane = threadIdx.x & 31;
  int leader = __ffs(mask) - 1;
  int base = 0;
  if (lane == leader) base = atomicAdd(ctr, __popc(mask));
  base = __shfl_sync(mask, base, leader);
  return base + __popc(mask & ((1u << lane) - 1u));
}

// Pick the histogram bin holding the `need`-th largest element (bins ascending in
// value; scan from the top). sel[0] = bin, sel[1] = remaining need within the bin.
template <int TB>
__device__ __forceinline__ void hist_select(const int* hist, int need, int* sel, int* scr,
                                            int tid) {
  int lane = tid & 31, wid = tid >> 5;
  int v = 0, x = 0;
  if (tid < 256) {
    v = hist[tid];
    x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int y = __shfl_down_sync(0xFFFFFFFFu, x, o);
      if (lane + o < 32) x += y;
    }
    if (lane == 0) scr[wid] = x;  // warp total
  }
  __syncthreads();
  if (tid < 256) {
    int S = x;
    for (int w = wid + 1; w < 8; ++w) S += scr[w];
    if (S >= need && S - v < need) {
      sel[0] = tid;
      sel[1] = need - (S - v);
    }
  }
  __syncthreads();
}

// ---------------------------------------------------------------------------
// Direct exact path: one CTA per row, full radix select over the row in smem.
// This is the analytic trivial-convergence limit of the threshold solve.
// ---------------------------------------------------------------------------
struct DCtl {
  int hist[256];
  int sel[2];
  int scr[8];
  int cnt_m, cnt_t;
};

template <int TB>
__global__ void __launch_bounds__(TB, 2) direct_kernel(const float* __restrict__ logits,
                                                       int* __restrict__ out, int npad, int n,
                                                       int K) {
  extern __shared__ unsigned skey[];
  __shared__ DCtl sm;
  int tid = threadIdx.x;
  int row = blockIdx.x;
  const float4* rp = (const float4*)(logits + (size_t)row * npad);
  int npad4 = npad >> 2;
  int M = n - K;  // complement size; tiny M -> min-descent complement path
  bool bottom = (M <= 8);
  unsigned mkey = bottom ? 0xFFFFFFFFu : 0u;  // masked-slot key
  for (int i = tid; i < npad4; i += TB) {
    float4 v = rp[i];
    int e = 4 * i;
    skey[e + 0] = (e + 0 < n) ? f2u(v.x) : mkey;
    skey[e + 1] = (e + 1 < n) ? f2u(v.y) : mkey;
    skey[e + 2] = (e + 2 < n) ? f2u(v.z) : mkey;
    skey[e + 3] = (e + 3 < n) ? f2u(v.w) : mkey;
  }
  if (tid == 0) {
    sm.cnt_m = 0;
    sm.cnt_t = 0;
  }
  __syncthreads();
  int* orow0 = out + (size_t)row * K;
  if (bottom) {
    // Emit the complement of the M smallest (tie-robust): successive min-descents.
    if (M <= 0) {
      for (int i = tid; i < K; i += TB) orow0[i] = i;
      return;
    }
    unsigned prev = 0;
    int total = 0, needLow = 0;
    unsigned mth = 0;
    for (;;) {
      unsigned mn = 0xFFFFFFFFu;
      int cnt = 0;
      for (int i = tid; i < npad; i += TB) {
        unsigned key = skey[i];
        if (key > prev) {
          if (key < mn) {
            mn = key;
            cnt = 1;
          } else if (key == mn) {
            cnt++;
          }
        }
      }
#pragma unroll
      for (int o = 16; o; o >>= 1) {
        unsigned omn = __shfl_down_sync(0xFFFFFFFFu, mn, o);
        int ocnt = __shfl_down_sync(0xFFFFFFFFu, cnt, o);
        if (omn < mn) {
          mn = omn;
          cnt = ocnt;
        } else if (omn == mn) {
          cnt += ocnt;
        }
      }
      int lane = tid & 31, wid = tid >> 5;
      if (lane == 0) {
        sm.hist[wid] = (int)mn;
        sm.hist[64 + wid] = cnt;
      }
      __syncthreads();
      if (tid == 0) {
        unsigned bmn = 0xFFFFFFFFu;
        int bcnt = 0;
        for (int w = 0; w < TB / 32; ++w) {
          unsigned omn = (unsigned)sm.hist[w];
          int ocnt = sm.hist[64 + w];
          if (omn < bmn) {
            bmn = omn;
            bcnt = ocnt;
          } else if (omn == bmn) {
            bcnt += ocnt;
          }
        }
        sm.hist[128] = (int)bmn;
        sm.hist[129] = bcnt;
      }
      __syncthreads();
      unsigned gmn = (unsigned)sm.hist[128];
      int gcnt = sm.hist[129];
      __syncthreads();
      total += gcnt;
      if (total >= M) {
        mth = gmn;
        needLow = M - (total - gcnt);
        break;
      }
      prev = gmn;
    }
    int npr = ((npad + TB - 1) / TB) * TB;
    for (int i = tid; i < npr; i += TB) {
      bool inb = i < n;
      unsigned key = inb ? skey[i] : 0u;
      bool hi = inb && key > mth;
      bool tie = inb && key == mth;
      int t = atomic_agg(&sm.cnt_t, tie);
      bool em = hi || (tie && t >= needLow);
      int p = atomic_agg(&sm.cnt_m, em);
      if (em) orow0[p] = i;
    }
    return;
  }
  unsigned prefix = 0;
  int need = K;
  for (int L = 3; L >= 0; --L) {
    if (tid < 256) sm.hist[tid] = 0;
    __syncthreads();
    int sh = L * 8;
    for (int i = tid; i < npad; i += TB) {
      unsigned key = skey[i];
      bool mt = (L == 3) || ((key >> (sh + 8)) == (prefix >> (sh + 8)));
      if (mt) atomicAdd(&sm.hist[(key >> sh) & 255], 1);
    }
    __syncthreads();
    hist_select<TB>(sm.hist, need, sm.sel, sm.scr, tid);
    prefix |= ((unsigned)sm.sel[0]) << sh;
    need = sm.sel[1];
    __syncthreads();
  }
  unsigned kth = prefix;
  int m0 = K - need;
  int* orow = out + (size_t)row * K;
  int npr = ((npad + TB - 1) / TB) * TB;
  for (int i = tid; i < npr; i += TB) {
    bool inb = i < npad;
    unsigned key = inb ? skey[i] : 0u;
    bool strict = inb && key > kth;
    bool tie = inb && key == kth;
    int p = atomic_agg(&sm.cnt_m, strict);
    if (strict) orow[p] = i;
    int t = atomic_agg(&sm.cnt_t, tie);
    if (tie && t < need) orow[m0 + t] = i;
  }
}

// ---------------------------------------------------------------------------
// GVR path
// ---------------------------------------------------------------------------
template <int TB, int R, int KC>
struct GSmem {
  static constexpr int NWARP = TB / 32;
  unsigned long long cand[KC];  // packed (key<<32 | index); CTA0 receives collect
  int ptcnt[R * TB];            // per-thread >= counts per rung (collect offsets)
  int hist[256];
  int sel[2];
  int scr8[8];
  float rungs[R];  // thresholds, descending
  int rcnt_local[R];
  int rcnt[R];  // cluster-wide counts
  int rpre[R];  // exclusive prefix of peer CTA counts
  int ipartial[2][R];
  float fpartial[2];
  float fwred[2 * NWARP];
  int iwred[NWARP];
  float hminmax[2];
  float lam;
  float t0;
  int ccnt;
  int cnt_m, cnt_t;
};

template <int NR, int CS, int TB, int R, int KC, int MAXV, int NA>
__device__ __forceinline__ void count_pass(GSmem<TB, R, KC>* s, const float4 (&a)[NA],
                                           const float4* rowp4, int f0, int f1, int tid) {
  float tr[NR];
  int c[NR];
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    tr[r] = s->rungs[r];
    c[r] = 0;
  }
  if constexpr (MAXV > 0) {
#pragma unroll
    for (int u = 0; u < MAXV; ++u) {
#pragma unroll
      for (int r = 0; r < NR; ++r)
        c[r] += (int)(a[u].x >= tr[r]) + (int)(a[u].y >= tr[r]) + (int)(a[u].z >= tr[r]) +
                (int)(a[u].w >= tr[r]);
    }
  } else {
    for (int i = f0 + tid; i < f1; i += TB) {
      float4 v = rowp4[i];
#pragma unroll
      for (int r = 0; r < NR; ++r)
        c[r] += (int)(v.x >= tr[r]) + (int)(v.y >= tr[r]) + (int)(v.z >= tr[r]) +
                (int)(v.w >= tr[r]);
    }
  }
#pragma unroll
  for (int r = 0; r < NR; ++r) {
    s->ptcnt[r * TB + tid] = c[r];
    int w = __reduce_add_sync(0xFFFFFFFFu, c[r]);
    if ((tid & 31) == 0) atomicAdd(&s->rcnt_local[r], w);
  }
}

template <int CS, int TB, int R, int KC>
__device__ __forceinline__ void exchange_counts(GSmem<TB, R, KC>* s, int par, int rank, int tid,
                                                int nr) {
  __syncthreads();  // rcnt_local complete
  if constexpr (CS == 1) {
    if (tid < nr) {
      s->rcnt[tid] = s->rcnt_local[tid];
      s->rpre[tid] = 0;
    }
    __syncthreads();
  } else {
    cg::cluster_group cl = cg::this_cluster();
    if (tid < nr) s->ipartial[par][tid] = s->rcnt_local[tid];
    cl.sync();
    if (tid < nr) {
      int tot = 0, pre = 0;
#pragma unroll
      for (int c = 0; c < CS; ++c) {
        GSmem<TB, R, KC>* ps = cl.map_shared_rank(s, c);
        int v = ps->ipartial[par][tid];
        tot += v;
        if (c < rank) pre += v;
      }
      s->rcnt[tid] = tot;
      s->rpre[tid] = pre;
    }
    __syncthreads();
  }
}

// Max row value strictly below thi (excludes anything <= TMIN territory implicitly:
// -inf register padding never wins unless nothing real is below, which contradicts
// the caller's count invariant).
template <int CS, int TB, int R, int KC, int MAXV, int NA>
__device__ __forceinline__ float max_below(GSmem<TB, R, KC>* s, const float4 (&a)[NA],
                                           const float4* rowp4, int f0, int f1, float thi,
                                           int par, int tid) {
  constexpr int NWARP = TB / 32;
  float m = -FLT_MAX;
  if constexpr (MAXV > 0) {
#pragma unroll
    for (int u = 0; u < MAXV; ++u) {
      if (a[u].x < thi) m = fmaxf(m, a[u].x);
      if (a[u].y < thi) m = fmaxf(m, a[u].y);
      if (a[u].z < thi) m = fmaxf(m, a[u].z);
      if (a[u].w < thi) m = fmaxf(m, a[u].w);
    }
  } else {
    for (int i = f0 + tid; i < f1; i += TB) {
      float4 v = rowp4[i];
      if (v.x < thi) m = fmaxf(m, v.x);
      if (v.y < thi) m = fmaxf(m, v.y);
      if (v.z < thi) m = fmaxf(m, v.z);
      if (v.w < thi) m = fmaxf(m, v.w);
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xFFFFFFFFu, m, o));
  int lane = tid & 31, wid = tid >> 5;
  if (lane == 0) s->fwred[wid] = m;
  __syncthreads();
  if (tid == 0) {
    float mm = -FLT_MAX;
    for (int w = 0; w < NWARP; ++w) mm = fmaxf(mm, s->fwred[w]);
    s->fpartial[par] = mm;
  }
  if constexpr (CS == 1) {
    __syncthreads();
    return s->fpartial[par];
  } else {
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();
    float mm = -FLT_MAX;
#pragma unroll
    for (int rr = 0; rr < CS; ++rr) {
      GSmem<TB, R, KC>* ps = cl.map_shared_rank(s, rr);
      mm = fmaxf(mm, ps->fpartial[par]);
    }
    return mm;
  }
}

template <int CS, int TB, int R, int KC, int MAXV, int NA>
__device__ __forceinline__ void collect(GSmem<TB, R, KC>* s, const float4 (&a)[NA],
                                        const float4* rowp4, int f0, int f1, float t, int ptrow,
                                        int cpre, int tid) {
  int mycnt = s->ptcnt[ptrow * TB + tid];
  int lane = tid & 31, wid = tid >> 5;
  int x = mycnt;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int y = __shfl_up_sync(0xFFFFFFFFu, x, o);
    if (lane >= o) x += y;
  }
  if (lane == 31) s->iwred[wid] = x;
  __syncthreads();
  int woff = 0;
  for (int w = 0; w < wid; ++w) woff += s->iwred[w];
  int off = cpre + woff + (x - mycnt);
  unsigned long long* dst;
  if constexpr (CS == 1) {
    dst = s->cand;
  } else {
    dst = cg::this_cluster().map_shared_rank(s, 0)->cand;
  }
  if constexpr (MAXV > 0) {
#pragma unroll
    for (int u = 0; u < MAXV; ++u) {
      int e0 = 4 * (f0 + tid + u * TB);
      if (a[u].x >= t) dst[off++] = ((unsigned long long)f2u(a[u].x) << 32) | (unsigned)(e0 + 0);
      if (a[u].y >= t) dst[off++] = ((unsigned long long)f2u(a[u].y) << 32) | (unsigned)(e0 + 1);
      if (a[u].z >= t) dst[off++] = ((unsigned long long)f2u(a[u].z) << 32) | (unsigned)(e0 + 2);
      if (a[u].w >= t) dst[off++] = ((unsigned long long)f2u(a[u].w) << 32) | (unsigned)(e0 + 3);
    }
  } else {
    for (int i = f0 + tid; i < f1; i += TB) {
      float4 v = rowp4[i];
      int e0 = 4 * i;
      if (v.x >= t) dst[off++] = ((unsigned long long)f2u(v.x) << 32) | (unsigned)(e0 + 0);
      if (v.y >= t) dst[off++] = ((unsigned long long)f2u(v.y) << 32) | (unsigned)(e0 + 1);
      if (v.z >= t) dst[off++] = ((unsigned long long)f2u(v.z) << 32) | (unsigned)(e0 + 2);
      if (v.w >= t) dst[off++] = ((unsigned long long)f2u(v.w) << 32) | (unsigned)(e0 + 3);
    }
  }
  __syncthreads();
}

// Plateau fallback: everything >= tstrict is mandatory (m of them, m < K), remainder
// filled by tie tickets at value w. Row-local counters live in CTA0 smem.
template <int CS, int TB, int R, int KC, int MAXV, int NA>
__device__ __forceinline__ void plateau_emit(GSmem<TB, R, KC>* s, const float4 (&a)[NA],
                                             const float4* rowp4, int f0, int f1, float tstrict,
                                             float w, int m, int K, int n, int* orow, int tid) {
  int *pm, *pt;
  if constexpr (CS == 1) {
    pm = &s->cnt_m;
    pt = &s->cnt_t;
  } else {
    GSmem<TB, R, KC>* s0 = cg::this_cluster().map_shared_rank(s, 0);
    pm = &s0->cnt_m;
    pt = &s0->cnt_t;
  }
  if constexpr (MAXV > 0) {
#pragma unroll
    for (int u = 0; u < MAXV; ++u) {
#pragma unroll
      for (int cmp = 0; cmp < 4; ++cmp) {
        float v = cmp == 0 ? a[u].x : (cmp == 1 ? a[u].y : (cmp == 2 ? a[u].z : a[u].w));
        int idx = 4 * (f0 + tid + u * TB) + cmp;
        bool strict = (v >= tstrict) && idx < n;
        bool tie = (!strict) && (v == w) && idx < n;
        int p = atomic_agg(pm, strict);
        if (strict) orow[p] = idx;
        int t = atomic_agg(pt, tie);
        if (tie && (m + t) < K) orow[m + t] = idx;
      }
    }
  } else {
    int span = f1 - f0;
    int f1r = f0 + ((span + TB - 1) / TB) * TB;
    for (int i = f0 + tid; i < f1r; i += TB) {
      bool inb = i < f1;
      float4 v = inb ? rowp4[i] : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
      int e0 = 4 * i;
#pragma unroll
      for (int cmp = 0; cmp < 4; ++cmp) {
        float vv = cmp == 0 ? v.x : (cmp == 1 ? v.y : (cmp == 2 ? v.z : v.w));
        int idx = e0 + cmp;
        bool strict = inb && (vv >= tstrict) && idx < n;
        bool tie = inb && (!strict) && (vv == w) && idx < n;
        int p = atomic_agg(pm, strict);
        if (strict) orow[p] = idx;
        int t = atomic_agg(pt, tie);
        if (tie && (m + t) < K) orow[m + t] = idx;
      }
    }
  }
}

// P4: exact radix refine of the K-th key among c candidates in s->cand, then
// strict-greater + tie-ticket writeback. Single CTA.
template <int TB, int R, int KC>
__device__ __forceinline__ void gvr_refine(GSmem<TB, R, KC>* s, int c, int K, int* orow,
                                           int tid) {
  unsigned prefix = 0;
  int need = K;
  for (int L = 3; L >= 0; --L) {
    if (tid < 256) s->hist[tid] = 0;
    __syncthreads();
    int sh = L * 8;
    for (int i = tid; i < c; i += TB) {
      unsigned key = (unsigned)(s->cand[i] >> 32);
      bool mt = (L == 3) || ((key >> (sh + 8)) == (prefix >> (sh + 8)));
      if (mt) atomicAdd(&s->hist[(key >> sh) & 255], 1);
    }
    __syncthreads();
    hist_select<TB>(s->hist, need, s->sel, s->scr8, tid);
    prefix |= ((unsigned)s->sel[0]) << sh;
    need = s->sel[1];
    __syncthreads();
  }
  unsigned kth = prefix;
  int m0 = K - need;
  int cpad = ((c + TB - 1) / TB) * TB;
  for (int i = tid; i < cpad; i += TB) {
    bool inb = i < c;
    unsigned long long e = inb ? s->cand[i] : 0ull;
    unsigned key = (unsigned)(e >> 32);
    bool strict = inb && key > kth;
    bool tie = inb && key == kth;
    int p = atomic_agg(&s->cnt_m, strict);
    if (strict) orow[p] = (int)(e & 0xFFFFFFFFull);
    int t = atomic_agg(&s->cnt_t, tie);
    if (tie && t < need) orow[m0 + t] = (int)(e & 0xFFFFFFFFull);
  }
}

template <int CS, int TB, int MAXV, int R, int KC, int MB>
__global__ void __launch_bounds__(TB, MB) gvr_kernel(const float* __restrict__ logits,
                                                     const int* __restrict__ pre_idx,
                                                     int* __restrict__ out, int npad, int n,
                                                     int K, int kC) {
  extern __shared__ __align__(16) unsigned char smraw[];
  GSmem<TB, R, KC>* s = (GSmem<TB, R, KC>*)smraw;
  constexpr int NA = MAXV > 0 ? MAXV : 1;
  constexpr int NWARP = TB / 32;
  int tid = threadIdx.x;
  int row = blockIdx.y;
  int rank = blockIdx.x;  // cluster rank: gridDim.x == CS
  const float* rowp = logits + (size_t)row * npad;
  const float4* rowp4 = (const float4*)rowp;
  int npad4 = npad >> 2;
  int chunk = (npad4 + CS - 1) / CS;
  int f0 = rank * chunk;
  int f1 = min(npad4, f0 + chunk);
  int* orow = out + (size_t)row * K;

  // Register preload first: the loads retire while the P1 hint gather runs.
  float4 a[NA];
  if constexpr (MAXV > 0) {
#pragma unroll
    for (int u = 0; u < MAXV; ++u) {
      int gi = f0 + tid + u * TB;
      a[u] = (gi < f1) ? rowp4[gi] : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
    }
  }

  // ---- P1: hint gather -> key-space min/max -> 64-bin CCDF -> rung ladder ----
  // The histogram runs in monotone f2u key space (log-like magnitude spacing),
  // which keeps quantile resolution near the hint minimum on heavy-tailed rows.
  const int* prow = pre_idx + (size_t)row * K;
  unsigned* hk = (unsigned*)s->cand;  // scratch (dead after the ladder is built)
  unsigned kmn = 0xFFFFFFFFu, kmx = 0u;
  for (int j = tid; j < K; j += TB) {
    int ix = prow[j];
    ix = max(0, min(n - 1, ix));
    unsigned key = f2u(__ldg(rowp + ix));
    hk[j] = key;
    kmn = min(kmn, key);
    kmx = max(kmx, key);
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    kmn = min(kmn, __shfl_down_sync(0xFFFFFFFFu, kmn, o));
    kmx = max(kmx, __shfl_down_sync(0xFFFFFFFFu, kmx, o));
  }
  {
    int lane = tid & 31, wid = tid >> 5;
    if (lane == 0) {
      ((unsigned*)s->fwred)[wid] = kmn;
      ((unsigned*)s->fwred)[NWARP + wid] = kmx;
    }
  }
  if (tid < 64) s->hist[tid] = 0;
  if (tid == 0) {
    s->cnt_m = 0;
    s->cnt_t = 0;
    s->ccnt = 0;
  }
  __syncthreads();
  if (tid == 0) {
    unsigned mn = 0xFFFFFFFFu, mx = 0u;
    for (int w = 0; w < NWARP; ++w) {
      mn = min(mn, ((unsigned*)s->fwred)[w]);
      mx = max(mx, ((unsigned*)s->fwred)[NWARP + w]);
    }
    ((unsigned*)s->hminmax)[0] = mn;
    ((unsigned*)s->hminmax)[1] = mx;
  }
  __syncthreads();
  unsigned hkmin = ((unsigned*)s->hminmax)[0], hkmax = ((unsigned*)s->hminmax)[1];
  unsigned krange = hkmax - hkmin + 1;
  float kscale = 64.f / (float)krange;
  for (int j = tid; j < K; j += TB) {
    int bb = min(63, (int)((float)(hk[j] - hkmin) * kscale));
    atomicAdd(&s->hist[bb], 1);
  }
  __syncthreads();
  if (tid == 0) {
    auto u2f = [](unsigned x) {
      return __uint_as_float(x ^ ((x & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu));
    };
    float hmin = u2f(hkmin);
    // key at the lower edge of the bin where the hint CCDF first reaches `target`
    auto tq = [&](float target) {
      int acc = 0;
      for (int j = 63; j >= 0; --j) {
        acc += s->hist[j];
        if ((float)acc >= target)
          return u2f(hkmin + (unsigned)((float)j * (float)krange * (1.f / 64.f)));
      }
      return hmin;
    };
    float t25 = tq(0.25f * (float)K);
    float lam = (t25 - hmin) * 0.7213475f;  // 1/ln(4)
    if (!(lam > 0.f)) lam = fmaxf(0.05f * fabsf(hmin), 1e-2f);
    s->lam = lam;
    // Fast-path guess: aim mid-window assuming the hint threshold undershoots.
    float tgt = sqrtf((float)K * (float)kC);
    s->t0 = fmaxf(hmin - lam * (logf(tgt / (float)K) + 0.30f), TMIN);
    // Undershoot-biased ladder: most rungs below hmin.
    float rg[R];
    if (R >= 8) {
      rg[0] = tq(0.60f * (float)K);
      rg[1] = hmin;
      rg[2] = hmin - 0.60f * lam;
      rg[3] = hmin - 1.20f * lam;
      rg[4] = hmin - 2.00f * lam;
      rg[5] = hmin - 3.00f * lam;
      rg[6] = hmin - 4.20f * lam;
      rg[7] = hmin - 6.00f * lam;
    } else {
      rg[0] = hmin;
      rg[1] = hmin - 1.00f * lam;
      rg[2] = hmin - 2.20f * lam;
      rg[3] = hmin - 4.00f * lam;
    }
    float prev = FLT_MAX;
#pragma unroll
    for (int r = 0; r < R; ++r) {
      float t = fminf(fmaxf(rg[r], TMIN), prev);
      s->rungs[r] = t;
      prev = t;
    }
  }
  if (tid < R) s->rcnt_local[tid] = 0;
  __syncthreads();

  // ---- Fast path (streaming tiers): fused guess-verify-collect ----
  // One guessed threshold from the hint CCDF; candidates land directly in CTA0
  // via warp-aggregated ticket atomics. If the collected count verifies inside
  // [K, kC] the count pass is never needed; otherwise fall back to the ladder.
  if constexpr (MAXV == 0) {
    float t0 = s->t0;
    unsigned long long* dst;
    int* pc;
    if constexpr (CS == 1) {
      dst = s->cand;
      pc = &s->ccnt;
    } else {
      GSmem<TB, R, KC>* s0 = cg::this_cluster().map_shared_rank(s, 0);
      dst = s0->cand;
      pc = &s0->ccnt;
      cg::this_cluster().sync();  // CTA0's ccnt=0 visible before remote atomics
    }
    int span = f1 - f0;
    int f1r = f0 + ((span + TB - 1) / TB) * TB;
    for (int i = f0 + tid; i < f1r; i += TB) {
      bool inb = i < f1;
      float4 v = inb ? rowp4[i] : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
      int e0 = 4 * i;
#pragma unroll
      for (int cmp = 0; cmp < 4; ++cmp) {
        float vv = cmp == 0 ? v.x : (cmp == 1 ? v.y : (cmp == 2 ? v.z : v.w));
        bool act = vv >= t0;
        int pos = atomic_agg(pc, act);
        if (act && pos < KC)
          dst[pos] = ((unsigned long long)f2u(vv) << 32) | (unsigned)(e0 + cmp);
      }
    }
    int c0;
    if constexpr (CS > 1) {
      cg::this_cluster().sync();
      c0 = cg::this_cluster().map_shared_rank(s, 0)->ccnt;
    } else {
      __syncthreads();
      c0 = s->ccnt;
    }
    if (c0 >= K && c0 <= kC) {
      if constexpr (CS > 1) {
        if (rank != 0) return;
      }
      gvr_refine<TB, R, KC>(s, c0, K, orow, tid);
      return;
    }
  }

  // ---- P2: ladder count + selection + log-secant refine ----
  int par = 0, fpar = 0;
  count_pass<R, CS, TB, R, KC, MAXV, NA>(s, a, rowp4, f0, f1, tid);
  exchange_counts<CS, TB, R, KC>(s, par, rank, tid, R);
  par ^= 1;

  float tsel = 0.f;
  int csel = -1, ptrow = 0, cpre = 0;
  float t_hi = 0.f, t_lo = 0.f;
  int c_hi = -1, c_lo = -1;
  {
    int rs = -1;
    for (int r = 0; r < R; ++r)
      if (s->rcnt[r] >= K) {
        rs = r;
        break;
      }
    if (rs >= 0 && s->rcnt[rs] <= kC) {
      tsel = s->rungs[rs];
      csel = s->rcnt[rs];
      ptrow = rs;
      cpre = s->rpre[rs];
    } else if (rs < 0) {
      t_hi = s->rungs[R - 1];
      c_hi = s->rcnt[R - 1];
    } else {
      t_lo = s->rungs[rs];
      c_lo = s->rcnt[rs];
      if (rs > 0) {
        t_hi = s->rungs[rs - 1];
        c_hi = s->rcnt[rs - 1];
      }
    }
  }

  float lam = s->lam;
  float target = sqrtf((float)K * (float)kC);
  for (int pass = 0; pass < MAXPASS && csel < 0; ++pass) {
    float tn;
    if (c_hi >= 0 && c_lo >= 0) {
      float lh = logf((float)max(c_hi, 1));
      float ll = logf((float)c_lo);
      float fr = (logf(target) - lh) / fmaxf(ll - lh, 1e-6f);
      tn = t_hi + (t_lo - t_hi) * fr;
      if (!(tn < t_hi && tn > t_lo)) tn = 0.5f * (t_hi + t_lo);
    } else if (c_hi >= 0) {
      tn = t_hi - lam * logf(target / fmaxf((float)c_hi, 1.f));
      lam *= 2.f;
    } else {
      tn = t_lo + lam * logf((float)c_lo / target);
      lam *= 2.f;
    }
    tn = fmaxf(tn, TMIN);
    __syncthreads();
    if (tid == 0) {
      s->rungs[0] = tn;
      s->rcnt_local[0] = 0;
    }
    __syncthreads();
    count_pass<1, CS, TB, R, KC, MAXV, NA>(s, a, rowp4, f0, f1, tid);
    exchange_counts<CS, TB, R, KC>(s, par, rank, tid, 1);
    par ^= 1;
    int cn = s->rcnt[0];
    if (cn >= K && cn <= kC) {
      tsel = tn;
      csel = cn;
      ptrow = 0;
      cpre = s->rpre[0];
    } else if (cn < K) {
      t_hi = tn;
      c_hi = cn;
    } else {
      t_lo = tn;
      c_lo = cn;
    }
  }

  // ---- Plateau fallback: max-below descent ----
  if (csel < 0) {
    if (c_hi < 0) {
      t_hi = FLT_MAX;
      c_hi = 0;
    }
    for (;;) {
      float w = max_below<CS, TB, R, KC, MAXV, NA>(s, a, rowp4, f0, f1, t_hi, fpar, tid);
      fpar ^= 1;
      w = fmaxf(w, TMIN);
      __syncthreads();
      if (tid == 0) {
        s->rungs[0] = w;
        s->rcnt_local[0] = 0;
      }
      __syncthreads();
      count_pass<1, CS, TB, R, KC, MAXV, NA>(s, a, rowp4, f0, f1, tid);
      exchange_counts<CS, TB, R, KC>(s, par, rank, tid, 1);
      par ^= 1;
      int cw = s->rcnt[0];
      if (cw < K) {
        t_hi = w;
        c_hi = cw;
        continue;
      }
      if (cw <= kC) {
        tsel = w;
        csel = cw;
        ptrow = 0;
        cpre = s->rpre[0];
        break;
      }
      plateau_emit<CS, TB, R, KC, MAXV, NA>(s, a, rowp4, f0, f1, t_hi, w, c_hi, K, n, orow, tid);
      return;
    }
  }

  // ---- P3: collect candidates >= tsel into CTA0 ----
  collect<CS, TB, R, KC, MAXV, NA>(s, a, rowp4, f0, f1, tsel, ptrow, cpre, tid);
  int c = csel;
  if constexpr (CS > 1) {
    cg::this_cluster().sync();
    if (rank != 0) return;
  } else {
    __syncthreads();
  }

  // ---- P4: exact radix refine + tie-ticket writeback (CTA0 only) ----
  gvr_refine<TB, R, KC>(s, c, K, orow, tid);
}

// ---------------------------------------------------------------------------
// Launch layer
// ---------------------------------------------------------------------------
template <int TB>
static void launch_direct(const float* logits, int* out, int b, int npad, int n, int K,
                          cudaStream_t stream) {
  static bool inited = false;
  if (!inited) {
    cudaFuncSetAttribute(direct_kernel<TB>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         12288 * 4);
    inited = true;
  }
  direct_kernel<TB><<<b, TB, npad * 4, stream>>>(logits, out, npad, n, K);
}

template <int CS, int TB, int MAXV, int R, int KC, int MB>
static void launch_gvr(const float* logits, const int* pre_idx, int* out, int b, int npad, int n,
                       int K, int kC, cudaStream_t stream) {
  using SM = GSmem<TB, R, KC>;
  auto kfn = gvr_kernel<CS, TB, MAXV, R, KC, MB>;
  static bool inited = false;
  if (!inited) {
    cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sizeof(SM));
    if (CS > 8) cudaFuncSetAttribute(kfn, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    inited = true;
  }
  if constexpr (CS == 1) {
    kfn<<<dim3(1, b, 1), TB, sizeof(SM), stream>>>(logits, pre_idx, out, npad, n, K, kC);
  } else {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CS, b, 1);
    cfg.blockDim = dim3(TB, 1, 1);
    cfg.dynamicSmemBytes = sizeof(SM);
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CS;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, kfn, logits, pre_idx, out, npad, n, K, kC);
  }
}

static int pow2_floor(int x) {
  int p = 1;
  while (p * 2 <= x) p *= 2;
  return p;
}

extern "C" void topk_launch(const float* logits, const int* pre_idx, int* out, int b, int npad,
                            int n, int K, cudaStream_t stream) {
  if (npad <= 12288) {
    if (npad <= 4608)
      launch_direct<256>(logits, out, b, npad, n, K, stream);
    else
      launch_direct<512>(logits, out, b, npad, n, K, stream);
    return;
  }
  int kC = (K >= 2048) ? 8192 : 6144;
  int capin = 256 / b;
  if (capin < 1) capin = 1;
  int cap = pow2_floor(capin);
  if (cap > 16) cap = 16;
  int champ;
  if (npad < 16384)
    champ = 1;
  else if (npad < 32768)
    champ = 4;
  else if (npad <= 131072)
    champ = 8;
  else
    champ = 16;
  int cs = champ < cap ? champ : cap;
  int npad4 = npad >> 2;
  int chunk = (npad4 + cs - 1) / cs;
  int mv = (chunk + 511) / 512;  // TB = 512

  // Register-resident tiers (rounded up to the instantiated MAXV set), else streaming.
#define REG(CSV, MVV) \
  launch_gvr<CSV, 512, MVV, 8, 8192, 1>(logits, pre_idx, out, b, npad, n, K, kC, stream)
#define STRM(CSV, RV, KCV) \
  launch_gvr<CSV, 512, 0, RV, KCV, 2>(logits, pre_idx, out, b, npad, n, K, kC, stream)
  if (mv <= 8) {
    switch (cs) {
      case 1:
        if (mv <= 7)
          REG(1, 7);
        else
          REG(1, 8);
        return;
      case 2:
        REG(2, 8);
        return;
      case 4:
        if (mv <= 3)
          REG(4, 3);
        else
          REG(4, 4);
        return;
      case 8:
        if (mv <= 3)
          REG(8, 3);
        else if (mv <= 4)
          REG(8, 4);
        else if (mv <= 6)
          REG(8, 6);
        else
          REG(8, 8);
        return;
      default:
        if (mv <= 5)
          REG(16, 5);
        else if (mv <= 6)
          REG(16, 6);
        else
          REG(16, 8);
        return;
    }
  }
  bool bigk = (K >= 2048);
  switch (cs) {
    case 1:
      if (bigk)
        STRM(1, 4, 8192);
      else
        STRM(1, 4, 6144);
      return;
    case 2:
      if (bigk)
        STRM(2, 4, 8192);
      else
        STRM(2, 4, 6144);
      return;
    case 4:
      if (bigk)
        STRM(4, 4, 8192);
      else
        STRM(4, 4, 6144);
      return;
    case 8:
      if (bigk)
        STRM(8, 4, 8192);
      else
        STRM(8, 4, 6144);
      return;
    default:
      if (bigk)
        STRM(16, 8, 8192);
      else
        STRM(16, 8, 6144);
      return;
  }
#undef REG
#undef STRM
}
