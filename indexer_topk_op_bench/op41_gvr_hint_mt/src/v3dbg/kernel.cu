// GVR (guess-verify-refine) top-k for DeepSeek indexer decode, BS=1, fp32, B200 (sm_100a).
//
// Skeleton (kept from the production GVR kernel, re-expressed in CUDA C++):
//   P1: gather logits[pre_idx] (temporal warm hint), build a rung ladder of
//       candidate thresholds from the hint value CCDF.
//   P2: one multi-threshold count pass over the row measures the global CCDF at
//       all 8 rungs at once; if no rung lands in [K, kC], iterate a log-space
//       secant solve (exponential-CCDF model) one threshold per pass.
//   P3: collect all (value, index) pairs >= threshold into CTA0 shared memory
//       (cluster DSMEM pushes, offsets from cached per-thread counts).
//   P4: exact refine: bitwise radix select of the K-th key among candidates,
//       then a strict-greater + tie-ticket writeback (tie-robust exactness).
// A value-plateau fallback (max-below descent + direct global emit) keeps the
// kernel exact even when no threshold can land a count inside [K, kC].
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cfloat>
#include <cmath>

namespace cg = cooperative_groups;

constexpr int RUNGS = 8;        // thresholds measured per count pass
constexpr int KCMAX = 8192;     // candidate buffer capacity
constexpr int MAXPASS = 8;      // multi-rung refine shrinks the bracket ~9x per pass

template <int TB>
struct Smem {
  static constexpr int NWARP = TB / 32;
  unsigned long long cand[KCMAX];  // packed (key<<32 | index) candidates (CTA0 only)
  int ptcnt[RUNGS * TB];            // per-thread >= counts per rung (P3 offsets)
  int hist[256];                   // P4 radix histogram / P1 hint histogram (64 bins)
  float rungs[RUNGS];              // thresholds, descending
  int rcnt_local[RUNGS];           // this CTA's slice counts
  int rcnt[RUNGS];                 // cluster-wide counts
  int rpre[RUNGS];                 // exclusive prefix of peer CTA counts
  int ipartial[2][RUNGS];          // parity-banked DSMEM exchange slots
  float fpartial[2];               // parity slots for max-below exchange
  float fwred[2 * NWARP];          // warp reduce scratch (min/max)
  int iwred[NWARP];                // warp scan scratch
  float hminmax[2];
  int sel_above[2];                // P4: selected bin, count strictly above it
  int sel_count;                   // P4: population of the selected radix bin
  int cnt_m;                       // emit: strict-greater slot counter
  int cnt_t;                       // emit: tie ticket counter
};

__device__ __forceinline__ unsigned f2u(float f) {
  unsigned u = __float_as_uint(f);
  return u ^ ((u & 0x80000000u) ? 0xFFFFFFFFu : 0x80000000u);
}

// Count elements >= rungs[r] over this CTA's slice for R thresholds in one read.
// Caches per-thread counts in s->ptcnt and accumulates CTA totals in s->rcnt_local.
// Caller must zero s->rcnt_local and __syncthreads() before; call exchange after.
template <int TB, int R>
__device__ __forceinline__ void count_pass(Smem<TB>* s, const float4* base, int v0, int v1, int tid) {
  constexpr int T = TB;
  constexpr int U = (R > 1) ? 4 : 8;  // batched independent loads to cover DRAM/L2 latency
  float tr[R];
#pragma unroll
  for (int r = 0; r < R; ++r) tr[r] = s->rungs[r];
  int c[R];
#pragma unroll
  for (int r = 0; r < R; ++r) c[r] = 0;
  int i = v0 + tid;
  for (; i + (U - 1) * T < v1; i += U * T) {
    float4 a[U];
#pragma unroll
    for (int u = 0; u < U; ++u) a[u] = __ldg(base + i + u * T);
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int u = 0; u < U; ++u)
        c[r] += (int)(a[u].x >= tr[r]) + (int)(a[u].y >= tr[r]) + (int)(a[u].z >= tr[r]) +
                (int)(a[u].w >= tr[r]);
  }
  for (; i < v1; i += T) {
    float4 a = __ldg(base + i);
#pragma unroll
    for (int r = 0; r < R; ++r)
      c[r] += (int)(a.x >= tr[r]) + (int)(a.y >= tr[r]) + (int)(a.z >= tr[r]) + (int)(a.w >= tr[r]);
  }
#pragma unroll
  for (int r = 0; r < R; ++r) s->ptcnt[r * T + tid] = c[r];
}

// Publish CTA counts, cluster-sum them (parity-banked DSMEM), compute peer prefix.
// Barrier fusion: the warp reduce writes its destination (rcnt for CS==1,
// ipartial for CS>1) DIRECTLY; the following block/cluster sync provides the
// cross-warp visibility a separate post-reduction __syncthreads used to give,
// saving one barrier per count pass on the latency-bound hot path.
template <int TB, int CS, int R = RUNGS>
__device__ __forceinline__ void exchange_counts(Smem<TB>* s, int par, int tid, int rank) {
  __syncthreads();  // ptcnt final
  {
    // warp w reduces rung w's per-thread counts (R <= NWARP always)
    int lane = tid & 31, wid = tid >> 5;
    if (wid < R) {
      int sum = 0;
      for (int i = lane; i < TB; i += 32) sum += s->ptcnt[wid * TB + i];
#pragma unroll
      for (int o = 16; o; o >>= 1) sum += __shfl_down_sync(0xFFFFFFFFu, sum, o);
      if (lane == 0) {
        if constexpr (CS == 1) { s->rcnt[wid] = sum; s->rpre[wid] = 0; }
        else s->ipartial[par][wid] = sum;
      }
    }
  }
  if constexpr (CS == 1) {
    __syncthreads();
  } else {
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();  // superset of __syncthreads(): ipartial writes visible cluster-wide
    if (tid < R) {
      int tot = 0, pre = 0;
#pragma unroll
      for (int rr = 0; rr < CS; ++rr) {
        Smem<TB>* ps = (Smem<TB>*)cl.map_shared_rank(s, rr);
        int v = ps->ipartial[par][tid];
        tot += v;
        if (rr < rank) pre += v;
      }
      s->rcnt[tid] = tot;
      s->rpre[tid] = pre;
    }
    __syncthreads();
  }
}

// Largest value strictly below thi across the whole row (cluster-reduced).
template <int TB, int CS>
__device__ __forceinline__ float max_below_pass(Smem<TB>* s, const float4* base, int v0, int v1,
                                                float thi, int par, int tid) {
  constexpr int T = TB;
  constexpr int NWARP = TB / 32;
  float m = -FLT_MAX;
  int i = v0 + tid;
  for (; i + 3 * T < v1; i += 4 * T) {
    float4 a[4];
#pragma unroll
    for (int u = 0; u < 4; ++u) a[u] = __ldg(base + i + u * T);
#pragma unroll
    for (int u = 0; u < 4; ++u) {
      if (a[u].x < thi) m = fmaxf(m, a[u].x);
      if (a[u].y < thi) m = fmaxf(m, a[u].y);
      if (a[u].z < thi) m = fmaxf(m, a[u].z);
      if (a[u].w < thi) m = fmaxf(m, a[u].w);
    }
  }
  for (; i < v1; i += T) {
    float4 a = __ldg(base + i);
    if (a.x < thi) m = fmaxf(m, a.x);
    if (a.y < thi) m = fmaxf(m, a.y);
    if (a.z < thi) m = fmaxf(m, a.z);
    if (a.w < thi) m = fmaxf(m, a.w);
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
      Smem<TB>* ps = (Smem<TB>*)cl.map_shared_rank(s, rr);
      mm = fmaxf(mm, ps->fpartial[par]);
    }
    return mm;
  }
}

// P1: hint gather + stats + rung ladder from the hint-value CCDF.
template <int TB, int AR = RUNGS, int HS = 1>
__device__ __forceinline__ void phase1(Smem<TB>* s, const float* __restrict__ logits,
                                       const int* __restrict__ pre_idx, int K, int npad, int tid) {
  // HS > 1: sample every HS-th hint. The initial threshold ladder comes from
  // the sample's quantiles; the secant loop's below-bracket extension covers
  // the lost >=K floor guarantee (skeleton unchanged: hint guess -> secant).
  const int Ks = (K + HS - 1) / HS;   // sampled hint count
  constexpr int T = TB;
  constexpr int NWARP = TB / 32;
  if (tid < 64) s->hist[tid] = 0;
  float hv[4];
  bool hok[4];
  float mn = FLT_MAX, mx = -FLT_MAX;
#pragma unroll
  for (int jj = 0; jj < 4; ++jj) {
    int j = (tid + jj * T) * HS;
    hok[jj] = (j < K);
    if (hok[jj]) {
      int hidx = __ldg(pre_idx + j);
      hidx = max(0, min(hidx, npad - 1));  // safety clamp (hint indices are
                                           // valid in-range in production)
      float v = __ldg(logits + hidx);
      hv[jj] = v;
      mn = fminf(mn, v);
      mx = fmaxf(mx, v);
    }
  }
#pragma unroll
  for (int o = 16; o; o >>= 1) {
    mn = fminf(mn, __shfl_down_sync(0xFFFFFFFFu, mn, o));
    mx = fmaxf(mx, __shfl_down_sync(0xFFFFFFFFu, mx, o));
  }
  int lane = tid & 31, wid = tid >> 5;
  if (lane == 0) { s->fwred[wid] = mn; s->fwred[NWARP + wid] = mx; }
  __syncthreads();
  if (tid == 0) {
    float a = FLT_MAX, b = -FLT_MAX;
    for (int w = 0; w < NWARP; ++w) {
      a = fminf(a, s->fwred[w]);
      b = fmaxf(b, s->fwred[NWARP + w]);
    }
    s->hminmax[0] = a;
    s->hminmax[1] = b;
  }
  __syncthreads();
  float hmin = s->hminmax[0], hmax = s->hminmax[1];
  if (!(hmax - hmin > 0.0f)) {  // degenerate: all hint values equal
    if (tid < AR) s->rungs[tid] = hmin;
    __syncthreads();
    return;
  }
  // Stage 1: coarse hist over [hmin, hmax]; a single sunk outlier can stretch
  // this range so far that the bulk collapses into one bin, so we only use it
  // to find a trim point covering ~97% of the hints, then re-histogram the
  // trimmed range finely for the actual rung quantiles.
  float scale = 64.0f / (hmax - hmin);
#pragma unroll
  for (int jj = 0; jj < 4; ++jj)
    if (hok[jj]) {
      int b = (int)((hv[jj] - hmin) * scale);
      b = max(0, min(63, b));
      atomicAdd(&s->hist[b], 1);
    }
  __syncthreads();
  if (tid < 32) {
    // warp-parallel suffix scan over 64 bins (2 bins/lane, high bins = high lanes)
    int h0 = s->hist[62 - 2 * tid], h1 = s->hist[63 - 2 * tid];
    // zero the two bins now (read complete): the trailing barrier publishes both
    // hminmax/rungs AND the cleared histogram, so stage 2 skips its zeroing sync.
    s->hist[62 - 2 * tid] = 0;
    s->hist[63 - 2 * tid] = 0;
    int S = h0 + h1;
    int x = S;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int v = __shfl_up_sync(0xFFFFFFFFu, x, o);
      if (tid >= o) x += v;
    }
    // lane covers bins [62-2t, 63-2t]; x = count of hints in bins >= 62-2t
    int A = x - S;  // count strictly above this lane's chunk
    float binw = (hmax - hmin) * (1.0f / 64.0f);
    int qtrim = (Ks * 97) / 100;
    if (A < qtrim && x >= qtrim) {
      int b = (A + h1 >= qtrim) ? (63 - 2 * tid) : (62 - 2 * tid);
      s->hminmax[0] = hmin + binw * (float)b;  // trim point
    }
    if (tid == 31 && x < qtrim) s->hminmax[0] = hmin;  // all bins can't reach qtrim
    if (tid == 0) s->rungs[AR - 1] = hmin;  // HS==1: guaranteed count >= K globally
  }
  __syncthreads();
  float tlow = s->hminmax[0];
  if (!(hmax - tlow > 0.0f)) {
    if (tid < AR - 1) s->rungs[tid] = hmax;
    __syncthreads();
    return;
  }
  // Stage 2: fine hist over [tlow, hmax] (hist already zeroed above)
  float scale2 = 64.0f / (hmax - tlow);
#pragma unroll
  for (int jj = 0; jj < 4; ++jj)
    if (hok[jj] && hv[jj] >= tlow) {
      int b = (int)((hv[jj] - tlow) * scale2);
      b = max(0, min(63, b));
      atomicAdd(&s->hist[b], 1);
    }
  __syncthreads();
  if (tid < 32) {
    int h0 = s->hist[62 - 2 * tid], h1 = s->hist[63 - 2 * tid];
    int S = h0 + h1;
    int x = S;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int v = __shfl_up_sync(0xFFFFFFFFu, x, o);
      if (tid >= o) x += v;
    }
    int A = x - S;
    float binw2 = (hmax - tlow) * (1.0f / 64.0f);
    // Ladder: one rung extrapolated ABOVE hmax (low-overlap layers can need a
    // threshold above every hint value), quantile rungs below, hmin floor.
    int qt[RUNGS - 2];
    if constexpr (AR == 4) {
      qt[0] = (Ks * 25) / 100;
      qt[1] = (Ks * 65) / 100;
    } else if constexpr (AR == 6) {
      qt[0] = (Ks * 15) / 100;
      qt[1] = (Ks * 40) / 100;
      qt[2] = (Ks * 70) / 100;
      qt[3] = (Ks * 92) / 100;
    } else {
      qt[0] = (Ks * 10) / 100;
      qt[1] = (Ks * 25) / 100;
      qt[2] = (Ks * 45) / 100;
      qt[3] = (Ks * 65) / 100;
      qt[4] = (Ks * 82) / 100;
      qt[5] = (Ks * 94) / 100;
    }
    int tot = __shfl_sync(0xFFFFFFFFu, x, 31);  // total hints in stage-2 range
#pragma unroll
    for (int r = 0; r < AR - 2; ++r) {
      if (A < qt[r] && x >= qt[r]) {
        int b = (A + h1 >= qt[r]) ? (63 - 2 * tid) : (62 - 2 * tid);
        s->rungs[r + 1] = tlow + binw2 * (float)b;  // lower bin edge: CCDF >= target
      }
      if (tid == 31 && tot < qt[r]) s->rungs[r + 1] = tlow;  // target below trim range
    }
    if (tid == 0) s->rungs[0] = hmax + (hmax - tlow);
  }
  __syncthreads();
}

template <int CS, int TB, int AR = RUNGS, int HS = 1>
__global__ void __launch_bounds__(TB, 1) gvr_topk_kernel(
    const float* __restrict__ logits, const int* __restrict__ pre_idx,
    int* __restrict__ out, int npad, int K, int kC) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  // batched rows: one cluster per row along grid.y; BS=1 keeps row 0 (no-op)
  logits += (size_t)blockIdx.y * npad;
  pre_idx += (size_t)blockIdx.y * K;
  out += (size_t)blockIdx.y * K;
  constexpr int T = TB;
  constexpr int NWARP = TB / 32;
  Smem<TB>* s = reinterpret_cast<Smem<TB>*>(smem_raw);
  int tid = threadIdx.x;
  int rank = 0;
  if constexpr (CS > 1) rank = (int)cg::this_cluster().block_rank();

  // slice in 64-float units (npad is a multiple of 64; pad = -FLT_MAX)
  int units = npad >> 6;
  int u0 = (int)(((long long)units * rank) / CS);
  int u1 = (int)(((long long)units * (rank + 1)) / CS);
  const float4* base = reinterpret_cast<const float4*>(logits);
  int v0 = u0 << 4, v1 = u1 << 4;  // float4-index range of this CTA's slice

  // async L2 prefetch: pre_idx first (feeds the P1 gather chain), then the slice
  // (covers the cold-L2 first count pass while P1 runs)
  if (tid * 32 < K) asm volatile("prefetch.global.L2 [%0];" ::"l"(pre_idx + tid * 32));
  for (int i = (v0 >> 3) + tid; i < (v1 >> 3); i += T)
    asm volatile("prefetch.global.L2 [%0];" ::"l"(base + ((long long)i << 3)));

  if (tid < RUNGS) s->rcnt_local[tid] = 0;

  int xch = 0;
  float thr = 0.0f;
  int chosen = -1, C = 0, cbase = 0;
  int m_gt = -1;  // >= 0 --> plateau direct-emit mode

  if (npad <= kC) {
    // whole row fits the candidate buffer: threshold solve converges trivially
    if (tid == 0) s->rungs[0] = -FLT_MAX;
    __syncthreads();
    count_pass<TB, 1>(s, base, v0, v1, tid);
    exchange_counts<TB, CS>(s, xch & 1, tid, rank);
    xch++;
    thr = -FLT_MAX;
    chosen = 0;
    C = s->rcnt[0];
    cbase = s->rpre[0];
  } else {
    phase1<TB, AR, HS>(s, logits, pre_idx, K, npad, tid);
    count_pass<TB, AR>(s, base, v0, v1, tid);
    exchange_counts<TB, CS>(s, xch & 1, tid, rank);
    xch++;

    float t_lo = -FLT_MAX, t_hi = INFINITY;  // count(t_lo) > kC, count(t_hi) < K
    int c_lo = 0x7fffffff, c_hi = 0;
    float span0 = fmaxf(s->hminmax[1] - s->hminmax[0], 1e-3f);
    float lct = __logf(sqrtf((float)K * (float)kC));  // secant target (geometric mid)
    int Rcur = AR;
    int dbg_pass = 0;
    for (int pass = 0;; ++pass) {
      dbg_pass = pass;
      // rungs descend in t, so counts ascend; first rung with count >= K
      int j = 0;
      while (j < Rcur && s->rcnt[j] < K) ++j;
      if (j < Rcur && s->rcnt[j] <= kC) {
        chosen = j;
        thr = s->rungs[j];
        C = s->rcnt[j];
        cbase = s->rpre[j];
        break;
      }
      if (j < Rcur && s->rungs[j] >= t_lo) { t_lo = s->rungs[j]; c_lo = s->rcnt[j]; }
      if (j > 0 && s->rungs[j - 1] <= t_hi) { t_hi = s->rungs[j - 1]; c_hi = s->rcnt[j - 1]; }

      bool descend = (pass >= MAXPASS);
      float nr[AR];  // next pass's rung ladder (descending)
      if (!descend) {
        if (isinf(t_hi)) {
          // every measured count > kC: geometric ladder walking up
          float step = span0 * (float)(1 << (pass * 3 < 24 ? pass * 3 : 24));
#pragma unroll
          for (int r = 0; r < AR; ++r) nr[r] = t_lo + step * (float)(1 << (AR - 1 - r));
          if (isinf(nr[0])) descend = true;
        } else if (t_lo == -FLT_MAX) {
          float step = span0 * (float)(1 << (pass * 3 < 24 ? pass * 3 : 24));
#pragma unroll
          for (int r = 0; r < AR; ++r) nr[r] = t_hi - step * (float)(1 << r);
        } else {
          // secant anchor: log-CCDF is near-linear in t, so counts spaced
          // geometrically between c_hi and c_lo map to a LINEAR ladder in t.
          // 8 inner rungs shrink the bracket >= 9x per pass.
          float dt = (t_hi - t_lo) * (1.0f / (float)(AR + 1));
#pragma unroll
          for (int r = 0; r < AR; ++r) nr[r] = t_hi - dt * (float)(r + 1);
          if (!(nr[AR - 1] > t_lo && nr[0] < t_hi)) descend = true;  // bracket collapsed: plateau
        }
      }
      if (descend) break;

      __syncthreads();
      if (tid < AR) {
        s->rcnt_local[tid] = 0;
        s->rungs[tid] = nr[tid];
      }
      __syncthreads();
      count_pass<TB, AR>(s, base, v0, v1, tid);
      exchange_counts<TB, CS>(s, xch & 1, tid, rank);
      xch++;
      Rcur = AR;
    }

  #ifdef DBG_PASSES
  if (tid == 0) out[0] = dbg_pass * 10 + (chosen < 0 ? 1 : 0);
  return;
#endif
  if (chosen < 0) {
      // Exact descent: step t_hi down through actual data values until the
      // count lands in [K, kC] or a plateau is confirmed (then direct emit).
      while (true) {
        float vstar = max_below_pass<TB, CS>(s, base, v0, v1, t_hi, xch & 1, tid);
        xch++;
        __syncthreads();
        if (tid < RUNGS) s->rcnt_local[tid] = 0;
        if (tid == 0) s->rungs[0] = vstar;
        __syncthreads();
        count_pass<TB, 1>(s, base, v0, v1, tid);
        exchange_counts<TB, CS>(s, xch & 1, tid, rank);
        xch++;
        int c = s->rcnt[0];
        if (c >= K && c <= kC) {
          chosen = 0;
          thr = vstar;
          C = c;
          cbase = s->rpre[0];
          break;
        }
        if (c < K) {
          t_hi = vstar;
          c_hi = c;
          continue;
        }
        // c > kC and count(> vstar) = c_hi < K: vstar IS the k-th value
        thr = vstar;
        m_gt = c_hi;
        break;
      }
    }
  }

  if (m_gt >= 0) {
    // plateau direct emit: strict-greater slots [0, m_gt), tie tickets fill the rest
    Smem<TB>* s0 = s;
    if constexpr (CS > 1) {
      cg::cluster_group cl = cg::this_cluster();
      if (rank == 0 && tid == 0) { s->cnt_m = 0; s->cnt_t = 0; }
      cl.sync();
      s0 = (Smem<TB>*)cl.map_shared_rank(s, 0);
    } else {
      if (tid == 0) { s->cnt_m = 0; s->cnt_t = 0; }
      __syncthreads();
    }
    int nt = K - m_gt;
    for (int i = v0 + tid; i < v1; i += T) {
      float4 a = __ldg(base + i);
      int gi = i << 2;
      float vv[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
      for (int q = 0; q < 4; ++q) {
        if (vv[q] > thr) {
          out[atomicAdd(&s0->cnt_m, 1)] = gi + q;
        } else if (vv[q] == thr) {
          int p = atomicAdd(&s0->cnt_t, 1);
          if (p < nt) out[m_gt + p] = gi + q;
        }
      }
    }
    return;
  }

  // ---- P3: collect candidates >= thr into CTA0 smem ----
  int myc = s->ptcnt[chosen * T + tid];
  int lane = tid & 31, wid = tid >> 5;
  int incl = myc;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int v = __shfl_up_sync(0xFFFFFFFFu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) s->iwred[wid] = incl;
  __syncthreads();
  if (wid == 0) {
    int v = (lane < NWARP) ? s->iwred[lane] : 0;
    int iv = v;
#pragma unroll
    for (int o = 1; o < NWARP; o <<= 1) {
      int u = __shfl_up_sync(0xFFFFFFFFu, iv, o);
      if (lane >= o) iv += u;
    }
    if (lane < NWARP) s->iwred[lane] = iv - v;
  }
  __syncthreads();
  int pos = cbase + s->iwred[wid] + (incl - myc);

  unsigned long long* dst = s->cand;
  if constexpr (CS > 1)
    dst = (unsigned long long*)cg::this_cluster().map_shared_rank(s->cand, 0);
  {
    int i = v0 + tid;
    for (; i + 3 * T < v1; i += 4 * T) {
      float4 a[4];
#pragma unroll
      for (int u = 0; u < 4; ++u) a[u] = __ldg(base + i + u * T);
#pragma unroll
      for (int u = 0; u < 4; ++u) {
        int gi = (i + u * T) << 2;
        if (a[u].x >= thr) dst[pos++] = ((unsigned long long)__float_as_uint(a[u].x) << 32) | (unsigned)gi;
        if (a[u].y >= thr) dst[pos++] = ((unsigned long long)__float_as_uint(a[u].y) << 32) | (unsigned)(gi + 1);
        if (a[u].z >= thr) dst[pos++] = ((unsigned long long)__float_as_uint(a[u].z) << 32) | (unsigned)(gi + 2);
        if (a[u].w >= thr) dst[pos++] = ((unsigned long long)__float_as_uint(a[u].w) << 32) | (unsigned)(gi + 3);
      }
    }
    for (; i < v1; i += T) {
      float4 a = __ldg(base + i);
      int gi = i << 2;
      if (a.x >= thr) dst[pos++] = ((unsigned long long)__float_as_uint(a.x) << 32) | (unsigned)gi;
      if (a.y >= thr) dst[pos++] = ((unsigned long long)__float_as_uint(a.y) << 32) | (unsigned)(gi + 1);
      if (a.z >= thr) dst[pos++] = ((unsigned long long)__float_as_uint(a.z) << 32) | (unsigned)(gi + 2);
      if (a.w >= thr) dst[pos++] = ((unsigned long long)__float_as_uint(a.w) << 32) | (unsigned)(gi + 3);
    }
  }
  if constexpr (CS > 1) {
    cg::this_cluster().sync();
    if (rank != 0) return;
  } else {
    __syncthreads();
  }

  // ---- P4 (CTA0 solo): exact K-th key via 4x8-bit radix select, then emit ----
  if (C == K) {
    for (int i = tid; i < C; i += T) out[i] = (int)(unsigned)(s->cand[i] & 0xFFFFFFFFull);
    return;
  }
  unsigned pref = 0;
  int want = K, m = 0;
  int final_shift = 0;
  // hist[0..255] prezeroed at the P3->P4 boundary; each pass re-zeros in-register
  // from the select warp after reading -> one fewer __syncthreads per radix pass.
  if (tid < 256) s->hist[tid] = 0;
  __syncthreads();
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    int shift = 24 - 8 * r;
    for (int i = tid; i < C; i += T) {
      unsigned long long kv = s->cand[i];
      if (r == 0) {
        unsigned u = f2u(__uint_as_float((unsigned)(kv >> 32)));
        s->cand[i] = ((unsigned long long)u << 32) | (kv & 0xFFFFFFFFull);
        atomicAdd(&s->hist[u >> 24], 1);
      } else {
        unsigned u = (unsigned)(kv >> 32);
        if ((u >> (shift + 8)) == pref) atomicAdd(&s->hist[(u >> shift) & 0xFFu], 1);
      }
    }
    __syncthreads();
    if (tid < 32) {
      int b8 = tid * 8, S = 0, h[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        h[q] = s->hist[b8 + q];
        S += h[q];
      }
#pragma unroll
      for (int q = 0; q < 8; ++q) s->hist[b8 + q] = 0;  // clear for next pass
      int x = S;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_down_sync(0xFFFFFFFFu, x, o);
        if (tid + o < 32) x += v;
      }
      int A = x - S;  // count in bins strictly above this lane's chunk
      if (A < want && want <= A + S) {
        int run = A;
#pragma unroll
        for (int q = 7; q >= 0; --q) {
          if (run < want && want <= run + h[q]) {
            s->sel_above[0] = b8 + q;
            s->sel_above[1] = run;
            s->sel_count = h[q];
          }
          run += h[q];
        }
      }
    }
    __syncthreads();
    int sel = s->sel_above[0], above = s->sel_above[1];
    m += above;
    want -= above;
    pref = (pref << 8) | (unsigned)sel;
    if (s->sel_count == want) {
      final_shift = shift;
      break;
    }
  }
  unsigned kth = pref;

  if (tid == 0) { s->cnt_m = 0; s->cnt_t = 0; }
  __syncthreads();
  int nt = K - m;
  for (int i = tid; i < C; i += T) {
    unsigned long long kv = s->cand[i];
    unsigned u = (unsigned)(kv >> 32);
    if (final_shift) u >>= final_shift;
    int idx = (int)(unsigned)(kv & 0xFFFFFFFFull);
    if (u > kth) {
      out[atomicAdd(&s->cnt_m, 1)] = idx;
    } else if (u == kth) {
      int p = atomicAdd(&s->cnt_t, 1);
      if (p < nt) out[m + p] = idx;
    }
  }
}


// ---- Register-resident GVR (npad <= CS*TB*MAXV*4) --------------------------
// Same phases as gvr_topk_kernel, but each thread loads its <=MAXV float4s of
// the row into REGISTERS once, up front (the loads retire while the P1 hint
// gather runs). Every count pass, the max-below descent, the collect, and the
// plateau emit then run from registers: the row is read from memory exactly
// once, so refine iterations cost only ALU + the count exchange. Out-of-range
// lanes hold -FLT_MAX (same as the row pad), which can never pass a threshold.

template <int TB, int R, int MAXV>
__device__ __forceinline__ void count_reg(Smem<TB>* s, const float4 (&a)[MAXV], int tid) {
  float tr[R];
#pragma unroll
  for (int r = 0; r < R; ++r) tr[r] = s->rungs[r];
  int c[R];
#pragma unroll
  for (int r = 0; r < R; ++r) c[r] = 0;
#pragma unroll
  for (int u = 0; u < MAXV; ++u)
#pragma unroll
    for (int r = 0; r < R; ++r)
      c[r] += (int)(a[u].x >= tr[r]) + (int)(a[u].y >= tr[r]) + (int)(a[u].z >= tr[r]) +
              (int)(a[u].w >= tr[r]);
#pragma unroll
  for (int r = 0; r < R; ++r) s->ptcnt[r * TB + tid] = c[r];
}

template <int TB, int CS, int MAXV>
__device__ __forceinline__ float max_below_reg(Smem<TB>* s, const float4 (&a)[MAXV], float thi,
                                               int par, int tid) {
  constexpr int NWARP = TB / 32;
  float m = -FLT_MAX;
#pragma unroll
  for (int u = 0; u < MAXV; ++u) {
    if (a[u].x < thi) m = fmaxf(m, a[u].x);
    if (a[u].y < thi) m = fmaxf(m, a[u].y);
    if (a[u].z < thi) m = fmaxf(m, a[u].z);
    if (a[u].w < thi) m = fmaxf(m, a[u].w);
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
      Smem<TB>* ps = (Smem<TB>*)cl.map_shared_rank(s, rr);
      mm = fmaxf(mm, ps->fpartial[par]);
    }
    return mm;
  }
}

template <int CS, int TB, int MAXV, int AR = RUNGS, int HS = 1>
__global__ void __launch_bounds__(TB, 1) gvr_topk_reg(
    const float* __restrict__ logits, const int* __restrict__ pre_idx,
    int* __restrict__ out, int npad, int K, int kC) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  // batched rows: one cluster per row along grid.y; BS=1 keeps row 0 (no-op)
  logits += (size_t)blockIdx.y * npad;
  pre_idx += (size_t)blockIdx.y * K;
  out += (size_t)blockIdx.y * K;
  constexpr int T = TB;
  constexpr int NWARP = TB / 32;
  Smem<TB>* s = reinterpret_cast<Smem<TB>*>(smem_raw);
  int tid = threadIdx.x;
  int rank = 0;
  if constexpr (CS > 1) rank = (int)cg::this_cluster().block_rank();

  int V4 = npad >> 2;
  int vpc = (V4 + CS - 1) / CS;  // launcher guarantees vpc <= MAXV*TB
  int v0 = rank * vpc;
  int v1 = min(v0 + vpc, V4);
  const float4* base = reinterpret_cast<const float4*>(logits);

  if (tid * 32 < K) asm volatile("prefetch.global.L2 [%0];" ::"l"(pre_idx + tid * 32));

  // one-time row load; retires while P1's gather chain runs
  float4 a[MAXV];
#pragma unroll
  for (int u = 0; u < MAXV; ++u) {
    int i = v0 + tid + u * T;
    a[u] = (i < v1) ? __ldg(base + i) : make_float4(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
  }

  if (tid < RUNGS) s->rcnt_local[tid] = 0;

  int xch = 0;
  float thr = 0.0f;
  int chosen = -1, C = 0, cbase = 0;
  int m_gt = -1;  // >= 0 --> plateau direct-emit mode

  phase1<TB, AR, HS>(s, logits, pre_idx, K, npad, tid);
  count_reg<TB, AR, MAXV>(s, a, tid);
  exchange_counts<TB, CS, AR>(s, xch & 1, tid, rank);
  xch++;

  {
    float t_lo = -FLT_MAX, t_hi = INFINITY;  // count(t_lo) > kC, count(t_hi) < K
    int c_lo = 0x7fffffff, c_hi = 0;
    float span0 = fmaxf(s->hminmax[1] - s->hminmax[0], 1e-3f);
    int Rcur = AR;
    int dbg_pass = 0;
    for (int pass = 0;; ++pass) {
      dbg_pass = pass;
      int j = 0;
      while (j < Rcur && s->rcnt[j] < K) ++j;
      if (j < Rcur && s->rcnt[j] <= kC) {
        chosen = j;
        thr = s->rungs[j];
        C = s->rcnt[j];
        cbase = s->rpre[j];
        break;
      }
      if (j < Rcur && s->rungs[j] >= t_lo) { t_lo = s->rungs[j]; c_lo = s->rcnt[j]; }
      if (j > 0 && s->rungs[j - 1] <= t_hi) { t_hi = s->rungs[j - 1]; c_hi = s->rcnt[j - 1]; }

      bool descend = (pass >= MAXPASS);
      float nr[AR];  // next pass's rung ladder (descending)
      if (!descend) {
        if (isinf(t_hi)) {
          float step = span0 * (float)(1 << (pass * 3 < 24 ? pass * 3 : 24));
#pragma unroll
          for (int r = 0; r < AR; ++r) nr[r] = t_lo + step * (float)(1 << (AR - 1 - r));
          if (isinf(nr[0])) descend = true;
        } else if (t_lo == -FLT_MAX) {
          float step = span0 * (float)(1 << (pass * 3 < 24 ? pass * 3 : 24));
#pragma unroll
          for (int r = 0; r < AR; ++r) nr[r] = t_hi - step * (float)(1 << r);
        } else {
          float dt = (t_hi - t_lo) * (1.0f / (float)(AR + 1));
#pragma unroll
          for (int r = 0; r < AR; ++r) nr[r] = t_hi - dt * (float)(r + 1);
          if (!(nr[AR - 1] > t_lo && nr[0] < t_hi)) descend = true;
        }
      }
      if (descend) break;

      __syncthreads();
      if (tid < AR) {
        s->rcnt_local[tid] = 0;
        s->rungs[tid] = nr[tid];
      }
      __syncthreads();
      count_reg<TB, AR, MAXV>(s, a, tid);
      exchange_counts<TB, CS, AR>(s, xch & 1, tid, rank);
      xch++;
      Rcur = AR;
    }

  #ifdef DBG_PASSES
  if (tid == 0) out[0] = dbg_pass * 10 + (chosen < 0 ? 1 : 0);
  return;
#endif
  if (chosen < 0) {
      while (true) {
        float vstar = max_below_reg<TB, CS, MAXV>(s, a, t_hi, xch & 1, tid);
        xch++;
        __syncthreads();
        if (tid < RUNGS) s->rcnt_local[tid] = 0;
        if (tid == 0) s->rungs[0] = vstar;
        __syncthreads();
        count_reg<TB, 1, MAXV>(s, a, tid);
        exchange_counts<TB, CS>(s, xch & 1, tid, rank);
        xch++;
        int c = s->rcnt[0];
        if (c >= K && c <= kC) {
          chosen = 0;
          thr = vstar;
          C = c;
          cbase = s->rpre[0];
          break;
        }
        if (c < K) {
          t_hi = vstar;
          c_hi = c;
          continue;
        }
        thr = vstar;
        m_gt = c_hi;
        break;
      }
    }
  }

  if (m_gt >= 0) {
    // plateau direct emit from registers
    Smem<TB>* s0 = s;
    if constexpr (CS > 1) {
      cg::cluster_group cl = cg::this_cluster();
      if (rank == 0 && tid == 0) { s->cnt_m = 0; s->cnt_t = 0; }
      cl.sync();
      s0 = (Smem<TB>*)cl.map_shared_rank(s, 0);
    } else {
      if (tid == 0) { s->cnt_m = 0; s->cnt_t = 0; }
      __syncthreads();
    }
    int nt = K - m_gt;
#pragma unroll
    for (int u = 0; u < MAXV; ++u) {
      int i = v0 + tid + u * T;
      if (i < v1) {
        int gi = i << 2;
        float vv[4] = {a[u].x, a[u].y, a[u].z, a[u].w};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
          if (vv[q] > thr) {
            out[atomicAdd(&s0->cnt_m, 1)] = gi + q;
          } else if (vv[q] == thr) {
            int p = atomicAdd(&s0->cnt_t, 1);
            if (p < nt) out[m_gt + p] = gi + q;
          }
        }
      }
    }
    return;
  }

  // ---- P3: collect candidates >= thr into CTA0 smem, from registers ----
  int myc = s->ptcnt[chosen * T + tid];
  int lane = tid & 31, wid = tid >> 5;
  int incl = myc;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int v = __shfl_up_sync(0xFFFFFFFFu, incl, o);
    if (lane >= o) incl += v;
  }
  if (lane == 31) s->iwred[wid] = incl;
  __syncthreads();
  if (wid == 0) {
    int v = (lane < NWARP) ? s->iwred[lane] : 0;
    int iv = v;
#pragma unroll
    for (int o = 1; o < NWARP; o <<= 1) {
      int u = __shfl_up_sync(0xFFFFFFFFu, iv, o);
      if (lane >= o) iv += u;
    }
    if (lane < NWARP) s->iwred[lane] = iv - v;
  }
  __syncthreads();
  int pos = cbase + s->iwred[wid] + (incl - myc);

  unsigned long long* dst = s->cand;
  if constexpr (CS > 1)
    dst = (unsigned long long*)cg::this_cluster().map_shared_rank(s->cand, 0);
#pragma unroll
  for (int u = 0; u < MAXV; ++u) {
    // thr > -FLT_MAX whenever we get here, so pad/dummy lanes can never pass
    int gi = (v0 + tid + u * T) << 2;
    if (a[u].x >= thr) dst[pos++] = ((unsigned long long)__float_as_uint(a[u].x) << 32) | (unsigned)gi;
    if (a[u].y >= thr) dst[pos++] = ((unsigned long long)__float_as_uint(a[u].y) << 32) | (unsigned)(gi + 1);
    if (a[u].z >= thr) dst[pos++] = ((unsigned long long)__float_as_uint(a[u].z) << 32) | (unsigned)(gi + 2);
    if (a[u].w >= thr) dst[pos++] = ((unsigned long long)__float_as_uint(a[u].w) << 32) | (unsigned)(gi + 3);
  }
  // prezero P4's radix histogram now; the P3-terminal barrier below publishes it
  // (hist is dead between phase1 and here), so the P4 loop skips its r=0 zeroing.
  if (tid < 256) s->hist[tid] = 0;
  if constexpr (CS > 1) {
    cg::this_cluster().sync();
    if (rank != 0) return;
  } else {
    __syncthreads();
  }

  // ---- P4 (CTA0 solo): exact K-th key via 4x8-bit radix select, then emit ----
  if (C == K) {
    for (int i = tid; i < C; i += T) out[i] = (int)(unsigned)(s->cand[i] & 0xFFFFFFFFull);
    return;
  }
  unsigned pref = 0;
  int want = K, m = 0;
  int final_shift = 0;
  // hist[0..255] is zeroed on entry (folded into the P3->P4 barrier below), and
  // each radix pass re-zeros it in-register from the select warp AFTER reading
  // it, so the trailing select barrier both selects and publishes the zeros for
  // the next pass -> one fewer __syncthreads per radix pass.
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    int shift = 24 - 8 * r;
    for (int i = tid; i < C; i += T) {
      unsigned long long kv = s->cand[i];
      if (r == 0) {
        unsigned u = f2u(__uint_as_float((unsigned)(kv >> 32)));
        s->cand[i] = ((unsigned long long)u << 32) | (kv & 0xFFFFFFFFull);
        atomicAdd(&s->hist[u >> 24], 1);
      } else {
        unsigned u = (unsigned)(kv >> 32);
        if ((u >> (shift + 8)) == pref) atomicAdd(&s->hist[(u >> shift) & 0xFFu], 1);
      }
    }
    __syncthreads();
    if (tid < 32) {
      int b8 = tid * 8, S = 0, h[8];
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        h[q] = s->hist[b8 + q];
        S += h[q];
      }
      // zero this lane's 8 bins now (read complete): the trailing barrier
      // publishes the cleared histogram to the next pass's atomicAdd.
#pragma unroll
      for (int q = 0; q < 8; ++q) s->hist[b8 + q] = 0;
      int x = S;
#pragma unroll
      for (int o = 1; o < 32; o <<= 1) {
        int v = __shfl_down_sync(0xFFFFFFFFu, x, o);
        if (tid + o < 32) x += v;
      }
      int A = x - S;
      if (A < want && want <= A + S) {
        int run = A;
#pragma unroll
        for (int q = 7; q >= 0; --q) {
          if (run < want && want <= run + h[q]) {
            s->sel_above[0] = b8 + q;
            s->sel_above[1] = run;
            s->sel_count = h[q];
          }
          run += h[q];
        }
      }
    }
    __syncthreads();
    int sel = s->sel_above[0], above = s->sel_above[1];
    m += above;
    want -= above;
    pref = (pref << 8) | (unsigned)sel;
    if (s->sel_count == want) {
      final_shift = shift;
      break;
    }
  }
  unsigned kth = pref;

  if (tid == 0) { s->cnt_m = 0; s->cnt_t = 0; }
  __syncthreads();
  int nt = K - m;
  for (int i = tid; i < C; i += T) {
    unsigned long long kv = s->cand[i];
    unsigned u = (unsigned)(kv >> 32);
    if (final_shift) u >>= final_shift;
    int idx = (int)(unsigned)(kv & 0xFFFFFFFFull);
    if (u > kth) {
      out[atomicAdd(&s->cnt_m, 1)] = idx;
    } else if (u == kth) {
      int p = atomicAdd(&s->cnt_t, 1);
      if (p < nt) out[m + p] = idx;
    }
  }
}

// ---- Direct exact path (npad <= DKCMAX): the GVR threshold solve converges
// trivially at t = -inf (count(-inf) = npad known analytically), so the kernel
// goes straight to exact collect + 11/11/10 radix refine. ----
constexpr int DCAP_RUNGS = 16;   // sizes the DSmem side-buffer union
constexpr int DKCMAX = 12288;    // direct-path candidate capacity
template <int TB>
struct DSmem {
  static constexpr int NWARP = TB / 32;
  static constexpr int SIDECAP = DCAP_RUNGS * TB / 2;
  unsigned long long cand[DKCMAX];  // packed (key<<32 | index) candidates (CTA0 only)
  union {
    int ptcnt[DCAP_RUNGS * TB];            // per-thread >= counts per rung (P3 offsets)
    unsigned long long side[SIDECAP]; // P4 boundary-bin compaction (ptcnt is dead by P4)
  };
  int hist[2048];                  // P4 radix histogram / P1 hint histogram (64 bins)
  float rungs[DCAP_RUNGS];              // thresholds, descending
  int rcnt_local[DCAP_RUNGS];           // this CTA's slice counts
  int rcnt[DCAP_RUNGS];                 // cluster-wide counts
  int rpre[DCAP_RUNGS];                 // exclusive prefix of peer CTA counts
  int ipartial[2][DCAP_RUNGS];          // parity-banked DSMEM exchange slots
  float fpartial[2];               // parity slots for max-below exchange
  float fwred[2 * NWARP];          // warp reduce scratch (min/max)
  int iwred[NWARP];                // warp scan scratch
  float hminmax[2];
  int sel_above[2];                // P4: selected bin, count strictly above it
  int cnt_m;                       // emit: strict-greater slot counter
  int cnt_t;                       // emit: tie ticket counter
  int cnt_side;                    // P4: side-buffer compaction counter
};
// ---- P4 helpers -----------------------------------------------------------

// Find bin b (0..NB-1) such that above(b) < want <= above(b) + hist[b], where
// above(b) = sum_{j>b} hist[j]. Writes s->sel_above = {b, above(b)}.
// Trailing __syncthreads.
template <int TB, int NB>
__device__ __forceinline__ void bin_select(DSmem<TB>* s, int want, int tid) {
  constexpr int PER = NB / TB;
  constexpr int NWARP = TB / 32;
  int h[PER];
  int S = 0;
#pragma unroll
  for (int q = 0; q < PER; ++q) {
    h[q] = s->hist[tid * PER + q];
    S += h[q];
  }
  int lane = tid & 31, wid = tid >> 5;
  // warp suffix-inclusive over lanes (higher lane = higher bins)
  int x = S;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int v = __shfl_down_sync(0xFFFFFFFFu, x, o);
    if (lane + o < 32) x += v;
  }
  if (lane == 0) s->iwred[wid] = x;  // warp total
  __syncthreads();
  if (tid < 32) {
    int wt = (tid < NWARP) ? s->iwred[tid] : 0;
    int wx = wt;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      int v = __shfl_down_sync(0xFFFFFFFFu, wx, o);
      if (tid + o < 32) wx += v;
    }
    if (tid < NWARP) s->iwred[tid] = wx - wt;  // strictly-above-warp sum
  }
  __syncthreads();
  int A = s->iwred[wid] + (x - S);  // count in bins strictly above this chunk
  if (A < want && want <= A + S) {
    int run = A;
#pragma unroll
    for (int q = PER - 1; q >= 0; --q) {
      if (run < want && want <= run + h[q]) {
        s->sel_above[0] = tid * PER + q;
        s->sel_above[1] = run;
      }
      run += h[q];
    }
  }
  __syncthreads();
}

// Emit indices of all candidates with (key >> shift) >= P; the caller
// guarantees exactly K of them. Warp-aggregated slot allocation.
template <int TB>
__device__ __forceinline__ void emit_prefix_ge(DSmem<TB>* s, int C, int shift, unsigned P,
                                               int* __restrict__ out, int tid) {
  if (tid == 0) s->cnt_m = 0;
  __syncthreads();
  int Cpad = (C + 31) & ~31;
  for (int i = tid; i < Cpad; i += TB) {
    bool v = i < C;
    unsigned long long kv = v ? s->cand[i] : 0ull;
    unsigned u = (unsigned)(kv >> 32);
    bool e = v && ((u >> shift) >= P);
    unsigned bal = __ballot_sync(0xFFFFFFFFu, e);
    if (e) {
      int lane = tid & 31;
      int leader = __ffs(bal) - 1;
      int base = 0;
      if (lane == leader) base = atomicAdd(&s->cnt_m, __popc(bal));
      base = __shfl_sync(bal, base, leader);
      out[base + __popc(bal & ((1u << lane) - 1u))] = (int)(unsigned)(kv & 0xFFFFFFFFull);
    }
  }
}

// Final emit: keys > kth are mandatory (slots [0, m)), keys == kth fill the
// remaining nt tie tickets (slots [m, m+nt)). Warp-aggregated.
template <int TB>
__device__ __forceinline__ void emit_final(DSmem<TB>* s, int C, unsigned kth, int m, int nt,
                                           int* __restrict__ out, int tid) {
  if (tid == 0) { s->cnt_m = 0; s->cnt_t = 0; }
  __syncthreads();
  int Cpad = (C + 31) & ~31;
  for (int i = tid; i < Cpad; i += TB) {
    bool v = i < C;
    unsigned long long kv = v ? s->cand[i] : 0ull;
    unsigned u = (unsigned)(kv >> 32);
    int idx = (int)(unsigned)(kv & 0xFFFFFFFFull);
    bool man = v && (u > kth);
    bool tie = v && (u == kth);
    unsigned bm = __ballot_sync(0xFFFFFFFFu, man);
    unsigned bt = __ballot_sync(0xFFFFFFFFu, tie);
    int lane = tid & 31;
    if (man) {
      int leader = __ffs(bm) - 1;
      int base = 0;
      if (lane == leader) base = atomicAdd(&s->cnt_m, __popc(bm));
      base = __shfl_sync(bm, base, leader);
      out[base + __popc(bm & ((1u << lane) - 1u))] = idx;
    }
    if (tie) {
      int leader = __ffs(bt) - 1;
      int base = 0;
      if (lane == leader) base = atomicAdd(&s->cnt_t, __popc(bt));
      base = __shfl_sync(bt, base, leader);
      int p = base + __popc(bt & ((1u << lane) - 1u));
      if (p < nt) out[m + p] = idx;
    }
  }
}

// P4: exact top-K among s->cand[0..C) (keys already f2u-transformed).
// s->hist[0..2047] must hold the histogram of (key >> 21) on entry.
// 11/11/10-bit radix with whole-bin early exit; the boundary bin is compacted
// into s->side so later passes sweep only its members.
template <int TB>
__device__ __forceinline__ void radix_select_emit(DSmem<TB>* s, int C, int K,
                                                  int* __restrict__ out, int tid) {
  constexpr int SIDECAP = DSmem<TB>::SIDECAP;
  if (C == K) {
    for (int i = tid; i < C; i += TB) out[i] = (int)(unsigned)(s->cand[i] & 0xFFFFFFFFull);
    return;
  }
  int want = K, m = 0;
  // level 0: top 11 bits (hist prebuilt during collect)
  bin_select<TB, 2048>(s, want, tid);
  int b0 = s->sel_above[0], A0 = s->sel_above[1];
  int h0 = s->hist[b0];
  if (want == A0 + h0) {  // k-th boundary == bin edge: whole bin admitted
    emit_prefix_ge<TB>(s, C, 21, (unsigned)b0, out, tid);
    return;
  }
  m += A0;
  want -= A0;
  bool docompact = (h0 <= SIDECAP);
  __syncthreads();  // everyone done reading hist/sel before reuse
  if (tid == 0) s->cnt_side = 0;
  for (int b = tid; b < 2048; b += TB) s->hist[b] = 0;
  __syncthreads();
  // level 1 sweep: histogram mid 11 bits of boundary-bin members; compact them
  for (int i = tid; i < C; i += TB) {
    unsigned long long kv = s->cand[i];
    unsigned u = (unsigned)(kv >> 32);
    if ((u >> 21) == (unsigned)b0) {
      atomicAdd(&s->hist[(u >> 10) & 0x7FFu], 1);
      if (docompact) s->side[atomicAdd(&s->cnt_side, 1)] = kv;
    }
  }
  __syncthreads();
  bin_select<TB, 2048>(s, want, tid);
  int b1 = s->sel_above[0], A1 = s->sel_above[1];
  int h1 = s->hist[b1];
  unsigned P01 = ((unsigned)b0 << 11) | (unsigned)b1;
  if (want == A1 + h1) {
    emit_prefix_ge<TB>(s, C, 10, P01, out, tid);
    return;
  }
  m += A1;
  want -= A1;
  __syncthreads();
  for (int b = tid; b < 1024; b += TB) s->hist[b] = 0;
  __syncthreads();
  // level 2 sweep: low 10 bits
  if (docompact) {
    for (int i = tid; i < h0; i += TB) {
      unsigned long long kv = s->side[i];
      unsigned u = (unsigned)(kv >> 32);
      if (((u >> 10) & 0x7FFu) == (unsigned)b1) atomicAdd(&s->hist[u & 0x3FFu], 1);
    }
  } else {
    for (int i = tid; i < C; i += TB) {
      unsigned long long kv = s->cand[i];
      unsigned u = (unsigned)(kv >> 32);
      if ((u >> 10) == P01) atomicAdd(&s->hist[u & 0x3FFu], 1);
    }
  }
  __syncthreads();
  bin_select<TB, 1024>(s, want, tid);
  int b2 = s->sel_above[0], A2 = s->sel_above[1];
  m += A2;
  want -= A2;
  unsigned kth = (P01 << 10) | (unsigned)b2;
  emit_final<TB>(s, C, kth, m, want, out, tid);
}

// ---- Direct kernel (npad <= candidate capacity) -----------------------------
// The GVR threshold solve converges trivially: count(-inf) = npad is known
// analytically and npad <= kC, so the exact collect admits the whole row and
// P4's exact refine does the selection (same P3/P4 stages as the GVR path).

template <int TB>
__global__ void __launch_bounds__(TB, 1) direct_topk_kernel(
    const float* __restrict__ logits, int* __restrict__ out, int npad, int K) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  // batched rows: one CTA per row along grid.y; BS=1 keeps row 0 (no-op)
  logits += (size_t)blockIdx.y * npad;
  out += (size_t)blockIdx.y * K;
  DSmem<TB>* s = reinterpret_cast<DSmem<TB>*>(smem_raw);
  int tid = threadIdx.x;
  for (int b = tid; b < 2048; b += TB) s->hist[b] = 0;
  __syncthreads();
  const float4* base = reinterpret_cast<const float4*>(logits);
  int V = npad >> 2;
  for (int i = tid; i < V; i += TB) {
    float4 a = __ldg(base + i);
    int gi = i << 2;
    float vv[4] = {a.x, a.y, a.z, a.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      unsigned ux = f2u(vv[q]);
      s->cand[gi + q] = ((unsigned long long)ux << 32) | (unsigned)(gi + q);
      atomicAdd(&s->hist[ux >> 21], 1);
    }
  }
  __syncthreads();
  radix_select_emit<TB>(s, npad, K, out, tid);
}

template <int TB>
static void launch_direct(const float* logits, int* out, int npad, int K, int BS,
                          cudaStream_t stream) {
  static bool inited = false;
  if (!inited) {
    cudaFuncSetAttribute(direct_topk_kernel<TB>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)sizeof(DSmem<TB>));
    inited = true;
  }
  direct_topk_kernel<TB><<<dim3(1, BS), TB, sizeof(DSmem<TB>), stream>>>(logits, out, npad, K);
}

template <int CS, int TB, int AR = RUNGS, int HS = 1>
static void launch_gvr(const float* logits, const int* pre_idx, int* out, int npad, int K, int kC,
                       int BS, cudaStream_t stream) {
  static bool inited = false;
  if (!inited) {
    cudaFuncSetAttribute(gvr_topk_kernel<CS, TB, AR, HS>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)sizeof(Smem<TB>));
    if (CS > 8)
      cudaFuncSetAttribute(gvr_topk_kernel<CS, TB, AR, HS>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    inited = true;
  }
  if constexpr (CS == 1) {
    gvr_topk_kernel<1, TB, AR, HS><<<dim3(1, BS), TB, sizeof(Smem<TB>), stream>>>(
        logits, pre_idx, out, npad, K, kC);
  } else {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CS, BS, 1);
    cfg.blockDim = dim3(TB, 1, 1);
    cfg.dynamicSmemBytes = sizeof(Smem<TB>);
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CS;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, gvr_topk_kernel<CS, TB, AR, HS>, logits, pre_idx, out, npad, K, kC);
  }
}


template <int CS, int TB, int MAXV, int AR = RUNGS, int HS = 1>
static void launch_reg(const float* logits, const int* pre_idx, int* out, int npad, int K, int kC,
                       int BS, cudaStream_t stream) {
  static bool inited = false;
  if (!inited) {
    cudaFuncSetAttribute(gvr_topk_reg<CS, TB, MAXV, AR, HS>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)sizeof(Smem<TB>));
    if (CS > 8)
      cudaFuncSetAttribute(gvr_topk_reg<CS, TB, MAXV, AR, HS>,
                           cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    inited = true;
  }
  if constexpr (CS == 1) {
    gvr_topk_reg<1, TB, MAXV, AR, HS><<<dim3(1, BS), TB, sizeof(Smem<TB>), stream>>>(
        logits, pre_idx, out, npad, K, kC);
  } else {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CS, BS, 1);
    cfg.blockDim = dim3(TB, 1, 1);
    cfg.dynamicSmemBytes = sizeof(Smem<TB>);
    cfg.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = CS;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;
    cudaLaunchKernelEx(&cfg, gvr_topk_reg<CS, TB, MAXV, AR, HS>, logits, pre_idx, out, npad, K, kC);
  }
}

// BS>1 dispatch (op38): probe-measured per-(npad, K, BS) winners.
// Three regimes per tier: low BS = the BS=1-optimal register-resident ladder
// (launch amortizes, latency dominates), mid BS = single-cluster register
// path (fewer sync partners, more rows in flight), high BS = streaming
// re-scan kernel (low registers -> 2 CTAs/SM occupancy; per-wave working set
// is L2-resident so count passes are cheap) with hint sampling (HS) and a
// short AR4/AR6 ladder. Skeleton everywhere: preIdx-hinted threshold guess ->
// secant-log refinement -> exact refine/emit.
static void launch_bs(const float* logits, const int* pre_idx, int* out, int npad, int K, int kC,
                      int BS, cudaStream_t stream) {
  if (npad <= DKCMAX) {  // direct tier: batched radix CTAs; streaming from BS>=256
    if (BS < 256)
      launch_direct<1024>(logits, out, npad, K, BS, stream);
    else
      launch_gvr<1, 512, 4, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
    return;
  }
  if (npad < 16896) {  // CS1 register-resident holds to BS<=128
    if (BS <= 128)
      launch_reg<1, 512, 9>(logits, pre_idx, out, npad, K, kC, BS, stream);
    else
      launch_gvr<1, 512, 4, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
    return;
  }
  if (npad <= 49152) {
    if (BS <= 8) {
      launch_reg<8, 512, 3>(logits, pre_idx, out, npad, K, kC, BS, stream);
    } else if (BS <= 128) {
      if (K <= 512)
        launch_reg<1, 1024, 9, 4, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
      else
        launch_reg<1, 1024, 9, 6, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
    } else {
      if (K <= 512)
        launch_gvr<1, 512, 4, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
      else
        launch_gvr<1, 512, 6, 4>(logits, pre_idx, out, npad, K, kC, BS, stream);
    }
    return;
  }
  if (npad <= 98304) {
    if (BS <= 8) {
      // vpc = ceil(npad/4/8): MAXV4 only holds to npad 65536 (original tiers)
      if (npad <= 65536)
        launch_reg<8, 512, 4>(logits, pre_idx, out, npad, K, kC, BS, stream);
      else
        launch_reg<8, 512, 8>(logits, pre_idx, out, npad, K, kC, BS, stream);
    } else if (BS <= 16 && K <= 512) {
      // v3 confirm probe (65600 BS16, all-layer min 1.045): 4-CTA streaming
      launch_gvr<4, 1024>(logits, pre_idx, out, npad, K, kC, BS, stream);
    } else if (BS > 16 && BS <= 32 && K == 1024) {
      // v3 confirm probe (65600 BS32, all-layer min 1.045)
      launch_gvr<4, 1024>(logits, pre_idx, out, npad, K, kC, BS, stream);
    } else if (BS <= 64) {
      launch_gvr<2, 1024, 6, 4>(logits, pre_idx, out, npad, K, kC, BS, stream);
    } else if (BS <= 128) {
      // v3 confirm probe BS128 (all-layer min: flash 1.053 / pro 1.043 /
      // v32 1.059): TB1024 streaming with short AR ladder
      if (K <= 1024)
        launch_gvr<1, 1024, 4, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
      else
        launch_gvr<1, 1024, 6, 4>(logits, pre_idx, out, npad, K, kC, BS, stream);
    } else {
      launch_gvr<1, 512, 4, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
    }
    return;
  }
  if (npad <= 131072) {
    if (BS <= 4)
      launch_reg<8, 512, 8>(logits, pre_idx, out, npad, K, kC, BS, stream);
    else if (BS <= 32)
      launch_reg<4, 1024, 9, 6, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
    else if (BS <= 64)
      launch_gvr<2, 1024, 4, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
    else
      launch_gvr<1, 1024, 4, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
    return;
  }
  if (npad <= 163840) {
    if (BS <= 4) {
      if (K >= 2048)
        launch_reg<16, 512, 5, 6>(logits, pre_idx, out, npad, K, kC, BS, stream);
      else
        launch_reg<16, 512, 5>(logits, pre_idx, out, npad, K, kC, BS, stream);
    } else if (BS <= 16) {
      if (BS > 8 && npad <= 131136) {
        // v3 confirm probe (131136 BS16, all-layer min: flash 1.066 /
        // pro 1.133 / v32 1.145): 4-CTA cluster reg with AR6 ladder
        if (K <= 512)
          launch_reg<4, 1024, 9, 6, 1>(logits, pre_idx, out, npad, K, kC, BS, stream);
        else
          launch_reg<4, 1024, 9, 6, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
      } else {
        launch_reg<8, 512, 10, 6, 4>(logits, pre_idx, out, npad, K, kC, BS, stream);
      }
    } else if (BS <= 32) {
      // v3 confirm probe BS32 (131136: min 1.068-1.123; 163776 v32: min 1.047)
      if (npad <= 131136)
        launch_reg<4, 1024, 9, 6, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
      else
        launch_gvr<4, 1024>(logits, pre_idx, out, npad, K, kC, BS, stream);
    } else if (BS <= 64) {
      launch_gvr<2, 1024, 4, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
    } else {
      launch_gvr<1, 1024, 4, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
    }
    return;
  }
  if (npad <= 262144) {
    if (BS <= 4) {
      if (K == 2048)
        launch_reg<16, 512, 8>(logits, pre_idx, out, npad, K, kC, BS, stream);
      else
        launch_reg<16, 512, 8, 6>(logits, pre_idx, out, npad, K, kC, BS, stream);
    } else if (BS <= 16) {
      if (K <= 512)
        launch_reg<8, 1024, 8, 4, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
      else
        launch_reg<8, 1024, 8, 6, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
    } else if (BS <= 32) {
      // v3 confirm probe (262144 BS32, all-layer min: flash 1.094 / pro 1.212)
      launch_gvr<4, 1024>(logits, pre_idx, out, npad, K, kC, BS, stream);
    } else if (BS <= 64) {
      if (K <= 512)
        launch_gvr<2, 1024, 4, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
      else
        launch_gvr<2, 1024, 6, 4>(logits, pre_idx, out, npad, K, kC, BS, stream);
    } else if (BS > 256 && BS <= 512 && K <= 512) {
      // v3 confirm probe (flash 262144 BS512, all-layer min 1.046): 2-CTA
      launch_gvr<2, 1024, 4, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
    } else {
      if (K <= 512)
        launch_gvr<1, 1024, 4, 2>(logits, pre_idx, out, npad, K, kC, BS, stream);
      else
        launch_gvr<1, 1024, 4, 4>(logits, pre_idx, out, npad, K, kC, BS, stream);
    }
    return;
  }
  launch_gvr<16, 512>(logits, pre_idx, out, npad, K, kC, BS, stream);
}

void gvr_topk_launch(const float* logits, const int* pre_idx, int* out, int npad, int K, int BS,
                     cudaStream_t stream) {
  int kC = (K >= 2048) ? 8192 : 6144;
  if (BS > 1) return launch_bs(logits, pre_idx, out, npad, K, kC, BS, stream);
  // BS==1: the r3_v11 ladder below is untouched (865-cell verdict carries over).
  // MAXV is matched tightly to each tier: mostly-dummy register slots cost
  // real time (predicated loads + dead compares), so keep slots nearly full.
  // Batched rows ride grid.y (one cluster per row); BS=1 grid is identical to
  // the single-row original, so the BS=1 verdict carries over unchanged.
  if (npad <= DKCMAX)
    launch_direct<1024>(logits, out, npad, K, BS, stream);
  else if (npad < 16384)
    launch_reg<1, 512, 8>(logits, pre_idx, out, npad, K, kC, BS, stream);   // vpc <= 4095
  else if (npad < 32768)
    launch_reg<4, 512, 4>(logits, pre_idx, out, npad, K, kC, BS, stream);   // vpc <= 2048
  else if (npad <= 49152)
    launch_reg<8, 512, 3>(logits, pre_idx, out, npad, K, kC, BS, stream);   // vpc <= 1536
  else if (npad <= 65536)
    launch_reg<8, 512, 4>(logits, pre_idx, out, npad, K, kC, BS, stream);   // vpc <= 2048
  else if (npad <= 131072)
    launch_reg<8, 512, 8>(logits, pre_idx, out, npad, K, kC, BS, stream);   // vpc <= 4096
  else if (npad <= 163840)
    // AR6's shifted quantile ladder measured faster on every K=2048 cell of
    // this tier (-0.5 to -3.1us); K<=1024 regressed (+3us convergence misses).
    if (K >= 2048)
      launch_reg<16, 512, 5, 6>(logits, pre_idx, out, npad, K, kC, BS, stream);  // vpc <= 2560
    else
      launch_reg<16, 512, 5>(logits, pre_idx, out, npad, K, kC, BS, stream);
  else if (npad <= 262144)
    // AR6 measured faster for K=512 (-0.7 to -1.3us) and K=1024 (r2-validated)
    // at this tier; K=2048 unmeasured here, keep the denser AR8 ladder.
    if (K == 2048)
      launch_reg<16, 512, 8>(logits, pre_idx, out, npad, K, kC, BS, stream);  // vpc <= 4096
    else
      launch_reg<16, 512, 8, 6>(logits, pre_idx, out, npad, K, kC, BS, stream);
  else
    launch_gvr<16, 512>(logits, pre_idx, out, npad, K, kC, BS, stream);     // streaming for huge n
}

// Probe entry: runtime (cs, maxv, ar) -> register-resident template variant.
// Caller must guarantee vpc = ceil(npad/4/cs) <= maxv*512. Used to measure the
// BS-aware (CS, MAXV) ladder empirically; production dispatch is built from
// the winners.
void gvr_topk_launch_cfg(const float* logits, const int* pre_idx, int* out, int npad, int K,
                         int BS, int tb, int cs, int maxv, int ar, int hs, cudaStream_t stream) {
  int kC = (K >= 2048) ? 8192 : 6144;
#define CFGH(TBV, CSV, MVV, ARV, HSV) \
  if (tb == TBV && cs == CSV && maxv == MVV && ar == ARV && hs == HSV) \
    return launch_reg<CSV, TBV, MVV, ARV, HSV>(logits, pre_idx, out, npad, K, kC, BS, stream);
  CFGH(1024, 1, 9, 8, 2) CFGH(1024, 1, 9, 8, 4) CFGH(1024, 1, 9, 6, 2) CFGH(1024, 1, 9, 4, 2)
  CFGH(1024, 1, 9, 4, 1) CFGH(1024, 1, 9, 6, 1)
  CFGH(1024, 2, 9, 8, 2) CFGH(1024, 2, 9, 8, 4) CFGH(1024, 2, 9, 6, 2)
  CFGH(1024, 4, 9, 8, 2) CFGH(1024, 4, 9, 8, 4) CFGH(1024, 4, 9, 6, 2) CFGH(1024, 4, 9, 6, 1)
  CFGH(1024, 8, 8, 6, 2) CFGH(1024, 8, 8, 6, 4) CFGH(1024, 8, 8, 4, 2)
  CFGH(512, 1, 9, 8, 2) CFGH(512, 1, 9, 8, 4) CFGH(512, 1, 9, 4, 2)
  CFGH(512, 8, 10, 6, 2) CFGH(512, 8, 10, 6, 4)
  CFGH(512, 16, 8, 6, 2)
#undef CFGH
  // maxv == 0 -> streaming kernel (global re-scan per pass; low-reg, high-occ)
#define CFGS(TBV, CSV, ARV, HSV) \
  if (tb == TBV && cs == CSV && maxv == 0 && ar == ARV && hs == HSV) \
    return launch_gvr<CSV, TBV, ARV, HSV>(logits, pre_idx, out, npad, K, kC, BS, stream);
  CFGS(512, 1, 8, 1) CFGS(512, 2, 8, 1) CFGS(512, 4, 8, 1)
  CFGS(1024, 1, 8, 1) CFGS(1024, 2, 8, 1) CFGS(1024, 4, 8, 1)
  CFGS(1024, 1, 8, 2) CFGS(1024, 1, 8, 4) CFGS(1024, 1, 6, 2) CFGS(1024, 1, 6, 4)
  CFGS(1024, 1, 4, 2) CFGS(1024, 1, 4, 4) CFGS(1024, 1, 6, 1) CFGS(1024, 1, 4, 1)
  CFGS(1024, 2, 8, 2) CFGS(1024, 2, 6, 2) CFGS(1024, 2, 4, 2) CFGS(1024, 2, 6, 4)
  CFGS(512, 1, 8, 2) CFGS(512, 1, 6, 2) CFGS(512, 1, 4, 2) CFGS(512, 1, 6, 4)
#undef CFGS
#define CFG(TBV, CSV, MVV, ARV) \
  if (tb == TBV && cs == CSV && maxv == MVV && ar == ARV && hs == 1) \
    return launch_reg<CSV, TBV, MVV, ARV>(logits, pre_idx, out, npad, K, kC, BS, stream);
  CFG(512, 1, 8, 8) CFG(512, 1, 9, 8) CFG(512, 1, 9, 6)
  CFG(512, 2, 5, 8) CFG(512, 2, 9, 8) CFG(512, 2, 9, 6)
  CFG(512, 4, 4, 8) CFG(512, 4, 5, 8) CFG(512, 4, 9, 8) CFG(512, 4, 9, 6)
  CFG(512, 8, 3, 8) CFG(512, 8, 5, 8) CFG(512, 8, 9, 8) CFG(512, 8, 10, 8)
  CFG(512, 8, 9, 6) CFG(512, 8, 10, 6)
  CFG(512, 16, 5, 8) CFG(512, 16, 8, 8) CFG(512, 16, 5, 6) CFG(512, 16, 8, 6)
  CFG(1024, 1, 5, 8) CFG(1024, 1, 8, 8) CFG(1024, 1, 9, 8) CFG(1024, 1, 9, 6)
  CFG(1024, 2, 5, 8) CFG(1024, 2, 9, 8) CFG(1024, 2, 9, 6)
  CFG(1024, 4, 5, 8) CFG(1024, 4, 9, 8) CFG(1024, 4, 9, 6)
  CFG(1024, 8, 5, 8) CFG(1024, 8, 8, 8) CFG(1024, 8, 5, 6) CFG(1024, 8, 8, 6)
#undef CFG
  // unknown combo: fall back to the production dispatch
  gvr_topk_launch(logits, pre_idx, out, npad, K, BS, stream);
}
