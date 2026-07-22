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
template <int TB, int CS>
__device__ __forceinline__ void exchange_counts(Smem<TB>* s, int par, int tid, int rank) {
  __syncthreads();  // ptcnt final
  {
    // warp w reduces rung w's per-thread counts (RUNGS <= NWARP always)
    int lane = tid & 31, wid = tid >> 5;
    if (wid < RUNGS) {
      int sum = 0;
      for (int i = lane; i < TB; i += 32) sum += s->ptcnt[wid * TB + i];
#pragma unroll
      for (int o = 16; o; o >>= 1) sum += __shfl_down_sync(0xFFFFFFFFu, sum, o);
      if (lane == 0) s->rcnt_local[wid] = sum;
    }
  }
  __syncthreads();
  if constexpr (CS == 1) {
    if (tid < RUNGS) { s->rcnt[tid] = s->rcnt_local[tid]; s->rpre[tid] = 0; }
    __syncthreads();
  } else {
    cg::cluster_group cl = cg::this_cluster();
    if (tid < RUNGS) s->ipartial[par][tid] = s->rcnt_local[tid];
    cl.sync();
    if (tid < RUNGS) {
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
template <int TB>
__device__ __forceinline__ void phase1(Smem<TB>* s, const float* __restrict__ logits,
                                       const int* __restrict__ pre_idx, int K, int npad, int tid) {
  constexpr int T = TB;
  constexpr int NWARP = TB / 32;
  if (tid < 64) s->hist[tid] = 0;
  float hv[4];
  bool hok[4];
  float mn = FLT_MAX, mx = -FLT_MAX;
#pragma unroll
  for (int jj = 0; jj < 4; ++jj) {
    int j = tid + jj * T;
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
    if (tid < RUNGS) s->rungs[tid] = hmin;
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
    int qtrim = (K * 97) / 100;
    if (A < qtrim && x >= qtrim) {
      int b = (A + h1 >= qtrim) ? (63 - 2 * tid) : (62 - 2 * tid);
      s->hminmax[0] = hmin + binw * (float)b;  // trim point
    }
    if (tid == 31 && x < qtrim) s->hminmax[0] = hmin;  // all bins can't reach qtrim
    if (tid == 0) s->rungs[RUNGS - 1] = hmin;  // guaranteed count >= K globally
  }
  __syncthreads();
  float tlow = s->hminmax[0];
  if (!(hmax - tlow > 0.0f)) {
    if (tid < RUNGS - 1) s->rungs[tid] = hmax;
    __syncthreads();
    return;
  }
  // Stage 2: fine hist over [tlow, hmax]
  if (tid < 64) s->hist[tid] = 0;
  __syncthreads();
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
    qt[0] = (K * 10) / 100;
    qt[1] = (K * 25) / 100;
    qt[2] = (K * 45) / 100;
    qt[3] = (K * 65) / 100;
    qt[4] = (K * 82) / 100;
    qt[5] = (K * 94) / 100;
    int tot = __shfl_sync(0xFFFFFFFFu, x, 31);  // total hints in stage-2 range
#pragma unroll
    for (int r = 0; r < RUNGS - 2; ++r) {
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

template <int CS, int TB>
__global__ void __launch_bounds__(TB, 1) gvr_topk_kernel(
    const float* __restrict__ logits, const int* __restrict__ pre_idx,
    int* __restrict__ out, int npad, int K, int kC) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
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
    phase1<TB>(s, logits, pre_idx, K, npad, tid);
    count_pass<TB, RUNGS>(s, base, v0, v1, tid);
    exchange_counts<TB, CS>(s, xch & 1, tid, rank);
    xch++;

    float t_lo = -FLT_MAX, t_hi = INFINITY;  // count(t_lo) > kC, count(t_hi) < K
    int c_lo = 0x7fffffff, c_hi = 0;
    float span0 = fmaxf(s->hminmax[1] - s->hminmax[0], 1e-3f);
    float lct = __logf(sqrtf((float)K * (float)kC));  // secant target (geometric mid)
    int Rcur = RUNGS;
    for (int pass = 0;; ++pass) {
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
      float nr[RUNGS];  // next pass's rung ladder (descending)
      if (!descend) {
        if (isinf(t_hi)) {
          // every measured count > kC: geometric ladder walking up
          float step = span0 * (float)(1 << (pass * 3 < 24 ? pass * 3 : 24));
#pragma unroll
          for (int r = 0; r < RUNGS; ++r) nr[r] = t_lo + step * (float)(1 << (RUNGS - 1 - r));
          if (isinf(nr[0])) descend = true;
        } else if (t_lo == -FLT_MAX) {
          float step = span0 * (float)(1 << (pass * 3 < 24 ? pass * 3 : 24));
#pragma unroll
          for (int r = 0; r < RUNGS; ++r) nr[r] = t_hi - step * (float)(1 << r);
        } else {
          // secant anchor: log-CCDF is near-linear in t, so counts spaced
          // geometrically between c_hi and c_lo map to a LINEAR ladder in t.
          // 8 inner rungs shrink the bracket >= 9x per pass.
          float dt = (t_hi - t_lo) * (1.0f / (float)(RUNGS + 1));
#pragma unroll
          for (int r = 0; r < RUNGS; ++r) nr[r] = t_hi - dt * (float)(r + 1);
          if (!(nr[RUNGS - 1] > t_lo && nr[0] < t_hi)) descend = true;  // bracket collapsed: plateau
        }
      }
      if (descend) break;

      __syncthreads();
      if (tid < RUNGS) {
        s->rcnt_local[tid] = 0;
        s->rungs[tid] = nr[tid];
      }
      __syncthreads();
      count_pass<TB, RUNGS>(s, base, v0, v1, tid);
      exchange_counts<TB, CS>(s, xch & 1, tid, rank);
      xch++;
      Rcur = RUNGS;
    }

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
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    int shift = 24 - 8 * r;
    if (tid < 256) s->hist[tid] = 0;
    __syncthreads();
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
  }
  unsigned kth = pref;

  if (tid == 0) { s->cnt_m = 0; s->cnt_t = 0; }
  __syncthreads();
  int nt = K - m;
  for (int i = tid; i < C; i += T) {
    unsigned long long kv = s->cand[i];
    unsigned u = (unsigned)(kv >> 32);
    int idx = (int)(unsigned)(kv & 0xFFFFFFFFull);
    if (u > kth) {
      out[atomicAdd(&s->cnt_m, 1)] = idx;
    } else if (u == kth) {
      int p = atomicAdd(&s->cnt_t, 1);
      if (p < nt) out[m + p] = idx;
    }
  }
}

template <int CS, int TB>
static void launch_gvr(const float* logits, const int* pre_idx, int* out, int npad, int K, int kC,
                       cudaStream_t stream) {
  static bool inited = false;
  if (!inited) {
    cudaFuncSetAttribute(gvr_topk_kernel<CS, TB>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)sizeof(Smem<TB>));
    if (CS > 8)
      cudaFuncSetAttribute(gvr_topk_kernel<CS, TB>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    inited = true;
  }
  if constexpr (CS == 1) {
    gvr_topk_kernel<1, TB><<<1, TB, sizeof(Smem<TB>), stream>>>(logits, pre_idx, out, npad, K, kC);
  } else {
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(CS, 1, 1);
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
    cudaLaunchKernelEx(&cfg, gvr_topk_kernel<CS, TB>, logits, pre_idx, out, npad, K, kC);
  }
}


void gvr_topk_launch(const float* logits, const int* pre_idx, int* out, int npad, int K,
                     cudaStream_t stream) {
  int kC = (K >= 2048) ? 8192 : 6144;
  if (npad <= kC && npad >= 3072)
    launch_gvr<1, 1024>(logits, pre_idx, out, npad, K, kC, stream);
  else if (npad < 16384)
    launch_gvr<1, 512>(logits, pre_idx, out, npad, K, kC, stream);
  else if (npad < 32768)
    launch_gvr<4, 512>(logits, pre_idx, out, npad, K, kC, stream);
  else if (npad < 131072)
    launch_gvr<8, 512>(logits, pre_idx, out, npad, K, kC, stream);
  else
    launch_gvr<16, 512>(logits, pre_idx, out, npad, K, kC, stream);
}
