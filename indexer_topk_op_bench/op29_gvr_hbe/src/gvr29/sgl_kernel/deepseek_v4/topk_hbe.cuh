/// \file topk_hbe.cuh
/// \brief op29 GVR-HBE: hint-boundary-exact streaming top-K.
///
/// Fuses sglang-v2's histogram pass and collect pass into ONE DRAM pass using
/// the GVR preIdx hint: two speculative collect columns (A = tight hint
/// quantile qA, B = safe quantile qB - margin) are derived from a mini
/// histogram of the K hint VALUES; the single fused pass then
///   (a) builds the full 4096-bin histogram (exact threshold bin b* known
///       post-pass, unconditionally),
///   (b) collects {val,idx} >= v(bA) into smem bufA and idx-only in
///       [v(bB), v(bA)) into smem bufB.
/// Tiered resolve:
///   b* >= bA and bufA intact          -> resolve inside bufA        (1 pass)
///   b* >= bB and bufA+bufB intact     -> resolve A + B (re-gather)  (1 pass
///                                        + <=capB random reads)
///   else                              -> stock second collect pass at b*
///                                        (== sglang v2 cost; fail-soft)
/// Exactness is UNCONDITIONAL: b* always comes from the full histogram and
/// the boundary bin is resolved by the same handle_tie machinery; hint
/// quality moves only speed. Derived from the vendored sglang v2 impl
/// (Apache-2.0); baseline structs untouched.
#pragma once

#include <sgl_kernel/deepseek_v4/topk_impl.cuh>

namespace device::topk {

/// Per-mille hint-quantile columns (compile-time; crux-calibrated 2026-07-13:
/// qA=0.75 cand med 1.7-3.5xK, qB=0.90 one-sided-safe 100% all scenarios).
struct HbeConfig {
  static constexpr uint32_t kQaPermille = 750;
  static constexpr uint32_t kQbPermille = 900;
  static constexpr uint32_t kQbMarginBins = 2;
  // capA entries hold {val,idx} (8 B), capB entries hold idx (4 B).
  // Budget: static smem ~52 KB + dyn <= ~60 KB keeps occupancy 2 (B200
  // 227 KB/SM). At K=2048: 2K*8 + 2K*4 = 48 KB dyn.
  static constexpr uint32_t capA(uint32_t topk) { return 2 * topk; }
  static constexpr uint32_t capB(uint32_t topk) { return 2 * topk; }
  static constexpr size_t dyn_smem_bytes(uint32_t topk) {
    return size_t(capA(topk)) * sizeof(TieValue)
         + size_t(capB(topk)) * sizeof(int32_t);
  }
};

struct TopKHbeStreaming : TopKStreaming {
  // dynamic-smem layout: [ TieValue bufA[capA] | int32 bufB[capB] ]
  struct HbeCounters {
    uint32_t cnt_a, cnt_b;
    uint32_t bin_a, bin_b;
  };

  /// pre_idx: K hint indices for this row (already offset for this batch
  /// element); values are clamped into [0, seq_len) — garbage hints only
  /// cost speed, never correctness.
  template <bool kUsePDL>
  SGL_DEVICE static void forward(const TopKProblem problem,
                                 const int32_t* __restrict__ pre_idx,
                                 void* _smem, void* _dyn) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<Smem*>(_smem);
    const auto topk = problem.topk;
    const uint32_t capA = HbeConfig::capA(topk);
    const uint32_t capB = HbeConfig::capB(topk);
    auto* bufA = static_cast<TieValue*>(_dyn);
    auto* bufB = reinterpret_cast<int32_t*>(bufA + capA);
    __shared__ HbeCounters hc;

    // ---- Phase H0: hint mini-histogram -> column bins bA >= bB ----------
    {
      typename Smem::kHistVec hist_vec;
      hist_vec.fill(0);
      smem->hist_vecs[tx] = hist_vec;
    }
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
      hc.cnt_a = hc.cnt_b = 0;
    }
    __syncthreads();
    PDLWaitPrimary<kUsePDL>();

    // gather K hint values; clamp indices for safety
    for (uint32_t t = tx; t < topk; t += kBlockSize) {
      const uint32_t hi = min(static_cast<uint32_t>(max(pre_idx[t], 0)),
                              problem.seq_len - 1);
      const float hv = problem.in[hi];
      atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(hv)], 1);
    }
    __syncthreads();
    // rank-from-top rA/rB over the K-entry hint histogram
    const uint32_t rA = max(1u, topk * HbeConfig::kQaPermille / 1000u);
    const uint32_t rB = max(1u, topk * HbeConfig::kQbPermille / 1000u);
    find_threshold(rA, topk, smem);
    __syncthreads();
    if (tx == 0) hc.bin_a = smem->threshold_bin;
    __syncthreads();
    find_threshold(rB, topk, smem);
    __syncthreads();
    if (tx == 0) {
      const uint32_t bb = smem->threshold_bin;
      hc.bin_b = bb > HbeConfig::kQbMarginBins ? bb - HbeConfig::kQbMarginBins
                                               : 0u;
      if (hc.bin_b > hc.bin_a) hc.bin_b = hc.bin_a;  // keep A above B
    }
    __syncthreads();
    const uint32_t binA = hc.bin_a, binB = hc.bin_b;
    const float vA = coarse_bin_lower_bound<kHistBits>(binA);
    const float vB = coarse_bin_lower_bound<kHistBits>(binB);

    // re-zero the histogram for the main pass
    {
      typename Smem::kHistVec hist_vec;
      hist_vec.fill(0);
      smem->hist_vecs[tx] = hist_vec;
    }
    __syncthreads();

    // ---- Phase H1: ONE fused pass: full histogram + dual-column collect --
    for_each_input(problem.in, problem.seq_len, [&](float val, uint32_t idx) {
      atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(val)], 1);
      if (val >= vA) {
        const auto p = atomicAdd(&hc.cnt_a, 1);
        if (p < capA) bufA[p] = {val, idx};
      } else if (val >= vB) {
        const auto p = atomicAdd(&hc.cnt_b, 1);
        if (p < capB) bufB[p] = static_cast<int32_t>(idx);
      }
    });
    __syncthreads();

    // ---- Phase H2: exact threshold + tiered resolve ---------------------
    find_threshold(topk, problem.seq_len, smem);
    __syncthreads();
    const uint32_t bstar = smem->threshold_bin;
    const uint32_t cntA = hc.cnt_a, cntB = hc.cnt_b;
    const float v_hi = coarse_bin_lower_bound<kHistBits>(bstar + 1);
    const float v_lo = coarse_bin_lower_bound<kHistBits>(bstar);

    const bool tierA = (bstar >= binA) && (cntA <= capA);
    const bool tierB = !tierA && (bstar >= binB) && (cntA <= capA)
                       && (cntB <= capB);
    if (tx == 0) {
      smem->count_gt = 0;
      smem->count_eq = 0;
    }
    __syncthreads();

    const auto classify = [&](float val, uint32_t idx) {
      if (val >= v_hi) {
        const auto pos = atomicAdd(&smem->count_gt, 1);
        if (pos < topk) [[likely]]
          problem.emit(pos, idx);
      } else if (val >= v_lo) {
        const auto ce = atomicAdd(&smem->count_eq, 1);
        if (ce < kMaxNumTie) [[likely]]
          smem->tie.values[ce] = {val, idx};
      }
    };

    if (tierA || tierB) {
      // resolve from smem candidates only (no further full pass)
      for (uint32_t t = tx; t < cntA; t += kBlockSize) {
        const auto e = bufA[t];
        classify(e.value, e.idx);
      }
      if (tierB) {
        for (uint32_t t = tx; t < cntB; t += kBlockSize) {
          const auto idx = static_cast<uint32_t>(bufB[t]);
          classify(problem.in[idx], idx);  // <=capB random re-gathers
        }
      }
    } else {
      // MISS: stock second pass at the exact bin (== rival's collect pass)
      for_each_input(problem.in, problem.seq_len,
                     [&](float val, uint32_t idx) { classify(val, idx); });
    }

    __syncthreads();
    const auto above_count = smem->count_gt;
    const auto equal_count = smem->count_eq;
    const auto remain_topk = above_count < topk ? topk - above_count : 0;
    const auto tie_count = min(equal_count, kMaxNumTie);
    handle_tie(smem->tie.values, problem, above_count, tie_count, remain_topk,
               &smem->tie.handle);
  }
};

}  // namespace device::topk
