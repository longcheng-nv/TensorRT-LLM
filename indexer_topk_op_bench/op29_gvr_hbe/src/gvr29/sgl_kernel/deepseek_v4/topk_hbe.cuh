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
  // 227 KB/SM). Dispatch guards K <= 1024, so 4K*8 + 2K*4 = 40 KB max dyn.
  // iter4: capA 2K->4K (worst-scenario cand med 3.4xK overflowed 2K).
  static constexpr uint32_t capA(uint32_t topk) { return 4 * topk; }
  static constexpr uint32_t capB(uint32_t topk) { return 2 * topk; }
  // iter4: subsample the hint gather 4x (quantile estimate needs only
  // ~hundreds of samples; the full-K gather was K*BS random reads).
  static constexpr uint32_t kHintStride = 4;
  // iter6: hint-free ROW-SAMPLE estimator (scenario-invariant; rescues the
  // uncorrelated-hint worst pole). s strided samples -> mini-hist -> bin at
  // rank ceil(s*K/N); columns take max(hint, sample-guard) per tier.
  static constexpr uint32_t kSampleTarget = 4096;
  static constexpr uint32_t kSampleGuardA = 1;
  static constexpr uint32_t kSampleGuardB = 6;
  // iter5: global spill (flashinfer-style) — candidates past the smem caps
  // go to a per-row global region instead of forcing a full redo pass.
  // Spill traffic ~(cand-cap)*8B*2 vs redo N*4B: e.g. worst K512 N=262144
  // cand~13xK -> ~57KB/row spill vs 1MB/row redo.
  static constexpr uint32_t spillA(uint32_t topk) { return 28 * topk; }
  static constexpr uint32_t spillB(uint32_t topk) { return 28 * topk; }
  static constexpr size_t spill_bytes_per_row(uint32_t topk) {
    return size_t(spillA(topk) + spillB(topk)) * sizeof(TieValue);
  }
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
                                 TieValue* __restrict__ spill_row,
                                 void* _smem, void* _dyn) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<Smem*>(_smem);
    const auto topk = problem.topk;
    const uint32_t capA = HbeConfig::capA(topk);
    const uint32_t capB = HbeConfig::capB(topk);
    auto* bufA = static_cast<TieValue*>(_dyn);
    auto* bufB = reinterpret_cast<int32_t*>(bufA + capA);
    const uint32_t spA = HbeConfig::spillA(topk);
    const uint32_t spB = HbeConfig::spillB(topk);
    TieValue* __restrict__ spillA_g = spill_row;          // [spA]
    TieValue* __restrict__ spillB_g = spill_row + spA;    // [spB]
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

    // gather a 1/kHintStride subsample of the hint values (clamped)
    const uint32_t n_hint = max(1u, topk / HbeConfig::kHintStride);
    for (uint32_t t = tx; t < n_hint; t += kBlockSize) {
      const uint32_t hi = min(
          static_cast<uint32_t>(max(pre_idx[t * HbeConfig::kHintStride], 0)),
          problem.seq_len - 1);
      const float hv = problem.in[hi];
      atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(hv)], 1);
    }
    __syncthreads();
    // rank-from-top rA/rB over the n_hint-entry hint histogram
    const uint32_t rA = max(1u, n_hint * HbeConfig::kQaPermille / 1000u);
    const uint32_t rB = max(1u, n_hint * HbeConfig::kQbPermille / 1000u);
    find_threshold(rA, n_hint, smem);
    __syncthreads();
    if (tx == 0) hc.bin_a = smem->threshold_bin;
    __syncthreads();
    find_threshold(rB, n_hint, smem);
    __syncthreads();
    if (tx == 0) {
      const uint32_t bb = smem->threshold_bin;
      hc.bin_b = bb > HbeConfig::kQbMarginBins ? bb - HbeConfig::kQbMarginBins
                                               : 0u;
      if (hc.bin_b > hc.bin_a) hc.bin_b = hc.bin_a;  // keep A above B
    }
    __syncthreads();

    // ---- Phase H0b (iter6): row-sample estimator, fused with hint cols ---
    {
      const uint32_t stride =
          max(1u, problem.seq_len / HbeConfig::kSampleTarget);
      const uint32_t n_samp = (problem.seq_len + stride - 1) / stride;
      // re-zero hist for the sample mini-hist
      {
        typename Smem::kHistVec hist_vec;
        hist_vec.fill(0);
        smem->hist_vecs[tx] = hist_vec;
      }
      __syncthreads();
      for (uint32_t t = tx; t < n_samp; t += kBlockSize) {
        const float sv = problem.in[t * stride];
        atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(sv)], 1);
      }
      __syncthreads();
      // sample-space rank of the row's K-th value
      const uint32_t rS = max(1u, static_cast<uint32_t>(
          (static_cast<uint64_t>(n_samp) * problem.topk) / problem.seq_len));
      find_threshold(rS, n_samp, smem);
      __syncthreads();
      if (tx == 0) {
        const uint32_t bs = smem->threshold_bin;
        const uint32_t sA =
            bs > HbeConfig::kSampleGuardA ? bs - HbeConfig::kSampleGuardA : 0u;
        const uint32_t sB =
            bs > HbeConfig::kSampleGuardB ? bs - HbeConfig::kSampleGuardB : 0u;
        // tighter (higher) of hint/sample per column; keep B <= A
        hc.bin_a = max(hc.bin_a, sA);
        hc.bin_b = max(hc.bin_b, sB);
        if (hc.bin_b > hc.bin_a) hc.bin_b = hc.bin_a;
      }
      __syncthreads();
    }
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
        if (p < capA) {
          bufA[p] = {val, idx};
        } else if (p < capA + spA) {
          spillA_g[p - capA] = {val, idx};
        }
      } else if (val >= vB) {
        const auto p = atomicAdd(&hc.cnt_b, 1);
        if (p < capB) {
          bufB[p] = static_cast<int32_t>(idx);
        } else if (p < capB + spB) {
          spillB_g[p - capB] = {val, idx};
        }
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

    const bool tierA = (bstar >= binA) && (cntA <= capA + spA);
    const bool tierB = !tierA && (bstar >= binB) && (cntA <= capA + spA)
                       && (cntB <= capB + spB);
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
      // resolve from smem candidates + global spill (no further full pass)
      const uint32_t nA_smem = min(cntA, capA);
      for (uint32_t t = tx; t < nA_smem; t += kBlockSize) {
        const auto e = bufA[t];
        classify(e.value, e.idx);
      }
      for (uint32_t t = capA + tx; t < cntA; t += kBlockSize) {
        const auto e = spillA_g[t - capA];
        classify(e.value, e.idx);
      }
      if (tierB) {
        const uint32_t nB_smem = min(cntB, capB);
        for (uint32_t t = tx; t < nB_smem; t += kBlockSize) {
          const auto idx = static_cast<uint32_t>(bufB[t]);
          classify(problem.in[idx], idx);  // <=capB random re-gathers
        }
        for (uint32_t t = capB + tx; t < cntB; t += kBlockSize) {
          const auto e = spillB_g[t - capB];
          classify(e.value, e.idx);
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
