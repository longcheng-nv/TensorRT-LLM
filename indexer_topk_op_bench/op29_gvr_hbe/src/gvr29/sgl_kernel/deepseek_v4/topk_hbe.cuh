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
  // iter8: CHUNKED sampling — 64 evenly-spaced chunks of 64 contiguous
  // elements (4096 samples). Per-element strided gather (iter6/7) was a
  // DRAM-burst disaster: 128-256B stride = 32B useful per 128B burst,
  // ~half a pass of waste at BS=1024. Chunked = coalesced 256B runs,
  // 16KB/row total. Positional trends average over the 64 chunks.
  static constexpr uint32_t kSampleChunks = 64;
  static constexpr uint32_t kChunkElems = 64;
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

  /// pre_idx: RESERVED (iter7 uses the hint-free row-sample estimator; the
  /// hint may return for sub-131K dispatch tiers). Unused.
  template <bool kUsePDL>
  SGL_DEVICE static void forward(const TopKProblem problem,
                                 const int32_t* __restrict__ /*pre_idx*/,
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

    // ---- Phase H0 (iter7): row-sample mini-hist, CAND-TARGETED columns ---
    // Columns are placed by candidate BUDGET, not by value-quantile guesses:
    //   binA = sample bin at cumulative rank ~2*rS_K  (targets cand ~2*K)
    //   binB = sample bin at cumulative rank ~8*rS_K  (targets cand ~8*K)
    // where rS_K = n_samp*K/N is the sample-space image of rank K. A column
    // targeting cand >= 2K sits at/below b* except under ~2x downward
    // sampling noise (tier B + miss fallback absorb the tail). Scenario-
    // invariant and hint-free (iter6 showed quantile-of-hint max() breaks
    // one-sided safety; iter3 showed hint gather is a real tax).
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

    constexpr uint32_t kNSamp =
        HbeConfig::kSampleChunks * HbeConfig::kChunkElems;
    const uint32_t chunk_stride =
        max(HbeConfig::kChunkElems, problem.seq_len / HbeConfig::kSampleChunks);
    const uint32_t n_samp = kNSamp;
    for (uint32_t t = tx; t < kNSamp; t += kBlockSize) {
      const uint32_t c = t / HbeConfig::kChunkElems;
      const uint32_t o = t % HbeConfig::kChunkElems;
      const uint32_t idx = min(c * chunk_stride + o, problem.seq_len - 1);
      const float sv = problem.in[idx];
      atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(sv)], 1);
    }
    __syncthreads();
    const uint32_t rS_K = max(1u, static_cast<uint32_t>(
        (static_cast<uint64_t>(n_samp) * topk) / problem.seq_len));
    const uint32_t rk_a = min(n_samp, 2 * rS_K);
    const uint32_t rk_b = min(n_samp, 8 * rS_K);
    find_threshold(rk_a, n_samp, smem);
    __syncthreads();
    if (tx == 0) hc.bin_a = smem->threshold_bin;
    __syncthreads();
    find_threshold(rk_b, n_samp, smem);
    __syncthreads();
    if (tx == 0) {
      hc.bin_b = smem->threshold_bin;
      if (hc.bin_b > hc.bin_a) hc.bin_b = hc.bin_a;  // keep B <= A
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
