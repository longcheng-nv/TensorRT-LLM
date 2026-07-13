/// \file topk_hbec.cuh
/// \brief op31 HBE-C: hint-ladder cluster single-pass top-K (tier 5).
///
/// Domain: the cluster path (BS <= 512, seq_len > cluster_floor). Replaces
/// the stock TopKCluster 2-pass body (Phase1 full-row hist build + all-reduce
/// + Phase3 full-row collect) with ONE speculative row pass:
///   C0  all 8 CTAs redundantly (redundant reads are L2 hits, op17 crux):
///       gather the preIdx hint values, stride-4 subsample (K/4 reads),
///       histogram them into the stock 1024-bin fp16-key hist, and place ONE
///       speculative collect boundary at the hint-rank kSpecPermille/1000
///       crossing (find_threshold reuse -> bin_spec; v_spec = the bin's exact
///       fp32 lower bound, so "val >= v_spec" == "bin >= bin_spec").
///   C1  each CTA, its N/8 chunk, ONE scan: val >= v_spec -> append
///       {val, global_idx} to the dyn-smem candidate buffer (cap 4*K/CTA)
///       AND count it into the (re-zeroed) 1024-bin candidate mini-hist.
///       Sub-boundary elements cost 1 fp32 cmp (vs stock F2F+twiddle+atomic).
///   C2  stock 1-shot DSMEM all-reduce of the mini-hist (+ 2 inlined scalars:
///       cand_count, overflow) -> every CTA has the cluster-wide candidate
///       hist. Candidate bins >= bin_spec are COMPLETE (count-validity), so
///       if total_cand >= K: find_threshold(topk, total_cand) gives the SAME
///       b* as the full-row hist, and the resolve scans run over the stored
///       candidates only (stock Phase 3.5 / handle_tie machinery verbatim).
///   MISS (total_cand < K or any CTA's buffer overflowed): fall back to the
///       UNTOUCHED TopKCluster<8>::forward (it fully re-initializes its own
///       state) -- exactness is unconditional, hint quality moves only speed.
///
/// Rung-0 crux (RUNG0_HBEC_RESULTS.md): frac 0.92, collect-at-loosest,
/// cap 32xK/row -> rr-real 0% miss, real axis 1.3% / E[passes] 1.03.
/// Rung-2 (RUNG2_HBEC_RESULTS.md): the DESIGN's multi-rung ladder + remote-
/// atomic mini-hist are DROPPED: the local mini-hist is candidate-count-
/// insensitive (1.4us flat), so a single boundary + local-build + dense
/// all-reduce strictly dominates (M x 8 scalar select saved nothing once the
/// resolve needs a dense all-reduce anyway).
#pragma once

#include <sgl_kernel/deepseek_v4/topk_impl.cuh>

namespace device::topk {

struct HbecConfig {
  // hint-rank boundary placement (per-mille). Rung-0: 0.92 covers h <= 0.92
  // on every axis incl the op30 worst pole (h 0.85-0.90); the op27 K2048
  // 0.75 top column has an h>0.75 lt_K hole (rr-real N=524288 h=0.82).
  static constexpr uint32_t kSpecPermille = 920;
  static constexpr uint32_t kHintStride = 4;  // K/4 hint gathers (HBE iter4)
  // per-CTA candidate cap = 4*K -> 32*K/row (rung-0 cap-32K policy: real
  // axis miss 1.3%; overflow -> stock fallback, fail-soft).
  static constexpr uint32_t cap(uint32_t topk) { return 4 * topk; }
  static constexpr size_t dyn_smem_bytes(uint32_t topk) {
    return size_t(cap(topk)) * sizeof(TieValue);
  }
};

template <uint32_t kClusterSize_>
struct TopKClusterHbec : TopKCluster<kClusterSize_> {
  using Cluster = TopKCluster<kClusterSize_>;
  using Base = typename Cluster::Base;  // TopKRadixBase<10>
  static constexpr uint32_t kClusterSize = kClusterSize_;
  using Base::kHistBits;
  using Base::kHistSize;
  using typename Base::Smem;  // Cluster::Smem layout reused as-is
  using Cfg = HbecConfig;

  // Extra cluster-visible scalars. Kept OUTSIDE Cluster::Smem so the miss
  // fallback (Cluster::forward) and the DSMEM hist all-reduce see the exact
  // stock layout at the same smem base address.
  struct HbecSmem : Cluster::Smem {
    uint32_t hint_valid;   // C0: # valid hint gathers (this CTA; identical
                           //     across CTAs -- same hint row)
    uint32_t cand_count;   // C1: local candidates (uncapped count)
    uint32_t red[2];       // C2: cluster-reduced {total_cand, overflow}
  };

  template <bool kUsePDL>
  SGL_DEVICE static void forward(const TopKProblem problem,
                                 const int32_t* __restrict__ pre_idx_row,
                                 void* _smem, void* _dyn) {
    const auto tx = threadIdx.x;
    const auto smem = static_cast<HbecSmem*>(_smem);
    const auto cluster = cg::this_cluster();
    const auto this_rank = blockIdx.y;
    const bool is_primary = (this_rank == 0);
    const auto topk = problem.topk;
    const uint32_t cap = Cfg::cap(topk);
    auto* __restrict__ cand = static_cast<TieValue*>(_dyn);
    constexpr uint32_t kBlockSize = Base::kBlockSize;
    constexpr uint32_t kMaxNumTie = Base::kMaxNumTie;

    // chunk geometry: identical to TopKCluster::forward
    constexpr uint32_t kAlignElems = kWarpSize * Base::kVecSize;
    const uint32_t chunk_size =
        div_ceil(problem.seq_len, kClusterSize * kAlignElems) * kAlignElems;
    const uint32_t chunk_start = min(this_rank * chunk_size, problem.seq_len);
    const uint32_t chunk_finish =
        min(chunk_start + chunk_size, problem.seq_len);
    const uint32_t local_seq_len = chunk_finish - chunk_start;

    // ---- C0: hint gather + boundary placement (block-local; all CTAs
    // compute the identical boundary from the identical hint row) ----------
    {
      typename Smem::kHistVec hist_vec;
      hist_vec.fill(0);
      smem->hist_vecs[tx] = hist_vec;
    }
    if (tx == 0) {
      smem->count_eq = 0;
      smem->count_gt = 0;
      smem->hint_valid = 0;
      smem->cand_count = 0;
    }
    __syncthreads();
    PDLWaitPrimary<kUsePDL>();

    const uint32_t n_sub = topk / Cfg::kHintStride;
    if (tx < n_sub) {
      const int32_t raw = pre_idx_row[tx * Cfg::kHintStride];
      if (raw >= 0 && static_cast<uint32_t>(raw) < problem.seq_len) {
        const float hv = problem.in[raw];
        atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(hv)], 1);
        atomicAdd(&smem->hint_valid, 1);
      }
    }
    __syncthreads();
    const uint32_t n_valid = smem->hint_valid;
    uint32_t bin_spec = 0;
    if (n_valid > 0) {
      const uint32_t rank =
          max(1u, (n_valid * Cfg::kSpecPermille) / 1000u);
      Base::find_threshold(rank, n_valid, smem);
      bin_spec = smem->threshold_bin;
    }
    const float v_spec = coarse_bin_lower_bound<kHistBits>(bin_spec);
    // n_valid == 0 -> bin_spec = 0 -> v_spec = -FLT_MAX -> every element is
    // a candidate -> guaranteed buffer overflow at cluster N -> fallback.
    __syncthreads();

    // ---- C1: single speculative chunk pass ------------------------------
    {
      typename Smem::kHistVec hist_vec;
      hist_vec.fill(0);
      smem->hist_vecs[tx] = hist_vec;  // re-zero for the candidate mini-hist
    }
    __syncthreads();
    Base::for_each_input(
        problem.in + chunk_start, local_seq_len,
        [&](float val, uint32_t local_idx) {
          if (val >= v_spec) {
            atomicAdd(&smem->histogram[extract_coarse_bin<kHistBits>(val)],
                      1);
            const auto pos = atomicAdd(&smem->cand_count, 1);
            if (pos < cap) [[likely]] {
              cand[pos] = {val, chunk_start + local_idx};
            }
          }
        });
    __syncthreads();

    // ---- C2: 1-shot DSMEM all-reduce (stock pattern) + 2 inlined scalars -
    {
      cluster.sync();
      static_assert(kHistSize == kBlockSize);
      constexpr uint32_t kPartition = kHistSize / kClusterSize;
      const auto start = this_rank * kPartition;
      const auto which = start + tx / kClusterSize;
      const auto peer_rank = tx % kClusterSize;
      const auto addr =
          cluster.map_shared_rank(&smem->histogram[which], peer_rank);
      const auto value = *addr;
      *addr = warp::reduce_sum<kClusterSize>(value);
      // 2 scalars ride the same sync pair on warp 0. The FULL warp must
      // enter (warp::reduce_sum shuffles with the full mask -- a tx<16
      // guard deadlocks); lanes 16-31 run a redundant duplicate segment.
      if (tx < kWarpSize) {
        const uint32_t m = (tx / kClusterSize) & 1u;  // 0:cand 1:ovf (x2)
        const uint32_t peer = tx % kClusterSize;
        const auto sm =
            static_cast<HbecSmem*>(cluster.map_shared_rank(smem, peer));
        const uint32_t v = (m == 0) ? sm->cand_count
                                    : (sm->cand_count > cap ? 1u : 0u);
        const uint32_t s = warp::reduce_sum<kClusterSize>(v);
        if (peer == 0 && tx < 2 * kClusterSize) smem->red[m] = s;
      }
      cluster.sync();
    }
    const uint32_t total_cand = smem->red[0];
    const uint32_t any_overflow = smem->red[1];

    // ---- MISS: stock fallback (re-initializes everything itself) --------
    if (total_cand < topk || any_overflow != 0) {
      __syncthreads();  // everyone done reading red[] / hist
      Cluster::template forward<false>(problem, _smem);
      return;
    }

    // ---- HIT: exact resolve over the stored candidates -------------------
    // Candidate bins >= bin_spec are complete cluster-wide and
    // total_cand >= topk, so b* == the full-row threshold bin.
    Base::find_threshold(topk, total_cand, smem);

    const auto threshold_bin = smem->threshold_bin;
    const float v_hi = coarse_bin_lower_bound<kHistBits>(threshold_bin + 1);
    const float v_lo = coarse_bin_lower_bound<kHistBits>(threshold_bin);
    const auto cur_out = is_primary ? problem.out : smem->tmp_out;
    const uint32_t n_stored = min(smem->cand_count, cap);
    for (uint32_t t = tx; t < n_stored; t += kBlockSize) {
      const TieValue cv = cand[t];
      if (cv.value >= v_hi) {
        const auto pos = atomicAdd(&smem->count_gt, 1);
        if (pos < topk) [[likely]] {
          cur_out[pos] = static_cast<int32_t>(cv.idx);
        }
      } else if (cv.value >= v_lo) {
        const auto count_eq = atomicAdd(&smem->count_eq, 1);
        if (count_eq < kMaxNumTie) [[likely]] {
          smem->tie.values[count_eq] = cv;
        }
      }
    }

    // ---- stock Phase 3.5 / 4 tail (verbatim TopKCluster) -----------------
    uint32_t start_write = 0;
    uint32_t num_write = 0;
    if (!is_primary) {
      __syncthreads();
      const auto local_above_count = smem->count_gt;
      const auto local_equal_count = min(smem->count_eq, kMaxNumTie);
      const auto smem_0 =
          static_cast<HbecSmem*>(cluster.map_shared_rank(smem, 0));
      if (tx == 0) {
        const auto gt = atomicAdd(&smem_0->count_gt, local_above_count);
        const auto eq = atomicAdd(&smem_0->count_eq, local_equal_count);
        smem->start_gt_local = gt;
        smem->start_eq_local = eq;
      }
      __syncthreads();
      const auto start_gt_local = smem->start_gt_local;
      const auto start_eq_local = smem->start_eq_local;
#pragma unroll
      for (uint32_t i = 0; i < Base::kTieItems; ++i) {
        const auto t = tx + i * kBlockSize;
        if (t < local_equal_count && start_eq_local + t < kMaxNumTie) {
          smem_0->tie.values[start_eq_local + t] = smem->tie.values[t];
        }
      }
      start_write = start_gt_local;
      num_write = local_above_count;
    }

    cluster.sync();
    if (!is_primary) {
#pragma unroll
      for (uint32_t i = 0; i < Base::kTopKItems; ++i) {
        if (const auto t = tx + i * kBlockSize;
            t < num_write && start_write + t < topk) {
          problem.emit(start_write + t, smem->tmp_out[t]);
        }
      }
    } else {
      const auto above_count = smem->count_gt;
      const auto equal_count = smem->count_eq;
      const auto remain_topk = above_count < topk ? topk - above_count : 0;
      const auto tie_count = min(equal_count, kMaxNumTie);
      Base::handle_tie(smem->tie.values, problem, above_count, tie_count,
                       remain_topk, &smem->tie.handle);
    }
  }
};

}  // namespace device::topk
