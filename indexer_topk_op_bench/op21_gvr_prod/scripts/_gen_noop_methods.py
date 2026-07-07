import cutlass  # noqa
import cutlass.cute as cute

@cute.jit
def _p3_leader_band_gather(self, rank, smem_keys, smem_vals, s_cluster, tidx):
    pass

@cute.jit
def block_count_collect_multi(self, input_row, N, s_mt_thr, smem_ptcnt_multi, smem_wcnt_multi, s_mt_cnt, smem_slotk, smem_slotv, tidx, warp_id, lane):
    pass

@cute.jit
def block_count_collect_multi_base(self, input_row, base, Ns, s_mt_thr, smem_ptcnt_multi, smem_wcnt_multi, s_mt_cnt, smem_slotk, smem_slotv, tidx, warp_id, lane):
    pass

@cute.jit
def block_count_collect_multi_smem(self, smem_rowbuf, N, s_mt_thr, smem_ptcnt_multi, smem_wcnt_multi, s_mt_cnt, smem_slotk, smem_slotv, tidx, warp_id, lane):
    pass

@cute.jit
def block_count_ge(self, input_row, N, threshold, smem_ptcnt, smem_wcnt, s_iscalars, tidx, warp_id, lane):
    pass

@cute.jit
def phase1_stats_stash(self, input_row, N, pre_idx_row, pre_idx_count, pre_idx_offset, smem_stash, smem_wmin_f32, smem_wmax_f32, smem_wsum_f32, smem_wcnt_i32, s_thr, s_iscalars, tidx, warp_id, lane):
    pass

@cute.jit
def phase1b_rank_quantile(self, smem_stash, pre_idx_count, smem_hist, s_thr, s_mt_thr, s_mt_cnt, tidx):
    pass

@cute.jit
def phase3_collect_candidates(self, input_row, N, smem_keys, smem_vals, smem_ptcnt, smem_wcnt, s_thr, s_iscalars, tidx, warp_id, lane):
    pass

@cute.jit
def phase3_from_slots(self, smem_slotk, smem_slotv, smem_keys, smem_vals, smem_ptcnt, smem_ptcnt_up, smem_ptcnt_multi, smem_wcnt, smem_didx, s_thr, s_swf, s_iscalars, output_indices_row, tidx, warp_id, lane):
    pass

@cute.jit
def phase3_from_slots_mc(self, smem_slotk, smem_slotv, smem_keys, smem_vals, smem_ptcnt, smem_ptcnt_up, smem_ptcnt_multi, smem_wcnt, s_thr, s_swf, s_iscalars, output_indices_row, d_off, b_off, rank, tidx, warp_id, lane):
    pass

@cute.jit
def phase3_sandwich(self, input_row, N, smem_keys, smem_vals, smem_ptcnt, smem_ptcnt_up, smem_wcnt, smem_didx, s_thr, s_swf, s_iscalars, output_values_row, output_indices_row, tidx, warp_id, lane):
    pass

@cute.jit
def phase4_band_rank_scatter(self, smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_swf, s_iscalars, output_values_row, output_indices_row, band, k_rem, m0, tidx, warp_id, lane):
    pass

@cute.jit
def phase4_band_snap(self, smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_swf, s_iscalars, output_values_row, output_indices_row, band, k_rem, m0, tidx, warp_id, lane):
    pass

@cute.jit
def phase4_band_snap_hist(self, smem_keys, smem_vals, smem_hist, smem_wcnt, s_thr, s_swf, s_iscalars, output_values_row, output_indices_row, band, k_rem, m0, tidx, warp_id, lane):
    pass

@cute.jit
def phase4_dist(self, rank, m0g, k_rem, smem_keys, smem_vals, smem_hist, smem_merged, smem_slotk, smem_slotv, smem_wcnt, s_cluster, s_thr, s_swf, s_iscalars, output_indices_row, tidx, warp_id, lane):
    pass
