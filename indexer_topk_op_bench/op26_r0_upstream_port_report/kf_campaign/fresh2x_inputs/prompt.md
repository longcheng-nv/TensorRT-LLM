# DeepSeek Indexer Top-K Decode (BS 1-1024, fp32, B200) — Optimize the Production GVR Kernel, Fresh Start

## Problem

Sparse-attention indexer top-K selection at decode time, batched.
`logits[b, npad]` fp32 holds REAL captured production indexer rows, valid
length `n_valid`, tail padded so pad never enters the top-k. For b > 1
every batch row is a materialized copy of the same captured row (distinct
memory — no L2-aliasing shortcuts). Return `indices[b, k]` int32: per-row
indices of the k largest values, any order; ties at the k-th value
boundary may be resolved either way (the correctness checker is
index-SET based and tie-robust, applied to EVERY row). Exactness is
non-negotiable: every index whose value is strictly greater than the
k-th value must appear, on every run, on every row.

`pre_idx[b, k]` int32 is the PREVIOUS decode step's top-k (temporal warm
hint; identical copies across rows). Overlap with the true top-k ranges
0.02-1.0 across workloads (typically >0.5). Exploiting it is REQUIRED
(see skeleton below), but correctness and the no-regression bar must
hold even at 0.02 overlap. You may NOT branch on any estimate of hint
quality computed outside the kernel (hit-rate is unknowable at
inference); in-kernel admission escape / lagged feedback is fine.

Workloads are REAL production captures from three models, n up to ~1.05M:
- V4-Flash: k=512,  ISL rungs 4K-1M
- V4-Pro:   k=1024, ISL rungs 4K-1M   (HIGHEST priority)
- V3.2:     k=2048, ISL rungs 4K-256K
b spans {1, 4, 32, 128, 256, 1024} on the platform; the external
acceptance grid covers BS 2-1024 densely, so treat b as fully dynamic.
The logits distribution is NOT random — heavy-tailed real indexer scores
(near-exponential CCDF); algorithms that look good on `randn` behave
differently here. Priority for effort allocation:
V4-Pro > V4-Flash ≈ V3.2; n in 32K-1M > n in 4K-32K.

## Baseline & starting point

The baseline you must beat is the CURRENT PRODUCTION kernel: the
guess-verify-refine (GVR) top-K from TensorRT-LLM PR#16457 (latest head,
including its K=2048 tail-ladder tuning), run NATIVELY BATCHED
([b, npad] in one launch). Its structure: seed a threshold guess from
`pre_idx`, verify/refine the threshold with a secant solve in log space,
then exactly collect the surviving candidates. It is written in CuTe DSL
(Python); a verbatim source digest — full config/dispatch/orchestration
layers plus the signature and docstring of every phase primitive — is in
the APPENDIX at the bottom of this brief. Read it first — your job is to
make THIS algorithm faster (re-expressed in CUDA C++), not to replace
it. This is a FRESH-START campaign: no prior optimized variant is
provided, and you must find the profitable directions yourself by
profiling and analysis.

The per-workload baseline timings (your speedup denominator) are
EXTERNAL nsys cold-L2 pure-kernel-time medians of exactly that
production kernel on an idle B200 — they contain none of the ~15µs
harness floor your own platform measurements include. Consequence: at
true kernel parity your platform speedup will read ~0.5-0.9x on
small-BS/small-n cells, NOT 1.0x. Do not be discouraged by sub-1.0
readings early on — track RELATIVE progress across submissions; at
BS>=32 the floor is negligible and platform numbers are close to truth.

Two measured facts about the production kernel you may rely on (from the
external report on the full 865-cell BS=1 grid):
- The FINAL-COLLECT block (threshold handoff + refine + writeback, "P4")
  dominates: largest phase on 827 of 865 cells, median ~37% of kernel
  time (range 23-58%). The mid scan/count passes are second.
- Real data is UNDERSHOOT-biased for hint-seeded thresholds (the seeded
  count almost always comes in below k, not above): guards that only
  fire on overshoot are dead code here.
How to attack is yours to work out.

## Target — two external acceptance bars

Bar 1 (BS=1): geomean speedup >= 1.60x over the production baseline on
the FULL external 865-cell BS=1 grid (the BS=1 cells here are a
stratified subset), with NO case slower than the baseline anywhere
(no-regression is a hard acceptance bar — a kernel that wins big on
average but loses any cell will be rejected downstream).

Bar 2 (BS>1): geomean speedup >= 2.0x over the natively-batched
production baseline across the external BS 2-1024 grid (all models, all
ISL rungs), again with no per-case regression.

The production baseline itself batches natively and amortizes launch
overhead well — your denominator is strongest at mid/high BS. Expect the
required curve shape: win latency at BS=1, win throughput at BS>=32.
Final acceptance re-measures externally with nsys cold-L2 (L2 flushed
between iterations). Do not overfit to the exact n values here: n is
dynamic (up to ~1.05M), k in {512, 1024, 2048} and b in {1..1024} at
runtime, hint quality is dynamic.

## Required algorithmic skeleton — HARD compliance rule

Keep the GVR skeleton per row: (a) `pre_idx` as the threshold prior,
(b) a secant+log-transform style exact threshold solve (or an equivalent
threshold-refinement structure), (c) an exact refine of the surviving
candidates. Any per-stage restructuring that preserves exactness is
allowed — in particular P1 (prior/seed) and P4 (final collect) may be
re-engineered freely as long as the stage contract holds. Mature
primitives (histogram ladders, radix digit passes, CUB block/warp
primitives) may be absorbed INTO stages.

**Non-negotiable:** a submission that abandons the `pre_idx` prior, or
replaces the threshold-prior structure wholesale with a prior-free
selection algorithm (plain radix-select, full sort, sampling-based
selection), is NON-COMPLIANT and will be rejected even if it is faster.
Likewise, do not build a per-case dispatcher across unrelated top-k
operators. The goal of this campaign is a better GVR, not a different
algorithm.

## Correctness traps

- The k-th-value tie boundary: the checker requires ALL indices with
  value strictly greater than the k-th value, plus any tie subset to
  fill the remainder. Arrival-order races on the boundary bin under
  concurrent compaction are the classic silent bug — never drop a
  strictly-greater element. At b > 1 the same race can fire on some rows
  only; the checker checks every row.
- Batch rows are independent copies: never share per-row state (counts,
  thresholds, candidate buffers) across rows without per-row isolation.

## Requirements

- CUDA C++ (sm_100a Blackwell B200). fp32 in, int32 indices out.
- Exact per the tie-robust set semantics above — no approximation.
- Dynamic n (up to ~1.05M, padded width `npad = ceil(n/64)*64`), dynamic
  b (1..1024), dynamic hint quality, k in {512, 1024, 2048} at runtime.
- Deterministic output not required (any tie resolution accepted), but
  the index set must be exactly right on every run.
- Few kernel launches preferred; launch overhead is material at BS=1
  (3-29 µs scale). CUDA graphs / framework kernels are banned by the
  compliance judge — win inside the kernel.

---

## APPENDIX — baseline kernel source digest (verbatim spans from the production CuTe DSL file, PR#16457 pinned head)

The production kernel is a 5,000-line CuTe DSL (Python) file; it cannot be shipped whole. Below are VERBATIM spans: full text of the per-K constants, constructor (every tuning knob + tuned default), launch-time config dispatch, and the per-row orchestration body; plus signature + docstring of every phase primitive. Timings in baselines.jsonl were measured from exactly this code (see Baseline section).

```python
class GvrParams:
    kFTarget: int
    kC: int  # candidate buffer cap
    kNumBins: int  # histogram bin count

    @staticmethod
    def get(dtype_name: str, top_k: int, compress_ratio: int = 1) -> "GvrParams":
        """Per-(dtype, K, cr) tuning constants, mirroring CUDA's
        ``GvrParams<T, K>`` template specialization. For K ∈ {512, 1024}
        cr=1 (DSv3.2) and cr=4 (DSv4, PR #14413) use different kFTarget —
        V4 aligns kFTarget with kK to avoid upper-clamp saturation on
        tight-sigma layers (1.5-2.2x fewer P2 iters on swe-bench). K=2048 is
        identical across cr (V4 doesn't natively use it).
        """
        TABLE = {
            # --- cr = 1 (DSv3.2): tuned on V3.2 swe-bench data ---
            ("float32", 512, 1): GvrParams(kFTarget=384, kC=5120, kNumBins=1024),
            ("float32", 1024, 1): GvrParams(kFTarget=2560, kC=5120, kNumBins=1024),
            ("float32", 2048, 1): GvrParams(kFTarget=3072, kC=6144, kNumBins=2048),
            ("bfloat16", 512, 1): GvrParams(kFTarget=384, kC=5120, kNumBins=512),
            ("bfloat16", 1024, 1): GvrParams(kFTarget=2560, kC=5120, kNumBins=512),
            ("bfloat16", 2048, 1): GvrParams(kFTarget=4096, kC=5120, kNumBins=2048),
            ("float16", 512, 1): GvrParams(kFTarget=384, kC=5120, kNumBins=512),
            ("float16", 1024, 1): GvrParams(kFTarget=2560, kC=5120, kNumBins=1024),
            ("float16", 2048, 1): GvrParams(kFTarget=4096, kC=5120, kNumBins=2048),
            # --- cr = 4 (DSv4): tuned on V4 Flash/Pro swe-bench data ---
            ("float32", 512, 4): GvrParams(kFTarget=512, kC=5120, kNumBins=1024),
            ("float32", 1024, 4): GvrParams(kFTarget=1024, kC=5120, kNumBins=1024),
            ("float32", 2048, 4): GvrParams(kFTarget=3072, kC=6144, kNumBins=2048),
            ("bfloat16", 512, 4): GvrParams(kFTarget=512, kC=5120, kNumBins=512),
            ("bfloat16", 1024, 4): GvrParams(kFTarget=1024, kC=5120, kNumBins=512),
            ("bfloat16", 2048, 4): GvrParams(kFTarget=4096, kC=5120, kNumBins=2048),
            ("float16", 512, 4): GvrParams(kFTarget=512, kC=5120, kNumBins=512),
            ("float16", 1024, 4): GvrParams(kFTarget=1024, kC=5120, kNumBins=1024),
            ("float16", 2048, 4): GvrParams(kFTarget=4096, kC=5120, kNumBins=2048),
        }
        key = (dtype_name, top_k, compress_ratio)
        if key not in TABLE:
            raise ValueError(f"Unsupported GvrParams<{dtype_name}, {top_k}, cr={compress_ratio}>")
        return TABLE[key]

class GvrTopKKernel:
    def __init__(
        self,
        ...

    @cute.jit
    def phase1_preidx_stats(
        self,
        input_row,  # cute.Tensor [N] fp32 (post-cast for half-prec)
        N,  # length of input_row
        pre_idx_row,  # cute.Tensor [M] int32
        pre_idx_count,
        pre_idx_offset,
        smem_wmin_f32,  # cute.Tensor [NUM_WARPS] float32
        smem_wmax_f32,  # cute.Tensor [NUM_WARPS] float32
        smem_wsum_f32,  # cute.Tensor [NUM_WARPS] float32
        smem_wcnt_i32,  # cute.Tensor [NUM_WARPS] int32
        s_thr,  # cute.Tensor [3] float32: [threshold, val_lo, val_hi]
        s_iscalars,  # cute.Tensor [6] int32: [cand_count, done, cnt_lo, cnt_hi, out_count, local_cand_count]
        tidx,
        warp_id,
        lane,
        smem_gath=None,  # cute.Tensor [top_k] f32 or None (p1b_cache): stash
        # the gathered value per preIdx slot so P1b skips a 2nd GMEM gather.
        s_mt_thr=None,  # r0_vseed: P1 also parks pmean in the last rung
        # column (visibility via P1's own trailing barrier -> zero extra sync).
    ):
        """preIdx scan + warp reduce + block aggregate + initial threshold.

        Smem layout split: floats kept in fp32 buffers, ints kept in int32
        buffers (no bit-cast tricks needed — avoids ArithValue/ir_value
        coupling and keeps types clean for the MLIR codegen).
        """
        ...

    @cute.jit
    def phase1b_hspace_rungs(
        self,
        ...

    @cute.jit
    def block_count_ge(
        self,
        input_row,  # cute.Tensor [N] fp32 (full row; this CTA only scans its slice)
        slice_start,  # int32: index in input_row where this CTA's slice starts
        slice_end,  # int32: index in input_row where this CTA's slice ends
        threshold,  # cutlass.Float32 scalar
        smem_ptcnt,  # cute.Tensor [BLOCK_SIZE] int32 (P3 cache)
        smem_wcnt,  # cute.Tensor [NUM_WARPS] int32 (block reduce scratch)
        s_iscalars,  # cute.Tensor [6] int32 (writes [0] = cand_count)
        s_cluster_partial,  # cute.Tensor [1] int32 (per-CTA partial scratch for DSMEM)
        tidx,
        warp_id,
        lane,
        do_cluster_sync,  # bool: False = skip DSMEM aggregation (cs=1 / short-row degrade)
        smem_input=None,  # optional SMEM-cached slice (smem_input[i] == input_row[slice_start+i])
        redundant=False,  # trace-time: every-warp reduce, return the total
        wcnt_off=None,  # int32 staging bank offset into smem_wcnt (parity)
    ):
        """Count input[i] >= threshold across this CTA's row slice, then
        DSMEM-aggregate across the cluster.

        ``redundant=True`` (p2_warp_redundant, cluster_size == 1 only):
        after the staging barrier EVERY warp reduces the warp counts
        lane-parallel and the block total RETURNS in a register —
        bit-identical across warps — instead of a leader writing
        s_iscalars[0] for a barrier-published broadcast. ``wcnt_off``
        parity-banks the smem_wcnt staging so a warp that has moved on
        to the next Phase-2 round cannot clobber a slot a slower warp is
        still reading (the per-round staging barrier bounds the drift to
        one round).

        Vectorized scan: each thread loads vec_w elements per iter (128 or
        256 bits) over ``input_row[slice_start : slice_end)``; scalar tail
        handles the remainder.

        Cluster aggregation (cluster_size > 1): every CTA stages its
        slice-local count into ``s_cluster_partial[call & 1]`` (parity
        double-buffer; slot 2 is the tid0-private call counter), syncs the
        cluster, then DSMEM-reads every peer's slot and sums into
        ``s_iscalars[0]``.
        After this every CTA's ``s_iscalars[0]`` holds the same
        cluster-wide cand_count, so Phase 2's secant update stays a
        leader-only scalar op on a value all CTAs agree on.
        """
        ...

    @cute.jit
    def phase2_secant_search(
        self,
        input_row,
        N,
        slice_start,
        slice_end,
        smem_ptcnt,
        smem_wcnt,
        s_thr,  # [threshold, val_lo, val_hi]
        s_iscalars,  # [cand_count, done, cnt_lo, cnt_hi, out_count]
        s_cluster_partial,  # [3] int32 cluster scratch (parity slots + counter)
        tidx,
        warp_id,
        lane,
        do_cluster_sync,  # bool: False = cs=1 / short-row degrade (skip cluster sync)
        smem_input=None,  # optional SMEM-cached slice
    ):
        """Refine s_thr[0] until cand_count lands in [kK, kCC].

        Each iter calls block_count_ge at the candidate threshold and
        updates the bracket (val_lo, val_hi, cnt_lo, cnt_hi). Sets
        s_iscalars[1] (done) = 1 on convergence, 2 on bracket exhaustion.
        """
        ...

    @cute.jit
    def phase3_collect_candidates(
        self,
        input_row,
        N,
        slice_start,
        slice_end,
        smem_keys,
        smem_vals,
        smem_ptcnt,
        smem_wcnt,
        s_thr,
        s_iscalars,
        s_cluster_partial,
        tidx,
        warp_id,
        lane,
        do_cluster_sync,  # bool: False = cs=1 / short-row degrade (skip cluster sync)
        smem_input=None,  # optional SMEM-cached slice
    ):
        """Retry-shrink (when P2 didn't converge) + prefix sum + stream-write.

        On exit, smem_keys[0 : cand_count] / smem_vals[0 : cand_count]
        hold every (value, index) pair with value >= threshold, in the
        scan order each thread produces them. Uses smem_ptcnt cached by
        the last block_count_ge in Phase 2 (or by the retry-shrink below).
        """
        ...

    @cute.jit
    def block_fused_snap_iter(
        self,
        keys_base,  # hoisted SMEM window base of smem_keys (iterator.toint())
        smem_wcnt,
        smem_hist,  # reused as scratch for s_up/s_down warp aggregates
        s_thr,
        s_iscalars,
        count,
        tidx,
        warp_id,
        lane,
    ):
        """One iteration of histogram snap. Updates s_iscalars[2]=cnt_lo (cge),
        s_iscalars[3]=cnt_hi (cgt), and s_thr[0]=threshold (moves toward
        the cnt-in-(kK_GT, kK_GE) bracket).
        """
        ...

    @cute.jit
    def _hist_build(self, keys_base, smem_hist, cand_count, lo, inv, tidx):
        """Zero smem_hist[0:kBins], then histogram keys[0:cand_count] with
        bin = clamp(int((v - lo) * inv), 0, kBins-1). Out-of-window values
        clamp into the edge bins, which keeps cumulative counts from the
        top exact for the k-th search (everything above the window lands
        in the top bin). Barrier after the zero pass and after the build."""
        ...

    @cute.jit
    def _kth_bin_search(
        self, smem_hist, smem_wcnt, s_thr, s_iscalars, lo, binw, tidx, warp_id, lane
    ):
        """Parallel k-th bin search (3-step, high→low). Writes
        s_thr[0] = lower edge of the selected bin (lo + bidx*binw) and
        s_iscalars[4] = selected bin's count (gates the level-2 histogram
        refinement). Clobbers s_iscalars[2]/[3] as staging (both are
        rewritten by the snap loop before anyone else reads them).
        Trailing barrier."""
        ...

    @cute.jit
    def phase4_rank_scatter(
        self,
        ...

    @cute.jit
    def phase4_histogram_snap(
        self,
        smem_keys,
        smem_vals,
        smem_hist,
        smem_wcnt,
        s_thr,
        s_iscalars,
        output_values_row,
        output_indices_row,
        cand_count,
        tidx,
        warp_id,
        lane,
    ):
        """Three branches by cand_count vs kK:
        == kK: direct emit (fast path)
        >  kK: histogram k-th bin search → snap → 2-pass writeback
        <  kK: emit cand_count + pad with -FLT_MAX
        """
        ...

    @cute.kernel
    def gvr_topk_kernel(
        self,
        input_data: cute.Tensor,  # [numRows, stride0] dtype
        pre_idx: cute.Tensor,  # [numRows / next_n, pre_idx_stride] int32
        seq_lens: cute.Tensor,  # [numRows / next_n] int32
        output_values: cute.Tensor,  # [numRows, top_k] dtype
        output_indices: cute.Tensor,  # [numRows, top_k] int32
        order_row: cute.Tensor,  # [batch_size] int32 (or None when seqlen_sorted=False)
    ):
        """Thin entry: bidx → row_idx → run_one_row.

        grid = (num_rows * cluster_size,) where num_rows = batch_size *
        next_n. cluster_id = bidx // cluster_size, cta_in_cluster ∈
        [0, cluster_size). CTA r scans row[r * N / cs : (r+1) * N / cs]
        in Phase 2, so the per-row GE-count scales as 1 / cs. At
        cluster_size == 1 this collapses to one CTA per row scanning
        the whole row.

        When ``self.seqlen_sorted`` is True, the LJF dispatch order
        operates at REQUEST granularity (``order_row`` has length
        batch_size = num_rows / next_n). The owning row is resolved as
        ``order_row[cluster_id // next_n] * next_n + cluster_id % next_n``
        so the ``next_n`` rows of one request stay contiguous in
        dispatch order. All ``cluster_size`` CTAs within a cluster see
        the same ``cluster_id`` and therefore the same row, preserving
        cluster-sync semantics.

        Body is extracted into :meth:`run_one_row` so other entries (e.g.
        the LB load-balance variant) can resolve ``row_idx`` differently
        from the mappings used here.
        """
        ...

    @cute.jit
    def run_one_row(
        self,
        row_idx,  # int32, owning row in [0, num_rows)
        input_data: cute.Tensor,  # [numRows, stride0] dtype
        pre_idx: cute.Tensor,  # [numRows / next_n, pre_idx_stride] int32
        seq_lens: cute.Tensor,  # [numRows / next_n] int32
        output_values: cute.Tensor,  # [numRows, top_k] dtype, optional
        output_indices: cute.Tensor,  # [numRows, top_k] int32
    ):
        """Dispatch: compute per-row slice + cluster sync mode, call _run_phases.

        ``run_one_row`` only handles row resolution, SMEM allocation, and
        the per-row long-vs-short decision. Phase 1-4 are in
        :meth:`_run_phases`.

        Short-row degrade: when the actual row workload fits within ONE
        CTA's design slice (``ceil(max_seq_len / cluster_size)``), CTA 0
        solo-scans the row (do_cluster_sync=False, no cluster sync) and
        the other cluster CTAs fall through ``run_one_row`` without
        calling ``_run_phases``. CuTe DSL doesn't support runtime
        ``return``, so non-leader CTAs naturally reach
        ``griddepcontrol_launch_dependents`` at the end.
        """
        ...

    @cute.jit
    def _run_phases(
        self,
        input_row,
        pre_idx_row,
        output_values_row,
        output_indices_row,
        N,
        pre_idx_offset,
        pre_idx_count,
        slice_start,
        slice_end,
        do_cluster_sync,
        cta_in_cluster,
        smem_keys,
        smem_vals,
        smem_hist,
        smem_ptcnt,
        smem_wcnt,
        smem_wmin,
        smem_wmax,
        smem_wsum,
        smem_wcnt_p1,
        s_thr,
        s_iscalars,
        s_cluster_partial,
        smem_input,
        s_mt_thr,
        smem_ptcnt_multi,
        smem_wcnt_multi,
        s_mt_cnt,
        s_r0col,
        s_cluster_partial_m,
        smem_gath,
        tidx,
        warp_id,
        lane,
    ):
        """Run Phase 1-4 + final cluster barrier on a given row slice.

        Caller (``run_one_row``) decides slice + do_cluster_sync per row:
          - cs=1                 → slice=[0,N), do_cluster_sync=False
          - cs>1, long row       → slice=N/cs per CTA, do_cluster_sync=True
          - cs>1, short row      → slice=[0,N), do_cluster_sync=False, CTA 0 only

        Non-leader CTAs in short-row mode never call this helper.
        """
        ...

    def pick_config(
        torch_dtype,
        num_rows: int,
        num_candidates: int,
        max_seq_len: Optional[int] = None,
        num_sms: Optional[int] = None,
    ) -> dict:
        """Pick the launch-shape ctor kwargs for ``(dtype, BS, N)``.

        Mirrors the production runner policy (cluster_size auto-pick +
        ``_pick_tuning``) so any caller instantiating the kernel directly
        gets the same shapes the custom op would use. Rationale (B200,
        nsys cold-L2, 2026-07-15 big-BS triage): a config frozen at the
        BS=1 optimum (cs = N>=65536 ? 4 : 1, T=1024, mbpm=1) is geomean
        2.27x slower (max 6.0x) than the op-bench anchor at BS in
        {64, 256, 1024}, while this policy is 0.95x (parity/better).
        Multi-CTA splitting only pays while the grid is a single wave
        (num_rows * cluster_size <= num_sms); past that, row parallelism
        already saturates the SMs and per-row splitting is pure overhead.

        ``max_seq_len``: pass the peak runtime N under CUDA-graph capture
        so the variant is picked for the replay shape, not the capture
        shape (same contract as the custom op's ``_pick_tuning``).

        Returns kwargs for ``GvrTopKKernel(...)``: ``cluster_size``,
        ``num_threads``, ``use_256bit_load``, ``min_blocks_per_mp``,
        ``enable_warp_parallel_reduce``.
        """
        import torch  # local: keep the module importable without torch

        if num_sms is None:
            num_sms = GvrTopKKernel._device_num_sms()
        n_row = max_seq_len if max_seq_len is not None else num_candidates
        is_fp32 = torch_dtype == torch.float32

        # cluster_size: B200 SXM5 synth-data tuning (matches the custom
        # op's auto-pick): N < 64K -> 1 (sync unrecouped); tiny grid at
        # large N -> 8; single-wave -> 4/2; multi-wave -> 1.
        if n_row < 65536:
            cluster_size = 1
        elif num_rows <= 4 and n_row >= 131072:
            cluster_size = 8
        elif num_rows * 4 <= num_sms:
            cluster_size = 4
        elif num_rows * 2 <= num_sms:
            cluster_size = 2
        else:
            cluster_size = 1

        # Cluster CTAs split the row, so tuning targets per-CTA work.
        n_per_cta = n_row // cluster_size
        # T=1024 needs 1 CTA/SM grid AND enough per-CTA vec work. Under
        # graph capture, raise the half-prec bar so a small capture-N
        # doesn't force T=1024 on small-N replays.
        n_thresh_t = 131072 if (max_seq_len is not None and not is_fp32) else 65536
        num_threads = 1024 if (num_rows <= num_sms and n_per_cta >= n_thresh_t) else 512
        # V=256-bit only helps fp32 at large N; half-prec cvt doubles reg
        # pressure. Caller must hand a 32B-aligned contiguous tensor
        # (``launch`` downgrades on misalignment).
        use_256bit_load = is_fp32 and n_per_cta >= 16384
        enable_warp_parallel_reduce = num_threads == 1024

        # min_blocks_per_mp: reg-vs-occupancy 3-tier (fp32 wants ~70 regs
        # for 4-LDG ILP -> mb<=2; half-prec fits 40 regs -> mb=3 packs
        # 3 CTA/SM when rows oversubscribe the device).
        vec_bits = 256 if use_256bit_load else 128
        vec_w = vec_bits // (32 if is_fp32 else 16)
        n_vec_iters = max(1, n_per_cta // (num_threads * vec_w))
        if is_fp32:
            if n_vec_iters < 4:
                min_blocks_per_mp = 0
            elif num_rows <= num_sms:
                min_blocks_per_mp = 1
            elif num_sms * 2 < num_rows <= num_sms * 3 and n_per_cta <= 32768:
                min_blocks_per_mp = 3
            else:
                min_blocks_per_mp = 2
        else:
            if num_rows > num_sms:
                min_blocks_per_mp = 3
            elif n_vec_iters < 4:
                min_blocks_per_mp = 0
            else:
                min_blocks_per_mp = 1

        return dict(
            cluster_size=cluster_size,
            num_threads=num_threads,
            use_256bit_load=use_256bit_load,
            min_blocks_per_mp=min_blocks_per_mp,
            enable_warp_parallel_reduce=enable_warp_parallel_reduce,
        )

    @classmethod
    def launch(
        cls,
        logits,
        pre_idx,
        seq_lens,
        output_indices,
        top_k: int,
        next_n: int = 1,
        compress_ratio: int = 1,
        max_seq_len: Optional[int] = None,
        num_sms: Optional[int] = None,
        **kernel_overrides,
    ) -> None:
        """Compile-and-launch with ``pick_config`` shapes (indices-only path).

        Owns a class-level compiled-variant cache keyed by every ctor knob,
        so repeated calls at any (BS, N, dtype) reuse the right variant.
        ``kernel_overrides`` (e.g. ``enable_r0=False``, ``cluster_size=8``)
        override the picked config and participate in the cache key.
        Mirrors the custom op's compile contract: sym_int shapes, tvm-ffi
        env stream (launches on the ambient torch stream), fixed
        ``return_output_values=False`` / ``seqlen_sorted=False``.
        """
        ...

```