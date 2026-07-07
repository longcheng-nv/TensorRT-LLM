# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PR-1 runner extension DRAFT for tensorrt_llm/_torch/custom_ops/
cute_dsl_custom_ops.py — NOT importable standalone (relative imports).

Insertion point: inside the ``if IS_CUTLASS_DSL_AVAILABLE:`` block, directly
after the existing ``CuteDSLGvrTopKDecodeRunner`` section (origin/main
~L6045-6560). The ``# >>> INSERT`` / ``# <<< INSERT`` markers delimit the
code to paste; everything referenced (``logger``, ``_get_num_sms``,
``_query_max_cluster_size``, ``_TORCH_TO_CUTLASS_DTYPE``, ``is_sm_100f``,
``get_sm_version``, ``cute``, ``cutlass``, ``torch``, ``Optional``) already
exists in that scope.

Design notes (op21 campaign → production):
- OPT-IN sibling path: new custom op ``trtllm::cute_dsl_gvr_topk_decode_ms``
  next to the classic op; nothing routes to it by default (default flip is
  PR-3 per UPSTREAM_ASSESSMENT §5).
- Non-LB / non-sort first step: no ``order_row`` / ``counters`` support;
  mixed long/short batches keep using the classic runner (op#9 lesson:
  complex dispatchers lose; keep LB orthogonal).
- Tuning (T / 256-bit / min_blocks) is the op21 bench ``_config`` verbatim —
  every SHIP_REVIEW verdict table was measured with exactly these rules.
- Cluster policy is op21 ``gvr_ms_auto`` verbatim (3 rules + hw clamp). All
  dispatch keys are capture-time constants (dtype, K, buffer max-N, BS) —
  CUDA-graph replay identity is preserved by construction.
- Env knobs from the bench (OP21_*) became constructor defaults; the
  A/B degrees of freedom stay compiled-in (p4_rank_scatter / p4_smallbin /
  p2_native / p3_push default True = the shipped configuration).
"""

# >>> INSERT (inside `if IS_CUTLASS_DSL_AVAILABLE:` after the GVR section)

    # ------------------------------------------------------------------ #
    #  CuTe DSL GVR-MS (multi-threshold sandwich) Top-K Decode            #
    # ------------------------------------------------------------------ #
    from ..cute_dsl_kernels.blackwell.top_k.gvr_topk_decode import \
        GvrParams as _GvrParams
    from ..cute_dsl_kernels.blackwell.top_k.gvr_topk_decode_ms import \
        GvrMsClusterKernel as _GvrMsClusterKernel
    from ..cute_dsl_kernels.blackwell.top_k.gvr_topk_decode_ms import \
        GvrMsKernel as _GvrMsKernel

    class CuteDSLGvrTopKDecodeMsRunner:
        """Runner for the GVR-MS Top-K kernel variant (Blackwell SM100).

        Opt-in sibling of :class:`CuteDSLGvrTopKDecodeRunner` (see
        ``gvr_topk_decode_ms.py`` for the algorithm). Single-CTA and
        row-chunked cluster paths only — no sort-indirect, no LB; route
        mixed long/short batches through the classic runner.
        """
        kernel_cache: dict = {}

        _DTYPE_NAME = {
            torch.float32: "float32",
            torch.bfloat16: "bfloat16",
            torch.float16: "float16",
        }

        @staticmethod
        def _pick_tuning(
            torch_dtype: torch.dtype,
            num_rows: int,
            N_row: int,
            num_sms: int,
            data_ptr: int,
        ) -> dict:
            """op21 bench ``_config`` verbatim: T=1024 needs a one-wave grid
            AND enough per-CTA vector work; 256-bit loads pay from N>=16K
            (both precisions — the 16-bit ladder loads packed u32 pairs);
            min_blocks 1/3 by one-wave vs multi-wave."""
            num_threads_per_block = (1024 if
                                     (num_rows <= num_sms and N_row >= 65536)
                                     else 512)
            use_256bit_load = N_row >= 16384
            if use_256bit_load:
                assert data_ptr % 32 == 0, (
                    f"use_256bit_load=True requires 32B-aligned "
                    f"logits.data_ptr(), got {data_ptr} % 32 = "
                    f"{data_ptr % 32}.")
            min_blocks_per_mp = 1 if num_rows <= num_sms else 3
            return dict(
                num_threads_per_block=num_threads_per_block,
                use_256bit_load=use_256bit_load,
                min_blocks_per_mp=min_blocks_per_mp,
            )

        @staticmethod
        def _pick_cluster(
            torch_dtype: torch.dtype,
            top_k: int,
            N_row: int,
            num_rows: int,
            num_sms: int,
        ) -> int:
            """op21 ``gvr_ms_auto`` dispatch (B200 nsys cold-L2 tuning,
            2026-07-05/06; HW-invariant on B300, NUM_SMS=148 both):

              16-bit AND N>=64K AND N>=32768*BS -> 8  (halved scan re-weights
                                                       the chunk tail; win
                                                       region measured exactly)
              K>=2048 AND N>=192K AND BS<=4     -> 8  (fp32 huge-N tiny-grid)
              N>=64K AND 4*BS<=SMs              -> 4  (one-wave aggregate BW)
              else                              -> 1
            """
            dt16 = torch_dtype in (torch.bfloat16, torch.float16)
            if dt16 and N_row >= 65536 and N_row >= 32768 * num_rows:
                return 8
            if top_k >= 2048 and N_row >= 196608 and num_rows <= 4:
                return 8
            if N_row >= 65536 and num_rows * 4 <= num_sms:
                return 4
            return 1

        @classmethod
        def _compile(
            cls,
            dtype,
            torch_dtype: torch.dtype,
            top_k: int,
            next_n: int,
            compress_ratio: int,
            cluster_size: int,
            fuse_collect: bool,
            num_threads_per_block: int,
            use_256bit_load: bool,
            min_blocks_per_mp: int,
        ) -> tuple:
            key = ("ms", dtype, top_k, next_n, compress_ratio, cluster_size,
                   fuse_collect, num_threads_per_block, use_256bit_load,
                   min_blocks_per_mp)
            if key in cls.kernel_cache:
                return key
            n_rows = cute.sym_int()
            n_cols = cute.sym_int()
            n_batch = cute.sym_int()
            in_align = 32 if use_256bit_load else 16
            input_fake = cute.runtime.make_fake_compact_tensor(
                dtype, (n_rows, n_cols),
                stride_order=(1, 0),
                assumed_align=in_align)
            pre_idx_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (n_batch, top_k),
                stride_order=(1, 0),
                assumed_align=16)
            seq_lens_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (n_batch, ), stride_order=(0, ))
            out_indices_fake = cute.runtime.make_fake_compact_tensor(
                cutlass.Int32, (n_rows, top_k),
                stride_order=(1, 0),
                assumed_align=16)
            fake_stream = cute.runtime.make_fake_stream(
                use_tvm_ffi_env_stream=True)
            common = dict(
                dtype=dtype,
                top_k=top_k,
                next_n=next_n,
                num_threads=num_threads_per_block,
                compress_ratio=compress_ratio,
                use_256bit_load=use_256bit_load,
                min_blocks_per_mp=min_blocks_per_mp,
                return_output_values=False,
            )
            if cluster_size > 1:
                # Cluster path always fuses (assert in the kernel ctor);
                # per-thread slot overflow falls back in-kernel to the
                # leader-exact classic path.
                kernel = _GvrMsClusterKernel(cluster_size=cluster_size,
                                             **common)
            else:
                kernel = _GvrMsKernel(fuse_collect=fuse_collect, **common)
            cls.kernel_cache[key] = cute.compile(
                kernel,
                input_fake,
                pre_idx_fake,
                seq_lens_fake,
                None,
                out_indices_fake,
                stream=fake_stream,
                options="--enable-tvm-ffi",
            )
            logger.debug(f"[compile cute_dsl gvr_topk_decode_ms] {key}")
            return key

        @classmethod
        def forward(
            cls,
            logits: torch.Tensor,
            pre_idx: torch.Tensor,
            seq_lens: torch.Tensor,
            output_indices: torch.Tensor,
            top_k: int,
            next_n: int = 1,
            compress_ratio: int = 1,
            max_seq_len: Optional[int] = None,
            cluster_size: Optional[int] = None,
        ) -> None:
            cute_dtype = _TORCH_TO_CUTLASS_DTYPE[logits.dtype]
            num_rows = logits.shape[0]
            assert num_rows % next_n == 0 and seq_lens.shape[
                0] == num_rows // next_n, (
                    f"shape contract: seq_lens.shape[0] (={seq_lens.shape[0]}) "
                    f"must equal logits.shape[0] / next_n "
                    f"(={num_rows} / {next_n} = {num_rows // next_n})")
            N_row = max_seq_len if max_seq_len is not None else logits.shape[1]
            num_sms = _get_num_sms()

            if cluster_size is None:
                cluster_size = cls._pick_cluster(logits.dtype, top_k, N_row,
                                                 num_rows, num_sms)
            assert cluster_size in (1, 2, 4, 8), (
                f"cluster_size must be 1, 2, 4 or 8; got {cluster_size}")
            if cluster_size > 1:
                hw_max_cluster = _query_max_cluster_size()
                if cluster_size > hw_max_cluster:
                    logger.warning_once(
                        f"cute_dsl_gvr_topk_decode_ms: cluster_size="
                        f"{cluster_size} exceeds device max "
                        f"({hw_max_cluster}); clamping.",
                        key="cute_dsl_gvr_topk_decode_ms_cluster_clamp",
                    )
                    cluster_size = hw_max_cluster

            tuning = cls._pick_tuning(logits.dtype, num_rows, N_row, num_sms,
                                      logits.data_ptr())
            # Fused P2+P3 slot collect (single-CTA): one-wave residency for
            # the slot smem AND the spec-collect buffer holds >= 4x K.
            params = _GvrParams.get(cls._DTYPE_NAME[logits.dtype], top_k,
                                    compress_ratio)
            fuse_collect = bool(num_rows <= num_sms
                                and 4 * top_k <= params.kC)
            if cluster_size > 1:
                # Cluster tuning is fixed (op21 bench msc conventions): the
                # C CTAs of one row split the scan, T=1024, one-wave grid.
                tuning = dict(num_threads_per_block=1024,
                              use_256bit_load=tuning["use_256bit_load"],
                              min_blocks_per_mp=1)
            key = cls._compile(
                cute_dtype,
                logits.dtype,
                top_k,
                next_n,
                compress_ratio=compress_ratio,
                cluster_size=cluster_size,
                fuse_collect=fuse_collect,
                **tuning,
            )
            cls.kernel_cache[key](logits, pre_idx, seq_lens, None,
                                  output_indices)

    @torch.library.custom_op("trtllm::cute_dsl_gvr_topk_decode_ms",
                             mutates_args=("output_indices", ),
                             device_types="cuda")
    def cute_dsl_gvr_topk_decode_ms(
        logits: torch.Tensor,
        pre_idx: torch.Tensor,
        seq_lens: torch.Tensor,
        output_indices: torch.Tensor,
        top_k: int,
        next_n: int = 1,
        compress_ratio: int = 1,
        max_seq_len: Optional[int] = None,
        cluster_size: Optional[int] = None,
    ) -> None:
        """CuTe DSL GVR-MS Top-K decode (opt-in variant, Blackwell).

        Same operator contract as ``trtllm::cute_dsl_gvr_topk_decode``
        (indices-only, exact top-K with pre_idx seeding), minus the
        sort-indirect / LB modes — pass those batches to the classic op.
        See ``gvr_topk_decode_ms.py`` for the algorithm and the measured
        kernel-level gains (nsys cold-L2 geomean 1.14x fp32 / 1.29x bf16 /
        1.27x fp16 vs the classic GVR kernel on the B200/B300 P0 grid).

        Args:
            logits: ``[num_rows, max_seq_len]`` fp32 / bf16 / fp16.
            pre_idx: ``[num_rows // next_n, top_k]`` int32.
                ``pre_idx[..., 0]`` must be the argmax index.
            seq_lens: ``[num_rows // next_n]`` int32, request-level
                (uncompressed space; kernel divides by compress_ratio).
            output_indices: ``[num_rows, top_k]`` int32.
            top_k: K in {512, 1024, 2048} — compile-time specialized.
            next_n: MTP temporal stride (cr=1 adds the per-row diagonal
                ``preIdxOffset = row % next_n + 1``).
            compress_ratio: 1 = DSv3.2, 4 = DSv4.
            max_seq_len: Peak N at replay; pass under CUDA Graph capture
                so tuning + cluster dispatch bind to the peak shape.
            cluster_size: None = auto (op21 policy); 1/2/4/8 to pin.
        """
        if not is_sm_100f():
            raise ValueError(
                f"CuteDSL: SM version {get_sm_version()} is not supported. "
                f"CuteDSL GVR-MS Top-K Decode only supports SM 100 family.")
        if logits.shape[0] % next_n != 0:
            raise ValueError(
                f"logits.shape[0] (={logits.shape[0]}) must be divisible by "
                f"next_n (={next_n}); kernel derives batch_size as "
                f"logits.shape[0] / next_n.")
        CuteDSLGvrTopKDecodeMsRunner.forward(
            logits=logits,
            pre_idx=pre_idx,
            seq_lens=seq_lens,
            output_indices=output_indices,
            top_k=top_k,
            next_n=next_n,
            compress_ratio=compress_ratio,
            max_seq_len=max_seq_len,
            cluster_size=cluster_size,
        )

    @torch.library.register_fake("trtllm::cute_dsl_gvr_topk_decode_ms")
    def _(
        logits: torch.Tensor,
        pre_idx: torch.Tensor,
        seq_lens: torch.Tensor,
        output_indices: torch.Tensor,
        top_k: int,
        next_n: int = 1,
        compress_ratio: int = 1,
        max_seq_len: Optional[int] = None,
        cluster_size: Optional[int] = None,
    ) -> None:
        return None

# <<< INSERT
