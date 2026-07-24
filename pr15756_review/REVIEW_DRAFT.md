# PR #15756 Review Draft — NOT YET POSTED

PR: `[None][feat] Improve cute dsl radix top-k` (limin2021)
Head reviewed: `c67fa93aa25c8ef24000385903cc555e36ae3574` (2026-07-24)
Reviewer: loncheng. Verdict proposal: **Request changes** on M2 (verified defect,
small fix) + clarifying question M1; everything else comment-level.

M2 was empirically verified on umbriel-b200-074 (B200, 148 SMs) against the PR
head kernel code loaded standalone (`repro_truncate_dup_indices.py` in this dir;
run with the mini-package harness under the session scratchpad, or rebuild:
symlink `blackwell/utils.py` + `top_k/{block_scan,filtered_top_k_varlen_util,
filtered_top_k_decode_varlen}.py` into `pk/` and `pk/top_k/`).

---

## M2 — VERIFIED DEFECT: TRUNCATE emits duplicate indices when top_k > smem_input_size

**File**: `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/filtered_top_k_varlen_util.py`
**Anchors**: L135 (TRUNCATE docstring), L517 (`elif cutlass.const_expr(self.enable_truncate)`)

Draft comment:

> I verified TRUNCATE behavior on B200 against this PR head with an adversarial
> input, and found a case where it violates the contract asserted by this PR's
> own test (`_compare_truncate_result` check 3, "no duplicate indices"):
>
> - Config: fp32, num_tokens=32768, **top_k=16384**, large_occupancy=True,
>   overflow_policy=TRUNCATE (S = filtered_topk_smem_input_size = 8192).
> - Input: all-identical logits (every element lands in one coarse bin, so the
>   threshold bin holds all N elements and truncation drops all but S).
> - Result: all 16384 output slots are written and in-range, but exactly **8192
>   are duplicates** — the S retained candidates are each emitted twice. No -1
>   padding, no "fewer than top_k" as the docstring suggests; downstream code
>   would silently attend duplicate columns and lose 8192 real candidates.
> - Same config with top_k=2048 is correct even on the all-tie input, and
>   REREAD is correct on both. So the trigger is specifically
>   `top_k > filtered_topk_smem_input_size` (total retained candidates <
>   topk_remaining), reachable within the supported top_k<=16384 range for fp32
>   large_occupancy (S=8192) and others.
>
> Since TRUNCATE is not the default and production uses top_k=2048, this is not
> release-blocking for the default path, but the failure mode (silent duplicate
> indices rather than the documented "may output fewer than top_k") is worth
> closing. Suggest either:
> 1. assert/raise at construction when `overflow_policy == "TRUNCATE" and
>    top_k > filtered_topk_smem_input_size`, or
> 2. initialize the unfilled `s_indices` tail to -1 in the TRUNCATE path and
>    update the docstring to describe the real failure mode.
>
> Repro script available on request (standalone, ~100 lines, runs in <1 min).

## M1 — QUESTION: warmup removal vs. runtime JIT compile stalls on non-captured shapes

**File**: `tensorrt_llm/_torch/attention_backend/sparse/dsa.py`
**Anchor**: L1813 (`# No explicit CuTe DSL top-k warmup needed: ...`)

Draft comment:

> The rationale covers CUDA-graph-captured geometries (WARMUP_STEPS runs a full
> forward per capture geometry before capture). But the deleted
> `warmup_cute_dsl_indexer_topk` also pre-compiled every power-of-2
> `bucketed_num_cols` in [2^10, 2^18], which protected the **non-captured**
> paths:
>
> 1. eager fallback (batch > max capture bs, or cuda_graph disabled), where a
>    new bucket is first hit mid-serving when context length crosses a
>    power-of-2 boundary — cuteDSL JIT compile is O(seconds) and would stall a
>    live decode step;
> 2. the new dispatch adds `cluster_size` (a function of runtime
>    num_rows/num_tokens) to the compile key, so eager mode can now generate
>    more distinct kernel variants at runtime than the old code did.
>
> Could you confirm how first-hit compiles are handled on the non-captured
> paths — is there an existing eager warmup that covers this, or should a
> slimmed-down warmup (per-policy, per-bucket, plus the auto_cluster_size
> ladder outputs) be kept for the non-graph case?

## M3 — API: three overlapping dispatch flags with implicit precedence

**File**: `tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py`
**Anchor**: L6568–L6580 (`cute_dsl_indexer_topk_decode` signature,
`radix_filter_single_pass_multi_cta: bool = True`)

Draft comment:

> `cute_dsl_indexer_topk_decode` now has three overlapping mode booleans with
> precedence implied only by if/elif ordering:
> `radix_filter_single_pass_multi_cta` (default True) >
> `single_pass_multi_cta` > 2-pass. A caller that today explicitly passes
> `single_pass_multi_cta=True` gets silently rerouted to the new radix-filter
> path unless they also pass `radix_filter_single_pass_multi_cta=False`.
> In-tree callers use defaults so nothing breaks here, but suggest either a
> single `mode: str` parameter, or at minimum documenting the precedence in
> the op docstring and asserting on conflicting combinations.

## m4 — prefill DSL op has no production caller; 262K x 2048 is 0.41x

**File**: `tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py`
**Anchor**: L5734 (`def cute_dsl_indexer_topk_prefill_blackwell`)

Draft comment:

> `cute_dsl_indexer_topk_prefill_blackwell` is registered and tested but not
> wired into dsa.py — prefill production dispatch still uses the CUDA
> `indexer_topk_prefill`. Worth stating in the PR description that prefill
> integration is a follow-up. Also, per your own table the DSL prefill is
> 0.41x (2.4x slower) at N=262144 / top_k=2048, so the eventual dispatch will
> need an N-gate; a short note in the op docstring would help whoever wires it.

## m5 — heuristic: exact `num_tokens == 32768` equality vs. bucketing; B200-randn tuning

**File**: `tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py`
**Anchor**: L6540–L6559 (`_radix_select_preferred`)

Draft comment:

> Two small points on `_radix_select_preferred`:
> 1. `num_tokens == 32768` is an exact equality on the raw shape[1], while the
>    kernel itself buckets via next_positive_power_of_2 — num_tokens in
>    (32768, 65536) with e.g. 32769 falls through to the filter path even
>    though it compiles the same 65536-bucket kernel; if the intent is
>    per-bucket routing this should compare the bucketed value (and if the
>    intent really is exact-32768 only, a comment would help).
> 2. The constants (num_rows >= 74, the 32768 split) are tuned on B200 randn
>    fixed-length inputs; the docstring caveat is appreciated — consider
>    upgrading it to a TODO to re-validate on real (concentrated) indexer
>    logit distributions, which in our experience can flip synth-tuned
>    routing verdicts.

## nits (bundle into one comment or skip)

- `filtered_top_k_decode_varlen.py` `_get_num_sms` caches on first call via a
  function attribute — wrong result in a heterogeneous multi-GPU process
  (rare; fine to leave with a comment).
- Leftover `# TODO: rename, CuteDSLTopKDecodeRadixFilterSPMultiCTARunner ...`
  and `# TODO: move this if to line 411-412?` in shipped code.
- 131 commits — squash on merge.

## Positive notes to include in the review summary

- `cluster_arrive_relaxed()` -> `cluster_arrive()` release-fence fix at DSMEM
  publish sites is correct and matches a race we independently root-caused in
  GVR work (relaxed arrive has no release semantics; peer
  `ld.shared::cluster` can read a stale histogram). Keeping WAR/liveness
  sites relaxed is also right.
- Perf methodology (CUPTI, cold-L2 8GB flush, median-of-40, full 60-config
  table including the regressing cells) is exemplary.
- REREAD default removes the O(num_gen_tokens * kv_len) scratch, making the
  dsa.py 256-token cap removal sound, and survives the all-tie adversarial
  input exactly (verified alongside M2).
- Test design: group2-heavy tie test, solo/degrade short-row cluster
  deadlock-safety test, REREAD no-overflow branch coverage; deleted tests
  were F811-shadowed duplicates (dead).
- Occupancy-aware SMEM sizing with the self-documented large-top_k L1 caveat.

## Empirical evidence log

```
umbriel-b200-074, torch 2.12.0a0+nv26.05, PR head c67fa93aa2
fp32 N=32768 top_k=16384 batch=4 large_occupancy=True  (S=8192, Uint16, nb=2)
TRUNCATE / all-tie : 4/4 rows BAD  (valid=16384, dups=8192, no -1, no OOR)
TRUNCATE / randn   : 4/4 rows OK
REREAD   / all-tie : 4/4 rows OK
REREAD   / randn   : 4/4 rows OK
fp32 N=32768 top_k=2048  (same S): all policies OK on both inputs
```
