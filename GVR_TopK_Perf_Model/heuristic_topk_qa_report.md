# Heuristic-Guided Top-K: Q&A on Algorithm Pseudocode & GPU Performance Model

> **Reference documents**:
> - [`heuristic_topk_perf_model.md`](heuristic_topk_perf_model.md) — Algorithm Pseudocode & GPU Performance Model
> - [`blog19_Temporal_Correlation_Meets_Sparse_Attention.md`](../docs/source/blogs/tech_blog/blog19_Temporal_Correlation_Meets_Sparse_Attention.md) — Tech Blog
> - [`heuristic_topk.cuh`](../cpp/tensorrt_llm/kernels/heuristic_topk.cuh) — CUDA Kernel Source (V2e)

---

## Q1: What's the math needed by `PreIndexStats`?

**Question:**
```
(pmin, pmax, pmean) ← PreIndexStats(input, preIdx, M)
what's the math needed by PreIndexStats?
```

**Answer:**

`PreIndexStats` computes three statistics — **min**, **max**, and **mean** — over the `M = 2048` input values at the positions specified by `preIdx`. From `heuristic_topk.cuh` lines 270–325:

```cuda
float local_min = FLT_MAX;
float local_max = -FLT_MAX;
float local_sum = 0.0f;
int local_cnt = 0;
for (int i = tid; i < M; i += BLOCK_SIZE)
{
    int idx = __ldg(&preIdx[i]) + preIdxOffset;
    if (idx >= 0 && idx < N)
    {
        float v = __ldg(&input[idx]);
        local_min = fminf(local_min, v);
        local_max = fmaxf(local_max, v);
        local_sum += v;
        local_cnt++;
    }
}
// ... warp reductions, cross-warp reductions ...
float pmean = (pcnt > 0) ? psum / (float) pcnt : (pmin + pmax) * 0.5f;
```

**Per-element math** (M = 2048 elements total, distributed across P = 512 threads, ~4 elements/thread):

| Instruction | Count per element | Purpose |
|---|---|---|
| `FMNMX` | 2 | `fminf` (min) + `fmaxf` (max) |
| `FADD` | 1 | Accumulate into `local_sum` |
| `IADD` | 1 | Increment `local_cnt` |
| `ISETP` | 1 | Bounds check `idx >= 0 && idx < N` |

**Cross-thread reduction** (after the per-element loop):

- **Warp-level**: `__reduce_min_sync` and `__reduce_max_sync` (single-instruction on sm80+ via `redux.sync`), `__shfl_down_sync` × 5 rounds for sum, and `__reduce_add_sync` for count.
- **Cross-warp** (tid=0 only): 16-iteration serial loop over `warp_counts[]`, then one `FDIV` for `pmean = psum / pcnt`.

**Bottleneck**: Global memory latency (not compute). The 2048 reads are only 16 KB of data but at **random, scattered** locations via `preIdx`, so L2 hit rate depends on `preIdx` locality. The ~5 FP/INT ops per element are negligible in comparison.

---

## Q2: `CountGE` — instruction-level characterization

**Question:**
```
C ← CountGE(input, N, T)
here we basically read every element, do one FMNMX (compare) and one IADD per element.
but we compare with pmin.
```

**Answer:**

The instruction characterization is correct — each element costs **1 × FSETP** (compare `v >= threshold`) + **1 × IADD** (accumulate count), with `float4` vectorized `__ldg` loads (128B coalesced).

However, a small clarification: the comparison is **not against `pmin`**, but against the **threshold T** (which is initialized to `pmean`, not `pmin`). From `heuristic_topk.cuh` line 318 and 342:

```cuda
smem->threshold = pmean;       // line 318 — T₀ = pmean
// ...
blockCountGE(input, N, smem->threshold, smem, tid, warp_id, lane);  // line 342
```

The `pmin` and `pmax` are used to initialize the **bracket** `[val_lo, val_hi]` for the interpolation search, not as the comparison threshold. The initial threshold `T₀ = pmean` is the mean of the pre-indexed values, which (with ~50% prediction overlap) approximates the true K-th value much more closely than `pmin` would.

Inside `blockCountGE` (lines 156–184):

```cuda
for (int i = tid * 4; i + 3 < N; i += BLOCK_SIZE * 4)
{
    float4 v4 = __ldg(reinterpret_cast<float4 const*>(input + i));
    c += (v4.x >= threshold) + (v4.y >= threshold)
       + (v4.z >= threshold) + (v4.w >= threshold);
}
// OPT7: cache per-thread count for Phase 3 sub-pass 1 reuse
smem->per_thread_counts[tid] = c;
```

Per element: 1 `FSETP` + 1 `IADD`. The `float4` vectorization means 4 elements per load instruction. The loop is **purely streaming, coalesced** — bandwidth-bound on a single SM.

---

## Q3: What is `R_max`?

**Question:**
```
for r = 1 to R_max while ¬done do
    T ← InterpolateThreshold(lo, hi, cnt_lo, cnt_hi, r)
    C ← CountGE(input, N, T)
here we do scan again, what is R_max?
```

**Answer:**

From `heuristic_topk.cuh` line 58:

```cuda
constexpr int MAX_REFINE_ITERS = 15;
```

**R\_max = 15**. This is the hard iteration cap on Phase 2. In practice:

| Scenario | Typical R | Explanation |
|---|---|---|
| **Real decode data** (good temporal prediction, α ≈ 35–50%) | **1–2** | `pmean` closely approximates the true K-th value |
| **Synthetic data** (static RoPE prior) | **4–6** | Less accurate initial estimate |
| **Worst case** (adversarial distribution) | ≤ 15 | Hard cap prevents runaway; fallback to `done=2` |

Each iteration does one full-N streaming scan (`blockCountGE`), so this cap prevents worst-case global memory traffic of `15 × N × 4B`. The 15-iteration cap is essentially never hit in practice — Phase 2 converges much earlier due to the superlinear convergence of secant interpolation.

---

## Q4: `InterpolateThreshold` — cost is small (confirmed)

**Question:**
```
sounds like each function of InterpolateThreshold will do some IADD, IMUL, some IMNMX.
but they are not per element. so the cost is small.
```

**Answer:**

Correct — the cost is negligible. This function runs on **tid=0 only** (single thread), executing ~10–15 scalar instructions. From `heuristic_topk.cuh` lines 366–401:

```cuda
if (tid == 0)
{
    float vlo = smem->val_lo, vhi = smem->val_hi;
    int clo = smem->cnt_lo, chi = smem->cnt_hi;
    int target = TOP_K + SAFETY_MARGIN / 2;
    float range = vhi - vlo;
    float nv;
    if (clo > chi && range > 1e-10f)
    {
        float f = (float) (clo - target) / (float) (clo - chi);
        f = fmaxf(0.05f, fminf(0.95f, f));
        if (iter == 0) f = fminf(f, 0.50f);
        nv = vlo + range * f;
    }
    else
        nv = (vlo + vhi) * 0.5f;
    // ... boundary guards ...
    smem->threshold = nv;
}
```

One note: the operations are actually **floating-point** (`FADD`, `FMUL`, `FDIV`, `FMNMX`), not integer, since threshold `T` and the bracket `[lo, hi]` are floats. The only integer ops are `clo - target` and `clo - chi` which are converted to float via `I2F` for the division.

The total cost per iteration is ~10–15 instructions on a single thread, compared to the `blockCountGE` scan that follows (which reads all N elements with 512 threads). Completely negligible in the performance model.

---

## Q5: What is `RetryNarrow`?

**Question:**
```
Phase 3:
if done ≠ EXACT then
    C ← CountGE(input, N, T)
    RetryNarrow(input, N, T, lo, hi) while C > C_max   ▷ ≤10 retries
what is RetryNarrow?
```

**Answer:**

`RetryNarrow` is a **simple bisection narrowing loop** (up to 10 retries) that handles the edge case where Phase 2 ended with `done=2` (fallback, not clean convergence) and the candidate count exceeds `MAX_CANDIDATES = 6144`. From `heuristic_topk.cuh` lines 440–469:

```cuda
if (smem->done != 1)
{
    blockCountGE(input, N, smem->threshold, smem, tid, warp_id, lane);
    if (tid == 0 && smem->cand_count > MAX_CANDIDATES)
        smem->val_lo = smem->threshold;
    __syncthreads();

    for (int retry = 0; retry < 10 && smem->cand_count > MAX_CANDIDATES; retry++)
    {
        if (tid == 0)
        {
            float lo = smem->val_lo, hi = smem->val_hi;
            float mid = (lo + hi) * 0.5f;
            if (mid == lo) mid = hi;
            smem->threshold = mid;
        }
        __syncthreads();
        blockCountGE(input, N, smem->threshold, smem, tid, warp_id, lane);
        if (tid == 0)
        {
            int c = smem->cand_count;
            if (c > MAX_CANDIDATES) smem->val_lo = smem->threshold;
            else if (c < TOP_K)    smem->val_hi = smem->threshold;
        }
        __syncthreads();
    }
}
```

Each retry iteration:
1. Bisects: `T = (lo + hi) / 2`
2. Runs a full-N `blockCountGE` scan
3. Updates the bracket: if `count > MAX_CANDIDATES`, raise `lo`; if `count < TOP_K`, lower `hi`

This is a safety-net path — it only activates when Phase 2 converged with `done=2` (fallback) and the threshold is too loose. Each retry costs one full-N scan. **In practice, this path is rare** — the interpolation search almost always finds a valid threshold with `done=1`, in which case the entire retry block is skipped (OPT5).

---

## Q6: PrefixSum — what does this do?

**Question:**
```
offsets[] ← PrefixSum(per_thread_counts[])   ▷ OPT7: reuse cached counts
what does this do? find the prefixsum for each thread?
is this implemented with library? shuffle and smem access might be needed.
```

**Answer:**

This computes an **exclusive prefix sum** over per-thread counts so each thread knows its exact write position in shared memory. **Not a library call** — it is a hand-written two-level prefix sum. From `heuristic_topk.cuh` lines 473–503:

```cuda
// Read cached count from OPT7
int my_total_qual = smem->per_thread_counts[tid];

// Level 1: Intra-warp inclusive prefix sum via __shfl_up_sync (Hillis-Steele)
int thread_prefix = my_total_qual;
#pragma unroll
for (int off = 1; off < WARP_SIZE; off *= 2)
{
    int other = __shfl_up_sync(full_mask, thread_prefix, off);
    if (lane >= off) thread_prefix += other;
}
int my_excl_offset = thread_prefix - my_total_qual;
int warp_total_qual = __shfl_sync(full_mask, thread_prefix, WARP_SIZE - 1);

// Level 2: Inter-warp exclusive prefix sum (tid=0, serial over 16 warps)
if (lane == 0) smem->warp_counts[warp_id] = warp_total_qual;
__syncthreads();
if (tid == 0)
{
    int total = 0;
    for (int w = 0; w < NUM_WARPS; w++)
    {
        int cnt = smem->warp_counts[w];
        smem->warp_counts[w] = total;
        total += cnt;
    }
    smem->cand_count = total;
}
__syncthreads();

// Final write position for each thread
int my_write_pos = smem->warp_counts[warp_id] + my_excl_offset;
```

**Two-level structure:**

| Level | Method | Cost |
|---|---|---|
| **Intra-warp** (32 threads) | 5 rounds of `__shfl_up_sync` (Hillis-Steele parallel prefix) | 5 shuffle + 5 IADD per thread |
| **Inter-warp** (16 warps) | tid=0 serial scan over `warp_counts[0..15]` | 16 IADD + 16 SMEM read/write (trivial) |
| **Combine** | `my_write_pos = warp_prefix + thread_offset` | 1 SMEM read + 1 IADD |

Plus 2 `__syncthreads` barriers. No external library (no CUB dependency). The total cost is dominated by the 5 shuffle rounds — negligible compared to the Phase 3 collect pass that follows.

---

## Q7: `StreamCollect` — what does this line do?

**Question:**
```
StreamCollect(input, N, T, offsets)  ▷ write qualifying elements → SMEM keys[]/vals[]
I don't quite understand this line.
```

**Answer:**

`StreamCollect` is the actual data-collection pass that writes qualifying elements (value + original index) into shared memory at pre-computed positions. From `heuristic_topk.cuh` lines 505–533:

```cuda
{
    float const thr = smem->threshold;
    for (int i = tid * 4; i + 3 < N; i += BLOCK_SIZE * 4)
    {
        float4 v4 = __ldg(reinterpret_cast<float4 const*>(input + i));
#pragma unroll
        for (int j = 0; j < 4; j++)
        {
            float val = (&v4.x)[j];
            if (val >= thr && my_write_pos < MAX_CANDIDATES)
            {
                smem->keys[my_write_pos] = val;    // store value
                smem->vals[my_write_pos] = i + j;  // store original index
                my_write_pos++;
            }
        }
    }
    // ... scalar tail for remainder ...
}
```

**How it works:**

1. Each thread re-scans the full input (vectorized `float4` loads, same pattern as `blockCountGE`).
2. For every element `≥ threshold`, it writes the **value** into `smem->keys[my_write_pos]` and the **original index** into `smem->vals[my_write_pos]`, then increments `my_write_pos`.
3. Because the prefix sum (Q6) pre-computed **non-overlapping write ranges** for each thread, **no atomics or ballots are needed** — each thread writes to its own contiguous SMEM region.

**Key insight**: `my_write_pos` was initialized to the exact exclusive offset for this thread (from Q6). Since the thread processes elements in the **same order** as when it counted them in `blockCountGE`, the writes fill the pre-allocated slots perfectly. This is the "**ballot-free**" design — the write offset was pre-allocated by the prefix sum, so there is no inter-thread contention.

This is critical for performance: `__ballot_sync` acts as a compiler barrier that serializes L2 load pipelining. The original per-element ballot approach costs ~30,000 cycles (5× the `blockCountGE` cost); the ballot-free approach costs ~16,000 cycles.

---

## Q8: Histogram `atomicAdd` is still needed (confirmed)

**Question:**
```
histogram[bin] += 1               ▷ SMEM atomicAdd
atomic add is still needed.
```

**Answer:**

Correct — `atomicAdd` is still needed for the histogram build. From `heuristic_topk.cuh` lines 585–590:

```cuda
for (int i = tid; i < cand_count; i += BLOCK_SIZE)
{
    int bin = (int) ((smem->keys[i] - block_min) * inv1);
    bin = min(max(bin, 0), NUM_BINS - 1);
    atomicAdd(&smem->histogram[bin], 1);
}
```

However, contention is **low** because:

| Factor | Value | Effect |
|---|---|---|
| Total atomic ops | ≤ 6144 (C candidates) | Much less than N |
| Number of bins | 2048 (`NUM_BINS`) | Wide distribution target |
| Average collisions per bin | ~3 (6144 / 2048) | Very low contention |

Compare this to the radix-select baseline, which funnels **all N elements** (up to 128K) through atomics on fewer hot bins — orders of magnitude more contention. The histogram is also operating on **SMEM** data (not global memory), so the atomic latency is ~20–30 cycles per conflict (vs. hundreds for global atomics).

---

## Q9: What is `FusedSnapScan`?

**Question:**
```
(cnt_ge, cnt_gt, succ_up, pred_down) ← FusedSnapScan(keys, cand_count, T_sel)
what is FusedSnapScan?
```

**Answer:**

`FusedSnapScan` is `blockFusedSnapIter` in the code — a single fused scan over SMEM candidates that computes **4 quantities simultaneously**. From `heuristic_topk.cuh` lines 190–249:

```cuda
__device__ __forceinline__ void blockFusedSnapIter(
    KernelSmem* smem, int count, int tid, int warp_id, int lane)
{
    float const thr = smem->threshold;
    int lge = 0, lgt = 0;
    float s_up = FLT_MAX, s_down = -FLT_MAX;

    for (int i = tid; i < count; i += BLOCK_SIZE)
    {
        float v = smem->keys[i];
        lge += (v >= thr);           // count ≥ T
        lgt += (v > thr);            // count > T
        if (v > thr)  s_up   = fminf(s_up, v);    // nearest value above T
        if (v < thr)  s_down = fmaxf(s_down, v);  // nearest value below T
    }

    // Warp reduce + cross-warp aggregate ...
    // Convergence / snap logic:
    if (cgt >= TOP_K)        smem->threshold = total_up;    // raise T
    else if (cge < TOP_K)    smem->threshold = total_down;  // lower T
    // else: converged (cgt < K ≤ cge)
}
```

**The 4 quantities and their purpose:**

| Quantity | Name | Meaning |
|---|---|---|
| `cnt_ge` | Count ≥ T | Number of candidates at or above threshold |
| `cnt_gt` | Count > T | Number of candidates strictly above threshold |
| `succ_up` | Successor | Smallest value strictly above T (nearest distinct value upward) |
| `pred_down` | Predecessor | Largest value strictly below T (nearest distinct value downward) |

**Convergence check**: If `cnt_gt < K ≤ cnt_ge`, then T exactly equals the K-th largest value — there are enough elements ≥ T to fill K slots, and the excess consists only of ties at exactly T. Done.

**If not converged** — the threshold "snaps" to a neighboring distinct value:
- If `cnt_gt ≥ K`: too many elements above T → **raise** T to `succ_up`
- If `cnt_ge < K`: not enough elements ≥ T → **lower** T to `pred_down`

This is why it's called "snap" — it jumps directly to the next distinct data value rather than doing arithmetic interpolation. Each snap iteration reduces the set of possible thresholds by at least one distinct value, **guaranteeing convergence**. In practice, with 2048 histogram bins providing a good initial pivot, only **1–3 snap iterations** are needed.

**Per-element cost**: 2 `FSETP` + 2 `IADD` + 2 `FMNMX` = 6 cheap ops, all on SMEM data (≤ 6144 elements). No global memory access.

---

## Q10: How is `BallotWrite` implemented?

**Question:**
```
mask_gt ← ballot(keys[i] > T_sel)
BallotWrite(mask_gt, keys, vals → outVal, outIdx, out_pos)
mask_eq ← ballot(keys[i] = T_sel)
BallotWrite(mask_eq, keys, vals → outVal, outIdx, out_pos)  ▷ truncate at K
and how was ballot write implemented?
```

**Answer:**

`BallotWrite` is a **warp-cooperative ballot + popcount + atomicAdd** pattern. From `heuristic_topk.cuh` lines 671–709:

```cuda
for (int base = warp_id * WARP_SIZE; base < cand_count; base += BLOCK_SIZE)
{
    int i = base + lane;
    float v = (i < cand_count) ? smem->keys[i] : -FLT_MAX;

    // --- BallotWrite for elements > T_sel ---
    bool emit_gt = (i < cand_count) && (v > sel_thr);
    unsigned mask_gt = __ballot_sync(full_mask, emit_gt);
    if (mask_gt)
    {
        int cnt  = __popc(mask_gt);                          // total qualifying in warp
        int moff = __popc(mask_gt & ((1u << lane) - 1u));    // my offset within batch
        int bp = 0;
        if (lane == 0)
            bp = atomicAdd(&smem->out_count, cnt);           // reserve cnt slots
        bp = __shfl_sync(full_mask, bp, 0);                  // broadcast base pos
        if (emit_gt && bp + moff < TOP_K)
        {
            outputValues[bp + moff]  = v;                    // write to global
            outputIndices[bp + moff] = smem->vals[i];
        }
    }

    // --- BallotWrite for elements == T_sel (fill remaining slots) ---
    bool emit_eq = (i < cand_count) && (v == sel_thr);
    unsigned mask_eq = __ballot_sync(full_mask, emit_eq);
    if (mask_eq)
    {
        int cnt  = __popc(mask_eq);
        int moff = __popc(mask_eq & ((1u << lane) - 1u));
        int bp = 0;
        if (lane == 0)
            bp = atomicAdd(&smem->out_count, cnt);
        bp = __shfl_sync(full_mask, bp, 0);
        if (emit_eq && bp + moff < TOP_K)
        {
            outputValues[bp + moff]  = v;
            outputIndices[bp + moff] = smem->vals[i];
        }
    }
}
```

**Step-by-step mechanism:**

| Step | Instruction | Purpose |
|---|---|---|
| 1 | `__ballot_sync(full_mask, emit)` | Each lane in the warp votes; returns 32-bit mask where bit `i` = 1 if lane `i` qualifies |
| 2 | `__popc(mask)` | Count total qualifying lanes in the warp → `cnt` |
| 3 | `__popc(mask & ((1u << lane) - 1u))` | **Intra-warp exclusive prefix popcount** — count qualifying lanes *below* current lane → `moff` (unique offset within warp batch) |
| 4 | `atomicAdd(&smem->out_count, cnt)` (lane 0 only) | Reserve `cnt` contiguous output slots; returned `bp` = base position for this warp's batch |
| 5 | `__shfl_sync(full_mask, bp, 0)` | Broadcast `bp` from lane 0 to all lanes |
| 6 | Write to `output[bp + moff]` | Each qualifying thread writes to its unique position |

**Two-pass structure (GT then EQ)**:
- First pass emits all elements **strictly greater** than `T_sel` — these are unconditionally in the Top-K.
- Second pass fills remaining slots with elements **exactly equal** to `T_sel` (tie-breaking).
- The `bp + moff < TOP_K` guard truncates at exactly K outputs, handling the case where there are more ties than remaining slots.

**Why ballot is acceptable here**: This Phase 4 ballot-compact operates on **SMEM candidates** (≤ 6144 elements), not the full N input. The compiler-barrier cost of `__ballot_sync` that hurts Phase 3 (where it serializes L2 load pipelining across N = 70K+ elements) is negligible here because Phase 4 has no global memory loads — only SMEM reads and global memory writes (K × 8B = 16 KB, coalesced).

After the loop, any unfilled slots (if `out_count < TOP_K`) are padded:

```cuda
int filled = min(smem->out_count, TOP_K);
for (int i = filled + tid; i < TOP_K; i += BLOCK_SIZE)
{
    outputValues[i]  = -FLT_MAX;
    outputIndices[i] = -1;
}
```
