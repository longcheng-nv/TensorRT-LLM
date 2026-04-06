# Heuristic-Guided Top-K: Algorithm Pseudocode & GPU Performance Model

> **Source**: `cpp/tensorrt_llm/kernels/heuristic_topk.cuh` (V2e)
>
> Sort-free, histogram-based Top-K selection optimised for a **single thread-block** on NVIDIA Blackwell (sm\_100).

---

## 1. Notation & Constants

| Symbol | Value | Description |
|--------|-------|-------------|
| \(N\) | runtime | Total number of input elements (global memory) |
| \(M\) | 2048 | Number of heuristic pre-indexed positions |
| \(K\) | 2048 | Top-K to select |
| \(C_{\max}\) | 6144 | `TOP_K + 2 × SAFETY_MARGIN` — max candidates in SMEM |
| \(B\) | 2048 | `NUM_BINS` — histogram resolution |
| \(P\) | 512 | `BLOCK_SIZE` — threads per CTA |
| \(W\) | 32 | `WARP_SIZE` |
| \(N_w\) | 16 | `NUM_WARPS = P / W` |
| \(R_{\max}\) | 15 | `MAX_REFINE_ITERS` — Phase 2 iteration cap |
| \(R\) | 1–2 (real decode), 4–6 (synthetic) | Actual Phase 2 iterations at runtime |
| \(S\) | 1–3 typical | Phase 4 snap iterations at runtime |

### Shared Memory Layout (~59 KB)

| Field | Size | Purpose |
|-------|------|---------|
| `keys[6144]` | 24 KB | Candidate values (float) |
| `vals[6144]` | 24 KB | Candidate original indices (int) |
| `histogram[2048]` | 8 KB | Histogram bins / scratch |
| `per_thread_counts[512]` | 2 KB | OPT7 — cached per-thread counts |
| `warp_counts[16]` + scalars | ~192 B | Reduction scratch, control variables |

---

## 2. Algorithm Pseudocode

### Algorithm 1: HeuristicTopK — Main Procedure

```
ALGORITHM HeuristicTopK(input[N], preIdx[M], K) → (outVal[K], outIdx[K])
────────────────────────────────────────────────────────────────────────

  // ─── Phase 1: Pre-Index Statistics ───────────────────────────────
  1   (pmin, pmax, pmean) ← PreIndexStats(input, preIdx, M)
  2   T ← pmean                           ▷ initial threshold = mean of heuristic samples
  3   lo ← pmin;  hi ← pmax
  4   done ← false

  // ─── Phase 2: Interpolation Threshold Search ────────────────────
  5   C ← CountGE(input, N, T)            ▷ count elements ≥ T (full-N scan)
  6   if K ≤ C ≤ C_max then done ← true
  7   else UpdateBounds(lo, hi, T, C)

  8   for r = 1 to R_max while ¬done do
  9   │   T ← InterpolateThreshold(lo, hi, cnt_lo, cnt_hi, r)
 10   │   C ← CountGE(input, N, T)        ▷ full-N scan
 11   │   if K ≤ C ≤ C_max then done ← true
 12   │   else UpdateBounds(lo, hi, T, C)
 13   end for

 14   if ¬done then T ← FallbackThreshold(lo, hi, cnt_lo)

  // ─── Phase 3: Candidate Collection (Prefix-Sum Compact) ─────────
 15   if done ≠ EXACT then               ▷ OPT5: skip re-count when Phase 2 was exact
 16   │   C ← CountGE(input, N, T)
 17   │   RetryNarrow(input, N, T, lo, hi) while C > C_max   ▷ ≤10 retries
 18   end if

 19   offsets[] ← PrefixSum(per_thread_counts[])   ▷ OPT7: reuse cached counts
 20   cand_count ← offsets[last]

 21   StreamCollect(input, N, T, offsets)  ▷ write qualifying elements → SMEM keys[]/vals[]

  // ─── Phase 4: Histogram-Based Exact Selection ───────────────────
 22   if cand_count = K then
 23   │   DirectCopy(keys, vals → outVal, outIdx)
 24   │   return

 25   (cmin, cmax) ← BlockMinMax(keys[0..cand_count-1])

      // ── 4a: Build histogram ──
 26   histogram[0..B-1] ← 0
 27   for each candidate keys[i]:
 28   │   bin ← ⌊(keys[i] - cmin) / (cmax - cmin) × B⌋
 29   │   histogram[bin] += 1               ▷ SMEM atomicAdd

      // ── 4b: Parallel K-th bin search (OPT6) ──
 30   warp_sums[w] ← Σ histogram[high_bins_of_warp_w]   ▷ each warp sums B/N_w = 128 bins
 31   target_warp ← first w s.t. prefix_sum(warp_sums) ≥ K
 32   T_sel ← ScanBinsInWarp(target_warp, histogram, K)  ▷ 128-step serial scan

      // ── 4c: Snap refinement ──
 33   for s = 1 to snap_limit do
 34   │   (cnt_ge, cnt_gt, succ_up, pred_down) ← FusedSnapScan(keys, cand_count, T_sel)
 35   │   if cnt_gt < K ≤ cnt_ge then break   ▷ converged: T_sel is exact pivot
 36   │   T_sel ← SnapAdjust(cnt_ge, cnt_gt, succ_up, pred_down)
 37   end for

      // ── 4d: Ballot-compact output ──
 38   out_pos ← 0
 39   for each warp-tile of candidates:
 40   │   mask_gt ← ballot(keys[i] > T_sel)
 41   │   BallotWrite(mask_gt, keys, vals → outVal, outIdx, out_pos)
 42   │   mask_eq ← ballot(keys[i] = T_sel)
 43   │   BallotWrite(mask_eq, keys, vals → outVal, outIdx, out_pos)  ▷ truncate at K
 44   PadRemaining(out_pos, K, -∞, -1)

 45   return (outVal, outIdx)
```

---

### Algorithm 2: CountGE — Block-Wide Count of Elements ≥ Threshold

```
FUNCTION CountGE(input[N], N, T) → count
────────────────────────────────────────────────────────────────────────
  PARALLEL for each thread tid ∈ [0, P):
  1   c ← 0
  2   for i = tid × 4;  i + 3 < N;  i += P × 4 do     ▷ vectorised float4 loads
  3   │   v4 ← __ldg( (float4*)(input + i) )            ▷ 128B coalesced read
  4   │   c += (v4.x ≥ T) + (v4.y ≥ T) + (v4.z ≥ T) + (v4.w ≥ T)
  5   end for
  6   // scalar tail: ≤ 3 elements (negligible)
  7   per_thread_counts[tid] ← c                         ▷ OPT7: cache for Phase 3
  8   c ← WarpReduceSum(c)                               ▷ redux.sync (sm≥80)
  9   if lane = 0 then warp_counts[warp_id] ← c
  BARRIER
 10  if tid = 0 then count ← Σ warp_counts[0..N_w-1]
```

### Algorithm 3: InterpolateThreshold

```
FUNCTION InterpolateThreshold(lo, hi, cnt_lo, cnt_hi, iter) → T_new
────────────────────────────────────────────────────────────────────────
  1   target ← K + SAFETY / 2
  2   range ← hi - lo
  3   if cnt_lo > cnt_hi AND range > ε then
  4   │   f ← (cnt_lo - target) / (cnt_lo - cnt_hi)
  5   │   f ← clamp(f, 0.05, 0.95)
  6   │   if iter = 1 then f ← min(f, 0.50)             ▷ conservative first step
  7   │   T_new ← lo + range × f
  8   else
  9   │   T_new ← (lo + hi) / 2                         ▷ fallback: bisection
 10  // Boundary guards: ensure T_new ∈ (lo, hi)
 11  if T_new ≤ lo then T_new ← lo + 0.05 × range
 12  if T_new ≥ hi then T_new ← hi - 0.05 × range
 13  return T_new
```

### Algorithm 4: FusedSnapIter — Threshold Refinement on SMEM Candidates

```
FUNCTION FusedSnapIter(keys[C], C, T) → (cnt_ge, cnt_gt, succ_up, pred_down, T')
────────────────────────────────────────────────────────────────────────
  PARALLEL for each thread tid:
  1   lge ← 0;  lgt ← 0;  s_up ← +∞;  s_down ← -∞
  2   for i = tid;  i < C;  i += P do                   ▷ SMEM reads
  3   │   v ← keys[i]
  4   │   lge += (v ≥ T);   lgt += (v > T)
  5   │   if v > T then s_up ← min(s_up, v)             ▷ nearest value above T
  6   │   if v < T then s_down ← max(s_down, v)         ▷ nearest value below T
  7   end for
  8   WarpReduce(lge, lgt, s_up, s_down)
  BARRIER
  9   tid=0: aggregate → cnt_ge, cnt_gt, total_up, total_down
 10   if cnt_gt ≥ K then T' ← total_up                  ▷ raise threshold
 11   elif cnt_ge < K then T' ← total_down               ▷ lower threshold
 12   else T' ← T                                        ▷ converged
  BARRIER
```

---

## 3. GPU Performance Model

### 3.1 Per-Phase Breakdown

#### Phase 1 — Pre-Index Statistics

| Metric | Value | Notes |
|--------|-------|-------|
| Global reads | \(2M \times 4\text{B} = 16\) KB | **Random** scatter-gather via `preIdx` |
| Compute | \(\sim 4M\) FP ops | min/max/sum — trivial |
| SMEM I/O | ~256 B | Warp partials → cross-warp reduce |
| Barriers | 2 | |
| **Bottleneck** | **Global memory latency** | Random access, L2 hit rate depends on `preIdx` locality |

#### Phase 2 — Interpolation Threshold Search

Each `CountGE` call performs one **full-N streaming pass**.

| Metric | Value (per iteration) | Notes |
|--------|----------------------|-------|
| Global reads | \(N \times 4\text{B}\) | **Coalesced**, vectorised `float4` |
| Compute | \(N\) compares + \(\lceil N/P \rceil\) adds | Arithmetic intensity ≈ 0.25 FLOP/B |
| SMEM I/O | ~128 B | Reduction scratch only |
| Barriers | 3 | sync after count, threshold update, next iter |

**Aggregate over \(R\) iterations (+ 1 initial call):**

| Metric | Formula |
|--------|---------|
| Total global reads | \((1 + R) \times N \times 4\) bytes |
| Total barriers | \(1 + 3R\) |
| **Bottleneck** | **Single-SM HBM bandwidth** (streaming, coalesced) |

#### Phase 3 — Prefix-Sum Candidate Collection

| Metric | Value | Notes |
|--------|-------|-------|
| Global reads | \(N \times 4\text{B}\) | 1 full-N pass (collect); OPT5 may add 1 more (re-count) |
| SMEM writes | \(C \times 8\text{B} \leq 48\) KB | `keys[]` + `vals[]`, non-contiguous → bank conflicts possible |
| Prefix-sum | \(5 \times\) `shfl_up` + 16-step serial | Intra-warp + inter-warp |
| Barriers | 2–3 | 2 if `done==1` (OPT5), 3 otherwise |
| **Bottleneck** | **Global BW** (read) + SMEM write throughput |

#### Phase 4 — Histogram-Based Exact Selection

All operations on **SMEM** data (\(C \leq 6144\) candidates).

| Sub-phase | Work | Barriers |
|-----------|------|----------|
| 4a: Min/Max | \(C\) SMEM reads + warp reduce | 1 |
| 4b: Histogram build | \(C\) SMEM reads + \(C\) `atomicAdd` (SMEM, 32-bit) | 2 (clear + build) |
| 4c: K-th bin search | **Parallel 2-step** (OPT6): \(N_w + B/N_w = 16 + 128 = 144\) serial steps (vs 2048 naive) | 3 |
| 4d: Snap refine | \(S \times C\) SMEM reads; 4 reductions/iter | \(2S\) |
| 4e: Output | \(C\) SMEM reads + \(K \times 8\text{B}\) global writes (ballot-compact) | 1 |

| Metric | Value |
|--------|-------|
| Total SMEM reads | \(\sim C \times (3 + S)\) words |
| Total SMEM atomics | \(C\) (histogram) |
| Global writes | \(K \times 8\text{B} = 16\) KB (coalesced) |
| Barriers | \(\sim 8 + 2S\) |
| **Bottleneck** | **SMEM throughput + instruction latency** (data fits entirely in SMEM) |

---

### 3.2 End-to-End Summary Table

```
┌─────────┬──────────────────────────┬───────────────────────┬──────────────┬─────────────────┐
│  Phase  │  Global Memory I/O       │  SMEM I/O             │ Barriers     │ Bottleneck      │
├─────────┼──────────────────────────┼───────────────────────┼──────────────┼─────────────────┤
│ Ph 1    │ 2M×4B = 16 KB (random)   │ ~256 B                │ 2            │ GMEM latency    │
│         │                          │                       │              │ (scatter-gather) │
├─────────┼──────────────────────────┼───────────────────────┼──────────────┼─────────────────┤
│ Ph 2    │ (1+R)×N×4B (coalesced)   │ ~128 B / iter         │ 1 + 3R       │ GMEM BW         │
│         │ R ∈ [1,2] real decode    │                       │              │ (streaming)     │
├─────────┼──────────────────────────┼───────────────────────┼──────────────┼─────────────────┤
│ Ph 3    │ N×4B read (coalesced)    │ C×8B write ≤ 48 KB    │ 2–3          │ GMEM BW         │
│         │ (+1 pass if done≠1)      │                       │              │                 │
├─────────┼──────────────────────────┼───────────────────────┼──────────────┼─────────────────┤
│ Ph 4    │ K×8B = 16 KB write       │ C×4B reads × ~S iters │ 8 + 2S       │ SMEM throughput │
│         │ (coalesced, ballot)      │ + C atomics (hist.)   │ S ∈ [1,3]    │ + ILP           │
└─────────┴──────────────────────────┴───────────────────────┴──────────────┴─────────────────┘
```

### 3.3 Total Global Memory Traffic (Dominant Term)

Best case (`done==1`, OPT5+OPT7 active — typical for real decode with good prediction):

$$
T_{\text{global}} \approx \bigl[(1 + R) \cdot N + M + 2K\bigr] \times 4 \;\text{bytes} \;\approx\; (1 + R) \times 4N
$$

Worst case (`done≠1`, Phase 3 requires re-count):

$$
T_{\text{global}} \approx \bigl[(2 + R) \cdot N + M + 2K\bigr] \times 4 \;\text{bytes} \;\approx\; (2 + R) \times 4N
$$

since \(N \gg M, K\).

**Example (real decode)**: \(N = 70\text{K}\), \(R = 2\), `done==1`

$$
T_{\text{global}} \approx 3 \times 280\;\text{KB} = 840\;\text{KB}
$$

**Example (synthetic/worst-case)**: \(N = 128\text{K}\), \(R = 5\), `done≠1`

$$
T_{\text{global}} \approx 7 \times 512\;\text{KB} = 3.5\;\text{MB}
$$

### 3.4 Bandwidth & Latency Estimates (NVIDIA B200)

| Parameter | Value |
|-----------|-------|
| B200 peak HBM BW | ~8 TB/s |
| SMs on B200 | **160** |
| L2 cache | 64 MB |
| Single-CTA effective BW | ~60–80 GB/s (1 SM, L2 streaming) |
| **Real decode example** (\(N{=}70\text{K}\), \(R{=}2\)) | \(840\;\text{KB} / 70\;\text{GB/s} \approx 12\;\mu\text{s}\) (global mem only) |
| **Synthetic example** (\(N{=}128\text{K}\), \(R{=}5\)) | \(3.5\;\text{MB} / 70\;\text{GB/s} \approx 50\;\mu\text{s}\) |
| Measured kernel latency (real, L2-cold) | GVR ~24 µs, Radix ~44 µs at \(N{=}70\text{K}\) |
| Barrier overhead (total) | ~15–30 × ~0.5 µs ≈ 10–15 µs |
| Phase 4 (SMEM-only) | < 10 µs (data ≤ 48 KB, all in SMEM) |

---

### 3.5 Roofline Parameters

| Parameter | Value |
|-----------|-------|
| Arithmetic Intensity (Ph 2, dominant) | ~0.25 FLOP/Byte (1 compare per 4 B) |
| Single-CTA achievable HBM BW | 60–80 GB/s |
| SMEM BW (Ph 4, dominant) | ~128 B/cycle × SM clock |
| Total global data volume | \((1{+}R) \times 4N\) bytes (best) to \((2{+}R) \times 4N\) (worst) |
| Total SMEM data volume (Ph 4) | \(\sim S \times C \times 4\) bytes, \(C \leq 6144\) |
| Total barrier overhead | ~15–30 barriers × ~0.5 µs each |

### 3.6 Regime Analysis

| Input Size \(N\) | Regime | Explanation |
|-------------------|--------|-------------|
| \(N \leq 64\text{K}\) | **Memory-latency bound** | Data fits in L2; single CTA cannot generate enough requests to saturate even L2 BW |
| \(64\text{K} < N \leq 256\text{K}\) | **Single-SM BW bound** | Streaming from HBM, 1 CTA limited to ~60–80 GB/s |
| \(N > 256\text{K}\) | **Single-SM BW bound (deep)** | Linear scaling; multi-CTA / multi-job batching becomes attractive |

---

## 4. Optimization Inventory

| ID | Optimization | Phase | Effect on Performance Model |
|----|-------------|-------|-----------------------------|
| OPT3 | `__ldg` (read-only cache) | All global reads | Higher L2/L1 hit rate; non-temporal hints |
| OPT4 | `redux.sync` (sm≥80) | All reductions | Single-instruction warp reduce; saves 5× `shfl` latency |
| OPT5 | Skip re-count when `done==1` | Ph 3 | **Saves 1 full-N pass** (\(4N\) bytes) |
| OPT6 | Parallel K-th bin search | Ph 4b | Serial depth 144 vs 2048 → **~14× speedup** in scan |
| OPT7 | Cached `per_thread_counts` | Ph 3 | **Saves 1 full-N pass** (\(4N\) bytes); reuse from last `CountGE` |

**Combined OPT5 + OPT7 savings** (best case, `done==1`):

$$
\Delta T = 2 \times N \times 4\;\text{bytes} \;/\; \text{BW}_{\text{eff}}
$$

For \(N = 128\text{K}\): \(\Delta T \approx 1\;\text{MB} / 70\;\text{GB/s} \approx 14\;\mu\text{s}\) saved.

---

## 5. Design Trade-offs

| Decision | Benefit | Cost |
|----------|---------|------|
| **Single CTA** | Zero inter-CTA sync; all communication via SMEM; no global atomics; minimal launch overhead | Uses 1/160 SMs → cannot saturate full HBM BW |
| **Interpolation search (vs bisection)** | Converges in ~1–2 iters (real decode) vs ~10–15 for pure bisection | Slightly more complex threshold computation (negligible FP cost) |
| **2048-bin histogram** | Fine-grained initial pivot for snap phase; reduces snap iterations | 8 KB SMEM; \(C\) atomic adds (low contention with 2048 bins) |
| **Snap refinement** | Exact convergence without sorting | Each snap iter scans all \(C\) candidates (up to 6144 × 4B reads from SMEM) |
| **`float4` vectorisation** | 4× fewer load instructions; 128B coalesced transactions | Requires \(N\) aligned to 4; tail loop for remainder |
| **Heuristic pre-index** | Provides informed initial threshold → fewer Phase 2 iterations | \(M\) random global reads (16 KB, latency-sensitive) |

---

## Appendix: Kernel Launch Configuration

```
Grid:             1 block
Block:            512 threads (16 warps)
Shared memory:    ~59 KB (dynamic, opt-in via cudaFuncSetAttribute)
Registers:        independently optimised (__noinline__ device function)
Launch bounds:    __launch_bounds__(512, 1)
```
