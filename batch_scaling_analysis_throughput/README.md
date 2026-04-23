# Batch-Scaling Throughput Analysis for GVR Top-K on B200

> This note consolidates the key conclusions from the batch-scaling `nsys` study, the final Scheme X dispatcher analysis, and the cuobjdump/NCU-verified register and block-limit investigation for the decode-side GVR Top-K kernel on NVIDIA Blackwell (B200, `sm_100`).

---

## 1. Scope

This discussion focuses on the decode-stage exact Top-K kernel used by DeepSeek Sparse Attention in TensorRT-LLM:

- **Baseline**: decode-side radix Top-K (`topKPerRowDecode<512, true, 0, 0>`)
- **Heuristic path**: production GVR decode kernel (`heuristicTopKMultiRowKernel`)
- **Problem size**: `N = 70,690`, `K = 2048`
- **Layers analyzed**: `L0, L1, L20, L21, L22, L40, L41, L42, L60`
- **Batch sizes analyzed**: `1, 2, 4, 8, 16, 32, 64, 128, 256, 512`
- **Primary timing source**: **nsys NVTX GPU-projected median latency**

The goal is to explain three related phenomena:

1. Why GVR is consistently faster than Radix at small and medium batch sizes,
2. Why the speedup compresses or regresses at very large batch sizes,
3. Why the most robust production solution is a **batch-threshold dispatcher** rather than further single-kernel tuning.

---

## 2. High-Level Result

The batch-scaling study shows a clear two-regime behavior:

- **BS = 1 to 128**: GVR is consistently superior, typically **1.7x-2.1x** on good layers and **~1.2x** even on the weakest layer.
- **BS = 256 to 512**: the advantage compresses sharply; some cells remain strong, some become unstable, and a small number of cells regress below `1.0x`.

The key point is that this large-batch degradation is **not** caused by a change in the kernel's static occupancy limits. Those limits are batch-size invariant. Instead, it is caused by:

- wave quantization,
- increased CTA competition once the grid exceeds the single-wave residency budget,
- and, most importantly, **HBM-latency amplification** on the heuristic miss path.

In other words, **theoretical occupancy stays fixed while achieved performance degrades because more CTAs become simultaneously resident and stalled on long-scoreboard memory latency**.

---

## 3. What the `nsys` Batch Sweep Actually Shows

### 3.1 Stable regime: BS 1-128

Across all 9 layers, the `nsys` data shows that both kernels scale nearly flat from `BS=1` to `BS=128`, consistent with a single-CTA-per-row design:

- **Radix** stays in the low-40-us range,
- **GVR** stays in the low-20-us to high-30-us range depending on layer,
- speedup remains strong and layer-dependent.

Representative observations from the `nsys` medians:

- **L21**: about `2.04x` at `BS=1`, rising to a peak of `2.14x` at `BS=16`
- **L60**: about `1.86x` at `BS=1`, around `1.96x` at `BS=16`
- **L0**: about `1.18x` at `BS=1`, around `1.25x` at `BS=16`

This regime is where GVR's algorithmic advantage is clearest:

- fewer full-row passes,
- lower synchronization overhead,
- shared-memory-local exact refinement,
- and strong temporal-prediction leverage on the good layers.

### 3.2 Inflection regime: BS 256

At `BS=256`, the average latency of both kernels jumps, but the heuristic path becomes much less stable:

- Radix remains structurally clean,
- GVR begins to show high layer-to-layer and row-to-row variance,
- and some layers enter a mixed win/loss zone.

This is the first sign that the single-kernel Pareto frontier is being overtaken by a **throughput scheduling effect** rather than a pure algorithm effect.

### 3.3 Degradation regime: BS 512

At `BS=512`, the entire workload is beyond the one-wave residency region for the decode kernels:

- Radix latency grows substantially but remains predictable,
- GVR still wins on many strong-correlation layers,
- but weak-correlation layers can now lose.

The most visible example is:

- **L0 @ BS=512**: GVR falls below Radix (`0.81x` in the `nsys` study)

while stronger layers still retain useful speedup:

- **L41 / L42 / L60 @ BS=512**: still around `1.39x-1.48x`

This large-batch regime is therefore **heterogeneous**, not uniformly bad: good layers still benefit, but the worst layers no longer have enough margin to survive miss-path inflation under SM contention.

---

## 4. Why Batch Size Changes Runtime but Not Static Block Limits

One of the central findings of the NCU + cuobjdump investigation is:

> **Block Limit Registers, Block Limit Shared Mem, Block Limit Warps, and Theoretical Occupancy are static per-CTA properties. They do not depend on grid size.**

For the decode kernels discussed here, the block shape is always:

- `512` threads / CTA
- `16` warps / CTA

so the block-limit quantities are determined entirely by:

- `regs_per_thread`,
- per-CTA shared memory,
- hardware warps per SM,
- and hardware CTA slots per SM.

The derived formulas are:

- `BL Registers = floor(65536 / (regs_per_thread × 512))`
- `BL Shared Mem = floor(Smem_Config / per_CTA_smem)`
- `BL Warps = floor(64 / 16) = 4`
- `Theoretical Active Warps / SM = min(BL_reg, BL_smem, BL_warps, BL_sm) × 16`
- `Theoretical Occupancy = active_warps / 64`

Therefore:

- **BS changes the number of CTAs in the grid**
- but **does not change per-CTA resource usage**
- so **it cannot change the block-limit values themselves**

What BS *does* change is:

- how many waves are required to drain the grid,
- how much tail inefficiency is introduced,
- and how much memory-latency overlap and long-scoreboard stalling accumulate once the grid exceeds the clean-residency region.

---

## 5. Per-Kernel Register Footprint Analysis

The cuobjdump-verified ptxas output gives the following authoritative resource usage:

| Kernel specialization | REG | Static SHARED | Per-CTA regs | BL Registers | Reg-limited Theor. Occ |
|---|---:|---:|---:|---:|---:|
| `heuristicTopKMultiRowKernel` (GVR decode) | 40 | 1024 B static + 59.50 KB dynamic at launch | 20,480 | 3 | 75% |
| `topKPerRowDecode<512, true, 0, 0>` (radix decode) | 40 | 19,568 B | 20,480 | 3 | 75% |
| `topKPerRowDecode<512, true, 1, 0>` (radix decode, multipleBlocksPerRow) | 40 | 19,568 B | 20,480 | 3 | 75% |
| `topKPerRowDecode<512, false, 0, 0>` (insertion decode) | 40 | 17,424 B | 20,480 | 3 | 75% |
| `topKPerRowDecode<1024, true, 0, 1>` (merge stage) | 52 | 38,096 B | 53,248 | 1 | 25% |
| `topKPerRowPrefill<512, true>` (radix prefill) | 40 | 19,568 B | 20,480 | 3 | 75% |
| `topKPerRowPrefill<512, false>` (insertion prefill) | 32 | 17,424 B | 16,384 | 4 | 100% |

### 5.1 Main implication

The most important observation is that:

> **GVR decode and radix decode use the same register count: 40 regs/thread.**

This matters because it means the decode-side theoretical occupancy ceiling is **shared** between the two kernels:

- both hit `BL Registers = 3`,
- both therefore cap at `3 CTA/SM`,
- both therefore cap at `48 active warps/SM = 75% theoretical occupancy`.

That is a strong negative result in the good sense: there is **no hidden register inflation penalty** in GVR relative to the radix decode baseline. Even though the algorithms are structurally different, ptxas lowers both into the same register bucket under `__launch_bounds__(512)`.

### 5.2 Why the register counts converge

This convergence is not forced by a pragma alone. It is the joint outcome of:

1. the shared block size (`512` threads),
2. the same launch-bound policy,
3. the `sm_100` register allocation granularity,
4. and a very similar per-thread live working set:
   - `float4` vector loads,
   - scan/count accumulators,
   - prefix-sum scratch state,
   - shared-memory pointers,
   - loop induction and address arithmetic.

The practical takeaway is:

> The decode-side register limiter is already "tight" and common to both kernels. Further occupancy growth does **not** come from small compiler flags; it requires a true reduction in the per-thread live range.

---

## 6. Per-Kernel Block Limit Analysis

NCU and cuobjdump together give the following decode-side block-limit picture:

| Kernel | Regs/Thread | Static + Dynamic + Driver Smem | BL Registers | BL Shared Mem | BL Warps | Theor. Occ | Limiting Resource |
|---|---:|---:|---:|---:|---:|---:|---|
| GVR Decode | 40 | `0 + 59.50 + 1.02 = 60.52 KB` | 3 | 3 | 4 | 75% | Registers **and** Shared Memory |
| Radix Decode | 40 | `18.54 + 8.19 + 1.02 = 27.75 KB` | 3 | 4 | 4 | 75% | Registers only |
| Radix Prefill radix | 40 | `~28.3 KB` | 3 | 4 | 4 | 75% | Registers only |
| Radix Prefill insertion | 32 | `~26.2 KB` | 4 | 5 | 4 | 100% | Registers and Warps |

### 6.1 GVR is double-bound; Radix decode is not

The decode-side difference is subtle but important:

- **GVR decode** is **double-bound**: registers and shared memory both stop it at `3 CTA/SM`
- **Radix decode** is **register-bound only**: shared memory would allow `4 CTA/SM`, but registers stop it at `3`

So although both kernels have the same **effective** theoretical occupancy, only Radix has **smem slack**.

### 6.2 Why this matters architecturally

This unused Radix shared-memory headroom is the main structural degree of freedom left on Blackwell:

> It is exactly the kind of slack that cluster distributed shared memory (DSM) could exploit, because DSM can trade extra inter-CTA collaboration against shared-memory capacity without first colliding with the register ceiling.

GVR does not have this option in the same form, because its shared-memory budget is already co-binding.

This is why the report identifies **cluster DSM** as the most plausible remaining architectural lever for decode-side residency expansion beyond the current single-CTA Pareto front.

---

## 7. Why Throughput Degrades: A Unified Interpretation

Putting the batch sweep and the resource audit together yields the following picture.

### 7.1 Small and medium batch sizes: algorithm dominates

At `BS <= 128`, throughput behavior is driven mainly by algorithmic efficiency:

- GVR performs fewer `N`-scale passes,
- avoids radix partitioning overhead,
- and converts temporal correlation into lower-latency exact selection.

Static occupancy is the same for GVR and radix decode, but that is not a problem because the kernel is still operating in the "good" residency regime.

### 7.2 Large batch sizes: memory latency dominates

At `BS >= 256`, the decisive effect is no longer the static occupancy ceiling. Instead:

- the grid spills into additional waves,
- the tail wave becomes more expensive,
- and, critically, **GVR miss-path CTAs amplify HBM demand** because they can trigger many full-row scans.

This is the true reason the heuristic path compresses or regresses at large BS:

> **Achieved occupancy rises, but the additional active warps are often stalled on HBM latency rather than doing useful work.**

So a higher achieved occupancy number in this regime is **not** a throughput win; it is often a sign of deeper long-scoreboard stasis.

### 7.3 The occupancy frontier is already reached

The key negative result from the report is:

> Decode-side throughput is no longer limited by a lack of resident warps. It is limited by what those resident warps are waiting for.

For this reason, trying to "push occupancy higher" without changing the per-thread live range or the HBM traffic pattern does not solve the real bottleneck.

---

## 8. Why Scheme X Is the Right Production Answer

The extended row sweep and integrated validation show that the safest deployable solution is not another heuristic variant, but a **batch-threshold dispatcher**:

```cpp
int const kBsLarge = sm_count * 3 - sm_count / 8;
bool const canUseHeuristic = (existing checks) && numRows < kBsLarge;
```

On B200 (`148` SMs), this yields `kBsLarge = 426`.

### 8.1 Why this works

Scheme X deliberately exploits a fact established by the throughput study:

- below the large-BS threshold, GVR is stably superior,
- above the threshold, the risk of row-level regressions rises sharply,
- and the regressions are driven by batch-scale architectural effects, not by layer-ID semantics that should be hardcoded into production.

### 8.2 Why this is better than per-layer logic

The final report explicitly rejects layer-ID hardcoding and per-layer α tables as the production answer. Scheme X is preferred because it is:

- **hardware-derived**
- **zero layer-ID hardcoded**
- **zero data-derived hyperparameters**
- **simple to integrate**
- and **sufficient to eliminate the remaining no-regression failures**

Integrated validation gave:

- **minimum** speedup vs Radix: `0.996x` (effectively parity / noise)
- **median** speedup vs Radix: `1.415x`
- **maximum** speedup vs Radix: `1.923x`

That is exactly the right trade-off for a production safeguard:

> preserve most of the heuristic upside in the sweet spot, while using Radix as the structural fallback in the throughput danger zone.

---

## 9. Practical Conclusions

### 9.1 What the batch-scaling study proves

1. **GVR is decisively better than Radix in the decode sweet spot** (`BS = 1..128`).
2. **The high-BS problem is architectural, not a simple occupancy deficiency**.
3. **GVR and Radix decode share the same register limiter** (`40 regs/thread`, `3 CTA/SM`, `75%` theoretical occupancy).
4. **GVR is additionally shared-memory bound**, while Radix retains one CTA of smem slack.
5. **HBM miss-path inflation**, not insufficient nominal occupancy, explains the large-BS collapse.

### 9.2 What the register/block analysis proves

1. The decode-side register limiter is real and binding.
2. It cannot be moved by launch-shape folklore alone.
3. Shared-memory reduction below the current GVR level would not, by itself, lift GVR above `3 CTA/SM`; the register ceiling would still hold.
4. The only remaining architectural slack belongs to the radix decode path on the shared-memory side, which is why **cluster DSM** is the main structural extension point.

### 9.3 What to do next

The analysis points to two future directions:

- **System-level**: use **Scheme X** style dispatch to protect throughput at large batch sizes.
- **Architecture-level**: explore **cluster DSM** and/or true per-thread live-range reduction if the goal is to change decode-side residency rather than just dispatch around it.

---

## 10. Verification Commands

### 10.1 ptxas ground truth

```bash
cuobjdump --dump-resource-usage tensorrt_llm/libs/libth_common.so \
    | grep -B1 -A1 -E "topKPerRow(Prefill|Decode)|heuristicTopKMultiRowKernel"
```

### 10.2 NCU block limits / occupancy sweep

```bash
grep -E "Block Limit (Registers|Shared Mem|Warps|SM)|Theoretical Occupancy|Achieved Occupancy|Achieved Active Warps" \
    ncu_l2_hypothesis/raw/{gvr,radix}_bs{1,128,256,400,512}.exported.csv
```

---

## 11. Bottom Line

The batch-scaling story is now consistent across algorithm analysis, runtime measurements, and resource accounting:

> **GVR wins because it is algorithmically better at the small-to-medium batch sizes that matter. It stops winning at very large batch sizes not because its theoretical occupancy is lower, but because the miss path turns memory latency into the dominant cost under multi-wave scheduling pressure.**

That is why the most robust production strategy is not to force one kernel to win everywhere, but to:

1. use **GVR** in the sweet spot,
2. use **Radix** when the grid crosses the architectural throughput danger zone,
3. and reserve deeper occupancy/residency gains for future architectural work such as **cluster DSM** or true live-range reduction.
