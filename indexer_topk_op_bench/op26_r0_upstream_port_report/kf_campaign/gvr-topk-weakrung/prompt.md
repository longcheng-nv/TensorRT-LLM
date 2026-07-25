# DeepSeek Indexer Top-K, Batched Decode (BS 16-1024, fp32, B200) — Beat the Production Kernel on Its Throughput Wall

## Problem

Sparse-attention indexer top-K at decode time, BATCHED: `logits[b, npad]`
fp32 (valid length `n_valid` per row, identical rows — every row is the same
real captured production row), `pre_idx[b, k]` int32 = previous decode step's
top-k (temporal warm hint, identical rows). Return `indices[b, k]` int32 per
row, any order. Exactness is non-negotiable and tie-robust set-based: every
index whose value is strictly greater than the k-th value must appear in
every row, on every run.

The 14 workloads are REAL production captures spanning the exact cells where
every known fast kernel LOSES to the production baseline:
- V4-Flash k=512: n=4099 (BS 256/1024), 8195 (256), 32771 (256), 65538 (32/256)
- V4-Pro k=1024: n=4099 (256/1024), 32771 (256), 65539 (16/256), 131075 (128), 262127 (16)
- V3.2 k=2048: n=65551 (256)

## Baseline and why these cells are special

The baseline timings are EXTERNAL nsys cold-L2 pure-kernel medians of the
production GVR top-K (TensorRT-LLM PR#16457, guess-verify-refine: pre_idx
threshold seed -> log-secant solve -> exact collect) launched NATIVELY
BATCHED on an idle B200. The production kernel amortizes batching well —
these 14 cells are its stronghold ("throughput wall"): two strong prior
optimized kernels (provided as seeds, see below) reach only 0.84-0.95x here
while winning 1.3-2.5x elsewhere. Your target is precisely this wall.

Platform timing note: your platform measurements carry a fixed harness
overhead of roughly 10-20us per call; the baselines do NOT contain it. At
true parity your platform speedup reads below 1.0 on the smaller cells.
Track relative progress; absolute wins live on the big (b x npad) cells
where the floor is a small fraction.

## Target

- **Required:** geomean speedup > 1.0x over baseline across the 14 workloads
  with NO workload below 0.95x (max 5% per-cell regression — hard bar).
  Stretch goal: geomean >= 1.15x.
- Any exact kernel that clears >1.0x on even a SUBSET of these cells is
  valuable (downstream integration dispatches per (b, npad, k) rung and
  falls back to the production kernel per rung) — but the submitted kernel
  itself must satisfy the no->5%-regression bar across all 14 workloads.

## Constraints (LLM judge + downstream code review enforce these)

1. GVR skeleton: exploit `pre_idx` as the threshold prior; equivalence-
   preserving redesign of any phase is allowed; sub-phases may absorb mature
   primitives (radix/histogram). No prior-free wholesale replacement.
2. Rows are identical COPIES by benchmark construction, but your kernel MUST
   compute every row's output from that row's own data. Computing one row
   and broadcasting/copying results across rows is CHEATING and is rejected
   (this has been caught and disqualified before). Legitimate cross-row
   amortization is limited to phase-1 threshold-prior construction shared
   across rows of the same batch (each row still verified/refined on its own
   data).
3. No CUDA graphs; no framework kernel imports (flashinfer/trtllm/...).
4. Exact output required at every hint quality; you may NOT branch on any
   host-side estimate of hint quality (unknowable at inference).

## SEED SOLUTIONS — start here, do not start from scratch

Two strong CUDA C++ kernels ship in the evaluation staging directory (same
place as `wl_*.safetensors`):
- **`seed_v3mt.cu` + `seed_v3mt_main.cpp`** — register-resident GVR, grid.y
  batching (one CTA group per row), per-K multi-threshold P1 ladder + log-
  secant P2, exact refine. Overall BS>1 envelope champion (gm 1.31) but
  0.84-0.95x on THESE cells.
- **`seed_e6.cu` + `seed_e6_main.cpp`** — sequential-emission arm: K0
  hint-min + clustered-sample quantile threshold, K1 fused tile collect +
  last-CTA exact 4-level reduce, K2 second-chance rescue; chunks ladder,
  ILP dispatched by BS (8 below 512, 4 at/above), __ldcs streaming loads at
  npad>=262144. Wins large-npad x BS>=256 (up to 2.56x) but loses here.

Establish your floor by submitting the better seed per your own measurement,
then attack the wall. Known verified facts (nsys, do not re-litigate):
- Uniform ILP-8 collect regresses BS>=512; dispatch ILP by BS.
- cp.async double-buffer loses (smem round-trip); __ldcs helps only at
  npad>=262144.
- CDP2 device-side tail launch loses globally (-rdc register reserve).
- The production baseline's advantage here is native batch amortization of
  its scan passes — the seeds pay per-row pass overhead the baseline shares
  across the batch. Closing THAT structural gap is the campaign.

## Output contract

torch extension binding: `run(logits, pre_idx, n_valid, indices)` — see the
seed `main.cpp` files. Compile flags: `-O3 --use_fast_math -std=c++17`,
target sm_100a.
