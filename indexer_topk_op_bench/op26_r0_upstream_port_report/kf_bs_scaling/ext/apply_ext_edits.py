# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Surgical edits turning compB kernel.cu into kernel_ext.cu (BS>1 extension).

Scope: aefm namespace ONLY (v30 keeps its own untouched copies; the v30 range
K=2048 & 16896<n<=140000 falls back to a sequential per-row loop).
Idempotent: re-running on an already-edited file fails the assertions loudly.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "kernel_ext.cu"
text = SRC.read_text()

CUT = "} // namespace aefm"
i = text.index(CUT)
aefm, rest = text[:i], text[i:]


def rep1(old, new):
    global aefm
    assert aefm.count(old) == 1, f"expected exactly 1 match:\n{old[:120]}"
    aefm = aefm.replace(old, new, 1)


# ---- E1: bottom3_kernel — grid.y row batching (out stride = n-3 == k) ----
rep1(
    """void bottom3_kernel(const float* __restrict__ logits, int n,
                    int* __restrict__ out) {
    const int tid = threadIdx.x;""",
    """void bottom3_kernel(const float* __restrict__ logits, long rstride, int n,
                    int* __restrict__ out) {
    logits += (size_t)blockIdx.y * (size_t)rstride;
    out += (size_t)blockIdx.y * (size_t)(n - 3);
    const int tid = threadIdx.x;""",
)

# ---- E2: topk_small — grid.y row batching ----
rep1(
    """void topk_small(const float* __restrict__ logits, int n, int k,
                int* __restrict__ out) {
    int tid = threadIdx.x;""",
    """void topk_small(const float* __restrict__ logits, long rstride, int n, int k,
                int* __restrict__ out) {
    logits += (size_t)blockIdx.y * (size_t)rstride;
    out += (size_t)blockIdx.y * (size_t)k;
    int tid = threadIdx.x;""",
)

# ---- E3: topk_mid — grid.y row batching ----
rep1(
    """void topk_mid(const float* __restrict__ logits, int n, int k,
              int* __restrict__ out) {
    const int tid = threadIdx.x;""",
    """void topk_mid(const float* __restrict__ logits, long rstride, int n, int k,
              int* __restrict__ out) {
    logits += (size_t)blockIdx.y * (size_t)rstride;
    out += (size_t)blockIdx.y * (size_t)k;
    const int tid = threadIdx.x;""",
)

# ---- E4: topk_fast — row-team partitioning (team == gridDim.x when BS=1) ----
rep1(
    """void topk_fast(const float* __restrict__ logits, int n, int k,
               unsigned int* __restrict__ scratch, int* __restrict__ out,
               unsigned int gen) {
    const int gridsz = gridDim.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int gtid = blockIdx.x * BLOCK + tid;
    const int gstride = gridsz * BLOCK;""",
    """void topk_fast(const float* __restrict__ logits, long rstride, int n, int k,
               unsigned int* __restrict__ scratch, int* __restrict__ out,
               unsigned int gen, int team) {
    const int row = blockIdx.x / team;
    const int bx = blockIdx.x - row * team;
    logits += (size_t)row * (size_t)rstride;
    out += (size_t)row * (size_t)k;
    scratch += (size_t)row * (size_t)SCRATCH_WORDS;
    const int gridsz = team;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int gtid = bx * BLOCK + tid;
    const int gstride = gridsz * BLOCK;""",
)

# ---- E5: update aefm::topk_launch call sites for the new signatures ----
rep1("bottom3_kernel<<<1, 512, 0, stream>>>(logits, n, out);",
     "bottom3_kernel<<<1, 512, 0, stream>>>(logits, 0L, n, out);")
for t, blk in (("<2>", 768), ("<3>", 768), ("<6>", 768), ("<11>", 768),
               ("<17>", 1024)):
    rep1(f"topk_small{t}<<<1, {blk}, 0, stream>>>(logits, n, k, out);",
         f"topk_small{t}<<<1, {blk}, 0, stream>>>(logits, 0L, n, k, out);")
for t in ("<2>", "<3>", "<4>"):
    rep1(f"topk_mid{t}<<<1, 1024, 0, stream>>>(logits, n, k, out);",
         f"topk_mid{t}<<<1, 1024, 0, stream>>>(logits, 0L, n, k, out);")
rep1("topk_fast<<<blocks, BLOCK, 0, stream>>>(logits, n, k, g_scratch, out, g_gen);",
     "topk_fast<<<blocks, BLOCK, 0, stream>>>(logits, 0L, n, k, g_scratch, out,"
     " g_gen, blocks);")

# ---- E6: batched launcher appended at the end of namespace aefm ----
BATCHED = r"""
// ---------------------------------------------------------------------------
// BS>1 extension (R3_LEDGER "BS>1 extension design analysis", steps A + B).
//   A: small/mid-n single-CTA tiers batch via grid.y (one launch, row per
//      blockIdx.y).
//   B: large-n topk_fast batches via row teams -- team = ceil(n/2048) CTAs
//      per row (register-resident constraint), per-row scratch slices, the
//      per-row grid barrier only spans that row's team.  Rows that fit the
//      co-residency cap run in ONE launch (single wave); larger BS is chunked
//      into ceil(BS/rows_per_wave) launches.  Slices are per-ROW (never
//      reused within a launch), so the fence-less L1 safety argument is
//      preserved: reuse only happens across launch boundaries.
// ---------------------------------------------------------------------------
static unsigned int* g_scratch_bs = nullptr;
static int g_scratch_bs_rows = 0;
static unsigned int g_gen_bs = 0;   // separate buffer => independent senses

void ext_caps(int* team_out, int* cap_out, int n) {
    if (!g_blocks_cap) {
        int active = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, topk_fast, BLOCK, 0);
        if (active < 1) active = 1;
        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
        g_blocks_cap = active * g_sms;
    }
    *team_out = (n + 2047) / 2048;
    *cap_out = g_blocks_cap;
}

void topk_launch_batched(const float* logits, long W, int n, int k, int* out,
                         int BS, cudaStream_t stream) {
    const dim3 gy(1, (unsigned int)BS);
    if (n - k == 3 && n <= 1536) {
        bottom3_kernel<<<gy, 512, 0, stream>>>(logits, W, n, out);
        return;
    }
    if (n <= 1536) {
        topk_small<2><<<gy, 768, 0, stream>>>(logits, W, n, k, out);
        return;
    }
    if (n <= 2304) {
        topk_small<3><<<gy, 768, 0, stream>>>(logits, W, n, k, out);
        return;
    }
    const bool tail_sel = 4 * k <= n;
    if (n <= 4608) {
        topk_small<6><<<gy, 768, 0, stream>>>(logits, W, n, k, out);
        return;
    }
    if (tail_sel && n <= 8195) {
        topk_mid<2><<<gy, 1024, 0, stream>>>(logits, W, n, k, out);
        return;
    }
    if (n <= 8448) {
        topk_small<11><<<gy, 768, 0, stream>>>(logits, W, n, k, out);
        return;
    }
    if (tail_sel && n <= 12291) {
        topk_mid<3><<<gy, 1024, 0, stream>>>(logits, W, n, k, out);
        return;
    }
    if (tail_sel && n <= 16387) {
        topk_mid<4><<<gy, 1024, 0, stream>>>(logits, W, n, k, out);
        return;
    }
    if (n <= 16896) {
        topk_small<17><<<gy, 1024, 0, stream>>>(logits, W, n, k, out);
        return;
    }

    int team, cap;
    ext_caps(&team, &cap, n);
    if (team > cap) {
        // row wider than one wave: keep the shipped single-row path per row
        for (int r = 0; r < BS; ++r)
            topk_launch(logits + (size_t)r * (size_t)W, n, k,
                        out + (size_t)r * (size_t)k, stream);
        return;
    }
    if (g_scratch_bs_rows < BS) {
        if (g_scratch_bs) cudaFree(g_scratch_bs);
        cudaMalloc(&g_scratch_bs,
                   (size_t)BS * SCRATCH_WORDS * sizeof(unsigned int));
        cudaMemset(g_scratch_bs, 0,
                   (size_t)BS * SCRATCH_WORDS * sizeof(unsigned int));
        g_scratch_bs_rows = BS;
    }
    int rows_per_wave = cap / team;
    for (int r0 = 0; r0 < BS; r0 += rows_per_wave) {
        const int rows = BS - r0 < rows_per_wave ? BS - r0 : rows_per_wave;
        ++g_gen_bs;
        topk_fast<<<rows * team, BLOCK, 0, stream>>>(
            logits + (size_t)r0 * (size_t)W, W, n, k,
            g_scratch_bs + (size_t)r0 * (size_t)SCRATCH_WORDS,
            out + (size_t)r0 * (size_t)k, g_gen_bs, team);
    }
}
"""
aefm += BATCHED

# ---- E7: top-level ext dispatch appended at end of file ----
TOP = r"""

// Extension dispatch: aefm batched paths; the v30 range (K=2048 mid-n) is out
// of the minimal experiment's scope and falls back to a sequential row loop.
void topk_launch_ext(const float* logits, long W, int n, int k, int* out,
                     int BS, cudaStream_t stream) {
    if (k == 2048 && n > 16896 && n <= 140000) {
        for (int r = 0; r < BS; ++r)
            v30::topk_launch(logits + (size_t)r * (size_t)W, n, k,
                             out + (size_t)r * (size_t)k, stream);
        return;
    }
    aefm::topk_launch_batched(logits, W, n, k, out, BS, stream);
}

// (path, team, cap, rows_per_wave, waves) for a given (n, k, BS).
// path: 0 = small-tier grid.y batch, 1 = row-team, 2 = v30 sequential
// fallback, 3 = row-wider-than-wave sequential fallback.
void topk_ext_info(int n, int k, int BS, int info[5]) {
    if (k == 2048 && n > 16896 && n <= 140000) {
        info[0] = 2; info[1] = 0; info[2] = 0; info[3] = 1; info[4] = BS;
        return;
    }
    if (n <= 16896) {
        info[0] = 0; info[1] = 1; info[2] = 0; info[3] = BS; info[4] = 1;
        return;
    }
    int team, cap;
    aefm::ext_caps(&team, &cap, n);
    if (team > cap) {
        info[0] = 3; info[1] = team; info[2] = cap; info[3] = 1; info[4] = BS;
        return;
    }
    const int rpw = cap / team;
    info[0] = 1; info[1] = team; info[2] = cap; info[3] = rpw;
    info[4] = (BS + rpw - 1) / rpw;
}
"""
rest = rest.rstrip("\n") + TOP

SRC.write_text(aefm + rest)
print("kernel_ext.cu written:", len((aefm + rest).splitlines()), "lines")
