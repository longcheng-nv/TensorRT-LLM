# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Register-diet edits on kernel_ext.cu: templetize topk_fast on
min-blocks-per-SM (__launch_bounds__(BLOCK, MINB) caps regs at
floor(65536/(512*MINB)): 1->none, 2->64, 3->42, 4->32), add per-variant
occupancy caches + max smem carveout, a variant-selecting batched launcher,
and a resource diagnosis entry. Idempotent-hostile: assertions fail loudly
on a second run."""
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


# ---- D1: templetize topk_fast on MINB ----
rep1(
    """__global__ __launch_bounds__(BLOCK, 1)
void topk_fast(const float* __restrict__ logits, long rstride, int n, int k,""",
    """template<int MINB>
__global__ __launch_bounds__(BLOCK, MINB)
void topk_fast(const float* __restrict__ logits, long rstride, int n, int k,""",
)

# ---- D2: existing call sites pin MINB=1 (bit-identical to validated ext) ----
rep1("cudaOccupancyMaxActiveBlocksPerMultiprocessor(\n            &active, topk_fast, BLOCK, 0);\n        if (active < 1) active = 1;\n        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);\n        g_blocks_cap = active * g_sms;\n    }\n    int blocks = (n + 2047) / 2048;",
     "cudaOccupancyMaxActiveBlocksPerMultiprocessor(\n            &active, topk_fast<1>, BLOCK, 0);\n        if (active < 1) active = 1;\n        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);\n        g_blocks_cap = active * g_sms;\n    }\n    int blocks = (n + 2047) / 2048;")
rep1("topk_fast<<<blocks, BLOCK, 0, stream>>>(logits, 0L, n, k, g_scratch, out,"
     " g_gen, blocks);",
     "topk_fast<1><<<blocks, BLOCK, 0, stream>>>(logits, 0L, n, k, g_scratch,"
     " out, g_gen, blocks);")
rep1("cudaOccupancyMaxActiveBlocksPerMultiprocessor(\n            &active, topk_fast, BLOCK, 0);\n        if (active < 1) active = 1;\n        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);\n        g_blocks_cap = active * g_sms;\n    }\n    *team_out",
     "cudaOccupancyMaxActiveBlocksPerMultiprocessor(\n            &active, topk_fast<1>, BLOCK, 0);\n        if (active < 1) active = 1;\n        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);\n        g_blocks_cap = active * g_sms;\n    }\n    *team_out")
rep1("topk_fast<<<rows * team, BLOCK, 0, stream>>>(",
     "topk_fast<1><<<rows * team, BLOCK, 0, stream>>>(")

# ---- D3: per-variant caps + carveout + variant launcher + diagnosis ----
DIET = r"""
// ---------------------------------------------------------------------------
// Register-diet variants (R3_LEDGER BS>1 ext "next lever"): MINB in {1..4}
// via __launch_bounds__(BLOCK, MINB). Per-variant occupancy caches; smem
// carveout raised to MAX once per variant (static smem 40KB/block would
// otherwise cap active at the default carveout).
// ---------------------------------------------------------------------------
static int g_cap_v[5] = {0, 0, 0, 0, 0};

template<int MINB>
static void diet_caps(int* team_out, int* cap_out, int n) {
    if (!g_cap_v[MINB]) {
        cudaFuncSetAttribute(topk_fast<MINB>,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        int active = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, topk_fast<MINB>, BLOCK, 0);
        if (active < 1) active = 1;
        if (!g_sms)
            cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
        g_cap_v[MINB] = active * g_sms;
    }
    *team_out = (n + 2047) / 2048;
    *cap_out = g_cap_v[MINB];
}

template<int MINB>
static void launch_fast_teams(const float* logits, long W, int n, int k,
                              int* out, int BS, cudaStream_t stream) {
    int team, cap;
    diet_caps<MINB>(&team, &cap, n);
    if (team > cap) {
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
    const int rows_per_wave = cap / team;
    for (int r0 = 0; r0 < BS; r0 += rows_per_wave) {
        const int rows = BS - r0 < rows_per_wave ? BS - r0 : rows_per_wave;
        ++g_gen_bs;
        topk_fast<MINB><<<rows * team, BLOCK, 0, stream>>>(
            logits + (size_t)r0 * (size_t)W, W, n, k,
            g_scratch_bs + (size_t)r0 * (size_t)SCRATCH_WORDS,
            out + (size_t)r0 * (size_t)k, g_gen_bs, team);
    }
}

void launch_fast_teams_v(const float* logits, long W, int n, int k, int* out,
                         int BS, int minb, cudaStream_t stream) {
    switch (minb) {
        case 2: launch_fast_teams<2>(logits, W, n, k, out, BS, stream); break;
        case 3: launch_fast_teams<3>(logits, W, n, k, out, BS, stream); break;
        case 4: launch_fast_teams<4>(logits, W, n, k, out, BS, stream); break;
        default: launch_fast_teams<1>(logits, W, n, k, out, BS, stream); break;
    }
}

// numRegs, staticSmemBytes, localBytes, active_default, active_maxcarveout
template<int MINB>
static void fast_stats_t(int out5[5]) {
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, topk_fast<MINB>);
    out5[0] = a.numRegs;
    out5[1] = (int)a.sharedSizeBytes;
    out5[2] = (int)a.localSizeBytes;
    int act = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act, topk_fast<MINB>,
                                                  BLOCK, 0);
    out5[3] = act;
    cudaFuncSetAttribute(topk_fast<MINB>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    act = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act, topk_fast<MINB>,
                                                  BLOCK, 0);
    out5[4] = act;
}

void fast_stats_v(int minb, int out5[5]) {
    switch (minb) {
        case 2: fast_stats_t<2>(out5); break;
        case 3: fast_stats_t<3>(out5); break;
        case 4: fast_stats_t<4>(out5); break;
        default: fast_stats_t<1>(out5); break;
    }
}
"""
aefm += DIET

# ---- D4: top-level variant entries appended at end of file ----
TOP = r"""

// Diet-variant entries. topk_launch_ext_v(minb=1) == topk_launch_ext for
// large-n modulo the smem-carveout attribute (cap may exceed active=2).
void topk_launch_ext_v(const float* logits, long W, int n, int k, int* out,
                       int BS, int minb, cudaStream_t stream) {
    if (k == 2048 && n > 16896 && n <= 140000) {
        for (int r = 0; r < BS; ++r)
            v30::topk_launch(logits + (size_t)r * (size_t)W, n, k,
                             out + (size_t)r * (size_t)k, stream);
        return;
    }
    if (n <= 16896) {
        aefm::topk_launch_batched(logits, W, n, k, out, BS, stream);
        return;
    }
    aefm::launch_fast_teams_v(logits, W, n, k, out, BS, minb, stream);
}

void topk_fast_stats(int minb, int out5[5]) { aefm::fast_stats_v(minb, out5); }
"""
rest = rest.rstrip("\n") + TOP

SRC.write_text(aefm + rest)
print("diet edits applied:", len((aefm + rest).splitlines()), "lines")
