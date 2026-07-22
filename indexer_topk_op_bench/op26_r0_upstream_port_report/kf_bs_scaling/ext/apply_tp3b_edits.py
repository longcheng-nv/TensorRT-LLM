# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tp3 stage-B optimizations (from the BS=16/32 nsys phase budget:
sample 2.2 + collect 12.6 + finish 7.5 us):

B1. Collect ILP: 4 independent float4 loads per thread per iteration
    (collect was 0.7-1.3 TB/s — short dependent loops, latency-bound).
B2. Fused candidate hist: collect accumulates the candidates' MSB hist in
    smem (sh_hist is free after the phase-0 merge) and merges it into a
    per-row global cand-hist; finish reads it directly and skips its whole
    hist-over-candbuf pass. Slice grows by 2048 words (own buffer — the
    sample hist cannot be reused: a fast CTA would overwrite it while a
    slow CTA is still reading b_safe).
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "kernel_ext.cu"
t = SRC.read_text()


def rep1(old, new):
    global t
    assert t.count(old) == 1, f"expected 1 match:\n{old[:100]}"
    t = t.replace(old, new, 1)


# ---- slice layout: + 2048-word candidate hist ----
rep1("#define TP3_RW (4 + 2 * CAP2)",
     "#define TP3_RW (4 + 2 * CAP2 + 2048)   // + per-row candidate MSB hist")

# ---- collect phase: ILP x4 + smem cand-hist + merge ----
rep1(
    """    // ---- phase 2: single full-read candidate collect (contiguous slices) --
    {
        const int q = nv4 / C, r = nv4 % C;
        const int beg = c * q + min(c, r);
        const int end = beg + q + (c < r ? 1 : 0);
        for (int i = beg + tid; i < end; i += 512) {
            float4 f = __ldcs(in4 + i);
            unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z), fkey(f.w)};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if ((int)(kk[j] >> 21) > b_safe) {
                    const unsigned int s = atomicAdd(cand_write, 1u);
                    if (s < CAP2)
                        candbuf[s] =
                            make_uint2(kk[j], (unsigned int)(4 * i + j));
                }
            }
        }
        if (c == C - 1) {
            for (int i = (nv4 << 2) + tid; i < n; i += 512) {
                const unsigned int key = fkey(__ldcs(lgr + i));
                if ((int)(key >> 21) > b_safe) {
                    const unsigned int s = atomicAdd(cand_write, 1u);
                    if (s < CAP2)
                        candbuf[s] = make_uint2(key, (unsigned int)i);
                }
            }
        }
    }
    global_barrier(g_arrive, g_release, C, gen * 8u + 1u);""",
    """    // ---- phase 2: single full-read candidate collect (contiguous slices,
    //      4x float4 ILP) + fused candidate MSB hist (sh_hist is free) ----
    unsigned int* candhist = d3_r + 4 + 2 * CAP2;
    for (int i = tid; i < 2048; i += 512) sh_hist[i] = 0u;
    __syncthreads();
    {
        const int q = nv4 / C, r = nv4 % C;
        const int beg = c * q + min(c, r);
        const int end = beg + q + (c < r ? 1 : 0);
        int i = beg + tid;
        for (; i + 3 * 512 < end; i += 4 * 512) {
            const float4 f0 = __ldcs(in4 + i);
            const float4 f1 = __ldcs(in4 + i + 512);
            const float4 f2 = __ldcs(in4 + i + 1024);
            const float4 f3 = __ldcs(in4 + i + 1536);
            const float4 ff[4] = {f0, f1, f2, f3};
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                const unsigned int kk[4] = {fkey(ff[u].x), fkey(ff[u].y),
                                            fkey(ff[u].z), fkey(ff[u].w)};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if ((int)(kk[j] >> 21) > b_safe) {
                        const unsigned int s = atomicAdd(cand_write, 1u);
                        if (s < CAP2)
                            candbuf[s] = make_uint2(
                                kk[j],
                                (unsigned int)(4 * (i + u * 512) + j));
                        atomicAdd(&sh_hist[kk[j] >> 21], 1u);
                    }
                }
            }
        }
        for (; i < end; i += 512) {
            const float4 f = __ldcs(in4 + i);
            const unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z),
                                        fkey(f.w)};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if ((int)(kk[j] >> 21) > b_safe) {
                    const unsigned int s = atomicAdd(cand_write, 1u);
                    if (s < CAP2)
                        candbuf[s] =
                            make_uint2(kk[j], (unsigned int)(4 * i + j));
                    atomicAdd(&sh_hist[kk[j] >> 21], 1u);
                }
            }
        }
        if (c == C - 1) {
            for (int i2 = (nv4 << 2) + tid; i2 < n; i2 += 512) {
                const unsigned int key = fkey(__ldcs(lgr + i2));
                if ((int)(key >> 21) > b_safe) {
                    const unsigned int s = atomicAdd(cand_write, 1u);
                    if (s < CAP2)
                        candbuf[s] = make_uint2(key, (unsigned int)i2);
                    atomicAdd(&sh_hist[key >> 21], 1u);
                }
            }
        }
    }
    __syncthreads();
    {
        const uint2* sh2 = reinterpret_cast<const uint2*>(sh_hist);
        unsigned long long* g2 = reinterpret_cast<unsigned long long*>(candhist);
        for (int i = tid; i < 1024; i += 512) {
            uint2 v = sh2[i];
            if (v.x | v.y)
                atomicAdd(&g2[i], ((unsigned long long)v.y << 32) | v.x);
        }
    }
    global_barrier(g_arrive, g_release, C, gen * 8u + 1u);""")

# ---- finish: read fused cand-hist, skip the hist-over-candbuf pass ----
rep1(
    """    bool fallback = Nc < k || Nc > CAP2;
    if (!fallback) {
        for (int i = tid; i < 2048; i += 512) sh_hist[i] = 0u;
        __syncthreads();
        for (int i = tid; i < Nc; i += 512)
            atomicAdd(&sh_hist[candbuf[i].x >> 21], 1u);
        __syncthreads();
        find_boundary_bins(sh_hist, 2048, warp_totals, &s_bin, &s_above, k);
        const int cb0 = s_bin;
        const int above_c = s_above;
        const int Tc = (int)sh_hist[cb0];
        const bool wholeC = above_c + Tc == k;""",
    """    bool fallback = Nc < k || Nc > CAP2;
    if (!fallback) {
        // candidate hist was fused into the collect phase
        find_boundary_bins(candhist, 2048, warp_totals, &s_bin, &s_above, k);
        const int cb0 = s_bin;
        const int above_c = s_above;
        const int Tc = (int)candhist[cb0];
        const bool wholeC = above_c + Tc == k;
        __syncthreads();                    // Tc loads before candhist zero
        for (int i = tid; i < 2048; i += 512) candhist[i] = 0u;""")

# finish's fallback + end-of-kernel zeroing must also clear candhist when
# the common path was not taken; zero it in the shared prologue instead is
# wrong (loads pending) — add zero to the fallback entry.
rep1(
    """    // rare fallback: full single-CTA recompute (same structure as
    // tp2_finish's fallback; row re-reads)
    if (tid == 0) atomicAdd(stats, 1u);""",
    """    // rare fallback: full single-CTA recompute (same structure as
    // tp2_finish's fallback; row re-reads)
    if (tid == 0) atomicAdd(stats, 1u);
    __syncthreads();
    for (int i = tid; i < 2048; i += 512) candhist[i] = 0u;""")

SRC.write_text(t)
print("tp3 stage-B edits applied")
