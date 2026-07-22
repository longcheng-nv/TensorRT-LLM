# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tp4: exact-hist fused 2-pass arm for the mid-BS regime.

Rationale (cold-L2 verdicts): at BS 16-64 the GPU is underfed, DRAM passes
are cheap (~2.4us/16.8MB) and LATENCY CHAINS are expensive. tp3's sampled
single-pass saves bytes but pays: sample phase, b_safe estimation, a full
candbuf write+read round trip, and fallback machinery. tp4 spends one extra
full read to know the EXACT boundary, then the second pass writes outputs
DIRECTLY (no candbuf):

  phase A: sliced full read -> smem 11-bit hist -> packed atomic merge into
           per-row global hist
  barrier; every CTA finds the exact b0/above0/T (plain loads: first touch
           of the merged lines is post-barrier -> fence-less legs hold)
  phase B: sliced re-read: bucket > b0 -> smem gt staging, ONE bulk cursor
           reservation per CTA -> outr; bucket == b0 -> whole? direct outr
           slots : global tiestage (bulk-staged as well; overflow falls
           back to direct per-hit cursor atomics, ranges stay disjoint)
  barrier; CTA0: tie refine over <= TP4_TIE staged pairs (compB fast-tail
           via d2_refine_arena) or in-CTA ladder for massive-tie rows;
           re-zeroes slice state.

Parallel-finish lesson applied: the refine stays on CTA0 (tiny); only the
BULK data motion (the two passes + emit) is parallel.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "kernel_ext.cu"
text = SRC.read_text()

CUT = "} // namespace aefm"
ci = text.index(CUT)
aefm, rest = text[:ci], text[ci:]

TP4 = r"""
// ---------------------------------------------------------------------------
// tp4: exact-hist fused 2-pass mid-BS arm. Slice: [arrive,release,gtcur,
// tiecur] + hist 2048 + tiestage TP4_TIE uint2.
// ---------------------------------------------------------------------------
#define TP4_TIE 2048
#define TP4_RW (4 + 2048 + 2 * TP4_TIE)

__global__ __launch_bounds__(512, 4)
void tp4_kernel(const float* __restrict__ logits, long rstride, int n, int k,
                unsigned int* __restrict__ d4, int* __restrict__ out,
                unsigned int gen, unsigned int* __restrict__ stats) {
    const int row = blockIdx.y;
    const int c = blockIdx.x;
    const int C = gridDim.x;
    const int tid = threadIdx.x;
    const float* lgr = logits + (size_t)row * (size_t)rstride;
    unsigned int* d4_r = d4 + (size_t)row * TP4_RW;
    unsigned int* g_arrive = d4_r + 0;
    unsigned int* g_release = d4_r + 1;
    unsigned int* gt_cur = d4_r + 2;
    unsigned int* tie_cur = d4_r + 3;
    unsigned int* hist_r = d4_r + 4;
    uint2* tiestage = reinterpret_cast<uint2*>(d4_r + 4 + 2048);
    int* outr = out + (size_t)row * (size_t)k;

    __shared__ __align__(16) unsigned int sh_hist[2048];
    __shared__ __align__(16) uint2 arena[TP4_TIE];
    __shared__ unsigned int warp_totals[16];
    __shared__ int s_bin, s_above, s_gt_n, s_tie_n, s_cnt, s_tie;
    __shared__ unsigned int s_gt_base, s_tie_base;

    const int nv4 = n >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(lgr);
    const int q = nv4 / C, r = nv4 % C;
    const int beg = c * q + min(c, r);
    const int end = beg + q + (c < r ? 1 : 0);

    // ---- phase A: sliced full read -> exact per-row hist ----
    for (int i = tid; i < 2048; i += 512) sh_hist[i] = 0u;
    __syncthreads();
    for (int i = beg + tid; i < end; i += 512) {
        float4 f = __ldcs(in4 + i);
        atomicAdd(&sh_hist[fkey(f.x) >> 21], 1u);
        atomicAdd(&sh_hist[fkey(f.y) >> 21], 1u);
        atomicAdd(&sh_hist[fkey(f.z) >> 21], 1u);
        atomicAdd(&sh_hist[fkey(f.w) >> 21], 1u);
    }
    if (c == C - 1) {
        for (int i = (nv4 << 2) + tid; i < n; i += 512)
            atomicAdd(&sh_hist[fkey(__ldcs(lgr + i)) >> 21], 1u);
    }
    __syncthreads();
    {
        const uint2* sh2 = reinterpret_cast<const uint2*>(sh_hist);
        unsigned long long* g2 = reinterpret_cast<unsigned long long*>(hist_r);
        for (int i = tid; i < 1024; i += 512) {
            uint2 v = sh2[i];
            if (v.x | v.y)
                atomicAdd(&g2[i], ((unsigned long long)v.y << 32) | v.x);
        }
    }
    global_barrier(g_arrive, g_release, C, gen * 8u);

    // ---- exact boundary (every CTA; first plain touch is post-barrier) ---
    find_boundary_bins(hist_r, 2048, warp_totals, &s_bin, &s_above, k);
    const int b0 = s_bin;
    const int above0 = s_above;
    const int T = (int)hist_r[b0];
    const int R = k - above0;
    const bool whole = T == R;
    const bool tie_ok = T <= TP4_TIE;

    // ---- phase B: sliced re-read, direct emit ----
    if (tid == 0) { s_gt_n = 0; s_tie_n = 0; }
    __syncthreads();
    for (int i = beg + tid; i < end; i += 512) {
        float4 f = __ldcs(in4 + i);
        const unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z),
                                    fkey(f.w)};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = (int)(kk[j] >> 21);
            if (d > b0) {
                const int sp = atomicAdd(&s_gt_n, 1);
                // gt per chunk <= above0 < k <= 2048 == arena size
                arena[sp] = make_uint2(kk[j], (unsigned int)(4 * i + j));
            } else if (d == b0 && tie_ok) {
                // ties are few (T <= TP4_TIE, real rows ~100-1200): direct
                // per-hit global cursor is fine; big-T rows go to the ladder
                const unsigned int s = atomicAdd(tie_cur, 1u);
                if (whole) {
                    if ((int)s < R) outr[above0 + s] = 4 * i + j;
                } else if ((int)s < TP4_TIE) {
                    tiestage[s] = make_uint2(kk[j], (unsigned int)(4 * i + j));
                }
            }
        }
    }
    if (c == C - 1) {
        for (int i = (nv4 << 2) + tid; i < n; i += 512) {
            const unsigned int key = fkey(__ldcs(lgr + i));
            const int d = (int)(key >> 21);
            if (d > b0) {
                const int sp = atomicAdd(&s_gt_n, 1);
                arena[sp] = make_uint2(key, (unsigned int)i);
            } else if (d == b0 && tie_ok) {
                const unsigned int s = atomicAdd(tie_cur, 1u);
                if (whole) {
                    if ((int)s < R) outr[above0 + s] = i;
                } else if ((int)s < TP4_TIE) {
                    tiestage[s] = make_uint2(key, (unsigned int)i);
                }
            }
        }
    }
    __syncthreads();
    if (tid == 0) s_gt_base = atomicAdd(gt_cur, (unsigned int)s_gt_n);
    __syncthreads();
    for (int i = tid; i < s_gt_n; i += 512)
        outr[s_gt_base + (unsigned int)i] = (int)arena[i].y;
    global_barrier(g_arrive, g_release, C, gen * 8u + 1u);
    if (c != 0) return;

    // ---- CTA0 tail: tie refine + slice re-zero ----
    __syncthreads();
    for (int i = tid; i < 2048; i += 512) hist_r[i] = 0u;
    if (tid == 0) { *gt_cur = 0u; *tie_cur = 0u; }
    if (whole) return;
    if (tie_ok) {
        for (int i = tid; i < T; i += 512) arena[i] = tiestage[i];
        __syncthreads();
        d2_refine_arena(arena, T, R, above0, outr, sh_hist, warp_totals);
        return;
    }
    // massive-tie fallback: in-CTA 11/11/10 ladder over row re-reads
    if (tid == 0) atomicAdd(stats, 1u);
    int remaining = R;
    int total_above = above0;
    unsigned int prefix = ((unsigned int)b0) << 21;
    int consumed = 11;
#pragma unroll
    for (int pass = 1; pass < 3; ++pass) {
        const int shift = pass == 1 ? 10 : 0;
        const int bits = pass == 1 ? 11 : 10;
        const int nb = 1 << bits;
        const unsigned int digit_mask = (unsigned int)(nb - 1);
        const unsigned int high_mask = 0xffffffffu << (shift + bits);
        for (int i = tid; i < nb; i += 512) sh_hist[i] = 0u;
        __syncthreads();
        for (int i = tid; i < nv4; i += 512) {
            float4 f = __ldcs(in4 + i);
            unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z), fkey(f.w)};
#pragma unroll
            for (int j = 0; j < 4; ++j)
                if ((kk[j] & high_mask) == (prefix & high_mask))
                    atomicAdd(&sh_hist[(kk[j] >> shift) & digit_mask], 1u);
        }
        for (int i = (nv4 << 2) + tid; i < n; i += 512) {
            unsigned int key = fkey(__ldcs(lgr + i));
            if ((key & high_mask) == (prefix & high_mask))
                atomicAdd(&sh_hist[(key >> shift) & digit_mask], 1u);
        }
        __syncthreads();
        find_boundary_bins(sh_hist, nb, warp_totals, &s_bin, &s_above,
                           remaining);
        bool whole_bucket = s_above + (int)sh_hist[s_bin] == remaining;
        prefix |= ((unsigned int)s_bin) << shift;
        total_above += s_above;
        remaining -= s_above;
        consumed += bits;
        __syncthreads();
        if (whole_bucket) break;
    }
    const int final_shift = 32 - consumed;
    const unsigned int threshold =
        final_shift ? (prefix >> final_shift) : prefix;
    if (tid == 0) { s_cnt = 0; s_tie = 0; }
    __syncthreads();
    for (int i = tid; i < nv4; i += 512) {
        float4 f = __ldcs(in4 + i);
        unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z), fkey(f.w)};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int key = kk[j];
            if ((int)(key >> 21) != b0) continue;
            if (final_shift) key >>= final_shift;
            if (key > threshold) {
                outr[above0 + atomicAdd(&s_cnt, 1)] = 4 * i + j;
            } else if (key == threshold) {
                int slot = atomicAdd(&s_tie, 1);
                if (slot < remaining) outr[total_above + slot] = 4 * i + j;
            }
        }
    }
    for (int i = (nv4 << 2) + tid; i < n; i += 512) {
        unsigned int key = fkey(__ldcs(lgr + i));
        if ((int)(key >> 21) != b0) continue;
        if (final_shift) key >>= final_shift;
        if (key > threshold) {
            outr[above0 + atomicAdd(&s_cnt, 1)] = i;
        } else if (key == threshold) {
            int slot = atomicAdd(&s_tie, 1);
            if (slot < remaining) outr[total_above + slot] = i;
        }
    }
}

static unsigned int* g_tp4 = nullptr;
static unsigned int* g_tp4_stats = nullptr;
static int g_tp4_rows = 0;
static int g_tp4_cap = 0;
static unsigned int g_tp4_gen = 0;

void tp4_launch(const float* logits, long W, int n, int k, int* out, int BS,
                cudaStream_t stream) {
    if (!g_tp4_cap) {
        cudaFuncSetAttribute(tp4_kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        int active = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, tp4_kernel, 512, 0);
        if (active < 1) active = 1;
        if (!g_sms)
            cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
        g_tp4_cap = active * g_sms;
    }
    int C = g_tp4_cap / BS;
    if (C > 32) C = 32;
    if (C < 1) {
        tp2_launch(logits, W, n, k, out, BS, stream);
        return;
    }
    if (g_tp4_rows < BS) {
        if (g_tp4) cudaFree(g_tp4);
        cudaMalloc(&g_tp4, (size_t)BS * TP4_RW * sizeof(unsigned int));
        cudaMemset(g_tp4, 0, (size_t)BS * TP4_RW * sizeof(unsigned int));
        g_tp4_rows = BS;
    }
    if (!g_tp4_stats) {
        cudaMalloc(&g_tp4_stats, sizeof(unsigned int));
        cudaMemset(g_tp4_stats, 0, sizeof(unsigned int));
    }
    ++g_tp4_gen;
    tp4_kernel<<<dim3((unsigned int)C, (unsigned int)BS), 512, 0, stream>>>(
        logits, W, n, k, g_tp4, out, g_tp4_gen, g_tp4_stats);
}

unsigned int tp4_read_reset_fallbacks() {
    if (!g_tp4_stats) return 0u;
    unsigned int v = 0;
    cudaMemcpy(&v, g_tp4_stats, sizeof(v), cudaMemcpyDeviceToHost);
    cudaMemset(g_tp4_stats, 0, sizeof(v));
    return v;
}
"""
aefm += TP4

TOP = r"""

// tp4 + unified dispatcher entries.
void topk_launch_tp4(const float* logits, long W, int n, int k, int* out,
                     int BS, cudaStream_t stream) {
    if (n <= 16896) {
        aefm::topk_launch_batched(logits, W, n, k, out, BS, stream);
        return;
    }
    aefm::tp4_launch(logits, W, n, k, out, BS, stream);
}

unsigned int topk_tp4_fallbacks() { return aefm::tp4_read_reset_fallbacks(); }
void topk_set_tp4_max_bs(int v) { aefm::g_tp4_max_bs = v; }

void topk_launch_auto(const float* logits, long W, int n, int k, int* out,
                      int BS, cudaStream_t stream) {
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
    switch (aefm::auto_pick(n, BS)) {
        case 0: aefm::launch_fast_teams_v(logits, W, n, k, out, BS, 4,
                                          stream); break;
        case 1: aefm::tp4_launch(logits, W, n, k, out, BS, stream); break;
        case 2: aefm::tp3_launch(logits, W, n, k, out, BS, stream); break;
        default: aefm::tp2_launch(logits, W, n, k, out, BS, stream); break;
    }
}

int topk_auto_pick(int n, int BS) { return aefm::auto_pick(n, BS); }
"""
rest = rest.rstrip("\n") + TOP

SRC.write_text(aefm + rest)
print("tp4 edits applied:", len((aefm + rest).splitlines()), "lines")
