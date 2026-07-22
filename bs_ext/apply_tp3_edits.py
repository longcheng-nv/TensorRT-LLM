# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""tp3 fused single-kernel mid-BS arm: sample + team barrier + collect +
team barrier + CTA0 finish, ONE launch. Kills the 3-launch latency chain
and the underfed finish kernel that make the BS 16-64 valley.

Safety: the two in-kernel barriers are the compB fence-less pattern — the
per-row shist/candbuf lines are never plain-read before the barrier that
publishes them (first plain touch is post-barrier, pre-barrier writes are
L2 atomics / plain stores from OTHER SMs never cached locally), and the
buffers are dedicated to this arm with generation-token senses. CTA0 is
the exclusive owner of the row slice after barrier 2 (all other CTAs have
exited), so its end-of-kernel re-zero is race-free; the load-then-zero
__syncthreads lesson is applied inside CTA0.
launch_bounds(512, 4) register diet (proven zero-spill on topk_fast) keeps
cap = 4*SM so C*BS fits one co-resident wave through BS ~= 512.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "kernel_ext.cu"
text = SRC.read_text()

CUT = "} // namespace aefm"
ci = text.index(CUT)
aefm, rest = text[:ci], text[ci:]

TP3 = r"""
// ---------------------------------------------------------------------------
// tp3: fused single-kernel sampled-single-pass arm for the mid-BS regime.
// Per-row slice in g_tp3: [cand, total, arrive, release] + CAP2 uint2 cands.
// ---------------------------------------------------------------------------
#define TP3_RW (4 + 2 * CAP2)

__global__ __launch_bounds__(512, 4)
void tp3_kernel(const float* __restrict__ logits, long rstride, int n, int k,
                unsigned int* __restrict__ shist,
                unsigned int* __restrict__ d3, int* __restrict__ out,
                unsigned int gen, unsigned int* __restrict__ stats) {
    const int row = blockIdx.y;
    const int c = blockIdx.x;
    const int C = gridDim.x;
    const int tid = threadIdx.x;
    const float* lgr = logits + (size_t)row * (size_t)rstride;
    unsigned int* shist_r = shist + (size_t)row * 2048;
    unsigned int* d3_r = d3 + (size_t)row * TP3_RW;
    unsigned int* cand_write = d3_r + 0;
    unsigned int* g_arrive = d3_r + 2;
    unsigned int* g_release = d3_r + 3;
    uint2* candbuf = reinterpret_cast<uint2*>(d3_r + 4);
    int* outr = out + (size_t)row * (size_t)k;

    __shared__ __align__(16) unsigned int sh_hist[2048];
    __shared__ __align__(16) uint2 arena[ARENA2];
    __shared__ unsigned int warp_totals[16];
    __shared__ int s_bin, s_above, s_cnt, s_tie;

    const int nv4 = n >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(lgr);

    // ---- phase 0: uniform block sampling, blocks interleaved over CTAs ----
    for (int i = tid; i < 2048; i += 512) sh_hist[i] = 0u;
    __syncthreads();
    unsigned int cnt = 0;
    for (int s = c; s * (512 * D2_SSTRIDE) < nv4; s += C) {
        const int i = s * (512 * D2_SSTRIDE) + tid;
        if (i < nv4) {
            float4 f = __ldcs(in4 + i);
            atomicAdd(&sh_hist[fkey(f.x) >> 21], 1u);
            atomicAdd(&sh_hist[fkey(f.y) >> 21], 1u);
            atomicAdd(&sh_hist[fkey(f.z) >> 21], 1u);
            atomicAdd(&sh_hist[fkey(f.w) >> 21], 1u);
            cnt += 4;
        }
    }
    __syncthreads();
    {
        const uint2* sh2 = reinterpret_cast<const uint2*>(sh_hist);
        unsigned long long* g2 =
            reinterpret_cast<unsigned long long*>(shist_r);
        for (int i = tid; i < 1024; i += 512) {
            uint2 v = sh2[i];
            if (v.x | v.y)
                atomicAdd(&g2[i], ((unsigned long long)v.y << 32) | v.x);
        }
    }
    // warp-reduce cnt then one atomic per warp into slice total
    for (int off = 16; off; off >>= 1)
        cnt += __shfl_down_sync(0xffffffffu, cnt, off);
    if ((tid & 31) == 0 && cnt) atomicAdd(d3_r + 1, cnt);
    global_barrier(g_arrive, g_release, C, gen * 8u);

    // ---- phase 1: budget-driven b_safe (each CTA independently) ----
    const int total = (int)d3_r[1];
    int rem_b = (int)(((long long)(CAP2 / 2) * (long long)total)
                      / (long long)n);
    if (rem_b < 1) rem_b = 1;
    if (rem_b > total) rem_b = total;
    find_boundary_bins(shist_r, 2048, warp_totals, &s_bin, &s_above, rem_b);
    const int b_safe = s_bin;

    // ---- phase 2: single full-read candidate collect (contiguous slices) --
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
    global_barrier(g_arrive, g_release, C, gen * 8u + 1u);
    if (c != 0) return;

    // ---- phase 3: CTA0 exact finish (exclusive slice owner from here) ----
    const int Nc = (int)*cand_write;
    __syncthreads();                        // Nc loads done before re-zero
    for (int i = tid; i < 2048; i += 512) shist_r[i] = 0u;
    if (tid == 0) { d3_r[0] = 0u; d3_r[1] = 0u; }

    bool fallback = Nc < k || Nc > CAP2;
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
        const bool wholeC = above_c + Tc == k;
        if (!wholeC && Tc > ARENA2) {
            fallback = true;
        } else {
            if (tid == 0) { s_cnt = 0; s_tie = 0; }
            __syncthreads();
            for (int i = tid; i < Nc; i += 512) {
                const uint2 e = candbuf[i];
                const int d = (int)(e.x >> 21);
                if (d > cb0) {
                    outr[atomicAdd(&s_cnt, 1)] = (int)e.y;
                } else if (d == cb0) {
                    const int s = atomicAdd(&s_tie, 1);
                    if (wholeC) outr[above_c + s] = (int)e.y;
                    else arena[s] = e;
                }
            }
            __syncthreads();
            if (!wholeC)
                d2_refine_arena(arena, Tc, k - above_c, above_c, outr,
                                sh_hist, warp_totals);
            return;
        }
    }

    // rare fallback: full single-CTA recompute (same structure as
    // tp2_finish's fallback; row re-reads)
    if (tid == 0) atomicAdd(stats, 1u);
    for (int i = tid; i < 2048; i += 512) sh_hist[i] = 0u;
    __syncthreads();
    for (int i = tid; i < nv4; i += 512) {
        float4 f = __ldcs(in4 + i);
        atomicAdd(&sh_hist[fkey(f.x) >> 21], 1u);
        atomicAdd(&sh_hist[fkey(f.y) >> 21], 1u);
        atomicAdd(&sh_hist[fkey(f.z) >> 21], 1u);
        atomicAdd(&sh_hist[fkey(f.w) >> 21], 1u);
    }
    for (int i = (nv4 << 2) + tid; i < n; i += 512)
        atomicAdd(&sh_hist[fkey(__ldcs(lgr + i)) >> 21], 1u);
    __syncthreads();
    find_boundary_bins(sh_hist, 2048, warp_totals, &s_bin, &s_above, k);
    const int b0 = s_bin;
    const int above0 = s_above;
    const int T = (int)sh_hist[b0];
    const int R = k - above0;
    const bool whole = T == R;
    const bool arena_ok = T <= ARENA2;
    __syncthreads();
    if (tid == 0) { s_cnt = 0; s_tie = 0; }
    __syncthreads();
    for (int i = tid; i < nv4; i += 512) {
        float4 f = __ldcs(in4 + i);
        unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z), fkey(f.w)};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = (int)(kk[j] >> 21);
            if (d > b0) {
                outr[atomicAdd(&s_cnt, 1)] = 4 * i + j;
            } else if (d == b0) {
                const int s = atomicAdd(&s_tie, 1);
                if (whole) outr[above0 + s] = 4 * i + j;
                else if (arena_ok)
                    arena[s] = make_uint2(kk[j], (unsigned int)(4 * i + j));
            }
        }
    }
    for (int i = (nv4 << 2) + tid; i < n; i += 512) {
        const unsigned int key = fkey(__ldcs(lgr + i));
        const int d = (int)(key >> 21);
        if (d > b0) {
            outr[atomicAdd(&s_cnt, 1)] = i;
        } else if (d == b0) {
            const int s = atomicAdd(&s_tie, 1);
            if (whole) outr[above0 + s] = i;
            else if (arena_ok) arena[s] = make_uint2(key, (unsigned int)i);
        }
    }
    __syncthreads();
    if (whole) return;
    if (arena_ok) {
        d2_refine_arena(arena, T, R, above0, outr, sh_hist, warp_totals);
        return;
    }
    // massive-tie in-CTA ladder (row re-reads) — same as tp2_finish
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

static unsigned int* g_tp3_shist = nullptr;
static unsigned int* g_tp3 = nullptr;
static unsigned int* g_tp3_stats = nullptr;
static int g_tp3_rows = 0;
static int g_tp3_cap = 0;
static unsigned int g_tp3_gen = 0;

void tp3_launch(const float* logits, long W, int n, int k, int* out, int BS,
                cudaStream_t stream) {
    if (!g_tp3_cap) {
        cudaFuncSetAttribute(tp3_kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        int active = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, tp3_kernel, 512, 0);
        if (active < 1) active = 1;
        if (!g_sms)
            cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
        g_tp3_cap = active * g_sms;
    }
    int C = g_tp3_cap / BS;
    if (C > 32) C = 32;
    if (C < 1) {                 // BS beyond one co-resident wave: not tp3's
        tp2_launch(logits, W, n, k, out, BS, stream);   // regime — defer
        return;
    }
    if (g_tp3_rows < BS) {
        if (g_tp3_shist) cudaFree(g_tp3_shist);
        if (g_tp3) cudaFree(g_tp3);
        cudaMalloc(&g_tp3_shist, (size_t)BS * 2048 * sizeof(unsigned int));
        cudaMalloc(&g_tp3, (size_t)BS * TP3_RW * sizeof(unsigned int));
        cudaMemset(g_tp3_shist, 0, (size_t)BS * 2048 * sizeof(unsigned int));
        cudaMemset(g_tp3, 0, (size_t)BS * TP3_RW * sizeof(unsigned int));
        g_tp3_rows = BS;
    }
    if (!g_tp3_stats) {
        cudaMalloc(&g_tp3_stats, sizeof(unsigned int));
        cudaMemset(g_tp3_stats, 0, sizeof(unsigned int));
    }
    ++g_tp3_gen;
    tp3_kernel<<<dim3((unsigned int)C, (unsigned int)BS), 512, 0, stream>>>(
        logits, W, n, k, g_tp3_shist, g_tp3, out, g_tp3_gen, g_tp3_stats);
}

unsigned int tp3_read_reset_fallbacks() {
    if (!g_tp3_stats) return 0u;
    unsigned int v = 0;
    cudaMemcpy(&v, g_tp3_stats, sizeof(v), cudaMemcpyDeviceToHost);
    cudaMemset(g_tp3_stats, 0, sizeof(v));
    return v;
}

void tp3_stats_out(int out3[3]) {
    if (!g_tp3_cap) {
        cudaFuncSetAttribute(tp3_kernel,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        int active = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, tp3_kernel, 512, 0);
        if (active < 1) active = 1;
        if (!g_sms)
            cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
        g_tp3_cap = active * g_sms;
    }
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, tp3_kernel);
    out3[0] = a.numRegs;
    out3[1] = (int)a.localSizeBytes;
    out3[2] = g_tp3_cap;
}
"""
aefm += TP3

TOP = r"""

// tp3 fused single-kernel entries.
void topk_launch_tp3(const float* logits, long W, int n, int k, int* out,
                     int BS, cudaStream_t stream) {
    if (n <= 16896) {
        aefm::topk_launch_batched(logits, W, n, k, out, BS, stream);
        return;
    }
    aefm::tp3_launch(logits, W, n, k, out, BS, stream);
}

unsigned int topk_tp3_fallbacks() { return aefm::tp3_read_reset_fallbacks(); }
void topk_tp3_stats(int out3[3]) { aefm::tp3_stats_out(out3); }
"""
rest = rest.rstrip("\n") + TOP

SRC.write_text(aefm + rest)
print("tp3 edits applied:", len((aefm + rest).splitlines()), "lines")
