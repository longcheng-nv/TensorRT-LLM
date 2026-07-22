# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D2 sampled-estimate single-pass edits: append the tp2 pipeline to
kernel_ext.cu (aefm namespace).

  tp2_sample : block-sampled (~1/16) 11-bit hist per row + sampled-total
               count (robust threshold scaling under slice quantization).
  tp2_collect: ONE full row read; b_safe = (sampled boundary) - DELTA;
               all elements with bucket > b_safe -> (key,idx) candbuf.
               Invariant: cand_count >= k  <=>  top-k subset of candidates.
  tp2_finish : per-row single CTA. Common path: exact top-k among <= CAP2
               candidates (MSB hist + arena refine, compB fast-tail).
               Fallback (cand_count < k or > CAP2, or cand boundary bucket
               > ARENA): full in-CTA recompute with row re-reads (rare,
               counted in a global stats word). Re-zeroes all per-row state.

Expected bytes: ~1/16 (sample) + 1.0 (collect) + fallback_rate * ~2
vs D1's 2.0. Race-lesson hygiene: every plain-load of state that the same
kernel later zeroes is fenced with __syncthreads first.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "kernel_ext.cu"
text = SRC.read_text()

CUT = "} // namespace aefm"
ci = text.index(CUT)
aefm, rest = text[:ci], text[ci:]

D2 = r"""
// ---------------------------------------------------------------------------
// D2 sampled-estimate single-pass arm (tp2). See RESULTS.md D2 section.
// ---------------------------------------------------------------------------
#define CAP2 8192               // candidate buffer entries per row
#define ARENA2 4096             // smem refine arena entries
#define D2_RW (4 + 2 * CAP2)    // per-row words in g_d2: [cand,flag,total,pad]
#define D2_DELTA 2              // b_safe = b_est - D2_DELTA
#define D2_SSTRIDE 16           // sample 512-float4 blocks at 1/16 duty

__global__ __launch_bounds__(512, 1)
void tp2_sample(const float* __restrict__ logits, long rstride, int n, int C,
                unsigned int* __restrict__ shist,
                unsigned int* __restrict__ d2) {
    const int row = blockIdx.y;
    const int slice = blockIdx.x;
    const int tid = threadIdx.x;
    const float* lgr = logits + (size_t)row * (size_t)rstride;
    unsigned int* shist_r = shist + (size_t)row * 2048;
    unsigned int* d2_r = d2 + (size_t)row * D2_RW;

    __shared__ unsigned int sh[2048];
    __shared__ unsigned int s_total;
    for (int i = tid; i < 2048; i += 512) sh[i] = 0u;
    if (tid == 0) s_total = 0u;
    __syncthreads();

    const int nv4 = n >> 2;
    const int q = nv4 / C, r = nv4 % C;
    const int beg = slice * q + min(slice, r);
    const int end = beg + q + (slice < r ? 1 : 0);
    const float4* in4 = reinterpret_cast<const float4*>(lgr);
    unsigned int cnt = 0;
    for (int i = beg + tid; i < end; i += 512 * D2_SSTRIDE) {
        float4 f = __ldcs(in4 + i);
        atomicAdd(&sh[fkey(f.x) >> 21], 1u);
        atomicAdd(&sh[fkey(f.y) >> 21], 1u);
        atomicAdd(&sh[fkey(f.z) >> 21], 1u);
        atomicAdd(&sh[fkey(f.w) >> 21], 1u);
        cnt += 4;
    }
    if (cnt) atomicAdd(&s_total, cnt);
    __syncthreads();
    const uint2* sh2 = reinterpret_cast<const uint2*>(sh);
    unsigned long long* g2 = reinterpret_cast<unsigned long long*>(shist_r);
    for (int i = tid; i < 1024; i += 512) {
        uint2 c = sh2[i];
        if (c.x | c.y)
            atomicAdd(&g2[i], ((unsigned long long)c.y << 32) | c.x);
    }
    if (tid == 0 && s_total) atomicAdd(&d2_r[2], s_total);
}

__global__ __launch_bounds__(512, 1)
void tp2_collect(const float* __restrict__ logits, long rstride, int n, int k,
                 int C, const unsigned int* __restrict__ shist,
                 unsigned int* __restrict__ d2) {
    const int row = blockIdx.y;
    const int slice = blockIdx.x;
    const int tid = threadIdx.x;
    const float* lgr = logits + (size_t)row * (size_t)rstride;
    const unsigned int* shist_r = shist + (size_t)row * 2048;
    unsigned int* d2_r = d2 + (size_t)row * D2_RW;
    unsigned int* cand_write = d2_r + 0;
    uint2* candbuf = reinterpret_cast<uint2*>(d2_r + 4);

    __shared__ unsigned int warp_totals[16];
    __shared__ int s_bin, s_above;
    const int total = (int)d2_r[2];
    int rem_s = (int)(((long long)k * (long long)total) / (long long)n);
    if (rem_s < 1) rem_s = 1;
    if (rem_s > total) rem_s = total;
    find_boundary_bins(shist_r, 2048, warp_totals, &s_bin, &s_above, rem_s);
    const int b_safe = s_bin - D2_DELTA;   // < 0 => everything is a candidate
                                           // => guaranteed overflow fallback

    const int nv4 = n >> 2;
    const int q = nv4 / C, r = nv4 % C;
    const int beg = slice * q + min(slice, r);
    const int end = beg + q + (slice < r ? 1 : 0);
    const float4* in4 = reinterpret_cast<const float4*>(lgr);
    for (int i = beg + tid; i < end; i += 512) {
        float4 f = __ldcs(in4 + i);
        unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z), fkey(f.w)};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if ((int)(kk[j] >> 21) > b_safe) {
                const unsigned int s = atomicAdd(cand_write, 1u);
                if (s < CAP2)
                    candbuf[s] = make_uint2(kk[j], (unsigned int)(4 * i + j));
            }
        }
    }
    if (slice == C - 1) {
        for (int i = (nv4 << 2) + tid; i < n; i += 512) {
            const unsigned int key = fkey(__ldcs(lgr + i));
            if ((int)(key >> 21) > b_safe) {
                const unsigned int s = atomicAdd(cand_write, 1u);
                if (s < CAP2) candbuf[s] = make_uint2(key, (unsigned int)i);
            }
        }
    }
}

// Exact top-R refine of Tc <= ARENA2 same-MSB-bucket (key, idx) pairs held
// in the shared arena, writing outr[base .. base+R). compB fast-tail
// structure (pass A bits 20..10, then warp-serial or pass B bits 9..0).
__device__ __forceinline__ void d2_refine_arena(
        const uint2* arena, int Tc, int R, int base, int* __restrict__ outr,
        unsigned int* sh_hist, unsigned int* warp_totals) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    __shared__ int a_bin, a_above, a_cnt, a_ntie;
    __shared__ unsigned int a_eqk[32];
    __shared__ int a_eqv[32];
    for (int i = tid; i < 2048; i += 512) sh_hist[i] = 0u;
    __syncthreads();
    for (int i = tid; i < Tc; i += 512)
        atomicAdd(&sh_hist[(arena[i].x >> 10) & 2047u], 1u);
    __syncthreads();
    find_boundary_bins(sh_hist, 2048, warp_totals, &a_bin, &a_above, R);
    const int bA = a_bin;
    const int aboveA = a_above;
    const int TA = (int)sh_hist[bA];
    const bool wholeA = aboveA + TA == R;
    if (tid == 0) { a_cnt = 0; a_ntie = 0; }
    __syncthreads();
    for (int i = tid; i < Tc; i += 512) {
        const uint2 e = arena[i];
        const int binA = (int)((e.x >> 10) & 2047u);
        if (wholeA ? (binA >= bA) : (binA > bA)) {
            outr[base + atomicAdd(&a_cnt, 1)] = (int)e.y;
        } else if (!wholeA && binA == bA) {
            int qi = atomicAdd(&a_ntie, 1);
            if (qi < 32) { a_eqk[qi] = e.x; a_eqv[qi] = (int)e.y; }
        }
    }
    __syncthreads();
    if (wholeA) return;
    const int R2 = R - aboveA;
    if (TA <= 32) {
        if (tid < 32) {
            unsigned int score = tid < TA ? ((a_eqk[tid] & 1023u) + 1u) : 0u;
            for (int rr = 0; rr < R2; ++rr) {
                unsigned int bs = score;
                int bl = tid;
#pragma unroll
                for (int off = 16; off; off >>= 1) {
                    unsigned int os = __shfl_down_sync(0xffffffffu, bs, off);
                    int ol = __shfl_down_sync(0xffffffffu, bl, off);
                    if (tid + off < 32 && os > bs) { bs = os; bl = ol; }
                }
                int winner = __shfl_sync(0xffffffffu, bl, 0);
                if (tid == winner) {
                    outr[base + aboveA + rr] = a_eqv[tid];
                    score = 0u;
                }
            }
        }
        return;
    }
    // big-tie pass B over bits 9..0
    __syncthreads();
    for (int i = tid; i < 1024; i += 512) sh_hist[i] = 0u;
    __syncthreads();
    for (int i = tid; i < Tc; i += 512) {
        const uint2 e = arena[i];
        if ((int)((e.x >> 10) & 2047u) == bA)
            atomicAdd(&sh_hist[e.x & 1023u], 1u);
    }
    __syncthreads();
    find_boundary_bins(sh_hist, 1024, warp_totals, &a_bin, &a_above, R2);
    const int bB = a_bin;
    // gt and exact-threshold ties share the running a_cnt cursor
    // (a_cnt == aboveA here from pass A); ties fill any remaining slot < R.
    for (int i = tid; i < Tc; i += 512) {
        const uint2 e = arena[i];
        if ((int)((e.x >> 10) & 2047u) != bA) continue;
        if ((int)(e.x & 1023u) > bB)
            outr[base + atomicAdd(&a_cnt, 1)] = (int)e.y;
    }
    __syncthreads();
    for (int i = tid; i < Tc; i += 512) {
        const uint2 e = arena[i];
        if ((int)((e.x >> 10) & 2047u) != bA) continue;
        if ((int)(e.x & 1023u) == bB) {
            const int s = atomicAdd(&a_cnt, 1);
            if (s < R) outr[base + s] = (int)e.y;
        }
    }
}

__global__ __launch_bounds__(512, 1)
void tp2_finish(const float* __restrict__ logits, long rstride, int n, int k,
                unsigned int* __restrict__ shist,
                unsigned int* __restrict__ d2, int* __restrict__ out,
                unsigned int* __restrict__ stats) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* lgr = logits + (size_t)row * (size_t)rstride;
    unsigned int* shist_r = shist + (size_t)row * 2048;
    unsigned int* d2_r = d2 + (size_t)row * D2_RW;
    const uint2* candbuf = reinterpret_cast<const uint2*>(d2_r + 4);
    int* outr = out + (size_t)row * (size_t)k;

    __shared__ __align__(16) unsigned int sh_hist[2048];
    __shared__ __align__(16) uint2 arena[ARENA2];
    __shared__ unsigned int warp_totals[16];
    __shared__ int s_bin, s_above, s_cnt, s_tie;

    const int Nc = (int)d2_r[0];
    __syncthreads();
    // re-zero per-row state for the next call (loads above are done).
    for (int i = tid; i < 2048; i += 512) shist_r[i] = 0u;
    if (tid == 0) { d2_r[0] = 0u; d2_r[1] = 0u; d2_r[2] = 0u; }

    bool fallback = Nc < k || Nc > CAP2;
    if (!fallback) {
        // ---- common path: exact top-k among the Nc candidates ----
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
            fallback = true;               // massive candidate tie bucket
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

    // ---- fallback: full single-CTA recompute (row re-reads; rare) ----
    if (tid == 0) atomicAdd(stats, 1u);
    const int nv4 = n >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(lgr);
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
    __syncthreads();                       // all T loads done before reuse
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
    // massive-tie ladder over bits 10 then 0 with row re-reads
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

static unsigned int* g_d2_shist = nullptr;
static unsigned int* g_d2 = nullptr;
static unsigned int* g_d2_stats = nullptr;
static int g_d2_rows = 0;

void tp2_launch(const float* logits, long W, int n, int k, int* out, int BS,
                cudaStream_t stream) {
    if (g_d2_rows < BS) {
        if (g_d2_shist) cudaFree(g_d2_shist);
        if (g_d2) cudaFree(g_d2);
        cudaMalloc(&g_d2_shist, (size_t)BS * 2048 * sizeof(unsigned int));
        cudaMalloc(&g_d2, (size_t)BS * D2_RW * sizeof(unsigned int));
        cudaMemset(g_d2_shist, 0, (size_t)BS * 2048 * sizeof(unsigned int));
        cudaMemset(g_d2, 0, (size_t)BS * D2_RW * sizeof(unsigned int));
        g_d2_rows = BS;
    }
    if (!g_d2_stats) {
        cudaMalloc(&g_d2_stats, sizeof(unsigned int));
        cudaMemset(g_d2_stats, 0, sizeof(unsigned int));
    }
    if (!g_sms)
        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
    int C = (4 * g_sms + BS - 1) / BS;
    if (C > 32) C = 32;
    if (C < 1) C = 1;
    const dim3 grid((unsigned int)C, (unsigned int)BS);
    tp2_sample<<<grid, 512, 0, stream>>>(logits, W, n, C, g_d2_shist, g_d2);
    tp2_collect<<<grid, 512, 0, stream>>>(logits, W, n, k, C, g_d2_shist,
                                          g_d2);
    tp2_finish<<<BS, 512, 0, stream>>>(logits, W, n, k, g_d2_shist, g_d2,
                                       out, g_d2_stats);
}

unsigned int tp2_read_reset_fallbacks() {
    if (!g_d2_stats) return 0u;
    unsigned int v = 0;
    cudaMemcpy(&v, g_d2_stats, sizeof(v), cudaMemcpyDeviceToHost);
    cudaMemset(g_d2_stats, 0, sizeof(v));
    return v;
}
"""
aefm += D2

TOP = r"""

// D2 sampled-estimate single-pass entries.
void topk_launch_tp2(const float* logits, long W, int n, int k, int* out,
                     int BS, cudaStream_t stream) {
    if (n <= 16896) {
        aefm::topk_launch_batched(logits, W, n, k, out, BS, stream);
        return;
    }
    aefm::tp2_launch(logits, W, n, k, out, BS, stream);
}

unsigned int topk_tp2_fallbacks() { return aefm::tp2_read_reset_fallbacks(); }
"""
rest = rest.rstrip("\n") + TOP

SRC.write_text(aefm + rest)
print("d2 edits applied:", len((aefm + rest).splitlines()), "lines")
