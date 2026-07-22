# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""D1 throughput-arm edits: append the barrier-free 3-kernel pipeline to
kernel_ext.cu (aefm namespace).

  tp_hist   : dim3(C, BS) CTAs, contiguous row slices, 11-bit MSB smem hist
              -> packed 64-bit atomic merge into per-row global hist.
  tp_collect: dim3(C, BS) CTAs, re-read slices; bucket > b0 -> out (global
              per-row counter), bucket == b0 -> out directly when the bucket
              closes the selection exactly, else (key,idx) -> tiebuf
              (skipped when T > CAP: the ladder fallback re-reads the row).
  tp_finish : one CTA per row; refine of the <= CAP boundary candidates
              (compB fast-tail structure) or the rare T > CAP in-CTA ladder
              over row re-reads; re-zeroes hist + counters for reuse.

No inter-CTA synchronization anywhere: kernel-launch boundaries provide the
ordering AND preserve the fence-less L1 safety legs (fresh launch => plain
loads of the merged hist are safe, unlike the B' persistent path).
All row reads use __ldcs (evict-first): rows are streamed 2x, the hist /
tiebuf working set should stay in L2.
"""
from pathlib import Path

SRC = Path(__file__).resolve().parent / "kernel_ext.cu"
text = SRC.read_text()

CUT = "} // namespace aefm"
ci = text.index(CUT)
aefm, rest = text[:ci], text[ci:]

TP = r"""
// ---------------------------------------------------------------------------
// D1 throughput arm (R3_LEDGER "BS>1 extension final picture" open item):
// barrier-free 3-kernel pipeline for the large-n high-BS regime.
// Per-row scratch: hist (2048 u32) in g_tp_hist; [gt_write, tie_write,
// pad, pad] + CAP uint2 tiebuf in g_tp_sc (16B-aligned slices).
// ---------------------------------------------------------------------------
#define TP_SCROW (4 + 2 * CAP)

__global__ __launch_bounds__(512, 1)
void tp_hist(const float* __restrict__ logits, long rstride, int n, int C,
             unsigned int* __restrict__ hist) {
    const int row = blockIdx.y;
    const int slice = blockIdx.x;
    const int tid = threadIdx.x;
    const float* lgr = logits + (size_t)row * (size_t)rstride;
    unsigned int* hist_r = hist + (size_t)row * 2048;

    __shared__ unsigned int sh[2048];
    for (int i = tid; i < 2048; i += 512) sh[i] = 0u;
    __syncthreads();

    const int nv4 = n >> 2;
    const int q = nv4 / C, r = nv4 % C;
    const int beg = slice * q + min(slice, r);
    const int end = beg + q + (slice < r ? 1 : 0);
    const float4* in4 = reinterpret_cast<const float4*>(lgr);
    for (int i = beg + tid; i < end; i += 512) {
        float4 f = __ldcs(in4 + i);
        atomicAdd(&sh[fkey(f.x) >> 21], 1u);
        atomicAdd(&sh[fkey(f.y) >> 21], 1u);
        atomicAdd(&sh[fkey(f.z) >> 21], 1u);
        atomicAdd(&sh[fkey(f.w) >> 21], 1u);
    }
    if (slice == C - 1) {
        for (int i = (nv4 << 2) + tid; i < n; i += 512)
            atomicAdd(&sh[fkey(__ldcs(lgr + i)) >> 21], 1u);
    }
    __syncthreads();
    const uint2* sh2 = reinterpret_cast<const uint2*>(sh);
    unsigned long long* g2 = reinterpret_cast<unsigned long long*>(hist_r);
    for (int i = tid; i < 1024; i += 512) {
        uint2 c = sh2[i];
        if (c.x | c.y)
            atomicAdd(&g2[i], ((unsigned long long)c.y << 32) | c.x);
    }
}

__global__ __launch_bounds__(512, 1)
void tp_collect(const float* __restrict__ logits, long rstride, int n, int k,
                int C, const unsigned int* __restrict__ hist,
                unsigned int* __restrict__ sc, int* __restrict__ out) {
    const int row = blockIdx.y;
    const int slice = blockIdx.x;
    const int tid = threadIdx.x;
    const float* lgr = logits + (size_t)row * (size_t)rstride;
    const unsigned int* hist_r = hist + (size_t)row * 2048;
    unsigned int* sc_r = sc + (size_t)row * TP_SCROW;
    unsigned int* gt_write = sc_r + 0;
    unsigned int* tie_write = sc_r + 1;
    uint2* tiebuf = reinterpret_cast<uint2*>(sc_r + 4);
    int* outr = out + (size_t)row * (size_t)k;

    __shared__ unsigned int warp_totals[16];
    __shared__ int s_bin, s_above;
    find_boundary_bins(hist_r, 2048, warp_totals, &s_bin, &s_above, k);
    const int b0 = s_bin;
    const int above0 = s_above;
    const int T = (int)hist_r[b0];
    const bool whole = above0 + T == k;
    const bool store_ties = !whole && T <= CAP;

    const int nv4 = n >> 2;
    const int q = nv4 / C, r = nv4 % C;
    const int beg = slice * q + min(slice, r);
    const int end = beg + q + (slice < r ? 1 : 0);
    const float4* in4 = reinterpret_cast<const float4*>(lgr);
    const int lane = tid & 31;
    for (int i = beg + tid; i < end; i += 512) {
        float4 f = __ldcs(in4 + i);
        unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z), fkey(f.w)};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = (int)(kk[j] >> 21);
            const bool gt = d > b0;
            const bool eq = d == b0;
            const unsigned int gm = __ballot_sync(0xffffffffu, gt);
            const unsigned int em = __ballot_sync(0xffffffffu, eq);
            if (gt) {
                int rk = __popc(gm & ((1u << lane) - 1));
                int leader = __ffs(gm) - 1;
                int base = 0;
                if (lane == leader)
                    base = (int)atomicAdd(gt_write, (unsigned int)__popc(gm));
                base = __shfl_sync(gm, base, leader);
                outr[base + rk] = 4 * i + j;
            }
            if (eq) {
                int rk = __popc(em & ((1u << lane) - 1));
                int leader = __ffs(em) - 1;
                int base = 0;
                if (lane == leader)
                    base = (int)atomicAdd(tie_write, (unsigned int)__popc(em));
                base = __shfl_sync(em, base, leader);
                if (whole) outr[above0 + base + rk] = 4 * i + j;
                else if (store_ties)
                    tiebuf[base + rk] = make_uint2(kk[j], (unsigned int)(4 * i + j));
            }
        }
    }
    if (slice == C - 1) {
        for (int i = (nv4 << 2) + tid; i < n; i += 512) {
            const unsigned int key = fkey(__ldcs(lgr + i));
            const int d = (int)(key >> 21);
            if (d > b0) {
                outr[atomicAdd(gt_write, 1u)] = i;
            } else if (d == b0) {
                const unsigned int s = atomicAdd(tie_write, 1u);
                if (whole) outr[above0 + s] = i;
                else if (store_ties)
                    tiebuf[s] = make_uint2(key, (unsigned int)i);
            }
        }
    }
}

__global__ __launch_bounds__(512, 1)
void tp_finish(const float* __restrict__ logits, long rstride, int n, int k,
               unsigned int* __restrict__ hist, unsigned int* __restrict__ sc,
               int* __restrict__ out) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const float* lgr = logits + (size_t)row * (size_t)rstride;
    unsigned int* hist_r = hist + (size_t)row * 2048;
    unsigned int* sc_r = sc + (size_t)row * TP_SCROW;
    uint2* tiebuf = reinterpret_cast<uint2*>(sc_r + 4);
    int* outr = out + (size_t)row * (size_t)k;

    __shared__ __align__(16) unsigned int sh_hist[2048];
    __shared__ unsigned int warp_totals[16];
    __shared__ int s_bin, s_above, s_cnt, s_ntie;
    __shared__ unsigned int s_eqk[32];
    __shared__ int s_eqv[32];

    find_boundary_bins(hist_r, 2048, warp_totals, &s_bin, &s_above, k);
    const int b0 = s_bin;
    const int above0 = s_above;
    const int T = (int)hist_r[b0];
    const int R = k - above0;

    // re-zero hist + counters for the next call (this kernel is the last
    // reader; launch boundary orders it for the next batch).
    for (int i = tid; i < 2048; i += 512) hist_r[i] = 0u;
    if (tid == 0) { sc_r[0] = 0u; sc_r[1] = 0u; }

    if (T == R) return;                      // whole bucket: K2 finished it

    if (T <= CAP) {
        // ---- refine low 21 bits of the T boundary candidates (compB
        // fast-tail structure, single 512-thread CTA) ----
        unsigned int tk[PT];
        int tv[PT];
#pragma unroll
        for (int j = 0; j < PT; ++j) {
            int p = tid + j * 512;
            if (p < T) {
                uint2 e = tiebuf[p];
                tk[j] = e.x; tv[j] = (int)e.y;
            } else { tk[j] = 0u; tv[j] = 0; }
        }
        __syncthreads();
        for (int i = tid; i < 2048; i += 512) sh_hist[i] = 0u;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < PT; ++j) {
            if (j * 512 >= T) break;
            if (tid + j * 512 < T)
                atomicAdd(&sh_hist[(tk[j] >> 10) & 2047u], 1u);
        }
        __syncthreads();
        find_boundary_bins(sh_hist, 2048, warp_totals, &s_bin, &s_above, R);
        const int bA = s_bin;
        const int aboveA = s_above;
        const int TA = (int)sh_hist[bA];
        const bool wholeA = aboveA + TA == R;
        if (tid == 0) { s_cnt = 0; s_ntie = 0; }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < PT; ++j) {
            if (j * 512 >= T) break;
            bool active = tid + j * 512 < T;
            int binA = active ? (int)((tk[j] >> 10) & 2047u) : -1;
            bool p = wholeA ? (binA >= bA) : (binA > bA);
            unsigned int m = __ballot_sync(0xffffffffu, p);
            if (p) {
                int rk = __popc(m & ((1u << lane) - 1));
                int leader = __ffs(m) - 1;
                int base = 0;
                if (lane == leader) base = atomicAdd(&s_cnt, __popc(m));
                base = __shfl_sync(m, base, leader);
                outr[above0 + base + rk] = tv[j];
            }
            if (!wholeA && binA == bA) {
                int qi = atomicAdd(&s_ntie, 1);
                if (qi < 32) { s_eqk[qi] = tk[j]; s_eqv[qi] = tv[j]; }
            }
        }
        __syncthreads();
        if (wholeA) return;
        const int R2 = R - aboveA;
        if (TA <= 32) {
            if (tid < 32) {
                unsigned int score =
                    tid < TA ? ((s_eqk[tid] & 1023u) + 1u) : 0u;
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
                        outr[above0 + aboveA + rr] = s_eqv[tid];
                        score = 0u;
                    }
                }
            }
            return;
        }
        // rare big-tie fallback: pass B over bits 9..0
        __syncthreads();
        for (int i = tid; i < 1024; i += 512) sh_hist[i] = 0u;
        __syncthreads();
#pragma unroll
        for (int j = 0; j < PT; ++j) {
            if (j * 512 >= T) break;
            if ((tid + j * 512 < T) && (int)((tk[j] >> 10) & 2047u) == bA)
                atomicAdd(&sh_hist[tk[j] & 1023u], 1u);
        }
        __syncthreads();
        find_boundary_bins(sh_hist, 1024, warp_totals, &s_bin, &s_above, R2);
        const int bB = s_bin;
#pragma unroll
        for (int j = 0; j < PT; ++j) {
            if (j * 512 >= T) break;
            bool inA = (tid + j * 512 < T) &&
                       (int)((tk[j] >> 10) & 2047u) == bA;
            bool p = inA && (int)(tk[j] & 1023u) > bB;
            unsigned int m = __ballot_sync(0xffffffffu, p);
            if (p) {
                int rk = __popc(m & ((1u << lane) - 1));
                int leader = __ffs(m) - 1;
                int base = 0;
                if (lane == leader) base = atomicAdd(&s_cnt, __popc(m));
                base = __shfl_sync(m, base, leader);
                outr[above0 + base + rk] = tv[j];
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < PT; ++j) {
            if (j * 512 >= T) break;
            bool inA = (tid + j * 512 < T) &&
                       (int)((tk[j] >> 10) & 2047u) == bA;
            bool p = inA && (int)(tk[j] & 1023u) == bB;
            unsigned int m = __ballot_sync(0xffffffffu, p);
            if (p) {
                int rk = __popc(m & ((1u << lane) - 1));
                int leader = __ffs(m) - 1;
                int base = 0;
                if (lane == leader) base = atomicAdd(&s_cnt, __popc(m));
                base = __shfl_sync(m, base, leader);
                int slot = base + rk;
                if (slot < R) outr[above0 + slot] = tv[j];
            }
        }
        return;
    }

    // ---- rare T > CAP: in-CTA 11/11/10 ladder over row re-reads ----
    const int nv4 = n >> 2;
    const float4* in4 = reinterpret_cast<const float4*>(lgr);
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
        if (whole_bucket) break;
        __syncthreads();
    }
    const int final_shift = 32 - consumed;
    const unsigned int threshold =
        final_shift ? (prefix >> final_shift) : prefix;
    if (tid == 0) { s_cnt = 0; s_ntie = 0; }
    __syncthreads();
    // final collect: winners strictly above threshold from bucket b0, plus
    // any-subset fill of full-key ties at the k-th value.
    for (int i = tid; i < nv4; i += 512) {
        float4 f = __ldcs(in4 + i);
        unsigned int kk[4] = {fkey(f.x), fkey(f.y), fkey(f.z), fkey(f.w)};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            unsigned int key = kk[j];
            if ((int)(key >> 21) != b0) continue;
            if (final_shift) key >>= final_shift;
            const bool gt = key > threshold;
            const bool eq = key == threshold;
            const unsigned int gm = __ballot_sync(0xffffffffu, gt);
            const unsigned int em = __ballot_sync(0xffffffffu, eq);
            if (gt) {
                int rk = __popc(gm & ((1u << lane) - 1));
                int leader = __ffs(gm) - 1;
                int base = 0;
                if (lane == leader) base = atomicAdd(&s_cnt, __popc(gm));
                base = __shfl_sync(gm, base, leader);
                outr[above0 + base + rk] = 4 * i + j;
            }
            if (eq) {
                int rk = __popc(em & ((1u << lane) - 1));
                int leader = __ffs(em) - 1;
                int base = 0;
                if (lane == leader) base = atomicAdd(&s_ntie, __popc(em));
                base = __shfl_sync(em, base, leader);
                int slot = base + rk;
                if (slot < remaining) outr[total_above + slot] = 4 * i + j;
            }
        }
    }
    for (int i = (nv4 << 2) + tid; i < n; i += 512) {
        unsigned int key = fkey(__ldcs(lgr + i));
        if ((int)(key >> 21) != b0) continue;
        if (final_shift) key >>= final_shift;
        if (key > threshold) outr[above0 + atomicAdd(&s_cnt, 1)] = i;
        else if (key == threshold) {
            int slot = atomicAdd(&s_ntie, 1);
            if (slot < remaining) outr[total_above + slot] = i;
        }
    }
}

static unsigned int* g_tp_hist = nullptr;
static unsigned int* g_tp_sc = nullptr;
static int g_tp_rows = 0;

void tp_launch(const float* logits, long W, int n, int k, int* out, int BS,
               cudaStream_t stream) {
    if (g_tp_rows < BS) {
        if (g_tp_hist) cudaFree(g_tp_hist);
        if (g_tp_sc) cudaFree(g_tp_sc);
        cudaMalloc(&g_tp_hist, (size_t)BS * 2048 * sizeof(unsigned int));
        cudaMalloc(&g_tp_sc, (size_t)BS * TP_SCROW * sizeof(unsigned int));
        cudaMemset(g_tp_hist, 0, (size_t)BS * 2048 * sizeof(unsigned int));
        cudaMemset(g_tp_sc, 0, (size_t)BS * TP_SCROW * sizeof(unsigned int));
        g_tp_rows = BS;
    }
    if (!g_sms)
        cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
    int C = (2 * g_sms + BS - 1) / BS;      // target ~2 CTAs per SM
    if (C > 8) C = 8;
    if (C < 1) C = 1;
    const dim3 grid((unsigned int)C, (unsigned int)BS);
    tp_hist<<<grid, 512, 0, stream>>>(logits, W, n, C, g_tp_hist);
    tp_collect<<<grid, 512, 0, stream>>>(logits, W, n, k, C, g_tp_hist,
                                         g_tp_sc, out);
    tp_finish<<<BS, 512, 0, stream>>>(logits, W, n, k, g_tp_hist, g_tp_sc,
                                      out);
}
"""
aefm += TP

TOP = r"""

// D1 throughput-arm entry (large-n only; small-n keeps grid.y batching).
void topk_launch_tp(const float* logits, long W, int n, int k, int* out,
                    int BS, cudaStream_t stream) {
    if (n <= 16896) {
        aefm::topk_launch_batched(logits, W, n, k, out, BS, stream);
        return;
    }
    aefm::tp_launch(logits, W, n, k, out, BS, stream);
}
"""
rest = rest.rstrip("\n") + TOP

SRC.write_text(aefm + rest)
print("tp edits applied:", len((aefm + rest).splitlines()), "lines")
