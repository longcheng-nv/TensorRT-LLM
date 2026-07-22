# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""B' persistent-queue edits: generate topk_fast_pq<MINB> from the (already
team-ized, diet-templated) topk_fast body in kernel_ext.cu.

Design (R3_LEDGER B' + C-remedies):
- ONE launch consumes the whole batch: nteams = min(cap/team, BS) teams,
  each looping rows round-robin (row = team_id + iter*nteams; uniform rows
  => static assignment == atomic queue without the grab-broadcast barrier).
- Per-TEAM scratch slice REUSED across row iterations -> the fence-less L1
  safety leg "L1 invalidated at launch boundary" dies. Remedies applied to
  the pq body only:
    * post-barrier global-hist loads via __ldcg (L2-only): boundary search
      over hist0_g / merged, the T = hist0_g[b0] read, tiebuf refine reads;
    * sense tokens widened to gen*8192 + iter*8 + pass (monotonic per slice
      across rows AND launches, iter <= 1023);
    * trailing per-team barrier (sense0+7) after each row stands in for the
      stream ordering that made cross-launch slice reuse safe (last
      arriver's re-zero must complete before the team starts the next row).
- The per-row body is wrapped in an immediately-invoked [&] lambda so the
  original early-return control flow (arrive-and-exit fast path) becomes
  per-row control flow unchanged.
"""
import re
from pathlib import Path

SRC = Path(__file__).resolve().parent / "kernel_ext.cu"
text = SRC.read_text()

CUT = "} // namespace aefm"
ci = text.index(CUT)
aefm, rest = text[:ci], text[ci:]

# ---------------------------------------------------------------- extract
SIG = """template<int MINB>
__global__ __launch_bounds__(BLOCK, MINB)
void topk_fast(const float* __restrict__ logits, long rstride, int n, int k,
               unsigned int* __restrict__ scratch, int* __restrict__ out,
               unsigned int gen, int team) {"""
assert aefm.count(SIG) == 1
body_start = aefm.index(SIG) + len(SIG)

END = """    if (last) {
        for (int i = tid; i < CNT; i += BLOCK) scratch[i] = 0u;
        if (tid == 0) atomicExch(tail_arrive, 0u);
    }
}"""
assert aefm.count(END) == 1
body_end = aefm.index(END) + len(END) - 1          # exclude closing brace
full_body = aefm[body_start:body_end]

M_SETUP = "    unsigned int* hist0_g = scratch + HIST0;"
M_PERROW = "    // ---- register-cached keys"
assert full_body.count(M_SETUP) == 1 and full_body.count(M_PERROW) == 1
setup = full_body[full_body.index(M_SETUP):full_body.index(M_PERROW)]
perrow = full_body[full_body.index(M_PERROW):]

# ------------------------------------------------------- transform perrow
def rep(s, old, new, cnt=None):
    c = s.count(old)
    assert c > 0 and (cnt is None or c == cnt), f"{c} matches for: {old[:80]}"
    return s.replace(old, new)

perrow = rep(perrow, "find_boundary_bins(hist0_g, 2048, warp_totals, &s_bin, &s_above, k);",
             "find_boundary_bins_cg(hist0_g, 2048, warp_totals, &s_bin, &s_above, k);", 1)
perrow = rep(perrow, "const int T = (int)hist0_g[b0];",
             "const int T = (int)__ldcg(&hist0_g[b0]);", 1)
perrow = rep(perrow, "find_boundary_bins(merged, nb, warp_totals, &s_bin, &s_above, remaining);",
             "find_boundary_bins_cg(merged, nb, warp_totals, &s_bin,\n"
             "                              &s_above, remaining);", 1)
perrow = rep(perrow, "bool whole_bucket = s_above + (int)merged[s_bin] == remaining;",
             "bool whole_bucket = s_above + (int)__ldcg(&merged[s_bin]) == remaining;", 1)
perrow = rep(perrow, "uint2 e = tiebuf[p];", "uint2 e = __ldcg(&tiebuf[p]);", 1)
perrow = rep(perrow, "global_barrier(g_arrive, g_release, gridsz, gen * 8u);",
             "global_barrier(g_arrive, g_release, gridsz, sense0);", 1)
perrow = rep(perrow, "global_barrier(g_arrive, g_release, gridsz, gen * 8u + (unsigned int)pass);",
             "global_barrier(g_arrive, g_release, gridsz,\n"
             "                       sense0 + (unsigned int)pass);", 1)
perrow = rep(perrow, "reinterpret_cast<const float4*>(logits)",
             "reinterpret_cast<const float4*>(lgr)", 1)
perrow = rep(perrow, "logits[ptail]", "lgr[ptail]", 1)
n_out = perrow.count("out[")
perrow = perrow.replace("out[", "outr[")
print(f"out[ -> outr[ : {n_out} sites")
perrow = perrow.replace("\n", "\n        ")        # indent into loop+lambda

FBB_CG = r"""
// __ldcg (L2-only) twin of find_boundary_bins for the persistent-queue path:
// per-team scratch slices are reused across row iterations WITHIN one launch,
// so the "L1 invalidated at launch boundary" leg of the fence-less safety
// argument does not apply -- post-barrier global-histogram loads must bypass
// L1.
__device__ __forceinline__ void find_boundary_bins_cg(
        const unsigned int* __restrict__ hist, int nb,
        unsigned int* warp_totals, int* s_bin, int* s_above,
        int remaining) {
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int bins_per_thread = nb >> 9;   // nb / BLOCK, BLOCK == 512
    const int base = tid * bins_per_thread;
    unsigned int b[4];
    unsigned int local = 0;
    if (bins_per_thread == 4) {
        uint4 v = __ldcg(reinterpret_cast<const uint4*>(hist + base));
        b[0] = v.x; b[1] = v.y; b[2] = v.z; b[3] = v.w;
        local = v.x + v.y + v.z + v.w;
    } else {
        uint2 v = __ldcg(reinterpret_cast<const uint2*>(hist + base));
        b[0] = v.x; b[1] = v.y; b[2] = 0u; b[3] = 0u;
        local = v.x + v.y;
    }
    unsigned int suffix = local;
#pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        unsigned int v = __shfl_down_sync(0xffffffffu, suffix, off);
        if (lane + off < 32) suffix += v;
    }
    if (lane == 0) warp_totals[warp] = suffix;
    __syncthreads();
    unsigned int higher_warps = 0;
#pragma unroll
    for (int w = 0; w < (BLOCK >> 5); ++w)
        if (w > warp) higher_warps += warp_totals[w];
    unsigned int higher = suffix - local + higher_warps;
    if ((int)higher < remaining && (int)(higher + local) >= remaining) {
        unsigned int cumulative = higher;
        int boundary = base;
        int above = (int)higher;
#pragma unroll
        for (int j = 3; j >= 0; --j) {
            if (j >= bins_per_thread) continue;
            unsigned int next = cumulative + b[j];
            if ((int)next >= remaining) {
                boundary = base + j;
                above = (int)cumulative;
                break;
            }
            cumulative = next;
        }
        *s_bin = boundary;
        *s_above = above;
    }
    __syncthreads();
}
"""

PQ = (FBB_CG + r"""
// B' persistent-queue kernel: one launch, nteams teams loop rows round-robin.
template<int MINB>
__global__ __launch_bounds__(BLOCK, MINB)
void topk_fast_pq(const float* __restrict__ logits, long rstride, int n, int k,
                  unsigned int* __restrict__ scratch, int* __restrict__ out,
                  unsigned int gen, int team, int BS) {
    const int nteams = gridDim.x / team;
    const int team_id = blockIdx.x / team;
    const int bx = blockIdx.x - team_id * team;
    scratch += (size_t)team_id * (size_t)SCRATCH_WORDS;
    const int gridsz = team;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int gtid = bx * BLOCK + tid;
    const int gstride = gridsz * BLOCK;
""" + setup + r"""
    for (int row = team_id, iter = 0; row < BS; row += nteams, ++iter) {
        const float* lgr = logits + (size_t)row * (size_t)rstride;
        int* outr = out + (size_t)row * (size_t)k;
        const unsigned int sense0 = gen * 8192u + (unsigned int)iter * 8u;
        [&]() {
        """ + perrow + r"""
        }();
        // stand-in for the launch boundary: the last arriver's scratch
        // re-zero must be complete (and ordered) before this team's next row
        // starts merging into the same slice.
        global_barrier(g_arrive, g_release, gridsz, sense0 + 7u);
    }
}

static unsigned int* g_scratch_pq = nullptr;
static int g_pq_teams = 0;
static unsigned int g_gen_pq = 0;
static int g_cap_pq[5] = {0, 0, 0, 0, 0};

template<int MINB>
static void pq_caps(int* team_out, int* cap_out, int n) {
    if (!g_cap_pq[MINB]) {
        cudaFuncSetAttribute(topk_fast_pq<MINB>,
                             cudaFuncAttributePreferredSharedMemoryCarveout,
                             cudaSharedmemCarveoutMaxShared);
        int active = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active, topk_fast_pq<MINB>, BLOCK, 0);
        if (active < 1) active = 1;
        if (!g_sms)
            cudaDeviceGetAttribute(&g_sms, cudaDevAttrMultiProcessorCount, 0);
        g_cap_pq[MINB] = active * g_sms;
    }
    *team_out = (n + 2047) / 2048;
    *cap_out = g_cap_pq[MINB];
}

template<int MINB>
static void launch_fast_pq(const float* logits, long W, int n, int k,
                           int* out, int BS, cudaStream_t stream) {
    int team, cap;
    pq_caps<MINB>(&team, &cap, n);
    if (team > cap || BS > 8192) {   // iter bound 1023 needs BS/nteams small
        for (int r = 0; r < BS; ++r)
            topk_launch(logits + (size_t)r * (size_t)W, n, k,
                        out + (size_t)r * (size_t)k, stream);
        return;
    }
    int nteams = cap / team;
    if (nteams > BS) nteams = BS;
    if (g_pq_teams < nteams) {
        if (g_scratch_pq) cudaFree(g_scratch_pq);
        cudaMalloc(&g_scratch_pq,
                   (size_t)nteams * SCRATCH_WORDS * sizeof(unsigned int));
        cudaMemset(g_scratch_pq, 0,
                   (size_t)nteams * SCRATCH_WORDS * sizeof(unsigned int));
        g_pq_teams = nteams;
    }
    ++g_gen_pq;
    topk_fast_pq<MINB><<<nteams * team, BLOCK, 0, stream>>>(
        logits, W, n, k, g_scratch_pq, out, g_gen_pq, team, BS);
}

void launch_fast_pq_v(const float* logits, long W, int n, int k, int* out,
                      int BS, int minb, cudaStream_t stream) {
    switch (minb) {
        case 2: launch_fast_pq<2>(logits, W, n, k, out, BS, stream); break;
        case 3: launch_fast_pq<3>(logits, W, n, k, out, BS, stream); break;
        case 4: launch_fast_pq<4>(logits, W, n, k, out, BS, stream); break;
        default: launch_fast_pq<1>(logits, W, n, k, out, BS, stream); break;
    }
}

template<int MINB>
static void pq_stats_t(int out5[5]) {
    cudaFuncAttributes a;
    cudaFuncGetAttributes(&a, topk_fast_pq<MINB>);
    out5[0] = a.numRegs;
    out5[1] = (int)a.sharedSizeBytes;
    out5[2] = (int)a.localSizeBytes;
    int act = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act, topk_fast_pq<MINB>,
                                                  BLOCK, 0);
    out5[3] = act;
    cudaFuncSetAttribute(topk_fast_pq<MINB>,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    act = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act, topk_fast_pq<MINB>,
                                                  BLOCK, 0);
    out5[4] = act;
}

void pq_stats_v(int minb, int out5[5]) {
    switch (minb) {
        case 2: pq_stats_t<2>(out5); break;
        case 3: pq_stats_t<3>(out5); break;
        case 4: pq_stats_t<4>(out5); break;
        default: pq_stats_t<1>(out5); break;
    }
}
""")
aefm += PQ

TOP = r"""

// B' persistent-queue entries.
void topk_launch_pq_v(const float* logits, long W, int n, int k, int* out,
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
    aefm::launch_fast_pq_v(logits, W, n, k, out, BS, minb, stream);
}

void topk_pq_stats(int minb, int out5[5]) { aefm::pq_stats_v(minb, out5); }
"""
rest = rest.rstrip("\n") + TOP

SRC.write_text(aefm + rest)
print("pq edits applied:", len((aefm + rest).splitlines()), "lines")
