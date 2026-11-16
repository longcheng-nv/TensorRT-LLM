"""Fused decode kernels ("megakernels") for BS=1 decoding on Apple GPUs.

Every kernel here replaces a chain of small MLX ops on the decode path (one token) and is checked
against that chain by tests/test_mk.py. Numerics follow the reference chain: fp8 (e4m3, power-of-
two block scales) activation fake-quant before every fp8/expert matmul, bf16 roundings where the
MLX chain rounds, fp32 accumulation.

MoE (routed experts, slab slots):
  moe_up    : for each of the k selected experts and each of its `inter` rows: g = w1.x, u = w3.x
              (x = act_quant(h) computed in-register), h_e = bf16(w_e * swiglu(bf16 g, bf16 u))
  moe_down  : y[r] = shared[r] + sum_e bf16( w2_e[r] . act_quant(h_e) )
"""

from __future__ import annotations

import os
from functools import lru_cache

import mlx.core as mx

# sparse attention as two parallel dispatches (sa_scores + sa_pv) instead of the single-dispatch
# mk_sparse_attn (16 threadgroups, latency-bound at long context). Same arithmetic; only the PV sum is
# split into 8 ordered partials.
SPLIT_ATTN = os.environ.get("DEEPINFER_MK_ATTN_SPLIT", "1") != "0"

def supported(a) -> bool:
    """The fused decode kernels are written for DeepSeek-V4.1-Flash's geometry; other shapes (the
    tiny test models) use the reference MLX path."""
    try:
        return (a.dim % 512 == 0 and a.head_dim == 512 and a.window_size == 128 and a.hc_mult == 4
                and a.rope_head_dim == 64 and a.q_lora_rank % 256 == 0 and a.n_routed_experts % 32 == 0
                and a.n_activated_experts <= 32 and a.moe_inter_dim % 256 == 0
                and (a.n_heads * a.head_dim // a.o_groups) % 256 == 0 and (a.o_groups * a.o_lora_rank) % 256 == 0)
    except AttributeError:
        return False


HEADER = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// PyTorch / MLX float -> fp8 e4m3fn (round to nearest even, saturating) and back.
inline float fp8_e4m3_round(float f) {
  uint32_t fp8_max = 543u << 21;
  uint32_t denorm_mask = 141u << 23;
  uint32_t f_bits = as_type<uint32_t>(f);
  uint32_t sign = f_bits & 0x80000000u;
  f_bits ^= sign;
  uint8_t bits;
  if (f_bits >= fp8_max) {
    bits = 0x7E;
  } else if (f_bits < (121u << 23)) {
    f_bits = as_type<uint32_t>(as_type<float>(f_bits) + as_type<float>(denorm_mask));
    bits = static_cast<uint8_t>(f_bits - denorm_mask);
  } else {
    uint8_t mant_odd = (f_bits >> 20) & 1;
    f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFF;
    f_bits += mant_odd;
    bits = static_cast<uint8_t>(f_bits >> 20);
  }
  bits |= static_cast<uint8_t>(sign >> 24);
  uint16_t v = bits & 127;
  uint16_t sign_bit = ((uint16_t)((bits >> 7) & 1)) << 15;
  uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
  half converted = as_type<half>(u);
  return static_cast<float>(converted) * 256.0f;
}

// 2**ceil(log2(x)) for positive normal float (ops._pow2_ceil_log2)
inline float pow2_ceil(float x) {
  uint32_t b = as_type<uint32_t>(x);
  uint32_t e = (b >> 23) & 0xFF;
  e += (b & 0x7FFFFF) != 0 ? 1u : 0u;
  return as_type<float>(e << 23);
}

// ops._act_quant scale for a block with absolute max `amax`
inline float act_scale(float amax) {
  return pow2_ceil(max(amax, 1e-4f) / 448.0f);
}

inline float bf_lo(uint w) { return as_type<float>(w << 16); }
inline float bf_hi(uint w) { return as_type<float>(w & 0xFFFF0000u); }

inline float fq8(float x, float s) {  // fake-quant one value with block scale s
  return fp8_e4m3_round(clamp(x / s, -448.0f, 448.0f)) * s;
}

inline float sigmoid_mlx(float x) {
  float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
  return (x < 0.0f) ? y : 1.0f - y;
}

// sum_j q_j * x[j] for 16 affine codes of `bits` bits starting at byte p (p aligned to the 16-code
// group's byte size: 3-bit -> 6 bytes (2-aligned), 2-bit -> 4 bytes, 4-bit -> 8 bytes), wide loads
// sum_j q_j * x[j] for 16 affine codes of `bits` bits starting at byte p, with xs[j] = x[j] pre-scaled
// by 2^-(bit offset of code j within its word) (exact: powers of two), so a code needs only a mask,
// a conversion and an FMA (no shift) -- the same trick as MLX's qdot. Wide loads: 3-bit -> 3 ushort
// (6 bytes, 2-aligned), 2-bit -> one uint, 4-bit -> uint2.
template <int bits>
inline void prescale16(thread const float* x, thread float* xs) {
  if (bits == 2) { for (int j = 0; j < 16; j++) xs[j] = x[j] * as_type<float>(uint(127 - 2 * j) << 23); }
  else if (bits == 4) { for (int j = 0; j < 16; j++) xs[j] = x[j] * as_type<float>(uint(127 - 4 * (j & 7)) << 23); }
  else {  // 3-bit: codes 0..9 in lo (shift 3j), code 10 straddles, 11..15 in hi (shift 3j-32)
    for (int j = 0; j < 10; j++) xs[j] = x[j] * as_type<float>(uint(127 - 3 * j) << 23);
    xs[10] = x[10];
    for (int j = 11; j < 16; j++) xs[j] = x[j] * as_type<float>(uint(127 - (3 * j - 32)) << 23);
  }
}
template <int bits>
inline float qdot16(const device uint8_t* p, thread const float* xs) {
  float acc = 0.0f;
  if (bits == 2) {
    const uint w = *((const device uint*)p);
    for (int j = 0; j < 16; j++) acc += float(w & (3u << (2 * j))) * xs[j];
  } else if (bits == 3) {
    const device ushort* q = (const device ushort*)p;
    const uint lo = uint(q[0]) | (uint(q[1]) << 16);
    const uint hi = uint(q[2]);
    for (int j = 0; j < 10; j++) acc += float(lo & (7u << (3 * j))) * xs[j];
    acc += float(((lo >> 30) | ((hi & 1u) << 2)) & 7u) * xs[10];
    for (int j = 11; j < 16; j++) acc += float(hi & (7u << (3 * j - 32))) * xs[j];
  } else if (bits == 4) {
    const uint2 w = *((const device uint2*)p);
    for (int j = 0; j < 8; j++) acc += float(w.x & (15u << (4 * j))) * xs[j];
    for (int j = 0; j < 8; j++) acc += float(w.y & (15u << (4 * j))) * xs[8 + j];
  }
  return acc;
}

// sum_j q_j * x[j] for 8 affine codes of `bits` bits packed little-endian starting at byte p
template <int bits>
inline float qdot8(const device uint8_t* p, thread const float* x) {
  float acc = 0.0f;
  if (bits == 2) {
    uint v = uint(p[0]) | (uint(p[1]) << 8);
    for (int j = 0; j < 8; j++) acc += float((v >> (2 * j)) & 3u) * x[j];
  } else if (bits == 3) {
    uint v = uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16);
    for (int j = 0; j < 8; j++) acc += float((v >> (3 * j)) & 7u) * x[j];
  } else if (bits == 4) {
    uint v = uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24);
    for (int j = 0; j < 8; j++) acc += float((v >> (4 * j)) & 15u) * x[j];
  } else {  // 8
    for (int j = 0; j < 8; j++) acc += float(p[j]) * x[j];
  }
  return acc;
}
"""

# ---------------------------------------------------------------------------------------- MoE up
# One threadgroup = 8 simdgroups x 4 rows = 32 consecutive rows of one expert (one act_quant block of
# h), so h is fake-quantized here, once. x is the already fake-quantized MoE input (bf16, from the
# router stage). Lane l of a simdgroup covers 16 values per step (K / 512 steps).
_MOE_UP = r"""
  constexpr int RPS = 4;
  constexpr int SG = 8;
  const int K = DIM;
  const int N = INTER;
  const int lane = thread_index_in_simdgroup;
  const int sg = simdgroup_index_in_threadgroup;
  const int blk = threadgroup_position_in_grid.y;           // over n_experts * N / 32
  const int e = (blk * 32) / N;
  const int rb = (blk * 32) % N;                             // first row of this 32-row block
  const int r0 = rb + sg * RPS;
  const uint slot = slots[e];
  constexpr int RB = (DIM * BITS) / 8;
  constexpr int G = DIM / 64;
  const device uint8_t* w1p = (const device uint8_t*)w1q + ((size_t)slot * N + r0) * RB;
  const device uint8_t* w3p = (const device uint8_t*)w3q + ((size_t)slot * N + r0) * RB;
  const device bfloat16_t* s1p = w1s + ((size_t)slot * N + r0) * G;
  const device bfloat16_t* b1p = w1b + ((size_t)slot * N + r0) * G;
  const device bfloat16_t* s3p = w3s + ((size_t)slot * N + r0) * G;
  const device bfloat16_t* b3p = w3b + ((size_t)slot * N + r0) * G;
  float acc1[RPS] = {0}, acc3[RPS] = {0};
  for (int it = 0; it < K / 512; it++) {
    const int c = it * 512 + lane * 16;
    float xv[16];
    const uint4 xa = *((const device uint4*)(x + c));
    const uint4 xb = *((const device uint4*)(x + c + 8));
    xv[0] = bf_lo(xa.x); xv[1] = bf_hi(xa.x); xv[2] = bf_lo(xa.y); xv[3] = bf_hi(xa.y);
    xv[4] = bf_lo(xa.z); xv[5] = bf_hi(xa.z); xv[6] = bf_lo(xa.w); xv[7] = bf_hi(xa.w);
    xv[8] = bf_lo(xb.x); xv[9] = bf_hi(xb.x); xv[10] = bf_lo(xb.y); xv[11] = bf_hi(xb.y);
    xv[12] = bf_lo(xb.z); xv[13] = bf_hi(xb.z); xv[14] = bf_lo(xb.w); xv[15] = bf_hi(xb.w);
    float xs = 0.0f;
    for (int j = 0; j < 16; j++) xs += xv[j];
    float xp[16];
    prescale16<BITS>(xv, xp);
    const int g = c / 64;
    const int boff = c * BITS / 8;
    for (int r = 0; r < RPS; r++) {
      const float q1 = qdot16<BITS>(w1p + r * RB + boff, xp);
      const float q3 = qdot16<BITS>(w3p + r * RB + boff, xp);
      acc1[r] += float(s1p[r * G + g]) * q1 + float(b1p[r * G + g]) * xs;
      acc3[r] += float(s3p[r * G + g]) * q3 + float(b3p[r * G + g]) * xs;
    }
  }
  threadgroup float hb[32];
  for (int r = 0; r < RPS; r++) {
    float gt = simd_sum(acc1[r]);
    float up = simd_sum(acc3[r]);
    if (lane == 0) {
      gt = float(bfloat16_t(gt));
      up = float(bfloat16_t(up));
      const float lim = limit[0];
      if (lim > 0.0f) { up = clamp(up, -lim, lim); gt = min(gt, lim); }
      hb[sg * RPS + r] = float(bfloat16_t(wts[e] * (gt * sigmoid_mlx(gt) * up)));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {  // act_quant over the 32-row block
    const float v = hb[lane];
    float amax = simd_max(fabs(v));
    const float s = act_scale(amax);
    h[e * N + rb + lane] = bfloat16_t(fq8(v, s));
  }
"""

# -------------------------------------------------------------------------------------- MoE down
# h is already fake-quantized (moe_up). Each simdgroup: RPS output rows over all experts.
_MOE_DOWN = r"""
  constexpr int RPS = 4;
  constexpr int SG = 2;
  const int K = INTER;
  const int N = DIM;
  const int lane = thread_index_in_simdgroup;
  const int sg = simdgroup_index_in_threadgroup;
  const int r0 = (threadgroup_position_in_grid.y * SG + sg) * RPS;
  constexpr int RB = (INTER * BITS) / 8;
  constexpr int G = INTER / 64;
  constexpr int ITF = INTER / 512;                 // full 512-value steps
  constexpr bool TAIL = (INTER % 512) != 0;         // + a half step (lanes 0..15)
  float out[RPS] = {0};
  for (int e = 0; e < NE; e++) {
    const uint slot = slots[e];
    const device uint8_t* wp = (const device uint8_t*)w2q + ((size_t)slot * N + r0) * RB;
    const device bfloat16_t* sp = w2s + ((size_t)slot * N + r0) * G;
    const device bfloat16_t* bp = w2b + ((size_t)slot * N + r0) * G;
    float acc[RPS] = {0};
    for (int it = 0; it < ITF + (TAIL ? 1 : 0); it++) {
      const int c = it * 512 + lane * 16;
      if (c >= K) break;
      float xv[16];
      const uint4 xa = *((const device uint4*)(h + e * K + c));
      const uint4 xb = *((const device uint4*)(h + e * K + c + 8));
      xv[0] = bf_lo(xa.x); xv[1] = bf_hi(xa.x); xv[2] = bf_lo(xa.y); xv[3] = bf_hi(xa.y);
      xv[4] = bf_lo(xa.z); xv[5] = bf_hi(xa.z); xv[6] = bf_lo(xa.w); xv[7] = bf_hi(xa.w);
      xv[8] = bf_lo(xb.x); xv[9] = bf_hi(xb.x); xv[10] = bf_lo(xb.y); xv[11] = bf_hi(xb.y);
      xv[12] = bf_lo(xb.z); xv[13] = bf_hi(xb.z); xv[14] = bf_lo(xb.w); xv[15] = bf_hi(xb.w);
      float xs = 0.0f;
      for (int j = 0; j < 16; j++) xs += xv[j];
      float xp[16];
      prescale16<BITS>(xv, xp);
      const int g = c / 64;
      const int boff = c * BITS / 8;
      for (int r = 0; r < RPS; r++) {
        acc[r] += float(sp[r * G + g]) * qdot16<BITS>(wp + r * RB + boff, xp) + float(bp[r * G + g]) * xs;
      }
    }
    for (int r = 0; r < RPS; r++) {
      float v = simd_sum(acc[r]);
      out[r] += float(bfloat16_t(v));
    }
  }
  if (lane == 0) {
    for (int r = 0; r < RPS; r++) y[r0 + r] = static_cast<OT>(shared[r0 + r] + out[r]);
  }
"""


@lru_cache(maxsize=None)
def _moe_up_kernel():
    return mx.fast.metal_kernel(name="mk_moe_up2", input_names=["x", "slots", "wts", "limit", "w1q", "w1s", "w1b", "w3q", "w3s", "w3b"],
                                output_names=["h"], header=HEADER, source=_MOE_UP)


@lru_cache(maxsize=None)
def _moe_down_kernel():
    return mx.fast.metal_kernel(name="mk_moe_down2", input_names=["h", "slots", "shared", "w2q", "w2s", "w2b"],
                                output_names=["y"], header=HEADER, source=_MOE_DOWN)


def moe_experts(xq: mx.array, slots: mx.array, wts: mx.array, limit: mx.array, p1, p3, p2, shared: mx.array,
                dim: int = 5120, inter: int = 2304, out_dtype=mx.float32) -> mx.array:
    """Routed experts of one decode token over slab slots, plus the shared expert's output.
    xq: [dim] bf16 MoE input AFTER act_quant; slots: [k] uint32; wts: [k] fp32 routing weights;
    p1/p3/p2: (packed, scales_bf16, biases_bf16, bits) slab projections; shared: [dim] fp32.
    Returns y [dim] fp32 = shared + sum_e expert_e(x)."""
    ne = slots.shape[0]
    b13, b2 = p1[3], p2[3]
    assert p3[3] == b13
    h = _moe_up_kernel()(
        inputs=[xq.reshape(-1), slots, wts.reshape(-1).astype(mx.float32), limit.reshape(1), p1[0], p1[1], p1[2], p3[0], p3[1], p3[2]],
        template=[("BITS", b13), ("DIM", dim), ("INTER", inter)],
        grid=(256, ne * inter // 32, 1), threadgroup=(256, 1, 1),
        output_shapes=[(ne * inter,)], output_dtypes=[mx.bfloat16],
    )[0]
    y = _moe_down_kernel()(
        inputs=[h, slots, shared.reshape(-1).astype(mx.float32), p2[0], p2[1], p2[2]],
        template=[("BITS", b2), ("DIM", dim), ("INTER", inter), ("NE", ne), ("OT", out_dtype)],
        grid=(64, dim // 8, 1), threadgroup=(64, 1, 1),
        output_shapes=[(dim,)], output_dtypes=[out_dtype],
    )[0]
    return y


# ------------------------------------------------------------------------------------------ mHC
# Decode (one token) residual-stream machinery of a Block as two dispatches per stage:
#   hc_partial  : [optional hc_post of the previous sublayer] -> X (bf16 [4, D]);
#                 per-threadgroup partials of mixes = X.flat @ fn.T (24), sum(X^2), and of
#                 h = bf16(sum_j premix[j] X[j]) its sum(h^2)
#   hc_finalize : reduce partials, scale by rsqrt(mean(X^2)+eps), Sinkhorn -> pre/post/comb;
#                 h_norm = bf16(w * h * rsqrt(mean(h^2)+eps))
_HC_PARTIAL = r"""
  constexpr int D = DIM;
  constexpr int HC = 4;
  constexpr int NM = 24;
  const int tid = thread_position_in_threadgroup.x;
  const int tg = threadgroup_position_in_grid.x;
  const int d = tg * TPG + tid;
  float xf[HC];
  if (HAS_POST) {
    const float hv = float(hin[d]);
    float res[HC];
    for (int j = 0; j < HC; j++) res[j] = float(xres[j * D + d]);
    for (int k = 0; k < HC; k++) {
      float acc = comb[0 * HC + k] * res[0];
      for (int j = 1; j < HC; j++) acc = acc + comb[j * HC + k] * res[j];
      const bfloat16_t o = bfloat16_t(post[k] * hv + acc);
      xnew[k * D + d] = o;
      xf[k] = float(o);
    }
  } else {
    for (int k = 0; k < HC; k++) { const bfloat16_t o = xres[k * D + d]; xnew[k * D + d] = o; xf[k] = float(o); }
  }
  float part[NM + 2];
  float ss = 0.0f;
  for (int k = 0; k < HC; k++) ss += xf[k] * xf[k];
  for (int i = 0; i < NM; i++) {
    float a = 0.0f;
    for (int k = 0; k < HC; k++) a += xf[k] * fn[(size_t)i * (HC * D) + k * D + d];
    part[i] = a;
  }
  float hs = premix[0] * xf[0];
  for (int j = 1; j < HC; j++) hs = hs + premix[j] * xf[j];
  const bfloat16_t hb = bfloat16_t(hs);
  hraw[d] = hb;
  const float hf = float(hb);
  part[NM] = ss;
  part[NM + 1] = hf * hf;
  threadgroup float red[TPG / 32][NM + 2];
  const int sg = tid / 32, lane = tid % 32;
  for (int i = 0; i < NM + 2; i++) {
    const float v = simd_sum(part[i]);
    if (lane == 0) red[sg][i] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < NM + 2) {
    float v = 0.0f;
    for (int s = 0; s < TPG / 32; s++) v += red[s][tid];
    partials[tg * (NM + 2) + tid] = v;
  }
"""

_HC_FINALIZE = r"""
  constexpr int D = DIM;
  constexpr int HC = 4;
  constexpr int NM = 24;
  const int tid = thread_position_in_threadgroup.x;
  threadgroup float tot[NM + 2];
  threadgroup float rstd_h;
  if (tid < NM + 2) {
    float v = 0.0f;
    for (int t = 0; t < NT; t++) v += partials[t * (NM + 2) + tid];
    tot[tid] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 32) {  // one simdgroup: lane = j * 4 + k holds comb[j][k] (lanes 0..15)
    const float eps = epsv[0];
    const float hc_eps = 1e-6f;
    const float rs = metal::precise::rsqrt(tot[NM] / float(HC * D) + eps);
    const int lane = tid;
    const int li = lane & 15;
    float c = tot[2 * HC + li] * rs * sc[2] + base[2 * HC + li];
    float mx_ = max(c, simd_shuffle_xor(c, 1));
    mx_ = max(mx_, simd_shuffle_xor(mx_, 2));
    c = metal::precise::exp(c - mx_);
    float s = c + simd_shuffle_xor(c, 1);
    s = s + simd_shuffle_xor(s, 2);
    c = c / s + hc_eps;
    s = c + simd_shuffle_xor(c, 4);
    s = s + simd_shuffle_xor(s, 8);
    c = c / (s + hc_eps);
    for (int it = 0; it < ITERS - 1; it++) {
      s = c + simd_shuffle_xor(c, 1);
      s = s + simd_shuffle_xor(s, 2);
      c = c / (s + hc_eps);
      s = c + simd_shuffle_xor(c, 4);
      s = s + simd_shuffle_xor(s, 8);
      c = c / (s + hc_eps);
    }
    if (lane < 16) comb_o[lane] = c;
    if (lane < HC) {
      pre_o[lane] = sigmoid_mlx(tot[lane] * rs * sc[0] + base[lane]) + hc_eps;
      post_o[lane] = 2.0f * sigmoid_mlx(tot[HC + lane] * rs * sc[1] + base[HC + lane]);
    }
    if (lane == 0) rstd_h = metal::precise::rsqrt(tot[NM + 1] / float(D) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float r = rstd_h;
  for (int d = tid; d < D; d += TPGF) {
    hnorm[d] = bfloat16_t(float(normw[d]) * (float(hraw[d]) * r));
  }
"""

_HC_FUSED = r"""
  constexpr int D = DIM;
  constexpr int HC = 4;
  constexpr int NM = 24;
  const int tid = thread_position_in_threadgroup.x;
  const int tg = threadgroup_position_in_grid.x;
  const int d = tg * TPG + tid;
  float xf[HC];
  if (HAS_POST) {
    const float hv = float(hin[d]);
    float res[HC];
    for (int j = 0; j < HC; j++) res[j] = float(xres[j * D + d]);
    for (int k = 0; k < HC; k++) {
      float acc = comb[0 * HC + k] * res[0];
      for (int j = 1; j < HC; j++) acc = acc + comb[j * HC + k] * res[j];
      const bfloat16_t o = bfloat16_t(post[k] * hv + acc);
      xnew[k * D + d] = o;
      xf[k] = float(o);
    }
  } else {
    for (int k = 0; k < HC; k++) { const bfloat16_t o = xres[k * D + d]; xnew[k * D + d] = o; xf[k] = float(o); }
  }
  float part[NM + 2];
  float ss = 0.0f;
  for (int k = 0; k < HC; k++) ss += xf[k] * xf[k];
  for (int i = 0; i < NM; i++) {
    float a = 0.0f;
    for (int k = 0; k < HC; k++) a += xf[k] * fn[(size_t)i * (HC * D) + k * D + d];
    part[i] = a;
  }
  float hs = premix[0] * xf[0];
  for (int j = 1; j < HC; j++) hs = hs + premix[j] * xf[j];
  const bfloat16_t hb = bfloat16_t(hs);
  hraw[d] = hb;
  const float hf = float(hb);
  part[NM] = ss;
  part[NM + 1] = hf * hf;
  threadgroup float red[TPG / 32][NM + 2];
  const int sg = tid / 32, lane = tid % 32;
  for (int i = 0; i < NM + 2; i++) {
    const float v = simd_sum(part[i]);
    if (lane == 0) red[sg][i] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < NM + 2) {
    float v = 0.0f;
    for (int s = 0; s < TPG / 32; s++) v += red[s][tid];
    partials[tg * (NM + 2) + tid] = v;
  }

  // ---- last-threadgroup epilogue (no second dispatch)
  threadgroup uint ticket;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  if (tid == 0) ticket = atomic_fetch_add_explicit((device atomic_uint*)counter, 1u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (ticket != uint(NT - 1)) return;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  if (tid == 0) atomic_store_explicit((device atomic_uint*)counter, 0u, memory_order_relaxed);

  threadgroup float tot[NM + 2];
  threadgroup float rstd_h;
  if (tid < NM + 2) {
    float v = 0.0f;
    for (int t = 0; t < NT; t++) v += partials[t * (NM + 2) + tid];
    tot[tid] = v;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid < 32) {  // one simdgroup: lane = j * 4 + k holds comb[j][k] (lanes 0..15)
    const float eps = epsv[0];
    const float hc_eps = 1e-6f;
    const float rs = metal::precise::rsqrt(tot[NM] / float(HC * D) + eps);
    const int lane = tid;
    const int li = lane & 15;
    float c = tot[2 * HC + li] * rs * sc[2] + base[2 * HC + li];
    float mx_ = max(c, simd_shuffle_xor(c, 1));
    mx_ = max(mx_, simd_shuffle_xor(mx_, 2));
    c = metal::precise::exp(c - mx_);
    float s = c + simd_shuffle_xor(c, 1);
    s = s + simd_shuffle_xor(s, 2);
    c = c / s + hc_eps;
    s = c + simd_shuffle_xor(c, 4);
    s = s + simd_shuffle_xor(s, 8);
    c = c / (s + hc_eps);
    for (int it = 0; it < ITERS - 1; it++) {
      s = c + simd_shuffle_xor(c, 1);
      s = s + simd_shuffle_xor(s, 2);
      c = c / (s + hc_eps);
      s = c + simd_shuffle_xor(c, 4);
      s = s + simd_shuffle_xor(s, 8);
      c = c / (s + hc_eps);
    }
    if (lane < 16) comb_o[lane] = c;
    if (lane < HC) {
      pre_o[lane] = sigmoid_mlx(tot[lane] * rs * sc[0] + base[lane]) + hc_eps;
      post_o[lane] = 2.0f * sigmoid_mlx(tot[HC + lane] * rs * sc[1] + base[HC + lane]);
    }
    if (lane == 0) rstd_h = metal::precise::rsqrt(tot[NM + 1] / float(D) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float r = rstd_h;
  for (int d = tid; d < D; d += TPG) {
    hnorm[d] = bfloat16_t(float(normw[d]) * (float(hraw[d]) * r));
  }
"""


HC_TPG = 64  # threads (= d values) per hc_partial threadgroup
HC_FUSED = True  # one dispatch per mHC stage (last-threadgroup epilogue)


@lru_cache(maxsize=None)
def _dummies(D):
    z = (mx.zeros((D,), dtype=mx.bfloat16), mx.zeros((4,), dtype=mx.float32), mx.zeros((16,), dtype=mx.float32))
    mx.eval(z)
    return z


@lru_cache(maxsize=None)
def _hc_partial_kernel():
    return mx.fast.metal_kernel(name="mk_hc_partial", input_names=["hin", "xres", "post", "comb", "premix", "fn"],
                                output_names=["xnew", "hraw", "partials"], header=HEADER, source=_HC_PARTIAL)


@lru_cache(maxsize=None)
def _hc_finalize_kernel():
    return mx.fast.metal_kernel(name="mk_hc_finalize", input_names=["partials", "hraw", "normw", "sc", "base", "epsv"],
                                output_names=["hnorm", "pre_o", "post_o", "comb_o"], header=HEADER, source=_HC_FINALIZE)


@lru_cache(maxsize=None)
def _hc_fused_kernel():
    return mx.fast.metal_kernel(name="mk_hc_fused", input_names=["hin", "xres", "post", "comb", "premix", "fn", "normw", "sc", "base", "epsv", "counter"],
                                output_names=["xnew", "hraw", "partials", "hnorm", "pre_o", "post_o", "comb_o"], header=HEADER, source=_HC_FUSED)


_COUNTERS: dict = {}


def hc_counter(key) -> mx.array:
    """A persistent ticket counter for one call site (the kernel resets it to 0 when done)."""
    c = _COUNTERS.get(key)
    if c is None:
        c = _COUNTERS[key] = mx.zeros((4096,), dtype=mx.uint32)  # large enough to be bound as device memory
        mx.eval(c)
    return c


def hc_stage(x_res: mx.array, premix: mx.array, fn: mx.array, sc: mx.array, base: mx.array, norm_w: mx.array,
             eps: mx.array, hin: mx.array | None = None, post: mx.array | None = None, comb: mx.array | None = None,
             counter: mx.array | None = None):
    """One mHC stage for a single token.
    x_res [4, D] bf16 residual streams; if hin is given, first X = hc_post(hin, x_res, post, comb).
    -> (X [4, D] bf16, h_norm [D] bf16, pre [4], post [4], comb [4, 4]) with h = sum_j premix[j] X[j]
    normalized by norm_w, and (pre, post, comb) the Sinkhorn mixes of X."""
    D = x_res.shape[-1]
    nt = D // HC_TPG
    has_post = hin is not None
    if not has_post:
        hin, post, comb = _dummies(D)
    if HC_FUSED and counter is not None:
        xnew, hraw, _, hnorm, pre_o, post_o, comb_o = _hc_fused_kernel()(
            inputs=[hin.reshape(-1), x_res.reshape(-1), post.reshape(-1), comb.reshape(-1), premix.reshape(-1), fn, norm_w, sc, base,
                    eps.reshape(1), counter],
            template=[("DIM", D), ("TPG", HC_TPG), ("HAS_POST", has_post), ("NT", nt), ("ITERS", 20)],
            grid=(D, 1, 1), threadgroup=(HC_TPG, 1, 1),
            output_shapes=[(4 * D,), (D,), (nt * 26,), (D,), (4,), (4,), (16,)],
            output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32, mx.bfloat16, mx.float32, mx.float32, mx.float32],
        )
        return xnew.reshape(4, D), hnorm, pre_o, post_o, comb_o.reshape(4, 4)
    xnew, hraw, partials = _hc_partial_kernel()(
        inputs=[hin.reshape(-1), x_res.reshape(-1), post.reshape(-1), comb.reshape(-1), premix.reshape(-1), fn],
        template=[("DIM", D), ("TPG", HC_TPG), ("HAS_POST", has_post)],
        grid=(D, 1, 1), threadgroup=(HC_TPG, 1, 1),
        output_shapes=[(4 * D,), (D,), (nt * 26,)], output_dtypes=[mx.bfloat16, mx.bfloat16, mx.float32],
    )
    hnorm, pre_o, post_o, comb_o = _hc_finalize_kernel()(
        inputs=[partials, hraw, norm_w, sc, base, eps.reshape(1)],
        template=[("DIM", D), ("NT", nt), ("ITERS", 20), ("TPGF", 256)],
        grid=(256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(D,), (4,), (4,), (16,)], output_dtypes=[mx.bfloat16, mx.float32, mx.float32, mx.float32],
    )
    return xnew.reshape(4, D), hnorm, pre_o, post_o, comb_o.reshape(4, 4)


# ------------------------------------------------------------------------------------- mxfp8 GEMV
# y[r] = sum_g 2^(s[r,g]-127) * sum_{i in g} e4m3(w[r,i]) * xin[i]   (MLX mxfp8 layout: uint32 words of
# 4 e4m3 codes, uint8 e8m0 scale per 32). Each simdgroup owns RS consecutive rows; its 32 lanes stride
# over K in 8-element chunks (4 lanes = one 32-block); SPLIT simdgroups of a threadgroup share the same
# rows and split the K iterations, reduced through threadgroup memory.
# PRO (activation prologue): 0 = x as given, 1 = fp8 fake-quant of x (ops.act_quant), 2 = RMSNorm(x, nw)
# then fp8 fake-quant (rms in fp32, weight applied, rounded to the dtype of x like ops.rms_norm).
_GEMV_MXFP8_HDR = HEADER + r"""
inline float e4m3_to_float(uint b) {
  uint16_t v = b & 127;
  uint16_t sign_bit = ((uint16_t)((b >> 7) & 1)) << 15;
  uint16_t u = (v << 7) | (((v + 1) >> 7) << 14) | sign_bit;
  return float(as_type<half>(u)) * 256.0f;
}
inline float e8m0_to_float(uint8_t s) { return as_type<float>(uint(s) << 23); }
// MLX LogAddExp(z, 0) with MLX's log1p (utils.h)
inline float log1p_mlx(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == FLT_MAX) return FLT_MAX;
  if (xp1 == 1.0f) return x;
  return x * (metal::log(xp1) / (xp1 - 1.0f));
}
inline float softplus_mlx(float z) {
  const float maxval = max(z, 0.0f), minval = min(z, 0.0f);
  if (isinf(maxval)) return maxval;
  return maxval + log1p_mlx(metal::exp(minval - maxval));
}
"""

_GEMV_MXFP8 = r"""
  const int lane = thread_index_in_simdgroup;
  const int sg = simdgroup_index_in_threadgroup;           // 0 .. SGS*SPLIT-1
  const int part = sg % SPLIT;                              // K split index
  const int rgrp = sg / SPLIT;                              // row group within the threadgroup
  const int r0 = (threadgroup_position_in_grid.y * SGS + rgrp) * RS;
  constexpr int ITERS = K / 256;
  if (GROWS > 0) x += (r0 / GROWS) * K;     // grouped (block-diagonal) weights: rows of group g read input slice g
  float rinv = 1.0f;
  if (PRO == 2) {  // RMSNorm statistic over the whole input (cheap: x is small and cached)
    float ss = 0.0f;
    for (int i = lane; i < K; i += 32) { const float v = float(x[i]); ss += v * v; }
    ss = simd_sum(ss);
    rinv = metal::precise::rsqrt(ss / float(K) + epsv[0]);
  }
  float acc[RS] = {0};
  const device uint8_t* wb = (const device uint8_t*)w;
  for (int it = part; it < ITERS; it += SPLIT) {
    const int c = it * 256 + lane * 8;
    float xv[8];
    for (int j = 0; j < 8; j++) {
      float v = float(x[c + j]);
      if (PRO == 2) v = float(static_cast<XT>(float(nw[c + j]) * (v * rinv)));
      xv[j] = v;
    }
    if (PRO >= 1) {
      float amax = 0.0f;
      for (int j = 0; j < 8; j++) amax = max(amax, fabs(xv[j]));
      amax = max(amax, simd_shuffle_xor(amax, 1));
      amax = max(amax, simd_shuffle_xor(amax, 2));
      const float s = act_scale(amax);
      for (int j = 0; j < 8; j++) xv[j] = fq8(xv[j], s);
    }
    const int g = c / 32;
    for (int r = 0; r < RS; r++) {
      const int row = r0 + r;
      const uint2 q = *((const device uint2*)(wb + (size_t)row * K + c));
      float d = 0.0f;
      d += e4m3_to_float(q.x & 0xFF) * xv[0];
      d += e4m3_to_float((q.x >> 8) & 0xFF) * xv[1];
      d += e4m3_to_float((q.x >> 16) & 0xFF) * xv[2];
      d += e4m3_to_float(q.x >> 24) * xv[3];
      d += e4m3_to_float(q.y & 0xFF) * xv[4];
      d += e4m3_to_float((q.y >> 8) & 0xFF) * xv[5];
      d += e4m3_to_float((q.y >> 16) & 0xFF) * xv[6];
      d += e4m3_to_float(q.y >> 24) * xv[7];
      acc[r] += e8m0_to_float(sc[(size_t)row * (K / 32) + g]) * d;
    }
  }
  if (SPLIT == 1 && EPI == 1) {           // RoPE on head dims [HD-RD, HD) (pairs), like ops.rope_tail
    float v[RS];
    for (int r = 0; r < RS; r++) v[r] = float(static_cast<OT>(simd_sum(acc[r])));
    if (lane == 0) {
      const int d0 = r0 % HD;
      for (int r = 0; r < RS; r += 2) {
        const int d = d0 + r;
        if (d >= HD - RD) {
          const int i = (d - (HD - RD)) / 2;
          const float c = cs[i], sn = cs[RD / 2 + i];
          const float a = v[r], b = v[r + 1];
          y[r0 + r] = static_cast<OT>(a * c - b * sn);
          y[r0 + r + 1] = static_cast<OT>(a * sn + b * c);
        } else {
          y[r0 + r] = static_cast<OT>(v[r]);
          y[r0 + r + 1] = static_cast<OT>(v[r + 1]);
        }
      }
    }
  } else if (SPLIT == 1) {
    for (int r = 0; r < RS; r++) {
      const float v = simd_sum(acc[r]);
      if (lane == 0) y[r0 + r] = static_cast<OT>(v);
    }
  } else {
    threadgroup float red[SGS * SPLIT][RS];
    for (int r = 0; r < RS; r++) {
      const float v = simd_sum(acc[r]);
      if (lane == 0) red[sg][r] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (part == 0 && lane < RS) {
      float v = 0.0f;
      for (int p = 0; p < SPLIT; p++) v += red[rgrp * SPLIT + p][lane];
      y[r0 + lane] = static_cast<OT>(v);
    }
  }
"""


@lru_cache(maxsize=None)
def _gemv_mxfp8_kernel():
    return mx.fast.metal_kernel(name="mk_gemv_mxfp8", input_names=["x", "w", "sc", "nw", "epsv", "cs"], output_names=["y"],
                                header=_GEMV_MXFP8_HDR, source=_GEMV_MXFP8)


def gemv_mxfp8(x: mx.array, w: mx.array, scales: mx.array, pro: int = 1, norm_w: mx.array | None = None,
               eps: mx.array | None = None, rs: int = 8, sgs: int = 1, split: int = 1, out_dtype=None,
               group_rows: int = 0, rope_cs: mx.array | None = None, head_dim: int = 512, rope_dim: int = 64) -> mx.array:
    """y = W @ prologue(x) for one token; W mxfp8 [N, K] (uint32 [N, K/4] + uint8 [N, K/32]).
    pro: 0 none, 1 act_quant(x), 2 act_quant(rms_norm(x, norm_w, eps)).
    group_rows > 0: block-diagonal weights, rows [g*group_rows, (g+1)*group_rows) read x[g*K:(g+1)*K].
    rope_cs: [rope_dim] (cos | sin) -> RoPE epilogue on dims [head_dim-rope_dim, head_dim) of every
    head_dim-row head of the output (rs must be even; split must be 1)."""
    N, K = w.shape[0], w.shape[1] * 4
    assert K % 256 == 0 and N % (rs * sgs) == 0, (N, K)
    x = x.reshape(-1)
    out_dtype = out_dtype or x.dtype
    nw = norm_w if norm_w is not None else x
    ev = (eps if eps is not None else mx.array(0.0, dtype=mx.float32)).reshape(1)
    epi = 1 if rope_cs is not None else 0
    if epi:
        assert split == 1 and rs % 2 == 0
    cs = rope_cs.reshape(-1) if rope_cs is not None else ev
    return _gemv_mxfp8_kernel()(
        inputs=[x, w, scales, nw, ev, cs],
        template=[("K", K), ("RS", rs), ("SGS", sgs), ("SPLIT", split), ("PRO", pro), ("XT", x.dtype), ("OT", out_dtype),
                  ("GROWS", group_rows), ("EPI", epi), ("HD", head_dim), ("RD", rope_dim)],
        grid=(32 * sgs * split, N // (rs * sgs), 1), threadgroup=(32 * sgs * split, 1, 1),
        output_shapes=[(N,)], output_dtypes=[out_dtype],
    )[0]


# ------------------------------------------------------------------------------- attention (decode)
# attn_aux: qr = rms_norm(qa, q_norm) (bf16, for wq_b and the indexer) and
#           kv = act_quant(rope_tail(rms_norm(kvr, kv_norm), cs)) (bf16) -- one threadgroup.
_ATTN_AUX = r"""
  const int tid = thread_position_in_threadgroup.x;     // 0..255
  const int lane = tid % 32, sg = tid / 32;
  threadgroup float red[2][8];
  float sq = 0.0f, sk = 0.0f;
  for (int i = tid; i < QA; i += 256) { const float v = float(qakv[i]); sq += v * v; }
  for (int i = tid; i < KVD; i += 256) { const float v = float(qakv[QA + i]); sk += v * v; }
  sq = simd_sum(sq); sk = simd_sum(sk);
  if (lane == 0) { red[0][sg] = sq; red[1][sg] = sk; }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float tq = 0.0f, tk = 0.0f;
  for (int i = 0; i < 8; i++) { tq += red[0][i]; tk += red[1][i]; }
  const float eps = epsv[0];
  const float rq = metal::precise::rsqrt(tq / float(QA) + eps);
  const float rk = metal::precise::rsqrt(tk / float(KVD) + eps);
  for (int i = tid; i < QA; i += 256) qr[i] = bfloat16_t(float(qnw[i]) * (float(qakv[i]) * rq));
  // kv: 512 values, 2 per thread (one RoPE pair), then act_quant per 32 (16 threads)
  const int d = tid * 2;
  float a = float(bfloat16_t(float(kvnw[d]) * (float(qakv[QA + d]) * rk)));
  float b = float(bfloat16_t(float(kvnw[d + 1]) * (float(qakv[QA + d + 1]) * rk)));
  if (d >= KVD - RD) {
    const int i = (d - (KVD - RD)) / 2;
    const float c = cs[i], sn = cs[RD / 2 + i];
    const float a2 = float(bfloat16_t(a * c - b * sn));
    const float b2 = float(bfloat16_t(a * sn + b * c));
    a = a2; b = b2;
  }
  float amax = max(fabs(a), fabs(b));
  amax = max(amax, simd_shuffle_xor(amax, 1));
  amax = max(amax, simd_shuffle_xor(amax, 2));
  amax = max(amax, simd_shuffle_xor(amax, 4));
  amax = max(amax, simd_shuffle_xor(amax, 8));
  const float s = act_scale(amax);
  kv[d] = bfloat16_t(fq8(a, s));
  kv[d + 1] = bfloat16_t(fq8(b, s));
"""

# sparse_attn: grid = 64/HPT threadgroups (HPT heads each), 256 threads. Keys: the window (128 slots
# indexing [ring | new], -1 empty) then NC compressed rows (cidx, -1 empty). Same math as
# Attention._core: fp32 scores * scale, max (>= -1e30), p = exp(s - m), denominator sum(p) +
# exp(sink - m), numerator sum(bf16(p) * k); then bf16, inverse RoPE on [D-RD, D), act_quant per 32.
_SPARSE_ATTN = r"""
  constexpr int D = 512;
  constexpr int NW = 128;
  constexpr int NK = NW + NC;
  const int tid = thread_position_in_threadgroup.x;
  const int lane = tid % 32, sg = tid / 32;
  const int h0 = threadgroup_position_in_grid.x * HPT;
  threadgroup float sc[HPT][NK];
  threadgroup float stat[HPT][2];
  // q slice of this lane for HPT heads: dims [lane*16, lane*16+16)
  float qv[HPT][16];
  for (int h = 0; h < HPT; h++)
    for (int j = 0; j < 16; j++) qv[h][j] = float(q[(h0 + h) * D + lane * 16 + j]);
  // pass 1: scores
  for (int k = sg; k < NK; k += 8) {
    int src = -1;
    const device bfloat16_t* kp = nullptr;
    if (k < NW) {
      src = widx[k];
      if (src >= 0) kp = (src == NW) ? kvnew : (ring + (size_t)src * D);
    } else {
      src = cidx[k - NW];
      if (src >= 0) kp = comp + (size_t)src * D;
    }
    if (src < 0) {
      if (lane < HPT) sc[lane][k] = -INFINITY;
      continue;
    }
    float kv_[16];
    for (int j = 0; j < 16; j++) kv_[j] = float(kp[lane * 16 + j]);
    for (int h = 0; h < HPT; h++) {
      float dsum = 0.0f;
      for (int j = 0; j < 16; j++) dsum += qv[h][j] * kv_[j];
      dsum = simd_sum(dsum);
      if (lane == 0) sc[h][k] = dsum * scale[0];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // max and denominator per head: simdgroup h (< HPT) reduces head h
  if (sg < HPT) {
    float m = -INFINITY;
    for (int k = lane; k < NK; k += 32) m = max(m, sc[sg][k]);
    m = max(simd_max(m), -1e30f);
    float sum = 0.0f;
    for (int k = lane; k < NK; k += 32) {
      const float p = metal::precise::exp(sc[sg][k] - m);
      sum += p;
      sc[sg][k] = float(bfloat16_t(p));              // numerator uses bf16(p)
    }
    sum = simd_sum(sum);
    if (lane == 0) { stat[sg][0] = m; stat[sg][1] = sum + metal::precise::exp(sink[h0 + sg] - m); }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // pass 2: o[h, d] for d = 2*tid, 2*tid+1
  const int d = tid * 2;
  float o[HPT][2];
  for (int h = 0; h < HPT; h++) { o[h][0] = 0.0f; o[h][1] = 0.0f; }
  for (int k = 0; k < NK; k++) {
    int src;
    const device bfloat16_t* kp;
    if (k < NW) { src = widx[k]; kp = (src == NW) ? kvnew : (ring + (size_t)max(src, 0) * D); }
    else { src = cidx[k - NW]; kp = comp + (size_t)max(src, 0) * D; }
    if (src < 0) continue;
    const float k0 = float(kp[d]), k1 = float(kp[d + 1]);
    for (int h = 0; h < HPT; h++) {
      const float p = sc[h][k];
      o[h][0] += p * k0;
      o[h][1] += p * k1;
    }
  }
  {  // ring update: threadgroup g copies rows [g*NW/NTG, (g+1)*NW/NTG) of [ring with row rslot = kvnew]
    const int ntg = threadgroups_per_grid.x;
    const int rows = NW / ntg;
    const int rs0 = threadgroup_position_in_grid.x * rows;
    const int rsl = rslot[0];
    for (int i = tid; i < rows * D; i += 256) {
      const int rr = rs0 + i / D, cc = i % D;
      ring_out[rr * D + cc] = (rr == rsl) ? kvnew[cc] : ring[rr * D + cc];
    }
  }
  for (int h = 0; h < HPT; h++) {
    const float den = stat[h][1];
    float a = float(bfloat16_t(o[h][0] / den));
    float b = float(bfloat16_t(o[h][1] / den));
    if (d >= D - RD) {                                // inverse RoPE: sin -> -sin
      const int i = (d - (D - RD)) / 2;
      const float c = cs[i], sn = -cs[RD / 2 + i];
      const float a2 = float(bfloat16_t(a * c - b * sn));
      const float b2 = float(bfloat16_t(a * sn + b * c));
      a = a2; b = b2;
    }
    float amax = max(fabs(a), fabs(b));
    amax = max(amax, simd_shuffle_xor(amax, 1));
    amax = max(amax, simd_shuffle_xor(amax, 2));
    amax = max(amax, simd_shuffle_xor(amax, 4));
    amax = max(amax, simd_shuffle_xor(amax, 8));
    const float s = act_scale(amax);
    out[(h0 + h) * D + d] = bfloat16_t(fq8(a, s));
    out[(h0 + h) * D + d + 1] = bfloat16_t(fq8(b, s));
  }
"""


@lru_cache(maxsize=None)
def _attn_aux_kernel():
    return mx.fast.metal_kernel(name="mk_attn_aux", input_names=["qakv", "qnw", "kvnw", "epsv", "cs"], output_names=["qr", "kv"],
                                header=HEADER, source=_ATTN_AUX)


@lru_cache(maxsize=None)
def _sparse_attn_kernel():
    return mx.fast.metal_kernel(name="mk_sparse_attn", input_names=["q", "ring", "kvnew", "widx", "comp", "cidx", "sink", "scale", "cs", "rslot"],
                                output_names=["out", "ring_out"], header=HEADER, source=_SPARSE_ATTN)


# Split sparse attention (default). mk_sparse_attn above runs H / 4 = 16 threadgroups that each walk every
# key twice in a serial loop, so at BS=1 it is latency-bound: 86 us/layer at short context and 195 us at 4K
# context, slower than the MLX op chain (133 us) once the 512 compressed keys are all valid. The same
# computation as two parallel dispatches:
#   sa_scores: threadgroup = 8 heads x 64 keys (8 per simdgroup); q of the 8 heads staged in threadgroup
#              memory; per key and head 16 products per lane in order, then simd_sum (as mk_sparse_attn)
#   sa_pv    : threadgroup = 8 heads x 64 output dims; per head the softmax with sink (lane-strided max /
#              sum, bf16(p) numerator, as mk_sparse_attn); the 8 simdgroups each sum NK/8 keys in order for
#              all 8 heads (one value load feeds 16 FMAs) and the 8 partials are added in a fixed order;
#              inverse RoPE, fp8 act-quant; the H threadgroups also write the new ring.
# Only the PV sum is regrouped (8 ordered partials). On real decode states (bench_attn_speedup.py
# --check-only): identical to mk_sparse_attn at short context, at most 1 of 32768 output elements per layer
# differs at 4K; against the MLX reference chain at most 12 elements per layer differ by one fp8 step, exactly
# as for mk_sparse_attn (fp32 summation order of QK). 49 / 82 us/layer at short / 4K context.
# DEEPINFER_MK_ATTN_SPLIT=0 selects the old kernel.
_SA_SCORES = r"""
  constexpr int D = 512;
  constexpr int NW = 128;
  constexpr int NK = NW + NC;
  constexpr int HB = 8;                            // heads per threadgroup
  constexpr int KB = 64;                           // keys per threadgroup, 8 per simdgroup
  const int lane = thread_index_in_simdgroup;
  const int sg = simdgroup_index_in_threadgroup;
  const int tid = thread_position_in_threadgroup.x;
  const int hb = threadgroup_position_in_grid.x % (H / HB);
  const int kb = threadgroup_position_in_grid.x / (H / HB);
  // q of this head block, transposed so lane L reads element L*16+j at [j][L] (no bank conflicts)
  threadgroup float qs[HB][16][32];
  for (int i = tid; i < HB * D; i += 256) {
    const int h = i / D, r = i % D;
    qs[h][r % 16][r / 16] = float(q[(hb * HB + h) * D + r]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float s = scale[0];
  for (int i = 0; i < KB / 8; i++) {
    const int k = kb * KB + sg * (KB / 8) + i;
    int src;
    const device bfloat16_t* kp = nullptr;
    if (k < NW) {
      src = widx[k];
      if (src >= 0) kp = (src == NW) ? kvnew : (ring + (size_t)src * D);
    } else {
      src = cidx[k - NW];
      if (src >= 0) kp = comp + (size_t)src * D;
    }
    if (src < 0) {
      if (lane < HB) sc[(hb * HB + lane) * NK + k] = -INFINITY;
      continue;
    }
    const device uint4* kp4 = (const device uint4*)(kp + lane * 16);
    const uint4 ka = kp4[0], kc = kp4[1];
    const float kv_[16] = {bf_lo(ka.x), bf_hi(ka.x), bf_lo(ka.y), bf_hi(ka.y), bf_lo(ka.z), bf_hi(ka.z), bf_lo(ka.w), bf_hi(ka.w),
                           bf_lo(kc.x), bf_hi(kc.x), bf_lo(kc.y), bf_hi(kc.y), bf_lo(kc.z), bf_hi(kc.z), bf_lo(kc.w), bf_hi(kc.w)};
    for (int h = 0; h < HB; h++) {
      float dsum = 0.0f;
      for (int j = 0; j < 16; j++) dsum += qs[h][j][lane] * kv_[j];
      dsum = simd_sum(dsum);
      if (lane == 0) sc[(hb * HB + h) * NK + k] = dsum * s;
    }
  }
"""

_SA_PV = r"""
  constexpr int D = 512;
  constexpr int NW = 128;
  constexpr int NK = NW + NC;
  constexpr int HB = 8;                            // heads per threadgroup
  constexpr int KC = NK / 8;                       // keys per simdgroup in the PV split
  const int lane = thread_index_in_simdgroup;
  const int sg = simdgroup_index_in_threadgroup;
  const int tid = thread_position_in_threadgroup.x;
  const int tgi = threadgroup_position_in_grid.x;
  const int hb = tgi / 8;                          // head block
  const int dl = lane * 2;
  const int d = (tgi % 8) * 64 + dl;               // this lane's output dims d, d+1
  threadgroup bfloat16_t pk[HB][NK];               // bf16(p) per head and key
  threadgroup int rowc[NK];                        // key source: -1 none, r < NW ring row, NW new kv, NW+1+c comp row c
  threadgroup float part[8][HB][64];               // per-simdgroup partial o
  threadgroup float dens[HB];
  for (int k = tid; k < NK; k += 256) {
    if (k < NW) rowc[k] = widx[k];
    else { const int c = cidx[k - NW]; rowc[k] = c < 0 ? -1 : NW + 1 + c; }
  }
  {  // softmax with sink for head hb*8+sg (same reduction order as mk_sparse_attn)
    const int h = hb * HB + sg;
    const device float* sh = sc + (size_t)h * NK;
    float m = -INFINITY;
    for (int k = lane; k < NK; k += 32) m = max(m, sh[k]);
    m = max(simd_max(m), -1e30f);
    float sum = 0.0f;
    for (int k = lane; k < NK; k += 32) {
      const float p = metal::precise::exp(sh[k] - m);
      sum += p;
      pk[sg][k] = bfloat16_t(p);                   // numerator uses bf16(p)
    }
    sum = simd_sum(sum);
    if (lane == 0) dens[sg] = sum + metal::precise::exp(sink[h] - m);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float o[HB][2];
  for (int h = 0; h < HB; h++) { o[h][0] = 0.0f; o[h][1] = 0.0f; }
  for (int k = sg * KC; k < (sg + 1) * KC; k++) {   // keys of this simdgroup, in order
    const int code = rowc[k];
    if (code < 0) continue;
    const device bfloat16_t* kp = code < NW ? ring + (size_t)code * D : (code == NW ? kvnew : comp + (size_t)(code - NW - 1) * D);
    const uint w = *((const device uint*)(kp + d));
    const float v0 = bf_lo(w), v1 = bf_hi(w);
    for (int h = 0; h < HB; h++) {
      const float p = float(pk[h][k]);
      o[h][0] += p * v0;
      o[h][1] += p * v1;
    }
  }
  for (int h = 0; h < HB; h++) { part[sg][h][dl] = o[h][0]; part[sg][h][dl + 1] = o[h][1]; }
  {  // ring update: threadgroup g copies rows [g*NW/NTG, (g+1)*NW/NTG) of [ring with row rslot = kvnew]
    const int rows = NW / (int)threadgroups_per_grid.x;
    const int rs0 = tgi * rows;
    const int rsl = rslot[0];
    for (int i = tid; i < rows * D; i += 256) {
      const int rr = rs0 + i / D, cc = i % D;
      ring_out[rr * D + cc] = (rr == rsl) ? kvnew[cc] : ring[rr * D + cc];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // combine the 8 key chunks in order (simdgroup sg finishes head hb*8+sg), then the epilogue
  const int h = hb * HB + sg;
  float a0 = 0.0f, a1 = 0.0f;
  for (int c = 0; c < 8; c++) { a0 += part[c][sg][dl]; a1 += part[c][sg][dl + 1]; }
  const float den = dens[sg];
  float a = float(bfloat16_t(a0 / den));
  float b = float(bfloat16_t(a1 / den));
  if (d >= D - RD) {                                // inverse RoPE: sin -> -sin
    const int i = (d - (D - RD)) / 2;
    const float c = cs[i], sn = -cs[RD / 2 + i];
    const float a2 = float(bfloat16_t(a * c - b * sn));
    const float b2 = float(bfloat16_t(a * sn + b * c));
    a = a2; b = b2;
  }
  float amax = max(fabs(a), fabs(b));
  amax = max(amax, simd_shuffle_xor(amax, 1));
  amax = max(amax, simd_shuffle_xor(amax, 2));
  amax = max(amax, simd_shuffle_xor(amax, 4));
  amax = max(amax, simd_shuffle_xor(amax, 8));
  const float s = act_scale(amax);
  out[h * D + d] = bfloat16_t(fq8(a, s));
  out[h * D + d + 1] = bfloat16_t(fq8(b, s));
"""


@lru_cache(maxsize=None)
def _sa_scores_kernel():
    return mx.fast.metal_kernel(name="mk_sa_scores", input_names=["q", "ring", "kvnew", "widx", "comp", "cidx", "scale"],
                                output_names=["sc"], header=HEADER, source=_SA_SCORES)


@lru_cache(maxsize=None)
def _sa_pv_kernel():
    return mx.fast.metal_kernel(name="mk_sa_pv", input_names=["sc", "ring", "kvnew", "widx", "comp", "cidx", "sink", "cs", "rslot"],
                                output_names=["out", "ring_out"], header=HEADER, source=_SA_PV)


def attn_aux(qakv: mx.array, q_norm: mx.array, kv_norm: mx.array, eps: mx.array, rope_cs: mx.array, qa_dim: int = 1280,
             kv_dim: int = 512, rope_dim: int = 64):
    """(qr bf16 [qa_dim], kv bf16 [kv_dim]) from the merged wq_a|wkv output."""
    qr, kv = _attn_aux_kernel()(
        inputs=[qakv.reshape(-1), q_norm, kv_norm, eps.reshape(1), rope_cs.reshape(-1)],
        template=[("QA", qa_dim), ("KVD", kv_dim), ("RD", rope_dim)],
        grid=(256, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(qa_dim,), (kv_dim,)], output_dtypes=[mx.bfloat16, mx.bfloat16],
    )
    return qr, kv


def sparse_attn(q: mx.array, ring: mx.array, kvnew: mx.array, widx: mx.array, comp: mx.array | None, cidx: mx.array | None,
                sink: mx.array, scale: mx.array, rope_cs: mx.array, rslot: mx.array, heads_per_tg: int = 4, rope_dim: int = 64):
    """One decode token's sparse attention (window + compressed) for all heads -> (act-quantized,
    inverse-RoPE'd output [H * 512] bf16 (the input of wo_a), the ring with row rslot = kvnew)."""
    H = q.size // 512
    nc = 0 if cidx is None else cidx.size
    if comp is None:
        comp, cidx = kvnew, _neg1()
    if SPLIT_ATTN and H % 8 == 0 and ring.shape[0] % H == 0 and nc % 64 == 0:
        nk = ring.shape[0] + nc
        sc = _sa_scores_kernel()(
            inputs=[q.reshape(-1), ring, kvnew.reshape(-1), widx, comp, cidx.reshape(-1), scale.reshape(1)],
            template=[("NC", nc), ("H", H)],
            grid=(256 * (H // 8) * (nk // 64), 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(H * nk,)], output_dtypes=[mx.float32],
        )[0]
        out, ring_out = _sa_pv_kernel()(
            inputs=[sc, ring, kvnew.reshape(-1), widx, comp, cidx.reshape(-1), sink, rope_cs.reshape(-1), rslot],
            template=[("NC", nc), ("H", H), ("RD", rope_dim)],
            grid=(256 * H, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(H * 512,), ring.shape], output_dtypes=[mx.bfloat16, ring.dtype],
        )
        return out, ring_out
    out, ring_out = _sparse_attn_kernel()(
        inputs=[q.reshape(-1), ring, kvnew.reshape(-1), widx, comp, cidx.reshape(-1), sink, scale.reshape(1), rope_cs.reshape(-1), rslot],
        template=[("NC", nc), ("HPT", heads_per_tg), ("RD", rope_dim)],
        grid=(256 * H // heads_per_tg, 1, 1), threadgroup=(256, 1, 1),
        output_shapes=[(H * 512,), ring.shape], output_dtypes=[mx.bfloat16, ring.dtype],
    )
    return out, ring_out


@lru_cache(maxsize=None)
def _neg1():
    a = mx.full((1,), -1, dtype=mx.int32)
    mx.eval(a)
    return a


# ---------------------------------------------------------------------------- router + shared expert
# rs_up  : rows [0, NG) are router logits x.f32 . gate_w (bf16 weights, fp32 math, like the reference's
#          fp32 matmul of bf16 values) for this layer's gate (and the next layer's, for speculation);
#          rows [NG, NG + 2*INTER) are the shared expert's w1|w3 (mxfp8, act_quant(x) in-register);
#          the first mxfp8 simdgroup also writes xq = act_quant(x) for the routed experts.
# rs_down: threadgroup 0 selects top-k (biased sqrt-softplus scores) + weights, and the next layer's
#          top-SPECK prediction; the others compute y = w2 . act_quant(bf16(swiglu(g, u))) (shared).
_RS_UP = r"""
  constexpr int SGN = 8;
  const int lane = thread_index_in_simdgroup;
  const int sg = simdgroup_index_in_threadgroup;
  const int tg = threadgroup_position_in_grid.y;
  constexpr int ITERS = K / 256;
  constexpr int GTG = NG / 32;                      // router threadgroups
  if (tg < GTG) {                                   // router rows (bf16 weights), 4 per simdgroup
    constexpr int RS = 4;
    const int r0 = (tg * SGN + sg) * RS;
    float acc[RS] = {0};
    for (int it = 0; it < ITERS; it++) {
      const int c = it * 256 + lane * 8;
      float xv[8];
      const uint4 xw = *((const device uint4*)(x + c));
      xv[0] = bf_lo(xw.x); xv[1] = bf_hi(xw.x); xv[2] = bf_lo(xw.y); xv[3] = bf_hi(xw.y);
      xv[4] = bf_lo(xw.z); xv[5] = bf_hi(xw.z); xv[6] = bf_lo(xw.w); xv[7] = bf_hi(xw.w);
      for (int r = 0; r < RS; r++) {
        const int gr = r0 + r;
        const device bfloat16_t* gbase = (gr < NE) ? (gw + (size_t)gr * K)
                                       : (gr < 2 * NE) ? (gwn + (size_t)(gr - NE) * K) : (gwn2 + (size_t)(gr - 2 * NE) * K);
        const uint4 ww = *((const device uint4*)(gbase + c));
        float d = bf_lo(ww.x) * xv[0];
        d += bf_hi(ww.x) * xv[1];
        d += bf_lo(ww.y) * xv[2];
        d += bf_hi(ww.y) * xv[3];
        d += bf_lo(ww.z) * xv[4];
        d += bf_hi(ww.z) * xv[5];
        d += bf_lo(ww.w) * xv[6];
        d += bf_hi(ww.w) * xv[7];
        acc[r] += d;
      }
    }
    for (int r = 0; r < RS; r++) {
      const float v = simd_sum(acc[r]);
      if (lane == 0) logits[r0 + r] = v;
    }
    return;
  }
  // shared expert: this threadgroup owns h rows [hb0, hb0 + 32): w1 rows and the same w3 rows
  constexpr int RS = 4;
  const int hb0 = (tg - GTG) * 32;
  const int m0 = hb0 + sg * RS;
  const bool write_xq = (tg == GTG && sg == 0);
  float acc1[RS] = {0}, acc3[RS] = {0};
  const device uint8_t* wb = (const device uint8_t*)w13;
  for (int it = 0; it < ITERS; it++) {
    const int c = it * 256 + lane * 8;
    float xv[8];
    float amax = 0.0f;
    const uint4 xw = *((const device uint4*)(x + c));
    xv[0] = bf_lo(xw.x); xv[1] = bf_hi(xw.x); xv[2] = bf_lo(xw.y); xv[3] = bf_hi(xw.y);
    xv[4] = bf_lo(xw.z); xv[5] = bf_hi(xw.z); xv[6] = bf_lo(xw.w); xv[7] = bf_hi(xw.w);
    for (int j = 0; j < 8; j++) amax = max(amax, fabs(xv[j]));
    amax = max(amax, simd_shuffle_xor(amax, 1));
    amax = max(amax, simd_shuffle_xor(amax, 2));
    const float s = act_scale(amax);
    for (int j = 0; j < 8; j++) xv[j] = fq8(xv[j], s);
    if (write_xq) for (int j = 0; j < 8; j++) xq[c + j] = static_cast<bfloat16_t>(xv[j]);
    const int g = c / 32;
    for (int r = 0; r < RS; r++) {
      for (int hp = 0; hp < 2; hp++) {
        const int row = m0 + r + hp * INTER;
        const uint2 q = *((const device uint2*)(wb + (size_t)row * K + c));
        float d = 0.0f;
        d += e4m3_to_float(q.x & 0xFF) * xv[0];
        d += e4m3_to_float((q.x >> 8) & 0xFF) * xv[1];
        d += e4m3_to_float((q.x >> 16) & 0xFF) * xv[2];
        d += e4m3_to_float(q.x >> 24) * xv[3];
        d += e4m3_to_float(q.y & 0xFF) * xv[4];
        d += e4m3_to_float((q.y >> 8) & 0xFF) * xv[5];
        d += e4m3_to_float((q.y >> 16) & 0xFF) * xv[6];
        d += e4m3_to_float(q.y >> 24) * xv[7];
        const float v = e8m0_to_float(w13s[(size_t)row * (K / 32) + g]) * d;
        if (hp == 0) acc1[r] += v; else acc3[r] += v;
      }
    }
  }
  threadgroup float hb[32];
  const float lim = limit[0];
  for (int r = 0; r < RS; r++) {
    float gt = simd_sum(acc1[r]);
    float up = simd_sum(acc3[r]);
    if (lane == 0) {
      gt = float(bfloat16_t(gt));
      up = float(bfloat16_t(up));
      if (lim > 0.0f) { up = clamp(up, -lim, lim); gt = min(gt, lim); }
      hb[sg * RS + r] = float(bfloat16_t(gt * sigmoid_mlx(gt) * up));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float v = hb[lane];
    const float s = act_scale(simd_max(fabs(v)));
    hq[hb0 + lane] = bfloat16_t(fq8(v, s));
  }
"""

_RS_DOWN = r"""
  const int lane = thread_index_in_simdgroup;
  const int sg = simdgroup_index_in_threadgroup;
  if (threadgroup_position_in_grid.y == 0) {
    if (sg != 0) return;
    // ---- routing: one simdgroup, each lane owns NE/32 experts
    constexpr int PER = NE / 32;
    float sc[PER], bs[PER];
    for (int i = 0; i < PER; i++) {
      const int e = lane + 32 * i;
      const float z = logits[e];
      sc[i] = metal::precise::sqrt(softplus_mlx(z));
      scores[e] = sc[i];
      bs[i] = sc[i] + bias[e];
    }
    float wsel[TOPK];
    uint isel[TOPK];
    float wsum = 0.0f;
    for (int k = 0; k < TOPK; k++) {
      float best = -INFINITY; int bi = 0x7fffffff;
      for (int i = 0; i < PER; i++) if (bs[i] > best) { best = bs[i]; bi = lane + 32 * i; }
      const float gbest = simd_max(best);
      const int cand = (best == gbest) ? bi : 0x7fffffff;
      const int gi = simd_min(cand);
      isel[k] = uint(gi);
      if ((gi % 32) == lane) { bs[gi / 32] = -INFINITY; }
      wsel[k] = simd_broadcast(sc[0], 0);  // placeholder, fixed below
    }
    // raw scores of the selected experts
    for (int k = 0; k < TOPK; k++) {
      const int e = int(isel[k]);
      float v = 0.0f;
      for (int i = 0; i < PER; i++) if (lane + 32 * i == e) v = sc[i];
      v = simd_sum(v);
      wsel[k] = v;
      wsum += v;
    }
    if (lane < TOPK) {
      idx[lane] = isel[lane];
      w[lane] = wsel[lane] / (wsum + 1e-20f) * route_scale[0];
    }
    if (SPECK2 > 0) {
      float pb2[PER];
      for (int i = 0; i < PER; i++) {
        const int e = lane + 32 * i;
        pb2[i] = metal::precise::sqrt(softplus_mlx(logits[2 * NE + e])) + bias_n2[e];
      }
      for (int k = 0; k < SPECK2; k++) {
        float best = -INFINITY; int bi = 0x7fffffff;
        for (int i = 0; i < PER; i++) if (pb2[i] > best) { best = pb2[i]; bi = lane + 32 * i; }
        const float gbest = simd_max(best);
        const int cand = (best == gbest) ? bi : 0x7fffffff;
        const int gi = simd_min(cand);
        if (lane == 0) pred2[k] = uint(gi);
        if ((gi % 32) == lane) pb2[gi / 32] = -INFINITY;
      }
    }
    if (SPECK > 0) {
      float pb[PER];
      for (int i = 0; i < PER; i++) {
        const int e = lane + 32 * i;
        const float z = logits[NE + e];
        pb[i] = metal::precise::sqrt(softplus_mlx(z)) + bias_n[e];
      }
      for (int k = 0; k < SPECK; k++) {
        float best = -INFINITY; int bi = 0x7fffffff;
        for (int i = 0; i < PER; i++) if (pb[i] > best) { best = pb[i]; bi = lane + 32 * i; }
        const float gbest = simd_max(best);
        const int cand = (best == gbest) ? bi : 0x7fffffff;
        const int gi = simd_min(cand);
        if (lane == 0) pred[k] = uint(gi);
        if ((gi % 32) == lane) pb[gi / 32] = -INFINITY;
      }
    }
    return;
  }
  // ---- shared expert down projection: rows r0 .. r0+RS of w2 (mxfp8 [DIM, INTER]) on hq
  constexpr int RS = 8;
  constexpr int SGS = 2;
  const int r0 = ((threadgroup_position_in_grid.y - 1) * SGS + sg) * RS;
  constexpr int K = INTER;
  float acc[RS] = {0};
  const device uint8_t* wb = (const device uint8_t*)w2;
  for (int it = 0; it < K / 256; it++) {
    const int c = it * 256 + lane * 8;
    float xv[8];
    const uint4 xw = *((const device uint4*)(hq + c));
    xv[0] = bf_lo(xw.x); xv[1] = bf_hi(xw.x); xv[2] = bf_lo(xw.y); xv[3] = bf_hi(xw.y);
    xv[4] = bf_lo(xw.z); xv[5] = bf_hi(xw.z); xv[6] = bf_lo(xw.w); xv[7] = bf_hi(xw.w);
    const int g = c / 32;
    for (int r = 0; r < RS; r++) {
      const int row = r0 + r;
      const uint2 q = *((const device uint2*)(wb + (size_t)row * K + c));
      float d = 0.0f;
      d += e4m3_to_float(q.x & 0xFF) * xv[0];
      d += e4m3_to_float((q.x >> 8) & 0xFF) * xv[1];
      d += e4m3_to_float((q.x >> 16) & 0xFF) * xv[2];
      d += e4m3_to_float(q.x >> 24) * xv[3];
      d += e4m3_to_float(q.y & 0xFF) * xv[4];
      d += e4m3_to_float((q.y >> 8) & 0xFF) * xv[5];
      d += e4m3_to_float((q.y >> 16) & 0xFF) * xv[6];
      d += e4m3_to_float(q.y >> 24) * xv[7];
      acc[r] += e8m0_to_float(w2s[(size_t)row * (K / 32) + g]) * d;
    }
  }
  for (int r = 0; r < RS; r++) {
    const float v = simd_sum(acc[r]);
    if (lane == 0) y[r0 + r] = float(bfloat16_t(v));
  }
"""


@lru_cache(maxsize=None)
def _rs_up_kernel():
    return mx.fast.metal_kernel(name="mk_rs_up3", input_names=["x", "gw", "gwn", "gwn2", "w13", "w13s", "limit"], output_names=["logits", "hq", "xq"],
                                header=_GEMV_MXFP8_HDR, source=_RS_UP)


@lru_cache(maxsize=None)
def _rs_down_kernel():
    return mx.fast.metal_kernel(name="mk_rs_down3", input_names=["logits", "bias", "bias_n", "bias_n2", "route_scale", "hq", "w2", "w2s"],
                                output_names=["idx", "w", "scores", "pred", "pred2", "y"], header=_GEMV_MXFP8_HDR, source=_RS_DOWN)


def router_shared(x: mx.array, gate_w: mx.array, w13: mx.array, w13s: mx.array, w2: mx.array, w2s: mx.array,
                  bias: mx.array, route_scale: mx.array, limit: mx.array, topk: int = 6,
                  gate_next: mx.array | None = None, bias_next: mx.array | None = None, spec_k: int = 0,
                  gate_next2: mx.array | None = None, bias_next2: mx.array | None = None, spec_k2: int = 0):
    """Decode router + shared expert in two dispatches.
    gate_w, gate_next, gate_next2: bf16 [NE, DIM] (this layer's router; the next two layers', used on
    this layer's input to predict their experts when spec_k / spec_k2 > 0); w13/w13s: shared w1|w3
    mxfp8 [2*INTER, DIM]; w2/w2s: shared w2 mxfp8 [DIM, INTER].
    -> idx [topk] uint32, w [topk] f32 (x route_scale), scores [NE] f32, pred [spec_k] uint32,
       pred2 [spec_k2] uint32, xq [DIM] bf16 (act_quant(x)), y_shared [DIM] f32."""
    D = x.shape[-1]
    NE = gate_w.shape[0]
    spec = spec_k > 0 and gate_next is not None
    spec2 = spec and spec_k2 > 0 and gate_next2 is not None
    NG = NE * (1 + spec + spec2)
    two_inter = w13.shape[0]
    inter = two_inter // 2
    x = x.reshape(-1)
    gn = gate_next if spec else gate_w
    gn2 = gate_next2 if spec2 else gate_w
    logits, hq, xq = _rs_up_kernel()(
        inputs=[x, gate_w, gn, gn2, w13, w13s, limit.reshape(1)],
        template=[("K", D), ("NG", NG), ("NE", NE), ("INTER", inter)],
        grid=(256, NG // 32 + inter // 32, 1), threadgroup=(256, 1, 1),
        output_shapes=[(NG,), (inter,), (D,)], output_dtypes=[mx.float32, mx.bfloat16, mx.bfloat16],
    )
    bn = bias_next if spec else bias
    bn2 = bias_next2 if spec2 else bias
    idx, w, scores, pred, pred2, y = _rs_down_kernel()(
        inputs=[logits, bias, bn, bn2, route_scale.reshape(1), hq, w2, w2s],
        template=[("NE", NE), ("TOPK", topk), ("SPECK", spec_k if spec else 0), ("SPECK2", spec_k2 if spec2 else 0),
                  ("INTER", inter), ("DIM", D)],
        grid=(64, 1 + D // 16, 1), threadgroup=(64, 1, 1),
        output_shapes=[(topk,), (topk,), (NE,), (max(spec_k, 1),), (max(spec_k2, 1),), (D,)],
        output_dtypes=[mx.uint32, mx.float32, mx.float32, mx.uint32, mx.uint32, mx.float32],
    )
    return idx, w, scores, pred, pred2, xq, y


# --------------------------------------------------------------------------------- GPU keep-alive
# Apple GPUs drop to a low-power state after ~1-2 ms without work; the next kernels then run 2-4x
# slower (measured: a 190 us GEMV takes ~1.2 ms after a 4 ms idle gap). While the decode thread
# waits for an SSD read, keep the GPU's submit/complete path busy with tiny kernels on the default
# stream instead of letting it idle.
_SPIN = r"""
  uint t = thread_position_in_grid.x;
  float a = float(t) * 1e-3f;
  for (int i = 0; i < ITERS; i++) a = metal::fma(a, 0.9999f, 1e-4f);
  if (a == 12345.0f) out[t] = a;
"""


@lru_cache(maxsize=None)
def _spin_kernel():
    return mx.fast.metal_kernel(name="mk_spin", input_names=["dummy"], output_names=["out"], source=_SPIN)


@lru_cache(maxsize=None)
def _spin_dummy():
    d = mx.zeros((1,))
    mx.eval(d)
    return d


SPIN_ITERS = 400


def gpu_tick():
    """Run one tiny kernel on the default GPU stream and wait for it (keeps the GPU clocked up)."""
    o = _spin_kernel()(inputs=[_spin_dummy()], template=[("ITERS", SPIN_ITERS)], grid=(32, 1, 1), threadgroup=(32, 1, 1),
                       output_shapes=[(32,)], output_dtypes=[mx.float32])[0]
    mx.eval(o)


def hot_wait(fut) -> None:
    """Block until `fut` (a read request) is done, keeping the GPU busy meanwhile."""
    while not fut.done():
        gpu_tick()
    fut.result()
