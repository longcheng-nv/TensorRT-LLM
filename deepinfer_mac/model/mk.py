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

from functools import lru_cache

import mlx.core as mx

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

inline float fq8(float x, float s) {  // fake-quant one value with block scale s
  return fp8_e4m3_round(clamp(x / s, -448.0f, 448.0f)) * s;
}

inline float sigmoid_mlx(float x) {
  float y = 1.0f / (1.0f + metal::precise::exp(metal::abs(x)));
  return (x < 0.0f) ? y : 1.0f - y;
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
# grid: (32 * SG, n_experts * inter / (RPS * SG), 1) threads; threadgroup (32 * SG, 1, 1)
# each simdgroup: RPS consecutive rows of one expert (w1 and w3), lane l covers 16 values per step
_MOE_UP = r"""
  constexpr int RPS = 4;
  constexpr int SG = 2;
  const int K = DIM;          // 5120
  const int N = INTER;        // 2304
  const int lane = thread_index_in_simdgroup;
  const int sg = simdgroup_index_in_threadgroup;
  const int row0 = (threadgroup_position_in_grid.y * SG + sg) * RPS;   // over n_experts * N
  const int e = row0 / N;
  const int r0 = row0 % N;
  const uint slot = slots[e];
  constexpr int RB = (DIM * BITS) / 8;                // packed bytes per row
  constexpr int G = DIM / 64;                          // groups per row
  const device uint8_t* w1p = (const device uint8_t*)w1q + ((size_t)slot * N + r0) * RB;
  const device uint8_t* w3p = (const device uint8_t*)w3q + ((size_t)slot * N + r0) * RB;
  const device bfloat16_t* s1p = w1s + ((size_t)slot * N + r0) * G;
  const device bfloat16_t* b1p = w1b + ((size_t)slot * N + r0) * G;
  const device bfloat16_t* s3p = w3s + ((size_t)slot * N + r0) * G;
  const device bfloat16_t* b3p = w3b + ((size_t)slot * N + r0) * G;

  float acc1[RPS] = {0}, acc3[RPS] = {0};
  for (int it = 0; it < K / 512; it++) {
    const int c = it * 512 + lane * 16;               // first of this lane's 16 values
    float xv[16];
    float amax = 0.0f;
    for (int j = 0; j < 16; j++) { xv[j] = float(x[c + j]); amax = max(amax, fabs(xv[j])); }
    amax = max(amax, simd_shuffle_xor(amax, 1));       // lanes 2i, 2i+1 share a 32-block
    const float s = act_scale(amax);
    float xs = 0.0f;
    for (int j = 0; j < 16; j++) { xv[j] = fq8(xv[j], s); xs += xv[j]; }
    const int g = c / 64;
    const int boff = c * BITS / 8;
    for (int r = 0; r < RPS; r++) {
      const float q1 = qdot8<BITS>(w1p + r * RB + boff, xv) + qdot8<BITS>(w1p + r * RB + boff + BITS, xv + 8);
      const float q3 = qdot8<BITS>(w3p + r * RB + boff, xv) + qdot8<BITS>(w3p + r * RB + boff + BITS, xv + 8);
      acc1[r] += float(s1p[r * G + g]) * q1 + float(b1p[r * G + g]) * xs;
      acc3[r] += float(s3p[r * G + g]) * q3 + float(b3p[r * G + g]) * xs;
    }
  }
  for (int r = 0; r < RPS; r++) {
    float gt = simd_sum(acc1[r]);
    float up = simd_sum(acc3[r]);
    if (lane == 0) {
      gt = float(bfloat16_t(gt));                      // gather_qmm output dtype
      up = float(bfloat16_t(up));
      const float lim = limit[0];
      if (lim > 0.0f) { up = clamp(up, -lim, lim); gt = min(gt, lim); }
      const float hv = wts[e] * (gt * sigmoid_mlx(gt) * up);
      h[e * N + r0 + r] = bfloat16_t(hv);
    }
  }
"""

# -------------------------------------------------------------------------------------- MoE down
# grid: (32 * SG, DIM / (RPS * SG), 1); each simdgroup RPS output rows, all experts
_MOE_DOWN = r"""
  constexpr int RPS = 4;
  constexpr int SG = 2;
  const int K = INTER;        // 2304
  const int N = DIM;          // 5120
  const int lane = thread_index_in_simdgroup;
  const int sg = simdgroup_index_in_threadgroup;
  const int r0 = (threadgroup_position_in_grid.y * SG + sg) * RPS;
  constexpr int RB = (INTER * BITS) / 8;
  constexpr int G = INTER / 64;
  float out[RPS] = {0};
  for (int e = 0; e < NE; e++) {
    const uint slot = slots[e];
    const device uint8_t* wp = (const device uint8_t*)w2q + ((size_t)slot * N + r0) * RB;
    const device bfloat16_t* sp = w2s + ((size_t)slot * N + r0) * G;
    const device bfloat16_t* bp = w2b + ((size_t)slot * N + r0) * G;
    float acc[RPS] = {0};
    for (int it = 0; it < K / 256; it++) {
      const int c = it * 256 + lane * 8;
      float xv[8];
      float amax = 0.0f;
      for (int j = 0; j < 8; j++) { xv[j] = float(h[e * K + c + j]); amax = max(amax, fabs(xv[j])); }
      amax = max(amax, simd_shuffle_xor(amax, 1));
      amax = max(amax, simd_shuffle_xor(amax, 2));     // 4 lanes per 32-block
      const float s = act_scale(amax);
      float xs = 0.0f;
      for (int j = 0; j < 8; j++) { xv[j] = fq8(xv[j], s); xs += xv[j]; }
      const int g = c / 64;
      const int boff = c * BITS / 8;
      for (int r = 0; r < RPS; r++) {
        acc[r] += float(sp[r * G + g]) * qdot8<BITS>(wp + r * RB + boff, xv) + float(bp[r * G + g]) * xs;
      }
    }
    for (int r = 0; r < RPS; r++) {
      float v = simd_sum(acc[r]);
      out[r] += float(bfloat16_t(v));                  // each expert's output rounds to bf16
    }
  }
  if (lane == 0) {
    for (int r = 0; r < RPS; r++) y[r0 + r] = shared[r0 + r] + out[r];
  }
"""


@lru_cache(maxsize=None)
def _moe_up_kernel():
    return mx.fast.metal_kernel(name="mk_moe_up", input_names=["x", "slots", "wts", "limit", "w1q", "w1s", "w1b", "w3q", "w3s", "w3b"],
                                output_names=["h"], header=HEADER, source=_MOE_UP)


@lru_cache(maxsize=None)
def _moe_down_kernel():
    return mx.fast.metal_kernel(name="mk_moe_down", input_names=["h", "slots", "shared", "w2q", "w2s", "w2b"],
                                output_names=["y"], header=HEADER, source=_MOE_DOWN)


def moe_experts(x: mx.array, slots: mx.array, wts: mx.array, limit: mx.array, p1, p3, p2, shared: mx.array,
                dim: int = 5120, inter: int = 2304) -> mx.array:
    """Routed experts of one decode token over slab slots, plus the shared expert's output.
    x: [dim] bf16 MoE input (pre act-quant); slots: [k] uint32; wts: [k] fp32 routing weights;
    p1/p3/p2: (packed, scales_bf16, biases_bf16, bits) slab projections; shared: [dim] fp32.
    Returns y [dim] fp32 = shared + sum_e expert_e(x)."""
    ne = slots.shape[0]
    b13, b2 = p1[3], p2[3]
    assert p3[3] == b13
    h = _moe_up_kernel()(
        inputs=[x.reshape(-1), slots, wts.reshape(-1).astype(mx.float32), limit.reshape(1), p1[0], p1[1], p1[2], p3[0], p3[1], p3[2]],
        template=[("BITS", b13), ("DIM", dim), ("INTER", inter)],
        grid=(64, ne * inter // 8, 1), threadgroup=(64, 1, 1),
        output_shapes=[(ne * inter,)], output_dtypes=[mx.bfloat16],
    )[0]
    y = _moe_down_kernel()(
        inputs=[h, slots, shared.reshape(-1).astype(mx.float32), p2[0], p2[1], p2[2]],
        template=[("BITS", b2), ("DIM", dim), ("INTER", inter), ("NE", ne)],
        grid=(64, dim // 8, 1), threadgroup=(64, 1, 1),
        output_shapes=[(dim,)], output_dtypes=[mx.float32],
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

HC_TPG = 64  # threads (= d values) per hc_partial threadgroup


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


def hc_stage(x_res: mx.array, premix: mx.array, fn: mx.array, sc: mx.array, base: mx.array, norm_w: mx.array,
             eps: mx.array, hin: mx.array | None = None, post: mx.array | None = None, comb: mx.array | None = None):
    """One mHC stage for a single token.
    x_res [4, D] bf16 residual streams; if hin is given, first X = hc_post(hin, x_res, post, comb).
    -> (X [4, D] bf16, h_norm [D] bf16, pre [4], post [4], comb [4, 4]) with h = sum_j premix[j] X[j]
    normalized by norm_w, and (pre, post, comb) the Sinkhorn mixes of X."""
    D = x_res.shape[-1]
    nt = D // HC_TPG
    has_post = hin is not None
    if not has_post:
        hin, post, comb = _dummies(D)
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
