# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op40 arm registry: arm name -> (package module name, ctor flag dict).

`base` is the byte-frozen PR#16457 @e612fc2f38 vendored package. Variants are
added as new packages under src/ (never by editing gvrpkg40b) and registered
here. Import side effect free; resolution happens in the harness.
"""

ARMS = {
    # name: (module, ctor_flags)
    "base": ("gvrpkg40b.top_k.gvr_topk_decode", {}),
    # iter1: op37 d2a/d2b/d1a P4 levers re-spliced onto e612 (non-KF primitives)
    "v1": ("gvrpkg40v1.top_k.gvr_topk_decode",
           dict(p4_rs_rw_search=True, p4_fine_skip=True, p4_peer_push=True)),
    # iter2: v1 + per-K R0 ladder recalibration (H8a host-replay winners:
    # hit% 89/65/21 -> 97/81/54, cand/K 1.90/1.56/1.33 -> 1.48/1.39/1.21)
    "v2lad": ("gvrpkg40v1.top_k.gvr_topk_decode",
              dict(p4_rs_rw_search=True, p4_fine_skip=True, p4_peer_push=True,
                   per_k={512: dict(r0_qfracs=(0.9, 0.6, 0.35)),
                          1024: dict(r0_qfracs=(0.9, 0.6, 0.35)),
                          2048: dict(r0_qfracs=(0.8, 0.6, 0.4, 0.25))})),
    # iter2b: v2lad + p1b_cache for fp32 K512/K1024 (kills the 2nd K-gather;
    # smem_gath +2/4KB; K2048 excluded pending occupancy check)
    "v2c": ("gvrpkg40v1.top_k.gvr_topk_decode",
            dict(p4_rs_rw_search=True, p4_fine_skip=True, p4_peer_push=True,
                 per_k={512: dict(r0_qfracs=(0.9, 0.6, 0.35), p1b_cache=True),
                        1024: dict(r0_qfracs=(0.9, 0.6, 0.35), p1b_cache=True),
                        2048: dict(r0_qfracs=(0.8, 0.6, 0.4, 0.25))})),
    # iter3: distributed MSB-radix-select P4 (v3 pkg), both cs paths, on top
    # of the v1 flag set (d2a/d2b/d1a inert where radix replaces them)
    "v3": ("gvrpkg40v3.top_k.gvr_topk_decode",
           dict(p4_rs_rw_search=True, p4_fine_skip=True, p4_peer_push=True,
                p4_radix_dist=True, p4_radix_cs1=True)),
    # iter3 composite: v3 radix P4 + the harvested K2048-only ladder
    "v3k": ("gvrpkg40v3.top_k.gvr_topk_decode",
            dict(p4_rs_rw_search=True, p4_fine_skip=True, p4_peer_push=True,
                 p4_radix_dist=True, p4_radix_cs1=True,
                 per_k={2048: dict(r0_qfracs=(0.8, 0.6, 0.4, 0.25))})),
    # iter5: v3k + exact full-row radix fallback on P2 fail-soft (fixes the
    # plateau/neartie baseline defect class)
    "v4": ("gvrpkg40v3.top_k.gvr_topk_decode",
           dict(p4_rs_rw_search=True, p4_fine_skip=True, p4_peer_push=True,
                p4_radix_dist=True, p4_radix_cs1=True, p2_radix_fallback=True,
                per_k={2048: dict(r0_qfracs=(0.8, 0.6, 0.4, 0.25))})),
    # iter6: ship-integrity composite — v1 levers + K2048 ladder + exact
    # fallback; radix PERF paths OFF (falsified iter3)
    "v5best": ("gvrpkg40v3.top_k.gvr_topk_decode",
               dict(p4_rs_rw_search=True, p4_fine_skip=True, p4_peer_push=True,
                    p2_radix_fallback=True,
                    per_k={2048: dict(r0_qfracs=(0.8, 0.6, 0.4, 0.25))})),
    # iter7 probe: T2 scan ILP via mt_unroll=8 (config-only)
    "v5mt8": ("gvrpkg40v3.top_k.gvr_topk_decode",
              dict(p4_rs_rw_search=True, p4_fine_skip=True, p4_peer_push=True,
                   p2_radix_fallback=True, mt_unroll=8,
                   per_k={2048: dict(r0_qfracs=(0.8, 0.6, 0.4, 0.25))})),
    # iter8: v5best + P3-fused coarse histogram (P4 skips minmax+build, cs1)
    "v6": ("gvrpkg40v3.top_k.gvr_topk_decode",
           dict(p4_rs_rw_search=True, p4_fine_skip=True, p4_peer_push=True,
                p2_radix_fallback=True, p3_hist_fuse=True,
                per_k={2048: dict(r0_qfracs=(0.8, 0.6, 0.4, 0.25))})),
}


def resolve(name):
    mod_name, flags = ARMS[name]
    import importlib
    mod = importlib.import_module(mod_name)
    return mod.GvrTopKKernel, dict(flags)
