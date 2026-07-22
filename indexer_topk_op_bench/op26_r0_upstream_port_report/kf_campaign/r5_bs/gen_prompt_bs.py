# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""R5 BS-campaign prompt: instructions + r3_v11 source digest (<=32KB total)."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = (HERE.parent / "harvest" / "r3_7d8272b7" / "gvr.cu").read_text().splitlines()

BODY = r"""# DeepSeek Indexer Top-K Decode, BATCHED (BS 1-1024, fp32, B200) — Scale the GVR Champion Across Batch

## Problem

Batched sparse-attention indexer top-K at decode time. `logits[b, npad]` fp32
(REAL captured indexer rows; every batch row is a materialized copy of the
same captured row — distinct memory, so no L2-aliasing shortcuts), valid
length `n_valid`, tail padded with -FLT_MAX. `pre_idx[b, k]` int32 is the
previous step's top-k per row (again identical copies). Return `indices[b, k]`
int32: per-row exact top-k index set (tie-robust checker per row). The extra
scalar input `cell_id` identifies the source capture for the input generator —
your kernel MUST accept it and ignore it:
  run(logits, pre_idx, n_valid, cell_id, indices)   # DPS, torch binding

Axes: b in {1..1024} (workloads sample 1/4/32/256/1024), k in {512,1024,2048},
n up to ~1.05M (`npad = ceil(n/64)*64`). Real heavy-tailed logits; hint
overlap 0.27-1.0. Do not overfit exact axis values.

## Baseline & starting point

Per-workload baseline timings are EXTERNAL nsys cold-L2 kernel times of the
PRODUCTION kernel (TensorRT-LLM PR#16457 GVR, CuTe DSL) run NATIVELY BATCHED
([b, npad] in one launch) on an idle B200. Beat THAT. The external acceptance
bars on the full 750-case real grid (75 cells x 10 BS points):
  1) average (geomean) speedup vs the production batched kernel >= 2.0x;
  2) EVERY case >= 1.0x (no regression anywhere), and the campaign objective
     beyond the bars is to MAXIMIZE THE MINIMUM per-case speedup;
  3) BS=1 must retain the existing BS=1 champion's performance (the champion
     source digest below IS the starting point — at BS=1 it already runs
     1.63x geomean over production on 865 cells; do not give that back).

Platform timings carry a fixed per-eval overhead; at BS>=32 it is negligible.
Track relative progress at small BS.

## Required algorithmic skeleton — HARD compliance rule (unchanged)

Keep the GVR skeleton per row: (a) `pre_idx` as threshold prior, (b) a
secant/log-style exact threshold solve (equivalent refinement structures OK),
(c) exact refine of surviving candidates. The npad<=12288 direct path (the
analytic trivial-convergence limit of the threshold solve) is compliant.
Wholesale prior-free replacements (plain radix-select/sort over rows) are
NON-COMPLIANT even if faster. No dispatch on hint quality (unknowable);
in-kernel admission escape is fine. CUDA graphs / framework kernels banned.

## What is already known about batch scaling THIS kernel (measured)

- The BS=1 champion below is a LATENCY design: 16-CTA cluster + full
  register residency per row. Naively batching it with grid.y=b is
  OCCUPANCY-LOCKED: high register counts pin it at 1 CTA/SM, and 16 CTAs
  PER ROW saturates 148 SMs at b>=10. At large b the problem flips from
  latency-bound to THROUGHPUT-bound: per-row resources must shrink
  (fewer CTAs/row, fewer registers, smaller rung ladders) as b grows.
- Measured on this data: a streaming 1-CTA/row variant with a reduced
  4-rung ladder + 2-deep hint seeding unlocked large-N high-BS
  (flash 512k BS=1024 ~1.9x vs production). That is a floor, not a target.
- The production baseline ITSELF batches natively and amortizes well —
  your denominator is strongest exactly where the champion is weakest
  (mid/high BS). Expect the required curve shape: keep the BS=1 wins,
  then win throughput at BS>=32 via occupancy, not per-row latency.
- Per-row work is IDENTICAL across rows (same data): divergence between
  rows is zero; per-row adaptive passes still differ only via atomics
  noise. Do not rely on that (correctness must hold for arbitrary rows),
  but you may exploit uniformity for performance (e.g. shared threshold
  seeding across rows is legitimate ONLY as a performance hint if
  correctness never depends on rows being equal).

## Dead ends already measured at BS=1 (do not re-discover; most still apply per row)

1. Prior-free radix/sort rewrites: banned and ~10x slow.
2. Private per-warp histograms (SM100 same-address atomics are fine).
3. Per-element ballot/popc fused count+collect: costs a full pass.
4. Staging whole row into smem before scanning: L2 re-reads are cheap.
5. >2 extra secant rounds; keep passes bounded.
6. cp.async pipelining on the scan loop (barrier overhead > latency hidden).
7. Launch-config-only retuning of the production kernel: ceiling 1.025x.
8. CUDA graphs / replay amortization: banned by the judge.

## Correctness traps (unchanged from BS=1 campaign)

- Tie boundary: never drop a strictly-greater element; ties fill remainder.
- Real data is undershoot-biased for hint-seeded thresholds.
- Cluster DSMEM: `arrive_relaxed` has no release semantics — parity-bank
  or fence your count exchanges.
- Batched twist: out-of-bounds writes to a NEIGHBOR ROW's output slice are
  the new silent killer; per-row slot counters must be row-local.

## Requirements

- CUDA C++ (sm_100a). fp32 in, int32 out, DPS, torch binding.
- Entry signature EXACTLY: run(logits, pre_idx, n_valid, cell_id, indices).
- Exact per-row (tie-robust), every run, all b in 1..1024.
- One launch preferred (or few); at b=1024 launch count matters less than
  occupancy, at b=1 it is 3-29us of the budget — dispatch on (b, npad, k)
  (all known at launch) is expected.
"""


def main():
    lines = SRC
    # digest spans: header comment+constants; register-resident section header
    # + count_reg; dispatch tail (launchers + gvr_topk_launch)
    head = "\n".join(lines[0:54])
    reg_hdr = "\n".join(lines[639:700])
    tail_start = next(i for i, l in enumerate(lines)
                      if l.startswith("static void launch_direct")) - 2
    tail = "\n".join(lines[tail_start:])
    digest = (
        "\n---\n\n## APPENDIX — BS=1 champion source digest (verbatim spans; "
        "this is the kernel whose BS=1 performance you must retain)\n\n"
        "Full file is 1344 lines; spans below give the constants, the "
        "register-resident core, and the complete launch/dispatch layer. "
        "Reconstruct the rest from structure (phases mirror the production "
        "GVR: P1 hint-CCDF rung ladder, P2 multi-threshold count + log-secant, "
        "P3 DSMEM collect, P4 radix refine + tie-ticket writeback; plus the "
        "direct exact path for npad<=12288).\n\n```cuda\n"
        + head + "\n\n// …(smem struct, count/exchange/max_below/phase1/"
        "gvr_topk_kernel: as in the production-mirroring structure)…\n\n"
        + reg_hdr + "\n\n// …(gvr_topk_reg body, direct path, P4 helpers)…\n\n"
        + tail + "\n```\n")
    out = BODY + digest
    (HERE / "prompt_bs.md").write_text(out)
    print(f"prompt_bs.md {len(out)} bytes (limit 32768)")
    assert len(out) <= 32768


if __name__ == "__main__":
    main()
