# op33 — GVR execution-perf (P5) + warp-tie-select (P3) campaign

Started 2026-07-13, node umbriel-b200-092, branch omni/op21-gvr-prod @a521404767.

## Mandate (user)
Starting points = **op26 R0 (`op26_r0auto`, auto 1CTA/MC dispatch)** and
**op21 ms_auto (`gvr_ms_auto`, HLS/op27 tail ladder)**. KEEP the GVR threshold
skeleton (P1 hint → count/secant → collect → P4). Optimize the *algorithm* and
*code-execution* performance. Test conditions IDENTICAL to op22
(`op22_temporal_fixed_hr_bench`): synth bundles `synth_data.get_bundle`
(byte-identical, cell_seed=42+crc32), nsys cold-L2 ×3 canonical. Use 8 GPUs.

## Why this is NOT a re-tread (falsification-history check)
- op29 HBE **SHIPPED** the boundary-exact restructure (N≥65536, ≥sglang_v2). ✅ done.
- op31 HBE-C cluster single-pass **CLOSED** (win only N≥524288, outside envelope). ✗
- op32 short-row register-resident (INSIGHTS-P0) **CLOSED NO-SHIP** — short-N
  fp32 BS=1 is latency/issue-bound structural wall; secant is load-bearing.
- **Untouched, skeleton-preserving, in-envelope**:
  - **P5 execution engineering** (op29 RESUME lists "短行 P5" pending): NCU shows
    fused pass issue-bound 81-84%, reg=44-61 pressing occupancy→50%. Levers:
    `__launch_bounds__(threads,minBlocks)` occupancy, Blackwell 256-bit (32B)
    vectorized coalesced loads on count/collect, runtime-K template collapse
    (4 multi-K specs each spilling reg), CUDA13 `enable_smem_spilling`. Applies
    across ALL N to BOTH incumbents. Literal match to "优化代码执行性能".
  - **P3 warp-register tie-select + shrink over-collect** (falsi-history live
    lever "shrink cand_count, P3 over-collects 3.96×K@K512"): warp ballot rank
    for cand≤128 (zero block barrier); tighter kFTarget → smaller P4. Targets
    mid-N/K2048 where P4/cand dominate. synth UNDER-represents tie work
    (synth_vs_real: real tail heavier) ⇒ synth win is a conservative floor.

## Protocol (omni-kernel v2, crux-first)
- **iter0 CRUX (cheap, no kernel)**: NCU on `gvr_ms_auto` + `op26_r0auto` at
  representative cells (fp32/bf16 × K512/1024/2048 × N{16K,65K,262K} × BS{1,64}).
  Gate metrics: `dram__throughput.pct`, `sm__sass_average_data_bytes_per_sector`
  (32B vec headroom?), `launch__registers_per_thread`, `sm__warps_active.pct`
  (achieved occupancy), `smsp__issue_active.pct`. Decide P5 headroom per regime
  BEFORE writing any kernel. If a lever shows no headroom in a regime → skip it there.
- Then per surviving lever: kernel (subclass/gated flag, zero vendored edits,
  baseline always regressible) → **exactness gate** (vdiff=0 sorted value-multiset
  + uniq==K, `get_bundle` data, adversarial + real-capture tracks) → cold-L2 pilot
  → **nsys ×3 median + anchor** verdict (REJECT/FALSIFIED/WASH/SHIP).
- 8-GPU fan-out: shard the (scenario×K×dtype×N×BS) grid by GPU; setsid single-line
  launches; pkill-triple to stop; anchor cell re-measured per batch (±3%).

## Ship rule (mirror op29)
New arm ≥ incumbent (op26_r0auto ∪ gvr_ms_auto, whichever is the per-cell best)
in every 9/9 (scenario×K) slice geomean; zero cell loses >5%; exactness 100%.

## Ledger
- iter0 CRUX DONE (NCU): P5 exec-throughput WALLED at BS=1 (idle DRAM + grid<<SM,
  W1) and up to BS~148 (sub-wave grid, W2). Live sliver = BS>=256 large-N cluster
  occupancy lift (sglang launch_bounds(1024,2) recipe). See ITERATIONS.md/WALLS.md.
