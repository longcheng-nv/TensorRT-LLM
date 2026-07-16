# op35 iteration log
(entries appended per omni-kernel protocol; verdicts ∈ SHIP/FALSIFIED/WASH/PIVOT)

## iter0 — 2026-07-16 — PROBE (rung 1+2)
Hypothesis: B1 block-skip viable if (a) tile coverage sparse on real data, (b) P3 scan
is a material kernel share. Ledger check: none (new granularity; ≠op16 value-peel, ≠ms fusion).
Probe A (host replay, 77 cells, results/replay_b1.csv):
  - (warp,window)=256-elem quantum is the right skip granularity:
    P3-scan-work saved ceiling: synth N≥64K med 79%, real N≥64K med 56%, small-N 16-19%.
  - whole-window (8K) granularity (proposal's literal B1): near-zero at real cells → dead.
  - warp early-exit (zero-sideband): 51%/24% — weaker than sideband but ~free.
Probe B (p3_oracle_frac=0.001 ablation, 39 N≥65536 cells, 8-GPU shard, CUDA-event):
  - event axis carries ~40µs host overhead → use DIFFERENCE (base-var) vs nsys pr time.
  - early shards: P3 scan share ≈ 0-25% at cluster cells (slice/CTA only 16-32K elems
    → P2/P3 scans are ~2-4 iters; kernel dominated by fixed phase-chain + gather + handoff).
Design settled (if GO): fp32 wwmax[warp][win] sideband in P2 (1 FMAX/elem) → P3 per-warp
ballot bitmap → bit-loop; EXACT for any thr (fallback recounts B2 ride free); gate cs≥4.

## iter1 — 2026-07-16 — FALSIFIED (H3-tail K2048 qfracs)
Hypothesis: op27 tail ladder (0.75,0.45,0.048) fixes K2048 worst tail (op27: 1.15→1.44×).
Probe: config-only A/B, 23 K2048 cells (synth best+worst all N + v32 real), 4 GPUs, event paired.
Result: geomean b/v 0.948-0.973 (variant SLOWER everywhere incl worst axis + real v32); exact 23/23.
Diagnosis: PR admission = R0+vseed; the vseed pmean rung already covers the tail interior that
op27's 0.45/0.048 columns bought in the HLS kernel; extra columns = pure M-ary P2 count tax.
Phase-level conclusions do not transfer across kernel families (omni-kernel rule confirmed).
Ledger write-back: FALSIFIED domain = qfracs tail-ladder on PR-R0+vseed, K2048, fp32 BS=1, event-paired.

## iter0-final — 2026-07-16 — ATTRIBUTION (nsys 4-arm oracle, 77 cells)
nsys decomposition (results/nsys_oracle_decomp.csv): P4blk (handoff2+P4+writeback)
= 23-58%, median ~37% — DOMINANT everywhere (small+large N, synth+real).
mid (P1b+P2+falsi+handoff1) 17-48%; floor 13-40%; P3 0-26% (only 512K-1M big).
UB(zero P4blk)=1.578; UB(zero P3+P4blk)=1.771. Campaign pivot: P4blk is the battleground;
the proposal's scan-side levers (B1/B2/HLS) cap at ~1.07-1.15 — cannot reach +40% alone.

## iter-L4 — 2026-07-16 — WASH (launch-config refinement)
cs2/cs4/cs8/nt512 overrides vs pick_config on 39 N>=64K cells: aggregate 0.89-0.99;
per-cell best-of-5 oracle ceiling (kernel-level, dilution-corrected) = 1.025 — noise-level.
pick_config confirmed near-optimal. Only residual micro-rule: cs8->cs4 boundary at
N=131072 (4-9% on 6 cells) — candidate for nsys confirm later.

## iter2 plan (from attribution)
iter2a p4_fused_hist (cs=1): hist built during P3 stream-write (bmin=thr, bmax=pmax
  snapshot, clamped) -> P4 skips minmax pass + zero + build (~3 barriers + 2 cand passes).
iter2b distP4 (cs>1): kill handoff2 value-gather; peers keep local cands; leader does
  scalar searches on DSMEM-merged hist; all CTAs scatter own cands via DSMEM atomic ranks.
iter2c kNumBins diet: screens running (256 needs scratch relocation >=272 if it wins).
