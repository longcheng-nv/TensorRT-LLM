# FALSIFIED.md — op42 ledger
# Scoped triples: (conclusion, domain, evidence) + root-cause class.
# Seeded ONLY from sanctioned materials (user-cited op37 BS-decay analysis).

- (Sequential per-row full-GPU launches lose to head batched arm from BS=2,
  domain: all K/N, BS>=2, evidence: nsys op37 198/198) — structural-wall:
  1 cluster <= 10.8% SMs, same-stream serialization. Revival: none — this IS
  the wall this campaign attacks via kernel-side row parallelism.
- (Naive Python multi-stream launching is SLOWER than sequential, domain:
  BS>1 host-loop, evidence: nsys op37) — structural-wall: host issue rate ~
  kernel duration 8-12us. Revival: C++ launcher or single batched launch.
- (CUDA-graph fork-join proves rows are HW-concurrent; BS64 8.4x recovery but
  ceiling ~148/CS co-resident clusters, domain: BS<=8 narrow win, evidence:
  nsys op37 graph_probe) — mechanism note, not a falsification: batched
  single-launch dominates graph fork-join; graphs are a fallback lever only.
- (Event-axis A/B vs cuteDSL head arm is INVALID on b200-073: head host issue
  latency up to 1.2ms at BS<128 mCTA variants dwarfs 24us kernel, domain: all
  head-arm ratios, evidence: nsys diag_head_bs1 + 2-GPU repro) —
  measurement-artifact. Revival: none; nsys pure-kernel is the only A/B axis.

## i10-C' sample-prefetch fusion (2026-07-25) — FALSIFIED
Claim: prefetching P2a sample float4s before P1 hides their DRAM latency
behind the hint-gather chain (save ~1.5-2us/tp row).
Result: smoke NET LOSS — BS256+ +4-12% (flash_64k 22.2->25.0, pro_64k BS512
40.9->43.7), BS16-64 flat. Mechanism: +8 regs (2 float4 slots) at
__launch_bounds__(TB,2)'s 64-reg cap -> spills; sfuse runtime branch adds
icache on a 22%-icache-stalled kernel. Same failure class as i9-B4.
Domain: any reg-resident prefetch added to gvr_topk_tp under TB512/LB2.
Evidence: i10cp smoke 6 cells vs E-baked baseline, results in ITERATIONS.

## i10-G P4 round-0 elimination via push-time f2u+hist (2026-07-25) — FALSIFIED
Claim: storing f2u keys + accumulating the 256-bin radix hist at push time
lets P4 skip its round-0 stream (C entries + rewrite + atomics), ~1-1.5us.
Result: NET LOSS. CS>1 cells +13-15% (v32_128k BS16/32): hist atomic per
candidate becomes a REMOTE DSMEM atomic to CTA0 inside the fused hot loop.
CS1 cells +2-5%: atomic dependency chain in the streaming loop costs more
than the ~C-entry round-0 it saves (C ~2-4K << npad).
Domain: any per-push side accounting added to fused_count_collect/collect_at.
Evidence: i10g smoke 6 cells vs E baseline.

## i10-F max-shared carveout for 2 CTA/SM (2026-07-25) — FALSIFIED
BS512/1024 +8-13% WORSE with cudaSharedmemCarveoutMaxShared on gvr_topk_tp.
Co-residency of two 85KB-smem CTAs hurts this kernel (L1/smem partition
pressure); occupancy was not the binding constraint the 25% number suggested.

## i10-D pivot tgt 3K->2K (2026-07-25) — FALSIFIED OFFLINE (no silicon)
8-rung ladder too coarse: avgC only -7% (3845->3559) but +2 new restream
cells incl flash_128k_L42 (C=477 < K=512 undershoot). probe_tgt.py, 865 cells.

## i10-c capn 8192->4096 (2026-07-25) — NO EFFECT
CS8 at npad 32832: DSMEM exchange growth offsets the extra fill. Noise-flat.

## i11-P4 2048-bin radix_select_emit in tp (2026-07-25) — FALSIFIED
Reusing the direct tier's 11/11/10 radix for tp's P4: +4-10% everywhere.
Fixed costs (2048-bin zero + bin_select scan) dominate at tp's C~2-4K;
the byte-radix's early exit averages ~1.5 effective rounds. 2048-bin only
pays at direct-tier C~npad. All exact; smoke-falsified, no silicon.

## i11-R launch_bounds (TB,2)->(TB,1) (2026-07-25) — FALSIFIED, KEY PHYSICS
BS>=256 CATASTROPHIC: +37-50% (flash_128k BS512 47->70us, v32_128k BS1024
184->253us); BS128 flat. INVERTS the i10-F reading: 2-CTA/SM co-residency
IS REAL at BS>=256 (64-reg cap + 83KB smem x2 both fit) and worth 35-50% —
the SM overlaps two rows' phases there. ncu's 25% occupancy was a BS128
artifact (grid 128 < 148 SMs: no second CTA EXISTS to co-reside).
COROLLARY 1: the 64-reg cap is load-bearing -> every reg-hungry idea
(C' prefetch, row-pairing interleave) is structurally dead.
COROLLARY 2: BS>=256 weakness is pure per-row WORK (overlap already active);
BS16-128 weakness is unfixable underfill (bs*CS < 148, no co-residency
partner possible; CS2@BS128 falsified iter7).
