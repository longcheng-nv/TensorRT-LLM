# GVR Top-K, BS 1-1024 — make the ORIGINAL GVR design win the whole batch envelope (+40% per K-group)

## Problem

Batched sparse-attention indexer top-K at decode. `logits[b, npad]` fp32 (REAL
captured rows; every batch row is a materialized copy of the same captured
row — distinct memory), valid length `n_valid`, tail padded -FLT_MAX.
`pre_idx[b, k]` int32 = previous step's top-k per row. Return `indices[b, k]`:
per-row EXACT top-k index set — the selected VALUES must equal torch.topk's
values exactly (index order free; k-th-value ties may resolve either way).
Entry (DPS, torch binding): run(logits, pre_idx, n_valid, indices).
Axes: b in 1..1024, k in {512,1024,2048}, n to ~1.05M (npad=ceil(n/64)*64).

## Acceptance (external, nsys cold-L2 pure kernel time on B200, full real envelope)

Two independent bars, BOTH must hold on the full grid (BS=1: all 865 layer
cells; BS 2-1024: the 75-cell x 10-BS real grid; baseline = TensorRT-LLM
PR#16457 GVR head — native batch at b>1):
  1) k=512 & k=1024 cases pooled (999 cases): geomean >= 1.40x AND every case >= 0.95x;
  2) k=2048 cases (616 cases): geomean >= 1.40x AND every case >= 0.95x.
Exactness everywhere is a precondition, not a bar.
Platform workloads are a 45-case stratified subset with a fixed per-eval
overhead (negligible at b>=32); the external grid is what counts.

## HARD skeleton rule — the ORIGINAL GVR design, no escape hatches

Every tier you build must be a GVR implementation per row:
 (a) use `pre_idx` (previous top-k) as the PRIOR to seed threshold guesses;
 (b) refine the threshold with a secant + log-transform solve (equivalent
     threshold-refinement structures allowed);
 (c) exact refine of surviving candidates.
P1 (prior) and P4 (refine) may be restructured freely as long as they stay
equivalent in role; mature primitives (radix digit passes, histogram ladders)
may be ABSORBED INSIDE a phase. What is BANNED: any tier that is a standalone
prior-free selection kernel (plain radix-select/sort/sampling select), and any
per-case dispatch to non-GVR top-k operators. Where the solve trivially
converges (candidate capacity >= npad) the threshold-solve structure must
still be the frame it degenerates from. No dispatch on hint quality
(unknowable at inference); in-kernel admission escape is fine.


## Cross-row reuse — the EXACT compliance line (read before writing any code)

The harness materializes identical batch rows. That is a property of the TEST
HARNESS, not the problem: your kernel must be a per-row GVR that would be
correct AND fast on ARBITRARY per-row data, and the external verdict treats it
that way.
- ALLOWED cross-row amortization: P1 ONLY — build the prior / threshold ladder
  once from `pre_idx` + gathered hint values and broadcast it to all rows
  (with per-row escape if a row's solve fails under the shared seed). The hint
  tensor is identical per row too; exploiting THAT is legitimate.
- REQUIRED per row: threshold verification / secant refinement against the
  ROW'S OWN data, candidate collect from the row's own data, exact refine.
  Every row's logits must actually be read and selected from.
- AUTO-DQ regardless of internal score: computing top-k for one row and
  broadcasting/copying the RESULT to other rows (with or without an equality
  verification pass); any row-equality/memcmp test used to skip per-row
  selection; prior-free radix-select/sort cores.

## Face the prior head-on (history you must beat, not dodge)

Previous campaigns repeatedly measured the pre_idx prior as UNPROFITABLE and
dropped it (three independent falsifications on round-1 kernels; an exact
warm-hint admission filter measured 1.0001x in its own activation zone; the
relaxed-skeleton lineage shipped hint-free). That lineage is NOT your starting
point and its conclusion is NOT accepted here. Your task is to make the prior
PAY within the GVR frame. Measured openings:
 - The production head itself profits from the prior (its P1 seeds cut secant
   passes; low-hint cells cost it ~+15-40% — the prior carries real signal).
 - At b>1 all rows share the hint: P1 gather + ladder construction can be
   computed ONCE per batch and broadcast (a pure amortization win no
   prior-free kernel can copy). Correctness must NOT assume rows are equal —
   amortize speculatively, verify per row cheaply, fall back per row.
 - Register/smem-resident multi-threshold count passes make extra secant
   iterations nearly free (BS=1 champion proof: 1.6531x vs head, 865 cells,
   skeleton fully compliant).

## Where the current best batched GVR (your seed, digest below) loses — measured zones

vs head native batch, 750-case BS2-1024 grid (this kernel is exact everywhere):
 - occupancy VALLEY, bs16-128 x n 16k-65k: gm 0.740, min 0.374. 1 CTA/row
   under-fills 148 SMs; multi-CTA/row pays cluster sync. Fix shape: rows-per-SM
   packing / persistent CTAs consuming (row, slice) work items; shared prior.
 - bandwidth zone, bs>=256 x n>=128k (and bs>=128 x n>=131k): gm 0.765-0.805.
   Head streams at ~2.4 TB/s effective; exact top-k must read everything once,
   so wins here come ONLY from fewer passes (better prior => fewer secant
   passes => less re-traffic) and perfect coalescing. Ceiling ~1.2-1.4x.
 - WIN zones to keep: npad<=12288 any b (gm 1.234, direct-form tier at high
   occupancy); b<=8 (gm 1.072 latency regime, register-resident).
 - BS=1 must hold the champion level (~1.65x): at b==1 nothing stops you from
   reusing the BS=1 champion structure (fully compliant GVR).

## Dead ends already measured (do not re-walk)

1. Prior-free tiers: banned AND falsified as a perf source at b=1.
2. grid.y naive batching of the register-resident bs1 kernel: occupancy-locked
   (1 CTA/SM from register pressure; 16 CTA/row saturates at b>=10).
3. cp.async pipelining of scan loops (barrier > latency hidden).
4. Private per-warp histograms (SM100 same-address atomics fine).
5. Whole-row smem staging before scans at b=1 (L2 re-reads cheap); at high b
   smem row caches DO pay when reused across passes — measure.
6. Cluster DSMEM count exchange without parity banks / proper fences: the
   row0-corruption race cost a whole prior campaign round. SELF-TEST any
   cluster tier on bs {2,3,4,8,16,64,96} x npad {16k,32k,64k,128k,256k} x
   k {512,1024,2048} with random data + tie-heavy adversarial rows before
   submitting.
7. CUDA graphs / framework kernel imports: banned by the judge.

## Correctness traps

- k-th tie boundary: never drop a strictly-greater element; ties fill rest.
- Real rows are undershoot-biased for hint-seeded thresholds.
- Batched twist: per-row output slot counters must be row-local; neighbor-row
  writes are the classic silent bug.
- Values equality vs torch.topk is checked per row on every workload.

## Seed digest — current batched champion dispatch (exact everywhere; improve on it)

Tiers: direct whole-row (npad<=12288, grid.x=b, per-row single CTA, fused
level-0 hist + 11/11/10 radix inside the degenerate solve frame);
register-resident per-row cluster tiers (cs=min(champ(npad), pow2(256/b)),
MAXV by chunk; exact sampled prior quantile ladders; RFAST warp-only
order-stat prior on some mid tiers); two-rung streaming ladders for
throughput tiers; ~20 measured dispatch rules on (b, npad, k) only.
Known result: gm 0.9862 overall vs head at b>1 — the valley and bandwidth
zones above are where it dies. BS=1 champion structure: P1 hint-CCDF two-level
64-bin histogram -> 8 quantile rungs; P2 one-pass 8-threshold count +
log-secant bracket /9 per pass (<=8) + plateau exact fallback; P3 prefix-sum
collect via parity-banked DSMEM; P4 CTA0 4x8-bit radix + tie-ticket writeback;
dispatch direct<=12288 / reg-resident<=262144 (1/4/8/16 CTA) / streaming.
