# op15 SMEM-resident — cross-iteration learnings

## Architecture notes (B300 sm_100)
- opt-in dynamic SMEM = 232448 B/block. Native-dtype row copy fits: fp32 N<=~37632, bf16/fp16 N<=~75264 (after ~80KB reserve for keys/vals/hist/scratch).
- cuteDSL: pass an smem tensor as the logits arg + flip make_ptr AddressSpace gmem->smem; `_load_fp32(t,i)=t[i]` works for smem scalar reads; vectorized `cute.copy(atom, smem_src, frag)` lowers to ld.shared.

## Effective techniques
- **Vectorize the staging copy.** iter2 scalar staging = +11% cold; iter3 vectorized
  staging (gmem→fragment→smem, same copy_atom as block_count_ge) = +2% cold. The scalar
  staging pass was the dominant penalty, NOT the smem reads.
- **warm-L2 A/B is the decisive isolator.** A memory-traffic optimization that is also
  slower warm (data hot in L2) cannot win — the traffic it targets is already free, so
  reject without chasing further impl variants. (smem-resident warm = +3% on B300.)

## Ineffective / falsified directions
- **SMEM-resident GVR (op15) — FALSIFIED on B300.** Best-effort (vectorized staging) =
  +2% cold / +3% warm slower across the small-N envelope; wins only at N=4096 (~3-4%);
  worst at K=2048 / N≥32768. Exact (vdiff=0) but no ship.
- Prior op8_gvr_turbo: smem-resident "win" at N=8192 was a cold-L2/launch artifact; nsys kernel-time = base; slower N>=65K. Reason: the one-time row load is itself a full gmem pass; at small N the re-read passes are L2 hits already (warm-L2 caps upside ~20%); BS=1 bottleneck is structural single-CTA ~24% occupancy, not L2-read traffic.

## Current best
- **baseline rank-scatter (op#7) — SMEM-resident does not beat it on B300.** Converged
  on a physical floor (warm-L2 also slower ⇒ no traffic to save). The genuine small-N
  lever is structural (intra-CTA warp pipelining to lift ~24% occupancy), not residency.