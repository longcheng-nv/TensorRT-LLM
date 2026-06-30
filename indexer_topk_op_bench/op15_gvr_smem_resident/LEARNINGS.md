# op15 SMEM-resident — cross-iteration learnings

## Architecture notes (B300 sm_100)
- opt-in dynamic SMEM = 232448 B/block. Native-dtype row copy fits: fp32 N<=~37632, bf16/fp16 N<=~75264 (after ~80KB reserve for keys/vals/hist/scratch).
- cuteDSL: pass an smem tensor as the logits arg + flip make_ptr AddressSpace gmem->smem; `_load_fp32(t,i)=t[i]` works for smem scalar reads; vectorized `cute.copy(atom, smem_src, frag)` lowers to ld.shared.

## Effective techniques
- (pending nsys A/B)

## Ineffective / falsified directions
- Prior op8_gvr_turbo: smem-resident "win" at N=8192 was a cold-L2/launch artifact; nsys kernel-time = base; slower N>=65K. Reason: the one-time row load is itself a full gmem pass; at small N the re-read passes are L2 hits already (warm-L2 caps upside ~20