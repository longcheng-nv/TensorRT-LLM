# op34 anchor — node umbriel-b200-048 (session 2026-07-15)

Node changed 074→048 since iter1. Absolute µs do NOT transfer; re-established here.

## Anchor cell (fixed for this session)
pro / ISL_256k / layer 32 / BS=1 fp32.  N=65539 (compressed valid), Npad=65600,
K=1024, hit_rate=0.489.

## Anchor numbers (cold-L2 512MB-evict wallclock median ×30 — L1 SCREEN only, NOT ship verdict)
| arm | us_cold | ratio vs sglang | exact |
|---|---|---|---|
| sglang_v2   | 21.61 | 1.000 | — |
| op26_r0auto | 31.24 | 1.446 | vdiff=0, recall=1.0 |

Ship goal at THIS cell: new ≤ sglang/1.30 = 16.6us ⇒ need **1.88× over op26_r0auto**.
(Wallclock includes ~12µs launch floor + host overhead; nsys pure-kernel ratios will
differ in absolute µs but the r0/sgl ratio is the anchor. §10 grand geomean was
sgl 7.88 / r0 12.31 on node 039 nsys — ratio 1.56 there; 1.45 here at one large-N cell.)

## HARNESS GOTCHA fixed this session (would corrupt every GVR number)
GVR `gvr_r0_op26(logits, pre, seq_lens, K, cr)` expects seq_lens = **UNCOMPRESSED**
length (kernel computes N_internal = seq_lens//cr). The v4cap bundle's `N` is the
COMPRESSED valid length. So GVR seq_lens MUST be `N*cr`. Passing `N` makes the kernel
scan only N/cr=16384 elements → recall≈0. The nsys_op34.py harness already does this
right (`_seq(N,cr)`); my env_anchor.py initially didn't — caught by the exactness check.
sglang seq_lens = N (compressed) directly. Convention divergence is a live trap.

## Cold-L2 refinement of the feasibility framing (this session's insight)
The FEASIBILITY_ANALYSIS pass-count framing is incomplete under cold-L2 (canonical):
- sglang 2 passes = 1 HBM read (cold, pass1 histogram) + 1 L2-hot read (pass2 collect).
- GVR   2 passes = 1 HBM read (cold, R0 count)      + 1 L2-hot read (P3 collect).
⇒ SAME cold-HBM traffic (1×N). Pass-fusion (eliminating GVR's L2-hot P3) saves only an
  L2 read (op29 measured 1.03–1.13× — matches). It is NOT the lever under cold-L2.
- The REAL difference: GVR reads with grid=(rows,1,1) = **1 SM** doing the cold-HBM read
  (single-SM bandwidth cap); sglang cluster uses **8 cooperating CTAs** = 8× the cold-HBM
  parallelism. The cold-HBM first read is what dominates at large N, and GVR does it with
  1/8 the SMs. ⇒ **multi-CTA-per-row is the decisive lever; pass-fusion has a low ceiling.**
This REDIRECTS iter2 (single-CTA pass-fusion, teed up by the prior session) toward the
multi-CTA structural lever. Next: NCU-confirm on this anchor cell (grid dims, DRAM bytes,
DRAM/SM throughput) before committing to a multi-CTA kernel.
