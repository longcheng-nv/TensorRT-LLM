# op16_gvr_dualthresh — Iteration log

Base op: **GVR cuteDSL rank-scatter P4** (op#7) =
`p4_recursive_digit/src/gvr_topk_decode_p4.py` → `src/gvr_topk_decode_dt.py`.
Wrapper mirrors `harness/gvr_cutedsl_rs_op.py`. HW: **B300 sm_103**.

User goal: two-threshold P2 (add `threshold_1` with count `M<K` ⇒ those M are
definite top-K winners), so P3 collects only the band `[threshold, threshold_1)`
and P4 refines only `M0-M` elements with a cheap select (bitonic/insertion).
Target: >95% of report cases beat Radix-cuteDSL AND SGLang, +40% avg per seqlen.

---

## Iter 0 — Baseline characterization + decisive P2-tax measurement (B300)

**Strategy**: before building, measure whether the mechanism can pay. Used the
clock64-instrumented rank-scatter kernel (`harness/measure_cute_phases_rs.py`)
with a `kC`/`kFTarget` override subclass (`scripts/p4_scaling.py`) to isolate
P1–P4 CYCLES as the candidate count shrinks (smaller kC ⇒ tighter threshold ⇒
fewer candidates). clock64 tot ≈ cold wall at these sizes (single-CTA,
18464 cyc ≈ 10µs ≈ the 9.7µs cold sweep number), so cycle deltas ≈ wall deltas.

### Results — K=512 fp32 cr=4, P4 cycles vs (kC, eval-optimal kFTarget)

| N | (kC,kFT) | P1 | P2 | P3 | P4 | tot | vs base |
|---:|---|---:|---:|---:|---:|---:|---:|
| 4096 | (5120,512) base | 3491 | 3874 | 1741 | 7050 | 16349 | — |
| 4096 | (1536,1280) | 3163 | 5152 | 1632 | 6078 | 16107 | −1.5% (wash) |
| 4096 | (1024,1024) | 3114 | 5132 | 1736 | 5959 | 16124 | −1.4% (wash) |
| 4096 | (768,640) | 3124 | 6881 | 1671 | 5997 | 17732 | +8% loss |
| 16384 | (5120,512) base | 3682 | 4425 | 3196 | 7120 | 18567 | — |
| 16384 | (1024,1024) | 3258 | 6085 | 3142 | 5823 | 18530 | −0.2% (wash) |
| 65536 | (5120,512) base | 3656 | 11030 | 9208 | 8656 | 32735 | — |
| 65536 | (1024,1024) | 3816 | 15999 | 8880 | 6972 | 35863 | +9.5% loss |

All variants EXACT (value-equiv torch.topk, uniq==K).

### Decisive findings
1. **P4 is only ~15% count-reducible at small N** (7050→5959 as cand 2K→1K);
   it only collapses (→~260) when cand→K, which needs kC≈K.
2. **Tightening the threshold explodes P2 by 1.3–2.5×** — the eval tax. With
   eval-optimal kFTarget the tax ≈ the P4 saving at small N (WASH) and EXCEEDS
   it at N≥65536 (LOSS). Exactly reproduces op13 on the rank-scatter P4.
3. The **band cannot be shrunk without the tax**: band = cand − c_hi ≥ K − c_hi,
   and c_hi < K, so band ≤ 256 requires cand ≤ c_hi+256 < 768 ⇒ kC≈K ⇒ max tax.

### Theoretical ceiling of the FULL two-threshold idea (free band-select P4)
Even assuming the band-refine P4 is FREE (replace rank-scatter with ~300cyc
single-warp select) at the tightest measured kC=768:

| N | tot with free select | vs baseline |
|---:|---:|---:|
| 4096 | 3124+6881+1671+300 = 11976 | **−27%** |
| 16384 | 3366+9757+3166+300 = 16589 | **−11%** |
| 65536 | 3729+20087+8393+300 = 32509 | **−0.7%** |
| ≥131072 | P2 tax explodes, P4 already ~small | **LOSS** |

⇒ Best-case ceiling: mild mid-N gain, decaying to zero by 65K, loss beyond.

### Why the +40%-everywhere TARGET is structurally unreachable (independent of P4)
- **Small N (4K):** best-case −27% → gvr 9.7→7.1µs, but Radix=6.45µs. Radix's
  floor is BELOW GVR's floor; cannot beat Radix here, let alone by 40%.
- **Large N (131K/262K):** GVR is P2+P3 (full-N streaming) bound (60–70%); at
  262K gvr=35µs vs Radix flat 19µs; even P4→0 leaves ≫Radix. A threshold-
  streaming kernel cannot beat flat radix-select here. (~26/60 cases.)
- The two-threshold idea attacks P4, which only dominates at small/mid N.

### Verdict
The user's specific mechanism (add threshold_1 in P2 via secant) is **tax-bound**
like op13/op14: net wash at small/mid N, loss at large N. The stated
+40%-over-Radix-at-95%-of-cases target is **not physically reachable** by any
P4-side optimization. The only lever that could unlock the P4 collapse is a
**cheaper P2** (pin a tight threshold in far fewer full-N passes — sampling /
better init / better root-finder), op13's unbuilt H2.

### Next action
Report to user with data + options: (A) build the achievable mid-N band-select
version (~10–25% mid-N, sub-target); (B) pursue cheaper-P2 lever (uncertain,
larger); (C) accept target unreachable and stop. Await direction.
