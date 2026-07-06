# op21 → upstream integration assessment (iter10 deliverable, 2026-07-06)

Scope (per SESSION_HANDOFF item 2b): diff op21's `gvr_ms`/`gvr_msc` against
the production GVR operator in tensorrt_llm, enumerate which levers port, the
code-surface delta, and the e2e validation plan. **This is the plan, not the
port.**

## 1. Production surface inventory (where the operator lives today)

| piece | location | state |
|---|---|---|
| Kernel | `tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode.py` | origin/main via #14602 → #15198 (multi-CTA DSMEM) → #15304 (load-balance) |
| Runner / custom op | `CuteDSLGvrTopKDecodeRunner` + `trtllm::cute_dsl_gvr_topk_decode` in `_torch/custom_ops/cute_dsl_custom_ops.py` | origin/main; 3 call modes: single-CTA / sort-indirect (`order_row`) / LB (`counters`+`order_row`, cluster {2,4,8}) |
| Model wiring | `_torch/attention_backend/sparse/dsa.py`: `enable_heuristic_topk` + `warmup_heuristic_topk_decode` + prefill→decode GVR handoff seeding | origin/main (this is the dsv4-pareto-bench GVR ON/OFF toggle) |
| Unit tests | `tests/unittest/_torch/.../sparse/test_cute_dsl_gvr_topk_decode.py` | origin/main — includes the 30 adversarial multi-bucket cases (large BS / cluster / varlen, uniform-random logits) |
| Kernel bench | `tests/scripts/cute_dsl_kernels/top_k/run_gvr_topk.py` | origin/main |
| Rank-scatter P4 | `fork/feat/gvr-rank-scatter-p4` (= PR #15709 route): `1872b84123` add → `1e07e506a1` default-ON → **`ec04147502` REVERT to snap-default, rank-scatter opt-in** | not yet in main at this checkout's fetch |

Upstream kernel algorithm (main): P1 `phase1_preidx_stats` (preIdx stats
seeds) → P2 `phase2_secant_search` (M=1 count/round, secant) → P3
`phase3_collect_candidates` → P4 `phase4_histogram_snap` (exact; rank-scatter
opt-in on the fork branch). Multi-CTA = slice scan + DSMEM count aggregation
(`mapa`/`ld.shared::cluster` helpers already in the file); host cluster policy
(2026-06-10 tuning): N<64K→1; BS≤4 ∧ N≥131072→8; 4·BS≤SMs→4; 2·BS≤SMs→2; else 1.

## 2. Algorithmic delta (op21 vs upstream main)

| phase | upstream main | op21 |
|---|---|---|
| P1 seed | preIdx min/mean stats → secant seeds | rank-quantile P1b: QBINS=256 histogram over gathered prev-K, parallel suffix scan, M=4 data-driven columns (iter1) |
| P2 | secant, 1 count column/round, ~1.3–2.1 rounds | ONE fused M=4 ladder round + speculative slot-collect (fuse gate `BS≤SMs ∧ 4K≤kC`), secant round-2 fallback (iter1) |
| P3 | separate collect pass | fused into P2 (happy path); msc: distributed direct-write at rank prefixes + **band remote-store push** via `st.shared::cluster` (iter7) |
| P4 | histogram-snap (exact default) / rank-scatter (opt-in, inexact) | rank-scatter + small-bin fast paths A/B (exact) + fixed-depth fine fallback path C (**inexact hazard — see §4**) (iter5+6) |
| multi-CTA | slice scan + DSMEM count agg; LB variant | row-chunked C-CTA cluster, replicated P1/P1b, slice ladder, DSMEM merge, distributed P3, leader P4 |
| 16-bit | dtype-generic via cvt→fp32 ladder | native `set.ge.{bf16x2,f16x2}` packed counts + threshold quantization-at-emit (iter9) |
| dispatch | 4-tier cluster policy + LB mode | 3 rules + fuse gate incl. **16-bit C8 rule** `N≥65536 ∧ N≥32768·BS` (iter8) and K2048-fp32-hugeN C8 (iter3) |
| features op21 LACKS | `next_n` (MTP rows), request-level varlen `seq_lens` contract w/ next_n, sort-indirect mode, LB mode, `_pick_tuning` per-shape tuning cache | — |

Measured stack-up on the op21 bench axis (nsys cold-L2, B200, vs per-cell
best rival): fp32 P0 gm 0.83 (iter1 single-CTA) → 1.051 (iter2 cluster) →
1.104 (iter5 RS-P4) → 1.125 (iter6 fast paths) → **1.249, 17/17 (iter7
push)**; bf16 1.028 → **1.091** (iter9 native ladder); fp16 **1.055**. vs
best existing GVR-family op (op8 lineage ≈ upstream main + RS): **1.139
fp32 / 1.285 bf16 / 1.267 fp16** — that last row is the honest estimate of
what porting op21 buys over the CURRENT production kernel.

## 3. Lever-by-lever portability

Ranked by (measured gain on the op21 axis) × (port cost). "Gain vs prod" uses
the gvrbest column (best GVR-family op) as the production-kernel proxy.

| # | lever (iter) | gain attribution | port target | surface | risk / dependency |
|---|---|---|---|---|---|
| L1 | P3 band remote-store push (iter7) | P0 gm 1.125→1.249 (the single biggest lever) | cluster path leader band assembly | new `st.shared::cluster` PTX helpers (~40 ln; file already has `mapa`/`ld` twins) + P3/P4 boundary rework | **depends on the op21 P3 structure** — push needs the global band prefix known before the walk (ladder count publish). Upstream secant publishes cnt_lo/cnt_hi so an analogous push is plausible, but it is NOT a drop-in patch on `phase3_collect_candidates`; medium confidence without a probe. Strongest argument for Strategy B (§5). |
| L2 | P4 small-bin fast paths A/B (iter6) + rank-scatter (iter5) | 1.104→1.125 + 1.054→1.104 | replaces P4 default | ~450 ln into the kernel (3-way branch + coarse search; op8-lineage code already reviewed in PR #15709) | **exactness precondition §4 — path C must become an exact fallback before any default flip.** Upstream already reverted a rank-scatter default once (`ec04147502`); a second attempt must pass the same 30 adversarial tests. |
| L3 | native 16-bit ladder + threshold quantization (iter9) | bf16 gm 1.028→1.091; fp16 1.043→1.055 | `block_count_ge` / ladder loops | ~250 ln, const_expr-gated (fp32 binaries provably unchanged — op21 demonstrated) | value only on 16-bit logits; DSv4 production indexer logits are fp32 today ⇒ future-proofing, ship last. Quantize-at-emit trick is general and low-risk. |
| L4 | dispatch rules (iter3+8) | K2048 fp32 262K BS1 +4%; 16-bit C8 cells +0.9–2.5µs | host `cluster_size` policy in the runner | ~20 ln | trivial; upstream policy is already the same shape (compare §1 policy vs op21 rules). Add: 16-bit C8 term + K2048-fp32 term. Independently landable. |
| L5 | order-stat P1b + fused M=4 ladder + slot collect (iter1-2) | the foundation: single-CTA gm 0.83→1.05 over secant lineage at P0; chain-length win at P1 | replaces P1/P2/P3 core | effectively a new kernel (~2.5K ln) | this is not a patch — it is the op21 kernel. See Strategy B. |

Inseparability note: L1 (push) consumes the ladder's per-CTA band prefix
publish, and L2's fast paths consume the band produced by the fused
slot-collect. The three top levers form a dependency chain rooted in L5.
**Incremental porting captures L4 (+ maybe L3) only; the 1.14–1.29× vs
production requires the op21 kernel body.**

## 4. Exactness precondition (P0 blocker, found in this assessment)

Upstream commit `ec04147502` (2026-07-01) reverted rank-scatter-P4-by-default
because **a fixed-depth histogram cannot separate two distinct values in the
same (sub-)bin** — the straddling bin can emit a value below the true K-th
rank; 30 `test_cute_dsl_gvr_topk_decode` adversarial failures (uniform-random
logits, large BS / cluster / varlen).

op21's P4 (`phase4_band_rank_scatter`) has the SAME latent mode in **path C**
(`_p4_band_fine_scatter`): 1024-coarse × 256-fine fixed depth, deepest-bin
members emitted in stash order. Paths A (whole-bin) and B (≤32 members, warp
register ranking) are exact; C fires only when cnt(b*)>32 ∧ ≠ r_need — never
on real/synth probe data (iter6 host probe: cnt(b*) max=4), and op21's gates
(synth 54/54, real 360/360, adversarial-preIdx 36/36) all pass because they
attack preIdx quality, not logits collision structure. 16-bit duplicates are
ties (tie-order emit is exact); the killer input is fp32 continuous values
straddling the cut inside one fine bin with >32 coarse-bin occupants.

**Port requirement**: replace path C's fixed-depth scatter with an exact
fallback — cleanest is `phase4_histogram_snap` on the residual straddling
bin (upstream's own exact default), or recurse-until-≤32 + path-B ranking.
Cost: path C never fires on production distributions, so the nsys tables in
SHIP_REVIEW.md are unaffected; exactness becomes unconditional. THEN run the
upstream 30-case adversarial suite + op21's real/adversarial gates on the
combined kernel. Also adopt upstream's adversarial multi-bucket cases into
op21's own smoke suite (gap found today).

## 5. Integration strategies

**Strategy A — incremental patches onto `gvr_topk_decode.py`**: lands L4
(dispatch, trivial) and L3 (16-bit ladder, self-contained). L1/L2 are not
honest patches (dependency chain §3). Captures a few % at 16-bit and the
K2048 fp32 cell; leaves the 1.14–1.29× core on the table.

**Strategy B (recommended) — op21 kernel as a sibling variant, PR-#15709
route**: land `gvr_ms`/`gvr_msc` (with §4 fix) as e.g.
`gvr_topk_decode_ms.py` next to the existing kernel, runner-selected.

Work items for Strategy B, in PR order (mirrors how #14602→#15304 landed):
1. **PR-1 kernel (opt-in, default OFF)**: port `src/gvr_ms_op.py` +
   `src/gvr_msc_op.py` (~3.4K ln today; expect ~2.5–3K after dropping bench
   scaffolding), with: §4 path-C fix; OP21_* env knobs → constructor flags
   (production code must not read env for behavior; keep env only in the
   bench scripts); copyright headers; `NUM_SMS` from device properties (B200
   = B300 = 148 today — thresholds coincide; policy must re-read per device).
2. **PR-1 runner extension** (~150 ln): new path in
   `CuteDSLGvrTopKDecodeRunner` behind a flag; compile-cache keys =
   (dtype, K, next_n, cr, BS, max-N, C, fuse) — all capture-time constants,
   CUDA-graph compatible (op21 dispatch already keys on buffer shape only).
3. **Contract gaps to close in the port** (op21 lacks these today):
   `next_n > 1` (MTP rows: row = req·next_n+nn indexing in P1 gather and
   varlen N per request); sort-indirect `order_row` mode; LB mode (either
   support C∈{2,4,8} slices under the LB partition or restrict the new path
   to non-LB and keep GvrTopKLBKernel for LB batches — RECOMMENDED first
   step); `return_output_values=False` (op21 already const_expr's this);
   GvrParams kC table vs op21's fixed kC=5120 fuse constant (map per
   (dtype, K, cr) — flash K512/cr4, pro K1024/cr4, v3.2 K2048/cr1 all
   already validated by op21's real-capture gates).
4. **PR-2 tests**: extend `test_cute_dsl_gvr_topk_decode.py` to run both
   kernels × the adversarial suite; add op21's real-capture exactness
   (real_data_v2 60L × C × {fp32,bf16,fp16}) as an LLM-models-root-gated
   test; adversarial-preIdx cases from `smoke_real_msc.py`.
5. **PR-3 dispatch flip** (after e2e soak): default the runner to the new
   kernel for non-LB decode; add the 16-bit C8 + K2048 rules to the policy.

## 6. E2E validation plan (dsv4-pareto-bench GVR ON/OFF A/B)

Stage 0 — unit: upstream suite (incl. 30 adversarial) + op21 gates, both
kernels, fp32/bf16/fp16, C∈{1,2,4,8}, varlen + next_n∈{1,2,4}.

Stage 1 — kernel nsys: op21 grids (drive_nsys_iter2.sh + drive_nsys_16bit.sh)
+ `run_gvr_topk.py` A/B old-vs-new on B200 AND B300 (B300 fp32 already
HW-invariant: gm 1.268 vs 1.249 B200, 17/17 both — B300_RESULTS.md).

Stage 2 — e2e perf: dsv4-pareto-bench, Flash (K512, cr=4) + Pro (K1024,
cr=4), TEP8, decode-heavy cells (ISL 64K/128K/256K, OSL 4-8K, BS sweep 1-32
covering the P0 grid's BS range), three arms: heuristic_topk OFF / ON-old /
ON-new. Acceptance: (a) indexer top-K kernel time in nsys decode traces
shrinks per the SHIP_REVIEW ratios, (b) e2e TPS/TPOT Pareto no-regress vs
ON-old on every cell, (c) no new G-series hazards (dispatch is compile-time,
CUDA-graph identity preserved — G3/G10 class risks nil by construction, but
verify graph capture with the new path once).

Stage 3 — accuracy invariance: exact top-K ⇒ selection identical to the
reference dispatcher; run dsv4-gsm8k-eval once on ON-new (Flash) as a cheap
end-to-end exactness canary (score must match ON-old run-to-run band).

Stage 4 — soak + flip: leave PR-1/2 opt-in for one release cycle of perf CI;
then PR-3 flips the default with the Stage-2 tables attached.

## 7. Open questions / follow-ups feeding this plan

1. B300 16-bit cross-check completion (in flight — B300_RELAUNCH_PROMPT.md).
2. K2048 16-bit BS1 tail ablation (iter10 item 3): outcome documents whether
   the ported kernel carries a known 0.88–0.96 cell family at 16-bit K2048
   (v3.2-geometry only; DSv4 Flash/Pro are K512/K1024 — not production-
   blocking for DSv4).
3. Whether PR #15709 (rank-scatter opt-in) merged upstream after this
   checkout's last fetch — determines PR-1's base.
4. LB-mode interaction: op21 never measured mixed long/short batches; the
   op#9/gvr-dispatch lesson (always-single wins; complex dispatchers lose)
   suggests keeping LB orthogonal initially (§5 item 3).
