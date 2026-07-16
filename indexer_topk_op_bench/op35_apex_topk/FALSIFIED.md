# op35 falsification ledger (seeded from GVR history 2026-07-16; scoped triples)
Inherited red lines (do NOT re-propose without citing revival condition):
- (fuse P3-collect into P2-count via per-element ballot/atomic slot-reserve; all K/N BS=1
  single-CTA; event+prod) = FALSIFIED Opt-L. REVIVAL CONDITION CITED by H0: append cost
  must scale with ADMITTED count (~3% of N) via warp-aggregated atomics, not per-element.
  To be re-proven at rung 2 before any kernel work.
- (smem-resident staging, N<=262k BS=1; nsys+warm-L2) = FALSIFIED op8/op15 — L2 already
  serves re-reads; irrelevant to H0 (single pass by construction).
- (cluster DSM multi-CTA at high BS; nsys) = FALSIFIED Opt-B/Q5f 0.36-0.45x. H0 uses NO
  cluster: global-atomic merge + last-CTA ticket.
- (P4-internal refine passes) = FALSIFIED; H0 tail = one small dense select, not refine.
- (P1 model-driven seed vs drift) = FALSIFIED for GVR P1; H0 seeds from SAMPLE+hint mix
  (distribution-free band), drift covered by band width — different mechanism, but if
  real-data miss% exceeds bound, revisit this entry.
- (2-way/4-way multi-threshold per pass, GVR skeleton) = wash — BUT that was threshold
  ladder WITHIN 2.5-pass structure; H0's ladder amortizes into a single pass. Cite: the
  wash was pass-count-neutral; H0 changes pass count.
- (BS=1 GVR-skeleton beat sglang) = op34 double-locked INFEASIBLE for GVR skeleton
  (oracle collect-only ~ sglang parity). H0 is NOT the GVR skeleton (no cluster barriers,
  1 pass); BS1 feasibility re-tested at rung0.1 with a leaner primitive.
