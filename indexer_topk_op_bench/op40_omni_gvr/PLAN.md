# op40_omni_gvr — GVR Top-K further optimization on PR #16457 head (omni-kernel v2)

Campaign kickoff: 2026-07-23 · node umb-b200-239 (8× B200, all idle 30-37 °C)

## Objective triple (user-supplied 2026-07-23; agent MAY NOT relax)

```yaml
objective:
  incumbent: PR #16457 head, pinned @ e612fc2f38112becffee4409c5cf07ed264c85d5
             (branch perf/gvr-topk-r0-histogram-ladder; vendored copy in src/baseline/,
              sha1 db7da478.. / f928b244..)
  rivals: []            # dispatch to non-GVR ops (radix-select etc.) explicitly OUT of scope
  envelope:
    data: real decode captures, same as op26_r0 REPORT.html §7b — V4 Pro (K=1024),
          V4 Flash (K=512), V3.2 (K=2048); all GVR-active layers; ISL 4K–1M; BS=1; fp32
    cells: 865
  verdict_axes: [per-model × per-ISL-band breakdown on the real 865 grid]
    # single real-capture dataset ⇒ worst/real/best collapses to the per-cell grid;
    # regression scan over all 865 cells plays the "worst" role
  goal: geomean speedup ≥ 1.60× vs FRESH same-node paired re-measured baseline
        (NOT vs REPORT.html absolute numbers) — STRETCH GOAL, not hard ship bar:
        deliverable = achieved gm + structural-wall attribution of remaining gap
  ship_rule: exactness green (tie-aware value-multiset vs torch.topk, 865/865 +
             synth + adversarial) AND zero regression (paired nsys median-of-3
             ratio ≥ 0.97 on every cell) AND internal dispatch rules ≤ 3
  priority: DSV4 Pro > Flash ≈ V3.2 · ISL 32K–1M > 4K–32K (effort allocation +
            wash tie-breaks only; verdict metric = unweighted gm over 865)
  hard_constraints:
    - GVR skeleton preserved: preIdx prior → secant+log threshold search → refine
    - P1/P4 (and other phases) may be equivalently restructured (multi-CTA/cluster,
      radix sub-stages inside refine, pass reordering, compile-time K-specialization: allowed)
    - exact algorithm (no approximation)
    - measurement: B200, cold L2, nsys as sole ship arbiter
```

## Firewall (user-mandated, this campaign is an omni-kernel vs Kernel-Factory comparison)

- FORBIDDEN material: anything KF-generated — `kf_campaign/` dirs, KF champion c74f_sbx,
  R3 compA/compB, and KF-lineage buckets `op37_bs_scaling/`, `op38_r3v11_bs/`, `op39_gvr_bsx/`.
  No code, data, reports, or hypotheses sourced from them.
- ALLOWED: op26_r0_upstream_port_report/REPORT.html, PR #16457 itself, and non-KF
  op-series campaign ledgers (op21–op37 FALSIFIED/WALLS/LEARNINGS, op37_p4opt, PR #16715
  conclusions) per user ruling 2026-07-23 ("只禁 KF").
- Session-memory boundary (disclosed to user): agent cannot un-know KF conclusions;
  protocol-level isolation = never cite them as evidence; every hypothesis must
  independently climb the probe ladder and be verified on silicon in this bucket.

## Red lines (from non-KF ledgers; full checkpoint per iter in ledger/)

- Event-axis wins are not verdicts; nsys only.
- No dispatch on hit-rate (inference-time unknowable).
- BS=1 small-N cells are latency/launch-floor bound (~10 µs nsys floor) —
  DRAM-traffic levers void there; check L2-trap prior before any traffic lever.
- Saturated multi-GPU sweeps are screening only; ship verdicts single-GPU paired.

## Baseline immutability

`src/baseline/` is byte-frozen (sha1 recorded above). All variants live in
`src/variant/` as separate modules; harness selects via explicit arg. One-revert
recovery guaranteed.
