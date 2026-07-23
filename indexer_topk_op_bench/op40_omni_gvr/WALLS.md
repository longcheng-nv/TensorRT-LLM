# WALLS.md — op40_omni_gvr

Structural walls (config-insensitive, mechanism understood). Inherited candidates
to re-verify on the PR-head kernel (they were measured on earlier variants):

- Launch/latency floor REVISED 2026-07-23 by floor probe: true floor is
  ~1.7 µs (GVR prologue+identity, nsys). The inherited "~10 µs" figure was
  algorithm latency, not a wall. Small-N cells carry ~6-8 µs of attackable
  phase-chain latency.
- Grid ≪ SM count at BS=1 ⇒ occupancy is structural; only more-CTAs-per-row
  forms move it (multi-CTA/cluster P4 is the allowed lever family).

## Campaign entries (append below)
