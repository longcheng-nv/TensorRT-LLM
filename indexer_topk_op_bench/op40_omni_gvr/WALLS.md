# WALLS.md — op40_omni_gvr

Structural walls (config-insensitive, mechanism understood). Inherited candidates
to re-verify on the PR-head kernel (they were measured on earlier variants):

- Launch/latency floor ~10 µs (nsys) for BS=1 small-N cells — bounds any gm
  contribution from N≤16K cells.
- Grid ≪ SM count at BS=1 ⇒ occupancy is structural; only more-CTAs-per-row
  forms move it (multi-CTA/cluster P4 is the allowed lever family).

## Campaign entries (append below)
