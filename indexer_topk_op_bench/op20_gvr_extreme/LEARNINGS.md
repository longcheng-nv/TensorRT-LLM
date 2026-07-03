# op20 learnings

## Inherited red lines (do not retry)
- op14: pass-count reduction moot (rows fit 126.5MB L2; dram bytes = 1× input already);
  no warp-collectives (ballot/shuffle) in the per-element streaming loop.
- op15: smem-resident staging falsified (warm-L2 also slower ⇒ no traffic to save);
  small-N lever is structural, not residency.
- op16: band-shrink / two-threshold peeling inside the secant framework is tax-bound;
  P2 sampling is an L2 trap.
- op19 iter14: smem savings that raise CTA residency 2→3 at K2048 highBS regress 0.918→0.745.

## Effective techniques (inherited)
- op18: CDF-aware round-1 threshold placement (offline 5-seed straddle fracs).
- op19: sandwich thr0/thr1 pair from one M-ary ladder pass; defer-direct P3 write;
  band-snap P4 with runtime-k; per-cell dispatch tables (240 keys/dtype).
- op7: rank-scatter exact P4 (wins large-N; P4 floor at small N is WORSE than snap —
  phases_rs shows 11.9µs vs 9.2µs snap at K512 N4096).

## Current best
- (iter0) op19 `gvr_sw_auto` verbatim copy = `src/gvr_x_op.py`.
