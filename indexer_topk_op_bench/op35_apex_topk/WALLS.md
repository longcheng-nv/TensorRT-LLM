# Structural walls (op35)
- (to be filled; candidates: BS1 launch floor, wave quantization at BS128-1024 small N)
- WALL (2026-07-17, iter10-17): +50%-over-composite-frontier is structurally
  infeasible for sampling-filter-select on the op26 envelope. Corrected floor:
  launch-bound cells' true floor = launch + threshold-acquisition + emission
  (frontier already at 1.1-1.5x of it); BW-bound cells need total <= 1.27x
  pure-read (below our measured filter-only tax 1.14-1.31). Micro-kernel
  pass/barrier cost ~3x boost math (IPC-bound, not DVFS). Final best:
  gm 0.507/0.516/0.512 (fp32/bf16/fp16), 1041/1041 exact.
- Per-CTA serial pass count is the currency at real clocks: single-CTA-per-row
  designs lose outright (iter14: N32768 1-CTA = 55us vs 20us sampled).
