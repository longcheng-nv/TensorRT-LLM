# op39 iterations

## iter 0 — 2026-07-23 — GO (feasibility cruxes)
Hypothesis: post-op38 the 1.8-mean gap lives in (a) DRAM-bound big cells and (b) L2-resident BS>=16 cells.
Probe rung 0 x2 (NCU, pro_1024k BS512 + pro_64k BS256):
- (a) both arms read 2.05x floor (1.09-1.10GB vs 0.537GB); pr HW BW already 5.8TB/s (~80% roofline) -> pass-cut lever real but bounded ~1.9-2.0x; only 51/750 cases are DRAM-bound -> moves envelope mean 1.33->1.39 alone. NOT sufficient.
- (b) L2-resident mid cell: ALL SOL% <40 (SM 25-33, Mem 23-30, L2 8-11), 20us vs ~4us data floor -> latency/occupancy structural bound, room ~3-5x, 699 cases -> THE battleground. Need non-DRAM mean 1.79 (now 1.35).
Verdict: GO — arm design = tile-parallel single-pass collect (grid ~2xSM, (row,chunk) tiles, hint->conservative t_lo -> atomic append -> small exact top-K), unified for BS>=16; also cuts the 2nd pass for the DRAM-bound 51.
Ledger: none violated (new shape, not the falsified (TB,CS,MAXV,AR,HS) family).
Next: rung-2 microbench — oracle-threshold collect structure speed on pro_64k BS256 + pro_1024k BS512.
