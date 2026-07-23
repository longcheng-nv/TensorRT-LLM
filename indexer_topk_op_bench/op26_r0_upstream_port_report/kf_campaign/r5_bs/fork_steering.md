# EXTERNAL FULL-GRID VERDICT — round-2 lineage has a CORRECTNESS BUG and two dispatch holes

The external referee ran your two best kernels (r2-v23 and r2-v39) on the FULL
750-case real grid (75 cells x BS{2..1024}, nsys cold-L2, per-row exactness).
Both are DISQUALIFIED. Read carefully — the platform workload subset cannot see
any of this, so you must fix it blind and self-test.

## 1. EXACTNESS FAILURES (inherited across your whole round-1/2 lineage)

Identical 8 failing cases in BOTH kernels, ALWAYS batch row 0 corrupted
(missing strictly-greater top-k elements, value diff up to 1.8):

  pro_512k  (k=1024, n=131063, npad=131072)  bs=2, 4, 8, 16
  flash_128k (k=512,  n=32771)               bs=64
  pro_128k  (k=1024, n=32771)                bs=64
  (row 0 ONLY, every time; rows 1..b-1 exact)

Signature suggests a cross-row race or wrong row-base in whatever tier serves
these (npad, bs) combos — e.g. a shared/prior stage that still reads/writes
row 0's slice while other rows proceed, or a cluster/DSMEM exchange keyed to
row 0. The platform subset has NO workload at (512k, bs 2-16) or bs=64, so
internal scores stay green while the kernel is wrong. Fix first; performance
is meaningless until every (npad, bs) shape is exact. SELF-TEST: build a
random-data unit test sweeping bs in {2,3,4,8,16,64,96} x npad in
{16384, 32768, 65536, 131072, 262144} x k in {512,1024,2048} and verify the
top-k index set per row before you submit anything.

## 2. FULL-GRID PERFORMANCE PICTURE (external, vs production native batch)

r2-v23: geomean 0.9348, 462/750 cases SLOWER than production.
r2-v39: geomean 0.9426, 458/750 slower. Your platform subset flatters you:
the probe subset (BS 4/32/256/1024 on 9 cells) reads ~1.07.

Two structural holes dominate:
- bs=128 at n~16387 (all three k): 0.40-0.47x — a tier/dispatch hole between
  your latency tiers and streaming tiers. bs=96..192 needs a real design,
  not a boundary fallthrough.
- K=1024 large-n (n=262127) at bs>=128: 0.42-0.47x — production's native
  batch amortizes the row scan; your per-row streaming ladder does not. At
  this shape the winning direction is throughput: rows-per-CTA > 1 or
  row-major persistent CTAs with per-SM slicing, minimal per-row setup, and
  ladder work shared across rows resident on the same SM.

Strong zones to KEEP: small-n all bs (up to 3.5x), large-n bs<=8, bs=1024
small/mid-n. Do not give these back while fixing the holes.

## 3. Acceptance reminder (unchanged, externally enforced)

average (geomean) >= 2.0x vs production on the full 750-case grid, EVERY case
>= 1.0x, per-row tie-robust exactness everywhere, BS=1 keeps the existing
champion's level. GVR skeleton hard rule unchanged.
