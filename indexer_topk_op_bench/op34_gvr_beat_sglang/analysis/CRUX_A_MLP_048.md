# op34 CRUX-A — MLP scan-scaling + large-N target (node 048, cold-L2)

## Bare multi-CTA count scan (Triton, pure-kernel NCU gpu_time, µs)
| N | C8 | C16 | C32 | C64 | C128 |
|---|---|---|---|---|---|
| 65539  | 13.44 | 9.54 | 7.65 | 6.75 | 6.53 |
| 262144 | 34.59 | 20.64 | 13.12 | 9.28 | 7.30 |
| 1048576| 124.99| 64.83| 35.94| 20.32| 12.77|

Scan latency KEEPS dropping well past C=8 (esp. large N: N=262144 C8→C64 = 34.6→9.3 = 3.7×).
sglang uses a FIXED 8-CTA cluster; at BS=1 (147 idle SMs) a GVR with C=32–64 gets 2–4× more
MLP on the scan. Even C128 stays <5% DRAM / <5% SM peak — still latency-bound, not saturated.

## Full-kernel targets (NCU pure-kernel, cold)
| N (compressed) | sglang total | op26_r0 (1-CTA) | beat-30% goal = sglang/1.3 |
|---|---|---|---|
| 65539  (256k ISL) | 28.16 µs (grid 8) | 43.26 µs (grid 1) | ≤ 21.66 µs |
| 262144 (1M ISL)   | 39.04 µs (grid 8) | 80.93 µs (grid 1) | ≤ 30.03 µs |

## The op34 winning hypothesis (large N)
The GVR **hint** lets it collect at a known threshold in ONE fused count+collect pass, where
sglang does TWO passes (histogram then collect). Combined with C>8 CTAs (each CTA collects only
~1.2K/C ≈ 40 candidates ⇒ tiny local buffer, a genuinely CHEAPER append than Opt-L's single-CTA
global-atomic that was falsified). Napkin at N=262144:
  fused scan C=32 (13µs) + P1b-threshold (~2µs) + cross-CTA merge + P4 rank-scatter (tail ?) 
  vs sglang 39µs; goal ≤30µs. Feasible IFF the merge+P4 tail ≤ ~15µs.
The **tail cost is the open variable** and decides both the large-N margin and the small-N floor.

## GO/NO-GO gate → rung-2 proxy (next)
Build a multi-CTA collect (Triton) + torch.topk tail, ORACLE threshold (UB best case), NCU-summed
cold. If UB(best-case multi-CTA top-K) > sglang/1.3 at large N ⇒ walled (report double-lock). If
UB < sglang/1.3 ⇒ GO build the CuTe multi-CTA GVR (last-CTA-merge, single launch — 2-kernel would
add a ~12µs launch floor that kills the win at these µs scales).

## Envelope note
GRAND 30% average is likely dragged down by small-N (4K–32K ISL ⇒ N=1024–8192) where op32's
latency floor + cross-CTA merge overhead wall the GVR skeleton. Expect the honest outcome to be a
CONDITIONAL large-N win (ISL≥256k) via multi-CTA + regime dispatch, EXCEEDING op31 (which only
reached parity because it kept C=8 and never tried C>8). Full grand geomean measured before ruling.
