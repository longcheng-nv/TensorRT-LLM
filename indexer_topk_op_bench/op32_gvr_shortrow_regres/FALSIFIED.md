# op32 falsification ledger  (scoped triples: conclusion / domain / evidence / root-class)
F1 — register-resident to save scan re-read traffic — DEAD.
  domain: fp32 BS=1 short-N (4-16K), single-CTA. evidence: NCU (dram__throughput 0.06%, sm 0.08%).
  root-class: structural-wall (L2-trap; the row is 32KB, all re-reads are L2/register hits). Same
  mechanism as op15 smem-resident (warm-L2 parity). Revival: only if a variant is memory-bound —
  it is not at BS=1. Register-residency may still SUPPORT a latency opt but cannot win on traffic.

F2 — raise threads/CTA 512->768/1024 at BS=1 short-N — WASH-to-loss.
  domain: fp32 BS=1 N4-16K K512. evidence: L1 cold-L2 A/B (t512 best 7/9, high variance);
  t768 exactness-broke 2 cells. root-class: structural-wall — critical path is the serial
  barrier-dependency chain (data-dependent secant), not warp occupancy; extra warps have no
  independent work to overlap. Revival: only alongside a restructuring that creates independent
  overlappable work across the barrier chain (none proposed yet).
