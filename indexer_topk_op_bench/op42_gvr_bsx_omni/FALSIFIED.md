# FALSIFIED.md — op42 ledger
# Scoped triples: (conclusion, domain, evidence) + root-cause class.
# Seeded ONLY from sanctioned materials (user-cited op37 BS-decay analysis).

- (Sequential per-row full-GPU launches lose to head batched arm from BS=2,
  domain: all K/N, BS>=2, evidence: nsys op37 198/198) — structural-wall:
  1 cluster <= 10.8% SMs, same-stream serialization. Revival: none — this IS
  the wall this campaign attacks via kernel-side row parallelism.
- (Naive Python multi-stream launching is SLOWER than sequential, domain:
  BS>1 host-loop, evidence: nsys op37) — structural-wall: host issue rate ~
  kernel duration 8-12us. Revival: C++ launcher or single batched launch.
- (CUDA-graph fork-join proves rows are HW-concurrent; BS64 8.4x recovery but
  ceiling ~148/CS co-resident clusters, domain: BS<=8 narrow win, evidence:
  nsys op37 graph_probe) — mechanism note, not a falsification: batched
  single-launch dominates graph fork-join; graphs are a fallback lever only.
- (Event-axis A/B vs cuteDSL head arm is INVALID on b200-073: head host issue
  latency up to 1.2ms at BS<128 mCTA variants dwarfs 24us kernel, domain: all
  head-arm ratios, evidence: nsys diag_head_bs1 + 2-GPU repro) —
  measurement-artifact. Revival: none; nsys pure-kernel is the only A/B axis.
