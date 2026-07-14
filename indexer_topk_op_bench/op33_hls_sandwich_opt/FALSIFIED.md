# op33 falsification ledger
F1 — D1 warp/register band tie-select as a NEW lever — already the incumbent default
  (p4_smallbin=True). p4_smallbin OFF = 0.835. No new headroom.
F2 — D3 qbins coarser (128/64) — negative (0.96-0.97); hist not the BS=1 bottleneck; precision loss.
F3 — D4 p4_rs OFF / slot_scale=1 — defaults optimal (0.91/0.99).
F4 — D2 M=3 (qfracs 0.85,0.35) for K512/1024 — **NO-SHIP: regresses the WORST scenario.**
  domain: fp32 BS=1 K512/1024. Clean single-idle-GPU PAIRED A/B (the only trustworthy verdict):
    K512 N8192 worst 0.884 | K512 N32768 worst 1.171 | K1024 N32768 worst 0.787 |
    K1024 N32768 real 1.108 | K512 N262144 worst 0.727.
  M=3 wins real (~1.1) but LOSES worst on 3/4 cells (−12..−27%). The deep 0.048 column M=3 removes
  is earning its keep on worst (op27 tail-ladder design). FAILS ship rule. root-class: the iter5
  "+9%" verdict was a MEASUREMENT ARTIFACT (real-only scenario + N≤65536 subset). Revival: none —
  the worst regression is mechanistic (deep-column removal → M0==0 fallback storm on low-hr rows).

MEASUREMENT META-LESSON (two omni-kernel violations this campaign, both cost a wrong "ship"):
  (1) NEVER headline one verdict axis. Testing scenario=real ONLY hid a worst catastrophe. Report
      [worst, real, best] every time.
  (2) 8-GPU-SATURATED sweeps corrupt A/B ratios — base & variant run on different GPUs under
      different contention, fabricating fake outliers (K1024 N32768 real read 0.227 saturated vs
      1.108 clean). Ship verdicts require single-GPU PAIRED back-to-back A/B (cf. anchor protocol).
