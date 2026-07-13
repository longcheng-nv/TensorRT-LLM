# op32 structural walls

W1 — multi-phase secant barrier chain (BS=1 short-N structural floor).
At BS=1 a single CTA runs a data-dependent 5-phase pipeline (P1 hint gather -> initial
count -> secant refine -> P3 collect -> P4 rank-scatter). Phases cannot overlap (each
consumes the prior's result); each transition needs a full-block barrier; at 25% structural
occupancy each barrier is a latency hit; issue rate is 15%. nsys floor ~9.7us (N=8192 K512
fp32 BS=1). Removing any phase is either falsified (register-resident F1; R0-ladder lost at
fp32 short-N) or harmful (refine=0 is SLOWER — P3/P4 blow up). The single-pass histogram that
beats it (sglang v2, ~7us, ~1.3-1.4x) is a DIFFERENT skeleton, excluded by the campaign
constraint. Config-insensitive: threads {512,768,1024} WASH; warp-parallel-reduce WASH.
Mechanism understood -> explore->productize switch: STOP exploring this axis.
