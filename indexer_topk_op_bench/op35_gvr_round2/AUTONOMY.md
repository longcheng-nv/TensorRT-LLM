# op35 autonomy contract (USER-granted 2026-07-16)
- USER directive: "中间不要停顿去等待人给予决策,自行判断最佳优化路径"。
- Self-decide: all experiment design, parameters, discarding negatives, checkpoints.
- Must stop for human: merging anything into PR#16457 (explicitly forbidden — new
  optimizations go to a SEPARATE follow-up PR after correctness+perf validation),
  changing the metric/envelope.
- Pre-authorized negative conclusion: if +40% is infeasible, say so plainly with
  the double-locked bound; harvest all real wins regardless.
- Budget: this campaign ~1-2 days GPU on b200-081 (8 GPUs, idle); token cost
  tracked in COST.md at official Claude pricing.
