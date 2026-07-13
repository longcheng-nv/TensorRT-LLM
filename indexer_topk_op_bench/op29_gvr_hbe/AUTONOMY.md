# AUTONOMY — the autonomy contract

Signed by: loncheng (user directive 2026-07-13: 以op27为起点继续优化, 目标全面超过SGLang v2) on 2026-07-13. The agent runs unattended within the self-decide
domain and stops only at the must-stop points.

## Self-decide (execute best judgment; do NOT pause to ask)
- Parameter choices, experiment design, probe ordering
- Discarding negative results and moving to the next hypothesis
- Checkpointing (commit early and often), RESUME upkeep
- Spending within the budget in PLAN.md

## Must stop for the human
- Shipping to production / opening the upstream PR
- Any change to baseline semantics or the incumbent's code
- Exceeding the cost budget (report burn rate at 80%)
- Changing the objective triple or the deployment envelope

## Pre-authorized negative conclusion
"If sglang_v2 remains unbeatable on some (scenario x K) slice — in particular WORST, where the hint carries no information and the structural floor is parity, state it plainly with numbers."
A clean negative report is a valid campaign outcome — never force a win.

## Reporting cadence
- ITERATIONS.md entry + commit per iteration (no batching)
- One-line decision announcements before acting on judgment calls
