# AUTONOMY — the autonomy contract (rmsnorm_campaign)

Signed by: human operator via tierB/KICKOFF.md on 2026-07-13. The agent runs
unattended within the self-decide domain and stops only at the must-stop points.

## Self-decide (execute best judgment; do NOT pause to ask)
- Parameter choices (Triton block/warp configs, autotune spaces), experiment design, probe ordering
- Discarding negative results and moving to the next hypothesis
- Checkpointing (commit early and often), RESUME upkeep
- Spending within the budget in PLAN.md (5 iters / 2 h)

## Must stop for the human
- Shipping to production / opening any PR (this campaign ends at a verdict report)
- Any change to baseline semantics or flashinfer source
- Exceeding 5 iterations or ~2 h wall-clock (report at 80% burn)
- Changing the objective triple or the token-grid envelope

## Pre-authorized negative conclusion
"If flashinfer.norm.rmsnorm remains the best option, state it plainly with
numbers." A clean negative report is a valid campaign outcome — never force a win.

## Reporting cadence
- ITERATIONS.md entry + git commit -s per iteration (no batching)
- One-line decision announcements before acting on judgment calls
