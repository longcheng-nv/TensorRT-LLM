# op37 cost ledger (running; finalized into REPORT.html at close)

> User directive 2026-07-20: 完成任务后统计 /cost 花费到本地文件/报告。
> Main-session token totals are only visible via the CLI `/cost` command —
> USER: please run `/cost` at campaign close and paste the block here (the
> assistant cannot read its own session cost). Sub-agent usage and wall/GPU
> time are tracked below by the assistant.

## Sub-agent usage
| date | agent | task | tokens | tool uses | duration |
|---|---|---|---|---|---|
| 07-20 | kernel-cute-specialist | gvrpkg37 dist_p4 splice + battery | 332,458 | 550 | 82 min |

## GPU time (umbriel-b200-028, 8×B200)
| date | run | GPUs | wall |
|---|---|---|---|
| 07-20 | T0 baseline sweep (12 batches, 2 arms) | 0-1 | ~35 min |
| 07-20 | L1 forced-cs probe (3 batches ×2 runs, arm-registry redo) | 2-3 | ~30 min |
| 07-20 | phase_bs clock64 breakdown (13 cells) | 6 | ~25 min |
| 07-20 | flash-512k 3-arm + base redo | 7 | ~15 min |
| 07-20 | dp4 splice agent battery/PTX/spotcheck | 4-5 | ~80 min |
| 07-20 | dp4-v1 verdict sweep (12 batches, 3 arms) | 0-1 | in flight |

## Notes
- nsys artifacts stay node-local (/tmp + results/*/nsys_reps, gitignored).
- Redo waste log: l1probe r1 discarded (arm-registry filter bug, ~10 min);
  f512k gvr_base needed a second batch (same bug) — fixed in ops_op37.
