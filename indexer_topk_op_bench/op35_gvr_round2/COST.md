# op35 cost ledger (2026-07-16, single-session campaign on umbriel-b200-081)

## GPU-hours (8×B200 node, per-phase estimates from wall-clock × GPUs used)
| phase | GPU-h |
|-------|------:|
| env rebuild + snapshots + smoke | 0.3 |
| iter0 replay_b1 (77 cells, 1 GPU) | 0.2 |
| iter0 p3-oracle event screen (8 GPU) | 1.3 |
| floor-oracle + iter1 h3tail (8 GPU) | 1.6 |
| nsys 4-arm oracle (4 GPU) | 1.7 |
| launch-cfg screens (4 GPU) | 1.3 |
| kb512/kb256 screens (8 GPU) | 2.0 |
| iter2a/probe smokes + ablations (1-2 GPU) | 1.0 |
| iter3 full-grid L1 (8 GPU) | 1.6 |
| NCU attribution (2 cells) | 0.3 |
| L2 nsys ×3 verdict (8 GPU, incl aborted 4-GPU round) | ~7.0 |
| **total** | **~18.3** |

## Token cost (estimate at official Claude API pricing; exact = /cost in session)
Assumptions: Claude Fable 5 billed at Opus-class list prices
(input $15/M, output $75/M, cache-read $1.50/M, cache-write $18.75/M);
long-session with ~1h prompt-cache TTL => bulk of input is cache-read.
| bucket | est tokens | est cost |
|--------|-----------:|---------:|
| input (mostly cache-read) | ~8-12M cache-read + ~0.5M fresh | ~$20-30 |
| output (incl thinking) | ~250-350k | ~$19-26 |
| **session total (est)** | | **~$40-60** |
NOTE: estimates; the authoritative number is the session /cost readout.
