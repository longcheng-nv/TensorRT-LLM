# Session API cost — rung/kb512 series (session d54cb523, 2026-07-18 → 07-20)

Computed from the session transcript's per-request `usage` records
(`~/.claude/projects/.../d54cb523-*.jsonl`, 844 billed requests,
2026-07-18T22:33Z → 2026-07-20T02:48Z), priced at **claude-fable-5** rates
($10 in / $50 out per MTok; cache read 0.1× = $1; cache write 1h-TTL 2× = $20).
`/cost` in the CLI is the authoritative number; this is the reproducible
breakdown.

| Component | Tokens | Rate ($/MTok) | Cost |
|---|---:|---:|---:|
| Uncached input | 1,728 | 10.00 | $0.02 |
| Output | 1,161,214 | 50.00 | $58.06 |
| Cache read | 367,346,725 | 1.00 | $367.35 |
| Cache write (1h TTL) | 8,957,547 | 20.00 | $179.15 |
| **Total** | | | **≈ $604.58** |

Notes:
- Cache reads dominate (61%) — the price of a very long single-session context
  (interactive analysis + A/B campaign in one thread); output is only 10%.
- All cache writes were 1h-TTL (2× premium) per the session's cache config.
- Work covered: decode-capture report §5c/§5c-CCDF/-b/§5d + rung recalibration
  study + qfracs silicon A/B + kb512 4-arm A/B + pre-PR validation + 2 pushed
  PR#16457 commits (0d6fc4f1f2, 1128c0544f). GPU compute cost is separate —
  ~60 nsys batches ≈ 2-3 GPU-hours on b200-027 across the campaign.
