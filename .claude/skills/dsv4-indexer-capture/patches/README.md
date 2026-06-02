# Patches

`dsa_hook_v2.diff` — minimal v2 hook delta to apply on top of the
`feat/gvr-v4-dispatch-tuning` branch (or any branch with the v1 hook).
Adds:
- Prefill `logits` capture (G1)
- Phase gate (`DSV4_INDEXER_CAPTURE_PHASE` ∈ {prefill, decode, both}, G2)
- Per-layer + `.pt` / `.npz` flush (`DSV4_INDEXER_CAPTURE_LAYOUT` /
  `DSV4_INDEXER_CAPTURE_FORMAT`, G3)

## Apply

```bash
cd /path/to/TensorRT-LLM-q9j         # the worktree, NOT the main checkout
patch -p1 < /path/to/.claude/skills/dsv4-indexer-capture/patches/dsa_hook_v2.diff
# Verify:
grep -c "Q9j capture hook v2" tensorrt_llm/_torch/attention_backend/sparse/dsa.py
# expected: 2
```

If the patch fails to apply cleanly (the worktree is on a different
branch), apply manually following `SKILL.md` §Hook layout — the v2 block
is self-contained between the two `# === Q9j capture hook v2 …` and
`# === End Q9j capture hook v2 …` sentinel comments.

The patch base is the local `feat/gvr-v4-dispatch-tuning` branch HEAD
at the time the skill was written. If you are on `origin/main` and the
patch fails, expect a 2300-line context-bound mismatch — go manual.
