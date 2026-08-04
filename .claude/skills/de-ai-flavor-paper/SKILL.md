---
name: de-ai-flavor-paper
description: Detect and remove LLM-flavored prose from computer-systems/architecture academic manuscripts (LaTeX) by comparing against pre-2022 human-expert best-paper norms (PPoPP/ASPLOS-class). Runs a quantitative grep battery, produces a severity-ranked finding list with line anchors, and applies minimal style-only revisions that never change technical claims or numbers. Triggers: "去AI味", "AI味道", "de-AI flavor", "humanize paper prose", "does this paper read like AI wrote it".
user-invocable: true
---

# De-AI-Flavor for Systems/Architecture Papers

Assess how much a manuscript "reads like an LLM wrote it" and revise it toward
the norms of pre-LLM-era human expert writing, using quantitative baselines
distilled from PPoPP Best Papers 2017–2022 (~74k words of human-expert systems
prose: Tapir, TxSampler, C-IST, SCCL, Lock-Free Locks Revisited, Software
Combining in Persistence). These norms transfer to PPoPP/ASPLOS/ISCA/MICRO/
SOSP/OSDI-class venues.

## Scope and ethics

- This skill improves **writing quality**, aligning prose with human-expert
  norms. It is NOT a tool to hide AI involvement: most venues (ACM policy,
  PPoPP/ASPLOS CFPs) **require** disclosing generative-AI assistance in the
  Acknowledgments. Always check the target venue's CFP and keep/restore a
  neutral disclosure statement; removing required disclosure risks
  desk-rejection or retraction.
- Never change technical claims, numbers, units, populations, or evidence
  scope. Style edits only. If a repetition exists because an evidence-
  discipline document requires it, keep one canonical instance and flag the
  rest instead of silently deleting.
- Never inject fake "human quirks" (deliberate typos, fabricated anecdotes).
  Authentic humanization comes from structure: worked examples with real
  measured numbers, first-person design rationale, uneven formatting.

## Inputs

- Path to the manuscript `.tex` (or Markdown) file(s). If the user gives a
  directory, locate the main file (`\documentclass`).
- Optional: a venue-specific corpus of pre-2022 papers for recalibration.
  Default baselines are in `reference.md` (read it before Phase 2).

## Workflow

### Phase 1 — Quantitative detection

Run the grep battery: `scripts/detect_ai_flavor.sh <file.tex>` (from this
skill's directory). It prints a scorecard covering both families of tells:

**Family A — classic LLM lexical markers** (any hit is a flag):
`comprehensive/novel/principled/carefully(self-praise)/crucially/notably/
importantly/furthermore/moreover/overall/systematically/extensive/leverage/
delve/robust/seamless/state-of-the-art(self)/holistic/meticulous/pivotal/
paradigm/landscape/realm/myriad/plethora/versatile/cutting-edge/key insight/
extensive experiments/in conclusion`.
Human baseline: **0 hits in 74k words** for almost all of these.

**Family B — structural tells** (the harder, more diagnostic family; a paper
can score 0 on Family A and still smell strongly of AI):

| # | Tell | Detection | Human norm |
|---|---|---|---|
| B1 | Mantra repetition: the same invariant/slogan restated in abstract, intro, captions, body, conclusion | grep top repeated 3–5-gram phrases; count key technical terms | An invariant is stated once at definition + once at theorem, then referenced |
| B2 | "X, not Y" antithesis aphorisms | `grep -cE ', not '` and `but (not|never)` | Near-zero; parallelism is functional, not ornamental |
| B3 | Template sentences with uniform geometry ("… are supplementary", "… is left to future work" repeated) | grep the repeated tail | 1–2 varied phrasings per paper |
| B4 | First-person density too LOW (system as sole actor, agentless passives) | count `\bwe\b|\bour\b` per 1000 words | 5–11 "we"/1000 words + "we believe / as far as we know" |
| B5 | Uniform caption geometry (every caption a polished same-length paragraph) | extract `\caption{}` word counts | Bimodal: 3-word fragments coexist with 150-word essays |
| B6 | Contribution bullets with bold/italic heads, equal length, parallel syntax | inspect the itemize | Terse, unequal, head-less bullets with §-refs |
| B7 | Title-Case concept branding beyond one system name | list capitalized multi-word coinages | ONE branded name per paper; everything else lowercase |
| B8 | Generic sweeping opener + zero worked example | read first page | Concrete worked example or exact number within first ~150 words of every major unit |
| B9 | Uniform defensive hedging on measured claims | scope-qualifier density per results sentence | Hedges attach only to unmeasured claims; measured claims stated flat |
| B10 | Connective glue (however/moreover alternating as paragraph openers) | sentence-initial connective counts | 0–5 per paper, several papers have 0 |
| B11 | Uniform paragraph geometry; "First,… Second,…" as argument scaffolding | paragraph length distribution | 4–10-sentence paragraphs; enumeration only for content taxonomy |
| B12 | Em-dash rhetoric ("not X—but Y") at high frequency | count `---`/`—` and classify function | 0 in half the corpus; where used, one consistent function |

### Phase 2 — Severity-ranked report

Read `reference.md` for the full corpus profile, then produce a finding list:

- Each finding: tell ID, evidence (line anchors + counts), severity
  (ordered by reader impact: B1 > B2 > B3 > B4 > B5 > B6 > B7 > B8 > others),
  and a minimal fix.
- Also report what is **already human-normal and must not be touched**
  (conditioned numbers, flat loss reporting, run-in bold mini-headings,
  restrained connectives). Guard against over-correction: do not add the
  Family A words while "fixing" Family B.

### Phase 3 — Minimal revision (only when the user asks for edits)

Apply in this order (deletions first — they free page budget for insertions):

1. **B1** Deduplicate mantras: keep 2 canonical statements, replace the rest
   with references or delete.
2. **B2** Keep ≤3 load-bearing antitheses; flatten the rest into plain
   declaratives.
3. **B3** Merge template sentences: one per section, varied wording.
4. **B5** Break caption uniformity: shorten captions of simple figures to 2–3
   sentences; keep dense result-figure captions long.
5. **B6** Strip bullet heads; allow unequal lengths; embed §-references.
6. **B4** Restore "we" as the actor for design decisions; add 1–2 honest
   epistemic sentences ("we believe…", "we do not have an explanation for…").
7. **B7** Lowercase coined action/phase names in running prose; keep the one
   system brand and pseudocode-internal names.
8. **B8** Insert one short worked example with real measured numbers from the
   paper's own data (a specific input, its behavior, the consequence) near the
   start of the introduction. Budget: use lines freed by steps 1–3.
9. **B9** Concentrate scope qualifiers in Methodology/Limitations; state
   measured results flat, one pointer per subsection.

Constraints during editing:
- Respect the venue page budget; track net line delta per edit.
- If the manuscript is under a concurrent revision protocol (evidence ledger,
  consensus checklist), coordinate: never batch style edits with numeric/
  claim edits, and re-anchor line numbers before each edit.

### Phase 4 — Verification

1. Re-run `scripts/detect_ai_flavor.sh` and compare the scorecard against the
   targets printed by the script.
2. Recompile the document; check page budget, unresolved references, overfull
   boxes.
3. Confirm no number, unit, population, claim, or evidence-state changed:
   diff should touch prose only. Spot-check every edited paragraph against
   its pre-edit meaning.
4. Confirm the AI-use disclosure statement required by the venue is present
   (anonymized if double-blind).

## Files

- `reference.md` — full corpus style profile (abstract shape, intro/
  contribution norms, sentence/vocabulary statistics, caption/paragraph
  rhythm, human quirks) with quoted examples and the norms-vs-tells table.
- `scripts/detect_ai_flavor.sh` — grep battery + scorecard with per-metric
  targets. Works on any single `.tex`/`.md` file.
