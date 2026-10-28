---
name: isc-paper-proofreader
description: Proofread, revise, or audit English research-paper prose for ISC High Performance. Use for ISC abstracts, introductions, related work, method descriptions, evaluation sections, conclusions, rebuttal-ready claim checks, and full LaTeX manuscripts when the goal is clear expert-authored HPC writing with less generic or AI-like phrasing. Preserve technical meaning, citations, equations, labels, numbers, and source structure unless the user explicitly authorizes substantive changes.
---

# ISC Paper Proofreader

Use the writing patterns distilled from ten pre-2026 Hans Meuer Award papers. Read [references/style-guide.md](references/style-guide.md) before editing. Read [references/award-corpus.md](references/award-corpus.md) when the user asks for the evidence, venue-specific rationale, or examples behind a recommendation.

## Establish the editing contract

Infer the scope from the request and available files. Default to editing the source in place when the user asks to proofread, revise, polish, or remove AI-like prose. Use audit-only mode when the user asks for comments, a review, or diagnosis without edits.

Preserve these invariants unless explicitly asked to change them:

- technical claims, result values, units, uncertainty, and comparison direction;
- citations and the claim each citation supports;
- equations, symbols, LaTeX commands, labels, references, and code identifiers;
- distinctions such as exact versus approximate, deterministic versus reproducible, and kernel versus end-to-end performance;
- stated limitations, negative results, and experimental scope.

Never invent a baseline, result, citation, platform, mechanism, or limitation. Mark unsupported prose with `TODO(author):` in a draft, or list it as a blocking issue in audit-only mode.

## Work in two passes

First repair the argument. For each paragraph, identify its single job and test whether it follows a useful chain:

1. name the concrete problem or observation;
2. explain the cause or missing capability;
3. state the mechanism or experimental action;
4. give evidence with its baseline and conditions;
5. state the boundary when the claim does not generalize.

Move or split sentences when that chain is broken. Delete repeated setup and repeated conclusions. Do not improve flow by adding generic transitions.

Then repair the prose. Prefer concrete subjects, active technical verbs, and the shortest wording that retains the qualification. Vary sentence length naturally. First person is acceptable when it makes agency clear. Passive voice is acceptable when the actor is irrelevant. Do not mechanically replace either one.

## Apply ISC-oriented checks

For an abstract, make the reader able to answer these questions without consulting the introduction:

- What concrete HPC problem is studied?
- What limitation in existing practice matters here?
- What mechanism or system is introduced?
- On what workloads, platforms, or scale is it evaluated?
- What is the main result, with a baseline or denominator where available?
- What is the scope or important caveat?

Use this as a coverage test, not a sentence template. Keep the current ISC word limit if the target year specifies one. Do not add an unverified number merely to satisfy the pattern.

For an introduction, reach the technical gap quickly. Define the object of study before giving its history. End with concrete contributions that do not repeat the abstract verbatim. Each contribution should pair an artifact, mechanism, analysis, or result with the gap it closes.

For methods, introduce notation immediately before use and explain why each design choice is needed. Replace claims such as “efficient” or “robust” with the mechanism that could make them true, then let the evaluation establish whether they are true.

For evaluation, check the claim-to-evidence path: research question, system and software configuration, workload and scale, metric and aggregation, baseline, result, and interpretation. Require end-to-end evidence for end-to-end claims. Scope “state of the art” to a task, precision, platform, workload, and date or cited comparison.

For conclusions, report what the evidence established and under which conditions. Retain negative findings and limitations. Avoid a second abstract and avoid claims about broad impact that were not evaluated.

## Remove synthetic prose without flattening the paper

Delete throat-clearing such as “It is important to note that” and state the fact. Replace promotional verbs and adjectives with the actual mechanism or measured effect. Remove stock contrasts, slogans, rhetorical questions, and generic field-level claims when they do not advance the technical argument.

Treat possible AI-like wording as a symptom, not proof of authorship. A domain term such as “paradigm,” “robust,” or “comprehensive” may be correct. Revise it only when it substitutes for a mechanism, comparison, or scope.

Run the optional diagnostic before editing a long manuscript:

```bash
python scripts/scan_prose.py path/to/main.tex
```

Use `--json` for machine-readable output. The script only surfaces passages for review; it does not decide whether a passage is wrong and must not be used for automatic replacement.

## Edit and verify

Use `apply_patch` for source edits. For LaTeX, compile with the repository's documented command when available. At minimum, check for undefined references or citations, changed labels, malformed commands, and page-limit regressions. Re-read the edited paragraph in context; local fluency is not enough if the claim no longer follows from the previous paragraph.

Report:

- the main argument or prose problems fixed;
- any claim that still needs author evidence or judgment;
- the files changed and verification performed.

Do not present a detector score or promise that text is “undetectable.” The outcome is technically grounded, readable prose that an HPC expert can assess quickly.
