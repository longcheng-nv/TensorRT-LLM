# omni-kernel v2 skill validation — experiment report

> 2026-07-13, umbriel-b200-027 (8× B200 idle, 33-39 °C). Validates the v2 skill
> (`gvr_agent_retrospective/skill_v2_draft/`, live via `.claude/skills/omni-kernel`)
> against the practice-only questions in OMNI_KERNEL_V2_GAPS.md §二 (P1-P5).
>
> Design: **Tier A** = direct pos/neg validation of the 4 measurement/gate
> scripts (P4); **Tier B** = bounded mini-campaign on a dense-class op with a
> mature production incumbent, then protocol-adherence audit (P1/P2/P5).

## Tier A — scripts robustness (P4) — DONE, 9/9 after fixes

Harness: `tierA/run_tierA.sh`. Every script tested in BOTH directions — a gate
that cannot fail is not a gate. Ops: Triton RMSNorm (dense, DSv4 hidden=7168,
bf16, 235 MB working set > L2) + tie-rich top-K (selection, N=262144 K=2048,
64 discrete levels). Negative controls: +5% scaled RMSNorm; duplicate-index topk.

| Test | Expectation | First run | After fix |
|---|---|---|---|
| A1 verify_exact dense (good impl) | PASS | OK | OK |
| A2 verify_exact dense (broken impl) | must FAIL | OK | OK |
| A3 verify_exact select, tie-rich (index sets legitimately differ) | PASS via value-multiset | OK | OK |
| A4 verify_exact select (duplicate index) | must FAIL | OK | OK |
| A5 bench_cold A/B | runs, cold ≥ warm | OK | OK |
| A6 nsys_verdict A/B | numbers coherent with A5 | **ran, but numbers wrong** (bug 1) | OK |
| A7 nsys_verdict anchor drift | must REJECT + refuse numbers | OK | OK |
| A8 ncu_attrib with INPUT_BYTES | structural verdicts printed | **rc 0 but verdicts silently missing** (bug 2) | OK |
| A9 ncu_attrib without INPUT_BYTES | runs (var documented optional) | **crash, set -u** (bug 3) | OK |

### Bugs found (all in the GAPS-P4 predicted class) — fixed in skill_v2_draft @c5dad99cb6

1. **nsys_verdict.py: naive `line.split(",")` on nsys CSV.** Kernel template
   names contain commas → `cols[-1]` is a name *fragment* → the evictor filter
   ("uniform"/"distribution") never matched and `--kernel-regex` matched
   fragments. Consequence: with the default regex, the L2-evictor kernel
   (~180 µs, 69.5% of the profiled window) was silently ADDED to every
   measurement — the L2 ship arbiter itself was lying. The candidate showed
   259 µs vs its true 80 µs; the L1/L2 "bias" was 0.33× vs the documented
   0.76-0.95×, which is what exposed it. Fix: `csv` module + thousands-sep
   tolerance. Post-fix bias: 0.95 (candidate), 0.97 (baseline) — in band.
2. **ncu_attrib.sh: structural verdicts were dead code.** `python3 - <csv <<EOF`
   — the heredoc overrides the file redirect, `python3 -` consumes stdin as the
   program, `csv.reader(sys.stdin)` reads EOF → the L2-trap/occupancy verdicts
   (the entire point of L3) never printed, rc still 0. Fix: csv path via argv.
3. **ncu_attrib.sh: `set -u` crash** when INPUT_BYTES unset despite the header
   documenting it optional. Fix: `${INPUT_BYTES:-}`.
4. **Footgun (mitigated): verdict block reads the FIRST profiled kernel.** A
   setup `randn` kernel produced a confident false "L2-TRAP: traffic levers
   VOID" verdict (dram_read 1.7e5 B vs the real kernel's 2.35e8 B). Added
   `KERNEL_REGEX` env → `ncu -k`. Post-fix: ratio 1.00, grid 16384 — correct.

### Tier A meta-conclusions

- The **criteria design validated**: tie-aware value-multiset correctly PASSes
  legitimate tie-boundary index divergence (A3) that index-equality would fail,
  while still catching a duplicate-index escape (A4). Anchor protocol actually
  refuses to emit numbers on drift (A7), not just warns.
- The most dangerous failure mode was **silent** (bugs 1, 2, 4 all produced
  rc 0 + plausible-looking output). Supports GAPS G4's thesis: the artifacts
  that lie are exactly the ones a single agent trusts. Script self-tests of
  this pos/neg form should ship WITH the skill (proposal for v2.1: a
  `scripts/selftest.sh` requiring a known-good/known-bad pair).

## Tier B — mini-campaign (P1 protocol adherence, P2 dense-class transfer, P5 cost)

Setup: `tierB/KICKOFF.md` (objective triple: incumbent =
flashinfer.norm.rmsnorm 0.6.11, DSv4 hidden 7168, token grid 1→16384, bf16;
ship rule geomean ≥1.00 & no cell <0.98 & gate green & ≤3 dispatch rules;
budget 5 iters / 2 h; pre-authorized negative conclusion). Audit rubric:
`tierB/AUDIT_CHECKLIST.md` — 25 items scored PASS/DEGRADED/SKIPPED against
on-disk evidence.

*Campaign in flight — results below appended on completion.*

## STATUS

- Tier A: DONE (9/9, 4 findings, fixes committed @c5dad99cb6).
- Tier B: campaign agent running (workspace `tierB/rmsnorm_campaign/`).
- Next: audit Tier B vs checklist → P1-P5 verdicts → v2.1 change list.
