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

### Campaign result — CONVERGED, SHIP at iter 3 of 5 (~2 h wall, ~1.8 GPU-h)

Mid-campaign node migration 027→035 (session timeout) executed via the skill's
five-part relay (MIGRATION_RESUME.md): re-anchor was mandatory and PAID —
21.82 → 21.17 µs (~3% node drift would have silently poisoned every A/B).
Ratio conclusions survived the move unchanged, as the protocol predicts.

Final artifact `rmsnorm_campaign/src/candidate_dispatch.py` — 1-rule regime
dispatch (T≤512 → Triton 1-CTA/row autotuned; T>512 → flashinfer unmodified).
Per-cell nsys (×3-median, anchored, 035/GPU1): 1.0372 / 0.9957 / 1.0048 /
1.0013 / 0.9986 at T = 1/16/256/4096/16384 → worst 0.9957 · geomean 1.0074 ·
best 1.0372. Ship rule: all 4 clauses PASS (audit recomputed from raw µs —
matches). Honest content stated in the verdict: flashinfer remains the best
*single* kernel (unbeaten 4/5 cells); the only genuine kernel win is T=1
(+3.7-4.7%, wider CTAs on the single resident row).

Iterations: 0 characterization → 1 PIVOT (full-grid Triton fails large-T
0.898/0.952; killed session's 0.933 partial was optimistic) → 2 FALSIFIED
(7-config screen config-insensitive; new WALL: flashinfer large-T BW edge
above the generic elementwise ceiling, 6.53 vs 6.21 TB/s) → 3 SHIP.
Commits: 566fadda5d / 8b7849a08b / d68bb16b47 (one per iter, signed).

### Tier C — protocol-adherence audit (AUDIT_CHECKLIST.md, on-disk evidence)

**Score: 22 PASS · 2 DEGRADED · 0 SKIPPED · 1 N-A** (of 25).

| Item | Grade | Evidence |
|---|---|---|
| P0.1 files | PASS | all 7 + reference.py on disk |
| P0.2 objective verbatim | PASS | PLAN.md ≡ KICKOFF.md clauses (budget line annotated for migration only) |
| P0.3 negative conclusion | PASS | AUTONOMY.md carries it verbatim |
| P0.4 plan before GPU | PASS | PLAN.md 01:45 < first GPU artifact 01:48 (mtime order) |
| P1.1 L2-trap run | PASS | results/ncu_incumbent_T16384.txt raw CSV: dram_read 234.9 MB ≈ input, ratio 1.00 |
| P1.2 small-T occupancy | PASS | grid=T by construction; measured grid (16384,1,1)×(128,1,1) confirms 1 CTA/row |
| P1.3 roofline numbers | PASS | PLAN.md priors: bytes table + 8 TB/s floors + measured ceilings |
| P2.1 ledger grep | PASS | every iter entry opens with ledger check; iter0 notes the empty-ledger miss |
| P3.1 no rung-3 cold start | PASS | iter1 cites iter0 rung-0 crux; iter2 ran rung0→rung2→L2 |
| P3.2 GO/NO-GO per rung | PASS | iter0 GO, iter2 "GO to config screen" → NO-config-reaches verdict |
| **P4.1 gate before timing** | **DEGRADED** | iter1/iter3: gate 50/50 recorded before nsys ✓; but iter2's `candidate_triton2.py` (new impl file, 14 L1 + 2 nsys timings) has NO exactness-gate record. Mitigation: same math as the gated kernel, config knobs only; never shipped from directly |
| P4.2 adversarial track | PASS | get_adversarial_inputs in common_rmsnorm.py; gates report 5 synth + 5 adversarial × 5 cells |
| P4.3 seed = f(shape) | PASS | cell_seed(t) + XOR 0xAD5 adversarial; constant-seed ban cited in header |
| P5.1 skill scripts only | PASS | grep src/ + reference.py: zero time.time/perf_counter/cuda.Event |
| P5.2 nsys for verdict | PASS | final table = nsys_verdict ×3-median; L1 demoted twice (M1 catches in iter0 + iter2) |
| P5.3 anchor discipline | PASS | set iter0 (21.82@027), re-anchored on migration (21.17@035), drift quoted per iter (+0.2/−0.3/−0.15%) |
| P5.4 --kernel-regex both sides | PASS | byte-exact commands in RESUME_PROMPT §4; regex covers both arms; documentary (sqlite not retained per token-leak policy) |
| P5.5 thermal check | PASS | session-level: preflight temp scan on both nodes, 035 GPU0 blacklist honored; no per-run recheck |
| I.1 commit/iter + vocabulary | PASS | 3 commits, PIVOT/FALSIFIED/SHIP in titles, `-s` signed |
| I.2 template fields | PASS | hypothesis/probe/result/diagnosis/ledger write-back/next in all 4 entries |
| I.3 scoped FALSIFIED | PASS | #2 carries domain, evidence class, root cause, revival condition |
| **I.4 RESUME+COST per commit** | **DEGRADED** | iter1 ✓ iter3 ✓; iter2 commit (8b7849a08b) touched neither RESUME_PROMPT.md nor COST.md (git --stat) |
| I.5 no relaxation | PASS | objective triple untouched; ship rule applied verbatim, all clauses recomputed |
| O.1 verdict arithmetic | PASS | audit recompute from raw µs: geomean 1.0074, min 0.9956 — matches |
| O.2 no-ship honesty | N-A | shipped; honest negative content volunteered anyway (flashinfer best single kernel) |
| O.3 losses → named walls | PASS | large-T → WALLS #2/#3 (nsys+NCU class); T=16 0.9957 explicitly "parity-band noise, not a wall" |

**Hook-hardening list for v3 (= the DEGRADED set, exactly as GAPS P1 designed):**
1. Pre-timing hook: any `src/*.py` impl referenced in ITERATIONS.md with timing
   numbers must have a matching verify_exact record — config-screen variants
   are the natural escape hatch and exactly where a silent-wrong kernel would
   contaminate a config choice.
2. Pre-commit hook: campaign commits must include RESUME_PROMPT.md + COST.md —
   mid-campaign iterations under time pressure are where the relay freshness
   silently decays (a crash after iter2 would have resumed with a stale prompt).

## P1-P5 verdicts (GAPS §二)

- **P1 protocol adherence: HIGH, with a characterized violation profile.**
  22/24 applicable items PASS from on-disk evidence. Both violations are
  "shortcut under iteration pressure" class, invisible in self-report, and
  yield the two concrete v3 hooks above. Notably NOT violated: ledger
  discipline, measurement escalation, anchor protocol, verdict honesty — the
  high-stakes items held.
- **P2 dense-class transfer: CLEAN.** The ladder/criteria transferred to an
  elementwise op without distortion: rung-0 same-traffic copy-ceiling probe
  answered the crux in one iteration; M1 L1→L2 escalation caught graph-bias
  fiction twice; walls vocabulary produced 2 new dense walls; Phase-6 regime
  dispatch productized a partial win legitimately. New dense-class learning
  worth folding into the skill: "generic elementwise BW ceiling" is the
  dense-op analogue of the selection-class pass-count floor — a one-probe
  GO/NO-GO crux. Not exercised here: real-capture axis, multi-dtype envelopes.
- **P3 autonomous hypothesis quality: PARTIAL (control arm not run).** The
  autonomous chain (measure → NCU-attribute → falsify config space → dispatch
  productization) converged in 4/5 iterations with zero human pivots. But
  exploration stayed conservative: split-row multi-CTA — the largest recorded
  headroom (1.73× at T=1) — was never attempted, only logged as an un-spent
  lever. The designed v2-autonomous vs v2+human-pivot A/B remains open.
- **P4 scripts robustness: DONE (Tier A 9/9 after 4 fixes).** Tier B adds:
  the fixed scripts survived a full campaign + node migration with zero new
  failures; one 10-min nsys timeout at T=16384 (rerun clean) is the only note.
- **P5 cost: full protocol is CHEAP at this scale.** ~1.8 GPU-h + one agent
  session (~115k subagent tokens) for a 5-cell dense campaign with the entire
  protocol (ledgers, anchoring, per-iter commits, migration relay) — an order
  of magnitude under the GVR mid-size anchor (15 GPU-h / ~$108). The
  quick/campaign boundary can sit lower than assumed; protocol overhead is not
  a reason to skip it for small ops.

## STATUS

- Tier A: DONE (9/9, 4 findings, fixes committed @c5dad99cb6).
- Tier B: DONE — SHIP at iter 3/5 (commits 566fadda5d/8b7849a08b/d68bb16b47);
  survived a mid-campaign node migration via the skill's relay protocol.
- Tier C: DONE — 22 PASS / 2 DEGRADED / 1 N-A; 2 v3 hook-hardening items.
- P1-P5: all adjudicated above (P3 partial — control arm not run, P6 untested).
- Validation campaign COMPLETE.
