---
name: omni-kernel
description: >
  Autonomous GPU kernel optimization agent that generates and optimizes kernels
  (CUDA C++, Triton, CuTe DSL, TileIR) under a rival-relative, verdict-driven
  campaign protocol. Distilled from AKO4ALL / CudaCoder / PerfBot / autoresearch
  and battle-hardened by the 6-month GVR top-K campaign record (op1-op27,
  148 commits, 28 falsifications / 12 ships). Use when: "optimize this kernel",
  "beat the incumbent implementation", "write a fast kernel for B200/H100",
  "autonomous kernel optimization".
tags: [cuda, triton, cute-dsl, tileir, optimization, autonomous, b200, h100]
license: LicenseRef-NvidiaProprietary
metadata:
  author: NVIDIA Corporation
  lineage: >
    v2 rewrite per indexer_topk_op_bench/gvr_agent_retrospective/OMNI_KERNEL_UPGRADE.md
    (II.8 change list). v1 was the generic SOL%-target loop.
---

# OmniKernel v2 — Autonomous GPU Kernel Optimization

Drive a hypothesis → probe → implement → gate → silicon-verdict → commit loop
until the ship rule is met or the remaining losses are attributed to structural
walls. The loop skeleton is cheap; what makes it work are **four mechanisms**:

1. **Trustworthy verdicts** — a four-level measurement ladder with nsys as the
   only ship arbiter, plus three a-priori vetoes.
2. **Dead ends stay dead** — a falsification ledger with a mandatory
   pre-proposal checkpoint.
3. **Session-loss immunity** — RESUME paste-blocks, idempotent progress
   markers, anchor re-calibration.
4. **Humans only at high-leverage points** — an autonomy contract signed at
   kickoff; everything else runs unattended.

**NOT the objective: SOL%.** SOL% is a diagnostic. Selection/search kernels can
be Pareto-optimal at 24% occupancy (structural, grid-limited), and a
high-SOL% kernel can still lose to a rival with a better algorithm (1 pass vs
2.5 passes). The objective is beating the incumbent on the deployment envelope
under a zero-regression ship rule.

## When this skill applies

- "optimize this kernel" / "beat <incumbent> by X%" / "make this faster on B200/H100"
- "write a high-performance kernel for <op>" with a shape specification
- "autonomous kernel optimization until converged"

**Does NOT apply when:** conceptual explanation only; model-level training
optimization (autoresearch/PerfBot); distributed training (PerfBot
distributed-specialist).

---

## Phase 0: Campaign Setup

### 0.1 Workspace scaffold

```bash
git status 2>/dev/null || git init .
git checkout -b omni/<kernel-name>
mkdir -p src scripts results templates
```

Required files (create from `templates/` in this skill dir):
- **PLAN.md** — objective triple + red lines (from the falsification ledger)
- **AUTONOMY.md** — the autonomy contract (see 0.3)
- **ITERATIONS.md** — the iteration log
- **FALSIFIED.md / WALLS.md** — campaign falsification ledger + structural walls
- **RESUME_PROMPT.md** — cross-session/cross-node relay (kept current every commit)
- **COST.md** — per-phase GPU-h + token accounting
- **reference.py** — reference implementation + `get_inputs()` (seed-policied, see Phase 4)

### 0.2 The objective triple (human-supplied; the agent MAY NOT relax it)

```yaml
objective:
  incumbent: <the current production-default implementation>   # A/B always against THIS
  rivals: [<other candidates worth beating/tracking>]
  envelope: {N: ..., K: ..., dtype: [...], BS: ...}            # deployment envelope;
                                                               # cells outside = stress probes only
  verdict_axes: [worst, real, best]        # report all three; never headline one axis
  ship_rule: "worst improves AND real/best regression-free AND exactness green
              AND dispatch rules <= 3"
  hard_constraints: [<frozen algorithm phases>, <CUDA-graph compatible>, <fail-soft>]
```

Rules learned the hard way:
- **A/B against the true incumbent**, not against your own previous variant
  (op12 nearly shipped a config that lost to the actual default).
- **Envelope rulings change conclusions** — record who set the envelope and when.
- Dispatch complexity is part of the objective, not an afterthought: a 240-key
  dispatch table is a dead deliverable even when every key wins.

### 0.3 The autonomy contract (AUTONOMY.md)

Replaces "NEVER STOP". Signed by the human at kickoff; then the agent runs
unattended inside its self-decide domain.

- **Self-decide (no asking):** parameter choices, experiment design, discarding
  negative results and moving on, checkpointing, spending within budget.
- **Must stop for a human:** shipping to production, changing baseline
  semantics, exceeding the cost budget, changing the objective/envelope.
- **Pre-authorized negative conclusion:** "if <incumbent> remains best, say so
  plainly with numbers" — written down so the agent never has to force a win.

### 0.4 Present the resolved plan

Show the objective triple, chosen language (Phase 1), probe plan (Phase 3),
budget, and the ledger red-lines before any computation.

---

## Phase 1: Characterization and Feasibility Priors

### 1.1 Detect architecture

```python
import torch
cap = torch.cuda.get_device_capability()
# (10,0) Blackwell SM100 (B200) · (10,3) B300 · (9,0) Hopper · (8,x) Ampere/Ada
```

### 1.2 Classify the operator and pick the language ONCE

Use the Arch-Strategy Matrix (`references/operator-routing.md`). Language choice
is a one-time decision revisited only as the LAST stall action — the GVR record
shows the productive search axis is primitive composition within one language,
not language hopping. (GVR: one CUDA→cuteDSL port, then 27 campaigns in cuteDSL.)

### 1.3 Roofline — kept, as a floor model

Compute-bound vs memory-bound time floors from bytes/FLOPs. Used for
feasibility, not as the success metric.

### 1.4 Feasibility priors (run BEFORE proposing directions)

Three a-priori vetoes; each is one command or one arithmetic line:

1. **L2-trap test**: if `input_bytes << L2_capacity`, every "fewer passes /
   less traffic" lever is a priori idle — the baseline's re-reads are already
   L2 hits. Verify with `ncu --metrics dram__bytes_read.sum` (one line killed
   op14 in a day). Corollary veto: if a traffic-saving variant is slower even
   with a **warm** L2, reject without further tuning (op15).
2. **Math floor**: min_passes × N × sizeof(dtype) / effective_BW. If the target
   speedup needs fewer passes than the algorithm's information floor, the
   target is infeasible — report it (op8, op10).
3. **Occupancy structure check**: if the grid at the target batch size covers
   ≪ SM count, occupancy is structural — register/pipelining levers are void;
   only more-CTAs-per-row / data-parallel forms can move it (op8 NCU proof).

**"INFEASIBLE" requires a double lock**: the math floor AND a
relaxed-constraint control experiment (drop a constraint, show even the relaxed
best misses the target — op10's cluster peaked 1.79× vs the 2× ask).

SOL% and occupancy are recorded in ITERATIONS.md as *diagnostics*.

---

## Phase 2: Exemplar Search and the Ledger Checkpoint

### 2.1-2.4 Exemplar search (unchanged in spirit)

Search the Kernel Arena, local knowledge bases, and known-good public kernels
(FlashAttention-4, DeepGEMM, FlashInfer, cutlass/examples). Adapt, don't copy.

### 2.5 Falsification-ledger checkpoint (MANDATORY, before implementing anything)

```bash
grep -il "<mechanism keyword>" FALSIFIED.md WALLS.md ../*/FALSIFIED.md 2>/dev/null
```

- On a hit: either cite the entry's **revival condition** and explain why it is
  now satisfied, or drop the idea. No third option.
- Ledger entries are **scoped triples**: `(conclusion, condition domain
  [K/N/dtype/BS/arch], evidence strength [host|event|nsys|NCU])` plus a
  root-cause class: `structural-wall | measurement-artifact | complexity-backfire`.
  Falsifications can be conditional — C8 clustering was noise at fp32 but a
  1.08-1.14× win at 16-bit. Record the domain, not just the verdict.
- This is the single highest-ROI mechanism in the source record: it repeatedly
  stopped re-proposals of the 12+2 Era-1 dead ends across 27 campaigns.

---

## Phase 3: The Probe Ladder (replaces blind 3-strategy generation)

**Never start with a kernel.** Every hypothesis climbs a cost ladder; each rung
can kill it for ~1% of the next rung's cost:

```
rung 0  CRUX experiment     — the one question the idea lives or dies on,
                              answered with any cheap tool (a Triton loop, a
                              10-line microbench). Example: "are 148 redundant
                              scans free at BS=1?" → yes (L2) → portfolio idea lives.
rung 1  HOST REPLAY         — a Python model of the kernel's control flow.
                              It earns prediction rights ONLY after replaying
                              the real kernel bit-for-bit on the full grid
                              (e.g. 720/720). Host projections are hypotheses,
                              not conclusions: expect nsys to slap them (op16 twice,
                              op26 secant2 — "pass count is not a latency proxy").
rung 2  MICROBENCH          — a standalone .cu/.py for the disputed primitive
                              (20 minutes ranked 3 compare schemes and predicted
                              the production win region for op21 iter9).
rung 3  KERNEL              — only now write/modify the real kernel.
```

GO/NO-GO is recorded at every rung in ITERATIONS.md.

### 3.2 Primitive recomposition is the productive search axis

The winning production algorithm (HLS) was a composition of six
already-verified primitives from earlier campaigns — zero new algorithms in 16
iterations. Maintain a **primitive inventory** in LEARNINGS.md (what was
verified, in which domain) and search over compositions before inventing.
Multi-hypothesis speculation is still useful — apply it to *parameters and
placement* (portfolio of thresholds/configs in one kernel), not to languages.

### 3.3 Implementation rules

- **The baseline is immutable**: new behavior enters via a gated flag
  (default-off = byte-identical) or a subclass override — never by editing the
  vendored/incumbent source. The baseline must stay recoverable in one revert.
- Compile-time keys > runtime branches: never-executed fallback code cost a
  systematic 4% fast-path tax until it was compile-gated (op21 iter14/16).
- Language quick-start templates: `references/` (Triton, CUDA C++ + binding,
  CuTe DSL) — unchanged from v1.

---

## Phase 4: Exactness Gate (GATE — MANDATORY, three tracks)

**Never benchmark an implementation that has not passed the gate.**

### 4.1 Pick the equivalence criterion by kernel class

| Kernel class | Criterion |
|---|---|
| Dense numeric (GEMM, norm, attention) | atol/rtol vs reference (fp32 1e-5, bf16 1e-2, fp16 1e-3) |
| Selection / search / sort (top-K, sampling) | **tie-aware value-multiset**: sorted output values identical + cardinality correct. NOT index equality (tie order is nondeterministic under atomics), NOT set equality (fails on duplicates at low precision) |

### 4.2 Three data tracks — all must pass

1. **Synthetic** from a validated generator with a seed policy
   (`cell_seed = f(shape)` so different cells get different draws — a constant
   seed silently tests one distribution). **torch.randn is banned** for
   low-precision selection inputs: bf16 randn collapses to ~256 levels and
   fabricates tie storms.
2. **Real captures** when available. Synth-exact ≠ real-exact: two GVR gate
   escapes (a fallback path, a pair=(0,1) route) were only ever triggered by
   real data. Synthetic performance conclusions are *upper-bound estimates*;
   real data is the verdict axis.
3. **Adversarial samples**: near-tie clusters (values 1-2 ULP apart), boundary
   padding, varlen rows, degenerate hints. op21 iter11: the adversarial track
   scored **0/72** on a kernel that was green everywhere else.

3 failed fix attempts → mark the implementation FAILED in ITERATIONS.md, move on.

---

## Phase 5: The Measurement Ladder (single code path; no ad-hoc timing)

All timing goes through `scripts/` — one entry point per rung. Discipline lives
in code paths, not prose.

| Rung | Tool | Use | NEVER |
|---|---|---|---|
| L0 | host replay / clock64 phase stamps | phase fractions, parameter screening | as a baseline or an op-vs-op claim (clock64 inflated a snap loop ~8µs and fabricated a "P4 win"; a printf'd baseline flipped an entire campaign's verdict) |
| L1 | `scripts/bench_cold.py` — cold-L2 + CUDA-graph median | full-grid sweeps | as a ship verdict (graph-launch bias 0.76-0.95×) |
| L2 | `scripts/nsys_verdict.py` — nsys pure-kernel, median of ×3 independent batches | **the only ship arbiter** | trusting a single batch (≥0.5µs variance) |
| L3 | `scripts/ncu_attrib.sh` | physical attribution only (occupancy structure, dram bytes) | optimizing to NCU numbers directly |

**Escalation triggers (mandatory):** small-N cells near the ~12-16µs launch
floor; any surprising win on L1; any ship claim. Event-axis "wins" that vanish
under nsys are the single most common artifact in the source record (≥5 cases).

**Cold-L2 is canonical** (warm-L2 understates memory-bound kernels 25-35%):
evict with a >L2 buffer write (`buf.uniform_()`) outside the timed window,
before *every* timed launch including warmup. Report warm separately (it
models fused-producer L2-hot deployments).

**Anchor protocol** (absolute numbers don't transfer across nodes/sessions):
- Nominate one anchor cell with an expected value at campaign start.
- Re-run the anchor before quoting any absolute number; drift >3% ⇒ re-baseline
  the whole grid, don't mix.
- Cross-node comparisons only via same-batch anchor-transfer ratios; report
  anchor-drift median/p10/p90 as the QA gate.
- A cross-node per-cell anomaly is transfer noise until reproduced same-node.

**Environment hygiene:** no timing on a GPU idling >50 °C (two nodes in the
record had broken cooling); detect co-tenancy by output-file growth (ps /
nvidia-smi are namespace-blind in sandboxes); paired same-process A/B for
compile-jitter immunity; a code-quality tax is real only if its sign is
consistent across independent binaries.

---

## Phase 6: Targeted Optimization (bottleneck-directed; unchanged core)

Memory-bound: vectorized access, fusion, coalescing, read-only cache.
Compute-bound: tensor cores (via CuTe DSL/TileIR on SM100 — never hand-write
tcgen05 PTX), warp specialization, software pipelining, occupancy tuning.
Selection/irregular: threshold/ladder speculation, slot-collect, dispatch by
regime. Phase-level conclusions do NOT transfer across kernel families (the
same P4 was cand-bound under snap and barrier-bound under rank-scatter — both
true). Consult `references/arch-guide-*.md`.

---

## Iteration Protocol (MANDATORY every change)

```
iter N = hypothesis (cites ledger §2.5)
       → probe rung(s) with GO/NO-GO          (Phase 3)
       → implement behind flag/subclass       (baseline byte-identical)
       → exactness three-track gate           (Phase 4)
       → L1 cold pilot → L2 nsys if ship-candidate
       → verdict ∈ {SHIP, FALSIFIED(+domain), WASH, PIVOT}
       → append ITERATIONS.md + write back FALSIFIED.md/WALLS.md
       → update RESUME_PROMPT.md + COST.md
       → git commit -s -m "[iter N] <verdict>: <one line>"
```

- Fixed verdict vocabulary. A falsified iteration is a *product*, not a
  failure — the source record shipped 12 levers off the back of 28
  falsifications, and the ledger is what kept the 28 from becoming 60.
- Backstop: before iter N+1, `git log -1` must show iter N.

### ITERATIONS.md entry

```markdown
## iter N — <date> — <VERDICT>
Hypothesis (ledger check: <hit/none>): ...
Probe: <rung, result, GO/NO-GO>
Result: <gate 3-track counts> · <L1/L2 numbers vs incumbent, per verdict_axes>
Diagnosis: <why — one mechanism sentence>
Ledger write-back: <FALSIFIED/WALLS entry added, with domain>
Next: ...
```

---

## Stall Handling (3 flat iterations)

In priority order — language switch is now LAST:

1. **Meta-analysis ops** (cheap, frame-changing; do them EARLY in a long
   campaign, not only at stalls):
   - *UB/LB bounding*: construct deterministic best/worst-case inputs; measure
     the remaining theoretical space. If UB < parity, stop tuning that axis.
   - *Favorability mapping*: sweep the data-parameter space for
     tailwind/headwind regions; they define scenario axes and ship rules.
2. **Mathematical formalization**: after an empirical plateau, formalize what
   is known (cost model with constants taken from measured campaigns; flag
   every interpolated constant and assign it a silicon-validation obligation).
   Math produces *structure*, not numbers — in the record it proved the fast
   path was already at its information floor and redirected all effort to the
   fallback tail, turning a 1.25×-mean campaign into a 2×-tail one.
3. **Primitive recomposition** over the LEARNINGS.md inventory.
4. Re-profile with `--set full`; hunt for a missed bottleneck.
5. Switch language (last resort; requires evidence the gap is language-level).

**Explore → productize switch signal**: when every residual loss is attributed
to a named structural wall (config-insensitive, mechanism understood), stop
exploring. Rewrite the objective with production constraints (dispatch-rule
budget, graph compatibility, fail-soft, real-capture exactness) and harvest.

---

## Convergence and Final State

- **CONVERGED**: ship_rule met on the envelope, verdicts from L2, gates green.
- **STOP (infeasible)**: double-locked per Phase 1.4 — report plainly; the
  negative conclusion is pre-authorized.
- HEAD ends at the best-performing iter (restore from git if needed).
- Keep: winning src/, ITERATIONS.md, FALSIFIED.md, WALLS.md, LEARNINGS.md,
  scripts/, RESUME_PROMPT.md, COST.md. Failed attempts stay in git history.

---

## Cross-Session, Multi-Node, and Ops Hygiene

- **RESUME_PROMPT.md is a mandatory deliverable**, refreshed at every commit.
  Five-part shape: 1-minute context / preflight checklist (git HEAD, env, GPU
  thermal blacklist, no co-resident driver, progress-marker count) / disjoint
  work split / byte-exact `setsid` launch commands / known gotchas. ~1/3 of the
  source record's handoffs were node-loss insurance — every one paid out.
- **Idempotence granularity is a design-time decision**: done-markers per
  batch, append-only jsonl per cell, parse caches per report file. Recovery
  from any incident = delete marker, re-issue the same command.
- **Launch/stop discipline**: long runs start as a single
  `setsid env ... > named.log &` line (no `&&` chains, no loops around it).
  Stop with a pkill-triple on driver/sweeper/profiler + a 30s respawn re-check
  + PGID kill for stragglers; task-runner stop buttons do NOT kill process
  trees (two dual-driver data-corruption incidents). Re-issued shard commands
  must copy env blocks whole and verify the log's header line (a missing env
  filter wasted 4 GPU-h).
- **Token hygiene**: profilers embed the process env in artifacts —
  `env -u GITHUB_TOKEN -u HF_TOKEN nsys ...`; gitignore `*.sqlite *.nsys-rep`
  BEFORE the first results commit (a live token was pushed publicly once).
- **COST.md**: per-phase GPU-hours and token spend. Calibration anchor from the
  record: one mid-size campaign ≈ 15 GPU-h + ~$108; the op21 flagship ≈ $797.
- Checkpoint analysis continuously: heredoc one-offs become committed .py
  files; conclusions become committed .md — never wait for campaign end.

### Long-lived multi-arm arena (enable when >3 candidates accumulate)

One shared report where every candidate ("arm") is timed on byte-identical
bundles against a co-located anchor arm; backfilled arms anchor-transfer with
drift quantiles as the QA gate. Report updaters are **self-contained
last-writers**: each re-derives ALL existing arms from raw roots before adding
its own, states successor obligations in its docstring, machine-checks
invariants (blob count, row counts), and backs up the report first. Never an
incremental patch updater. Shard nodes run and stop; one coordinating session
does parse/update/commit.

---

## Anti-Patterns (NEVER DO) — includes the measurement-artifact catalog

1. Reward hacking (timing tricks, pre-computed results).
2. Optimizing without measuring; guessing bottlenecks.
3. **Event-only ship claims** — the event axis fabricated ≥5 reproducible lies
   in the source record.
4. **Instrumented baselines** — clock64/printf builds are diagnosis-only.
5. **torch.randn for low-precision selection inputs**; constant seeds across cells.
6. **A/B against anything but the true incumbent.**
7. Editing the vendored/incumbent source instead of flag/subclass.
8. Warm-L2 tight-loop numbers reported as canonical.
9. Quoting absolute µs across nodes/sessions without the anchor protocol.
10. Dispatch tables that grow past the ship rule's budget (240-key cautionary tale).
11. Batch-committing; skipping the ledger write-back.
12. `.item()`/hidden syncs in wrappers; non-fresh `get_inputs()`.
13. Stopping a sweep with the task-runner button and relaunching immediately
    (dual-driver corruption); launching long runs without `setsid`.
14. Trusting a cross-node per-cell anomaly without a same-node re-test.

---

## Simplicity Criterion (unchanged)

Within 5% of target and comparable performance → prefer the simpler
implementation. A 0.5% gain that doubles complexity is a loss. Code that never
executes still costs (icache/codegen tax) — delete or compile-gate it.

---

## Reference Files

- `templates/PLAN.md`, `templates/AUTONOMY.md`, `templates/FALSIFIED.md`,
  `templates/WALLS.md`, `templates/RESUME_PROMPT.md`, `templates/COST.md`
- `scripts/bench_cold.py` — L1: cold-L2 + CUDA-graph sweep timing
- `scripts/nsys_verdict.py` — L2: nsys ×3-median ship arbiter + anchor check
- `scripts/ncu_attrib.sh` — L3: attribution metrics (occupancy, dram bytes)
- `scripts/verify_exact.py` — Phase 4 gate (atol/rtol + tie-aware multiset modes)
- `scripts/benchmark.py`, `scripts/verify.py` — v1 quick tools (superseded for
  verdicts; fine for smoke tests)
- `references/arch-guide-b200.md`, `references/ncu-metrics.md`
- `LEARNINGS.md` — seeded with the GVR campaign ledger (M/C/S/E)
- Full provenance: `indexer_topk_op_bench/gvr_agent_retrospective/`
  (RETROSPECTIVE.md + OMNI_KERNEL_UPGRADE.md)
