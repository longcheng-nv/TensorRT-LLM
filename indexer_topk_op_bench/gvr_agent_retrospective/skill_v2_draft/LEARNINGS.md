# OmniKernel Learnings — Cross-Campaign Knowledge Base

Seeded 2026-07-12 from the GVR top-K campaign record (6 months, Era-0/1/2,
op1-op27, 148 commits, 28 falsifications / 12 ships). Full provenance:
`indexer_topk_op_bench/gvr_agent_retrospective/{RETROSPECTIVE.md,RETROSPECTIVE.html}`.
Update per campaign: append findings, never rewrite history.

## M — Measurement (all executable; encode in scripts/, not prose)

- **M1 The timing ladder**: host-replay/clock64 (diagnosis) → cold-L2 graph
  event (`bench_cold.py`, sweeps) → nsys ×3-median (`nsys_verdict.py`, the ONLY
  ship arbiter) → NCU (`ncu_attrib.sh`, attribution). Escalate on: small-N
  cells near the ~12-16µs launch floor, any suspicious L1 win, any ship claim.
- **M2 Cold-L2 is canonical** (warm understates memory-bound kernels 25-35%);
  the warm-L2 A/B is a one-shot veto for any traffic-saving lever.
- **M3 Instrumented kernels never serve as baselines** (a printf'd baseline
  flipped a whole multi-CTA campaign; clock64 inflated a snap loop ~8µs and
  fabricated a "P4 win"). Instrumented data = phase fractions only.
- **M4 Headline numbers only from the production call surface** (same SASS was
  20.9-34.4% slower through a standalone JIT harness — different preIdx
  semantics changed the seed quality). A/B only against the true incumbent.
- **M5 Anchor protocol**: run the anchor cell before quoting absolutes; >3%
  drift ⇒ re-baseline. Cross-node comparisons only via same-batch
  anchor-transfer ratios with drift quantiles as the QA gate; a cross-node
  per-cell anomaly is noise until reproduced same-node.
- **M6 Environment hygiene**: no timing on GPUs idling >50 °C; co-tenancy
  detection via output-file growth (ps/nvidia-smi are namespace-blind); paired
  same-process A/B; a code-quality tax is real only if its sign is consistent
  across independent binaries.
- **M7 One-line L2-trap test**: `ncu --metrics dram__bytes_read.sum` vs input
  bytes. Input ≪ L2 ⇒ all "fewer passes" levers a priori void (killed 3
  campaigns: op14/15/16-sampling).

## C — Correctness

- **C1** Exactness gate is up-front, not post-hoc. Selection kernels:
  tie-aware value-multiset (sorted values equal + cardinality), never index
  equality, never set equality.
- **C2** Three tracks: synth + real captures + adversarial (near-tie clusters,
  boundary padding, varlen). The adversarial track scored 0/72 on a kernel
  green everywhere else; two escapes were real-data-only.
- **C3** torch.randn banned for low-precision selection inputs (bf16 collapses
  to ~256 levels → tie storms); per-cell seed policy (`seed=f(shape)`).
  Synthetic perf conclusions are upper-bound estimates; real data is the
  verdict axis.

## S — Search strategy

- **S1** iter0 measures the mechanism ceiling before any kernel:
  crux → host-replay → microbench → kernel. Host models earn prediction rights
  only after bit-exact replay of the real kernel; expect nsys to overturn host
  projections (pass count is NOT a latency proxy — loop overhead beat the
  saved passes in op26-secant2).
- **S2** The falsification ledger is first-class: mandatory pre-proposal grep;
  entries are scoped triples (conclusion, condition domain, evidence strength)
  — a verdict can flip across dtype (C8: noise at fp32, 1.08-1.14× win at 16-bit).
- **S3** Composing verified primitives out-produces inventing algorithms: the
  shipped HLS = six primitives from five earlier campaigns; 16 production
  iterations contained zero new algorithms.
- **S4** INFEASIBLE = math floor + relaxed-constraint control, double-locked.
- **S5** Meta-analysis ops early: deterministic UB/LB bounds (how much
  theoretical space is left) + favorability mapping (which cells are worth
  chasing; parameters can be non-monotonic in speedup).
- **S6** After an empirical plateau, formalize: a cost model with
  measured-campaign constants (flag every interpolated constant with a
  silicon-validation obligation). Math produces structure, not numbers — it
  redirected a 1.25×-mean campaign to a 2×-tail one.
- **S7** Dispatch-by-regime is the usual shape of a shippable answer, but its
  complexity belongs in the objective (a 240-key table is dead on arrival;
  ship rules cap it, e.g. ≤3 rules; "dispatch rules stop where the data stops").
- **S8** Explore→productize switch: when every residual loss maps to a named
  structural wall. Phase-level conclusions do not transfer across kernel
  families (the same phase was cand-bound under one P4 and barrier-bound
  under another — both true).

## E — Engineering & organization

- **E1** Discipline lives in single code paths (one timing function, one nsys
  wrapper) — every campaign imports, none re-implements.
- **E2** Every iter: hypothesis→probe→implement(flag/subclass; baseline
  byte-identical)→gate→verdict {SHIP|FALSIFIED+domain|WASH|PIVOT}→ledger
  write-back→commit. Never batch commits.
- **E3** Idempotence granularity at design time (done-markers/jsonl/parse
  caches); `setsid` single-line launches; stop = pkill-triple + respawn
  re-check (task-runner stops don't kill process trees — two dual-driver
  corruption incidents); re-issued env blocks pasted whole + header-line check.
- **E4** RESUME_PROMPT.md refreshed every commit (five-part paste-block);
  shard nodes run-and-stop, one coordinator parses/updates/commits.
- **E5** Long-lived reports: self-contained last-writer updaters (re-derive
  all arms, docstring successor obligations, machine-checked invariants, .bak
  first). Never incremental patch updaters.
- **E6** Profiler artifacts embed the env → `env -u GITHUB_TOKEN -u HF_TOKEN`;
  gitignore *.sqlite/*.nsys-rep before the first results commit (a live token
  was once pushed publicly). COST.md per phase (mid campaign ≈ 15 GPU-h +
  ~$108; flagship ≈ $797).
- **E7** Checkpoint continuously: heredocs → committed .py; conclusions →
  committed .md; never wait for campaign end.

## Primitive inventory (verified, with domains — grow this per campaign)

| Primitive | Verified domain | Source |
|---|---|---|
| Speculative threshold portfolio, winner reuses its own smem counts | selection, single cooperative kernel, BS≤16 | GVR op17 |
| M-ary ladder single-pass counting + CDF-aware rung placement | selection, M≤4 (rungs are currency: M=5 taxed +7-19%) | GVR op18/op25 |
| Sandwich pair (guaranteed direct-writes + band refine) | selection, large N | GVR op19 |
| Fused count+collect slot pass | selection, large N | GVR op20 |
| Exact rank-scatter (histogram→prefix→scatter, fixed 256-bin fine level) | P4-class barrier-bound selection | GVR op7/PR#15709 |
| Remote-store push (st.shared::cluster into leader smem, kills gather pass) | cluster kernels | GVR op21 iter7 |
| Log-domain regula-falsi root aim from ladder-known counts | threshold fallback tails | GVR op13/op21 iter13 |
| Row-chunked C-CTA data-parallel decomposition | single-row long-scan, occupancy-walled | GVR op21 iter2 |
| Replicated seeding (recompute per CTA; L2 makes copies free, barriers don't) | cluster kernels | GVR op21 iter3 (dist-P1 falsified) |
| Compile-time gating of fallback code mass (n in the compile key) | any kernel with rarely-taken paths (4% icache tax) | GVR op21 iter16 |

## Architecture Notes (B200/SM100) — retained from v1

- Compile with `TORCH_CUDA_ARCH_LIST=10.0` / `sm_100`; B300 is compute 10.3 (`103-real`)
- CuTe DSL requires `nvidia-cutlass-dsl`; TileIR requires `nv-triton` (NVIDIA PyPI)
- tcgen05 MMA is 2-SM cooperative — use CuTe DSL / TileIR, never hand-write PTX
- TMEM is 512 KB/SM; tcgen05 accumulates there (no SMEM pressure for the accumulator)
- Triton works well for elementwise/reduction on B200 without special tuning
- BF16 preferred over FP16 (native); FP4/FP6 via tcgen05 for peak throughput;
  TF32 default for torch.mm / tl.dot (~1e-2 abs drift vs FP32 expected)
- Occupancy on SM100 large tensors: >33% is often enough; check structure first (M7/WALLS)

## Current Best / Open Questions

*Per-campaign sections start here — append below, one block per campaign.*
