# TensorRT-LLM Top-K / DSA Indexer Skills

Claude Code skills for developing, benchmarking and validating the DeepSeek
sparse-attention (DSA) indexer Top-K path in TensorRT-LLM (GVR / heuristic
Top-K vs Radix Top-K), plus supporting model-staging and onboarding tooling.

**How skills are used.** Each subdirectory is a self-contained skill: a
`SKILL.md` (frontmatter `description` drives automatic invocation inside
Claude Code, and the body is the operating manual) plus optional `src/`
scripts and `assets/` data. Two ways to use them:

1. **Agent-driven** — open Claude Code in a checkout that carries
   `.claude/skills/` and phrase a request matching a skill's trigger keywords;
   the agent loads the skill and follows it.
2. **Direct CLI** — most data/bench skills ship plain Python entry points
   under `src/` that run standalone (documented per skill below).

Unless stated otherwise, GPU-side scripts target NVIDIA Blackwell
(B200 sm_100 / B300 sm_100a) and the PyTorch backend of TensorRT-LLM.

---

## 1 · Synthetic input generation

### `indexer-topk-temporal-synth`  ← recommended
Unified generator of realistic decode **logits + temporally-coherent preIdx**
for the indexer Top-K operator, covering **DeepSeek V3.2 (K=2048, cr=1),
V4-Flash (K=512, cr=4) and V4-Pro (K=1024, cr=4)**. Supersedes the three
`swebench-temporal-synth*` skills below.

- Marginal: per-layer **empirical inverse-CDF + GPD tail** calibrated from
  real 64K production captures (fixes the legacy single-Beta tail collapse:
  synthetic mass at the real top-K selection boundary was 0.00× at N≥128K,
  now 0.99–1.15×). Default `--cfg aggregate` mixes rows over the real layer
  family.
- Temporal: rank-conditional retention curve + real miss-depth samples +
  per-row hit-rate sampled from the real per-step distribution; optional
  Gaussian-copula AR(1) multi-step chain (`--steps T`).
- Calibration assets (~1 MB/model) are committed; synthesis needs no access
  to the raw captures. Kernel contracts (preIdx offset, padding, seq_lens,
  radix_aux) are preserved per model.

```bash
SKILL=.claude/skills/indexer-topk-temporal-synth
python3 $SKILL/src/synth_temporal_data.py --model v4flash --N 64K \
    --cfg aggregate --bs 8 --outdir /tmp/synth_out
# nsys GVR-vs-Radix on the generated bundles:
#   src/bench_nsys.py + src/parse_nsys.py (see SKILL.md)
# refresh calibration from new captures: src/calibrate_from_real.py
# acceptance gates (5 gates × 3 models): src/validate_against_real.py
```

### `dsv4pro-indexer-synth`
Synthesizes the **full indexer kernel input tensor set** — Q (FP4), K-cache
(FP4), weights (fp32), logits (fp32), topK (int32), preIdx (int32) — for
DSV4 **Pro (K=1024)** or **Flash (K=512)** at any BS/ISL/OSL, via a
Logits-First + Rank-Transform + Temporal-Bias algorithm calibrated from real
SWE-bench 64K B300 captures. Use this when benchmarking the whole indexer
kernel (not just the Top-K stage).

```bash
python3 .claude/skills/dsv4pro-indexer-synth/src/synth_indexer_inputs.py --help
```

### `swebench-temporal-synth`, `-v4flash`, `-v4pro`  (superseded)
Legacy per-model generators: single moment-matched Beta marginal +
iid-Gaussian-noise preIdx calibrated to a fixed scalar hit-rate
(V3.2 / V4-Flash / V4-Pro respectively). **Superseded by
`indexer-topk-temporal-synth`** — a validation study showed the Beta marginal
flattens the real heavy positive tail exactly where top-K selects (see the
banner in each SKILL.md). Kept for provenance and for reproducing historical
benchmark numbers.

---

## 2 · Real-data capture & kernel benchmarking

### `dsv4-indexer-capture`
Drives a single-prompt, BS=1, greedy end-to-end DSv4 inference and dumps
per-layer **(indexer logits, indexer top-K)** from the real production GVR
path (hooked `dsa.py`; capture kept in CPU RAM, atexit-flushed).
Parameterized over prompt, model variant (Flash/Pro/raw path), phase
(prefill/decode), OSL, layer subset, and save format. Data-collection only —
not a perf benchmark (`cuda_graph_config=null`, greedy). This is the source
of the captures that calibrate the synthesis skills above.

```bash
bash .claude/skills/dsv4-indexer-capture/src/launch_capture.sh --help
```

### `gvr-vs-radix-swebench`
End-to-end performance comparison of the **GVR (heuristic-guided) Top-K
kernel vs the TRT-LLM Radix Top-K kernel on real data**: the 9-layer
SWE-Bench-64K decode-logits dataset (DeepSeek V3.2). Produces nsys-measured
per-row kernel times and an English report covering BS=1 (real
variable-length rows) and BS scaling 1..512, with per-BS min/max/mean
speedups. Prefer this (or the V4 captures) over synthetic data for
tail-sensitive final verdicts.

---

## 3 · End-to-end benchmarking, profiling & accuracy

### `dsv4-pareto-bench`
The comprehensive playbook for benchmarking **DeepSeek V4 Flash and Pro on
Blackwell**: dispatches (hardware × model × dataset) into cluster-native
sbatch sweeps or single-node resumable drivers; canonical YAML/CLI defaults,
per-(HW, model) `max_num_tokens`/KV-fraction tables, MTP/BS sweep recipes,
GVR ON/OFF A/B pairing, known gotchas (G1–G8: DEP hangs, OOM patterns,
CUDA-graph interactions) with recovery procedures, and Pareto-frontier
reporting. Agent-driven; start by asking e.g. *"run Flash throughput sweep on
B300, synthetic dataset"*.

### `dsv4-nsys-profile`
Single-shot **Nsight Systems decode-phase profiling** for DSv4 Flash/Pro:
wraps `trtllm-bench` in an nsys `cudaProfilerApi` window so the report
contains only the requested steady-state decode iteration range. Auto-detects
model path, MoE backend, TEP/DEP, MTP and the GVR Top-K toggle, and renders
the same engine config as `dsv4-pareto-bench` so kernel comparisons land on
identical engine state.

### `dsv4-gsm8k-eval`
**GSM8K accuracy evaluation** for DeepSeek-V4 (Flash / Flash-Base / Pro) with
MTP speculative decoding + CUDA graphs on 8× Blackwell via `trtllm-eval`:
env preflight (fast-hadamard-transform, transformers pin), background launch,
progress polling, final strict-match / flexible-extract scores. Use to check
that kernel/feature changes did not regress accuracy.

---

## 4 · Kernel integration & model onboarding

### `automerge-trtllm-topk`
Integrates a **standalone single-CTA heuristic Top-K micro-kernel**
(`heuristic_topk.cuh`) into TensorRT-LLM's DSA indexer decode path: extends
it to multi-row (multi-batch, next_n>1), wires the thop operator plumbing,
keeps CUDA-Graph compatibility, and runs end-to-end tests. Use when a new or
modified micro-kernel needs productization.

### `ad-model-onboard`
Translates a HuggingFace model into a **prefill-only AutoDeploy custom
model** using reference custom ops, and validates it with hierarchical
equivalence tests. Input: HF model ID; output: custom model file + tests +
summary report.

---

## 5 · Infrastructure

### `computelab-hf-stage`
Stages large model weights (DeepSeek-V4 Pro/Flash, Llama-405B, …) **directly
from the HF CDN to local fast storage** (`/raid` NVMe or `/dev/shm`) on
computelab nodes at ~0.5–2 GB/s via `hf_transfer` + a 16-worker pool —
bypassing NFS cold-reads (hours) and slow cross-host copies. Use after any
fresh host login / container restart / reboot, before launching benchmarks.

---

## 6 · Paper & writing tooling

### `de-ai-flavor-paper`
Detects and removes **LLM-flavored prose** from systems/architecture paper
manuscripts (LaTeX/Markdown) by comparing against quantitative style
baselines distilled from six pre-2022 PPoPP Best Papers (~74k words of
human-expert prose). Two detection families: classic lexical markers
(`comprehensive/novel/moreover/…` — target 0) and the harder structural
tells (mantra repetition, "X, not Y" antithesis chains, template sentences,
low first-person density, uniform caption/bullet geometry, Title-Case
concept branding). Style-only revisions — never changes numbers, claims, or
evidence scope, and keeps venue-required generative-AI disclosure intact.

```bash
SKILL=.claude/skills/de-ai-flavor-paper
$SKILL/scripts/detect_ai_flavor.sh path/to/main.tex   # scorecard w/ targets
# full corpus norms + quoted examples: $SKILL/reference.md
```

---

## Conventions & gotchas

- **Never commit nsys artifacts** (`*.sqlite`, `*.nsys-rep`): they embed the
  full process environment, including tokens. Profile with
  `env -u GITHUB_TOKEN -u HF_TOKEN nsys profile …`.
- Kernel micro-benchmarks quote **cold-L2, nsys-projected** kernel times
  (128 MiB L2 flush before every rep); in-process `cuda.Event` walls include
  launch latency and understate GVR-vs-Radix speedups.
- Synthetic data is for controllable sweeps over arbitrary (N, BS, seed,
  dtype); **real-data paths are the final word** for tail-sensitive
  conclusions.
- Some hosts have a broken-cooling GPU0 (e.g. umbriel-b200-019/035): check
  idle temperature and pin `CUDA_VISIBLE_DEVICES=1` for timing work.
