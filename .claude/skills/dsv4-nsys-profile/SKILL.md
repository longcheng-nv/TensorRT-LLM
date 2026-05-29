---
name: dsv4-nsys-profile
description: >
  Single-shot Nsight Systems (nsys) decode-phase profiling launcher for
  DeepSeek-V4 Flash / Pro on Blackwell GPUs. Drives trtllm-bench inside an
  nsys-managed cudaProfilerApi window so the .nsys-rep only contains the
  requested iteration range of steady-state decode — none of the prefill or
  warmup overhead. Auto-detects model path, MoE backend (TRTLLM/MEGAMOE_DEEPGEMM),
  TEP/DEP axes, MTP, and the GVR Top-K toggle. Renders the same G1/G10-correct
  YAML as `dsv4-pareto-bench` so kernel/feature comparisons land on identical
  engine state. Trigger keywords: "nsys profile DSv4 decode", "trace DSv4
  GVR off", "decode kernel timeline", "single-config nsys for V4", "BS=1
  long-ISL profile".
license: LicenseRef-NvidiaProprietary
metadata:
  author: loncheng
  related: dsv4-pareto-bench, computelab-hf-stage
---

# DSv4 nsys decode-phase profiling — single-shot launcher

Companion to [`dsv4-pareto-bench`](../dsv4-pareto-bench/SKILL.md): same model,
same YAML knobs, same G1/G10 fixes, same auto-detection — but one cell at a
time, wrapped in `nsys profile -c cudaProfilerApi`, with `TLLM_PROFILE_START_STOP`
bracketing the iteration window so the `.nsys-rep` only contains the slice you
asked for. The pareto skill is for **landscape** (24-384 cells, no profiler);
this skill is for **kernel-level forensics** (1 cell, full nsys trace).

## When to invoke

Triggers:
- "I need an nsys trace of DSv4 decode" / "profile V4 kernel timeline"
- "compare GVR Top-K ON vs OFF in Nsight" → run twice (`GVR=1`, `GVR=0`)
- "profile V4 Pro DEP at BS=4 ISL=64K" / "trace MTP=3 decode"
- After a kernel change in `cpp/tensorrt_llm/kernels/indexerTopK.cu` or
  `cpp/tensorrt_llm/kernels/heuristicTopKDecode.{cu,h}` — capture before/after.
- Investigating a regression where pareto cell ΔTPOT moved unexpectedly.

Not for:
- Frontier / Pareto sweeps — use `dsv4-pareto-bench` (24-384 cells, no nsys).
- Accuracy validation — use `dsv4-gsm8k-eval`.
- Pure throughput numbers without kernel breakdown — use `dsv4-pareto-bench`
  single-cell (`BSS=… MTPS=… FEATURES=…`) instead of paying nsys overhead.

## Quickstart — the exact configuration the script defaults to

GVR OFF · BS=1 · ISL/OSL=128K/2K · MTP=0 · TEP · 8×B300 · Flash MXFP4 ·
nsys window iters 500-550 (in decode steady state, well past prefill).

```bash
# From the repo root:
bash .claude/skills/dsv4-nsys-profile/scripts/run_nsys.sh
# → produces $PWD/dsv4_nsys_out/ds4_Flash_ISL131072_OSL2048_BS1_MTP0_TEP_GVRfalse_<TS>/
#     decode_trace_iter500_550.nsys-rep
#     decode_trace_iter500_550.sqlite
#     extra-llm-api-config.yml   sampler-options.yml   dataset.jsonl   run.log
```

That single line satisfies "采集关闭 GVR top-K, BS=1, ISL/OSL=128K/2K, MTP=0 的 decode 阶段 nsys 数据".

## Parameter-driven invocation

All axes are env vars; positional args also work for legacy callers. Defaults
reproduce the Quickstart above.

```bash
# 1. Same shape, but Heuristic GVR ON (for paired comparison)
GVR=1 bash .claude/skills/dsv4-nsys-profile/scripts/run_nsys.sh

# 2. DEP4 BS=4 ISL=64K OSL=8K MTP=3 (long-context DEP profile, late-decode window)
ISL=65536 OSL=8192 BS=4 MTP=3 GVR=0 MODE=DEP \
PROFILE_ITER_RANGE=1000-1050 \
bash .claude/skills/dsv4-nsys-profile/scripts/run_nsys.sh

# 3. Pro variant on lyris1 (Pareto-skill auto-detect picks /lustre path)
MODEL_VARIANT=Pro \
ISL=32768 OSL=4096 BS=8 MTP=0 GVR=1 MODE=TEP \
bash .claude/skills/dsv4-nsys-profile/scripts/run_nsys.sh

# 4. Multiple narrow windows in one run (e.g. early vs late decode)
PROFILE_ITER_RANGE=50-60,500-510,1500-1510 \
bash .claude/skills/dsv4-nsys-profile/scripts/run_nsys.sh

# 5. Positional shim — same effect as Quickstart
bash .claude/skills/dsv4-nsys-profile/scripts/run_nsys.sh 131072 2048 1 0 0 TEP

# 6. Real SWE-bench prompt (64K shortcut) — auto-resolves to longseqtasks/swe_bench_64k.jsonl,
#    tokenizes through the V4 chat template, then runs nsys decode-window trace.
PROMPTS_INPUT=swe64k OSL=2048 BS=1 MTP=0 GVR=0 MODE=TEP \
bash .claude/skills/dsv4-nsys-profile/scripts/run_nsys.sh

# 7. Real SWE-bench, single entry × BS replication (entry #2 replicated to BS=4)
PROMPTS_INPUT=swe100k PROMPTS_ENTRY=2 PROMPTS_REPLICATE=4 \
OSL=2048 BS=4 MTP=0 GVR=1 MODE=TEP \
bash .claude/skills/dsv4-nsys-profile/scripts/run_nsys.sh

# 8. Real prompts from your own {system,user} JSONL
PROMPTS_INPUT=/abs/path/to/my_prompts.jsonl OSL=4096 BS=1 \
bash .claude/skills/dsv4-nsys-profile/scripts/run_nsys.sh
```

### Real-prompt input — SWE-bench & custom JSONL

For kernel-forensics on **realistic** decode (real token distributions, real
MTP acceptance, real preIdx behavior — important for GVR Top-K comparisons),
prefer real prompts over `token-norm-dist`. Two shapes are accepted:

1. **`{system, user}` JSONL** (raw SWE-bench format):
   ```json
   {"system": "", "user": "You will be provided with a partial code base ..."}
   ```
   The runner pipes it through
   [`scripts/prepare_swebench_dataset.py`](scripts/prepare_swebench_dataset.py)
   (vendored from `CodeRepos/GVR_TopK_supplementaty_materials/`), which
   applies `tokenizer.apply_chat_template(add_generation_prompt=True)`
   against the **V4 `MODEL_PATH`** so the resulting `input_ids` are exactly
   what the V4 forward pass would see.

2. **Already-tokenized JSONL** (`{task_id, input_ids, output_tokens}` per line —
   `trtllm-bench`'s native format): copied through verbatim. Useful if the
   tokenized cache already exists (e.g. `DSV32/.../dataset_swebench_*.json`).

Auto-resolved canonical SWE-bench sources (first existing wins):

| Shortcut | Resolves to |
|---|---|
| `PROMPTS_INPUT=swe16k`  | `…/GVR_TopK_supplementaty_materials/longseqtasks/swe_bench_16k.jsonl`  → `tllm_toolbox/.../tasks/swe_bench_16k.jsonl`  → DSV32 mirror |
| `PROMPTS_INPUT=swe32k`  | same chain for 32K (ISL ~22-34K real tokens) |
| `PROMPTS_INPUT=swe64k`  | same chain for 64K (ISL **52,244 – 68,656**, 5 entries) |
| `PROMPTS_INPUT=swe100k` | same chain for 100K (ISL **100,200 – 104,749**, 5 entries) |

ISL handling: when `PROMPTS_INPUT` is set the runner re-derives `max_isl`
from the tokenized rows and **promotes** `ISL` / `max_input_len` / `max_seq_len`
(re-rendering the YAML) — your `ISL=…` env var becomes informational. This
means `PROMPTS_INPUT=swe64k` automatically configures the engine for the
correct 52K-69K ISL range without manual coordination.

Single-entry + replication pattern (the BS-sweep workflow Xianjie's reference
uses): `PROMPTS_ENTRY=N` picks one prompt, `PROMPTS_REPLICATE=R` replicates
it R× so `trtllm-bench --num_requests R --concurrency BS` is feasible.

Tokenizer note: all V4 checkpoints (Flash / Pro / *-Base) ship an **empty**
`chat_template` in `tokenizer_config.json` — the V4 runtime applies the
template externally. For offline tokenization the script auto-falls-back
to a sibling DeepSeek-V3.x template (priority: `V3.2-Exp-FP4-v2` →
`V3.2-Exp-hf` → `V3-0324` → `V3`); they share the V4 tokenizer vocab and
special tokens, so the produced `input_ids` are byte-identical to the V3.2
reference pipeline for the same prompt. Override with
`--chat-template-source /path/to/dir-or-jinja-file` on
`prepare_swebench_dataset.py` if you have a custom template. The script
errors loudly if no candidate is reachable.

### Env vars

| Group | Env var | Default | Notes |
|---|---|---|---|
| **Axes** | `ISL` | `131072` | input seq length (tokens) |
| | `OSL` | `2048` | output seq length |
| | `BS` | `1` | batch size; also passed to `--concurrency` and `--max_batch_size` |
| | `MTP` | `0` | speculative MTP layers; `0` drops the speculative_config block entirely |
| | `GVR` | `0` | `1` = Heuristic Top-K ON, `0` = Radix OFF (= `enable_heuristic_topk`) |
| | `MODE` | `TEP` | `TEP` (attn TP-sharded) or `DEP` (attn DP'd); same convention as `dsv4-pareto-bench` |
| **Model / HW** | `MODEL_VARIANT` | `Flash` | `Flash` or `Pro`; drives auto-detect path |
| | `MODEL_PATH` | auto | explicit absolute path overrides auto-detect |
| | `TP`, `EP` | `8`, `8` | tensor / expert parallel |
| | `MAX_NUM_TOKENS` | `8192` | chunked-prefill chunk size; ISL=128K → 16 chunks |
| | `MOE_BACKEND` | auto | derived from `config.json:expert_dtype` × `MODE` |
| | `KV_FRACTION` | `0.8` | `kv_cache_free_gpu_mem_fraction` |
| | `KV_CACHE_DTYPE` | `fp8` | |
| | `STREAM_INTERVAL` | `100` | YAML `stream_interval` (lower → more iter prints) |
| **Profiling** | `PROFILE_ITER_RANGE` | `500-550` | `TLLM_PROFILE_START_STOP` value (`a-b` or `a-b,c-d`) |
| | `NSYS_TRACE` | `cuda,nvtx,python-gil` | `-t` flag |
| | `NSYS_EXTRA_ARGS` | `""` | appended to nsys command (e.g. `--gpu-metrics-devices=all`) |
| **Bench** | `NUM_PROMPTS` | `BS` (floor 1) | `--num_requests` |
| | `WARMUP` | `1` | `--warmup` |
| **Real prompts** | `PROMPTS_INPUT` | (unset) | `swe16k\|swe32k\|swe64k\|swe100k` shortcut, or a path to a `{system,user}` JSONL, or a path to an already-tokenized `{task_id,input_ids,output_tokens}` JSONL. When set, ISL is auto-promoted from the real tokenized length. |
| | `PROMPTS_ENTRY` | (unset) | Keep only entry N (0-indexed) from `PROMPTS_INPUT`. |
| | `PROMPTS_REPLICATE` | auto | Replicate kept entries N× (default: `NUM_PROMPTS` when `PROMPTS_ENTRY` set, else 1). |
| **Output** | `OUT_DIR` | `$PWD/dsv4_nsys_out` | run dir created underneath as `ds4_<...>_<ts>/` |
| **Misc** | `MODEL_CARD` | `deepseek-v4/DeepSeek-V4` | trtllm-bench model card |
| | `TRTLLM_REPO` | `$PWD` | must contain `benchmarks/cpp/prepare_dataset.py` |
| | `DRY_RUN` | `0` | `1` = render YAML/dataset, skip nsys+bench |

### Output layout

```
$OUT_DIR/ds4_<variant>_ISL<ISL>_OSL<OSL>_BS<BS>_MTP<MTP>_<MODE>_GVR<bool>_<UTC>/
├── decode_trace_iter500_550.nsys-rep    # open in Nsight Systems
├── decode_trace_iter500_550.sqlite      # for nsys-stats / custom analyzers
├── extra-llm-api-config.yml             # exact YAML used (G1/G10 compliant)
├── sampler-options.yml                  # greedy top_k=1 seed=0
├── dataset.jsonl                        # generated synthetic prompts
└── run.log                              # bench stdout (full)
```

## Iteration counting cheat sheet — how to pick `PROFILE_ITER_RANGE`

`TLLM_PROFILE_START_STOP` is consumed at `tensorrt_llm/_torch/pyexecutor/py_executor.py:81`
and brackets `cudaProfilerStart/Stop` around the named iter ranges (each iter
= one engine forward pass). Per-cell iter math:

```
prefill_iters = ceil(ISL / max_num_tokens) * NUM_PROMPTS
decode_iters  = OSL * NUM_PROMPTS         (each step emits ≥1 token; MTP layers
                                           are subsumed into the same iter)
warmup_iters  = WARMUP * (prefill_iters + ~few decode)   (rough)
total_iters   ≈ warmup_iters + prefill_iters + decode_iters
```

### Examples

| ISL | OSL | BS | max_num_tokens | WARMUP | Prefill iters | Decode iters | Steady-state window |
|---:|---:|---:|---:|---:|---:|---:|:---|
| 131072 | 2048 | 1 | 8192 | 1 | 16 + 16 (warmup) | 2048 | **500-550** (default) |
| 65536  | 8192 | 4 | 8192 | 1 | 8 + 8            | 8192 | 1000-1050 (mid-decode) |
| 32768  | 4096 | 8 | 8192 | 1 | 4 + 4            | 4096 | 800-850 |
| 524288 | 2048 | 1 | 16384 | 1 | 32 + 32          | 2048 | 200-250 (still decode; prefill <70 iters) |

Heuristics:
- Default `500-550` is safe for any `(ISL ≤ 256K, OSL ≥ 1024, BS ≤ 16)` combo.
- For OSL < 1024 or BS > 16, pull the window earlier (e.g. `200-250`).
- For multi-prompt analysis, use a windowed list: `50-100,500-550,1500-1550`.
- nsys overhead per captured iter is ~1-10 ms additional; a 50-iter window adds ≤ 1 s and produces a 100-500 MB `.nsys-rep`.

### What's outside the window costs nothing

`-c cudaProfilerApi` tells nsys to **only** collect when the program calls
`cudaProfilerStart()`. TRT-LLM does so on every iter inside the configured
range and `cudaProfilerStop()` immediately after. Outside the window: full
native speed, no nsys overhead, no trace bytes. That's why the script always
runs the **full** bench (warmup → prefill → all decode) — the window just
selects which iters land on disk.

## Comparing GVR ON vs OFF (the canonical workflow)

```bash
# Pair 1: Radix (GVR OFF)
GVR=0 OUT_DIR=./nsys_gvr_on_off \
bash .claude/skills/dsv4-nsys-profile/scripts/run_nsys.sh

# Pair 2: Heuristic (GVR ON) — identical shape, only the indexer kernel differs
GVR=1 OUT_DIR=./nsys_gvr_on_off \
bash .claude/skills/dsv4-nsys-profile/scripts/run_nsys.sh

# Compare side-by-side in Nsight Systems:
nsys-ui ./nsys_gvr_on_off/ds4_*_GVRfalse_*/decode_trace_*.nsys-rep \
        ./nsys_gvr_on_off/ds4_*_GVRtrue_*/decode_trace_*.nsys-rep
```

Useful nsys-stats queries to run on the SQLite sidecars:

```bash
# Per-kernel decode-only time
nsys stats --report cuda_kern_exec_sum --format csv \
    ./nsys_gvr_on_off/ds4_*_GVRfalse_*/decode_trace_*.sqlite

# Indexer kernel time
nsys stats --report cuda_kern_exec_sum --format csv \
    --filter '%indexer%' \
    ./nsys_gvr_on_off/ds4_*_GVRtrue_*/decode_trace_*.sqlite
```

## Auto-detected paths (same priority as dsv4-pareto-bench + computelab-hf-stage)

`MODEL_PATH` resolution order (first existing wins):

```
1. /dev/shm/DeepSeek-V4-<variant>                         (RAM, 30 GB/s)
2. /raid/data/${USER}-stage/DeepSeek-V4-<variant>          (local NVMe, 5 GB/s)
3. /home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V4-<variant>   (SC NFS)
4. /lustre/.../common/DeepSeek-V4-<variant>                (lyris1 / coreai_comparch_trtllm)
5. /lustre/.../users/ashanbhag/DeepSeek-V4-<variant>       (dfw alt)
6. /lustre/.../coreai_comparch_inferencex/models/dsv4-<variant_lc>   (lyris1 alt)
```

Steps 1-2 are local-fast staged copies (see [`computelab-hf-stage`](../computelab-hf-stage/SKILL.md)).
If you've staged the model with that skill, this script will pick it up automatically
and decode-iteration latency will be lower (less page-cache contention).

## Gotchas (inherited from `dsv4-pareto-bench` — verify before running)

The full list is in [`dsv4-pareto-bench` G1-G10](../dsv4-pareto-bench/SKILL.md#known-dsv4-gotchas--recipes).
The ones that **always** apply to this script:

- **G1** — KV reuse OFF (template enforces `enable_block_reuse: false` +
  `enable_partial_reuse: false`).
- **G2** — `fast_hadamard_transform` must be importable:
  ```bash
  python3 -c "import fast_hadamard_transform" \
      || pip install --user --no-build-isolation \
         git+https://github.com/Dao-AILab/fast-hadamard-transform.git
  ```
- **G6** — nsys + MPI can leave workers blocked at shutdown; bench's
  auto-scancel works only without nsys. After this script returns
  successfully (or fails), check `nvidia-smi --query-compute-apps=pid` —
  manually `kill -9 <pid>` any straggling python workers before re-running.
- **G10** — the script exports all G10 env vars (NCCL_GRAPH_MIXING_SUPPORT=0,
  NCCL_NVLS_ENABLE=0, TRTLLM_ENABLE_PDL=1, etc.) and emits the matching
  YAML blocks for DEP. Don't override these unless you've read the G10 entry
  in `dsv4-pareto-bench` SKILL.md and understand the trade.

Mode-specific:

- **DEP + Radix (GVR=0) + BS ≥ 8 + ISL ≥ 128K** → likely stack crash (G4).
  Profile in TEP mode for those points instead.
- **DEP + Heuristic (GVR=1) + BS ≥ 16** → ~5-10× slowdown (G5); the trace
  will be huge and the indexer band will dominate the timeline. Useful only
  to *demonstrate* the saturation, not to compare to TEP.
- **Pro + DEP + MTP ≥ 1** → known hang (G3). Use TEP for any MTP-spec
  profile on Pro.

## Reading the trace — what to look for

In Nsight Systems with the default window:

1. **Top-level NVTX**: look for `LLM/forward_step` ribbons — one per iter.
   They should be tight (~20-100 ms at ISL=128K BS=1 MTP=0) with no idle
   gaps between them. Gaps indicate Python-side scheduling drag (the
   `python-gil` trace will show why).
2. **Indexer kernels** (the GVR comparison target):
   - GVR OFF (Radix): `radix_top_k_kernel` / `radix_topK_*` (multiple passes)
     followed by `selectTopK*`. Time goes up with BS × ISL.
   - GVR ON (Heuristic): `heuristic_top_k_decode_kernel` (single fused pass).
     For BS ≤ 8, this should be 2-5× faster than the Radix sum.
3. **Sparse-MLA attention**: `deepseek_v4_sparse_mla_*` block. Should
   dominate decode wall time at long ISL.
4. **MoE**: `fp4_quantize` → `trtllm_fp4_block_scale_moe` block (TEP) or
   `megamoe_deepgemm_*` block (DEP). On Flash MXFP4 the TRTLLM-backend
   GroupedGEMM dominates MoE time.

## Cross-references

- Sibling skill: [`dsv4-pareto-bench`](../dsv4-pareto-bench/SKILL.md) — frontier sweeps (no nsys).
- Sibling skill: [`computelab-hf-stage`](../computelab-hf-stage/SKILL.md) — stage weights to /dev/shm or /raid for fast loading.
- Source: `tensorrt_llm/_torch/pyexecutor/py_executor.py:81` (`TLLM_PROFILE_START_STOP`).
- Source: `cpp/tensorrt_llm/kernels/indexerTopK.cu` (GVR vs Radix dispatcher).
- Predecessor script: `/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/run_perf_DS.sh` (DSv3.2; this skill supersedes it for V4).

## Verification

```bash
# 1. DRY_RUN — render YAML + dataset, no nsys/bench. Should finish in seconds.
DRY_RUN=1 bash .claude/skills/dsv4-nsys-profile/scripts/run_nsys.sh
# Inspect dsv4_nsys_out/ds4_*/extra-llm-api-config.yml for correct YAML.

# 2. Smoke run — small window, ISL=4K (≤ 30 s wall)
ISL=4096 OSL=128 BS=1 MTP=0 GVR=0 \
PROFILE_ITER_RANGE=20-25 \
bash .claude/skills/dsv4-nsys-profile/scripts/run_nsys.sh

# 3. Verify .nsys-rep is non-empty
ls -la dsv4_nsys_out/ds4_*/decode_trace_iter*.nsys-rep
```
