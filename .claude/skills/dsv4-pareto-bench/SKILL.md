---
name: dsv4-pareto-bench
description: Comprehensive playbook for benchmarking DeepSeek-V4 Flash and Pro on Blackwell. Dispatches on THREE user-supplied axes — Hardware (B200 GB200 / B300 GB300), Model variant (Flash / Pro), Dataset mode (synthetic token-norm-dist / real random-v2 ratio=0.8) — into one of 8 (HW × Model × Dataset) recipes mapped to either Workflow A (Xianjie Qiao's cluster-native sbatch sweep, `bench-dsv4/agg_bench/`) or Workflow B (single-node resumable paired-feature driver, `perf_logs/pareto_v4_flash_gvr/`). Includes canonical YAML/CLI defaults, per-(HW, Model) max_num_tokens/kv_frac tables, sweep recipes (F1 Flash, P1 Pro), known DSv4 gotchas (G1-G8: DEP small-bs hang, DEP+Radix+BS≥8 stack crash, cuda_graph+KV-reuse OOB, fast_hadamard_transform requirement, GVR numRows-saturation, MPI shutdown hang, OOM patterns, bwti50), hang/crash recovery, and Pareto-frontier reporting. Use when the user asks to "run Flash/Pro throughput sweep on B200/B300", "generate Pareto curve", "MTP/BS-scan", "compare GVR/feature-X ON vs OFF", "reproduce Xianjie's frontier", "best benchmark config for Flash/Pro", or specifies any subset of (HW, Model, Dataset).
---

# DeepSeek-V4 Pareto throughput benchmarking — Flash & Pro

Unified playbook combining Xianjie Qiao's cluster-native sweep methodology (`bench-dsv4`) with the single-node paired-feature sweep methodology proven on `perf_logs/pareto_v4_flash_gvr`. Both target **DeepSeek-V4 Flash** (MXFP4 routed experts) and **DeepSeek-V4 Pro** (FP8 block-scale routed experts) on Blackwell GPUs (sm_100 / sm_103).

## When to invoke

Triggers (Chinese or English):
- "run Flash/Pro throughput sweep" / "测试 V4 性能"
- "generate Pareto curve" / "build frontier report" / "publish Flash bench numbers"
- "MTP-scan" / "BS-scan" / "concurrency ramp"
- "compare GVR/feature-X ON vs OFF" — for kernel-feature A/B work (PR §6 style)
- "best benchmark config for Flash/Pro on GB200/GB300"
- After a kernel/dispatcher change to DSv4 attention or MoE, when the user wants a Pareto-frontier sanity check

## Quickstart — 4-axis paired sweep (BS × MTP × {GVR_ON,GVR_OFF} × {TEP,DEP})

The skill ships a portable driver in `scripts/` and YAML templates in `templates/`. Five env vars are enough to launch a sweep on any 8-GPU Blackwell node (no SLURM required):

```bash
SKILL_DIR=.claude/skills/dsv4-pareto-bench          # path inside this repo
export PERFDIR=$PWD/dsv4_bench                       # sweep workdir (plan.csv, logs/, results/)
export MODEL_PATH=/path/to/DeepSeek-V4-Flash         # absolute; Pro: also set MOE_BACKEND=DEEPGEMM
export TRTLLM_REPO=$PWD                              # must contain benchmarks/cpp/prepare_dataset.py

# Axes (any subset of these env vars; defaults reproduce B-flash-B300 384 cells)
export BSS="1 2 4 8 16 32"                           # batch sizes
export MTPS="0 3"                                    # MTP draft layers (0 = MTP off)
export MODES="TEP DEP"                               # TEP = attn-DP on, DEP = attn-DP off
export FEATURES="1 0"                                # GVR enable_heuristic_topk: 1=ON first, 0=OFF
export ISLS="131072:4096 262144:4096"                # "ISL:OSL[ ISL:OSL]" pairs

# 1. Generate plan.csv
python3 $SKILL_DIR/scripts/00_generate_plan.py

# 2. (optional) Preview what would run
DRY_RUN=1 bash $SKILL_DIR/scripts/02_master.sh

# 3. Run the sweep (resumable: re-running picks up where it stopped)
setsid nohup bash $SKILL_DIR/scripts/02_master.sh \
  > $PERFDIR/driver_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 < /dev/null &
disown

# 4. Aggregate ON-vs-OFF deltas into REPORT.md + heatmaps
python3 $SKILL_DIR/scripts/03_summarize.py
```

### Axis → env var → YAML/CLI field cheatsheet

| Axis | env (00_generate_plan) | YAML / CLI | Notes |
|---|---|---|---|
| **BS** | `BSS` | `--max_batch_size` + `--concurrency`; `cuda_graph_config.batch_sizes` should include it | DEP+Radix+BS≥8 ISL≥128K crashes (G4); BS≥16 DEP+GVR saturates (G5) |
| **MTP** | `MTPS` | `speculative_config.num_nextn_predict_layers` (block dropped when MTP=0) | Pro DEP4 hangs for MTP∈{1,2,3} — set `MTPS="0"` (G3) |
| **GVR Top-K** | `FEATURES` | `sparse_attention_config.enable_heuristic_topk: true/false` | Requires SM≥100; `compress_ratio=4` indexerTopK path |
| **TEP/DEP** | `MODES` | `enable_attention_dp: true(=TEP) / false(=DEP)`; same `--tp ${TP} --ep ${EP}` for both | TP/EP themselves fixed via `TP` `EP` env vars (default 8/8) |
| **ISL/OSL** | `ISLS="ISL:OSL,..."` | `max_input_len`, `max_seq_len` | OSL drives wall-clock; ISL drives indexer cost |

### Other tunables for `01_run_one_cell.sh` (all env vars, see header)

| Env var | Default | When to change |
|---|---|---|
| `MOE_BACKEND` | `TRTLLM` | `DEEPGEMM` for Pro |
| `MAX_NUM_TOKENS` | `8192` | `16384` on B300 / Flash |
| `KV_FRACTION` | `0.8` | `0.9` for TEP-only sweeps with plenty of VRAM |
| `MOE_MAX_NUM_TOKENS` | `131072` | rarely changed |
| `TP`, `EP` | `8`, `8` | TP4 / TP2 sweeps |
| `MULTI_ROUND` | `2` | bigger = lower variance, longer wall-clock |
| `CELL_TIMEOUT` | `7200` | per-cell hard timeout (s) |
| `CUDA_GRAPH_BS_LIST` | `[1,2,3,4,5,6,7,8]` | extend if running BS>8 cells |
| `EXTRA_YAML_TEMPLATE` | `templates/extra-llm-api-config.yml.tpl` | override for custom YAML shape |

### Per-axis filter (master driver) — slice the matrix without regenerating plan.csv

```bash
ISL_FILTER=131072  MTP_FILTER=3  MODE_FILTER=TEP  GVR_FILTER=1  \
  bash $SKILL_DIR/scripts/02_master.sh
```

### Dataset is identical across every BS cell

Both modes (synthetic, real) generate ONE dataset of size `DATASET_NUM_PROMPTS` and reuse it across all BS cells; smaller-BS cells consume a strict prefix. This is mandatory for paired BS scaling — without it, BS=1 and BS=32 would see different prompts and the BS effect would be confounded with prompt-content noise.

| Quantity | Set by | Default |
|---|---|---|
| `DATASET_NUM_PROMPTS` | `00_generate_plan.py` → `$PERFDIR/bench.env` (sourced by `02_master.sh`) | `max(BSS) * MULTI_ROUND` |
| `MULTI_ROUND` | env at plan-gen time | `2` |

Override `DATASET_NUM_PROMPTS` when re-using a pre-generated dataset of a different size (e.g. real-prompt JSONL has 256 rows total).

### Real-prompt mode (long-sequence SWE-bench)

To benchmark against real prompts instead of synthetic random tokens, pre-generate a trtllm-bench-compatible JSONL once with `scripts/prepare_real_prompts.py`, then point `DATASET_FILE` at it. The sweep skips synthetic generation entirely:

```bash
# 1. Pick the ISL bucket matching your sweep (16k / 32k / 64k / 100k)
curl -sL -O https://raw.githubusercontent.com/longcheng-nv/GVR_TopK_supplementaty_materials/main/longseqtasks/swe_bench_64k.jsonl

# 2. Convert: source JSONL is {system, user}; output is {task_id, prompt, output_tokens}.
#    If --model-path is given AND it contains encoding/encoding_dsv4.py, the
#    DSv4 chat template is applied (preserves thinking_mode/reasoning_effort).
#    Otherwise plain "system\n\nuser" concat is used.
python3 $SKILL_DIR/scripts/prepare_real_prompts.py \
    --input swe_bench_64k.jsonl \
    --output $PERFDIR/datasets/swe_bench_64k.jsonl \
    --num-prompts 256 \
    --output-tokens 4096 \
    --model-path $MODEL_PATH \
    --thinking-mode thinking

# 3. Wire the sweep to use this file (all BS cells share it)
export DATASET_FILE=$PERFDIR/datasets/swe_bench_64k.jsonl
bash $SKILL_DIR/scripts/02_master.sh
```

Source files (4 ISL buckets at `longcheng-nv/GVR_TopK_supplementaty_materials/longseqtasks/`):

| File | Rough ISL after templating | Source rows |
|---|---|---|
| `swe_bench_16k.jsonl`  | ~16 384 | small (cycled if `--num-prompts` larger) |
| `swe_bench_32k.jsonl`  | ~32 768 | — |
| `swe_bench_64k.jsonl`  | ~65 536 | — |
| `swe_bench_100k.jsonl` | ~102 400 | — |

`prepare_real_prompts.py` cycles input rows modulo their length if `--num-prompts` exceeds the source row count — guaranteeing identical per-row content across all BS cells in the sweep (BS=1 sees row 0; BS=4 sees rows 0..7; BS=32 sees rows 0..63; etc.).

### Skill files

```
.claude/skills/dsv4-pareto-bench/
├── SKILL.md
├── scripts/
│   ├── 00_generate_plan.py       # axis cross-product → plan.csv + bench.env
│   ├── 01_run_one_cell.sh        # render YAML, run trtllm-bench, parse log
│   ├── 02_master.sh              # resumable driver, idempotent on restart
│   ├── 03_summarize.py           # paired ON/OFF deltas + heatmaps
│   ├── _parse_one_cell.py        # PERFORMANCE OVERVIEW + [Scheme X] dispatch parser
│   └── prepare_real_prompts.py   # real-prompt JSONL converter (SWE-bench long-seq)
└── templates/
    ├── extra-llm-api-config.yml.tpl   # envsubst template, all knobs as ${VAR}
    └── sampler-options.yml.tpl
```

The rest of this document covers cluster sbatch workflow (A), known DSv4 gotchas (G1-G8), and per-(HW × Model × Dataset) recipes when the Quickstart defaults don't fit.

## Skill arguments — resolve these THREE axes BEFORE planning

The skill must be parameterized by 3 orthogonal user-supplied inputs. Ask if any is unspecified:

1. **Hardware**: `B200` (sm_100, GB200, ~80 GB VRAM/GPU) | `B300` (sm_103, GB300, ~141 GB VRAM/GPU)
2. **Model variant**: `Flash` (Instruct, MXFP4 routed experts, MoE backend = TRTLLM) | `Pro` (Instruct, FP8 block-scale routed experts, MoE backend = DEEPGEMM)
3. **Dataset mode**:
   - `synthetic` — `prepare_dataset.py token-norm-dist`, fixed length (stdev=0), no prefix sharing. Best for **paired feature A/B** comparison (variance floor ≈ ±2 %).
   - `real` — `random-v2 ratio=0.8` (80 % prefix-shared chat-like prompts) from Xianjie's `aa_synthetic_generation`. Best for **production-representative Pareto frontier**.

All 8 combinations (HW × Model × Dataset) are valid — but defaults change per combination. Use the dispatch table below.

## (HW × Model) defaults

| HW | Model | max_num_tokens | kv_frac | MoE backend | TP×EP combos | DEP4 MTP allowed | Notes |
|---|---|---|---|---|---|---|---|
| **B200** | Flash | 16384 | 0.9 | TRTLLM | TP1/2/4, TEP/DEP up to TP4 | 0,1,2,3 | Recipe F1; cluster=OCI/dfw |
| **B200** | Pro   | 8192 (TP4) / 16384 (TP8) | 0.8 | DEEPGEMM | TP4/8, TEP/DEP | **0 only on DEP4** (G3) | Recipe P1; check Pro path on B200 cluster |
| **B300** | Flash | 16384 | 0.9 | TRTLLM | TP8 EP8 (single 8-GPU node) or multi-node TP4 on lyris1 | 0,1,2,3 | Recipe F1 (params unchanged from B200 — B300 has MORE VRAM); also workflow B if doing paired A/B |
| **B300** | Pro   | 16384 | 0.8 | DEEPGEMM | TP4/8, TEP/DEP | **0 only on DEP4** (G3 verified on lyris1 2026-04) | Recipe P1; lyris1 is the canonical B300 Pro cluster |

For ALL (HW, Model) combos: KV reuse OFF (G1), `fast_hadamard_transform` installed (G2), `enable_padding=true`, `kv_cache_config.dtype=fp8`, `tokens_per_block=128`.

## Dataset mode → workflow + dataset path

| Dataset | Preferred workflow | Dataset source | When to override |
|---|---|---|---|
| `synthetic` (token-norm-dist) | **B (paired ON/OFF)** — `perf_logs/pareto_v4_flash_gvr/scripts/02_master.sh` | generated on-the-fly by `01_run_one_cell.sh`, cached at `datasets/synth_isl${ISL}_osl${OSL}_n${N}.jsonl` | Override to workflow A for synthetic cross-cluster scaling tests — pre-generate the JSONL once, manually copy onto each cluster, edit `run.sh` dataset_file path |
| `real` (random-v2 ratio=0.8) | **A (Xianjie Pareto)** — `bench-dsv4/agg_bench/submit_*.sh` | pre-generated via `aa_synthetic_generation` repo, lives at `${home_dir}/dataset/random-v2/...` per cluster | Override to workflow B only if you need paired-comparison metrics on the realistic workload AND have copied the random-v2 JSONL into `pareto_v4_flash_gvr/datasets/` |

**Default recipe lookup** — pick from the 8 cells below given (HW, Model, Dataset):

| HW | Model | Dataset | Recipe | Driver pattern | Cells | Approx walltime |
|---|---|---|---|---|---|---|
| B200 | Flash | real      | **F1** | sbatch A | ~152 | 6-8 h on OCI/dfw |
| B200 | Flash | synthetic | **B-flash-B200** (port B from B300 — edit `01_run_one_cell.sh` MODEL_PATH + verify TP fits the 80GB-VRAM budget at BS=64+) | nohup B | 384 (filter to <128 cells) | depends on filter |
| B200 | Pro   | real      | **P1**, but cap TP4 cells at MTP=0 if DEP4 (G3) | sbatch A | ~134 | 6-8 h on dfw |
| B200 | Pro   | synthetic | **B-pro-B200** + ATTN_DP-only sweep (skip DEP MTP>0) | nohup B | ≤192 | 6 h+ |
| B300 | Flash | real      | F1 reused on lyris1 / single-node interactive — params unchanged | sbatch A on lyris1, or single-node loop | ~152 | 4-6 h |
| B300 | Flash | synthetic | **B (this is the proven recipe in `pareto_v4_flash_gvr`)** | nohup B | up to 384 (filter ISL+MTP) | 4-6 h per ISL+MTP slice |
| B300 | Pro   | real      | **P1** (Xianjie's lyris1 published spec) | sbatch A on lyris1 | ~134 | 6-8 h |
| B300 | Pro   | synthetic | **B-pro-B300** + ATTN_DP-only (DEP4 MTP>0 hang G3) + `cuda_graph_config: null` for accuracy-critical paths | nohup B | ≤192 | 6 h+ |

If the user asks for "best Pareto curve" without specifying axes → default to **(B300, Flash, real, F1)** on lyris1.

If the user asks for "is feature X a win" without specifying axes → default to **(B300, Flash, synthetic, B-paired)** on the single-node interactive host (`umb-b300-dp-*`).

## HW → cluster reverse lookup

The "Hardware platform matrix" below lists by cluster. This reverse table answers "I have hardware X, which cluster do I submit to?":

| Hardware | Canonical clusters | Submission style |
|---|---|---|
| **B200 GB200** | **OCI** (`batch`, `--gres=gpu:4` required, account=`coreai_comparch_trtllm` or `coreai_comparch_inferencex`) <br/> **dfw** (`batch`, `--gres=gpu:4` required, account=`coreai_comparch_inferencex`) | `sbatch --wrap=srun --container-image=... bash run.sh ...` |
| **B300 GB300** | **lyris1** (`gb300`, **NO `--gres`**, account=`coreai_comparch_inferencex`) <br/> **Interactive single-node** (e.g. `umb-b300-dp-184`, 8× B300, NO SLURM) | sbatch on lyris1; nohup/setsid on single-node |

Pre-FairShare check is mandatory on every cluster (see Pre-flight below).

## Pre-flight checklist (run BEFORE first sbatch / nohup)

Verify all 8 before launching ANY sweep cell. Failures here cause silent mid-run crashes that waste hours of budget.

1. **FairShare account** — pick the account with the LARGEST FairShare on the target cluster:
   ```bash
   ssh <cluster> "sshare -u \$USER --format=Account,FairShare -P"
   # Update `account=` in submit script to the winner. FairShare rotates over time.
   ```
2. **Model path exists** — `ls "$MODEL_PATH" | head` returns weight files. Pick from the (HW × Model × Cluster) section above; double-check on the destination host (mounts differ between login and compute nodes).
3. **Dataset present**:
   - real mode: `ls ${home_dir}/dataset/random-v2/<...>_for_bench.json`
   - synthetic mode: `ls ${PERFDIR}/datasets/synth_isl${ISL}_osl${OSL}_n${N}.jsonl` — if missing, 01_run_one_cell.sh will generate at first call (one-time ~10 min per (ISL, OSL, n) triple).
4. **`fast_hadamard_transform` installed in the run env** (G2):
   ```bash
   python3 -c "import fast_hadamard_transform; print(fast_hadamard_transform.__version__)"
   # If ImportError → pip install --user --no-build-isolation \
   #   git+https://github.com/Dao-AILab/fast-hadamard-transform.git
   ```
5. **`transformers` version matches branch pin** — `pip show transformers | grep Version` must equal what `tensorrt_llm/requirements.txt` pins (DSv4 branch: 4.57.3). Newer raises `ImportError: AutoModelForVision2Seq` at `import tensorrt_llm`.
6. **KV reuse OFF in YAML template** (G1) — `enable_block_reuse: false` AND `enable_partial_reuse: false`. Mandatory for Pro + cuda_graph, recommended for Flash for consistency.
7. **For Pro DEP4: enforce MTP=0 only** (G3) — confirm submit script does NOT enumerate MTP ∈ {1,2,3} for any `tp4_ep4_adp1` cell unless willing to debug hangs.
8. **GPU clean** — `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` returns empty on every target node. Stale tensors from prior crashed runs cause OOM-on-init.

If any of 1-8 fails, **abort and fix before submitting**. None of these are recoverable mid-sweep without restarting the cell.

## Two complementary workflows

| Workflow | Path | Best for | NOT for |
|---|---|---|---|
| **A. Cluster-native full sweep** | `bench-dsv4/agg_bench/` (Xianjie Qiao's repo) | Production-quality Pareto frontier across many (TP/EP/adp/BS/MTP) cells; cross-cluster comparison HTML; using shared random-v2 dataset | A/B kernel comparisons (no paired ON/OFF baked in) |
| **B. Single-node paired sweep** | `perf_logs/pareto_v4_flash_gvr/scripts/` (our driver) | Tight paired ON/OFF kernel/feature comparisons on a single 8-GPU node; full debug trace (n_scheme_x, numRows distribution); resumable across SLURM step kills | Multi-cluster scaling story; >8 GPU configs |

When the user asks for "best Pareto curve" → workflow **A**. When the user asks "is feature X a win on DSv4?" → workflow **B**. Both share the YAML/CLI knobs below.

## Hardware platform matrix

| Cluster | GPU | Connectivity | TP/EP combos used | Notes |
|---|---|---|---|---|
| **OCI batch** | B200 (sm_100) GB200 | NVLink full mesh, 4 GPU/node | TP1/2/4 | partition=`batch`, `--gres=gpu:4` required |
| **dfw batch** | B200 GB200 | NVLink, 4 GPU/node | TP4, TP8 | same as OCI; `--gres=gpu:4` required |
| **lyris1 gb300** | B300 (sm_103) GB300 | NVLink, 4 GPU/node | TP4, TP8 (multi-node) | partition=`gb300`; **do NOT pass `--gres=gpu:N`** |
| **umb-b300-dp-184** (interactive) | B300 ×8 | NVLink full mesh | TP8 EP8 | single-node sweep target; `141GB` VRAM / GPU |

## Model paths (per cluster, per variant)

| Cluster | Flash | Pro |
|---|---|---|
| **OCI** | `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/common/DeepSeek-V4-Flash` | `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/common/DeepSeek-V4-Pro/` |
| **lyris1** | `/lustre/fsw/coreai_comparch_inferencex/models/dsv4-flash` | `/lustre/fsw/coreai_comparch_inferencex/models/dsv4-pro` |
| **dfw** | `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/models/DeepSeek-V4-Flash` | `/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/ashanbhag/DeepSeek-V4-Pro` |
| **Interactive single-node** | `/home/scratch.jinshik_gpu/DeepSeek-V4-Flash` | `/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V4-Pro` |

Workflow A's `run.sh` auto-detects path by cluster; override with `MODEL_PATH` env var. Workflow B hardcodes the interactive single-node path; edit `01_run_one_cell.sh` to switch clusters.

## Standard YAML defaults (apply to BOTH workflows)

```yaml
# Decode shape
max_input_len:  <ISL + 512>          # margin for chat-template prefix
max_seq_len:    <ISL + OSL + 512>

# CUDA Graph — broad bs list for Xianjie-style ramp, narrow for paired comparison
cuda_graph_config:
    enable_padding: true
    batch_sizes: [1,2,4,8,16,32,64,128,192,256,320,384,448,512,768,1024]
    # For paired feature comparison (workflow B), use the narrower
    # [1,2,3,4,5,6,7,8] to keep per-cell warmup time bounded.

# KV cache — V4 ALWAYS needs reuse OFF (workaround for sparse-MLA OOB)
kv_cache_config:
    enable_block_reuse:   false       # mandatory for Pro + cuda_graph;
                                      # Flash sometimes survives with true but use false to match Pro
    enable_partial_reuse: false       # paired with the above
    tokens_per_block:     128         # Xianjie default; works on both variants
    dtype:                fp8         # 2× KV capacity vs bf16

# MoE backend
moe_config:
    # Flash (MXFP4 routed experts):
    backend: TRTLLM
    # Pro (FP8 block-scale routed experts):
    # backend: DEEPGEMM   # workflow A default
    # backend: WIDEEP     # alternate for Pro-Base completion model
    max_num_tokens: 131072            # workflow B's larger value;
                                      # workflow A uses 16384 (Flash GB200) / 8192-16384 (Pro)

# Logging + stream throttling
print_iter_log: true
stream_interval: 100                  # workflow A only (reduces tqdm churn)

# Attention parallelism — controls TEP vs DEP
enable_attention_dp: true             # TEP (Tensor + AttnDP + Expert parallel)
# enable_attention_dp: false          # DEP (Tensor + Expert parallel, no AttnDP)

# Autotuner — disable for reproducible numbers
enable_autotuner: false               # workflow B default; workflow A leaves on

# Speculative decoding (MTP layers, V4 native)
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: <MTP>   # 0 → drop the whole block; 1-3 typical

# AllReduce path — only for tp=8 ep=8 + DEP (attn_dp=false)
# allreduce_strategy: MNNVL          # Xianjie sets this conditionally

# Feature toggles (workflow B's added knobs)
sparse_attention_config:
    algorithm: deepseek_v4
    enable_heuristic_topk: <bool>     # GVR Heuristic vs Radix top-K
```

## Standard CLI defaults (`trtllm-bench throughput`)

```bash
trtllm-llmapi-launch numactl -m 0,1 \
trtllm-bench \
  -m mewtwo/Mewtwo                 \   # canonical model card (works for both)
  --model_path  ${MODEL_PATH}       \
  throughput                        \
  --tp          ${TP_SIZE}          \
  --ep          ${EP_SIZE}          \
  --warmup      0                   \   # workflow A: 0 (skip);
                                       # workflow B: 1 (warmup matters for ON/OFF parity)
  --dataset     ${DATASET}          \
  --backend     pytorch             \
  --max_batch_size  ${BS}           \
  --max_num_tokens  ${MAX_NUM_TOKENS} \
  --kv_cache_free_gpu_mem_fraction ${KV_FRAC} \   # 0.8 default; 0.9 if not OOMing
  --concurrency ${BS}               \   # workflow A: linear sweep value (1..max_bs)
                                       # workflow B: = max_batch_size (= BS)
  --num_requests ${BS * MULTI_ROUND} \  # workflow A multi_round=5; workflow B=2
  --extra_llm_api_options ${YAML}   \
  --sampler_options ${SAMPLER}      \   # workflow B only — top_k=1 seed=0 for greedy
  --disable_chunked_context         \   # workflow A only — important for steady-state metric
  --streaming                       \
  --report_json ${PREFIX}_report.json   # workflow A — structured metrics dump
```

`trtllm-llmapi-launch` is the MPI-rank-aware launcher (rank 0 = driver, rank N = MGMN worker). Required for multi-node TP via `--mpi=pmix` in SLURM. `numactl -m 0,1` binds memory to NUMA nodes 0 + 1 — essential for large-BS perf on dual-socket GB200/GB300 hosts.

## Sweep recipes — what "best" Pareto coverage means

### Recipe F1 — Flash full bench (HW-agnostic; Xianjie's published spec, originally B200)

Applies to **both B200 GB200 and B300 GB300 Flash**. B300 has more VRAM so the same parameter set runs without OOM headroom issues. All cells × `MTP ∈ {0,1,2,3}`, cuda-graph only, `max_num_tokens=16384`, `kv_frac=0.9` unless noted.

| Config | TP×EP | adp | max_bs | sweep (concurrency) |
|---|---|---|---|---|
| TEP1 | 1×1 | off | 256 | 1,2,4,8,16,32,64,128,256 |
| TEP2 | 2×2 | off | 256 | 1,2,4,8,16,32,64,128,256 |
| DEP2 | 2×2 | on  | 1024 | 1,2,4,8,16,32,64,128,256,512,1024 |
| TEP4 | 4×4 | off | 256 | 1,2,4,8,16,32,64,128,256 |
| DEP4 | 4×4 | on  | 1024 | 1,2,4,8,16,32,64,128,256,512,1024 |

Total: ~38 concurrencies × 4 MTP = **~152 cells**.

Hardware deployment notes:
- **B200**: Recipe used as-is on OCI/dfw (cluster=`batch`, `--gres=gpu:4` required).
- **B300**: Same recipe on lyris1 (cluster=`gb300`, **no `--gres` flag**). Also valid on single 8-GPU interactive B300 (e.g. `umb-b300-dp-184`) by running TEP4/DEP4/TEP8/DEP8 cells natively without multi-node MPI.

### Recipe P1 — Pro full bench (HW-agnostic; Xianjie's lyris1 published spec, originally B300)

Applies to **both B200 GB200 and B300 GB300 Pro**. On B200 use `max_num_tokens=8192` for TP4 (KV+activation fits in 80 GB at high BS); B300 has headroom for 16384 on TP4 too. `kv_frac=0.8` on DEP configs (G7 OOM history on `dep8 conc=2048 mtp=0 kv=0.9 dfw`); 0.9 on TEP.

| Config | TP×EP | adp | max_bs | kv_frac | sweep | MTP |
|---|---|---|---|---|---|---|
| TEP4 | 4×4 | off | 256  | 0.9 | 1..256 (9 pts) | 0,1,2,3 |
| **DEP4** | 4×4 | on  | 64   | **0.8** | 1..64 (7 pts) | **0 only** (1/2/3 hang — see Gotchas) |
| TEP8 | 8×8 | off | 512  | 0.9 | 1..512 (10 pts) | 0,1,2,3 |
| TP8  | 8×1 | off | 512  | 0.9 | 1 only          | 0,1,2,3 |
| DEP8 | 8×8 | on  | 1024 | **0.8** | 1..1024 (11 pts) | 0,1,2,3 |

Total: **~134 cells**. SLURM walltime: use `--time=02:00:00`; 1 h is not enough (DEP4 BS=64 hits TIME LIMIT on lyris1).

### Recipe B — Paired feature ON/OFF sweep (our `perf_logs/pareto_v4_flash_gvr` shape)

5-axis matrix: `BS × ISL × OSL × MTP × Mode × Feature` where Feature is the kernel knob being A/B-tested (e.g. `enable_heuristic_topk`). Cells laid out so paired ON/OFF cells run back-to-back for thermal/cache-state parity:

```python
# Baseline (B-flash-B300, the proven recipe from pareto_v4_flash_gvr)
BSS    = [1, 2, 4, 8, 16, 32, 64, 128]
ISLS   = [(128 * 1024, 4 * 1024), (256 * 1024, 4 * 1024), (512 * 1024, 4 * 1024)]
MTPS   = [0, 1, 2, 3]
MODES  = ["TEP", "DEP"]
FEATURES = [True, False]    # ON first (paired block)
# → 384 cells, ordered (ISL asc, BS asc, MTP asc, Mode {TEP first}, Feature {ON first})
```

Use the filter env vars on master driver to subset (e.g. `MTP_FILTER=3 ISL_FILTER=131072` to focus on a single ISL × MTP cross-section).

#### Recipe B variants — adjust per (HW, Model)

Edit `00_generate_plan.py` axis arrays and the YAML template in `01_run_one_cell.sh` per the table. The driver/parser scripts are HW/Model-agnostic.

| Variant | BSS | ISLS | MTPS | MODES | YAML deltas vs baseline | Cell count | Notes |
|---|---|---|---|---|---|---|---|
| **B-flash-B300** (baseline) | `[1,2,4,8,16,32,64,128]` | `[128K, 256K, 512K]` | `[0,1,2,3]` | `[TEP, DEP]` | (none — uses baseline YAML) | 384 | Proven on `umb-b300-dp-184`; BS=64/128 OOM expected, BS≥8 DEP+Radix crash expected (G4) |
| **B-flash-B200** | `[1,2,4,8,16,32]` (drop 64+; B200 80GB VRAM cap) | `[64K, 128K, 256K]` (drop 512K) | `[0,1,2,3]` | `[TEP, DEP]` | `max_num_tokens: 16384`; model_path → B200 cluster (OCI/dfw) | 192 | Verify TP8 EP8 single-node B200 host exists; else split across multi-node TP4 + sbatch wrap |
| **B-pro-B300** | `[1,2,4,8,16,32,64,128]` | `[128K, 256K, 512K]` | TEP: `[0,1,2,3]`, DEP4: `[0]` only (G3) | `[TEP, DEP]` | `moe_config.backend: DEEPGEMM`; `kv_cache_free_gpu_mem_fraction: 0.8`; model_path → Pro B300 (`lyris1` or interactive) | ≤192 | Skip BS≥16 DEP MTP>0 cells (G3 hang); BS=64/128 likely OOM even on 141GB |
| **B-pro-B200** | `[1,2,4,8,16,32]` (drop 64+) | `[64K, 128K]` (drop 256K/512K; Pro KV is heavier) | TEP: `[0,1,2,3]`, DEP4: `[0]` only | `[TEP, DEP]` | `max_num_tokens: 8192` (TP4) / `16384` (TP8); `kv_frac: 0.8`; `moe_config.backend: DEEPGEMM` | ≤96 | Aggressive prune — Pro on B200 hits OOM at concurrency≥1024 with kv=0.9; stick to TP≤4 |

After picking a variant, regenerate plan + restart driver:
```bash
cd ${PERFDIR}
# Edit scripts/00_generate_plan.py axis arrays to match the variant.
# Edit scripts/01_run_one_cell.sh: MODEL_PATH, moe_backend, kv_frac, max_num_tokens.
python3 scripts/00_generate_plan.py   # regenerates plan.csv
# DELETE completed.txt/failed.txt if variant changed parameters that invalidate prior cells:
# rm completed.txt failed.txt   # only if NOT reusing prior results
setsid nohup env MAX_WALL_SECONDS=21600 ISL_FILTER=... MTP_FILTER=... \
  bash scripts/02_master.sh > driver_logs/driver_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 < /dev/null &
disown
```

## Two driver patterns (pick one)

### Pattern A — SLURM `sbatch --wrap` (Xianjie / bench-dsv4)

```bash
# Inside submit_<variant>_<cluster>.sh — one sbatch per cell.
sbatch -N $nodes -n $tp \
  --partition=$partition --account=$account --time=02:00:00 \
  --output=$log --error=$log \
  --export=ALL,MODEL_VARIANT=Pro,MAX_NUM_TOKENS=16384 \
  --wrap="srun --no-container-mount-home --container-image=${image} \
          --container-mounts=/lustre:/lustre --mpi=pmix \
          bash ${script_dir}/run.sh $tp $ep $bs $adp $kv $home_dir $pre nvfp4 \"\" $conc $mtp"
```

**Pre-flight check (always before submitting)**:
```bash
ssh <cluster> "sshare -u \$USER --format=Account,FairShare -P"
# Pick the account with the LARGEST FairShare; update account= in your submit script.
```

Critical sbatch quirks (from Xianjie's CLAUDE.md):
- **lyris1 gb300 does NOT accept `--gres=gpu:N`** — omit it. dfw/OCI require it.
- For >50 jobs: use `sbatch --wrap` not `srun ... &`; parallel `srun &` on the login node hits the pthread fork limit and many submissions fail silently.
- Each `run.sh` rank 0 calls `scancel` after the bench writes a non-empty `_report.json` → releases the allocation immediately (don't wait for walltime).

### Pattern B — Single-node resumable driver (our `02_master.sh`)

```bash
cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/perf_logs/pareto_v4_flash_gvr
setsid nohup env \
  MAX_WALL_SECONDS=21600   \   # 6 h budget; honors SLURM_JOB_END_TIME if set
  ISL_FILTER=131072        \   # optional axis filters
  MTP_FILTER=3             \
  bash scripts/02_master.sh > driver_logs/driver_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 < /dev/null &
disown
```

Driver semantics:
- Reads `plan.csv`, skips cells in `completed.txt` or `failed.txt` (unless listed in `force_retry.txt`).
- Per-cell `timeout --kill-after=30s 7200` wraps `trtllm-bench` — runaway cell can't burn the whole budget.
- Each cell's `01_run_one_cell.sh` writes the YAML, runs bench, parses log → appends one row to `results/per_cell.csv`, records cell_id in `completed.txt` or `failed.txt` with reason (OOM / BENCH_FAIL_exitN / TIMEOUT / PARSE_FAIL).
- Stdin isolation: master pipes `< /dev/null` into each cell. Without this, `trtllm-bench`'s distributed launcher consumes the outer `tail | while read` pipe → master exits after 1 iteration.

## Dataset preparation (technical details for each mode)

> The choice of dataset mode is an explicit skill argument (axis 3 above). This section is the implementation details once a mode is picked. Both modes can be combined with either workflow A or B — the "preferred workflow" column in the dispatch table is the recommended default, not a hard constraint.

### Dataset mode "real" — random-v2 prefix-matching (Xianjie's `aa_synthetic_generation`)

Generated separately via Xianjie's `aa_synthetic_generation` repo:
```bash
git clone https://gitlab-master.nvidia.com/xqiao/aa_synthetic_generation.git
# Produces:
#   ${home_dir}/dataset/random-v2/<basename(model)>-${isl}-${osl}-${num}-ratio-${ratio}_for_bench.json
# Canonical:
#   DeepSeek-V4-Pro-1024-1024-20000-ratio-08_for_bench.json   (1k/1k, ratio=0.8)
#   DeepSeek-V4-Pro-1024-1-20000-ratio-08_for_bench.json      (ctx-only, osl=1)
```

`ratio=0.8` = 80% prefix-shared prompts (realistic chat workload); `ratio=1.0` = unique prompts (worst-case).

### Dataset mode "synthetic" — token-norm-dist (in-repo `prepare_dataset.py`)

Generated on-the-fly by `01_run_one_cell.sh`, cached per (ISL, OSL, num_prompts):
```bash
python3 benchmarks/cpp/prepare_dataset.py \
    --tokenizer "${MODEL_PATH}" --stdout \
    token-norm-dist \
        --num-requests "${NUM_PROMPTS}" \
        --input-mean   "${ISL}" --input-stdev  0 \
        --output-mean  "${OSL}" --output-stdev 0 \
    > "${DATASET}"
```

`stdev=0` = fixed-length prompts → eliminates length-variance noise for paired comparison.

## Known DSv4 gotchas + recipes

### G1 — KV reuse + cuda_graph crash on V4

**Symptom**: async CUDA OOB at `_torch/attention_backend/sparse/deepseek_v4/deepseek_v4.py:_compute_ctx_compressed_position_ids` (torch.arange surfaces OOB from preceding sparse-MLA graph replay).

**Trigger**: `cuda_graph_config` enabled (any bs list) AND `kv_cache_config.enable_block_reuse=true`.

**Fix**: always set both `enable_block_reuse: false` and `enable_partial_reuse: false`. Mandatory for Pro, recommended for Flash for consistency.

### G2 — `fast_hadamard_transform` missing → mid-run illegal address

**Symptom**: startup warning `Sparse MLA will skip hadamard transformation`, then `CUDA error: an illegal memory access` mid-decode.

**Fix**:
```bash
pip install --user --no-build-isolation \
  git+https://github.com/Dao-AILab/fast-hadamard-transform.git
```
`--no-build-isolation` is mandatory (setup.py imports torch at build time).

### G3 — Pro DEP4 small-bs hang (lyris1)

**Symptom**: Pro `tp4_ep4_adp1 (DEP4)` with `bs ∈ {1..64}` AND `MTP ∈ {1,2,3}` AND `kv=0.8` hangs after weight load (cuda graph capture phase) — never enters the bench loop. Pure MTP=0 works.

**Mitigation**: Only sweep MTP=0 for DEP4 on Pro (Xianjie's published spec encodes this). Untested: `kv=0.9` might recover MTP=1/2/3 — try if needed.

### G4 — DEP+Radix+BS≥8+ISL=128K stack crash (TEP fine)

**Symptom**: `bench` crashes mid-decode with one of two stacks depending on BS:
- BS=8 DEP `enable_heuristic_topk=false`: DeepGEMM `smxx_layout.hpp:97` → `CUDA_ERROR_ILLEGAL_ADDRESS`.
- BS=16 DEP `enable_heuristic_topk=false`: cuBLAS `cublasGemmEx` → `CUBLAS_STATUS_EXECUTION_FAILED`.

Both followed by `Child job N terminated normally, but 1 process returned a non-zero exit code`.

**Root cause**: Pre-existing TRT-LLM stack fragility with DEP attention sharding at large BS, ISL=128K decode. **Independent of GVR** — same-point GVR-ON cells (Heuristic indexer path) complete successfully (though slowly at BS≥16 — see G5).

**Mitigation**: When sweeping DEP at BS≥8 ISL≥128K, expect 1-2 cells per BS step to crash. Manual kill workflow (see "Hang/crash recovery" below) is faster than waiting for the 7200s `timeout` wrapper.

### G5 — GVR Heuristic numRows-saturated slowdown (DEP large BS)

**Symptom**: BS=16/32 DEP `enable_heuristic_topk=true` completes but is **5-10× slower than TEP same point**. Trace shows `numRows_distribution` peaked at `64:264264` (BS=16 DEP) or `128:191688` (BS=32 DEP); `n_scheme_x_lines` 10× higher than TEP.

**Root cause**: DEP keeps attention rows un-sharded across ranks; BS=B × MTP_layers → numRows = B × (1 + MTP). GVR Heuristic K=512 secant iterations are exposed at large numRows → each indexer call ~100 μs × 500k calls = ~50 s pure-indexer work.

**Mitigation (for code path, not runtime)**: Add a `numRows ≤ N` guard in `canUseHeuristic` to fall back to Radix when DEP+large-BS would saturate. For benchmarks, mark these cells as "anti-pattern" in the report rather than waiting for them to complete.

### G6 — MPI shutdown hang under nsys

**Symptom**: nsys-wrapped jobs sometimes leave MPI workers (rank>0) blocked during shutdown → `mpirun` never returns → `run.sh` auto-scancel block never runs.

**Mitigation**: manual `scancel` for nsys jobs. (Auto-scancel only works for non-profiled bench runs.)

### G7 — OOM patterns to watch

- OCI `tep4 conc≥768 mtp=2 kv=0.9` (Flash) — OOM during executor init.
- dfw Pro `dep8 conc=2048 mtp=0 kv=0.9` — same OOM.
- B300 single-node `BS=64+ ISL=128K MTP=3` — likely OOM on 8× B300 (KV cache exceeds ~80 GB per rank).
- **Fix**: drop `kv_fraction` from 0.9 → 0.8; or reduce `concurrency`/`max_batch_size`/`mtp_size`.

### G8 — bwti50 stabilizer (`batch_wait_timeout_iters: 50`)

Acts as a stabilizer for "collapsed" cases (e.g. dep2 mtp=3 sees +50% throughput with bwti50). But slightly hurts well-tuned mtp=0/1 configs. **Targeted fix, NOT a general improvement** — only add via `BATCH_WAIT_TIMEOUT_ITERS=50` env var for the specific configs that collapse without it.

## Hang/crash recovery workflow

When a bench cell hangs (frequent for G3, G4, G5 cells), the `timeout` wrapper waits the full 7200s before giving up — wasting 1-2 h of budget. Detect and kill manually:

**Symptoms of hang (vs slow but progressing)**:
1. Cell's `<cell_id>.log` mtime stale > 5 min.
2. `nvidia-smi`: 1 GPU shows ~700 MB used, other 7 show ~4 MB (main bench process waiting for dead MPI workers).
3. Cell log already contains a `Traceback`/`RuntimeError`/`terminate called` near the tail.

**Recovery**:
```bash
# 1. Confirm:
PERFDIR=...your sweep dir
CELL=cell_063_128K_008_M3_DEP_GVRF
grep -nE "(RuntimeError|terminate called|CUDA error)" \
    "$PERFDIR/logs/$CELL.log" | head -5
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader

# 2. Kill the timeout wrapper (NOT the python process directly):
TW_PID=$(pgrep -f "timeout.*7200.*trtllm-bench.*${CELL}")
kill -TERM $TW_PID
# Wait ~5 s; SIGTERM cascades to bench process group → mpirun reaps → runner sees non-zero exit
# Runner records BENCH_FAIL_exit143 in failed.txt → master moves to next cell within ~30 s
```

**Why TW_PID and not the python PID**: killing the timeout wrapper makes `timeout` exit with 143 (= 128 + SIGTERM). The runner's `if [[ ${BENCH_EXIT} -eq 124 ]]` branch is for clean `timeout` expiry; with 143 it falls through to the generic `BENCH_FAIL_exitN` recorder, which is the path you want. Killing the python directly leaves the wrapper alive (with exit 0 from its sub-process) and might confuse the runner state machine.

This recovery saved ~1 h of budget per occurrence in the BS=8/16/32 DEP-Radix crash cluster observed on 2026-05-18.

## Reporting — pick by audience

### R1 — Cross-cluster HTML with Pareto frontier (Xianjie's `bench-report` skill)

```bash
# After bench logs land at $log_dir on a cluster:
cd bench-dsv4
ssh ${cluster} "bash agg_bench/gather_perf.sh \
    /lustre/.../bench-mewtwo/agg_bench/${log_dir}" > /tmp/${cluster}_all.txt

python3 agg_bench/report_perf.py \
    --primary ${cluster}:/tmp/${cluster}_all.txt \
    --compare lyris1:/tmp/lyris1_all.txt \
    --compare dfw:/tmp/dfw_all.txt \
    --label flash_full --account ...
# → agg_bench/report_<cluster>_<YYYYMMDD>_<label>.html
```

Or invoke the `bench-report` skill directly (per `bench-dsv4/.claude/skills/bench-report.md`) with `log_dir` as argument.

Report includes:
- **Status grid** per MTP (left=MTP=0, right=MTP=1+): rows=configs, cols=concurrency, cells=OK/TIMEOUT/HANG/RAWREF/ERROR.
- **Per-config charts**: Per-User TPS vs Output TPS/GPU; Per-User TPS vs Total TPS/GPU.
- **Pareto frontier** (the most important chart): scatter of ALL successful runs as faint background, frontier overlay as bold line. Frontier definition: a point `(x_i, y_i)` is on the frontier iff no other point has both `x ≥ x_i` AND `y ≥ y_i` with at least one strictly greater. Top-right is best.
- **Tables**: successful runs (highlight frontier members in yellow); failed runs with reason.

### R2 — Per-cell CSV + heatmap (workflow B's `03_summarize.py`)

```bash
python3 ${PERFDIR}/scripts/03_summarize.py
# → results/REPORT.md (paired ΔTPOT table)
# → results/figures/dtpot_isl_bs_<Mode>_M<MTP>.png   (one heatmap per Mode/MTP slice)
```

Each row in `per_cell.csv` carries 24 fields including:
- Throughput: `req_per_sec`, `total_output_tok_s`, `total_token_tok_s`, `per_gpu_output_tok_s`, `per_user_output_ctx`
- Latency: `tpot_avg_ms`, `ttft_avg_ms`, `avg_req_latency_ms`, `total_latency_ms`
- Energy: `total_energy_j`, `tps_per_w`, `avg_gpu_power_w`
- Dispatch trace: `dispatch_path ∈ {Heuristic, Radix, Mixed, Unknown}`, `n_scheme_x_lines`, `numRows_distribution`

The `numRows_distribution` field (e.g. `"4:12516,8:27447,16:1008,...,64:672"`) is crucial for G5-style diagnoses — saturation at a single large nR bucket is the smoking gun for indexer anti-patterns.

## Pareto-frontier definition (precise)

For a 2-D metric pair `(per_user_tps, per_gpu_tps)`:

```python
def pareto_frontier(points):
    """A point (x,y) is on the frontier iff no other point dominates it.
    Dominates = both metrics ≥ AND at least one strictly >. Top-right is best."""
    pts = sorted(points, key=lambda p: -p[0])     # x descending
    frontier, y_max = [], -float("inf")
    for p in pts:
        if p[1] > y_max:                          # strictly better y given lower x
            frontier.append(p)
            y_max = p[1]
    return frontier
```

For paired ΔTPOT comparison (workflow B):

```python
def paired_delta_tpot(on_cell, off_cell):
    # Same (BS, ISL, OSL, MTP, Mode), only the feature toggle differs.
    return (on_cell.tpot_avg_ms - off_cell.tpot_avg_ms) / off_cell.tpot_avg_ms
# Negative = ON faster than OFF = feature wins on per-token latency.
# Pair with req/s ratio: cells where ΔTPOT < 0 AND Δreq/s > 0 are unambiguous wins.
```

## Usage examples — 5 dispatch walkthroughs

Each example shows the full chain: **user prompt → 3-axis dispatch → concrete commands → expected outputs → combo-specific failure modes**. Copy-paste ready as templates.

### Example 1 — Production Pareto on B300 Flash with real dataset

**User**: "在 lyris1 上跑 Flash 的全 Pareto 曲线，对比 NVFP4 vs 现有 baseline"

Dispatch:

| Axis | Value | Reason |
|---|---|---|
| HW | B300 | "lyris1" → GB300 |
| Model | Flash | explicit |
| Dataset | real | "生产 Pareto" → random-v2 ratio=0.8 |
| Recipe | F1 | (B300, Flash, real) lookup |
| Driver | Workflow A (sbatch) | recipe F1 default |

Commands:

```bash
# 1. Pre-flight (all 8 items from the Pre-flight section above).
ssh lyris1 "sshare -u \$USER --format=Account,FairShare -P"      # FairShare pick
ssh lyris1 "ls /lustre/fsw/coreai_comparch_inferencex/models/dsv4-flash | head"

# 2. Submit Recipe F1 on lyris1 (152 sbatch jobs, auto-scancel on success):
ssh lyris1 bash /lustre/fsw/coreai_comparch_infbench/xqiao/bench-mewtwo/agg_bench/submit_flash_full_lyris1.sh

# 3. Report (R1) after ~6 h:
ssh lyris1 "bash agg_bench/gather_perf.sh logs_<TS>/" > /tmp/lyris1_all.txt
python3 bench-dsv4/agg_bench/report_perf.py \
    --primary lyris1:/tmp/lyris1_all.txt --label flash_full --account ...
# → bench-dsv4/agg_bench/report_lyris1_<YYYYMMDD>_flash.html
```

Combo failure modes: BS=512+1024 DEP4 conc OOM (G7) → set kv=0.8 for the bad cell. Flash on lyris1 does NOT hit G3 (Pro-only DEP4 hang). No `--gres=gpu:4` needed (gb300 partition).

### Example 2 — Kernel A/B on B300 Flash with synthetic dataset (the GVR pattern)

**User**: "测一下我的新 Heuristic Top-K 在 V4 上是不是 win，跨 BS/MTP 全比较一遍"

Dispatch: (B300, Flash, synthetic, B-flash-B300, Workflow B). HW + Model defaulted because user didn't specify; dataset inferred from "比较 ON/OFF" needing low noise floor.

```bash
hostname  # confirm umb-b300-dp-*; if not, ssh there
nvidia-smi --query-compute-apps=pid --format=csv,noheader   # GPU clean
cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/perf_logs/pareto_v4_flash_gvr

# Feature toggle already in plan.csv (GVRS=[True,False] paired ON/OFF).
setsid nohup env MAX_WALL_SECONDS=21600 \
  ISL_FILTER=131072 MTP_FILTER=3 \
  bash scripts/02_master.sh > driver_logs/driver_$(date -u +%Y%m%dT%H%M%SZ).log 2>&1 < /dev/null &
disown

# Aggregate (R2):
python3 ${PERFDIR}/scripts/03_summarize.py
# → results/REPORT.md + figures/dtpot_isl_bs_<Mode>_M<MTP>.png
```

Combo failure modes (will hit, plan for them):
- BS≥8 DEP `Feature=false` (Radix path) → G4 stack crash (DeepGEMM @ BS=8 / cuBLAS @ BS≥16).
- BS≥16 DEP `Feature=true` (Heuristic) → G5 numRows-saturated 5-10× slowdown.
- Apply hang/crash recovery: GPU 0% + log mtime > 5 min → kill timeout wrapper.

### Example 3 — Pro full sweep on B200 with real dataset

**User**: "在 dfw 上测 Pro 模型的全 sweep，对比 MTP=0/1/2/3"

Dispatch: (B200, Pro, real, P1+B200deltas, Workflow A).

P1 deltas for B200:
- `max_num_tokens: 8192` (TP4) / `16384` (TP8) — B200 VRAM cap
- `kv_frac=0.8` on DEP, `0.9` on TEP
- DEP4 strictly MTP=0 (G3 — applies to ALL clusters, not lyris1-only)
- `--gres=gpu:4` required (dfw partition=batch)

```bash
ssh dfw "sshare -u \$USER --format=Account,FairShare -P"
# Edit submit script's account= to FairShare winner, then:
ssh dfw bash /lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/xqiao/bench-mewtwo/agg_bench/submit_pro_full_dfw.sh
```

Combo failure modes: G3 (Pro DEP4 MTP>0 hang — submit script MUST enforce MTP=0 for DEP4), G7 (dfw `dep8 conc=2048 mtp=0 kv=0.9` OOM → drop to kv=0.8).

### Example 4 — Cross-cluster overlay (B200 + B300 Pro real)

**User**: "我想看 Pro 在 B300 vs B200 上的 Pareto frontier 对比图，跨集群 overlay"

Dispatch: two sweeps + R1 overlay.

```bash
# Sweep B200 + Sweep B300:
ssh dfw    bash bench-mewtwo/agg_bench/submit_pro_full_dfw.sh        # → logs_<TS_dfw>
ssh lyris1 bash bench-mewtwo/agg_bench/submit_pro_full_lyris1.sh     # → logs_<TS_lyris1>

# Overlay report (use --compare for cross-cluster):
ssh dfw    "bash agg_bench/gather_perf.sh logs_<TS_dfw>"    > /tmp/dfw_all.txt
ssh lyris1 "bash agg_bench/gather_perf.sh logs_<TS_lyris1>" > /tmp/lyris1_all.txt
python3 bench-dsv4/agg_bench/report_perf.py \
    --primary lyris1:/tmp/lyris1_all.txt \
    --compare dfw:/tmp/dfw_all.txt \
    --label pro_b300_vs_b200 --account ...
```

Combo failure modes: not strict apples-to-apples (B200 caps max_num_tokens lower) — footnote the difference in the report; expect FP8 GEMM kernel selection to differ between sm_100 and sm_103 (real signal, not bug).

### Example 5 — Dispatcher asks when user under-specifies

**User**: "测一下 Flash 的性能"

SKILL response per dispatch protocol:

> Need to resolve 3 axes before planning:
> 1. **Hardware**: B200 (GB200, OCI/dfw) or B300 (GB300, lyris1 / single-node)?
> 2. **Dataset mode**: `synthetic` (token-norm-dist, low-noise paired ON/OFF, single-node OK) or `real` (random-v2 ratio=0.8, production Pareto, needs cluster)?
> 3. **Goal**: reproduce Xianjie's frontier (→ A) or validate a feature ON/OFF (→ B)?

If user replies "B300 + GVR comparison" → auto-completes to **(B300, Flash, synthetic, B-flash-B300, Workflow B)** = Example 2 path.

### Cheat sheet

```
User句式                                      → (HW, Model, Dataset, Recipe, Workflow, Host)
─────────────────────────────────────────────────────────────────────────────────────────────
"reproduce Xianjie's Flash frontier"          → (B300, Flash, real, F1,           A, lyris1)
"reproduce Xianjie's Pro frontier"            → (B300, Pro,   real, P1,           A, lyris1)
"Flash bench on dfw"                          → (B200, Flash, real, F1,           A, dfw)
"Pro bench on OCI"                            → (B200, Pro,   real, P1+B200deltas, A, OCI)
"is GVR a win on Flash"                       → (B300, Flash, synth, B-flash-B300, B, umb-b300-dp-*)
"is feature X a win on Pro"                   → (B300, Pro,   synth, B-pro-B300,   B, B300 interactive)
"cross-cluster Pareto for Pro"                → (B200+B300, Pro, real, P1×2 + R1 overlay)
"BS=4 MTP=3 Flash check"                      → (B300 interactive, Flash, synth, B-flash-B300 filtered)
"Pareto curve" (no other detail)              → dispatcher asks the 3 axis questions first
```

## Production deployment recommendations distilled from prior sweeps

### Flash (B200 GB200 sm_100 / B300 GB300 sm_103)

| Scenario | Recommended config |
|---|---|
| Throughput-prioritized server | TEP4 max_bs=256 kv=0.9 MTP=1 |
| Latency-prioritized server    | TEP4 max_bs=8  kv=0.9 MTP=3 |
| Long-context BS=1             | **DEP4 + MTP=3** (highest per-user TPS at long ISL) |
| **GVR enable_heuristic_topk** | ON for BS ∈ [2, 8]; OFF for BS=1 corner AND BS≥16 (U-shaped curve at MTP=3 TEP) |

### Pro (B200 GB200 sm_100 / B300 GB300 sm_103; canonical cluster=lyris1)

| Scenario | Recommended config |
|---|---|
| Throughput-prioritized server | DEP8 max_bs=1024 kv=0.8 MTP=1 |
| Latency-prioritized server    | TEP4 max_bs=256  kv=0.9 MTP=3 |
| Single-node throughput        | TEP8 max_bs=512 kv=0.9 MTP=2 |
| Mid-BS DEP4                   | ONLY MTP=0; MTP=1/2/3 hang on lyris1 (G3) |
| Pro accuracy-critical path    | `cuda_graph_config: null` + MTP=0 (CUDA-graph + sparse-MLA OOB bug, see GVR_ON_VALIDATION_REPORT.md) |

## Cross-references

- **Xianjie's bench-dsv4**: `/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/bench-dsv4/` (cloned 2026-05-18). Cluster-native sweep + cross-cluster HTML.
- **Our sweep**: `/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/perf_logs/pareto_v4_flash_gvr/` (PR §6 GVR validation source). Paired ON/OFF workflow.
- **Sister skill**: `dsv4-gsm8k-eval` (this repo, `.claude/skills/dsv4-gsm8k-eval/`) — accuracy eval, complementary to perf.
- **Session state (master log of all DSv4 work)**: `/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/session-state-20260511/RESUME.md`.
- **Source code**:
  - `tensorrt_llm/_torch/models/modeling_deepseekv4.py` — V4 model
  - `tensorrt_llm/_torch/attention_backend/sparse/deepseek_v4/` — sparse MLA + indexer
  - `cpp/tensorrt_llm/kernels/indexerTopK.cu` — GVR vs Radix dispatcher (Scheme X log lines)
  - `cpp/tensorrt_llm/kernels/heuristicTopKDecode.{cu,h}` — Heuristic Top-K kernel
