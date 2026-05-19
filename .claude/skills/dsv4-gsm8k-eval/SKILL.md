---
name: dsv4-gsm8k-eval
description: Run GSM8K accuracy evaluation on DeepSeek-V4 (Flash / Flash-Base / Pro) with MTP speculative decoding + CUDA Graph on 8× Blackwell GPUs via trtllm-eval. Handles env preflight (fast-hadamard-transform, transformers pin), launches the eval as a background process, polls progress, reports final strict-match / flexible-extract scores. Use when the user asks to "run DSv4 GSM8K eval", "test DeepSeek-V4 Flash GSM8K accuracy", "reproduce 95.11 reference on B300", or similar.
---

# DeepSeek-V4 GSM8K accuracy eval

End-to-end GSM8K eval for DeepSeek-V4 Flash / Flash-Base / Pro on **8× Blackwell (sm_100 / sm_103)** with MTP + CUDA Graph, via the `trtllm-eval` CLI. Reference Flash accuracy on the in-tree YAML is **95.11**; an unconstrained 8× B300 run in this skill's session reproduced **96.85** (strict-match 96.89 / flexible-extract 96.82).

This skill assumes `tensorrt_llm` is already installed (editable or wheel) in the env. It does NOT build TRT-LLM.

## When to invoke

Triggers:
- "run DSv4 GSM8K eval" / "test DeepSeek-V4 GSM8K"
- "reproduce 95.11 reference on B300"
- "validate Flash / Flash-Base / Pro accuracy"
- After a code change to DSv4 attention / MoE / MTP, when the user wants a fast sanity check

## Pre-requisites (the skill must verify, not assume)

1. **Hardware**: 8 visible Blackwell GPUs (B200 SM 10.0 or B300 SM 10.3). DSv4 is Blackwell-only — fail loudly on other archs.
2. **`fast_hadamard_transform` Python package installed.** If missing, `import tensorrt_llm` succeeds and warmup completes, but inference crashes mid-run with `CUDA error: an illegal memory access` from `_torch/attention_backend/sparse/deepseek_v4/deepseek_v4.py:_compute_ctx_compressed_position_ids`. The startup warning `Sparse MLA will skip hadamard transformation` is the signal — treat it as a HARD ERROR for DSv4. Install with:
   ```bash
   pip install --user --no-build-isolation \
     git+https://github.com/Dao-AILab/fast-hadamard-transform.git
   ```
   `--no-build-isolation` is mandatory: the package's `setup.py` imports `torch` at build time.
3. **`transformers` version matches the branch's pin** (`requirements.txt`). The DSv4 branch pins `transformers==4.57.3`; running with `transformers>=5.x` causes `ImportError: cannot import name 'AutoModelForVision2Seq'` etc. at top-level `import tensorrt_llm`.
4. **Model checkpoint accessible** (one of):
   - Flash (Instruct, MXFP4 routed experts → MoE backend `TRTLLM`): `/home/scratch.jinshik_gpu/DeepSeek-V4-Flash`
   - Flash-Base (Completion, FP8 block-scale → MoE backend `WIDEEP`): `/home/scratch.jinshik_gpu/DeepSeek-V4-Flash-Base`
   - Pro (Instruct, MoE backend `TRTLLM`): `/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V4-Pro` (default; override via `PRO_PATH` on hosts where the CI mount is not present)
5. **GSM8K dataset accessible**: `/home/scratch.trt_llm_data/llm-models/datasets/openai/gsm8k` (or override via `DATASET_PATH`).

The preflight in `scripts/run_eval.sh` checks 1–5 and aborts with a clear message if anything is missing.

## Pipeline

### Stage 1 — preflight + launch

```bash
cd .claude/skills/dsv4-gsm8k-eval/scripts
DSV4_VARIANT=flash bash run_eval.sh           # default: Flash
DSV4_VARIANT=flash-base bash run_eval.sh      # Flash-Base (no chat template)
DSV4_VARIANT=pro bash run_eval.sh             # Pro (Instruct, like Flash)
```

`run_eval.sh` does:
1. Resolve `$DSV4_VARIANT` → model path + Instruct/Completion flag.
2. Verify all 5 preconditions; fail-fast if any missing. **DO NOT proceed if `fast_hadamard_transform` is missing.**
3. Emit the run-specific YAML to `/tmp/dsv4_gsm8k_<variant>_<timestamp>.yaml` (cuda_graph + MTP).
4. Launch `python3 -m tensorrt_llm.commands.eval` with the verified-good flag set:
   - `--max_batch_size 32 --max_num_tokens 8192 --max_seq_len 8192`
   - `--tp_size 8 --ep_size 8 --kv_cache_free_gpu_memory_fraction 0.8`
   - `--custom_tokenizer deepseek_v4` (Instruct only; OMIT for Flash-Base)
   - `gsm8k --apply_chat_template --system_prompt "<answer-format prompt>"` (Instruct only)
   - `--max_input_length 4096 --max_output_length 512`
   - Logs to `/tmp/dsv4_gsm8k_<variant>_<timestamp>.log`.
5. Print the launched PID and log path. Return immediately; eval keeps running.

ETA: **~10 min model init/warmup + ~3 min inference** for the full 1319-problem GSM8K set on 8× B300 (Flash, MXFP4 routed experts). Pro will be longer; Flash-Base similar.

### Stage 2 — monitor progress

The agent should poll the log file. Useful greps:

```bash
LOG=/tmp/dsv4_gsm8k_<variant>_<timestamp>.log
# Phase 1: weight load
grep "Model init total" "$LOG"
# Phase 2: warmup / autotune
grep "TRTLLM initialization time" "$LOG"
# Phase 3: inference progress (lm-eval prints tqdm to stderr → log)
tail -c 8192 "$LOG" | tr '\r' '\n' | grep "Fetching responses:" | tail -3
# Failure detection
grep -E "illegal memory access|RequestError|Traceback" "$LOG" | head
```

Use `nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader` to confirm liveness — log mtime stays stale during long phases because tqdm uses `\r`.

`scripts/check_progress.sh <log>` wraps these greps.

### Stage 3 — collect results

When the lm-eval table prints, the run is done. Extract:

```bash
grep -E "exact_match|gsm8k average accuracy" "$LOG" | tail -5
ls /tmp/tekit_gsm8k_script_outputs/<variant>/samples_gsm8k.json
```

The accuracy table includes both `strict-match` and `flexible-extract` filters at `n-shot=5`. Report both, plus the average.

Compare against:
- **In-tree YAML reference** (`tests/integration/defs/accuracy/references/gsm8k.yaml`): Flash 95.11 (TP=8 / 8× B200, fraction=0.15, cuda_graph=None, max_num_tokens=2048).
- **B300 TP=4 default-config measurement**: 95.38.
- **B300 TP=8 + MTP + CUDA Graph + Hadamard** (this skill's recipe): **96.85** observed.

Anything ≤ 93 → suspect missing fast-hadamard-transform, missing `--custom_tokenizer`, or wrong checkpoint type (Instruct flags applied to Base, or vice versa).

## Hard-coded configuration (don't change without asking the user)

These reproduce the verified 96.85 number. Changing them invalidates the comparison.

| Knob | Value |
|---|---|
| Parallelism | `tp=8, ep=8, pp=1` |
| `max_batch_size` / `max_num_tokens` / `max_seq_len` | `32 / 8192 / 8192` |
| `kv_cache_free_gpu_memory_fraction` | `0.8` |
| `cuda_graph_config.batch_sizes` | `[1, 2, 4, 8, 16, 32]` |
| `cuda_graph_config.enable_padding` | `false` |
| `speculative_config.decoding_type` | `MTP` |
| `speculative_config.num_nextn_predict_layers` | `3` |
| GSM8K subcommand `max_input_length / max_output_length` | `4096 / 512` |
| Instruct system prompt | "Solve the problem carefully. End your response with a final line exactly in the form `#### <answer>`, using the simplest numeric form without units or trailing zeros." |

Why these specific values:
- `max_seq_len ≥ max_input_length + max_output_length` (4096 + 512 = 4608); the eval CLI's `--max_seq_len` is **total**, not just prompt. Setting to 8192 aligns with `max_num_tokens` and leaves headroom.
- `enable_padding: false` keeps CUDA Graph specialized to exact batch sizes — `MAX_UTILIZATION` KV scheduler still works without it.
- MTP `num_nextn_predict_layers=3` matches the in-tree DSv4 default for serving runs.

## Anti-patterns / gotchas

- **`fast_hadamard_transform` is non-negotiable for DSv4 eval.** The startup warning is easy to miss; the crash is async and surfaces in `_compute_ctx_compressed_position_ids` (a `torch.arange` call) which makes the stack misleading. Install before launching.
- **Flash-Base ≠ Flash:** Flash-Base is a completion checkpoint. Do NOT pass `--custom_tokenizer deepseek_v4`, `--apply_chat_template`, or `--system_prompt` to Flash-Base — they are no-ops at best and chat-template-corrupt the prompt at worst. Drop the GSM8K accuracy floor by 10+ pts if applied wrongly.
- **MoE backend depends on quant**: Flash (MXFP4 routed) ⇒ `moe_config.backend: TRTLLM` (auto). Flash-Base / Pro-Base FP8 block-scale ⇒ `WIDEEP`. CUTLASS is Hopper-only on Blackwell; expect `Unsupported quantization mode: [65536]` if WIDEEP is forced on Flash.
- **Stuck MPI parent on rank-0 crash**: when a child rank dies from a CUDA error, the rank-0 Python process can hang on RPC waiting. Children release GPU memory, parent doesn't exit. After observing the crash, the agent must `kill -9` the entire process tree (`python3 -m tensorrt_llm.commands.eval` + its bash parents) before launching a retry. `nvidia-smi memory.used == 0` confirms cleanup.
- **`torch._dynamo hit config.recompile_limit (8)` on `prepare_drafter_inputs`** is benign for DSv4 MTP. Fallback to eager, minor perf hit, no correctness impact. Don't chase it.
- **lm-eval `--output_path` is a DIRECTORY, not a file.** The evaluator does `mkdir -p path; write path/samples_gsm8k.json`. Do not pass `.../samples_gsm8k.json` directly.

## Files in this skill

```
scripts/
├── run_eval.sh        # preflight + launch, parameterized by DSV4_VARIANT
└── check_progress.sh  # tail/grep helper for the live log
```

## Resume behavior

Each invocation produces a fresh timestamped log under `/tmp/dsv4_gsm8k_<variant>_<UTC>.log` and writes results under `/tmp/tekit_gsm8k_script_outputs/<variant>/`. There is no checkpoint/resume mid-run — if eval dies, kill the parent tree and re-launch. The full Flash run takes ~13 min, so re-run cost is low.
