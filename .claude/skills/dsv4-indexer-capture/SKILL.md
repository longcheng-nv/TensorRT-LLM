---
name: dsv4-indexer-capture
description: >
  Drive a single-prompt, BS=1, greedy end-to-end DSv4 inference and dump
  per-layer `(indexer logits, indexer top-K)` data from the real production
  GVR path. Parameterized over prompt (raw text or jsonl[idx]), model
  variant (Flash|Pro|raw path), phase (prefill|decode|both), OSL, save
  format (pt|npz), per-layer vs single-file layout, layer subset, and
  number of GPUs. Wraps the q9j worktree hook in
  `TensorRT-LLM-q9j/.../dsa.py` so the capture lives in CPU memory through
  generation and is atexit-flushed. Strictly a DATA-collection skill, NOT
  a performance benchmark — `cuda_graph_config=null`, `enable_heuristic_topk=true`,
  `temperature=0.0`. Mirrors the Q9j (Flash) / Q9k (Pro) capture
  methodology in `auto_optimization_v1/.../09_precision_ablation/11_*/` and
  `.../12_*/` and supersedes their hard-coded TP=EP=8 + swe_bench_*.jsonl
  invocations. Trigger keywords: "抓 dsv4 indexer logits", "dump V4
  Flash/Pro per-layer top-K", "capture indexer logits for prompt …",
  "harvest dsv4 indexer data per layer", "real-loop indexer capture".
license: LicenseRef-NvidiaProprietary
metadata:
  author: loncheng@nvidia.com
  parent_studies:
    - auto_optimization_v1/.../09_precision_ablation/11_dsv4_trtllm_indexer_data_capture (Q9j, Flash K=512)
    - auto_optimization_v1/.../09_precision_ablation/12_dsv4_pro_indexer_data_capture     (Q9k, Pro K=1024)
  worktree: /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM-q9j
  hook_file: tensorrt_llm/_torch/attention_backend/sparse/dsa.py
  hook_version: v2 (2026-06-02)
---

# DSv4 indexer logits + top-K capture (Flash / Pro)

## What this skill does

Run **one** real prompt through DSv4 (Flash or Pro) end-to-end with
GVR enabled, and harvest these tensors on TP rank 0:

| Stream | When | Shape | Default keyed by |
|---|---|---|---|
| `prefill.logits.in` | BEFORE `indexer_topk_prefill` (each chunk) | fp32 `[num_q_in_chunk, kv_len_chunk]` | `(layer, chunk_idx)` |
| `prefill.topk.out`  | AFTER prefill block finishes | int32 `[num_ctx, K]` | `layer` |
| `decode.logits.in`  | BEFORE `indexer_topk_decode` (each step) | fp32 `[kv_len_at_step]` | `(layer, step)` |
| `decode.preidx.in`  | BEFORE `indexer_topk_decode` (GVR warm-start) | int32 `[K]` | `(layer, step)` |
| `decode.topk.out`   | AFTER `indexer_topk_decode` | int32 `[K]` | `(layer, step)` |

All tensors are CPU-cloned at the call site (inside dsa.py), accumulated
in a process-local dict, and flushed at `atexit` to one of two layouts:

- **single-file** (default, v1-compat): one file per stream
- **per-layer**: one subdir per layer with one file per stream

Format is selectable: `.pt` (PyTorch dict) or `.npz` (NumPy archive
with `L<l>_S<s>` keys).

## When to use

| ✅ Use | ❌ Don't use |
|---|---|
| Single prompt, single capture for downstream offline analysis | Throughput / latency benchmarks → use `dsv4-pareto-bench` |
| Want per-layer GVR `(logits, preIdx, topK)` triples | Want decode TPS / TTFT numbers → use `dsv4-nsys-profile` |
| Want to feed Q9j/Q9k-style A1/A2/A4 distribution analyses | Quantization accuracy → use `dsv4-gsm8k-eval` |
| First time on a new prompt that isn't in `swe_bench_*.jsonl` | Re-running the exact 21-layer × 300-step Q9j corpus → use `11_/src/run_capture.sh` (frozen) |

## Hard preconditions

0. **Works in a standard "tensorrt_llm already installed" env**, INCLUDING
   editable installs (e.g. `pip install -e .` on the main checkout).
   The launcher injects `PYTHONPATH=<q9j-worktree>:…` + `PYTHONSAFEPATH=1`,
   which triggers the worktree's `sitecustomize.py` to drop the
   `_EditableFinder` from `sys.meta_path` so `import tensorrt_llm` resolves
   to the worktree (where the capture hook lives). Verified by:
   ```bash
   PYTHONPATH=$WT:$PYTHONPATH PYTHONSAFEPATH=1 \
     python3 -c "import tensorrt_llm; print(tensorrt_llm.__file__)"
   # → /…/TensorRT-LLM-q9j/tensorrt_llm/__init__.py
   ```

1. **q9j worktree must exist + contain the v2 hook + `sitecustomize.py`**.
   Confirm:
   ```bash
   WT=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM-q9j
   test -f $WT/sitecustomize.py && \
   grep -q "Q9j capture hook v2" $WT/tensorrt_llm/_torch/attention_backend/sparse/dsa.py
   ```
   If missing, see [§Bootstrap](#bootstrap-creating-the-worktree-from-scratch).

2. **fast_hadamard_transform** must be importable (DSv4 will crash mid-run otherwise).

3. **CUDA Graph disabled** — the hook does `.cpu()` per layer per step, which is a
   host-device sync illegal during graph capture. The driver script sets
   `cuda_graph_config=None` already.

4. **GVR enabled** — `enable_heuristic_topk=True` is the only path the hook
   intercepts. If you want Radix-path captures, this skill does NOT apply
   without an additional hook on the `else:` branch in dsa.py.

5. **TP rank 0 only writes.** All ranks see identical post-reduction tensors.
   Multi-GPU is supported but only rank 0 produces output.

6. **BS = concurrency = num_request = 1** is fixed. This is a data-collection
   skill, not a perf skill.

## CLI

Invoke via `src/launch_capture.sh`:

```bash
launch_capture.sh \
  --model         flash|pro|<absolute-path>    # required
  --prompt        "<raw text>"  |  @<jsonl-path>[#idx]    # required (one of)
  --osl           300                           # max_new_tokens, default 300
  --phase         prefill|decode|both           # default both
  --layers        all|even|<comma-list>         # default even (GVR-active subset)
  --num-gpus      1|2|4|8                       # default 8
  --save-format   pt|npz                        # default pt
  --layout        single-file|per-layer         # default per-layer
  --out-dir       <dir>                         # default ./capture_<UTC>_<model>_<phase>
  --index-topk    auto|512|1024                 # default auto (= config.json value)
  --kv-cache-frac 0.7                           # default 0.7
```

### Examples

**Flash, full both-phase capture from raw text, per-layer .pt:**
```bash
src/launch_capture.sh \
  --model flash \
  --prompt "Write a Python function that reverses a string." \
  --osl 64 \
  --phase both \
  --num-gpus 8 \
  --out-dir /tmp/cap_flash_raw_demo
```

**Pro, decode-only K=1024 capture from swe_bench_64k, .npz per-layer:**
```bash
src/launch_capture.sh \
  --model pro \
  --prompt "@/home/scratch.loncheng_gpu/workspace/tllm_toolbox/indexer_topK_perf/data_distri/deepseek-v3.2-logging/tasks/swe_bench_64k.jsonl#0" \
  --osl 300 \
  --phase decode \
  --save-format npz \
  --num-gpus 8
```

**Flash, prefill-only on 4 GPUs (TP=EP=4), specific layers, single-file legacy layout:**
```bash
src/launch_capture.sh \
  --model flash \
  --prompt "@swe_bench_32k.jsonl#0" \
  --osl 1 \
  --phase prefill \
  --layers 2,20,22,36,42 \
  --num-gpus 4 \
  --layout single-file
```

## Output layout

### `single-file` (legacy / Q9j-compat)

```
<out-dir>/
  manifest.json
  q9j_logits.in.pt              # dict[(layer, step) -> fp32 tensor]    (decode)
  q9j_preidx.in.pt              # dict[(layer, step) -> int32 tensor]   (decode)
  q9j_topk.out.pt               # dict[(layer, step) -> int32 tensor]   (decode)
  q9j_prefill.logits.in.pt      # dict[(layer, chunk_idx) -> fp32]      (new in v2)
  q9j_prefill.topk.out.pt       # dict[layer -> int32 [num_ctx, K]]     (legacy)
```

`Q9jCapture` in `11_/src/q9j_load.py` reads this layout (only the v1 files —
the new `q9j_prefill.logits.in.pt` is ignored by the legacy loader).

### `per-layer` (default, recommended for new analyses)

```
<out-dir>/
  manifest.json
  layer_02/
    prefill.logits.in.pt   # dict[chunk_idx -> fp32 [num_q, kv_len]]
    prefill.topk.out.pt    # int32 [num_ctx, K]   (single tensor, not a dict)
    decode.logits.in.pt    # dict[step -> fp32 [kv_len_at_step]]
    decode.preidx.in.pt    # dict[step -> int32 [K]]
    decode.topk.out.pt     # dict[step -> int32 [K]]
  layer_04/ ...
  ...
```

With `--save-format npz`, the `.pt` extensions become `.npz` (each containing
`L<layer>_S<step>` keys, or `L<layer>` for scalar tensors).

## Manifest

`manifest.json` is written by the driver before exit and contains every
parameter that affects the capture, so downstream analysis can reconstruct
the run context:

```json
{
  "model_variant": "flash",
  "model_path": "/dev/shm/DeepSeek-V4-Flash",
  "prompt_source": "@/path/to/swe_bench_32k.jsonl#0",
  "prompt_token_count": 34376,
  "max_new_tokens": 300,
  "actual_output_tokens": 297,
  "temperature": 0.0,
  "phase": "both",
  "layers_logged": [2,4,6,...,42],
  "tp_size": 8, "ep_size": 8,
  "index_topk": 512,
  "max_num_tokens": 50000,
  "kv_cache_frac": 0.7,
  "sparse_attention": "deepseek_v4",
  "enable_heuristic_topk": true,
  "mtp": "off",
  "cuda_graph": "disabled",
  "save_format": "pt",
  "layout": "per-layer",
  "capture_dir": "...",
  "elapsed_seconds": 56.5,
  "hook_version": "v2",
  "worktree_sha": "<git rev-parse HEAD>"
}
```

## Auto-derived knobs

The driver computes these from the prompt + model variant so the user
doesn't need to:

| Knob | Rule |
|---|---|
| `model_path` | `flash` → first existing of `/dev/shm/DeepSeek-V4-Flash`, `/raid/data/$USER-stage/DeepSeek-V4-Flash`, `/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V4-Flash`, `/home/scratch.jinshik_gpu/DeepSeek-V4-Flash`. Same for `pro` with `DeepSeek-V4-Pro`. |
| `index_topk` | Read from `config.json::index_topk` (Flash=512, Pro=1024) unless `--index-topk` overrides. |
| `layers` | `even` → `tuple(range(2, num_hidden_layers, 2))` (Flash: 2..42; Pro: 2..60). `all` → `range(num_hidden_layers)`. `<list>` → as given. |
| `max_num_tokens` | `prompt_token_count + osl + 1024` rounded up to nearest 1024. |
| `ep_size` | = `tp_size` (DSv4 production layout). |
| `kv_cache_frac` | 0.7 by default (matches Pro 100K headroom). |

## Phase semantics

- **prefill**: captures `prefill.logits.in` (per chunk) + `prefill.topk.out`
  (per layer). Decode hooks are skipped.
- **decode**: captures `decode.logits.in/preidx.in/topk.out` (per step).
  Prefill snapshot is skipped.
- **both** (default): all 5 streams.

**Note**: even with `--phase prefill`, the model still generates `osl` tokens
(you can set `--osl 1` to minimize wasted decode work). The hook simply
doesn't dump decode tensors.

## Bootstrap (creating the worktree from scratch)

If `TensorRT-LLM-q9j/` does not exist on this host yet, or its `dsa.py`
lacks the v2 hook, do this **once**:

```bash
WT=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM-q9j
MAIN=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM

# 1. Clone (or worktree-link) the main checkout to the q9j path.
#    The cheapest path is a git worktree on the SAME branch the user runs.
git -C $MAIN worktree add $WT HEAD
#    Or full clone if worktree-add is unsuitable:
#    git clone $MAIN $WT

# 2. Copy sitecustomize.py from THIS skill (or hand-author per the §Hook
#    layout shown below). It must drop `_EditableFinder` and prepend $WT
#    to sys.path before user code runs.
test -f $WT/sitecustomize.py || \
    cp $MAIN/.claude/skills/dsv4-indexer-capture/src/sitecustomize.py $WT/  # if shipped

# 3. Apply the v2 hook patch to $WT/tensorrt_llm/_torch/attention_backend/sparse/dsa.py.
#    The diff is the difference between v1 (upstream) and v2 (this skill).
#    See §Hook layout for the call sites — patch into the same dsa.py the
#    user just cloned. If a tracked diff exists at
#    `.claude/skills/dsv4-indexer-capture/patches/dsa_hook_v2.diff`,
#    apply it; otherwise paste the v2 hook block + 3 call-site edits
#    documented in §Hook layout.

# 4. Sanity check the redirect:
PYTHONPATH=$WT:$PYTHONPATH PYTHONSAFEPATH=1 \
    python3 -c "import tensorrt_llm, sys; \
        print(tensorrt_llm.__file__); \
        assert '/TensorRT-LLM-q9j/' in tensorrt_llm.__file__; \
        assert not any('Editable' in type(f).__name__ for f in sys.meta_path); \
        print('OK')"
```

**Why a separate worktree** instead of editing the main checkout:
the main checkout is the editable-installed `tensorrt_llm` used by every
other DSv4 workflow on this host (Pareto sweeps, GSM8K, nsys, etc.).
Direct edits would inject `.cpu()` syncs into all of them, breaking
CUDA-Graph runs and slowing throughput benchmarks. The worktree +
sitecustomize.py + env-gated `DSV4_INDEXER_CAPTURE_DIR` design keeps the
capture cost strictly opt-in per process.

## Hook layout

The v2 hook lives in:

`/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM-q9j/tensorrt_llm/_torch/attention_backend/sparse/dsa.py`

| Block | Lines | Function |
|---|---|---|
| Hook config & buffers | ~85-185 | env parsing, buffer dicts, save helpers |
| Prefill `logits` capture (chunked path) | inside `for _q9j_chunk_idx, chunk in enumerate(...)` | calls `_q9j_log_prefill_logits` after `_call_mqa_logits` |
| Prefill `logits` capture (single-pass fallback) | after the fallback `_call_mqa_logits` | calls `_q9j_log_prefill_logits` with chunk_idx=0 |
| Prefill `topk` snapshot | `_q9j_log_prefill(self.layer_idx, topk_indices_buffer[:num_ctx_tokens, :])` | legacy Plan B path |
| Decode `(logits, preIdx)` capture | BEFORE `indexer_topk_decode` | `_q9j_log("logits.in", ...)`, `_q9j_log("preidx.in", ...)` |
| Decode `topk` capture | AFTER `indexer_topk_decode` | `_q9j_log("topk.out", ...)` |
| atexit flush | `_q9j_flush()` | branches on `_Q9J_LAYOUT` + `_Q9J_FORMAT` |

The hook is no-op unless `DSV4_INDEXER_CAPTURE_DIR` is set. Default phase
is `both` (matches v1 behavior); default layout is `single-file` (v1-compat);
default format is `pt`. **The driver script always overrides defaults
explicitly via env vars** so the run record matches the CLI flags.

## Gotchas (G1–G7)

- **G1 — `/dev/shm` model wins**: if `/dev/shm/DeepSeek-V4-{Flash,Pro}` exists,
  use it; it's faster than `/raid` and dodges the NFS cold-read.
- **G2 — Pro 100K previously failed**: 11_/12_ historical Pro-100K captures
  show 4 KB empty dirs — atexit flush raced with MPI teardown on multi-GB
  prefill data. Workaround: keep ISL ≤ 64K for prefill captures, or set
  `--phase decode` so prefill snapshot is skipped.
- **G3 — Chunked prefill rank-0 slice**: prefill `logits` captured under
  `q_split` is only this rank's slice (chunk[chunk_q_start:chunk_q_end]).
  For full-prompt prefill logits across all ranks, capture only when
  `tp_size==1` or aggregate offline using the manifest's `tp_size`.
- **G4 — DeepseekV4Tokenizer chat template**: HF AutoTokenizer **cannot** be
  used directly — the template is baked into
  `tensorrt_llm.tokenizer.deepseek_v4.DeepseekV4Tokenizer`. The driver uses
  this; raw `--prompt` text is passed verbatim into one chat message.
- **G5 — `enable_heuristic_topk=True` is mandatory**. With it off, the
  decode path takes a different branch and no hook fires.
- **G6 — `cuda_graph_config=null` is mandatory**. Setting cuda_graph_config
  to a non-null value silently disables decode capture (the hook detects
  `is_current_stream_capturing()` and short-circuits).
- **G7 — TP=EP=N pairing**: DSv4 production runs use TP=EP. The driver
  forces `ep_size = tp_size`. To break this, edit `run_capture.py` directly.

## Size estimates

For Flash K=512, 21 layers, 1 prompt × 1024 decode steps:

| Stream | Per cell | Cells | Subtotal |
|---|---|---|---|
| `decode.logits.in` (fp32) | ~30 KB (avg kv_len) | 21 × 1024 | ~640 MB |
| `decode.preidx.in` (int32) | 2 KB | 21 × 1024 | ~43 MB |
| `decode.topk.out` (int32) | 2 KB | 21 × 1024 | ~43 MB |
| `prefill.logits.in` (fp32) | ~70 MB per chunk | 21 × 9 chunks | ~13 GB |
| `prefill.topk.out` (int32) | 70 MB (num_ctx × K) | 21 | ~1.5 GB |

**Decode-only**: ~750 MB. **Both-phase ISL≤32K**: ~15 GB. **Both-phase
ISL=100K**: 40+ GB and risks G2 flush race.

## File layout under this skill

```
.claude/skills/dsv4-indexer-capture/
  SKILL.md              # this file
  src/
    launch_capture.sh   # bash wrapper (env setup + worktree validation)
    run_capture.py      # python driver
```

## See also

- `dsv4-pareto-bench` — for throughput/latency sweeps, NOT data dumps.
- `dsv4-nsys-profile` — for nsys profiling, also NOT data dumps.
- `swebench-temporal-synth-v4flash` / `-v4pro` — synthesize realistic
  `(logits, preIdx)` without running the model; uses the captures
  produced by THIS skill as fit data.
- `11_dsv4_trtllm_indexer_data_capture/REPORT.md` — Q9j Flash all-layers
  + cross-prompt analyses derived from these captures.
- `12_dsv4_pro_indexer_data_capture/REPORT.md` — Q9k Pro K=1024 / K=512
  cross-arch validation.
