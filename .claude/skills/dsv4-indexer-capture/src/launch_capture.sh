#!/usr/bin/env bash
# DSv4 indexer capture launcher.
#
# Sets up the env so `import tensorrt_llm` resolves to the q9j worktree,
# expands --model / --layers / --phase / --layout / --save-format into the
# env vars the dsa.py v2 hook expects, then invokes run_capture.py.
#
# Zero impact on the shared TRT-LLM editable install: all redirection is
# per-process via PYTHONPATH + PYTHONSAFEPATH + DSV4_INDEXER_CAPTURE_* envs.
set -euo pipefail

# === Defaults ===
WT=${Q9J_WORKTREE:-/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM-q9j}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_PY="${SCRIPT_DIR}/run_capture.py"

MODEL=""
PROMPT=""
OSL=300
PHASE="both"
LAYERS_SPEC="even"
NUM_GPUS=8
SAVE_FORMAT="pt"
LAYOUT="per-layer"
OUT_DIR=""
INDEX_TOPK="auto"
KV_FRAC="0.7"

usage() {
  cat <<EOF
Usage: $0 [options]

  --model         flash|pro|<abs-path>            (required)
  --prompt        "<raw text>" | @<jsonl>[#idx]   (required)
  --osl           N             default ${OSL}
  --phase         prefill|decode|both             default ${PHASE}
  --layers        all|even|<csv>                  default ${LAYERS_SPEC}
  --num-gpus      1|2|4|8                         default ${NUM_GPUS}
  --save-format   pt|npz                          default ${SAVE_FORMAT}
  --layout        single-file|per-layer           default ${LAYOUT}
  --out-dir       <dir>                           default auto-generated
  --index-topk    auto|512|1024                   default ${INDEX_TOPK}
  --kv-cache-frac <float>                         default ${KV_FRAC}
  -h | --help

Example:
  $0 --model flash --prompt "Hello world" --osl 32 --phase both \\
     --num-gpus 8 --save-format pt --layout per-layer \\
     --out-dir /tmp/cap_demo
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)         MODEL="$2"; shift 2 ;;
    --prompt)        PROMPT="$2"; shift 2 ;;
    --osl)           OSL="$2"; shift 2 ;;
    --phase)         PHASE="$2"; shift 2 ;;
    --layers)        LAYERS_SPEC="$2"; shift 2 ;;
    --num-gpus)      NUM_GPUS="$2"; shift 2 ;;
    --save-format)   SAVE_FORMAT="$2"; shift 2 ;;
    --layout)        LAYOUT="$2"; shift 2 ;;
    --out-dir)       OUT_DIR="$2"; shift 2 ;;
    --index-topk)    INDEX_TOPK="$2"; shift 2 ;;
    --kv-cache-frac) KV_FRAC="$2"; shift 2 ;;
    -h|--help)       usage; exit 0 ;;
    *) echo "[launch] unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$MODEL" ]]  || { echo "[launch] FAIL: --model required" >&2; exit 2; }
[[ -n "$PROMPT" ]] || { echo "[launch] FAIL: --prompt required" >&2; exit 2; }

# === Preflight ===
echo "[launch] preflight..."

[[ -d "$WT" ]] || {
    cat >&2 <<EOF
  FAIL: q9j worktree missing: $WT

  The skill REQUIRES a separate TRT-LLM checkout at this path holding
  the capture hooks. To bootstrap (one-time, per host):

    git clone /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM \\
      $WT
    # Then apply the v2 hook patches (see SKILL.md §Hook layout).
EOF
    exit 1
}
[[ -f "$WT/tensorrt_llm/_torch/attention_backend/sparse/dsa.py" ]] \
    || { echo "  FAIL: dsa.py missing in worktree" >&2; exit 1; }
[[ -f "$WT/sitecustomize.py" ]] || {
    echo "  FAIL: $WT/sitecustomize.py missing — without it, PYTHONPATH cannot win over an editable install. See SKILL.md §Bootstrap." >&2
    exit 1
}
grep -q "Q9j capture hook v2" \
    "$WT/tensorrt_llm/_torch/attention_backend/sparse/dsa.py" \
    || { echo "  FAIL: dsa.py in worktree lacks v2 hook — see SKILL.md §Hook layout" >&2; exit 1; }
echo "  OK: dsa.py has Q9j v2 hook + sitecustomize.py present"

command -v nvidia-smi >/dev/null \
    || { echo "  FAIL: nvidia-smi not found" >&2; exit 1; }
NUM_VISIBLE=$(nvidia-smi -L | wc -l)
[[ "$NUM_VISIBLE" -ge "$NUM_GPUS" ]] \
    || { echo "  FAIL: --num-gpus=$NUM_GPUS but only $NUM_VISIBLE GPUs visible" >&2; exit 1; }
echo "  OK: $NUM_VISIBLE GPUs visible (using $NUM_GPUS)"

python3 -c "import fast_hadamard_transform" 2>/dev/null \
    || echo "  WARN: fast_hadamard_transform not importable — DSv4 will crash mid-run" >&2

# === Resolve model path early so we can read config.json for layer expansion ===
resolve_model() {
  local arg="$1"
  case "$arg" in
    flash)
      for p in \
          /dev/shm/DeepSeek-V4-Flash \
          "/raid/data/${USER}-stage/DeepSeek-V4-Flash" \
          /home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V4-Flash \
          /home/scratch.jinshik_gpu/DeepSeek-V4-Flash; do
        [[ -d "$p" ]] && { echo "$p"; return 0; }
      done
      echo "[launch] FAIL: no Flash checkpoint found" >&2; exit 1 ;;
    pro)
      for p in \
          /dev/shm/DeepSeek-V4-Pro \
          "/raid/data/${USER}-stage/DeepSeek-V4-Pro" \
          /home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V4-Pro; do
        [[ -d "$p" ]] && { echo "$p"; return 0; }
      done
      echo "[launch] FAIL: no Pro checkpoint found" >&2; exit 1 ;;
    *)
      [[ -d "$arg" ]] || { echo "[launch] FAIL: path missing: $arg" >&2; exit 1; }
      echo "$arg" ;;
  esac
}

MODEL_PATH=$(resolve_model "$MODEL")
echo "  OK: model = $MODEL_PATH"

# === Expand --layers via Python (reads config.json::compress_ratios) ===
LAYERS_CSV=$(python3 - "$MODEL_PATH" "$LAYERS_SPEC" <<'PY'
import json, sys
model_path, spec = sys.argv[1], sys.argv[2]
cfg = json.load(open(f"{model_path}/config.json"))
nh = cfg["num_hidden_layers"]
crs = cfg.get("compress_ratios")
if spec == "all":
    layers = list(range(nh))
elif spec == "even":
    layers = [i for i, cr in enumerate(crs)] if crs else []
    layers = [i for i, cr in enumerate(crs) if cr == 4] if crs else list(range(2, nh, 2))
else:
    layers = [int(x) for x in spec.split(",") if x.strip()]
print(",".join(str(x) for x in layers))
PY
)
echo "  OK: layers = $LAYERS_CSV"

# === Default out-dir ===
if [[ -z "$OUT_DIR" ]]; then
  TS=$(date -u +%Y%m%dT%H%M%SZ)
  MODEL_TAG=$(basename "$MODEL_PATH" | tr '[:upper:]' '[:lower:]')
  OUT_DIR="${SCRIPT_DIR}/../captures/capture_${TS}_${MODEL_TAG}_${PHASE}"
fi
mkdir -p "$OUT_DIR"
OUT_DIR=$(readlink -f "$OUT_DIR")
echo "  OK: out-dir = $OUT_DIR"

# === Env setup ===
export PYTHONPATH="$WT:${PYTHONPATH:-}"
export PYTHONSAFEPATH=1
export PYTHONDONTWRITEBYTECODE=1
export DSV4_INDEXER_CAPTURE_DIR="$OUT_DIR"
export DSV4_INDEXER_CAPTURE_LAYERS="$LAYERS_CSV"
export DSV4_INDEXER_CAPTURE_PHASE="$PHASE"
export DSV4_INDEXER_CAPTURE_LAYOUT="$LAYOUT"
export DSV4_INDEXER_CAPTURE_FORMAT="$SAVE_FORMAT"

echo ""
echo "[launch] env:"
echo "  PYTHONPATH=$WT:..."
echo "  DSV4_INDEXER_CAPTURE_DIR=$OUT_DIR"
echo "  DSV4_INDEXER_CAPTURE_LAYERS=$LAYERS_CSV"
echo "  DSV4_INDEXER_CAPTURE_PHASE=$PHASE"
echo "  DSV4_INDEXER_CAPTURE_LAYOUT=$LAYOUT"
echo "  DSV4_INDEXER_CAPTURE_FORMAT=$SAVE_FORMAT"
echo ""

# === Launch ===
exec python3 "$RUN_PY" \
    --model         "$MODEL_PATH" \
    --prompt        "$PROMPT" \
    --osl           "$OSL" \
    --phase         "$PHASE" \
    --layers        "$LAYERS_SPEC" \
    --num-gpus      "$NUM_GPUS" \
    --save-format   "$SAVE_FORMAT" \
    --layout        "$LAYOUT" \
    --out-dir       "$OUT_DIR" \
    --index-topk    "$INDEX_TOPK" \
    --kv-cache-frac "$KV_FRAC"
