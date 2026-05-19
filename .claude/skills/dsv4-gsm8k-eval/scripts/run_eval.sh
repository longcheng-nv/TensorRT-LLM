#!/usr/bin/env bash
# DSv4 GSM8K eval launcher.
# Preflight env + launch trtllm-eval. Prints PID + log path; eval runs in background.
#
# Env vars (all optional, sane defaults):
#   DSV4_VARIANT       flash | flash-base | pro                       (default: flash)
#   FLASH_PATH         override Flash checkpoint dir
#   FLASH_BASE_PATH    override Flash-Base checkpoint dir
#   PRO_PATH           override Pro checkpoint dir
#   DATASET_PATH       override GSM8K dataset dir
#   OUTPUT_ROOT        override results root  (default /tmp/tekit_gsm8k_script_outputs)
#   LOG_DIR            override log dir       (default /tmp)
#   SKIP_HADAMARD_CHECK=1  bypass the fast-hadamard-transform precheck (NOT recommended)

set -euo pipefail

VARIANT="${DSV4_VARIANT:-flash}"
FLASH_PATH="${FLASH_PATH:-/home/scratch.jinshik_gpu/DeepSeek-V4-Flash}"
FLASH_BASE_PATH="${FLASH_BASE_PATH:-/home/scratch.jinshik_gpu/DeepSeek-V4-Flash-Base}"
PRO_PATH="${PRO_PATH:-/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V4-Pro}"
DATASET_PATH="${DATASET_PATH:-/home/scratch.trt_llm_data/llm-models/datasets/openai/gsm8k}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/tmp/tekit_gsm8k_script_outputs}"
LOG_DIR="${LOG_DIR:-/tmp}"

GSM8K_SYS_PROMPT="Solve the problem carefully. End your response with a final line exactly in the form #### <answer>, using the simplest numeric form without units or trailing zeros."

# ---------------------------------------------------------------------------
# Resolve variant → model path + Instruct/Completion flag
# ---------------------------------------------------------------------------
case "$VARIANT" in
  flash)
    MODEL_PATH="$FLASH_PATH"
    IS_INSTRUCT=1
    ;;
  flash-base)
    MODEL_PATH="$FLASH_BASE_PATH"
    IS_INSTRUCT=0  # completion checkpoint — do NOT apply chat template
    ;;
  pro)
    MODEL_PATH="$PRO_PATH"
    IS_INSTRUCT=1
    ;;
  *)
    echo "ERROR: DSV4_VARIANT must be one of: flash, flash-base, pro (got '$VARIANT')" >&2
    exit 2
    ;;
esac

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
preflight_fail() {
  echo "PREFLIGHT FAILED: $*" >&2
  exit 3
}

# 1. GPU count + arch
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l) || preflight_fail "nvidia-smi not found"
[ "$NUM_GPUS" -ge 8 ] || preflight_fail "need 8 GPUs, found $NUM_GPUS"
ARCH=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -d '\n')
case "$ARCH" in
  *B200*|*B300*) ;;
  *) preflight_fail "DSv4 is Blackwell-only (need B200/B300). Got: $ARCH" ;;
esac

# 2. Model + dataset paths
[ -d "$MODEL_PATH" ] || preflight_fail "checkpoint dir not found: $MODEL_PATH (set ${VARIANT^^}_PATH to override)"
[ -f "$MODEL_PATH/config.json" ] || preflight_fail "no config.json under $MODEL_PATH"
[ -d "$DATASET_PATH" ] || preflight_fail "GSM8K dataset dir not found: $DATASET_PATH (set DATASET_PATH to override)"

# 3. tensorrt_llm importable
python3 -c "import tensorrt_llm" >/dev/null 2>&1 \
  || preflight_fail "import tensorrt_llm failed. Check transformers pin (DSv4 wants transformers==4.57.3) and env."

# 4. fast_hadamard_transform installed (DSv4 sparse MLA WILL crash without it)
if [ "${SKIP_HADAMARD_CHECK:-0}" != "1" ]; then
  python3 -c "import fast_hadamard_transform" >/dev/null 2>&1 \
    || preflight_fail "fast_hadamard_transform not installed. Without it, DSv4 sparse MLA falls back and crashes mid-eval with 'CUDA error: illegal memory access'. Install: pip install --user --no-build-isolation git+https://github.com/Dao-AILab/fast-hadamard-transform.git"
fi

# ---------------------------------------------------------------------------
# Per-run files
# ---------------------------------------------------------------------------
TS=$(date -u +%Y%m%dT%H%M%S)
LOG_FILE="${LOG_DIR}/dsv4_gsm8k_${VARIANT}_${TS}.log"
YAML_FILE="${LOG_DIR}/dsv4_gsm8k_${VARIANT}_${TS}.yaml"
OUTPUT_DIR="${OUTPUT_ROOT}/${VARIANT}"
mkdir -p "$OUTPUT_ROOT"

cat >"$YAML_FILE" <<'YAML'
cuda_graph_config:
  batch_sizes: [1, 2, 4, 8, 16, 32]
  max_batch_size: 32
  enable_padding: false
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 3
# Enable GVR heuristic Top-K for the DSv4 indexer (Blackwell-only).
sparse_attention_config:
  algorithm: deepseek_v4
  enable_heuristic_topk: true
YAML

# ---------------------------------------------------------------------------
# Build CLI
# ---------------------------------------------------------------------------
CMD=(python3 -m tensorrt_llm.commands.eval
  --model "$MODEL_PATH"
  --backend pytorch
  --max_batch_size 32
  --max_num_tokens 8192
  --max_seq_len 8192
  --kv_cache_free_gpu_memory_fraction 0.8
  --tp_size 8
  --ep_size 8
  --config "$YAML_FILE")

if [ "$IS_INSTRUCT" -eq 1 ]; then
  CMD+=(--custom_tokenizer deepseek_v4)
fi

CMD+=(gsm8k
  --dataset_path "$DATASET_PATH"
  --max_input_length 4096
  --max_output_length 512
  --output_path "$OUTPUT_DIR")

if [ "$IS_INSTRUCT" -eq 1 ]; then
  CMD+=(--apply_chat_template --system_prompt "$GSM8K_SYS_PROMPT")
fi

# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
echo "=== DSv4 GSM8K eval ==="
echo "variant : $VARIANT"
echo "model   : $MODEL_PATH"
echo "dataset : $DATASET_PATH"
echo "yaml    : $YAML_FILE"
echo "log     : $LOG_FILE"
echo "output  : $OUTPUT_DIR/samples_gsm8k.json"
echo
echo "cmd     : ${CMD[*]}"
echo

nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
PID=$!
echo "launched PID=$PID"
echo
echo "monitor with:"
echo "  bash $(dirname "$0")/check_progress.sh $LOG_FILE"
echo
echo "kill with:"
echo "  kill -9 $PID  # plus any 'python3 -m tensorrt_llm.commands.eval' child mpi workers"
