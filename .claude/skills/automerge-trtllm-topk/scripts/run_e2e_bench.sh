#!/bin/bash
# End-to-end TRT-LLM benchmark for heuristic TopK validation.
# Usage:
#   ./run_e2e_bench.sh [--profile] [--heuristic]
#
# Options:
#   --profile     Enable nsys profiling for decode iterations
#   --heuristic   Set enable_heuristic_topk=true (default: false)
#
# Environment variables (override defaults):
#   MODEL_CARD, MODEL_PATH, MAX_BATCH, ISL, OSL, EP, MTP, KV_FRAC, TRT_LLM_HOME

set -ex

MODEL_CARD="${MODEL_CARD:-deepseek-ai/DeepSeek-V3.2-Exp}"
MODEL_PATH="${MODEL_PATH:-/home/scratch.trt_llm_data/llm-models/DeepSeek-V3.2-Exp-FP4-v2/}"
MAX_BATCH="${MAX_BATCH:-1}"
ISL="${ISL:-4096}"
OSL="${OSL:-131072}"
EP="${EP:-8}"
MTP="${MTP:-1}"
KV_FRAC="${KV_FRAC:-0.8}"

ENABLE_PROFILE=false
ENABLE_HEURISTIC=false
for arg in "$@"; do
    case "$arg" in
        --profile)    ENABLE_PROFILE=true ;;
        --heuristic)  ENABLE_HEURISTIC=true ;;
    esac
done

NUM_PROMPTS=$((MAX_BATCH * 1))
MAX_NUM_TOKENS=$(( (MAX_BATCH + ISL + 128 + 63) / 64 * 64 ))

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TRT_LLM_HOME="${TRT_LLM_HOME:-$(cd "$SCRIPT_DIR/../../../../" && pwd)}"

sudo nvidia-smi -pm 0 2>/dev/null; sudo nvidia-smi -pm 1 2>/dev/null
sudo nvidia-smi boost-slider --vboost 4 2>/dev/null || true

HEUR_TAG=$( $ENABLE_HEURISTIC && echo "heuristic_on" || echo "heuristic_off" )
LOG_DIR="tmp/e2e_bench_B${MAX_BATCH}_ISL${ISL}_${HEUR_TAG}"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +'%m%d%H%M')
LOG_PREFIX="${LOG_DIR}/run_${TIMESTAMP}"

cat <<EOF > extra-llm-api-config.yml
cuda_graph_config:
    enable_padding: true
    max_batch_size: ${MAX_BATCH}
kv_cache_config:
    free_gpu_memory_fraction: ${KV_FRAC}
    enable_block_reuse: false
    tokens_per_block: 64
    dtype: fp8
enable_chunked_prefill: false
print_iter_log: true
enable_attention_dp: true
moe_config:
    backend: CUTLASS
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: ${MTP}
sparse_attention_config:
    algorithm: dsa
    enable_heuristic_topk: ${ENABLE_HEURISTIC}
EOF

python "${TRT_LLM_HOME}/benchmarks/cpp/prepare_dataset.py" \
    --tokenizer="${MODEL_PATH}" --stdout token-norm-dist \
    --num-requests="${NUM_PROMPTS}" \
    --input-mean="${ISL}" --output-mean="${OSL}" \
    --input-stdev=0 --output-stdev=0 > dataset.json

BENCH_CMD="trtllm-bench -m ${MODEL_CARD} --model_path ${MODEL_PATH} throughput \
    --tp ${EP} --ep ${EP} \
    --warmup 1 \
    --dataset dataset.json \
    --backend pytorch \
    --max_batch_size ${MAX_BATCH} \
    --max_num_tokens ${MAX_NUM_TOKENS} \
    --kv_cache_free_gpu_mem_fraction ${KV_FRAC} \
    --concurrency ${MAX_BATCH} \
    --extra_llm_api_options extra-llm-api-config.yml \
    --num_requests ${NUM_PROMPTS} \
    --streaming"

if $ENABLE_PROFILE; then
    export TLLM_PROFILE_RECORD_GC=1
    export TLLM_PROFILE_START_STOP=16000-16050
    export TLLM_LLMAPI_ENABLE_NVTX=1
    nsys profile -o "${LOG_PREFIX}_trace" \
        -f true \
        -t 'cuda,nvtx,python-gil' \
        -c cudaProfilerApi \
        --cuda-graph-trace node \
        -e --trace-fork-before-exec=true \
        ${BENCH_CMD} |& tee "${LOG_PREFIX}.txt"
    echo "--- nsys kernel summary ---"
    nsys stats --report cuda_gpu_kern_sum "${LOG_PREFIX}_trace.nsys-rep" \
        | grep -i "topk\|heuristic" || true
else
    ${BENCH_CMD} |& tee "${LOG_PREFIX}.txt"
fi

echo "Done. Logs: ${LOG_PREFIX}.txt"
