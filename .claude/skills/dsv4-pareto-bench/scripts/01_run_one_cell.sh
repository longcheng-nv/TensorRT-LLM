#!/bin/bash
# Run ONE cell of the DSv4 paired-feature sweep. Idempotent and safe to re-run.
#
# Usage: bash 01_run_one_cell.sh <cell_id> <ISL> <OSL> <BS> <MTP> <Mode> <GVR>
# Where GVR is 1 (ON) or 0 (OFF). Mode is TEP or DEP.
#
# Required env:
#   MODEL_PATH       absolute path to DSv4 Flash / Pro weights
#   PERFDIR          sweep working dir (must contain plan.csv, scripts/, ...)
#                    defaults to parent of this script's dir
#
# Optional env (with defaults matching the proven B-flash-B300 recipe):
#   MODEL_CARD          deepseek-v4/DeepSeek-V4
#   TRTLLM_REPO         pwd when not set (must contain benchmarks/cpp/prepare_dataset.py)
#   TP                  8         (tensor parallel)
#   EP                  8         (expert parallel)
#   MAX_NUM_TOKENS      8192      (per-cell trtllm-bench --max_num_tokens)
#   MOE_MAX_NUM_TOKENS  131072    (MoE config max_num_tokens)
#   KV_FRACTION         0.8       (--kv_cache_free_gpu_mem_fraction)
#   KV_CACHE_DTYPE      fp8
#   MOE_BACKEND         TRTLLM    (Pro variant: DEEPGEMM)
#   MULTI_ROUND         4         (NUM_PROMPTS = BS * MULTI_ROUND; was 2 before
#                                  2026-05-19 when steady-state analysis showed
#                                  2 waves = ~50% tail-drain on ISL≥64K. Bump
#                                  to 8 for ISL≥100K + high CONC. See SKILL.md
#                                  "Sizing num_requests".)
#   DATASET_NUM_PROMPTS sweep-wide dataset size (default: NUM_PROMPTS).
#                       Normally set in PERFDIR/bench.env by 00_generate_plan.py
#                       to max(BSS) * MULTI_ROUND so every BS cell consumes the
#                       SAME prompts (smaller BS = strict prefix of larger BS).
#   DATASET_FILE        absolute path to a pre-made trtllm-bench JSONL.
#                       When set, synthetic generation is skipped entirely —
#                       used for real prompts produced by prepare_real_prompts.py
#                       or any external dataset (e.g., Xianjie's random-v2).
#   TOKEN_BUDGET_MARGIN 512
#   CUDA_GRAPH_BS_LIST  "[1, 2, 3, 4, 5, 6, 7, 8]"
#   SAMPLER_TOP_K       1
#   SAMPLER_SEED        0
#   CELL_TIMEOUT        7200      (seconds; 124 exit -> recorded as TIMEOUT)
#   EXTRA_YAML_TEMPLATE custom envsubst template (default: templates/extra-llm-api-config.yml.tpl)
#   SAMPLER_TEMPLATE    custom envsubst template (default: templates/sampler-options.yml.tpl)
#
# Exit codes:
#   0  success — row appended to results/per_cell.csv, cell_id in completed.txt
#   1  cell already completed (no-op)
#   2  OOM — added to failed.txt with reason=OOM
#   3  bench non-zero exit or TIMEOUT — added to failed.txt
#   4  PERFORMANCE OVERVIEW missing — added to failed.txt
#   5  invalid args / missing required env
set -uo pipefail

if [[ $# -ne 7 ]]; then
    echo "Usage: $0 <cell_id> <ISL> <OSL> <BS> <MTP> <Mode> <GVR>" >&2
    exit 5
fi

CELL_ID="$1"; ISL="$2"; OSL="$3"; BS="$4"; MTP="$5"; MODE="$6"; GVR="$7"

# Required env
: "${MODEL_PATH:?MODEL_PATH env var required}"
[[ -d "${MODEL_PATH}" ]] || { echo "MODEL_PATH ${MODEL_PATH} does not exist" >&2; exit 5; }

# Defaults
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PERFDIR="${PERFDIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
MODEL_CARD="${MODEL_CARD:-deepseek-v4/DeepSeek-V4}"
TRTLLM_REPO="${TRTLLM_REPO:-${PWD}}"
TP="${TP:-8}"
EP="${EP:-8}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-8192}"
MOE_MAX_NUM_TOKENS="${MOE_MAX_NUM_TOKENS:-131072}"
KV_FRACTION="${KV_FRACTION:-0.8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
# Auto-derive MoE backend from quantization unless user overrides.
# expert_dtype=fp4 → MXFP4 routed experts → TRTLLM (W4A8_MXFP4_MXFP8 kernel)
# expert_dtype absent/other → FP8 block-scale routed → DEEPGEMM
if [[ -z "${MOE_BACKEND:-}" ]] && [[ -f "${MODEL_PATH}/config.json" ]]; then
    _expert_dtype=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('expert_dtype',''))" "${MODEL_PATH}/config.json" 2>/dev/null || echo "")
    if [[ "${_expert_dtype}" == "fp4" ]]; then
        MOE_BACKEND="TRTLLM"
        echo "[moe-auto] expert_dtype=fp4 → MOE_BACKEND=TRTLLM"
    elif [[ -n "${_expert_dtype}" ]]; then
        MOE_BACKEND="DEEPGEMM"
        echo "[moe-auto] expert_dtype=${_expert_dtype} → MOE_BACKEND=DEEPGEMM"
    fi
fi
MOE_BACKEND="${MOE_BACKEND:-TRTLLM}"
MULTI_ROUND="${MULTI_ROUND:-4}"
# Decouple --max_batch_size (engine-side cap) from BS (which doubles as
# --concurrency). Set MAX_BS_OVERRIDE to fix the engine-side max batch
# (e.g., MAX_BS_OVERRIDE=128 for a concurrency sweep at fixed BS=128).
MAX_BS_OVERRIDE="${MAX_BS_OVERRIDE:-}"
# Likewise, decouple --num_requests from BS * MULTI_ROUND when you need a
# fixed dataset window across all cells (e.g., NUM_PROMPTS_OVERRIDE=256).
NUM_PROMPTS_OVERRIDE="${NUM_PROMPTS_OVERRIDE:-}"
TOKEN_BUDGET_MARGIN="${TOKEN_BUDGET_MARGIN:-512}"
CUDA_GRAPH_BS_LIST="${CUDA_GRAPH_BS_LIST:-[1, 2, 3, 4, 5, 6, 7, 8]}"
SAMPLER_TOP_K="${SAMPLER_TOP_K:-1}"
SAMPLER_SEED="${SAMPLER_SEED:-0}"
CELL_TIMEOUT="${CELL_TIMEOUT:-7200}"
EXTRA_YAML_TEMPLATE="${EXTRA_YAML_TEMPLATE:-${SKILL_DIR}/templates/extra-llm-api-config.yml.tpl}"
SAMPLER_TEMPLATE="${SAMPLER_TEMPLATE:-${SKILL_DIR}/templates/sampler-options.yml.tpl}"

[[ -f "${EXTRA_YAML_TEMPLATE}" ]] || { echo "missing template ${EXTRA_YAML_TEMPLATE}" >&2; exit 5; }
[[ -f "${SAMPLER_TEMPLATE}" ]]   || { echo "missing template ${SAMPLER_TEMPLATE}"   >&2; exit 5; }

mkdir -p "${PERFDIR}/logs" "${PERFDIR}/configs" "${PERFDIR}/datasets" "${PERFDIR}/results"
LOG_FILE="${PERFDIR}/logs/${CELL_ID}.log"
YAML_FILE="${PERFDIR}/configs/${CELL_ID}.yaml"
SAMPLER_FILE="${PERFDIR}/configs/${CELL_ID}.sampler.yaml"

# Fast-skip if already completed.
if grep -qxF "${CELL_ID}" "${PERFDIR}/completed.txt" 2>/dev/null; then
    echo "[skip] ${CELL_ID} already in completed.txt"
    exit 1
fi
# Fast-skip if previously failed unless force_retry.txt lists it.
if grep -qP "^${CELL_ID}\b" "${PERFDIR}/failed.txt" 2>/dev/null \
   && ! grep -qxF "${CELL_ID}" "${PERFDIR}/force_retry.txt" 2>/dev/null; then
    echo "[skip] ${CELL_ID} previously failed; remove from failed.txt or add to force_retry.txt to retry"
    exit 1
fi

if [[ -n "${NUM_PROMPTS_OVERRIDE}" ]]; then
    NUM_PROMPTS="${NUM_PROMPTS_OVERRIDE}"
else
    NUM_PROMPTS=$(( BS * MULTI_ROUND ))
    # Floor at 8 prompts so very-low-BS cells still get a meaningful steady-state
    # window (BS=1 × MULTI_ROUND=4 = 4 is too short on long-ISL workloads).
    (( NUM_PROMPTS < 8 )) && NUM_PROMPTS=8
fi
# MAX_BS = max_batch_size sent to --max_batch_size. Defaults to BS but can be
# pinned via MAX_BS_OVERRIDE (e.g., concurrency-sweep style at fixed engine BS).
MAX_BS="${MAX_BS_OVERRIDE:-${BS}}"
# Default to per-cell NUM_PROMPTS for backwards compat; 02_master.sh sources
# bench.env which sets this to max(BSS) * MULTI_ROUND so all BS cells share
# the same dataset (smaller BS consumes a strict prefix).
DATASET_NUM_PROMPTS="${DATASET_NUM_PROMPTS:-${NUM_PROMPTS}}"
MAX_INPUT_LEN=$(( ISL + TOKEN_BUDGET_MARGIN ))
MAX_SEQ_LEN=$(( ISL + OSL + TOKEN_BUDGET_MARGIN ))

case "${MODE}" in
    # Both modes use `--tp ${TP} --ep ${EP}`; difference is enable_attention_dp.
    # Convention matches Xianjie's bench-dsv4/agg_bench/submit_pro_full_lyris1.sh:
    #   adp=0 → TEP (Tensor + Expert Parallel; attention TP-sharded)
    #   adp=1 → DEP (Data  + Expert Parallel; attention DP'd per rank)
    # Pre-2026-05-26 versions of this script had this mapping INVERTED — see
    # SKILL.md "Naming convention fix 2026-05-26" banner.
    TEP) ATTN_DP=false ;;
    DEP) ATTN_DP=true  ;;
    *) echo "Mode must be TEP or DEP, got ${MODE}" >&2; exit 5 ;;
esac

GVR_BOOL=$([[ "${GVR}" == "1" ]] && echo true || echo false)
ENABLE_HEURISTIC_TOPK="${GVR_BOOL}"

# When MTP=0, drop the speculative_config block entirely
# (decoding_type=MTP with 0 layers is invalid).
if [[ "${MTP}" == "0" ]]; then
    SPECULATIVE_CONFIG_BLOCK=""
else
    SPECULATIVE_CONFIG_BLOCK=$(cat <<EOF
speculative_config:
    decoding_type: MTP
    num_nextn_predict_layers: ${MTP}
EOF
)
fi

# MNNVL allreduce: enable on Blackwell only for TP=8 EP=8 non-attention-DP cases
# (bench-dsv4 pattern; NVLink ring becomes worth it at TP8 EP8).
if [[ "${TP}" == "8" ]] && [[ "${EP}" == "8" ]] && [[ "${ATTN_DP}" == "false" ]]; then
    ALLREDUCE_BLOCK=$'allreduce_strategy: MNNVL\n'
else
    ALLREDUCE_BLOCK=""
fi

# Render YAML via envsubst.
export MAX_INPUT_LEN MAX_SEQ_LEN CUDA_GRAPH_BATCH_SIZES="${CUDA_GRAPH_BS_LIST}"
export KV_CACHE_DTYPE MOE_MAX_NUM_TOKENS MOE_BACKEND ATTN_DP
export SPECULATIVE_CONFIG_BLOCK ENABLE_HEURISTIC_TOPK ALLREDUCE_BLOCK
# Prefer the system envsubst (gettext-base) when available; fall back to
# the pure-Python equivalent in the skill scripts dir. Some hosts (internal
# VMs without gettext) need the Python path.
if command -v envsubst >/dev/null 2>&1; then
    ENVSUBST_CMD=(envsubst)
else
    ENVSUBST_CMD=(python3 "${SCRIPT_DIR}/_envsubst.py")
fi
"${ENVSUBST_CMD[@]}" < "${EXTRA_YAML_TEMPLATE}" > "${YAML_FILE}"
export SAMPLER_TOP_K SAMPLER_SEED
"${ENVSUBST_CMD[@]}" < "${SAMPLER_TEMPLATE}" > "${SAMPLER_FILE}"

echo "==================== ${CELL_ID} ===================="
echo "ISL/OSL=${ISL}/${OSL} BS(conc)=${BS} max_bs=${MAX_BS} num_prompts=${NUM_PROMPTS} MTP=${MTP} Mode=${MODE} GVR=${GVR_BOOL} TP=${TP} EP=${EP}"
echo "MODEL_PATH=${MODEL_PATH}  MOE_BACKEND=${MOE_BACKEND}  KV_FRACTION=${KV_FRACTION}"
echo "YAML: ${YAML_FILE}"
echo "log:  ${LOG_FILE}"
echo "Start: $(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Enable the [Scheme X] dispatch trace lines so _parse_one_cell.py can
# classify Heuristic vs Radix vs Mixed.
export TRTLLM_SCHEMEX_DEBUG=1

# DRY_RUN=1 short-circuits before the bench (used by smoke tests).
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "[dry] DRY_RUN=1 — skipping dataset generation and trtllm-bench; YAML and sampler rendered."
    exit 0
fi

# Dataset selection:
#   - If DATASET_FILE env is set, use it directly (real prompts produced by
#     prepare_real_prompts.py, or any pre-made trtllm-bench JSONL). Bench
#     consumes the first NUM_PROMPTS rows of this file.
#   - Otherwise, lazily generate a synthetic token-norm-dist dataset sized to
#     DATASET_NUM_PROMPTS (= max(BSS) * MULTI_ROUND when sourced from
#     bench.env). Smaller-BS cells consume a strict prefix of the same file,
#     so per-prompt content is identical across every BS cell in the sweep.
if [[ -n "${DATASET_FILE:-}" ]]; then
    DATASET="${DATASET_FILE}"
    [[ -s "${DATASET}" ]] || { echo "DATASET_FILE=${DATASET} does not exist or is empty" >&2; exit 5; }
    echo "[dataset] using pre-made ${DATASET}"
else
    DATASET="${PERFDIR}/datasets/synth_isl${ISL}_osl${OSL}_n${DATASET_NUM_PROMPTS}.jsonl"
    if [[ ! -s "${DATASET}" ]]; then
        echo "[dataset] generating ${DATASET} (n=${DATASET_NUM_PROMPTS}; same file across all BS cells)"
        PREPARE="${TRTLLM_REPO}/benchmarks/cpp/prepare_dataset.py"
        [[ -f "${PREPARE}" ]] || { echo "TRTLLM_REPO/${PREPARE} missing; set TRTLLM_REPO env" >&2; exit 5; }
        # Generate to a temp file first so a failing prepare_dataset.py doesn't
        # leave a 0-byte cache that future runs would consider "present".
        tmp_dataset="${DATASET}.partial"
        if ! ( cd "${TRTLLM_REPO}" && python3 "${PREPARE}" \
                --tokenizer "${MODEL_PATH}" --stdout \
                token-norm-dist \
                    --num-requests "${DATASET_NUM_PROMPTS}" \
                    --input-mean "${ISL}" --input-stdev 0 \
                    --output-mean "${OSL}" --output-stdev 0 \
                > "${tmp_dataset}" ); then
            rm -f "${tmp_dataset}"
            echo "[fail] dataset generation failed" >&2
            exit 3
        fi
        mv "${tmp_dataset}" "${DATASET}"
    fi
fi

REPORT_JSON="${PERFDIR}/results/${CELL_ID}_report.json"

# NUMA pin to nodes 0,1 (bench-dsv4 pattern) when the host has ≥2 NUMA nodes.
NUMA_PREFIX=()
if command -v numactl >/dev/null 2>&1; then
    _numa_nodes=$(numactl --hardware 2>/dev/null | awk '/available:/{print $2}')
    if [[ -n "${_numa_nodes}" ]] && (( _numa_nodes >= 2 )); then
        NUMA_PREFIX=(numactl -m 0,1)
    fi
fi

set +e
timeout --kill-after=30s "${CELL_TIMEOUT}" \
    "${NUMA_PREFIX[@]}" \
    trtllm-bench -m "${MODEL_CARD}" --model_path "${MODEL_PATH}" throughput \
        --tp "${TP}" --ep "${EP}" \
        --warmup 1 \
        --dataset "${DATASET}" \
        --backend pytorch \
        --max_batch_size "${MAX_BS}" \
        --max_num_tokens "${MAX_NUM_TOKENS}" \
        --kv_cache_free_gpu_mem_fraction "${KV_FRACTION}" \
        --concurrency "${BS}" \
        --extra_llm_api_options "${YAML_FILE}" \
        --sampler_options "${SAMPLER_FILE}" \
        --num_requests "${NUM_PROMPTS}" \
        --streaming \
        --report_json "${REPORT_JSON}" \
        > "${LOG_FILE}" 2>&1
BENCH_EXIT=$?
set -e

if [[ ${BENCH_EXIT} -eq 124 ]]; then
    printf '%s\tTIMEOUT_%ds\n' "${CELL_ID}" "${CELL_TIMEOUT}" >> "${PERFDIR}/failed.txt"
    echo "[fail] cell exceeded ${CELL_TIMEOUT}s — killed"
    exit 3
fi

echo "End:   $(date -u +%Y-%m-%dT%H:%M:%SZ)  exit=${BENCH_EXIT}"

if grep -qE "OutOfMemoryError|CUDA error: out of memory|CUDA out of memory" "${LOG_FILE}"; then
    printf '%s\tOOM\n' "${CELL_ID}" >> "${PERFDIR}/failed.txt"
    echo "[fail] OOM"
    exit 2
fi
if [[ ${BENCH_EXIT} -ne 0 ]]; then
    printf '%s\tBENCH_FAIL_exit%d\n' "${CELL_ID}" "${BENCH_EXIT}" >> "${PERFDIR}/failed.txt"
    echo "[fail] bench non-zero exit"
    exit 3
fi
if ! grep -q "PERFORMANCE OVERVIEW" "${LOG_FILE}"; then
    printf '%s\tPARSE_FAIL_no_perf_overview\n' "${CELL_ID}" >> "${PERFDIR}/failed.txt"
    echo "[fail] PERFORMANCE OVERVIEW missing"
    exit 4
fi

python3 "${SCRIPT_DIR}/_parse_one_cell.py" \
    --cell-id "${CELL_ID}" \
    --isl "${ISL}" --osl "${OSL}" --bs "${BS}" \
    --mtp "${MTP}" --mode "${MODE}" --gvr "${GVR_BOOL}" \
    --log "${LOG_FILE}" \
    --out "${PERFDIR}/results/per_cell.csv"

echo "${CELL_ID}" >> "${PERFDIR}/completed.txt"
echo "[ok] ${CELL_ID} done"
exit 0
