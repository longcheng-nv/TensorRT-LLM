#!/bin/bash
# DSv4 (Flash / Pro) nsys decode-phase profiling — one-shot launcher.
#
# Wraps `trtllm-bench throughput` with `nsys profile -c cudaProfilerApi` and
# `TLLM_PROFILE_START_STOP=<iter_range>` so the resulting .nsys-rep only
# contains the requested iteration window (typically a slice of steady-state
# decode). Everything outside the window is skipped at zero overhead via
# cudaProfilerStart/Stop bracketing.
#
# Usage (env-var driven; all args optional):
#   ISL=131072 OSL=2048 BS=1 MTP=0 GVR=0 MODE=TEP \
#   PROFILE_ITER_RANGE=500-550 OUT_DIR=./nsys_out \
#   bash run_nsys.sh
#
# Or with positional shim (for legacy callers):
#   bash run_nsys.sh <ISL> <OSL> <BS> <MTP> <GVR{0,1}> <MODE{TEP,DEP}>
#
# Output (under OUT_DIR):
#   ds4_<variant>_ISL<ISL>_OSL<OSL>_BS<BS>_MTP<MTP>_<mode>_GVR<bool>_<ts>/
#       run.log                              # bench stdout/stderr (filtered)
#       extra-llm-api-config.yml             # rendered YAML
#       dataset.jsonl                        # generated dataset (cached per shape)
#       decode_trace_<window>.nsys-rep       # nsys report (open in Nsight Systems)
#       decode_trace_<window>.sqlite         # nsys sqlite export (auto)
#
# Env vars — model / hardware (all auto-detected or inherited from dsv4-pareto-bench):
#   MODEL_VARIANT       Flash | Pro                 default: Flash
#   MODEL_PATH          explicit absolute path      default: auto-detect by hostname
#   TP, EP              tensor / expert parallel    default: 8 / 8
#   MOE_BACKEND         auto from config.json       fp4 → TRTLLM (TEP) / MEGAMOE_DEEPGEMM (DEP)
#   MAX_NUM_TOKENS      chunked-prefill chunk size  default: 8192
#   MOE_MAX_NUM_TOKENS  MoE config max_num_tokens   default: 131072
#   KV_FRACTION         kv_cache_free_gpu_mem_fraction  default: 0.8
#   KV_CACHE_DTYPE      fp8                         default: fp8
#
# Env vars — workload axes:
#   ISL                 input sequence length       default: 131072  (128 K)
#   OSL                 output sequence length      default: 2048    (2 K)
#   BS                  batch size = concurrency    default: 1
#   MTP                 num_nextn_predict_layers    default: 0       (drop spec block)
#   GVR                 enable_heuristic_topk       default: 0       (0 = Radix; 1 = Heuristic)
#   MODE                TEP | DEP                   default: TEP     (enable_attention_dp = false)
#
# Env vars — profiling window:
#   PROFILE_ITER_RANGE  TLLM_PROFILE_START_STOP value (start-stop or list)
#                       default: 500-550  (50-iter decode window after prefill+warmup)
#                       Each iter = one forward pass. At ISL=128K BS=1 max_num_tokens=8192,
#                       prefill ≈ 16 iters; with --warmup 1 the decode steady state
#                       begins around iter ~40. Iter 500-550 is comfortably in decode.
#   STREAM_INTERVAL     YAML stream_interval        default: 100
#                       Lower to 10 if you want more frequent decode iter prints in the log.
#   NSYS_TRACE          nsys -t flag                default: cuda,nvtx,python-gil
#   NSYS_EXTRA_ARGS     extra args appended to nsys default: empty
#
# Env vars — misc:
#   NUM_PROMPTS         --num_requests             default: max(BS, 1)
#                       For BS=1 OSL=2048 a single request is enough — the profile
#                       window will land in its decode phase. Bump (e.g., 4) only if
#                       you need warmup-vs-steady-state comparison across requests.
#   WARMUP              trtllm-bench --warmup       default: 1
#   OUT_DIR             output root                 default: $PWD/dsv4_nsys_out
#   MODEL_CARD          --model_path companion      default: deepseek-v4/DeepSeek-V4
#   TRTLLM_REPO         must contain benchmarks/cpp/prepare_dataset.py  default: $PWD
#   DRY_RUN             1 = render YAML + dataset cmd, no bench/nsys     default: 0
#
# Env vars — real-prompt input (mutually exclusive with the synthetic default):
#   PROMPTS_INPUT       Switches the dataset source. Accepted values:
#                         (a) shortcut keyword: swe16k | swe32k | swe64k | swe100k
#                             → auto-resolves to a longseqtasks/swe_bench_*.jsonl
#                               via the candidate path list (see resolver below).
#                         (b) explicit path to a {system,user} JSONL
#                             → tokenized with prepare_swebench_dataset.py against MODEL_PATH.
#                         (c) explicit path to an already-tokenized JSONL with
#                             {task_id, input_ids, output_tokens} per line
#                             → copied verbatim into dataset.jsonl (no re-tokenize).
#                       Default: unset (use synthetic token-norm-dist).
#                       When set, ISL env var becomes informational only (the real
#                       prompt's tokenized ISL drives max_input_len / max_seq_len).
#   PROMPTS_ENTRY       Keep only this 0-indexed entry from PROMPTS_INPUT.
#                       Default: unset (keep all entries).
#   PROMPTS_REPLICATE   Replicate kept entries this many times (rows out =
#                       entries × replicate). Useful for BS>1 same-prompt sweeps.
#                       Default: max(NUM_PROMPTS / entries_after_select, 1).
#
# Exit codes:
#   0  success — .nsys-rep written
#   2  OOM
#   3  bench non-zero exit
#   5  invalid args / env / missing template
set -uo pipefail

# ----------------- Args + defaults ----------------------------------------
# Positional shim — if 6 args provided, treat them as the axes:
if [[ $# -eq 6 ]]; then
    ISL="${ISL:-$1}"; OSL="${OSL:-$2}"; BS="${BS:-$3}"
    MTP="${MTP:-$4}"; GVR="${GVR:-$5}"; MODE="${MODE:-$6}"
elif [[ $# -gt 0 ]]; then
    echo "Usage (positional): $0 <ISL> <OSL> <BS> <MTP> <GVR{0,1}> <MODE{TEP,DEP}>" >&2
    echo "       (or no args: drive entirely via env vars)" >&2
    exit 5
fi

ISL="${ISL:-131072}"
OSL="${OSL:-2048}"
BS="${BS:-1}"
MTP="${MTP:-0}"
GVR="${GVR:-0}"
MODE="${MODE:-TEP}"

MODEL_VARIANT="${MODEL_VARIANT:-Flash}"
case "${MODEL_VARIANT}" in Flash|Pro) ;; *)
    echo "MODEL_VARIANT must be Flash or Pro, got '${MODEL_VARIANT}'" >&2; exit 5 ;;
esac

# Model-path auto-detection (mirrors dsv4-pareto-bench/01_run_one_cell.sh).
if [[ -z "${MODEL_PATH:-}" ]]; then
    case "$(hostname -f 2>/dev/null || hostname)" in
        *.colossus.nvidia.com|umb-b*) _cluster="SC-computelab" ;;
        lyris*)                       _cluster="lyris1" ;;
        *)                            _cluster="unknown" ;;
    esac
    # Expected safetensors shard count per variant — used to skip partial stages
    # (e.g., an in-progress /dev/shm or /raid HF download).
    case "${MODEL_VARIANT}" in
        Flash) _expected_shards=46 ;;
        Pro)   _expected_shards=64 ;;
        *)     _expected_shards=0  ;;
    esac
    for _cand in \
        "/dev/shm/DeepSeek-V4-${MODEL_VARIANT}" \
        "/raid/data/${USER}-stage/DeepSeek-V4-${MODEL_VARIANT}" \
        "/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V4-${MODEL_VARIANT}" \
        "/lustre/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/common/DeepSeek-V4-${MODEL_VARIANT}" \
        "/lustre/fsw/portfolios/coreai/projects/coreai_comparch_inferencex/users/ashanbhag/DeepSeek-V4-${MODEL_VARIANT}" \
        "/lustre/fsw/coreai_comparch_inferencex/models/dsv4-$(echo ${MODEL_VARIANT} | tr A-Z a-z)" \
    ; do
        if [[ -d "${_cand}" ]]; then
            # Reject incomplete stages: require ≥ _expected_shards safetensors
            # AND a non-empty config.json. Prevents /dev/shm partial-download
            # from masking the NFS canonical copy.
            _n_st=$(ls "${_cand}"/*.safetensors 2>/dev/null | wc -l)
            if (( _expected_shards > 0 )) && (( _n_st < _expected_shards )); then
                echo "[model-auto] skip ${_cand}: ${_n_st}/${_expected_shards} safetensors (partial)"
                continue
            fi
            [[ -s "${_cand}/config.json" ]] || { echo "[model-auto] skip ${_cand}: config.json missing/empty"; continue; }
            MODEL_PATH="${_cand}"
            echo "[model-auto] cluster=${_cluster} variant=${MODEL_VARIANT} (${_n_st} safetensors) → MODEL_PATH=${MODEL_PATH}"
            break
        fi
    done
    unset _cand _cluster
fi
: "${MODEL_PATH:?MODEL_PATH unset and auto-detect failed — set MODEL_PATH or MODEL_VARIANT=Flash|Pro}"
[[ -d "${MODEL_PATH}" ]] || { echo "MODEL_PATH ${MODEL_PATH} does not exist" >&2; exit 5; }

TP="${TP:-8}"
EP="${EP:-8}"
MAX_NUM_TOKENS="${MAX_NUM_TOKENS:-8192}"
MOE_MAX_NUM_TOKENS="${MOE_MAX_NUM_TOKENS:-131072}"
KV_FRACTION="${KV_FRACTION:-0.8}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
STREAM_INTERVAL="${STREAM_INTERVAL:-100}"
NUM_PROMPTS="${NUM_PROMPTS:-${BS}}"
(( NUM_PROMPTS < 1 )) && NUM_PROMPTS=1
WARMUP="${WARMUP:-1}"

# MoE backend — auto from config.json:expert_dtype, override Mode-dependent for fp4.
if [[ -z "${MOE_BACKEND:-}" ]] && [[ -f "${MODEL_PATH}/config.json" ]]; then
    _expert_dtype=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('expert_dtype',''))" "${MODEL_PATH}/config.json" 2>/dev/null || echo "")
    if [[ "${_expert_dtype}" == "fp4" ]]; then
        if [[ "${MODE}" == "DEP" ]]; then
            MOE_BACKEND="MEGAMOE_DEEPGEMM"
            echo "[moe-auto] expert_dtype=fp4 Mode=DEP → MOE_BACKEND=MEGAMOE_DEEPGEMM"
        else
            MOE_BACKEND="TRTLLM"
            echo "[moe-auto] expert_dtype=fp4 Mode=TEP → MOE_BACKEND=TRTLLM"
        fi
    elif [[ -n "${_expert_dtype}" ]]; then
        MOE_BACKEND="DEEPGEMM"
        echo "[moe-auto] expert_dtype=${_expert_dtype} → MOE_BACKEND=DEEPGEMM"
    fi
fi
MOE_BACKEND="${MOE_BACKEND:-TRTLLM}"

PROFILE_ITER_RANGE="${PROFILE_ITER_RANGE:-500-550}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,python-gil}"
NSYS_EXTRA_ARGS="${NSYS_EXTRA_ARGS:-}"

MODEL_CARD="${MODEL_CARD:-deepseek-v4/DeepSeek-V4}"
TRTLLM_REPO="${TRTLLM_REPO:-${PWD}}"
DRY_RUN="${DRY_RUN:-0}"

# Decode-phase nsys runs typically use a NARROW cuda_graph_bs list since BS is
# small and there's no concurrency sweep. Capture BS itself + a power-of-2
# ladder up to it for safety against in-flight padding.
if [[ -z "${CUDA_GRAPH_BS_LIST:-}" ]]; then
    _gbs="1, 2, 3, 4, 5, 6, 7, 8"
    for _sz in 16 32 64 128 256 512 1024; do
        (( _sz <= BS )) && _gbs="${_gbs}, ${_sz}"
    done
    if (( BS > 8 )) && [[ ",${_gbs}," != *", ${BS},"* ]]; then
        _gbs="${_gbs}, ${BS}"
    fi
    CUDA_GRAPH_BS_LIST="[${_gbs}]"
fi

case "${MODE}" in
    TEP) ATTN_DP=false ;;
    DEP) ATTN_DP=true  ;;
    *) echo "MODE must be TEP or DEP, got '${MODE}'" >&2; exit 5 ;;
esac

GVR_BOOL=$([[ "${GVR}" == "1" ]] && echo true || echo false)
ENABLE_HEURISTIC_TOPK="${GVR_BOOL}"

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

# MNNVL allreduce only for TP=8 EP=8 TEP cases (matches dsv4-pareto-bench).
if [[ "${TP}" == "8" ]] && [[ "${EP}" == "8" ]] && [[ "${ATTN_DP}" == "false" ]]; then
    ALLREDUCE_BLOCK=$'allreduce_strategy: MNNVL\n'
else
    ALLREDUCE_BLOCK=""
fi

if [[ "${ATTN_DP}" == "true" ]]; then
    G10_ADP_BLOCK=$'# G10 DEP fix — required for cute_dsl path on V4 Pro DEP.\nenable_lm_head_tp_in_adp: true\nuse_cute_dsl_blockscaling_bmm: true\n'
else
    G10_ADP_BLOCK=""
fi

if [[ "${MOE_BACKEND}" == "MEGAMOE_DEEPGEMM" ]]; then
    MOE_LP_COMBINE_BLOCK=$'    # G10 DEP fix — required by MEGAMOE_DEEPGEMM path.\n    use_low_precision_moe_combine: true\n'
else
    MOE_LP_COMBINE_BLOCK=""
fi

# ----------------- Output dir + sidecar paths -----------------------------
TS="$(date -u +%Y%m%dT%H%M%SZ)"
TAG="ds4_${MODEL_VARIANT}_ISL${ISL}_OSL${OSL}_BS${BS}_MTP${MTP}_${MODE}_GVR${GVR_BOOL}_${TS}"
OUT_DIR="${OUT_DIR:-${PWD}/dsv4_nsys_out}"
RUN_DIR="${OUT_DIR}/${TAG}"
mkdir -p "${RUN_DIR}"

YAML_FILE="${RUN_DIR}/extra-llm-api-config.yml"
SAMPLER_FILE="${RUN_DIR}/sampler-options.yml"
DATASET_FILE="${RUN_DIR}/dataset.jsonl"
LOG_FILE="${RUN_DIR}/run.log"
NSYS_OUT_BASE="${RUN_DIR}/decode_trace_iter${PROFILE_ITER_RANGE//[,-]/_}"

# Token budgets — leave a 512-token margin (matches dsv4-pareto-bench).
MAX_INPUT_LEN=$(( ISL + 512 ))
MAX_SEQ_LEN=$(( ISL + OSL + 512 ))

# ----------------- Render YAML + sampler ----------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXTRA_YAML_TEMPLATE="${EXTRA_YAML_TEMPLATE:-${SKILL_DIR}/templates/extra-llm-api-config.yml.tpl}"
[[ -f "${EXTRA_YAML_TEMPLATE}" ]] || { echo "missing template ${EXTRA_YAML_TEMPLATE}" >&2; exit 5; }

export MAX_INPUT_LEN MAX_SEQ_LEN CUDA_GRAPH_BATCH_SIZES="${CUDA_GRAPH_BS_LIST}"
export KV_CACHE_DTYPE MOE_MAX_NUM_TOKENS MOE_BACKEND ATTN_DP STREAM_INTERVAL
export SPECULATIVE_CONFIG_BLOCK ENABLE_HEURISTIC_TOPK ALLREDUCE_BLOCK
export G10_ADP_BLOCK MOE_LP_COMBINE_BLOCK

if command -v envsubst >/dev/null 2>&1; then
    envsubst < "${EXTRA_YAML_TEMPLATE}" > "${YAML_FILE}"
else
    # Lightweight Python envsubst fallback.
    python3 -c "
import os, sys, re
src = open(sys.argv[1]).read()
out = re.sub(r'\\\$\\{([A-Z_][A-Z0-9_]*)\\}', lambda m: os.environ.get(m.group(1), ''), src)
open(sys.argv[2], 'w').write(out)
" "${EXTRA_YAML_TEMPLATE}" "${YAML_FILE}"
fi

# Sampler: greedy (top_k=1 seed=0) for reproducibility — same as dsv4-pareto-bench.
cat > "${SAMPLER_FILE}" <<EOF
top_k: 1
seed: 0
EOF

echo "==================== ${TAG} ===================="
echo "MODEL_PATH=${MODEL_PATH}  variant=${MODEL_VARIANT}"
echo "ISL=${ISL} OSL=${OSL} BS=${BS} MTP=${MTP} GVR=${GVR_BOOL} Mode=${MODE} TP=${TP} EP=${EP}"
echo "MOE_BACKEND=${MOE_BACKEND}  KV_FRACTION=${KV_FRACTION}  STREAM_INTERVAL=${STREAM_INTERVAL}"
echo "max_input_len=${MAX_INPUT_LEN}  max_seq_len=${MAX_SEQ_LEN}  max_num_tokens=${MAX_NUM_TOKENS}"
echo "PROFILE_ITER_RANGE=${PROFILE_ITER_RANGE}  NSYS_TRACE=${NSYS_TRACE}"
echo "RUN_DIR=${RUN_DIR}"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry] DRY_RUN=1 — YAML + sampler rendered; skipping dataset gen + nsys + bench."
    echo "      run:  ls ${RUN_DIR}"
    exit 0
fi

# ----------------- Dataset generation -------------------------------------
# Three modes:
#   (1) PROMPTS_INPUT unset  → synthetic token-norm-dist (default, unchanged)
#   (2) PROMPTS_INPUT set    → real SWE-bench / custom JSONL pipeline:
#         (2a) shortcut keyword  → resolve via SWE_BENCH_CANDIDATES then tokenize
#         (2b) {system,user}.jsonl path → tokenize with prepare_swebench_dataset.py
#         (2c) tokenized {input_ids,...} path → copy through verbatim
if [[ -n "${PROMPTS_INPUT:-}" ]] && [[ -s "${DATASET_FILE}" ]]; then
    :  # already materialized in this RUN_DIR; skip regen
elif [[ -n "${PROMPTS_INPUT:-}" ]]; then
    # ----- 2. Real-prompt mode -------------------------------------------
    # Resolve keyword shortcuts to an absolute path against the canonical
    # locations seen on this cluster (first existing wins).
    _resolve_swebench_keyword() {
        local kw="$1"; local tag
        case "${kw}" in
            swe16k)  tag="swe_bench_16k.jsonl"  ;;
            swe32k)  tag="swe_bench_32k.jsonl"  ;;
            swe64k)  tag="swe_bench_64k.jsonl"  ;;
            swe100k) tag="swe_bench_100k.jsonl" ;;
            *) return 1 ;;
        esac
        for cand in \
            "/home/scratch.loncheng_gpu/workspace/CodeRepos/GVR_TopK_supplementaty_materials/longseqtasks/${tag}" \
            "/home/scratch.loncheng_gpu/workspace/tllm_toolbox/indexer_topK_perf/data_distri/deepseek-v3.2-logging/tasks/${tag}" \
            "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV32/E2E_exp/deepseek-v3.2-logging/tasks/${tag}" \
        ; do
            if [[ -f "${cand}" ]]; then printf '%s\n' "${cand}"; return 0; fi
        done
        return 1
    }
    _src="${PROMPTS_INPUT}"
    if [[ "${_src}" == swe16k || "${_src}" == swe32k || "${_src}" == swe64k || "${_src}" == swe100k ]]; then
        if ! _src=$(_resolve_swebench_keyword "${PROMPTS_INPUT}"); then
            echo "[fail] PROMPTS_INPUT=${PROMPTS_INPUT}: no candidate path exists. Check longseqtasks/." >&2
            exit 5
        fi
        echo "[dataset] PROMPTS_INPUT shortcut '${PROMPTS_INPUT}' → ${_src}"
    fi
    [[ -f "${_src}" ]] || { echo "[fail] PROMPTS_INPUT=${_src} not found" >&2; exit 5; }

    # Tokenized vs. {system,user} probe: peek at first non-empty line.
    _first_line=$(awk 'NF{print; exit}' "${_src}" 2>/dev/null)
    if [[ "${_first_line}" == *'"input_ids"'* ]]; then
        # (2c) Already-tokenized — copy verbatim with optional --entry / replicate handled by sed/awk.
        echo "[dataset] already-tokenized input detected; copying ${_src} → ${DATASET_FILE}"
        if [[ -n "${PROMPTS_ENTRY:-}" ]]; then
            sed -n "$((PROMPTS_ENTRY + 1))p" "${_src}" > "${DATASET_FILE}.partial"
        else
            cp "${_src}" "${DATASET_FILE}.partial"
        fi
        _rep="${PROMPTS_REPLICATE:-1}"
        if (( _rep > 1 )); then
            : > "${DATASET_FILE}.expanded"
            for ((i = 0; i < _rep; i++)); do cat "${DATASET_FILE}.partial" >> "${DATASET_FILE}.expanded"; done
            mv "${DATASET_FILE}.expanded" "${DATASET_FILE}"
            rm -f "${DATASET_FILE}.partial"
        else
            mv "${DATASET_FILE}.partial" "${DATASET_FILE}"
        fi
    else
        # (2a/2b) {system,user} JSONL — tokenize with vendored prepare_swebench_dataset.py.
        PREP_SWE="${SKILL_DIR}/scripts/prepare_swebench_dataset.py"
        [[ -f "${PREP_SWE}" ]] || { echo "[fail] missing ${PREP_SWE}" >&2; exit 5; }

        _entry_args=()
        [[ -n "${PROMPTS_ENTRY:-}" ]] && _entry_args+=(--entry "${PROMPTS_ENTRY}")

        # Default replicate: enough rows to satisfy NUM_PROMPTS.
        if [[ -z "${PROMPTS_REPLICATE:-}" ]]; then
            if [[ -n "${PROMPTS_ENTRY:-}" ]]; then
                PROMPTS_REPLICATE="${NUM_PROMPTS}"
            else
                # _src has N entries; trtllm-bench caps num_requests at line count,
                # so 1× is fine unless user explicitly asks for more rows.
                PROMPTS_REPLICATE=1
            fi
        fi

        echo "[dataset] tokenizing ${_src} → ${DATASET_FILE} (entry=${PROMPTS_ENTRY:-all} replicate=${PROMPTS_REPLICATE} osl=${OSL})"
        if ! MAX_ISL_FROM_TOK=$(python3 "${PREP_SWE}" \
                --input "${_src}" \
                --tokenizer "${MODEL_PATH}" \
                --osl "${OSL}" \
                --output "${DATASET_FILE}.partial" \
                --num-replicate "${PROMPTS_REPLICATE}" \
                "${_entry_args[@]}" \
                2> >(tee -a "${LOG_FILE}.tokenize" >&2) \
                | tail -n 1 ); then
            rm -f "${DATASET_FILE}.partial"
            echo "[fail] prepare_swebench_dataset.py failed; see ${LOG_FILE}.tokenize" >&2
            exit 3
        fi
        mv "${DATASET_FILE}.partial" "${DATASET_FILE}"
        echo "${MAX_ISL_FROM_TOK}" > "${DATASET_FILE}.max_isl"
        echo "[dataset] tokenized max_isl=${MAX_ISL_FROM_TOK} cached in ${DATASET_FILE}.max_isl"
    fi

    # Real prompts: re-derive ISL / max_seq_len from the tokenized rows.
    _real_max_isl=$(python3 -c "
import json, sys
m = 0
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if line:
            d = json.loads(line)
            m = max(m, len(d.get('input_ids', [])))
print(m)
" "${DATASET_FILE}")
    if (( _real_max_isl > ISL )); then
        echo "[dataset] real-prompt max_isl=${_real_max_isl} > ISL=${ISL}; promoting ISL/max_input_len/max_seq_len."
        ISL="${_real_max_isl}"
        MAX_INPUT_LEN=$(( ISL + 512 ))
        MAX_SEQ_LEN=$(( ISL + OSL + 512 ))
        export MAX_INPUT_LEN MAX_SEQ_LEN
        # Re-render the YAML so max_input_len / max_seq_len reflect the real ISL.
        if command -v envsubst >/dev/null 2>&1; then
            envsubst < "${EXTRA_YAML_TEMPLATE}" > "${YAML_FILE}"
        else
            python3 -c "
import os, sys, re
src = open(sys.argv[1]).read()
out = re.sub(r'\\\$\\{([A-Z_][A-Z0-9_]*)\\}', lambda m: os.environ.get(m.group(1), ''), src)
open(sys.argv[2], 'w').write(out)
" "${EXTRA_YAML_TEMPLATE}" "${YAML_FILE}"
        fi
    fi
    # Adjust NUM_PROMPTS to dataset line count (trtllm-bench caps at it anyway).
    _real_rows=$(wc -l < "${DATASET_FILE}")
    if (( NUM_PROMPTS > _real_rows )); then
        echo "[dataset] capping NUM_PROMPTS ${NUM_PROMPTS} → ${_real_rows} (dataset line count)"
        NUM_PROMPTS="${_real_rows}"
    fi
elif [[ ! -s "${DATASET_FILE}" ]]; then
    # ----- 1. Synthetic mode (default) -----------------------------------
    PREPARE="${TRTLLM_REPO}/benchmarks/cpp/prepare_dataset.py"
    [[ -f "${PREPARE}" ]] || { echo "TRTLLM_REPO=${TRTLLM_REPO} missing benchmarks/cpp/prepare_dataset.py" >&2; exit 5; }
    tmp_dataset="${DATASET_FILE}.partial"
    echo "[dataset] generating ${DATASET_FILE} (synthetic n=${NUM_PROMPTS} ISL=${ISL} OSL=${OSL})"
    if ! ( cd "${TRTLLM_REPO}" && python3 "${PREPARE}" \
            --tokenizer "${MODEL_PATH}" --stdout \
            token-norm-dist \
                --num-requests "${NUM_PROMPTS}" \
                --input-mean "${ISL}" --input-stdev 0 \
                --output-mean "${OSL}" --output-stdev 0 \
            > "${tmp_dataset}" ); then
        rm -f "${tmp_dataset}"
        echo "[fail] dataset generation failed" >&2
        exit 3
    fi
    mv "${tmp_dataset}" "${DATASET_FILE}"
fi

# ----------------- G10 env vars (mandatory for DSv4 cuda_graph stability) -
export NCCL_GRAPH_MIXING_SUPPORT=0
export NCCL_NVLS_ENABLE=0
export TRTLLM_ENABLE_PDL=1
export TRTLLM_MLA_EXTRA_OVERLAP=1
export TRTLLM_FUSED_FP8_QUANT_PACK=1
export ENABLE_PERFECT_ROUTER=1
export TRTLLM_DISABLE_KV_CACHE_RATIO_UPDATE=1
export TRTLLM_SERVER_DISABLE_GC=1
export TRTLLM_WORKER_DISABLE_GC=1
export MIMALLOC_PURGE_DELAY=0

# ----------------- Profiling env vars -------------------------------------
export TLLM_PROFILE_RECORD_GC=1
export TLLM_LLMAPI_ENABLE_NVTX=1
export TLLM_PROFILE_START_STOP="${PROFILE_ITER_RANGE}"

# ----------------- NUMA pin (≥2 nodes) ------------------------------------
NUMA_PREFIX=()
if command -v numactl >/dev/null 2>&1; then
    _numa_nodes=$(numactl --hardware 2>/dev/null | awk '/available:/{print $2}')
    if [[ -n "${_numa_nodes}" ]] && (( _numa_nodes >= 2 )); then
        NUMA_PREFIX=(numactl -m 0,1)
    fi
fi

# ----------------- Launch nsys + trtllm-bench -----------------------------
# nsys flags:
#   -o <base>                     output prefix; nsys appends .nsys-rep
#   -f true                       force overwrite
#   -t cuda,nvtx,python-gil       trace what matters for decode kernels + bench markers
#   -c cudaProfilerApi            obey TLLM_PROFILE_START_STOP (cudaProfilerStart/Stop)
#   --cuda-graph-trace=node       expand cuda graphs into per-node spans (essential
#                                  for kernel-level decode analysis)
#   --export=sqlite               auto-export sqlite alongside .nsys-rep (NOT `-e`,
#                                  which is `--env-var` in nsys 2026.x and silently
#                                  consumes the next positional arg)
#   --trace-fork-before-exec=true follow fork into trtllm-bench's worker procs
echo "[nsys] start at $(date -u +%FT%TZ)"
set +e
nsys profile \
    -o "${NSYS_OUT_BASE}" -f true \
    -t "${NSYS_TRACE}" \
    -c cudaProfilerApi \
    --cuda-graph-trace=node \
    --export=sqlite \
    --trace-fork-before-exec=true \
    ${NSYS_EXTRA_ARGS} \
    "${NUMA_PREFIX[@]}" \
    trtllm-bench -m "${MODEL_CARD}" --model_path "${MODEL_PATH}" throughput \
        --tp "${TP}" --ep "${EP}" \
        --warmup "${WARMUP}" \
        --dataset "${DATASET_FILE}" \
        --backend pytorch \
        --max_batch_size "${BS}" \
        --max_num_tokens "${MAX_NUM_TOKENS}" \
        --max_seq_len "${MAX_SEQ_LEN}" \
        --kv_cache_free_gpu_mem_fraction "${KV_FRACTION}" \
        --concurrency "${BS}" \
        --extra_llm_api_options "${YAML_FILE}" \
        --sampler_options "${SAMPLER_FILE}" \
        --num_requests "${NUM_PROMPTS}" \
        --streaming \
    2>&1 | tee "${LOG_FILE}"
BENCH_EXIT=${PIPESTATUS[0]}
set -e
echo "[nsys] done at $(date -u +%FT%TZ)  exit=${BENCH_EXIT}"

if grep -qE "OutOfMemoryError|CUDA error: out of memory|CUDA out of memory" "${LOG_FILE}"; then
    echo "[fail] OOM — drop KV_FRACTION (0.8→0.7) or shrink ISL/BS"
    exit 2
fi
if [[ ${BENCH_EXIT} -ne 0 ]]; then
    echo "[fail] bench exited ${BENCH_EXIT}; inspect ${LOG_FILE}"
    exit 3
fi

# ----------------- Smoke-check the .nsys-rep ------------------------------
NSYS_REP="${NSYS_OUT_BASE}.nsys-rep"
if [[ ! -s "${NSYS_REP}" ]]; then
    echo "[warn] expected ${NSYS_REP} not produced — check log for nsys errors" >&2
    exit 3
fi
NSYS_SIZE=$(du -h "${NSYS_REP}" | awk '{print $1}')
echo "[ok] .nsys-rep ${NSYS_REP} (${NSYS_SIZE})"
echo "[ok] view with: nsys-ui ${NSYS_REP}"
echo "[ok] all artifacts in: ${RUN_DIR}"
