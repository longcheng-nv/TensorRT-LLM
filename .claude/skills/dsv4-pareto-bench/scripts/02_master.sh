#!/bin/bash
# Master driver — loops over pending cells in plan.csv, respects job budget,
# idempotent on restart. Skips cells listed in completed.txt or failed.txt
# (unless listed in force_retry.txt).
#
# Required env (inherited by 01_run_one_cell.sh):
#   MODEL_PATH         absolute path to DSv4 Flash / Pro weights;
#                      OR omit and set MODEL_VARIANT=Flash|Pro to let
#                      01_run_one_cell.sh auto-detect by cluster
#                      (SC computelab → /home/scratch.trt_llm_data_ci/llm-models/)
#
# Optional env:
#   PERFDIR            sweep working dir (default: parent of this script's dir)
#   MAX_WALL_SECONDS   default 14000 (3h 53m, leaves ~7 min cleanup margin)
#   ISL_FILTER         e.g. "131072" to run only ISL=128K cells
#   BS_FILTER          e.g. "1"
#   MTP_FILTER         e.g. "3"
#   GVR_FILTER         "1" or "0"
#   MODE_FILTER        "TEP" or "DEP"
#   DRY_RUN            "1" prints the would-run list and exits

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PERFDIR="${PERFDIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PLAN="${PERFDIR}/plan.csv"
COMPLETED="${PERFDIR}/completed.txt"
FAILED="${PERFDIR}/failed.txt"

# MODEL_PATH may be auto-detected by 01_run_one_cell.sh via MODEL_VARIANT.
# Pre-check upfront so we don't enumerate the plan and then fail every cell.
if [[ -z "${MODEL_PATH:-}" ]]; then
    _variant="${MODEL_VARIANT:-Flash}"
    for _cand in \
        "/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V4-${_variant}" \
        "/lustre/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/common/DeepSeek-V4-${_variant}" \
        ; do
        if [[ -d "${_cand}" ]]; then MODEL_PATH="${_cand}"; break; fi
    done
    unset _cand _variant
fi
: "${MODEL_PATH:?MODEL_PATH unset and auto-detect failed — set MODEL_PATH explicitly or MODEL_VARIANT=Flash|Pro}"

MAX_WALL_SECONDS="${MAX_WALL_SECONDS:-14000}"
START_EPOCH=$(date +%s)
BUDGET_END=$(( START_EPOCH + MAX_WALL_SECONDS ))

# If running under SLURM, take the earlier of MAX_WALL_SECONDS and SLURM end.
if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
    if [[ "${SLURM_JOB_END_TIME}" -lt "${BUDGET_END}" ]]; then
        BUDGET_END="${SLURM_JOB_END_TIME}"
        echo "[budget] using SLURM_JOB_END_TIME=${SLURM_JOB_END_TIME}"
    fi
fi

ISL_FILTER="${ISL_FILTER:-}"
BS_FILTER="${BS_FILTER:-}"
MTP_FILTER="${MTP_FILTER:-}"
GVR_FILTER="${GVR_FILTER:-}"
MODE_FILTER="${MODE_FILTER:-}"
DRY_RUN="${DRY_RUN:-0}"

touch "${COMPLETED}" "${FAILED}"
[[ -f "${PLAN}" ]] || { echo "missing ${PLAN}; run 00_generate_plan.py first" >&2; exit 1; }
mkdir -p "${PERFDIR}/results"

# Pull sweep-wide DATASET_NUM_PROMPTS / MULTI_ROUND from bench.env so all
# BS cells consume identical prompts (smaller-BS cells use a strict prefix).
if [[ -f "${PERFDIR}/bench.env" ]]; then
    # shellcheck disable=SC1091
    source "${PERFDIR}/bench.env"
fi

echo "================================================================"
echo "DSv4 paired-feature master driver"
echo "PERFDIR : ${PERFDIR}"
echo "started : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "budget  : ${MAX_WALL_SECONDS}s (until $(date -u -d @${BUDGET_END} +%Y-%m-%dT%H:%M:%SZ))"
echo "filters : ISL=${ISL_FILTER:-*} BS=${BS_FILTER:-*} MTP=${MTP_FILTER:-*} Mode=${MODE_FILTER:-*} GVR=${GVR_FILTER:-*}"
[[ "${DRY_RUN}" == "1" ]] && echo "DRY_RUN=1 (no cells will execute)"
echo "================================================================"

ran=0
skipped=0
failed=0
budget_exits=0

# Read plan.csv (skip header).
tail -n +2 "${PLAN}" | while IFS=',' read -r cell_id isl osl bs mtp mode gvr est_sec; do
    [[ -n "${ISL_FILTER}"  && "${isl}"  != "${ISL_FILTER}"  ]] && continue
    [[ -n "${BS_FILTER}"   && "${bs}"   != "${BS_FILTER}"   ]] && continue
    [[ -n "${MTP_FILTER}"  && "${mtp}"  != "${MTP_FILTER}"  ]] && continue
    [[ -n "${GVR_FILTER}"  && "${gvr}"  != "${GVR_FILTER}"  ]] && continue
    [[ -n "${MODE_FILTER}" && "${mode}" != "${MODE_FILTER}" ]] && continue

    if grep -qxF "${cell_id}" "${COMPLETED}"; then
        skipped=$(( skipped + 1 ))
        continue
    fi
    if grep -qP "^${cell_id}\b" "${FAILED}" \
       && ! grep -qxF "${cell_id}" "${PERFDIR}/force_retry.txt" 2>/dev/null; then
        skipped=$(( skipped + 1 ))
        continue
    fi

    now=$(date +%s)
    remaining=$(( BUDGET_END - now ))
    need=$(( est_sec * 3 / 2 ))
    if [[ ${remaining} -lt ${need} ]]; then
        echo "[budget] only ${remaining}s left, next cell needs ~${need}s — exiting cleanly"
        budget_exits=1
        break
    fi

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "[dry] would run ${cell_id} (est ${est_sec}s, remaining ${remaining}s)"
        continue
    fi

    set +e
    # `< /dev/null` isolates the cell's stdin from the outer `tail | while read`
    # pipe. Without this, trtllm-bench's distributed launcher (torchrun/mpirun)
    # can consume the pipe, prematurely ending the while loop after 1 iteration.
    PERFDIR="${PERFDIR}" \
    bash "${SCRIPT_DIR}/01_run_one_cell.sh" \
        "${cell_id}" "${isl}" "${osl}" "${bs}" "${mtp}" "${mode}" "${gvr}" \
        < /dev/null
    rc=$?
    set -e
    case "${rc}" in
        0) ran=$(( ran + 1 )) ;;
        1) skipped=$(( skipped + 1 )) ;;
        *) failed=$(( failed + 1 )) ;;
    esac
done

echo "================================================================"
echo "ended   : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "ran=${ran} skipped=${skipped} failed=${failed} budget_exit=${budget_exits}"
echo "completed total: $(wc -l < "${COMPLETED}")"
echo "failed total:    $(wc -l < "${FAILED}")"
echo "================================================================"
