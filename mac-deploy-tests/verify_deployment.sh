#!/bin/zsh
# Post-conversion verification loop for the DeepSeek-V4.1-Flash deployment on this Mac.
# Runs every gate in order, logs each to logs/verify/<step>.log, and writes a PASS/FAIL summary.
# Steps are independent: a failure is recorded and the loop continues.
#
#   CACHE_GB=10 /Users/loncheng/Models/dsv41-mac-tests/verify_deployment.sh [step ...]
# With no args all steps run; pass step names to run a subset, e.g. `verify_deployment.sh doctor api`.
set -u
REPO=/Users/loncheng/Models/DeepInfer-Mac
PY=$REPO/.venv/bin/python
MODEL=/Users/loncheng/Models/dsv41-mac
TESTS=/Users/loncheng/Models/dsv41-mac-tests
LOGS=/Users/loncheng/Models/logs/verify
CACHE_GB=${CACHE_GB:-10}
PORT=${PORT:-8000}
mkdir -p "$LOGS"
SUMMARY=$LOGS/summary.txt
: > "$SUMMARY"
export DSV41_REF=/Users/loncheng/Models/DeepSeek-V4.1-Flash-1017-ref/inference
export DSV41_TOKENIZER=/Users/loncheng/Models/DeepSeek-V4.1-Flash-1017-ref/tokenizer.json
export PYTHONDONTWRITEBYTECODE=1
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy   # direct Hub access for the scripts that range-read originals

want() { [ $# -eq 0 ] && return 0; for s in "$@"; do [ "$s" = "$STEP" ] && return 0; done; return 1; }
record() { local st=$1 name=$2 note=$3; printf '%-6s %-22s %s\n' "$st" "$name" "$note" | tee -a "$SUMMARY"; }
run_step() {  # run_step NAME "grep-pattern-for-pass" cmd...
  local name=$1 pat=$2; shift 2
  local log=$LOGS/$name.log
  echo "=== [$(date '+%H:%M:%S')] $name ===" | tee -a "$LOGS/all.log"
  local t0=$(date +%s)
  ( cd "$REPO" && "$@" ) > "$log" 2>&1
  local rc=$? dt=$(( $(date +%s) - t0 ))
  if [ $rc -eq 0 ] && grep -q -E "$pat" "$log"; then record PASS "$name" "${dt}s $(grep -E "$pat" "$log" | tail -n1 | cut -c1-150)"
  else record FAIL "$name" "rc=$rc ${dt}s $(grep -v '^\[hf\] retry' "$log" | tail -n 2 | tr '\n' ' ' | cut -c1-200)"; fi
}
ARGS=("$@")

STEP=doctor;      want "${ARGS[@]}" && run_step doctor "ready to run" "$PY" -m deepinfer_mac doctor --model "$MODEL"
STEP=weightfree;  want "${ARGS[@]}" && run_step weightfree "passed" "$PY" -m pytest -q -p no:cacheprovider tests/
STEP=encoding;    want "${ARGS[@]}" && run_step encoding "passed" "$PY" -I -m pytest -q -p no:cacheprovider /Users/loncheng/Models/DeepSeek-V4.1-Flash-1017-ref/encoding/test_encoding.py
STEP=oracles;     want "${ARGS[@]}" && run_step oracles "overall: PASS" python3 -I "$TESTS/gate_official_oracles.py" --workdir /private/tmp/claude-502/-Users-loncheng-Models/60697234-ef74-4285-9fb6-da2f8011ce9b/scratchpad/oracle-gate
STEP=quantgate;   want "${ARGS[@]}" && run_step quantgate "ALL PASS" "$PY" "$TESTS/gate_quant_layouts.py"
STEP=engram;      want "${ARGS[@]}" && run_step engram "ENGRAM CHECK OK" "$PY" scripts/check_engram.py
STEP=realcheck;   want "${ARGS[@]}" && run_step realcheck "RESULT: worst layer cosine 0\.999[9]|RESULT: worst layer cosine 1\.0" "$PY" scripts/real_check.py 6 --noquant
STEP=quantdamage; want "${ARGS[@]}" && run_step quantdamage "cos|cosine" "$PY" scripts/quant_damage.py 6
STEP=speed;       want "${ARGS[@]}" && run_step speed "tok/s" "$PY" scripts/decode_speed.py "$CACHE_GB" --real-engram
STEP=bench;       want "${ARGS[@]}" && run_step bench '"decode_tok_s"' "$PY" -m deepinfer_mac bench --model "$MODEL" --cache-gb "$CACHE_GB" --prompt-len 256 --gen 32 --rounds 2
STEP=cli;         want "${ARGS[@]}" && run_step cli "ds> .*391" sh -c "printf 'What is 17 * 23? Answer with the number only.\n\n' | $PY -m deepinfer_mac chat --model $MODEL --cache-gb $CACHE_GB --temperature 0 --max-tokens 16"
STEP=cli_zh;      want "${ARGS[@]}" && run_step cli_zh "ds> .*北京" sh -c "printf '中国的首都是哪里？只回答城市名。\n\n' | $PY -m deepinfer_mac chat --model $MODEL --cache-gb $CACHE_GB --temperature 0 --max-tokens 16"
STEP=structured;  want "${ARGS[@]}" && run_step structured "STRUCTURED OUTPUT PASS" "$PY" "$TESTS/structured_output_check.py" --cache-gb "$CACHE_GB" --thinking
STEP=offline;     want "${ARGS[@]}" && run_step offline "OFFLINE PROMPTS PASS" "$PY" "$TESTS/offline_prompts_check.py" --cache-gb "$CACHE_GB"

STEP=api
if want "${ARGS[@]}"; then
  echo "=== [$(date '+%H:%M:%S')] api: starting server on :$PORT ===" | tee -a "$LOGS/all.log"
  pkill -f "deepinfer_mac serve" 2>/dev/null; sleep 1
  ( cd "$REPO" && nohup "$PY" -m deepinfer_mac serve --model "$MODEL" --cache-gb "$CACHE_GB" --host 127.0.0.1 --port "$PORT" > "$LOGS/serve.log" 2>&1 & )
  for i in $(seq 1 120); do curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" && break; sleep 2; done
  if curl -s "http://127.0.0.1:$PORT/v1/models" | grep -q deepseek-flash; then
    t0=$(date +%s)
    ( cd "$TESTS" && rm -f results.jsonl && DSV41_BASE_URL="http://127.0.0.1:$PORT" DSV41_SKIP_SLOW=${DSV41_SKIP_SLOW:-0} "$PY" -m pytest -q -p no:cacheprovider -rA test_api_e2e.py ) > "$LOGS/api.log" 2>&1
    rc=$?; dt=$(( $(date +%s) - t0 ))
    tailline=$(grep -E "passed|failed|error" "$LOGS/api.log" | tail -n1 | cut -c1-150)
    if [ $rc -eq 0 ]; then record PASS api "${dt}s $tailline"; else record FAIL api "rc=$rc ${dt}s $tailline"; fi
    curl -s "http://127.0.0.1:$PORT/stats" > "$LOGS/stats.json"; record INFO api_stats "$(cut -c1-200 "$LOGS/stats.json")"
  else
    record FAIL api "server did not come up; see $LOGS/serve.log: $(tail -n 2 "$LOGS/serve.log" | tr '\n' ' ' | cut -c1-200)"
  fi
  [ "${KEEP_SERVER:-0}" = "1" ] || pkill -f "deepinfer_mac serve"
fi

echo; echo "===== SUMMARY ($(date '+%F %T'), cache ${CACHE_GB} GiB) ====="; cat "$SUMMARY"
grep -q '^FAIL' "$SUMMARY" && exit 1 || exit 0
