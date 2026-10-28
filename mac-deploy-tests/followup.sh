#!/bin/zsh
# Follow-up verification after the server threading patch: full API suite, then the GPU-exclusive
# re-runs (offline prompts, thinking-mode structured output, quant_damage N=2, runtime knob sweep),
# then bring the API server back up and leave it running.
set -u
REPO=/Users/loncheng/Models/DeepInfer-Mac; PY=$REPO/.venv/bin/python; MODEL=/Users/loncheng/Models/dsv41-mac
TESTS=/Users/loncheng/Models/dsv41-mac-tests; LOGS=/Users/loncheng/Models/logs/verify; CACHE_GB=${CACHE_GB:-9}
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export PYTHONDONTWRITEBYTECODE=1
S=$LOGS/summary-followup.txt; : > "$S"
rec() { printf '%-6s %-22s %s\n' "$1" "$2" "$3" | tee -a "$S"; }
step() { local name=$1 pat=$2; shift 2; local log=$LOGS/followup-$name.log; local t0=$(date +%s); ( cd "$REPO" && "$@" ) > "$log" 2>&1; local rc=$? dt=$(( $(date +%s)-t0 ));
  if [ $rc -eq 0 ] && grep -q -E "$pat" "$log"; then rec PASS "$name" "${dt}s $(grep -E "$pat" "$log" | tail -n1 | cut -c1-140)"; else rec FAIL "$name" "rc=$rc ${dt}s $(grep -v '^\[hf\]' "$log" | tail -n 2 | tr '\n' ' ' | cut -c1-200)"; fi; }

# 1. full API suite against the already-running patched server
for i in $(seq 1 60); do curl -s -o /dev/null http://127.0.0.1:8000/v1/models && break; sleep 2; done
( cd "$TESTS" && rm -f results.jsonl && DSV41_BASE_URL=http://127.0.0.1:8000 "$PY" -m pytest -q -p no:cacheprovider -rA test_api_e2e.py ) > "$LOGS/followup-api.log" 2>&1
rc=$?; line=$(grep -E "passed|failed" "$LOGS/followup-api.log" | tail -n1 | cut -c1-120); [ $rc -eq 0 ] && rec PASS api_full "$line" || rec FAIL api_full "$line"
curl -s http://127.0.0.1:8000/stats > "$LOGS/followup-stats.json"
pkill -f "deepinfer_mac serve"; sleep 5

# 2. GPU-exclusive re-runs
step offline "OFFLINE PROMPTS PASS" "$PY" "$TESTS/offline_prompts_check.py" --cache-gb "$CACHE_GB"
step structured_thinking "STRUCTURED OUTPUT PASS" env SO_THINK_MAX_TOKENS=3000 "$PY" "$TESTS/structured_output_check.py" --cache-gb "$CACHE_GB" --thinking
step quantdamage2 "cos" perl -e 'alarm 1500; exec @ARGV' "$PY" scripts/quant_damage.py 2
( CACHE_GB=$CACHE_GB "$TESTS/tune_runtime_knobs.sh" ) > "$LOGS/followup-knobs.log" 2>&1 && rec PASS knobs "see $LOGS/tune/results.tsv" || rec FAIL knobs "see $LOGS/followup-knobs.log"

# 3. bring the server back
( cd "$REPO" && nohup "$PY" -m deepinfer_mac serve --model "$MODEL" --cache-gb "$CACHE_GB" --host 127.0.0.1 --port 8000 > "$LOGS/serve3.log" 2>&1 < /dev/null & )
for i in $(seq 1 90); do curl -s -o /dev/null http://127.0.0.1:8000/v1/models && break; sleep 2; done
curl -s http://127.0.0.1:8000/v1/models | grep -q deepseek-flash && rec PASS server_up "127.0.0.1:8000 /v1/models ok" || rec FAIL server_up "server not responding"
echo "===== FOLLOWUP DONE $(date '+%F %T') ====="; cat "$S"
