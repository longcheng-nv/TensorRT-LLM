#!/bin/bash
# Final validation of the shipped engine (branch deepinfer-mac-dsv41-flash):
#   tps  : start the server with NO mode flag (the CLI default, i.e. --fast) and measure streamed decode
#          tok/s through HTTP with mac-deploy-tests/mk/serve_tps.py (4 prompts x 300 tokens)
#   e2e  : API end-to-end suite (test_api_e2e.py) against the default mode (expect 26/28: only the two
#          determinism tests fail, by design) and against --lossless (expect 28/28)
# usage: final_validation.sh tps|e2e|all        results: /Users/loncheng/Models/logs/final/
set -u
REPO=/Users/loncheng/Models/DeepInfer-Mac
PY=$REPO/.venv/bin/python
MODEL=/Users/loncheng/Models/dsv41-mac
TESTS=/Users/loncheng/Models/dsv41-mac-tests
OUT=/Users/loncheng/Models/logs/final
PORT=8000
CACHE_GB=${CACHE_GB:-16}
mkdir -p "$OUT"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

start_server() {  # $1 = label, rest = extra flags
  local label=$1; shift
  pkill -f "deepinfer_mac serve" 2>/dev/null; sleep 2
  ( cd "$REPO" && nohup "$PY" -m deepinfer_mac serve "$@" --model "$MODEL" --cache-gb "$CACHE_GB" --host 127.0.0.1 --port "$PORT" > "$OUT/serve_$label.log" 2>&1 & )
  for i in $(seq 1 150); do curl -s --noproxy '*' -m 2 -o /dev/null "http://127.0.0.1:$PORT/v1/models" && break; sleep 2; done
  curl -s --noproxy '*' "http://127.0.0.1:$PORT/v1/models" | grep -q deepseek-flash || { echo "[$label] server did not come up"; tail -n 5 "$OUT/serve_$label.log"; return 1; }
  echo "[$label] $(grep -m1 'decode mode' "$OUT/serve_$label.log")"
  caffeinate -i -w "$(pgrep -f 'deepinfer_mac serve' | head -1)" &
}

stop_server() { pkill -f "deepinfer_mac serve" 2>/dev/null; sleep 2; }

run_e2e() {  # $1 = label
  ( cd "$TESTS" && DSV41_BASE_URL="http://127.0.0.1:$PORT" "$PY" -m pytest -q -p no:cacheprovider -rA test_api_e2e.py ) > "$OUT/api_$1.log" 2>&1
  echo "[$1] api e2e: $(grep -E 'passed|failed|error' "$OUT/api_$1.log" | tail -n1 | cut -c1-160)"
  grep -E '^FAILED' "$OUT/api_$1.log" | cut -c1-140
}

echo "power: $(pmset -g batt | head -1 | sed "s/Now drawing from //")  $(date '+%F %T')"
MODE=${1:-all}  # macOS ships bash 3.2: no case fall-through, plain ifs
if [ "$MODE" = tps ] || [ "$MODE" = all ]; then
  if start_server default; then
    "$PY" "$REPO/mac-deploy-tests/mk/serve_tps.py" --url "http://127.0.0.1:$PORT" --max-tokens 300 2>&1 | tee "$OUT/serve_tps_default.log"
    if [ "$MODE" = all ]; then run_e2e default; fi
    stop_server
  fi
fi
if [ "$MODE" = e2e ]; then
  if start_server default; then run_e2e default; stop_server; fi
fi
if [ "$MODE" = e2e ] || [ "$MODE" = all ]; then
  if start_server lossless --lossless; then run_e2e lossless; stop_server; fi
fi
echo "done $(date '+%T')"
