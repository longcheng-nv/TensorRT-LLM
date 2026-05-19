#!/usr/bin/env bash
# Summarize the live state of a dsv4-gsm8k-eval log.
# Usage: bash check_progress.sh <log_file>

set -euo pipefail

LOG="${1:-}"
if [ -z "$LOG" ] || [ ! -f "$LOG" ]; then
  echo "usage: $0 <log_file>" >&2
  exit 2
fi

echo "=== file ==="
ls -la "$LOG"

echo
echo "=== process(es) ==="
ps -ef | grep -E "tensorrt_llm.commands.eval" | grep -v grep | awk '{print "pid="$2, "cputime="$7, "cmd="$8}' | head -3 \
  || echo "no eval process running"

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader 2>/dev/null \
  || echo "nvidia-smi unavailable"

echo
echo "=== failure markers ==="
if grep -qE "illegal memory access|RequestError|Traceback" "$LOG"; then
  echo "FAILURES DETECTED:"
  grep -nE "illegal memory access|RequestError|Traceback" "$LOG" | head -5
else
  echo "(none)"
fi

echo
echo "=== hadamard warning (DSv4 hard dependency) ==="
if grep -q "Sparse MLA will skip hadamard" "$LOG"; then
  echo "WARNING: 'Sparse MLA will skip hadamard' present. Install fast_hadamard_transform; eval WILL crash."
else
  echo "(not present — good)"
fi

echo
echo "=== phase markers ==="
grep -E "Model init total|TRTLLM initialization time|TRTLLM execution time" "$LOG" | head -10 \
  || echo "(none yet — still loading)"

echo
echo "=== last inference progress (tqdm uses \\r so we split it) ==="
# tqdm writes carriage-return updates; convert to newlines and grab last few
tail -c 16384 "$LOG" | tr '\r' '\n' | grep -E "Fetching responses:" | tail -3 \
  || echo "(no Fetching-responses line yet)"

echo
echo "=== final accuracy table (if present) ==="
if grep -q "gsm8k average accuracy" "$LOG"; then
  echo "RUN COMPLETE."
  grep -E "exact_match|gsm8k average accuracy|Results saved to" "$LOG" | tail -10
else
  echo "(not yet — run still in progress or failed before scoring)"
fi
