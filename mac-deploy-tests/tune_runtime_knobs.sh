#!/bin/zsh
# Compare DeepInfer-Mac's runtime (weights-unchanged) speed knobs at one expert-cache size.
# For each setting: `bench` (2 rounds; round 2 = warm cache) + two greedy CLI probes for correctness.
#   CACHE_GB=9 /Users/loncheng/Models/dsv41-mac-tests/tune_runtime_knobs.sh
set -u
REPO=/Users/loncheng/Models/DeepInfer-Mac; PY=$REPO/.venv/bin/python; MODEL=/Users/loncheng/Models/dsv41-mac
CACHE_GB=${CACHE_GB:-9}; LOGS=/Users/loncheng/Models/logs/tune; mkdir -p "$LOGS"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
OUT=$LOGS/results.tsv; printf 'setting\tround2_decode_tok_s\tround2_hit_rate\tround2_GB_per_32tok\tprefill_tok_s\tprobe_391\tprobe_beijing\n' > "$OUT"
typeset -A SETTINGS
SETTINGS=(
  baseline ""
  moe_topk4 "--moe-topk 4"
  skip_miss_008 "--skip-miss 0.08"
  prefill_topk4 "--prefill-topk 4"
  topk4_skip008 "--moe-topk 4 --skip-miss 0.08"
)
for name in baseline moe_topk4 skip_miss_008 prefill_topk4 topk4_skip008; do
  flags=${SETTINGS[$name]}
  echo "=== [$(date '+%H:%M:%S')] $name ($flags) ==="
  (cd "$REPO" && eval "$PY -m deepinfer_mac bench --model $MODEL --cache-gb $CACHE_GB --prompt-len 256 --gen 32 --rounds 2 $flags") > "$LOGS/bench-$name.log" 2>&1
  r2=$(grep '"round": 1' "$LOGS/bench-$name.log" | tail -1)
  tok=$(echo "$r2" | sed -n 's/.*"decode_tok_s": \([0-9.]*\).*/\1/p'); hit=$(echo "$r2" | sed -n 's/.*"expert_hit_rate": \([0-9.]*\).*/\1/p')
  gb=$(echo "$r2" | sed -n 's/.*"expert_GB_read": \([0-9.]*\).*/\1/p'); pf=$(echo "$r2" | sed -n 's/.*"prefill_tok_s": \([0-9.]*\).*/\1/p')
  a=$(cd "$REPO" && printf 'What is 17 * 23? Answer with the number only.\n\n' | eval "$PY -m deepinfer_mac chat --model $MODEL --cache-gb $CACHE_GB --temperature 0 --max-tokens 12 $flags" 2>/dev/null | grep 'ds>' | grep -q 391 && echo ok || echo WRONG)
  b=$(cd "$REPO" && printf '中国的首都是哪里？只回答城市名。\n\n' | eval "$PY -m deepinfer_mac chat --model $MODEL --cache-gb $CACHE_GB --temperature 0 --max-tokens 12 $flags" 2>/dev/null | grep 'ds>' | grep -q 北京 && echo ok || echo WRONG)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "${tok:-?}" "${hit:-?}" "${gb:-?}" "${pf:-?}" "$a" "$b" | tee -a "$OUT"
done
echo; echo "===== knob comparison (cache ${CACHE_GB} GiB) ====="; column -t -s $'\t' "$OUT"
