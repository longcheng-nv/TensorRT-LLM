#!/bin/zsh
# Lossless decode-overlap sweep: expert cache size x idle-window speculation (SPEC_K) x router
# lookahead (LOOKAHEAD_K). Uses the engine's decode_speed.py (real Engram, greedy, 20 tokens).
set -u
setopt shwordsplit
REPO=/Users/loncheng/Models/DeepInfer-Mac; PY=$REPO/.venv/bin/python; OUT=/Users/loncheng/Models/logs/opt/sweep_overlap.tsv
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
printf 'cache_gb\tspec_k\tlookahead_k\tprefill_s\tms_per_token\ttok_s\thit_rate\tGB_per_token\tssd_wait_ms\n' > "$OUT"
for cache in ${CACHES:-9 18}; do for spec in ${SPECS:-0 3 6}; do for la in ${LOOKS:-0 3 6}; do
  line=$(cd "$REPO" && SPEC_K=$spec LOOKAHEAD_K=$la "$PY" scripts/decode_speed.py $cache --real-engram 2>&1 | grep -v '^\[hf\]' | grep 'decode:' | tail -1)
  pf=$(echo "$line" | sed -n 's/.*tok \([0-9.]*\)s |.*/\1/p'); ms=$(echo "$line" | sed -n 's/.*median \([0-9]*\) ms.*/\1/p'); tk=$(echo "$line" | sed -n 's/.*= \([0-9.]*\) tok\/s.*/\1/p')
  hr=$(echo "$line" | sed -n 's/.*hit rate \([0-9.]*\).*/\1/p'); gb=$(echo "$line" | sed -n 's/.*| \([0-9.]*\) GB\/token.*/\1/p'); sw=$(echo "$line" | sed -n 's/.*SSD wait \([0-9]*\) ms.*/\1/p')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$cache" "$spec" "$la" "${pf:-?}" "${ms:-?}" "${tk:-?}" "${hr:-?}" "${gb:-?}" "${sw:-?}" | tee -a "$OUT"
done; done; done
echo; column -t -s $'\t' "$OUT"
