#!/bin/zsh
# Record one optimization iteration of the DeepInfer-Mac engine so the whole history can be replayed.
#   opt_iter.sh "<short description>"
# - commits the current engine tree on branch opt/compute-side and tags it opt-iter<N>
# - measures at a FIXED configuration: cache 18 GiB, SPEC_K=3, lookahead off, real Engram
#     * decode_speed.py (14-token prompt, 20 greedy tokens)   -> ms/token, tok/s, hit rate, SSD wait
#     * bench --prompt-len 1024 --gen 32 --rounds 2            -> ISL=1024 prefill/decode (round 2)
#     * correctness gate: tests/test_parity.py + tests/test_engine.py (MLX == PyTorch reference) + greedy probe 17*23
# - appends one JSON line to logs/opt/journal/iterations.jsonl and a markdown row to journal.md
set -u
setopt shwordsplit
REPO=/Users/loncheng/Models/DeepInfer-Mac; PY=$REPO/.venv/bin/python; MODEL=/Users/loncheng/Models/dsv41-mac
J=/Users/loncheng/Models/logs/opt/journal; mkdir -p "$J"
SPEC_K=${SPEC_K:-4}; DESC="${1:-"(no description)"} [cache 18, SPEC_K $SPEC_K]"
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
export PYTHONDONTWRITEBYTECODE=1 DSV41_REF=/Users/loncheng/Models/DeepSeek-V4.1-Flash-1017-ref/inference DSV41_TOKENIZER=/Users/loncheng/Models/DeepSeek-V4.1-Flash-1017-ref/tokenizer.json
cd "$REPO"
git rev-parse --verify -q opt/compute-side >/dev/null || git branch opt/compute-side
git checkout -q opt/compute-side
N=$(ls "$J"/iter-*.json 2>/dev/null | wc -l | tr -d ' ')
git add -A deepinfer_mac scripts && git commit -q -m "opt-iter$N: $DESC" --allow-empty && git tag -f "opt-iter$N" >/dev/null
SHA=$(git rev-parse --short HEAD)
LOG=$J/iter-$N.log; : > "$LOG"
echo "=== opt-iter$N @ $SHA: $DESC ($(date '+%F %T')) ===" | tee -a "$LOG"
pkill -f "deepinfer_mac serve" 2>/dev/null; sleep 2

# correctness gate (fast, weights-free parity vs official PyTorch reference)
"$PY" -m pytest -q -p no:cacheprovider tests/test_parity.py tests/test_engine.py tests/test_server.py >> "$LOG" 2>&1; PAR=$?
par_line=$(grep -E "passed|failed" "$LOG" | tail -n1)

# decode speed at the fixed configuration (3 runs, take median ms/token)
ms_list=(); tk=""; hr=""; gb=""; sw=""
for r in 1 2 3; do
  line=$(SPEC_K=$SPEC_K LOOKAHEAD_K=0 "$PY" scripts/decode_speed.py 18 --real-engram 2>&1 | grep -v '^\[hf\]' | grep 'decode:' | tail -1); echo "decode_speed run $r: $line" >> "$LOG"
  ms_list+=($(echo "$line" | sed -n 's/.*median \([0-9]*\) ms.*/\1/p'))
  tk=$(echo "$line" | sed -n 's/.*= \([0-9.]*\) tok\/s.*/\1/p'); hr=$(echo "$line" | sed -n 's/.*hit rate \([0-9.]*\).*/\1/p'); gb=$(echo "$line" | sed -n 's/.*| \([0-9.]*\) GB\/token.*/\1/p'); sw=$(echo "$line" | sed -n 's/.*SSD wait \([0-9]*\) ms.*/\1/p')
done
ms_med=$(printf '%s\n' "${ms_list[@]}" | sort -n | sed -n '2p')

# ISL=1024 bench
"$PY" -m deepinfer_mac bench --model "$MODEL" --cache-gb 18 --spec-k $SPEC_K --prompt-len 1024 --gen 32 --rounds 2 2>&1 | grep round >> "$LOG"
r2=$(grep '"round": 1' "$LOG" | tail -1)
pf=$(echo "$r2" | sed -n 's/.*"prefill_s": \([0-9.]*\).*/\1/p'); dtk=$(echo "$r2" | sed -n 's/.*"decode_tok_s": \([0-9.]*\).*/\1/p'); bhr=$(echo "$r2" | sed -n 's/.*"expert_hit_rate": \([0-9.]*\).*/\1/p')

# greedy probe
probe=$(printf 'What is 17 * 23? Answer with the number only.\n\n' | "$PY" -m deepinfer_mac chat --model "$MODEL" --cache-gb 18 --temperature 0 --max-tokens 8 2>/dev/null | grep 'ds>' | grep -q 391 && echo ok || echo WRONG)

python3 - "$N" "$SHA" "$DESC" "$ms_med" "$tk" "$hr" "$gb" "$sw" "$pf" "$dtk" "$bhr" "$PAR" "$par_line" "$probe" <<'EOF'
import json, sys, time
n, sha, desc, ms, tk, hr, gb, sw, pf, dtk, bhr, par, par_line, probe = sys.argv[1:]
rec = {"iter": int(n), "sha": sha, "desc": desc, "ts": time.strftime("%F %T"), "spec_k": int(__import__("os").environ.get("SPEC_K", "4")), "cache_gb": 18,
       "decode_ms_per_token_median3": float(ms or 0), "decode_tok_s_last": float(tk or 0), "hit_rate": float(hr or 0),
       "GB_per_token": float(gb or 0), "ssd_wait_ms": float(sw or 0),
       "isl1024_prefill_s": float(pf or 0), "isl1024_decode_tok_s": float(dtk or 0), "isl1024_hit_rate": float(bhr or 0),
       "parity_tests_rc": int(par), "parity_tests": par_line.strip(), "probe_391": probe}
J = "/Users/loncheng/Models/logs/opt/journal"
json.dump(rec, open(f"{J}/iter-{n}.json", "w"), indent=1)
open(f"{J}/iterations.jsonl", "a").write(json.dumps(rec, ensure_ascii=False) + "\n")
md = f"{J}/journal.md"
try: body = open(md).read()
except FileNotFoundError: body = "| iter | sha | description | decode ms/tok (median of 3) | tok/s | hit | GB/tok | SSD wait ms | ISL1024 prefill s | ISL1024 decode tok/s | parity | probe |\n|---|---|---|---|---|---|---|---|---|---|---|---|\n"
body += f"| {n} | {sha} | {desc} | {ms} | {tk} | {hr} | {gb} | {sw} | {pf} | {dtk} | {'PASS' if par=='0' else 'FAIL'} | {probe} |\n"
open(md, "w").write(body)
print(json.dumps(rec, ensure_ascii=False))
EOF
