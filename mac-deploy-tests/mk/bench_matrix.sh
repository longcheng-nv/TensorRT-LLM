#!/bin/zsh
# Decode-speed matrix for the fused path: fixed-expert compute path + real routing configs.
#   bench_matrix.sh LABEL [TOKENS]   (results appended to logs/mk/bench.jsonl, summary printed)
cd /Users/loncheng/Models/DeepInfer-Mac
LABEL=${1:-run}; N=${2:-48}
OUT=/Users/loncheng/Models/logs/mk/bench.jsonl
B=/Users/loncheng/Models/dsv41-mac-tests/mk/mk_bench.py
run() { env -u HTTP_PROXY -u HTTPS_PROXY PYTHONDONTWRITEBYTECODE=1 .venv/bin/python $B --cache-gb 16 --tokens $N --label "$LABEL" --out $OUT "$@" 2>&1 | grep '^{' | tail -1; }
run --mode fixed
run --mode real --spec-k 4
run --mode real --spec-k 2 --spec-k2 2
run --mode real --sub-rank 4 --spec-k 2
run --mode real --sub-rank 4 --spec-k 2 --spec-k2 2
run --mode real --sub-rank 3 --spec-k 2
run --mode real --sub-rank 3 --spec-k 2 --spec-k2 2
run --mode real --sub-rank 2 --spec-k 2
run --mode real --sub-rank 2 --spec-k 2 --spec-k2 2
python3 - "$LABEL" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open("/Users/loncheng/Models/logs/mk/bench.jsonl") if l.strip()]
rows = [r for r in rows if r.get("label") == sys.argv[1]]
print(f"{'mode':6s} {'sub':>4s} {'K':>2s} {'K2':>2s} | {'ms/tok':>7s} {'tok/s':>6s} {'p10':>6s} {'p90':>6s} | {'hit':>5s} {'GB/tok':>6s} {'wait':>6s} {'subs':>5s}")
for r in rows:
    print(f"{r['mode']:6s} {str(r.get('sub_rank')):>4s} {r['spec_k']:>2d} {r.get('spec_k2', 0) or 0:>2d} | {r['ms_median']:7.1f} {r['tok_s_median']:6.2f} {r['ms_p10']:6.1f} {r['ms_p90']:6.1f} | "
          f"{r['hit_rate']:5.3f} {r['GB_per_token']:6.3f} {r['ssd_wait_ms']:6.1f} {r.get('subs_per_tok', 0):5.1f}")
EOF
