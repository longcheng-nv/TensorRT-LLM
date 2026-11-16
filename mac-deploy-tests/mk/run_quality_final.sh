#!/bin/zsh
cd /Users/loncheng/Models/DeepInfer-Mac
Q=/Users/loncheng/Models/logs/mk/quality
run() { local name=$1; shift
  echo "=== $name $(date '+%T')" >> $Q/final.log
  env -u HTTP_PROXY -u HTTPS_PROXY PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /Users/loncheng/Models/dsv41-mac-tests/mk/mk_quality.py $Q/$name.json --cache-gb 16 "$@" > $Q/$name.log 2>&1
  tail -1 $Q/$name.log >> $Q/final.log; }
run sub2_40 --set general --max-cases 40 --sub-rank 2 --spec-k 2
run router_base60 --set router --manifest manifest-short.tsv --max-cases 60 --spec-k 4
run router_sub2_60 --set router --manifest manifest-short.tsv --max-cases 60 --sub-rank 2 --spec-k 2
echo "=== done $(date '+%T')" >> $Q/final.log
