#!/bin/zsh
cd /Users/loncheng/Models/DeepInfer-Mac
Q=/Users/loncheng/Models/logs/mk/quality
run() { local name=$1; shift
  echo "=== $name $(date '+%T')" >> $Q/sweep3.log
  env -u HTTP_PROXY -u HTTPS_PROXY PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /Users/loncheng/Models/dsv41-mac-tests/mk/mk_quality.py $Q/$name.json --set general --max-cases 20 --cache-gb 16 "$@" > $Q/$name.log 2>&1
  tail -1 $Q/$name.log >> $Q/sweep3.log; }
run sub2_spec2 --sub-rank 2 --spec-k 2
run sub3g010_spec2 --sub-rank 3 --sub-gap 0.10 --spec-k 2
echo "=== done $(date '+%T')" >> $Q/sweep3.log
