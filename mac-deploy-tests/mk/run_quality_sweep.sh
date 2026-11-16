#!/bin/zsh
# Quality sweep: official-API teacher-forced NLL / top-1 agreement for routing policies.
cd /Users/loncheng/Models/DeepInfer-Mac
Q=/Users/loncheng/Models/logs/mk/quality
N=${N:-40}
run() {  # name sub_rank spec_k
  local name=$1; shift
  echo "=== $name $(date '+%T')" >> $Q/sweep.log
  env -u HTTP_PROXY -u HTTPS_PROXY PYTHONDONTWRITEBYTECODE=1 .venv/bin/python /Users/loncheng/Models/dsv41-mac-tests/mk/mk_quality.py $Q/$name.json --set general --max-cases $N --cache-gb 16 "$@" > $Q/$name.log 2>&1
  tail -1 $Q/$name.log >> $Q/sweep.log
}
run base_spec4 --spec-k 4
run sub4_spec4 --sub-rank 4 --spec-k 4
run sub3_spec2 --sub-rank 3 --spec-k 2
# run sub2_spec2 --sub-rank 2 --spec-k 2
echo "=== done $(date '+%T')" >> $Q/sweep.log
