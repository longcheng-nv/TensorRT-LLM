#!/bin/zsh
# Replay the decode benchmark of a recorded optimization iteration (or all of them) from its git tag,
# in a detached worktree so the working tree is untouched.
#   opt_replay.sh <N> | all
set -u
REPO=/Users/loncheng/Models/DeepInfer-Mac; PY=$REPO/.venv/bin/python
J=/Users/loncheng/Models/logs/opt/journal; WT=/Users/loncheng/Models/logs/opt/replay-worktree
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
replay_one() {
  local n=$1 tag=opt-iter$1
  rm -rf "$WT"; git -C "$REPO" worktree prune; git -C "$REPO" worktree add -q --detach "$WT" "$tag" || { echo "iter $n: tag $tag missing"; return 1; }
  # the worktree has no venv: run the recorded engine code with the shared interpreter via PYTHONPATH
  local sk; sk=$(python3 -c "import json;print(json.load(open('$J/iter-$n.json')).get('spec_k',3))")
  local line; line=$(cd "$WT" && SPEC_K=$sk LOOKAHEAD_K=0 PYTHONPATH="$WT" "$PY" scripts/decode_speed.py 18 --real-engram 2>&1 | grep -v '^\[hf\]' | grep 'decode:' | tail -1)
  local rec; rec=$(python3 -c "import json;d=json.load(open('$J/iter-$n.json'));print(d['decode_ms_per_token_median3'], '|', d['desc'])")
  echo "iter $n ($tag): recorded $rec"
  echo "          replay : $line"
  git -C "$REPO" worktree remove --force "$WT"
}
if [ "${1:-}" = "all" ]; then for f in "$J"/iter-*.json; do replay_one "$(basename "$f" .json | sed 's/iter-//')"; done; else replay_one "$1"; fi
