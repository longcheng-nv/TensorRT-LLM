#!/bin/bash
# Poll the KernelFactory campaign; append status snapshots to status_log.jsonl.
# Exits (-> notifies the agent) when: terminal phase, round advances, or best
# speedup changes vs the state captured at loop start.
CID=${1:-tfb91bvwm972kfyf1bc1trj5e0}
DIR=$(dirname "$0")
LOG=$DIR/status_log.jsonl
snap() { kf --format json campaign show "$CID" 2>/dev/null; }
S0=$(snap)
R0=$(echo "$S0" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('current_round'))" 2>/dev/null)
SP0=$(echo "$S0" | python3 -c "import sys,json; print(json.load(sys.stdin).get('best_speedup') or 0)" 2>/dev/null)
echo "$S0" | python3 -c "import sys,json,time; d=json.load(sys.stdin); d['_ts']=time.strftime('%FT%TZ',time.gmtime()); print(json.dumps(d))" >> "$LOG" 2>/dev/null
while true; do
  sleep 120
  S=$(snap) || continue
  [ -z "$S" ] && continue
  echo "$S" | python3 -c "import sys,json,time; d=json.load(sys.stdin); d['_ts']=time.strftime('%FT%TZ',time.gmtime()); print(json.dumps(d))" >> "$LOG" 2>/dev/null
  PH=$(echo "$S" | python3 -c "import sys,json; print(json.load(sys.stdin).get('phase',''))" 2>/dev/null)
  R=$(echo "$S" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('current_round'))" 2>/dev/null)
  SP=$(echo "$S" | python3 -c "import sys,json; print(json.load(sys.stdin).get('best_speedup') or 0)" 2>/dev/null)
  case "$PH" in Completed|Failed|Cancelled|Converged)
    echo "TERMINAL phase=$PH round=$R"; exit 0;; esac
  UP=$(python3 -c "print(1 if float('$SP') - float('$SP0') >= 0.02 else 0)" 2>/dev/null)
  if [ "$UP" = "1" ]; then echo "SPEEDUP IMPROVED $SP0 -> $SP round=$R"; exit 0; fi
  if [ "$R" != "$R0" ]; then echo "ROUND ADVANCE $R0 -> $R"; exit 0; fi
done
