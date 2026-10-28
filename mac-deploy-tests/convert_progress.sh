#!/bin/zsh
# One-line conversion status: tasks done/total, GB in, MB/s, disk free, ETA, and whether the process is alive.
OUT=/Users/loncheng/Models/dsv41-mac
LOG=/Users/loncheng/Models/logs/convert-full.log
TOTAL=2748
done_n=$(wc -w < "$OUT/progress.log" 2>/dev/null | tr -d ' ')
last=$(grep -a '^\[convert\] [0-9]*/' "$LOG" 2>/dev/null | tail -n 1)
alive=$(pgrep -f "deepinfer_mac.convert.remote" >/dev/null && echo alive || echo STOPPED)
# bytes left from doctor-equivalent arithmetic: 510.29 GB total payload; use the log's "GB in" for this run only
gb_in=$(echo "$last" | sed -n 's/.* \([0-9.]*\) GB in, \([0-9.]*\) MB\/s.*/\1/p')
rate=$(echo "$last" | sed -n 's/.* \([0-9.]*\) GB in, \([0-9.]*\) MB\/s.*/\2/p')
errs=$(grep -a -c -E 'Traceback|Error|retry' "$LOG" 2>/dev/null)
free=$(df -g /System/Volumes/Data | awk 'NR==2{print $4}')
eta=""
if [[ -n "$rate" && "$rate" != "0.0" ]]; then
  # remaining bytes ~ proportional to remaining tasks (expert tasks dominate, ~154 MB each; EG tasks ~264 MB)
  left_tasks=$(( TOTAL - done_n ))
  eta_h=$(python3 -c "print(round(($left_tasks*0.19e3)/($rate)/3600,1))" 2>/dev/null)
  eta="ETA~${eta_h}h"
fi
echo "$(date '+%H:%M:%S') $alive tasks ${done_n}/${TOTAL} | run: ${gb_in:-?} GB in @ ${rate:-?} MB/s | disk free ${free} GiB | log errors/retries: ${errs} | ${eta}"
