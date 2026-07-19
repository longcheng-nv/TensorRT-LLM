#!/bin/bash
# Checkpoint sync: copy qab3 jsonl + .done markers + batch logs (NEVER nsys
# reps — they embed env tokens) from /tmp to NFS every 60s until all 9 done.
SRC=/tmp/gvrqab/qab3_results
DST="$(dirname "$(readlink -f "$0")")/qab3_ckpt"
mkdir -p "$DST"
while true; do
  cp -u "$SRC"/*.jsonl "$SRC"/.done_* "$SRC"/*.log "$DST"/ 2>/dev/null
  n=$(ls "$SRC"/.done_* 2>/dev/null | wc -l)
  [ "$n" -ge 9 ] && { cp -u "$SRC"/*.jsonl "$SRC"/.done_* "$SRC"/*.log "$DST"/ 2>/dev/null; echo "CKPT_SYNC_DONE ($n)"; break; }
  sleep 60
done
