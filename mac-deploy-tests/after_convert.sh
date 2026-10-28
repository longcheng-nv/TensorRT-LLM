#!/bin/zsh
# Stage 1 of the hand-off: wait for the full conversion to finish, confirm completeness with
# `doctor`, size the expert cache from the memory actually available, then launch the whole
# verification suite detached (stage 2) and exit so the orchestrator gets notified.
set -u
REPO=/Users/loncheng/Models/DeepInfer-Mac
PY=$REPO/.venv/bin/python
MODEL=/Users/loncheng/Models/dsv41-mac
TESTS=/Users/loncheng/Models/dsv41-mac-tests
LOGS=/Users/loncheng/Models/logs
mkdir -p "$LOGS/verify"
while pgrep -f "deepinfer_mac.convert.remote" >/dev/null; do sleep 60; done
echo "converter exited at $(date '+%F %T')"
"$TESTS/convert_progress.sh"
"$PY" -m deepinfer_mac doctor --model "$MODEL" | tee "$LOGS/verify/doctor-after-convert.txt"
if ! grep -q "conversion: complete" "$LOGS/verify/doctor-after-convert.txt"; then
  echo "CONVERSION INCOMPLETE — not starting verification"; tail -n 5 "$LOGS/convert-full.log"; exit 1
fi
du -sh "$MODEL"; df -h /System/Volumes/Data | tail -n 1
# expert cache: what macOS can hand out now minus the ~7.6 GiB dense weights and a 4 GiB margin, clamped to [6, 16] GiB
avail_gib=$(vm_stat | awk '/Pages free/{f=$3}/Pages inactive/{i=$3}/Pages purgeable/{p=$3}/Pages speculative/{s=$3}END{printf "%d", (f+i+p+s)*16384/2^30}')
cache=$(( avail_gib - 8 - 4 )); [ $cache -lt 6 ] && cache=6; [ $cache -gt 16 ] && cache=16
echo "available ~${avail_gib} GiB -> CACHE_GB=${cache}"
echo "$cache" > "$LOGS/verify/cache_gb.txt"
nohup env CACHE_GB=$cache KEEP_SERVER=1 "$TESTS/verify_deployment.sh" > "$LOGS/verify/run.log" 2>&1 < /dev/null &
disown
echo "verification launched (pid $!), log $LOGS/verify/run.log"
