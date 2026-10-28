#!/bin/zsh
# Launch (or resume) the full DeepInfer-Mac conversion of DeepSeek-V4.1-Flash-1017, detached.
#  - proxy env vars are unset: the local privoxy caps Hugging Face at ~9 MB/s, direct Wi-Fi gives ~31 MB/s
#  - caffeinate -dims keeps the Mac awake on AC power; progress.log makes any restart resume
#  - the same command re-run after an interruption continues where it stopped
set -euo pipefail
REPO=/Users/loncheng/Models/DeepInfer-Mac
OUT=/Users/loncheng/Models/dsv41-mac
LOG=/Users/loncheng/Models/logs/convert-full.log
mkdir -p /Users/loncheng/Models/logs
if pgrep -f "deepinfer_mac.convert.remote" >/dev/null; then
  echo "a converter is already running (pid $(pgrep -f deepinfer_mac.convert.remote | head -1)); not starting another"; exit 2
fi
cd "$REPO"
echo "=== $(date '+%F %T') launch/resume full conversion -> $OUT" >> "$LOG"
env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy \
  nohup caffeinate -dims "$REPO/.venv/bin/python" -u -m deepinfer_mac.convert.remote --out "$OUT" --workers 6 >> "$LOG" 2>&1 < /dev/null &
disown
sleep 2
echo "started pid $(pgrep -f deepinfer_mac.convert.remote | head -1); log: $LOG"
