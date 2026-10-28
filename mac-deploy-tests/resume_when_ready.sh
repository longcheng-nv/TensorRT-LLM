#!/bin/zsh
# Guardian: resume the DeepSeek-V4.1-Flash conversion only when the Mac is on AC power and on a
# real network (not an iPhone hotspot, direct Hugging Face throughput >= MIN_MBPS), then hand off to
# after_convert.sh, which waits for completion and launches the full verification suite.
# Re-checks every 2 minutes; if the converter dies mid-way (sleep, network loss) it is relaunched
# under the same conditions. Progress is in /Users/loncheng/Models/dsv41-mac/progress.log.
set -u
TESTS=/Users/loncheng/Models/dsv41-mac-tests
LOGS=/Users/loncheng/Models/logs
MODEL=/Users/loncheng/Models/dsv41-mac
MIN_MBPS=${MIN_MBPS:-5}
PROBE="https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/1bae0e1088993b9b82de08f59de5be5a34e42170/model-00021-of-00048.safetensors"
GLOG=$LOGS/resume-guardian.log
mkdir -p "$LOGS/verify"
log() { echo "$(date '+%F %T') $*" | tee -a "$GLOG"; }

ready() {
  pmset -g batt | grep -q "AC Power" || { log "waiting: on battery"; return 1; }
  local ip; ip=$(ipconfig getifaddr en0 2>/dev/null || true)
  case "$ip" in 172.20.10.*) log "waiting: iPhone hotspot ($ip)"; return 1;; "") log "waiting: en0 has no address"; return 1;; esac
  local bps; bps=$(env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy curl --noproxy '*' -sL -r 0-16777215 -o /dev/null -w '%{speed_download}' --max-time 40 "$PROBE" 2>/dev/null || echo 0)
  local mbps=$(( ${bps%.*} / 1000000 ))
  if [ "$mbps" -lt "$MIN_MBPS" ]; then log "waiting: HF direct ${mbps} MB/s < ${MIN_MBPS} MB/s (ip $ip)"; return 1; fi
  log "ready: AC power, ip $ip, HF direct ${mbps} MB/s"; return 0
}

complete() { grep -q "^EG-14-" "$MODEL/progress.log" 2>/dev/null && [ "$(wc -w < "$MODEL/progress.log" | tr -d ' ')" -ge 2748 ]; }

log "guardian started (pid $$); tasks done: $(wc -w < "$MODEL/progress.log" | tr -d ' ')/2748"
while ! complete; do
  if pgrep -f "deepinfer_mac.convert.remote" >/dev/null; then
    # running: if conditions degrade badly (battery or hotspot), pause to save data/battery
    if ! pmset -g batt | grep -q "AC Power" || ipconfig getifaddr en0 2>/dev/null | grep -q '^172\.20\.10\.'; then
      log "conditions degraded while running -> pausing converter"; pkill -TERM -f "deepinfer_mac.convert.remote"; sleep 5
    fi
    sleep 120; continue
  fi
  if ready; then
    "$TESTS/run_full_convert.sh" | tee -a "$GLOG"
    sleep 120
  else
    sleep 120
  fi
done
log "conversion complete ($(wc -w < "$MODEL/progress.log" | tr -d ' ') tasks); handing off to after_convert.sh"
"$TESTS/after_convert.sh" 2>&1 | tee -a "$GLOG"
touch "$LOGS/verify/verify-launched.marker"
log "guardian done"
