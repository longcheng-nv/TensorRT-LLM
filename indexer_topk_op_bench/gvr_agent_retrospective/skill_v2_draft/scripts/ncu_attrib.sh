#!/usr/bin/env bash
# L3 ATTRIBUTION (OmniKernel measurement ladder): physical root-cause metrics.
# Use for attribution ONLY — never optimize to these numbers directly, and
# never quote an instrumented run as a performance baseline.
#
# Prints the raw metrics plus the two structural verdicts:
#   - L2-trap:   dram__bytes_read ~= input working set?  (if bytes << input,
#                the baseline is already L2-resident: traffic levers are void)
#   - occupancy: grid covers << SM count? (occupancy is structural; register/
#                pipelining levers are void — need data-parallel decomposition)
#
# Usage: bash scripts/ncu_attrib.sh <runner.py> [args...]
#        INPUT_BYTES=1048576 bash scripts/ncu_attrib.sh <runner.py>   # enables L2-trap verdict
#        KERNEL_REGEX=my_kernel bash scripts/ncu_attrib.sh <runner.py>
#
# The verdict block reads the FIRST profiled kernel's metrics. If the runner
# launches anything before the kernel of interest (input generation counts!),
# set KERNEL_REGEX so ncu -k profiles only the target (validated footgun: a
# randn setup kernel produced a false L2-TRAP verdict).
set -euo pipefail

RUNNER=${1:?usage: ncu_attrib.sh <runner.py> [args...]}; shift || true
KFLAGS=()
[ -n "${KERNEL_REGEX:-}" ] && KFLAGS=(-k "$KERNEL_REGEX")

METRICS=(
  gpu__time_duration.sum                                   # kernel time
  dram__bytes_read.sum                                     # L2-trap test
  dram__bytes_write.sum
  lts__t_bytes.sum                                         # L2 traffic
  sm__warps_active.avg.pct_of_peak_sustained_active        # achieved occupancy
  sm__throughput.avg.pct_of_peak_sustained_elapsed         # compute SOL% (diagnostic)
  gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
  launch__grid_size                                        # occupancy-structure test
  launch__registers_per_thread
)
IFS=,; MET="${METRICS[*]}"; unset IFS

# Token hygiene: profiler artifacts embed the process env.
CSV=/tmp/ncu_attrib_$$.csv
env -u GITHUB_TOKEN -u HF_TOKEN \
  ncu --metrics "$MET" --target-processes all --csv ${KFLAGS[@]+"${KFLAGS[@]}"} \
  python3 "$RUNNER" "$@" | tee "$CSV"

# NOTE: the csv path is passed as argv — `python3 -` takes its PROGRAM from
# stdin, so `<file` + heredoc silently starves csv.reader (validated dead-code
# failure: verdicts never printed). INPUT_BYTES defaulted for `set -u`.
python3 - "$CSV" "${INPUT_BYTES:-}" <<'EOF' || true
import csv, os, sys
input_bytes = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else 0
with open(sys.argv[1]) as f:
    rows = [r for r in csv.reader(f) if r]
hdr = next((r for r in rows if "Metric Name" in r), None)
if not hdr:
    sys.exit(0)
i_name, i_val = hdr.index("Metric Name"), hdr.index("Metric Value")
vals = {}
for r in rows:
    if len(r) > max(i_name, i_val) and r[i_name] not in ("Metric Name",):
        vals.setdefault(r[i_name], r[i_val])
def num(k):
    try: return float(vals.get(k, "nan").replace(",", ""))
    except ValueError: return float("nan")

print("\n=== STRUCTURAL VERDICTS ===")
dram = num("dram__bytes_read.sum")
if input_bytes:
    ratio = dram / input_bytes if input_bytes else float("nan")
    trap = "L2-TRAP: baseline already ~1 HBM pass -> traffic levers VOID" \
           if ratio < 1.5 else "no L2 trap (real DRAM re-reads present)"
    print(f"dram_read={dram:.3e}B vs input={input_bytes:.3e}B (ratio {ratio:.2f}) -> {trap}")
else:
    print(f"dram_read={dram:.3e}B (set INPUT_BYTES=<bytes> to enable the L2-trap verdict)")
grid = vals.get("launch__grid_size", "?")
occ = num("sm__warps_active.avg.pct_of_peak_sustained_active")
print(f"grid={grid}, achieved_occupancy={occ:.1f}% -> if grid << SM count, occupancy "
      f"is STRUCTURAL (register/pipeline levers void; go data-parallel)")
EOF
rm -f "$CSV"
