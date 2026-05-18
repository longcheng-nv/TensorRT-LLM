#!/usr/bin/env python3
"""Generate plan.csv enumerating all cells in the DSv4 paired-feature sweep.

Axes (all read from env, with defaults that reproduce the proven B-flash-B300
recipe used in perf_logs/pareto_v4_flash_gvr):

  ISLS     "<isl>:<osl>[,<isl>:<osl>...]"  default "131072:4096,262144:4096,524288:4096"
  BSS      space- or comma-separated ints  default "1 2 4 8 16 32 64 128"
  MTPS     space- or comma-separated ints  default "0 1 2 3"     (0 disables MTP block)
  MODES    space- or comma-separated      default "TEP DEP"     (TEP=attn-DP on, DEP=attn-DP off)
  FEATURES space- or comma-separated 0/1  default "1 0"          (GVR enable_heuristic_topk)

Ordering: (ISL asc, BS asc, MTP asc, Mode {first listed}, Feature {first listed}).
Paired blocks run back-to-back for thermal/cache-state parity.

Re-running is idempotent — overwrites plan.csv with the same content.

PERFDIR override (env): default = parent of this script's dir.
"""

import csv
import os
import sys


def _parse_int_list(raw: str) -> list[int]:
    return [int(t) for t in raw.replace(",", " ").split()]


def _parse_isls(raw: str) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for item in raw.replace(",", " ").split():
        isl_str, _, osl_str = item.partition(":")
        if not osl_str:
            raise SystemExit(f"ISLS entry {item!r} must be 'ISL:OSL'")
        pairs.append((int(isl_str), int(osl_str)))
    return pairs


def _parse_mode_list(raw: str) -> list[str]:
    out = [t for t in raw.replace(",", " ").split()]
    for m in out:
        if m not in {"TEP", "DEP"}:
            raise SystemExit(f"Mode must be TEP or DEP, got {m!r}")
    return out


def _parse_bool_list(raw: str) -> list[int]:
    out: list[int] = []
    for t in raw.replace(",", " ").split():
        if t in {"1", "true", "True", "on", "ON"}:
            out.append(1)
        elif t in {"0", "false", "False", "off", "OFF"}:
            out.append(0)
        else:
            raise SystemExit(f"Feature flag must be 0/1/true/false, got {t!r}")
    return out


PERFDIR = os.environ.get(
    "PERFDIR",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
PLAN_PATH = os.path.join(PERFDIR, "plan.csv")

ISLS = _parse_isls(os.environ.get("ISLS", "131072:4096 262144:4096 524288:4096"))
BSS = _parse_int_list(os.environ.get("BSS", "1 2 4 8 16 32 64 128"))
MTPS = _parse_int_list(os.environ.get("MTPS", "0 1 2 3"))
MODES = _parse_mode_list(os.environ.get("MODES", "TEP DEP"))
FEATURES = _parse_bool_list(os.environ.get("FEATURES", "1 0"))


def isl_bucket(isl: int) -> str:
    return f"{isl // 1024}K"


def estimate_seconds(isl: int, bs: int, mtp: int, osl: int = 4096) -> int:
    """Per-cell wall-clock estimate. Used by 02_master.sh budget logic.

    Calibrated against perf_logs/long_isl: Flash ISL=64K OSL=512 BS=4 MTP=3 ~ 8 min.
    Hard-capped at 120 min so each cell fits a single 4 h SLURM job
    (master.sh requires est_sec * 1.5 < remaining; bench wraps the same timeout).
    """
    isl_term = 5 * (isl / 65536)
    osl_term = 5 * (osl / 512)
    base_min = max(isl_term + osl_term, 8)
    base_min *= 1.0 + 0.10 * mtp
    base_min *= 1.0 + 0.02 * max(0, bs - 4)
    base_min = max(5.0, min(base_min, 120.0))
    return int(base_min * 60)


def main() -> None:
    rows = []
    for isl, osl in ISLS:
        for bs in BSS:
            for mtp in MTPS:
                for mode in MODES:
                    for feat in FEATURES:
                        rows.append({
                            "ISL": isl,
                            "OSL": osl,
                            "BS": bs,
                            "MTP": mtp,
                            "Mode": mode,
                            "GVR": feat,
                            "est_sec": estimate_seconds(isl, bs, mtp, osl),
                        })

    for i, row in enumerate(rows):
        row["cell_id"] = (
            f"cell_{i:03d}_{isl_bucket(row['ISL'])}"
            f"_{row['BS']:03d}_M{row['MTP']}"
            f"_{row['Mode']}_GVR{'T' if row['GVR'] else 'F'}"
        )

    fieldnames = ["cell_id", "ISL", "OSL", "BS", "MTP", "Mode", "GVR", "est_sec"]
    # Force Unix line endings — bash `read` chokes on csv.writer's default \r\n.
    os.makedirs(os.path.dirname(PLAN_PATH) or ".", exist_ok=True)
    with open(PLAN_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: row[k] for k in fieldnames})

    print(f"Wrote {len(rows)} cells to {PLAN_PATH}", file=sys.stderr)
    print(f"  ISLS={ISLS}", file=sys.stderr)
    print(f"  BSS={BSS}  MTPS={MTPS}  MODES={MODES}  FEATURES={FEATURES}", file=sys.stderr)
    print(f"Total est. runtime: {sum(r['est_sec'] for r in rows) / 3600:.1f} h",
          file=sys.stderr)
    for isl, osl in ISLS:
        sub = [r for r in rows if r["ISL"] == isl]
        h = sum(r["est_sec"] for r in sub) / 3600
        print(f"  ISL={isl_bucket(isl)}: {len(sub)} cells, ~{h:.1f} h", file=sys.stderr)


if __name__ == "__main__":
    main()
