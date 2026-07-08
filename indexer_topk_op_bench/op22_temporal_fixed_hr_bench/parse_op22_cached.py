# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""op22rr — parse_op22.py with a per-rep kernel-sum cache.

Identical output to parse_op22.py, but each rep's parse_rep() result is
cached to <rep>.kern.json keyed by the rep's (mtime, size), so re-running
after new batches land only exports the NEW reps (nsys export of a bs rep
costs 30-60 s; the full 81-rep grid would otherwise re-export every time).

Usage: python3 parse_op22_cached.py [<out_root>]  (default ../results_b200_op22rr)
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "report"))
from parse_nsys_full import parse_rep as _parse_rep_raw  # noqa: E402

import parse_op22  # noqa: E402


def parse_rep_cached(rep):
    rep = Path(rep)
    if not rep.exists():
        return {}
    st = rep.stat()
    key = [int(st.st_mtime), st.st_size]
    cache = rep.with_suffix(rep.suffix + ".kern.json")
    if cache.exists():
        try:
            c = json.loads(cache.read_text())
            if c.get("key") == key:
                return c["kern"]
        except (json.JSONDecodeError, KeyError):
            pass
    kern = _parse_rep_raw(rep)
    if kern:  # never cache an empty parse (export may have raced a writer)
        cache.write_text(json.dumps({"key": key, "kern": kern}))
    return kern


def main():
    parse_op22.parse_rep = parse_rep_cached
    if len(sys.argv) < 2:
        sys.argv.append(str(HERE.parents[0] / "results_b200_op22rr"))
    parse_op22.main()


if __name__ == "__main__":
    main()
