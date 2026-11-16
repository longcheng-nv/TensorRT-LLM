#!/usr/bin/env python3
"""Embed a decode-trace JSON into the timeline viewer.

usage: python3 embed.py DATA.json OUT.html [--template trace_viewer_template.html]

The template must contain the placeholder __TRACE_JSON__ exactly once, inside
<script id="trace-data" type="application/json">. The JSON is re-serialised compactly; every
"</" becomes "<\\/" and every "<!--" becomes "<\\u0021--" so the data can never close the script
element early or switch the HTML tokenizer into its escaped-script state. Both are still valid
JSON escapes, so JSON.parse in the page returns the original strings. NaN / Infinity (which
JSON.parse rejects) are written as null.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

PLACEHOLDER = "__TRACE_JSON__"
HERE = os.path.dirname(os.path.abspath(__file__))


def finite(obj):
    """Replace non-finite floats with None (JSON has no NaN / Infinity)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: finite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [finite(v) for v in obj]
    return obj


def script_safe_json(obj) -> str:
    text = json.dumps(finite(obj), ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # "<" can only occur inside JSON strings, so these replacements keep the JSON valid
    return text.replace("</", "<\\/").replace("<!--", "<\\u0021--")


def main() -> int:
    ap = argparse.ArgumentParser(description="Embed a trace JSON into the DSV4.1-Flash timeline viewer.")
    ap.add_argument("data", help="trace JSON (schema: runs[] with tokens / cpu / gpu / io / reqs)")
    ap.add_argument("out", help="output HTML file")
    ap.add_argument("--template", default=os.path.join(HERE, "trace_viewer_template.html"),
                    help="viewer template (default: trace_viewer_template.html next to this script)")
    a = ap.parse_args()

    with open(a.template, encoding="utf-8") as f:
        template = f.read()
    n = template.count(PLACEHOLDER)
    if n != 1:
        print(f"embed.py: {a.template} must contain {PLACEHOLDER} exactly once (found {n})", file=sys.stderr)
        return 2
    with open(a.data, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not isinstance(data.get("runs"), list):
        print("embed.py: warning: no 'runs' list in the data; the viewer will show an empty state", file=sys.stderr)

    html = template.replace(PLACEHOLDER, script_safe_json(data))
    out_dir = os.path.dirname(os.path.abspath(a.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)

    runs = data.get("runs") if isinstance(data, dict) else None
    desc = ", ".join(f"{r.get('id', '?')} {len(r.get('tokens') or [])} tokens"
                     for r in (runs or []) if isinstance(r, dict)) or "no runs"
    print(f"wrote {a.out} ({len(html.encode('utf-8')) / 1024:.0f} KB; {desc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
