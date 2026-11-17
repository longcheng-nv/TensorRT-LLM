"""Combine several trace_decode.py runs into one viewer data file, renaming each run.

usage: combine_runs.py OUT.json ID LABEL TRACE.json [ID LABEL TRACE.json ...]
"""
import json
import sys
import time

args = sys.argv[1:]
if len(args) < 4 or (len(args) - 1) % 3:
    sys.exit(__doc__)
runs = []
for i in range(1, len(args), 3):
    rid, label, path = args[i:i + 3]
    r = json.load(open(path))
    r["id"], r["label"] = rid, label
    runs.append(r)
json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "runs": runs}, open(args[0], "w"), ensure_ascii=False)
print(f"{args[0]}: " + ", ".join(f"{r['id']} ({len(r['tokens'])} tokens)" for r in runs))
