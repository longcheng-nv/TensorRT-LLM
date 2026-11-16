"""Convert Chrome / Perfetto trace-event JSON into the timeline viewer's data schema, so a trace can be
embedded in an HTML page (embed.py) instead of being opened in ui.perfetto.dev.

Written for the traces of mac-deploy-tests/mk/trace_decode.py (tid 1 = CPU main thread, tid 2 = GPU
(inferred), tid >= 10 = SSD IO threads, instant events "token N" mark token starts), but any trace with
complete ("X") events works: tid 1 / 2 map to the CPU / GPU lanes, every other tid becomes an IO lane,
and event names that are not known kinds render as "other" with the original name kept as the note.

What a trace-event file cannot carry is reconstructed or left out: read requests are rebuilt from their
chunks (submit time ~ first chunk start; promote / cancel times are unknown).

usage: perfetto_to_viewer.py ID LABEL TRACE.json [ID LABEL TRACE.json ...] OUT.json
"""
from __future__ import annotations

import json
import re
import sys
import time

CPU_KINDS = {"build", "route_build", "sync", "post", "launch_hit", "io_issue", "wait", "launch_miss", "tail", "other"}
GPU_KINDS = {"attn", "moe", "hot", "head"}
IO_NAME = re.compile(r"^(spec|demand) L(\d+) e(\d+)$")


def convert(run_id: str, label: str, path: str) -> dict:
    d = json.load(open(path))
    ev = d["traceEvents"] if isinstance(d, dict) else d
    unit = 1e-3  # trace-event timestamps are microseconds; the viewer uses milliseconds
    xs = [e for e in ev if e.get("ph") == "X" and "ts" in e]
    marks = sorted(e["ts"] for e in ev if e.get("ph") in ("i", "I") and str(e.get("name", "")).startswith("token"))
    t_base = marks[0] if marks else min((e["ts"] for e in xs), default=0)
    ms = lambda t: round((t - t_base) * unit, 4)  # noqa: E731
    io_tids = sorted({e.get("tid") for e in xs if e.get("tid") not in (1, 2)})
    lane = {tid: i for i, tid in enumerate(io_tids)}
    cpu, gpu, io, reqs = [], [], [], {}
    for e in xs:
        a = e.get("args", {}) or {}
        name = str(e.get("name", ""))
        t0, t1 = ms(e["ts"]), ms(e["ts"] + e.get("dur", 0))
        if e.get("tid") == 1:
            k = name if name in CPU_KINDS else "other"
            note = a.get("note") or ("" if k == name else name)
            cpu.append({"k": k, "l": a.get("layer", -1), "t0": t0, "t1": t1, **({"n": note} if note else {})})
        elif e.get("tid") == 2:
            est = name.endswith("(est)")
            k = name.replace("(est)", "").strip()
            note = a.get("note") or ("" if k in GPU_KINDS else name)
            gpu.append({"k": k if k in GPU_KINDS else "other", "l": a.get("layer", -1), "t0": t0, "t1": t1, "est": est,
                        **({"n": note} if note else {})})
        else:
            m = IO_NAME.match(name)
            spec, layer, expert = (int(m.group(1) == "spec"), int(m.group(2)), int(m.group(3))) if m else (0, -1, -1)
            req = a.get("req", -1)
            kb = a.get("kb", 0)
            io.append({"th": lane[e.get("tid")], "t0": t0, "t1": t1, "spec": spec, "req": req, "l": layer, "e": expert, "kb": kb,
                       **({} if m else {"n": name})})
            if req != -1:
                r = reqs.setdefault(req, {"req": req, "l": layer, "e": expert, "spec": spec, "t_sub": t0, "t_done": t1,
                                          "t_promote": None, "t_cancel": None, "kb": 0})
                r["t_sub"], r["t_done"], r["kb"] = min(r["t_sub"], t0), max(r["t_done"], t1), r["kb"] + kb
    end = max([x["t1"] for x in cpu + gpu + io] or [0])
    starts = [ms(t) for t in marks] or [0.0]
    tokens = [{"i": i, "t0": s, "t1": starts[i + 1] if i + 1 < len(starts) else end} for i, s in enumerate(starts)]
    return {
        "id": run_id, "label": label,
        "meta": {"io_threads": len(io_tids), "traced_tokens": len(tokens),
                 "ms_per_token": [round(t["t1"] - t["t0"], 2) for t in tokens],
                 "notes": [f"由 Perfetto / Chrome Trace JSON 转换（{path.split('/')[-1]}）：读请求由其数据块重建，提交时刻取首块开始，提升与取消时刻缺失。"]},
        "tokens": tokens, "cpu": cpu, "gpu": gpu, "io": io, "reqs": sorted(reqs.values(), key=lambda r: r["t_sub"]),
    }


def main():
    args = sys.argv[1:]
    if len(args) < 4 or (len(args) - 1) % 3:
        sys.exit(__doc__)
    runs = [convert(*args[i:i + 3]) for i in range(0, len(args) - 1, 3)]
    json.dump({"generated": time.strftime("%Y-%m-%d %H:%M"), "runs": runs}, open(args[-1], "w"), ensure_ascii=False)
    for r in runs:
        print(f"{r['id']}: {len(r['cpu'])} cpu / {len(r['gpu'])} gpu / {len(r['io'])} io / {len(r['reqs'])} requests, {len(r['tokens'])} tokens")


if __name__ == "__main__":
    main()
