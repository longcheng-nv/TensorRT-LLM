#!/usr/bin/env python3
"""Build the agent-session timeline page: the decode viewer (trace_viewer_template.html) with the runs of
mac-deploy-tests/mk/trace_agent.py, retitled for prefill + decode segments.

usage: agent_page.py OUT.html RUN.json [RUN.json ...]
"""
from __future__ import annotations

import bisect
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from embed import PLACEHOLDER, script_safe_json  # noqa: E402

TEXT = [
    ("<title>DSV4.1-Flash 解码时间线</title>", "<title>DSV4.1 agent 流水线时间线</title>"),
    ('<h1 id="t-intro">单 token 解码时间线</h1>', '<h1 id="t-intro">agent 会话的推理流水线（prefill + 解码）</h1>'),
    ("每条泳道是一个执行者：CPU 主线程是 Python 解码循环里逐层的各个阶段，",
     "回放一段真实的 DeepSeek Harness 会话（服务端录下的 token 序列，按服务端的方式 prefill，再逐个喂回当时生成的 token）。"
     "左上角选「Token」可跳到某个请求的 prefill、首个输出 token 或前几个解码 token。每条泳道是一个执行者：CPU 主线程是 Python 里逐层的各个阶段"
     "（prefill 时每层按 16 个专家一批：预读下一批 → 等这一批读完 → 构建 → 等 GPU 算完），"),
    ("注意力块的结束取路由同步返回的时刻，MoE 块的时长是估计（斜线）。",
     "注意力块的结束取路由同步返回的时刻；prefill 的专家批以 eval 返回为结束（实测），解码的 MoE 块时长是估计（斜线）。"
     "prefill 段里同一 IO 线程上间隔不到 2 ms 的 2 MB 读块合并成一条占用条（层交界的空泡是几十到几百毫秒，照样看得见；每块的细节在 Perfetto JSON 里）。"),
    ("解码时间线 · <span", "流水线时间线 · <span"),
    ("<span class=\"num\">01</span>逐 token 汇总", "<span class=\"num\">01</span>逐段汇总（prefill / 首个 token / 解码）"),
]


def compact(run, gap=2.0):
    """Inside prefill segments, one SSD bar per IO thread for each stretch of back-to-back chunks (kind = the majority
    by bytes) and one bar per stretch of hot-wait blocks: a prefill holds ~10^5 chunk events, and what the page needs
    to show is each thread's occupancy and its gaps. Per-chunk detail stays in the Perfetto JSON."""
    pf = sorted((s["t0"], s["t1"]) for s in run["tokens"] if s.get("kind") == "prefill")
    starts = [w[0] for w in pf]

    def in_pf(t):
        j = bisect.bisect_right(starts, t) - 1
        return j >= 0 and t <= pf[j][1]

    def merge(events, key):
        out, last = [], {}
        for e in sorted(events, key=lambda e: (key(e), e["t0"])):
            m = last.get(key(e))
            if m is not None and e["t0"] - m["t1"] <= gap:
                m["t1"] = max(m["t1"], e["t1"])
                if "kb" in m:
                    m["kb"] += e["kb"]
                    m["_skb"] += e["kb"] if e["spec"] else 0
                    m["n"] = m.get("n", 1) + e.get("n", 1)
                    m["e"], m["req"] = "", ""
                continue
            m = dict(e)
            if "kb" in m:
                m["_skb"] = e["kb"] if e["spec"] else 0
            last[key(e)] = m
            out.append(m)
        for m in out:
            if "_skb" in m:
                m["spec"] = 1 if 2 * m.pop("_skb") > m["kb"] else 0
        return out

    io_pf = [e for e in run["io"] if in_pf(e["t0"])]
    run["io"] = sorted([e for e in run["io"] if not in_pf(e["t0"])] + merge(io_pf, lambda e: e["th"]), key=lambda e: e["t0"])
    hot = [e for e in run["gpu"] if e["k"] == "hot" and in_pf(e["t0"])]
    rest = [e for e in run["gpu"] if not (e["k"] == "hot" and in_pf(e["t0"]))]
    run["gpu"] = sorted(rest + merge(hot, lambda e: 0), key=lambda e: e["t0"])
    return run


def main() -> int:
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    out, paths = sys.argv[1], sys.argv[2:]
    runs = []
    for p in paths:
        runs += [compact(r) for r in json.load(open(p, encoding="utf-8"))["runs"]]
    html = open(os.path.join(HERE, "trace_viewer_template.html"), encoding="utf-8").read()
    for a, b in TEXT:
        if a not in html:
            print(f"agent_page.py: template text not found: {a[:40]}", file=sys.stderr)
            return 2
        html = html.replace(a, b, 1)
    html = html.replace(PLACEHOLDER, script_safe_json({"generated": time.strftime("%Y-%m-%d %H:%M"), "runs": runs}))
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {out} ({len(html.encode('utf-8')) / 1e6:.1f} MB; " + ", ".join(f"{r['id']} {len(r['tokens'])} segments" for r in runs) + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
