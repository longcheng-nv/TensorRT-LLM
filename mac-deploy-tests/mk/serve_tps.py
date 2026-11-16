"""End-to-end decode speed through the OpenAI-compatible server (what a local user sees): streams a
chat completion and measures tokens/s after the first token (time between first and last content
chunk). usage: serve_tps.py [--url http://127.0.0.1:8000] [--max-tokens 300] [--temperature 0]"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

PROMPTS = [
    "Write a long paragraph about the history of tea.",
    "Write a Python function that parses a CSV file and computes per-column statistics, with comments.",
    "用三段话解释量子计算的基本原理，并举一个实际应用的例子。",
    "Solve step by step: a train travels 120 km at 60 km/h and then 180 km at 90 km/h. What is its average speed?",
]


def one(url, prompt, max_tokens, temperature):
    body = {"model": "deepseek-flash", "messages": [{"role": "user", "content": prompt}], "stream": True,
            "max_tokens": max_tokens, "temperature": temperature, "thinking": {"type": "disabled"},
            "stream_options": {"include_usage": True}}
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = last = None
    n = 0
    usage = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for ch in d.get("choices", []):
                if ch.get("delta", {}).get("content"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    n += 1
    toks = usage["completion_tokens"] if usage else n
    return {"ttft_s": round(first - t0, 2), "tokens": toks, "decode_tok_s": round((toks - 1) / max(1e-9, last - first), 2)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.0)
    a = ap.parse_args()
    rs = []
    for p in PROMPTS:
        r = one(a.url, p, a.max_tokens, a.temperature)
        rs.append(r)
        print(json.dumps({"prompt": p[:30], **r}, ensure_ascii=False), flush=True)
    tot = sum(r["tokens"] - 1 for r in rs)
    print(f"mean decode tok/s over prompts: {sum(r['decode_tok_s'] for r in rs) / len(rs):.2f}")
