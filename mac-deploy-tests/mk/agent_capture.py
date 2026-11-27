"""Record a real DeepSeek Harness agent session on the local server, for trace_agent.py to replay.

Starts the same server as `python -m deepinfer_mac serve` (the CLI default --fast unless --lossless)
on 127.0.0.1:PORT and appends one JSON line per request to OUT.jsonl: the prompt token ids, the
generated token ids (with their times), the number of images and the engine's last_stats (reused
prefix, prefill seconds, expert GB read). Drive it with the real client -- `dsv41 -p "..." -p "..."`
finds a server already answering on port 8000 and uses it -- then stop it with Ctrl+C / SIGTERM.

usage: agent_capture.py OUT.jsonl [--cache-gb 8] [--port 8000] [--lossless]
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
MODEL = "/Users/loncheng/Models/dsv41-mac"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--cache-gb", type=float, default=8.0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--lossless", action="store_true")
    a = ap.parse_args()

    import uvicorn

    from deepinfer_mac.__main__ import _llm, build_parser
    from deepinfer_mac.server import create_app

    sa = build_parser().parse_args(["serve", "--model", MODEL, "--cache-gb", str(a.cache_gb), "--port", str(a.port)]
                                   + (["--lossless"] if a.lossless else []))
    llm = _llm(sa)
    inner = llm.stream
    lock = threading.Lock()

    def stream(prompt_token_ids, sp=None, images=None):
        rec = {"t_start": time.time(), "prompt": [int(t) for t in prompt_token_ids], "images": len(images or []),
               "max_tokens": getattr(sp, "max_tokens", None), "generated": [], "t_tokens": []}
        t0 = time.perf_counter()
        g = inner(prompt_token_ids, sp, images)
        try:
            for out in g:
                rec["generated"].append(int(out.token_id))
                rec["t_tokens"].append(round(time.perf_counter() - t0, 4))
                yield out
        finally:
            g.close()  # runs the engine's own finally (live state, last_stats) before we read them
            rec["t_end"] = round(time.perf_counter() - t0, 4)
            rec["stats"] = dict(llm.last_stats)
            with lock, open(a.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[capture] request: prompt {len(rec['prompt'])} (reused {rec['stats'].get('reused_tokens')}), "
                  f"generated {len(rec['generated'])}, first token {rec['t_tokens'][0] if rec['t_tokens'] else None} s, "
                  f"total {rec['t_end']} s", file=sys.stderr, flush=True)

    llm.stream = stream
    uvicorn.run(create_app(llm), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
