"""Prefill lookahead (MoE.PREFILL_LOOKAHEAD) on a prompt with an image: LLM._prefill_vl alternating off / on; times,
bytes read, and the runs grouped by bit-identical last logits. The prompt is the dsh system text of a recorded session
(agent_capture.py) plus the harness test image and a question, about the size of the image request in `dsv41 test`.

Prefill is not bit-reproducible run to run by itself: the routed experts are added into the rows with MLX scatter-add
(`y.at[r].add(o)`), which uses atomics, so a row with two experts in one batch sums in no fixed order (fp32 rounding,
amplified by the later layers). On this prompt the runs fall into groups that do not follow the lookahead setting --
the lookahead moves reads only. (A text prompt, test_prefill_lookahead.py, happened to give one group.)

usage: bench_prefill_vl.py [--cache-gb 8] [--rounds 2] [--k 16]
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
import time

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
MODEL = "/Users/loncheng/Models/dsv41-mac"
CAPTURE = "/Users/loncheng/Models/logs/mk/agent/capture-20261127.jsonl"
IMAGE = "/Users/loncheng/Models/DeepInfer-Mac/harness/testdata/carrots.jpeg"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-gb", type=float, default=8.0)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--k", type=int, default=16)
    a = ap.parse_args()

    from deepinfer_mac import LLM
    from deepinfer_mac.model import model as M

    llm = LLM(MODEL, expert_cache_gb=a.cache_gb)
    rec = json.loads(open(CAPTURE).readline())
    text = llm.tokenizer.decode(rec["prompt"], skip_special_tokens=False)
    system = text.split("<｜System｜>", 1)[1].split("<｜User｜>", 1)[0]
    url = "data:image/jpeg;base64," + base64.b64encode(open(IMAGE, "rb").read()).decode()
    msgs = [{"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": "图里是什么蔬菜？只回答名字。"}, {"type": "image_url", "image_url": {"url": url}}]}]
    ids, images, _ = llm.chat_inputs(msgs, thinking=False)
    print(f"prompt {len(ids)} tokens, {len(images)} image(s)", flush=True)
    store, runs = llm.weights.experts, []
    for _ in range(a.rounds):
        for k in (0, a.k):
            M.MoE.PREFILL_LOOKAHEAD = k
            llm.reset()
            b0, t = store.stats["bytes"], time.perf_counter()
            st, logits, _, _ = llm._prefill_vl(ids, images, False)
            lg = np.array(logits[-1])
            dt, gb = time.perf_counter() - t, (store.stats["bytes"] - b0) / 1e9
            runs.append((f"#{len(runs) + 1} k={k}", lg))
            print(f"#{len(runs)} lookahead {k:>3}: {dt:6.2f} s, {gb:6.1f} GB, argmax {int(lg.argmax())}", flush=True)
    groups = []
    for name, lg in runs:
        for g in groups:
            if np.array_equal(g[1], lg):
                g[0].append(name)
                break
        else:
            groups.append(([name], lg))
    print("bit-identical groups:", [g[0] for g in groups])
    if len(groups) > 1:
        x, y = groups[0][1], groups[1][1]
        print(f"between the first two groups: max |logit diff| {float(np.abs(x - y).max()):.3g}, same argmax {int(x.argmax()) == int(y.argmax())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
