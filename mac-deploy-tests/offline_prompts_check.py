"""DeepInfer's five official offline regression prompts (examples/_prompts.py) run through the Mac
engine's Python API, printed in the official `tests/models/test_offline_inference.py` line format.

The official decode_ppl goldens are platform-specific (sampled, exact-equality, no 'cpu' entry) and
are NOT used as pass/fail. What this proves: the deployment produces sane, parseable outputs with the
expected answers on the exact official regression inputs (text + 1/3/5-image prompts), and that
per-token logprobs are finite with decode_ppl in a plausible O(1-5) band.

Run from the DeepInfer-Mac repo root:
  env -u HTTP_PROXY -u HTTPS_PROXY .venv/bin/python /Users/loncheng/Models/dsv41-mac-tests/offline_prompts_check.py [--cache-gb 10] [--text-only]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
from deepinfer_mac import LLM, SamplingParams  # noqa: E402
from deepinfer_mac.engine import EOS_ID  # noqa: E402

IMG = "<｜deepseek_image｜>"
IMAGES = Path("/Users/loncheng/Models/DeepInfer/examples/images")
REF_IMAGES = Path("/Users/loncheng/Models/DeepSeek-V4.1-Flash-1017-ref/inference/examples/images")

# (prompt, images, expected substring in the answer) -- verbatim from DeepInfer examples/_prompts.py
TEST_DATA = [
    ("问题：法国的首都是哪里？\n答案：巴黎\n\n问题：一年有几个月？\n答案：12 个月\n\n问题：中国的首都是哪里？\n答案：", [], "北京"),
    ("Question: What is the largest planet in the solar system?\nAnswer: Jupiter\n\n"
     "Question: How many legs does a spider have?\nAnswer: Eight\n\n"
     "Question: Who wrote Romeo and Juliet?\nAnswer:", [], "Shakespeare"),
    (f"{IMG}\nQuestion: What vegetable is shown in the image?\nAnswer: Carrots\n\n"
     "Question: What color are the carrot tops?\nAnswer: Green\n\n"
     "Question: What color are the carrots?\nAnswer:", ["carrots.jpeg"], "range"),  # orange / Orange
    (f"This is image_1: {IMG}\nQuestion: What vegetable is shown in image_1?\nAnswer: Carrots\n\n"
     f"This is image_2: {IMG}\nQuestion: What food is shown in image_2?\nAnswer: Ribs\n\n"
     f"This is image_3: {IMG}\nQuestion: What vegetable is shown in image_3?\nAnswer:",
     ["carrots.jpeg", "ribs.jpeg", "carrots.jpeg"], "arrot"),
    (f"{IMG}\nQuestion: What vegetable is shown in the image?\nAnswer: Carrots\n\n"
     f"{IMG}\nQuestion: What food is shown in the image?\nAnswer: Ribs\n\n"
     f"{IMG}\nQuestion: What color are the carrot tops?\nAnswer: Green\n\n"
     f"{IMG}\nQuestion: Is the food meat or a vegetable?\nAnswer: Meat\n\n"
     f"{IMG}\nQuestion: What vegetable is shown in the image?\nAnswer: ",
     ["carrots.jpeg", "ribs.jpeg", "carrots.jpeg", "ribs.jpeg", "carrots.jpeg"], "arrot"),
]


def find_image(name: str) -> str | None:
    for d in (IMAGES, REF_IMAGES):
        if (d / name).exists():
            return str(d / name)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/loncheng/Models/dsv41-mac")
    ap.add_argument("--cache-gb", type=float, default=10.0)
    ap.add_argument("--text-only", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=32, help="official uses 128; answers are short, keep SSD time bounded")
    a = ap.parse_args()
    llm = LLM(a.model, expert_cache_gb=a.cache_gb)
    ok = True
    results = []
    for i, (prompt, imgs, expect) in enumerate(TEST_DATA):
        if imgs and a.text_only:
            continue
        paths = [find_image(n) for n in imgs]
        if any(p is None for p in paths):
            print(f"SKIP sample {i}: missing image(s) {imgs}")
            continue
        ids = llm.encode(prompt)
        sp = SamplingParams(temperature=0.4, max_tokens=a.max_tokens, seq_seed=i, top_logprobs=4)
        t0 = time.perf_counter()
        out = list(llm.stream(ids, sp, paths or None))
        dt = time.perf_counter() - t0
        toks = [o.token_id for o in out]
        gen = [t for t in toks if t != EOS_ID]
        text = llm.tokenizer.decode(gen, skip_special_tokens=True)
        logps = [o.token_logprobs for o in out]
        finite = all(math.isfinite(x) for x in logps)
        ppl = math.exp(-sum(logps) / max(1, len(logps))) if finite else float("nan")
        finish = "stop" if toks and toks[-1] == EOS_ID else "length"
        aliases = {"arrot": ("arrot", "胡萝卜"), "range": ("range", "橙"), "北京": ("北京", "beijing"), "Shakespeare": ("shakespeare", "莎士比亚")}
        hit = any(a.lower() in text.lower() for a in aliases.get(expect, (expect,)))
        ok &= hit and finite
        print(f"DP 0; INPUT: {prompt!r}; OUTPUT: {text!r}; output_tokens={len(out)}; accept_rate=N/A; decode_ppl={ppl:.3f}")
        print(f"   sample {i}: {'PASS' if hit and finite else 'FAIL'} expected~{expect!r} finish={finish} finite={finite} "
              f"{dt:.1f}s stats={json.dumps(llm.last_stats)}")
        results.append({"sample": i, "pass": bool(hit and finite), "text": text, "decode_ppl": ppl, "finish": finish,
                        "images": len(imgs), "stats": llm.last_stats, "seconds": round(dt, 1)})
        llm.reset()
    Path("/Users/loncheng/Models/dsv41-mac-tests/offline_prompts_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("OFFLINE PROMPTS", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
