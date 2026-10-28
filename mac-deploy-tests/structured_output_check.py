"""DeepInfer's official structured-output oracle (examples/_structured_output.py) run through the Mac
engine's Python API: regex, JSON-schema and EBNF constraints on adversarial prompts that beg for
free-form text, at temperature 0. The grammar must win, the final token must be EOS (finish 'stop').

Run from anywhere:
  env -u HTTP_PROXY -u HTTPS_PROXY /Users/loncheng/Models/DeepInfer-Mac/.venv/bin/python \
      /Users/loncheng/Models/dsv41-mac-tests/structured_output_check.py [--cache-gb 10] [--thinking]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
from deepinfer_mac import LLM, SamplingParams  # noqa: E402
from deepinfer_mac.engine import EOS_ID  # noqa: E402

THINK_END_ID = 128822

JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "color": {"type": "string", "enum": ["red", "green", "blue"]},
        "count": {"type": "integer", "enum": [1, 2, 3]},
    },
    "required": ["color", "count"],
    "additionalProperties": False,
}
CASES = [  # verbatim from DeepInfer examples/_structured_output.py
    ("一只会唱歌的冰箱飞到了月亮上。请随便续写一个很长的故事，不要写编号。", {"schema_type": "regex", "schema": r"[A-Z]{2}-[0-9]{3}"}),
    ("火星上的面条为什么会跳舞？请用自然语言胡思乱想，不要输出 JSON 或代码。", {"schema_type": "json", "schema": json.dumps(JSON_SCHEMA)}),
    ("请写一首关于紫色拖鞋和时间旅行的长诗，不要用英文，也不要给出简短答案。", {"schema_type": "ebnf", "schema": 'root ::= "answer=" ("yes" | "no") ";level=" [1-3]'}),
]


def check(kind: str, schema: str, text: str) -> tuple[bool, str]:
    if kind == "regex":
        return bool(re.fullmatch(schema, text)), "fullmatch"
    if kind == "json":
        try:
            v = json.loads(text)
        except Exception as e:  # noqa: BLE001
            return False, f"json.loads failed: {e}"
        ok = isinstance(v, dict) and set(v) == {"color", "count"} and v["color"] in {"red", "green", "blue"} and type(v["count"]) is int and v["count"] in {1, 2, 3}
        return ok, "schema"
    return bool(re.fullmatch(r"answer=(yes|no);level=[1-3]", text)), "ebnf-fullmatch"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/Users/loncheng/Models/dsv41-mac")
    ap.add_argument("--cache-gb", type=float, default=10.0)
    ap.add_argument("--thinking", action="store_true", help="also run each case in thinking mode (grammar triggers after </think>)")
    a = ap.parse_args()
    llm = LLM(a.model, expert_cache_gb=a.cache_gb)
    ok = True
    modes = [False, True] if a.thinking else [False]
    for thinking in modes:
        for prompt, so in CASES:
            kind = so["schema_type"]
            so_args = dict(so)
            ids, _ = llm.chat_prompt([{"role": "user", "content": prompt}], thinking=thinking, **({"reasoning_effort": "low"} if thinking else {}))
            if thinking:
                so_args["trigger_token_ids"] = [THINK_END_ID]
            import os
            think_max = int(os.environ.get("SO_THINK_MAX_TOKENS", "512"))
            sp = SamplingParams(temperature=0, max_tokens=think_max if thinking else 128, so_args=so_args)
            t0 = time.perf_counter()
            out = list(llm.stream(ids, sp))
            dt = time.perf_counter() - t0
            toks = [o.token_id for o in out]
            finished_stop = bool(toks) and toks[-1] == EOS_ID
            body = toks[:-1] if finished_stop else toks
            text_all = llm.tokenizer.decode(body, skip_special_tokens=False)
            text = text_all.split("</think>")[-1] if thinking else text_all
            good, how = check(kind, so["schema"], text)
            # the official oracle requires a final EOS; a grammar that completed but did not emit EOS is a failure
            passed = good and finished_stop
            ok &= passed
            print(f"{'PASS' if passed else 'FAIL'} [{'thinking' if thinking else 'chat'}] SO {kind}: {text!r} | {how}={good} final_eos={finished_stop} "
                  f"tokens={len(out)} {dt:.1f}s stats={json.dumps(llm.last_stats)}")
            llm.reset()
    print("STRUCTURED OUTPUT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
