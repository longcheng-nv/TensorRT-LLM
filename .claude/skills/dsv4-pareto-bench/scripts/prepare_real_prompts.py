#!/usr/bin/env python3
"""Convert real {system, user} chat JSONL into a trtllm-bench input JSONL.

Source format (one JSON object per line):
    {"system": "...", "user": "..."}

Output format (trtllm-bench-compatible, one JSON object per line):
    {"task_id": int, "prompt": str, "output_tokens": int}

Reference prompts (per-ISL buckets) are published at:
    https://github.com/longcheng-nv/GVR_TopK_supplementaty_materials/tree/main/longseqtasks

When --model-path points at a DSv4 checkpoint that ships an
`encoding/encoding_dsv4.py` module (Flash/Pro both do), the DSv4
chat template is applied via `encoding_dsv4.encode_messages` —
preserving thinking_mode / reasoning_effort tags. Otherwise falls
back to a plain "system\\n\\nuser" concatenation (trtllm-bench will
re-tokenise either way using --tokenizer at runtime).

If --num-prompts > len(input rows), the script cycles through the
input modulo its length so the output is exactly --num-prompts long
(matches `--num-requests` semantics in 01_run_one_cell.sh's
NUM_PROMPTS = BS * MULTI_ROUND).
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


def load_dsv4_encoder(model_path: str | None):
    if not model_path:
        return None
    p = Path(model_path) / "encoding" / "encoding_dsv4.py"
    if not p.exists():
        return None
    spec = importlib.util.spec_from_file_location("encoding_dsv4", p)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def render_prompt(encoder, system: str, user: str,
                  thinking_mode: str, reasoning_effort: str | None) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    if encoder is not None:
        return encoder.encode_messages(
            messages,
            thinking_mode=thinking_mode,
            reasoning_effort=reasoning_effort,
        )
    # Fallback when encoding_dsv4 is unavailable.
    parts = []
    if system:
        parts.append(system)
    parts.append(user)
    return "\n\n".join(parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--input", required=True, help="raw {system,user} JSONL")
    p.add_argument("--output", required=True, help="trtllm-bench JSONL output")
    p.add_argument("--num-prompts", type=int, required=True,
                   help="output row count; cycles input rows if larger")
    p.add_argument("--output-tokens", type=int, required=True,
                   help="output_tokens per row (= OSL in the sweep)")
    p.add_argument("--model-path", default=None,
                   help="optional; enables encoding_dsv4 chat template")
    p.add_argument("--thinking-mode", default="thinking",
                   choices=["chat", "thinking"])
    p.add_argument("--reasoning-effort", default="",
                   help="empty / 'high' / 'max'; only used with thinking_mode=thinking")
    args = p.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"input does not exist: {args.input}")

    with open(args.input) as f:
        rows = [json.loads(l) for l in f if l.strip()]
    if not rows:
        sys.exit(f"empty input: {args.input}")

    encoder = load_dsv4_encoder(args.model_path)
    if args.model_path and encoder is None:
        print(
            f"[warn] encoding_dsv4.py not found under "
            f"{args.model_path}/encoding; falling back to plain concat",
            file=sys.stderr,
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as out:
        for i in range(args.num_prompts):
            src = rows[i % len(rows)]
            prompt = render_prompt(
                encoder,
                src.get("system", ""),
                src.get("user", ""),
                args.thinking_mode,
                args.reasoning_effort or None,
            )
            out.write(json.dumps(
                {"task_id": i, "prompt": prompt, "output_tokens": args.output_tokens},
                ensure_ascii=False,
            ) + "\n")

    template_state = "dsv4 chat template" if encoder is not None else "plain concat"
    print(
        f"wrote {args.num_prompts} rows to {args.output} "
        f"(cycled from {len(rows)} source rows, template={template_state})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
