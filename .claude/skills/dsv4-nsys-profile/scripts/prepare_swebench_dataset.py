#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
"""
Convert SWE-bench JSONL ({system, user} chat format) to trtllm-bench dataset
format ({task_id, input_ids, output_tokens}) using the target model's chat
template + tokenizer. Vendored from:
   CodeRepos/GVR_TopK_supplementaty_materials/realistic_dataset_revised_swebench_E2E_decode_GVR_topK/

Extensions over the upstream copy:
  --entry N         keep only entry N (0-indexed) from the input
  --num-replicate N replicate the kept entries N times (entry × N rows out)
                    NOTE: replication happens AFTER --entry, so the common
                    "single-prompt × BS" pattern is `--entry 2 --num-replicate ${BS}`.

Tokenizer note: pass the V4 MODEL_PATH directly — the script uses
`AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)` and
`apply_chat_template(...)`, so the resulting input_ids are exactly what
the V4 forward pass would see for the same prompt.

Prints max_isl to stdout for shell capture (`MAX_ISL=$(... | tail -1)`).
"""
import argparse
import json
import os
import sys

# DeepSeek-V4 checkpoints (Flash / Pro / *-Base) ship an empty chat_template
# in tokenizer_config.json — the V4 runtime applies the template externally.
# For offline tokenization we need an explicit template. Fall back to the
# DeepSeek-V3.2 family (same tokenizer vocab + special tokens, so the
# formatted output is byte-equivalent for the chat-style {system,user} prompts
# used by SWE-bench).
CHAT_TEMPLATE_FALLBACK_CANDIDATES = [
    "/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V3.2-Exp-FP4-v2",
    "/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V3.2-Exp-hf",
    "/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V3-0324",
    "/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V3",
]


def load_chat_template_from_dir(path):
    """Read non-empty chat_template from tokenizer_config.json at `path`."""
    cfg_path = os.path.join(path, "tokenizer_config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except Exception:
        return None
    tmpl = cfg.get("chat_template")
    if isinstance(tmpl, str) and tmpl.strip():
        return tmpl
    return None


def resolve_chat_template(tokenizer, explicit_source):
    """
    Returns a non-empty chat_template string OR None.
    Priority: explicit CLI arg → tokenizer.chat_template → fallback sibling dirs.
    """
    if explicit_source:
        if os.path.isdir(explicit_source):
            tmpl = load_chat_template_from_dir(explicit_source)
            if tmpl:
                print(f"[chat_template] from --chat-template-source dir: {explicit_source}",
                      file=sys.stderr)
                return tmpl
        elif os.path.isfile(explicit_source):
            with open(explicit_source) as f:
                tmpl = f.read()
            if tmpl.strip():
                print(f"[chat_template] from --chat-template-source file: {explicit_source}",
                      file=sys.stderr)
                return tmpl
        print(f"[chat_template] --chat-template-source={explicit_source} did not yield a template",
              file=sys.stderr)

    tmpl = getattr(tokenizer, "chat_template", None)
    if isinstance(tmpl, str) and tmpl.strip():
        print("[chat_template] using tokenizer.chat_template from MODEL_PATH",
              file=sys.stderr)
        return tmpl

    for cand in CHAT_TEMPLATE_FALLBACK_CANDIDATES:
        tmpl = load_chat_template_from_dir(cand)
        if tmpl:
            print(f"[chat_template] FALLBACK: tokenizer at MODEL_PATH has no template; "
                  f"borrowed from {cand}", file=sys.stderr)
            return tmpl
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Tokenize SWE-bench JSONL into trtllm-bench format.")
    parser.add_argument("--input", required=True,
                        help="Path to swe_bench_*.jsonl ({system, user} per line)")
    parser.add_argument("--tokenizer", required=True,
                        help="Path to tokenizer / model directory")
    parser.add_argument("--osl", type=int, default=2048,
                        help="output_tokens per record (default: 2048)")
    parser.add_argument("--output", default="dataset_swebench.json",
                        help="Output JSONL path")
    parser.add_argument("--entry", type=int, default=None,
                        help="Keep only entry N (0-indexed). Default: keep all.")
    parser.add_argument("--num-replicate", type=int, default=1,
                        help="Replicate the kept entries this many times. "
                             "Applied AFTER --entry. Default: 1 (no replication).")
    parser.add_argument("--chat-template-source", default=None,
                        help="Path to a model dir (containing tokenizer_config.json) "
                             "or a raw Jinja template file. Overrides the V3.2 "
                             "fallback list. Useful if you have a custom DeepSeek "
                             "template not in CHAT_TEMPLATE_FALLBACK_CANDIDATES.")
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer,
                                              trust_remote_code=True)

    chat_template = resolve_chat_template(tokenizer, args.chat_template_source)
    if not chat_template:
        print("ERROR: no usable chat_template found. Pass --chat-template-source "
              "or stage a sibling DeepSeek V3.x model dir in "
              "CHAT_TEMPLATE_FALLBACK_CANDIDATES.", file=sys.stderr)
        sys.exit(2)

    entries = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))

    print(f"Loaded {len(entries)} SWE-bench entries from {args.input}",
          file=sys.stderr)

    if args.entry is not None:
        if not (0 <= args.entry < len(entries)):
            print(f"ERROR: --entry {args.entry} out of range "
                  f"[0, {len(entries) - 1}]", file=sys.stderr)
            sys.exit(2)
        entries = [entries[args.entry]]
        print(f"Selected entry #{args.entry}", file=sys.stderr)

    rep = max(1, args.num_replicate)
    dataset = []
    task_id = 0
    for entry in entries:
        messages = []
        if entry.get("system"):
            messages.append({"role": "system", "content": entry["system"]})
        messages.append({"role": "user", "content": entry["user"]})
        # return_dict=False is required: when chat_template= is passed explicitly,
        # newer `transformers` returns a BatchEncoding by default, which breaks
        # downstream `len(input_ids)` and JSON serialization. We want the bare
        # List[int] form.
        input_ids = tokenizer.apply_chat_template(
            messages,
            chat_template=chat_template,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
            return_tensors=None)
        for _ in range(rep):
            dataset.append({
                "task_id": task_id,
                "input_ids": input_ids,
                "output_tokens": args.osl,
            })
            task_id += 1
        print(f"  Entry kept: {len(input_ids)} input tokens, "
              f"replicated ×{rep}, osl={args.osl}", file=sys.stderr)

    with open(args.output, "w") as f:
        for record in dataset:
            f.write(json.dumps(record) + "\n")

    isls = [len(r["input_ids"]) for r in dataset]
    max_isl = max(isls)
    print(f"\nGenerated {len(dataset)} requests -> {args.output}",
          file=sys.stderr)
    print(f"  ISL range: [{min(isls)}, {max_isl}]  OSL: {args.osl}",
          file=sys.stderr)

    print(max_isl)  # stdout: machine-readable for shell capture


if __name__ == "__main__":
    main()
