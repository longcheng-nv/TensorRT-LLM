"""Mac-side oracle dump for gate_official_oracles.py.

Runs under the DeepInfer-Mac venv (mlx, tokenizers, deepseek_recipe).  Weights-free: it uses the
engine's real code paths that do not need a converted checkpoint -- `LLM.encode` (called unbound on a
stand-in object carrying the tokenizer), `build_chat` / `encode_conversation`, `Args.from_dict`, the
engram compressed-vocabulary builder, and the converter's `Converter.tasks()` (with the one network
read, inference/config.json, redirected to the local reference copy).

The golden-fixture comparison is a port of the prior audit's recipe_vs_goldens.py
(scratchpad/completeness-critic), kept here so the gate is self-contained.

    python _oracle_mac.py --repo <DeepInfer-Mac> --ref <ckpt-ref> --inputs inputs.json --out mac.json --workdir <dir>
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import difflib
import importlib.metadata as md
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace


def _version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "missing"


# ------------------------------------------------------------------ recipe helpers (from recipe_vs_goldens.py)


def add_tool_call_ids(messages):
    """The fixtures omit OpenAI's tool_call `id`; deepseek-recipe requires it.  Pair ids with the
    following tool messages' tool_call_id so the structure is unchanged.  Relative image paths get a
    syntactically valid URL (rendering never fetches)."""
    msgs = copy.deepcopy(messages)
    counter = 0
    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and m.get("tool_calls"):
            following = [t for t in msgs[i + 1:] if t.get("role") == "tool"][: len(m["tool_calls"])]
            for k, tc in enumerate(m["tool_calls"]):
                tid = following[k].get("tool_call_id") if k < len(following) else None
                if tid is None:
                    tid = f"call_{counter}"
                    counter += 1
                    if k < len(following):
                        following[k]["tool_call_id"] = tid
                tc.setdefault("id", tid)
        if m.get("role") == "user" and isinstance(m.get("content"), list):
            for part in m["content"]:
                u = (part.get("image_url") or {}).get("url") if isinstance(part, dict) else None
                if u and not u.startswith(("http", "data:")):
                    part["image_url"]["url"] = "https://example.invalid/" + u
    return msgs


GEN_HEADERS = ("<｜Assistant｜><think>", "<｜Assistant｜></think>")
EFFORT_HEADER_RE = re.compile(r"Reasoning Effort: \d+ \(range 1-100, the higher the value, the more thorough the reasoning\)\n\n")
ACTION_TOKEN = "<｜action｜>"
# encoding.py:481-506 TOOLS_TEMPLATE, as rendered after a system message ("\n\n" + render_tools(...))
TOOLS_BLOCK_RE = re.compile(r"\n\n## Tools\n\n.*?You MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls\.\n", re.S)


def strip_generation_header(prompt: str, golden: str) -> str:
    """Goldens are complete conversations ending in EOS; a request renderer appends the next
    generation header.  Compare the conversation body only."""
    for h in GEN_HEADERS:
        if prompt.endswith(h) and not golden.endswith(h):
            return prompt[: -len(h)]
    return prompt


def classify_diff(recipe: str, golden: str) -> str:
    """Name the smallest documented normalisation that makes recipe == golden, else 'other'."""
    if recipe == golden:
        return "none"
    r_no_effort = EFFORT_HEADER_RE.sub("", recipe, count=1)
    g_no_action = golden.replace(ACTION_TOKEN, "")
    if r_no_effort == golden:
        return "reasoning_effort_header_only"
    if recipe == g_no_action:
        return "action_token_only"
    if r_no_effort == g_no_action:
        return "reasoning_effort_header_and_action_token"
    r_blocks, g_blocks = TOOLS_BLOCK_RE.findall(recipe), TOOLS_BLOCK_RE.findall(golden)
    if len(g_blocks) == 1 and not r_blocks and recipe == TOOLS_BLOCK_RE.sub("", golden, count=1):
        return "tools_block_missing_only"
    if len(g_blocks) == 1 and len(r_blocks) == 1 and r_blocks == g_blocks and TOOLS_BLOCK_RE.sub("", recipe, count=1) == TOOLS_BLOCK_RE.sub("", golden, count=1):
        return "tools_block_position_only"
    return "other"


def unified(a: str, b: str, limit: int = 12) -> list[str]:
    lines = list(difflib.unified_diff(a.splitlines(keepends=True), b.splitlines(keepends=True), fromfile="golden", tofile="recipe", n=0))
    return [l.rstrip("\n") for l in lines[:limit]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="DeepInfer-Mac repo root")
    ap.add_argument("--ref", required=True, help="reference checkpoint dir")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True, help="scratch dir (the converter stand-in reads its progress.log here; nothing is written)")
    a = ap.parse_args()
    repo, ref = Path(a.repo).resolve(), Path(a.ref).resolve()
    sys.path.insert(0, str(repo))
    inputs = json.load(open(a.inputs))
    out: dict = {
        "python": sys.version.split()[0],
        "versions": {k: _version(k) for k in ("tokenizers", "deepseek-recipe", "mlx", "transformers", "numpy")},
        "errors": {},
    }

    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(ref / "tokenizer.json"))

    # ------------------------------------------------------------------ tokenizer + engine.encode (H1)
    try:
        from deepinfer_mac.engine import EOS_ID, IMAGE_PLACEHOLDER, IMAGE_TOKEN, LLM, SamplingParams, build_chat

        stand_in = SimpleNamespace(tokenizer=tok)  # LLM.encode only touches self.tokenizer (engine.py:383-385)

        def engine_encode(text: str) -> list[int]:
            return LLM.encode(stand_in, text)

        def replace_image_tokens(prompt: str) -> str:  # same normalisation the official examples apply
            for placeholder in inputs["candidate_image_placeholders"]:
                if placeholder != IMAGE_TOKEN:
                    prompt = prompt.replace(placeholder, IMAGE_TOKEN)
            return prompt

        t = {
            "class": "tokenizers.Tokenizer (tokenizer.json)",
            "vocab_size_with_added": tok.get_vocab_size(with_added_tokens=True),
            "vocab_size_base": tok.get_vocab_size(with_added_tokens=False),
            "post_processor": json.load(open(ref / "tokenizer.json")).get("post_processor", {}).get("type"),
            "EOS_ID": EOS_ID,
            "IMAGE_PLACEHOLDER": IMAGE_PLACEHOLDER,
            "IMAGE_TOKEN": IMAGE_TOKEN,
            "image_token_id_from_tokenizer": tok.token_to_id(IMAGE_TOKEN),
            "blacklist": list(SamplingParams().blacklist),
            "encode_empty": engine_encode(""),
            "specials": {s: tok.token_to_id(s) for s in inputs["special_tokens"]},
            "mm_span": {s: tok.token_to_id(s) for s in inputs["mm_span_tokens"]},
            "raw_prompts": [],
            "goldens": [],
        }
        for p in inputs["offline_prompts"]:
            q = replace_image_tokens(p)
            ids = engine_encode(q)
            t["raw_prompts"].append({
                "prompt_after_replace": q,
                "ids_engine_encode": ids,
                "ids_no_special": tok.encode(q, add_special_tokens=False).ids,
                "ids_tokenizers_default": tok.encode(q).ids,
                "n_image_tokens": ids.count(tok.token_to_id(IMAGE_TOKEN)),
            })
        for g in inputs["goldens"]:
            t["goldens"].append({
                "ids_no_special": tok.encode(g, add_special_tokens=False).ids,
                "ids_tokenizers_default": tok.encode(g).ids,  # ByteLevel post-processor adds nothing
                "ids_engine_encode": engine_encode(g),  # raw-text path: BOS prepended -> doubled on rendered text
            })
        out["tokenizer"] = t
    except Exception as e:  # noqa: BLE001
        out["errors"]["tokenizer"] = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ Args (H2)
    try:
        from deepinfer_mac.model.model import Args
        from deepinfer_mac.model.engram import NgramHasher, build_compressed_token_map

        inf_cfg = json.load(open(ref / "inference" / "config.json"))  # the converter copies this to inference_config.json (remote.py:257-260)
        args = Args.from_dict(inf_cfg)
        fields = Args.__dataclass_fields__
        _, compressed_vocab = build_compressed_token_map(str(ref / "tokenizer.json"))
        import inspect

        pad_default = inspect.signature(NgramHasher.__init__).parameters["pad_id"].default
        out["args"] = {
            "args": {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(args).items()},
            "inference_config": inf_cfg,
            "dropped_keys": sorted(k for k in inf_cfg if k not in fields),
            "defaulted_fields": sorted(k for k in fields if k not in inf_cfg),
            "derived": {
                "n_prefill_layer_from_ratios": list(args.compress_ratios).index(1),
                "last_kv_source": max(args.kv_source_layers),
                "kv_lora_rank": args.head_dim - args.rope_head_dim,
                "engram_compressed_vocab_from_tokenizer": compressed_vocab,
                "engram_pad_id_default": pad_default,
                "n_tables": args.engram_max_ngram_size - 1,
                "n_embed_per_ngram": args.engram_head_dim * args.engram_n_heads,
            },
        }
    except Exception as e:  # noqa: BLE001
        out["errors"]["args"] = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ converter task coverage (H3)
    try:
        from deepinfer_mac.convert import remote

        class LocalConverter(remote.Converter):
            def _small(self, path: str) -> bytes:  # the only network read in __init__: inference/config.json
                return (ref / path).read_bytes()

        conv = LocalConverter(Path(a.workdir) / "never-written", workers=1)
        idx = conv.idx
        tasks = conv.tasks()
        consumed: set[str] = set()
        problems: list[str] = []
        kind_counts: dict[str, int] = {}
        expert_re = re.compile(r"experts\.(\d+)\.(w\d\.(?:weight|scale))$")  # remote.py:232
        for tid, kind, payload in tasks:
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            if kind == "EG":  # remote.py:176-182 / 189 / 239-247: rows of embed.weight + embed.scale of that layer
                for suf in ("weight", "scale"):
                    name = f"layers.{payload['layer']}.engram.embed.{suf}"
                    if name not in idx:
                        problems.append(f"{tid}: {name} missing from index")
                    consumed.add(name)
                continue
            names = list(payload)
            consumed.update(names)
            if kind in ("XL", "XM"):
                for n in names:
                    if not expert_re.search(n):
                        problems.append(f"{tid}: {n} does not match the expert regex in Converter.write")
            elif kind in ("globals", "L", "M"):  # convert_dense (remote.py:79-97)
                for n in names:
                    m = payload[n]
                    if m["dtype"] == "F8_E4M3" and n[: -len(".weight")] + ".scale" not in payload:
                        problems.append(f"{tid}: fp8 weight {n} has no .scale in the same task")
                    if n.endswith(".scale") and n[: -len(".scale")] + ".weight" not in payload:
                        problems.append(f"{tid}: orphan scale {n} would be written verbatim")
            elif kind == "embed":
                if names != ["embed.weight"]:
                    problems.append(f"{tid}: unexpected payload {names}")
        ignored = sorted(set(idx) - consumed)
        hf_weight_map = json.load(open(ref / "model.safetensors.index.json"))["weight_map"]
        shard_mismatch = sorted(n for n in idx if hf_weight_map.get(n) != idx[n]["shard"])
        out["converter"] = {
            "n_shard_mismatch_vs_hf_index": len(shard_mismatch),
            "shard_mismatch_examples": shard_mismatch[:10],
            "n_index": len(idx),
            "n_tasks": len(tasks),
            "kind_counts": kind_counts,
            "n_consumed": len(consumed),
            "n_ignored": len(ignored),
            "ignored": ignored[:50],
            "ignored_non_scale": [n for n in ignored if not n.endswith(".scale")][:50],
            "problems": problems[:50],
            "engram_rows_in_index": {f"layers.{l}.engram.embed.weight": idx[f"layers.{l}.engram.embed.weight"]["shape"] for l in conv.inf["engram_layer_ids"]},
        }
    except Exception as e:  # noqa: BLE001
        out["errors"]["converter"] = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ recipe rendering (H4)
    try:
        from deepseek_recipe import ChatCompletionRequest, ConversionError, ConversionOptions, DeepseekV41Encoding, ModelVersion

        from deepinfer_mac.engine import IMAGE_PLACEHOLDER, IMAGE_TOKEN, build_chat

        def opts():
            return ConversionOptions().with_model_alias("deepseek-flash", ModelVersion.DeepseekV41Flash)

        def render(body: dict):
            conv = ChatCompletionRequest(json.dumps(body, ensure_ascii=False).encode()).convert(opts())
            r = DeepseekV41Encoding().render_conversation(conv.conversation)
            return conv, r.prompt.replace(IMAGE_PLACEHOLDER, IMAGE_TOKEN), len(r.image_sources)  # engine.py:191 respelling

        tests_dir = ref / "encoding" / "tests"
        golden_rows = []

        def compare_case(i: int, body: dict, golden: str, label: str) -> dict:
            row = {"case": i, "label": label}
            try:
                conv, prompt, n_img = render(body)
                stripped = strip_generation_header(prompt, golden)
                row.update(status="MATCH" if stripped == golden else "DIFF", diff_kind=classify_diff(stripped, golden), n_images=n_img,
                           golden_chars=len(golden), recipe_chars=len(prompt), recipe_tail=prompt[-32:],
                           thinking_stage=str(conv.parsing_options.thinking_initial_stage))
                if stripped != golden:
                    row["diff"] = unified(golden, stripped)
            except ConversionError as e:
                row.update(status="ConversionError", http_status=e.status_code, message=e.body[:300])
            except Exception as e:  # noqa: BLE001
                row.update(status="EXC", message=f"{type(e).__name__}: {str(e)[:300]}")
            return row

        for i in range(1, 6):
            fx = json.loads((tests_dir / f"test_input_{i}.json").read_text())
            golden = (tests_dir / f"test_output_{i}.txt").read_text()
            if isinstance(fx, dict):
                messages, tools, modes, effort = fx["messages"], fx.get("tools"), [fx.get("thinking_mode", "chat")], fx.get("reasoning_effort")
            else:
                messages, tools, modes, effort = fx, None, ["chat", "thinking"], None
            for mode in modes:
                body = {"model": "deepseek-flash", "messages": add_tool_call_ids(messages), "thinking": {"type": "enabled" if mode == "thinking" else "disabled"}}
                if tools:
                    body["tools"] = tools
                if effort is not None:
                    body["reasoning_effort"] = effort
                golden_rows.append(compare_case(i, body, golden, f"thinking_mode={mode}" + (f", effort={effort}" if effort else "")))
            if i == 3:  # probes: what remains once the rejected `tool_calls: []` is dropped?
                msgs = [{k: v for k, v in m.items() if not (k == "tool_calls" and v == [])} for m in messages]
                body = {"model": "deepseek-flash", "messages": add_tool_call_ids(msgs), "thinking": {"type": "disabled"}}
                golden_rows.append(compare_case(i, body, golden, "thinking_mode=chat, empty tool_calls removed (probe A)"))
                # the fixture attaches `tools` to the mid-conversation system message (per-message key, an encoding.py
                # extension); the OpenAI request schema only has request-level tools -> hoist them and retry
                per_msg_tools = next((m["tools"] for m in msgs if m.get("tools")), None)
                hoisted = [{k: v for k, v in m.items() if k != "tools"} for m in msgs]
                body = {"model": "deepseek-flash", "messages": add_tool_call_ids(hoisted), "thinking": {"type": "disabled"}, "tools": per_msg_tools}
                golden_rows.append(compare_case(i, body, golden, "thinking_mode=chat, empty tool_calls removed + per-message tools hoisted to request level (probe B)"))

        # the 5 official offline prompts as single-user-turn conversations, through the engine's own build_chat
        offline_rows = []
        for body in inputs["offline_bodies"]:
            row = {"index": body["index"], "renders": {}}
            for mode in ("chat", "thinking"):
                extra = {"reasoning_effort": inputs["reasoning_effort"]} if mode == "thinking" else {}
                rec: dict = {}
                try:
                    ids, images, conv = build_chat(ref, body["messages"], thinking=(mode == "thinking"), **extra)
                    prompt = DeepseekV41Encoding().render_conversation(conv.conversation).prompt.replace(IMAGE_PLACEHOLDER, IMAGE_TOKEN)
                    rec.update(prompt=prompt, ids=list(ids), n_images=len(images), chars=len(prompt),
                               ids_tokenizers_no_special=tok.encode(prompt, add_special_tokens=False).ids)
                except ConversionError as e:
                    rec.update(error=f"ConversionError {e.status_code}: {e.body[:300]}")
                except Exception as e:  # noqa: BLE001
                    rec.update(error=f"{type(e).__name__}: {str(e)[:300]}")
                row["renders"][mode] = rec
            offline_rows.append(row)

        # golden fixtures through the engine's chat path -> ids (fixtures the recipe accepts)
        chat_ids = []
        for i, mode in ((1, "thinking"), (2, "chat"), (5, "thinking")):
            fx = json.loads((tests_dir / f"test_input_{i}.json").read_text())
            golden = (tests_dir / f"test_output_{i}.txt").read_text()
            messages = fx["messages"] if isinstance(fx, dict) else fx
            extra = {}
            if isinstance(fx, dict):
                if fx.get("tools"):
                    extra["tools"] = fx["tools"]
                if fx.get("reasoning_effort") is not None:
                    extra["reasoning_effort"] = fx["reasoning_effort"]
            try:
                ids, images, conv = build_chat(ref, add_tool_call_ids(messages), thinking=(mode == "thinking"), **extra)
                prompt = DeepseekV41Encoding().render_conversation(conv.conversation).prompt.replace(IMAGE_PLACEHOLDER, IMAGE_TOKEN)
                # same rule as strip_generation_header: golden 5 is a current-turn prompt that already ends with the header
                header = next((h for h in GEN_HEADERS if prompt.endswith(h) and not golden.endswith(h)), "")
                header_ids = tok.encode(header, add_special_tokens=False).ids if header else []
                chat_ids.append({"case": i, "mode": mode, "ids": list(ids), "n_images": len(images), "header": header, "header_ids": header_ids,
                                 "ids_minus_header": list(ids)[: len(ids) - len(header_ids)] if header_ids else list(ids),
                                 "golden_ids_no_special": tok.encode(golden, add_special_tokens=False).ids})
            except Exception as e:  # noqa: BLE001
                chat_ids.append({"case": i, "mode": mode, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        out["recipe"] = {"goldens": golden_rows, "offline": offline_rows, "chat_ids": chat_ids}
    except Exception as e:  # noqa: BLE001
        out["errors"]["recipe"] = f"{type(e).__name__}: {e}"

    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[mac] wrote {a.out}; errors: {out['errors'] or 'none'}")


if __name__ == "__main__":
    main()
