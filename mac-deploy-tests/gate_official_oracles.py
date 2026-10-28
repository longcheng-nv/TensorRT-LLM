#!/usr/bin/env python3
"""Weights-free parity gates: DeepInfer-Mac (MLX port) vs the OFFICIAL deepseek-ai/DeepInfer oracles.

The official engine cannot run on a Mac, but its pure-Python contract code can: this runner drives
it (in a transformers-4.50 venv) and the Mac port (in its own venv) over identical inputs, dumps
JSON from each side, and compares.  Nothing downloads, nothing is written into either repo.

  H1 tokenizer-policy parity   official get_hf_tokenizer ids vs the Mac tokenizer / engine.encode
  H2 config parity             official load_hf_model_config(...).gpt vs the Mac Args
  H3 weight coverage           hf_to_model over the HF index, index == converter tensor_index, converter task coverage
  H4 prompt-rendering parity   deepseek_recipe vs the checkpoint's encoding.py (goldens + official offline prompts)

Exit status 0 only if H1-H4 PASS (with the documented, asserted exceptions in H4).

Re-run:
  python3 /Users/loncheng/Models/dsv41-mac-tests/gate_official_oracles.py \
      --workdir /private/tmp/claude-502/-Users-loncheng-Models/60697234-ef74-4285-9fb6-da2f8011ce9b/scratchpad/oracle-gate

The official-side venv is a scratch venv from the audit; if it is gone, recreate it with
  python3.12 -m venv <dir> && <dir>/bin/pip install transformers==4.50.0 tokenizers==0.21.4 \
      huggingface_hub==0.36.2 safetensors==0.8.0 torch==2.13.0 einops numpy psutil pyzmq posix_ipc orjson pillow
and pass --official-python <dir>/bin/python.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULTS = {
    "official_repo": "/Users/loncheng/Models/DeepInfer",
    "mac_repo": "/Users/loncheng/Models/DeepInfer-Mac",
    "ref": "/Users/loncheng/Models/DeepSeek-V4.1-Flash-1017-ref",
    "official_python": "/private/tmp/claude-502/-Users-loncheng-Models/60697234-ef74-4285-9fb6-da2f8011ce9b/scratchpad/official-deepinfer/.venv-torch/bin/python",
    "mac_python": "/Users/loncheng/Models/DeepInfer-Mac/.venv/bin/python",
}
OFFICIAL_MODEL_NAME = "deepseek-ai/DeepSeek-V4.1-Flash-1017"  # examples/_config.py DEFAULT_MODEL_NAME; resolved via DEEPINFER_HF_MODEL_DIR
REASONING_EFFORT = "high"
IMAGE_MARKERS = ("<｜deepseek_image｜>", "<image>")  # spellings used by examples/_prompts.py offline prompts
SPECIAL_TOKENS = [
    "<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>", "<｜▁pad▁｜>",
    "<｜User｜>", "<｜Assistant｜>", "<｜System｜>", "<think>", "</think>", "｜DSML｜",
    "<｜latest_reminder｜>", "<｜action｜>", "<｜query｜>", "<｜authority｜>", "<｜domain｜>", "<｜title｜>", "<｜read_url｜>",
    "<｜deepseek_image｜>", "<｜rl_image_pad｜>", "<｜rl_image_start｜>",
    "<｜ref｜>", "<｜/ref｜>", "<｜box｜>", "<｜/box｜>", "<｜point｜>", "<｜/point｜>", "<｜polygon｜>", "<｜/polygon｜>",
    "<image>",  # must be absent on both sides (otherwise the official candidate resolution would pick it first)
]
MM_SPAN_TOKENS = [f"<|place_holder_mm_span_{i:04d}|>" for i in range(21)]


# ------------------------------------------------------------------ reporting


class Gate:
    def __init__(self, name: str, title: str):
        self.name, self.title = name, title
        self.fails: list[str] = []
        self.n = 0
        print(f"\n=== {name}: {title} ===")

    def check(self, ok: bool, label: str, detail: str = "") -> bool:
        self.n += 1
        print(f"  [{'ok' if ok else 'FAIL'}] {label}" + (f": {detail}" if detail else ""))
        if not ok:
            self.fails.append(label)
        return ok

    def info(self, label: str, detail: str = "") -> None:
        print(f"  [info] {label}" + (f": {detail}" if detail else ""))

    def result(self) -> bool:
        ok = not self.fails
        print(f"{self.name} {self.title}: {'PASS' if ok else 'FAIL'} ({self.n - len(self.fails)}/{self.n} checks)")
        return ok


def short(ids, k: int = 8) -> str:
    if ids is None:
        return "None"
    return f"len={len(ids)} {list(ids[:k])}{'...' if len(ids) > k else ''}"


def norm(v):
    """Structural equality helper: lists/tuples -> tuples, dicts -> sorted tuples; ints == floats."""
    if isinstance(v, dict):
        return tuple(sorted((str(k), norm(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(norm(x) for x in v)
    return v


# ------------------------------------------------------------------ inputs


def extract_offline_prompts(prompts_py: Path) -> list[str]:
    """examples/_prompts.py keeps the 5 offline prompts in a local `test_data` literal; read it with ast."""
    tree = ast.parse(prompts_py.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "prepare_offline_prompts":
            for stmt in ast.walk(node):
                if isinstance(stmt, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "test_data" for t in stmt.targets):
                    data = ast.literal_eval(stmt.value)
                    return [p for p, _images in data]
    raise RuntimeError(f"could not find prepare_offline_prompts.test_data in {prompts_py}")


def prompt_to_openai_content(prompt: str):
    """Split a raw offline prompt at its image markers into OpenAI content parts (identical input for
    both renderers).  Text-only prompts stay plain strings."""
    pattern = "(" + "|".join(map(re.escape, IMAGE_MARKERS)) + ")"
    blocks, k = [], 0
    for part in re.split(pattern, prompt):
        if part in IMAGE_MARKERS:
            blocks.append({"type": "image_url", "image_url": {"url": f"https://example.invalid/image_{k}.jpeg"}})
            k += 1
        elif part:
            blocks.append({"type": "text", "text": part})
    return blocks if k else prompt


def build_inputs(args) -> dict:
    ref = Path(args.ref)
    prompts = extract_offline_prompts(Path(args.official_repo) / "examples" / "_prompts.py")
    goldens = [(ref / "encoding" / "tests" / f"test_output_{i}.txt").read_text() for i in range(1, 6)]
    bodies = [{"index": i, "messages": [{"role": "user", "content": prompt_to_openai_content(p)}]} for i, p in enumerate(prompts)]
    return {
        "official_model_name": OFFICIAL_MODEL_NAME,
        "reasoning_effort": REASONING_EFFORT,
        "special_tokens": SPECIAL_TOKENS,
        "mm_span_tokens": MM_SPAN_TOKENS,
        "candidate_image_placeholders": list(IMAGE_MARKERS[::-1]),  # ("<image>", "<｜deepseek_image｜>") as deepinfer.multimodal_tokens
        "offline_prompts": prompts,
        "goldens": goldens,
        "offline_bodies": bodies,
    }


# ------------------------------------------------------------------ subprocess drivers


def run_side(label: str, python: str, script: Path, cwd: str, extra: list[str], env: dict, log: Path) -> int:
    cmd = [python, "-u", str(script), *extra]
    t0 = time.time()
    with open(log, "w") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=900).returncode
    print(f"[{label}] exit {rc} in {time.time() - t0:.1f}s (log: {log})")
    if rc != 0:
        print(log.read_text()[-3000:])
    return rc


# ------------------------------------------------------------------ gates


def gate_h1(O: dict, M: dict, inputs: dict) -> bool:
    g = Gate("H1", "tokenizer-policy parity")
    ot, mt = O.get("tokenizer"), M.get("tokenizer")
    if not g.check(ot is not None and mt is not None, "both tokenizer dumps present", f"official_err={O['errors'].get('tokenizer')} mac_err={M['errors'].get('tokenizer')}"):
        return g.result()
    g.info("official", f"{ot['class']} len={ot['len']} add_bos_token={ot['add_bos_token']} add_eos_token={ot['add_eos_token']} "
                       f"(transformers {O['versions']['transformers']}, tokenizers {O['versions']['tokenizers']})")
    g.info("mac", f"{mt['class']} vocab={mt['vocab_size_with_added']} (base {mt['vocab_size_base']}) post_processor={mt['post_processor']} "
                  f"(tokenizers {M['versions']['tokenizers']}, deepseek-recipe {M['versions']['deepseek-recipe']})")
    # special ids
    g.check(ot["bos"] == ["<｜begin▁of▁sentence｜>", 0] and mt["specials"]["<｜begin▁of▁sentence｜>"] == 0, "BOS", f"official={ot['bos']} mac_id={mt['specials']['<｜begin▁of▁sentence｜>']}")
    g.check(ot["eos"] == ["<｜end▁of▁sentence｜>", 1] and mt["specials"]["<｜end▁of▁sentence｜>"] == 1 and mt["EOS_ID"] == 1, "EOS",
            f"official={ot['eos']} mac_id={mt['specials']['<｜end▁of▁sentence｜>']} engine.EOS_ID={mt['EOS_ID']}")
    g.check(ot["pad"] == ["<｜▁pad▁｜>", 2] and mt["specials"]["<｜▁pad▁｜>"] == 2 and 2 in mt["blacklist"], "PAD",
            f"official={ot['pad']} mac_id={mt['specials']['<｜▁pad▁｜>']} (in engine sampling blacklist {mt['blacklist']})")
    mism = {s: (ot["specials"][s], mt["specials"][s]) for s in inputs["special_tokens"] if ot["specials"][s] != mt["specials"][s]}
    present = {s: ot["specials"][s] for s in inputs["special_tokens"] if ot["specials"][s] is not None}
    g.check(not mism, f"{len(inputs['special_tokens'])} special-token strings -> same ids (or absent on both)",
            f"mismatches={mism or 'none'}; e.g. <think>={present.get('<think>')} </think>={present.get('</think>')} <｜User｜>={present.get('<｜User｜>')} "
            f"<｜Assistant｜>={present.get('<｜Assistant｜>')} <｜System｜>={present.get('<｜System｜>')} ｜DSML｜={present.get('｜DSML｜')} <｜action｜>={present.get('<｜action｜>')}; "
            f"'<image>' absent on both={ot['specials']['<image>'] is None and mt['specials']['<image>'] is None}")
    # image token
    img_ok = ot["image_token_id_config"] == ot["image_token_id_resolved"] == mt["image_token_id_from_tokenizer"] == M["args"]["args"]["image_token_id"] == 129264 and ot["image_token_str"] == mt["IMAGE_TOKEN"]
    g.check(img_ok, "image token", f"config.json={ot['image_token_id_config']} official-resolved({ot['candidate_image_placeholders']})={ot['image_token_id_resolved']} "
                                   f"spelling={ot['image_token_str']!r}; mac tokenizer={mt['image_token_id_from_tokenizer']} Args.image_token_id={M['args']['args']['image_token_id']} engine.IMAGE_TOKEN={mt['IMAGE_TOKEN']!r} "
                                   f"(recipe renders {mt['IMAGE_PLACEHOLDER']!r}, respelled by engine.py:191)")
    # mm-span tokens
    off_span = [ot["mm_span"][s] for s in inputs["mm_span_tokens"]]
    g.check(off_span == list(range(129280, 129301)) and ot["len"] == 129301, "official mm-span registration", f"ids={off_span[0]}..{off_span[-1]} len(tokenizer)={ot['len']}")
    first_free = mt["vocab_size_with_added"]
    g.check(all(v is None for v in mt["mm_span"].values()) and first_free == 129280 == ot["vocab_size_config"] == off_span[0],
            "mm-span layout parity", f"mac tokenizer.json has no mm-span tokens; its first free id {first_free} == vocab_size {ot['vocab_size_config']} == official first mm-span id {off_span[0]}; "
                                     f"official internal image ids vocab_size+ImageTokenOffset={ot['image_token_offsets']} (Mac spans use image_token_id + token types, vision.py:9)")
    g.check(ot["encode_empty"] == [0] == mt["encode_empty"], "encode('')", f"official={ot['encode_empty']} mac engine.encode={mt['encode_empty']}")
    # raw offline prompts: official tokenizer.encode(prompt) (add_bos_token=True) vs Mac engine.encode ([0] + no-special ids)
    for i, (o, m) in enumerate(zip(ot["raw_prompts"], mt["raw_prompts"])):
        same = o["ids_default"] == m["ids_engine_encode"]
        one_bos = m["ids_engine_encode"][:1] == [0] and m["ids_engine_encode"].count(0) == 1 and o["ids_default"].count(0) == 1
        g.check(same and one_bos and o["n_image_tokens"] == m["n_image_tokens"] and o["prompt_after_replace"] == m["prompt_after_replace"],
                f"offline prompt {i}: official encode(prompt) == mac engine.encode(prompt)",
                f"{short(o['ids_default'])} vs {short(m['ids_engine_encode'])}; image tokens {o['n_image_tokens']}/{m['n_image_tokens']}; single BOS={one_bos}")
    # rendered golden prompts: official add_special_tokens=False (eval/run_eval.py:222) vs Mac tokenizer; both default paths double BOS identically
    for i, (o, m) in enumerate(zip(ot["goldens"], mt["goldens"]), start=1):
        same = o["ids_no_special"] == m["ids_no_special"]
        g.check(same and m["ids_no_special"][:1] == [0] and m["ids_no_special"].count(0) == 1,
                f"golden {i}: official encode(text, add_special_tokens=False) == mac tokenizer ids", f"{short(o['ids_no_special'])} vs {short(m['ids_no_special'])}")
        g.check(o["ids_default"] == m["ids_engine_encode"] and o["ids_default"][:2] == [0, 0],
                f"golden {i}: official encode(text) [add_bos] == mac engine.encode(text) (both double the literal BOS -> raw-text path is NOT for rendered prompts)",
                f"{short(o['ids_default'], 4)} vs {short(m['ids_engine_encode'], 4)}")
    # the engine's rendered-prompt path (deepseek_recipe) must carry exactly one BOS and agree with the goldens
    for row in M["recipe"]["chat_ids"]:
        if "error" in row:
            g.check(False, f"golden {row['case']} via engine.chat path", row["error"])
            continue
        ids = row["ids"]
        g.check(ids[:1] == [0] and ids.count(0) == 1 and row["ids_minus_header"] == row["golden_ids_no_special"],
                f"golden {row['case']} via engine build_chat ({row['mode']}): exactly one BOS, ids == golden ids + generation header",
                f"{short(ids)} header={row['header']!r}->{row['header_ids']} golden {short(row['golden_ids_no_special'])}")
    g.info("BOS reconciliation", "official: tokenizer_config add_bos_token=false is overridden to True (tokenizer.py:59-61) so raw prompts get BOS via encode(); "
                                 "rendered prompts already start with the BOS string and are encoded with add_special_tokens=False (eval/run_eval.py:222, inference/generate.py:156 "
                                 "with the stock add_bos_token=false). Mac: engine.encode prepends [0] to a no-special encode (engine.py:385) == official raw path; "
                                 "rendered prompts go through deepseek_recipe.with_tokenizer().encode (engine.py:190) whose only BOS is the literal one == official rendered path.")
    return g.result()


def gate_h2(O: dict, M: dict) -> bool:
    g = Gate("H2", "config parity (official load_hf_model_config(...).gpt vs Mac Args)")
    oc, ma = O.get("config"), M.get("args")
    if not g.check(oc is not None and ma is not None, "both config dumps present", f"official_err={O['errors'].get('config')} mac_err={M['errors'].get('args')}"):
        return g.result()
    gpt, A, inf, D = oc["gpt"], ma["args"], ma["inference_config"], ma["derived"]
    hf, aml = oc["hf_raw"], oc["attn_module_list"]
    text = hf["text_config"]
    g.info("sources", "official gpt <- config.json (HF); Mac Args <- inference/config.json (copied to inference_config.json by the converter, remote.py:257) via Args.from_dict (model.py:84)")
    g.info("Mac Args.from_dict drops inference-config keys", f"{ma['dropped_keys']} (consumed elsewhere or hardcoded; asserted below); defaulted Args fields={ma['defaulted_fields'] or 'none'}")

    def cmp(label, official, mac, note=""):
        g.check(norm(official) == norm(mac), label, f"official={official!r} mac={mac!r}" + (f"  [{note}]" if note else ""))

    cmp("num_hidden_layers / n_layers", gpt["num_hidden_layers"], A["n_layers"])
    cmp("n_prefill_layer / first ratio-1 layer == last KV source (model.py:937)", gpt["n_prefill_layer"], [D["n_prefill_layer_from_ratios"], D["last_kv_source"]][0], f"mac last_kv_source={D['last_kv_source']}")
    g.check(D["n_prefill_layer_from_ratios"] == D["last_kv_source"] == gpt["n_prefill_layer"], "prefill boundary agrees three ways", f"ratios->{D['n_prefill_layer_from_ratios']} kv_source->{D['last_kv_source']} official->{gpt['n_prefill_layer']}")
    cmp("kv_lora_rank / head_dim - rope_head_dim", gpt["kv_lora_rank"], D["kv_lora_rank"], f"mac head_dim={A['head_dim']} rope_head_dim={A['rope_head_dim']} (official: head_dim - qk_rope_head_dim, config.py:289)")
    cmp("window_size", gpt["window_size"], [A["window_size"], 0], "official (left, right); Mac SWA window is left-only")
    cmp("dflash_cfg.sliding_window / window_size", gpt["dflash_cfg"]["sliding_window"], A["window_size"])
    cmp("vocab_size", gpt["vocab_size"], A["vocab_size"])
    cmp("hidden_size / dim", gpt["hidden_size"], A["dim"])
    cmp("num_attention_heads / n_heads", gpt["num_attention_heads"], A["n_heads"])
    cmp("head_dim", gpt["head_dim"], A["head_dim"])
    cmp("qk_rope_head_dim / rope_head_dim", gpt["qk_rope_head_dim"], A["rope_head_dim"])
    cmp("q_lora_rank", gpt["q_lora_rank"], A["q_lora_rank"])
    cmp("o_lora_rank", gpt["o_lora_rank"], A["o_lora_rank"])
    cmp("o_groups", gpt["o_groups"], A["o_groups"])
    cmp("rms_norm_eps / norm_eps", gpt["rms_norm_eps"], A["norm_eps"])
    cmp("swiglu_limit", gpt["swiglu_limit"], A["swiglu_limit"])
    cmp("rope_theta", gpt["rope_theta"], A["rope_theta"])
    cmp("hc_mult", gpt["hc_mult"], A["hc_mult"])
    cmp("hc_sinkhorn_iters", gpt["hc_sinkhorn_iters"], A["hc_sinkhorn_iters"])
    cmp("hc_eps", gpt["hc_eps"], A["hc_eps"])
    cmp("num_nextn_predict_layers / n_mtp_layers", gpt["num_nextn_predict_layers"], A["n_mtp_layers"])
    cmp("tie_word_embeddings == False / separate embed + head tensors", gpt["tie_word_embeddings"], False, "HF index has embed.weight and head.weight; converter writes embed.bf16 + head (remote.py:92,218)")
    # MoE
    moe = gpt["moe"]
    cmp("moe.n_routed_experts", moe["n_routed_experts"], A["n_routed_experts"])
    cmp("moe.num_experts_per_tok / n_activated_experts", moe["num_experts_per_tok"], A["n_activated_experts"])
    cmp("moe.moe_intermediate_size / moe_inter_dim", moe["moe_intermediate_size"], A["moe_inter_dim"])
    cmp("moe.routed_scaling_factor / route_scale", moe["routed_scaling_factor"], A["route_scale"])
    cmp("moe.n_shared_experts / inference n_shared_experts (Mac hardcodes one shared expert, model.py:590)", moe["n_shared_experts"], [inf.get("n_shared_experts"), 1][0], "mac hardcoded=1")
    g.check(moe["n_shared_experts"] == 1, "moe.n_shared_experts == 1 (Mac assumption)", f"official={moe['n_shared_experts']}")
    cmp("moe.scoring_func / inference score_func (Mac hardcodes sqrt(softplus), model.py:594)", moe["scoring_func"], inf.get("score_func"))
    g.check(moe["scoring_func"] == "sqrtsoftplus", "moe.scoring_func == sqrtsoftplus (Mac assumption)", f"official={moe['scoring_func']}")
    g.check(moe["topk_method"] == "noaux_tc" and moe["norm_topk_prob"] is True, "moe.topk_method noaux_tc + norm_topk_prob (Mac: bias-corrected pick, raw-score weights renormalised, model.py:595-598)",
            f"official topk_method={moe['topk_method']} norm_topk_prob={moe['norm_topk_prob']}")
    # engram
    eg = gpt["engram"]
    cmp("engram.engram_layer_ids", eg["engram_layer_ids"], A["engram_layer_ids"])
    cmp("engram.engram_max_ngram_size", eg["engram_max_ngram_size"], A["engram_max_ngram_size"])
    cmp("engram.n_head_per_ngram (per table)", eg["n_head_per_ngram"], [A["engram_n_heads"]] * D["n_tables"])
    cmp("engram.n_embed_per_ngram (per table) / engram_head_dim*engram_n_heads", eg["n_embed_per_ngram"], [D["n_embed_per_ngram"]] * D["n_tables"])
    cmp("engram.vocab_size (per table) / inference engram_vocab_size (engine.py:238)", eg["vocab_size"], [inf.get("engram_vocab_size")] * D["n_tables"])
    cmp("engram.enable_tokenizer_compression / Mac always builds the compressed map (engram.py:62)", eg["enable_tokenizer_compression"], True)
    cmp("engram compressed vocab: config engram_compressed_vocab_size vs Mac build_compressed_token_map(tokenizer.json)", text["engram_compressed_vocab_size"], D["engram_compressed_vocab_from_tokenizer"])
    cmp("engram.engram_pad_token_id / Mac NgramHasher pad_id default (engram.py:90) & inference engram_pad_id", eg["engram_pad_token_id"], [D["engram_pad_id_default"], inf.get("engram_pad_id")][0], f"inference engram_pad_id={inf.get('engram_pad_id')}")
    cmp("engram.tokenizer_vocab_size", eg["tokenizer_vocab_size"], A["vocab_size"])
    cmp("engram.fp8_embed/fp8_proj / inference dtype fp8", [eg["fp8_embed"], eg["fp8_proj"]], [inf.get("dtype") == "fp8"] * 2)
    conv = M.get("converter", {})
    rows = conv.get("engram_rows_in_index", {})
    cmp("engram table sizes: inference engram_num_embeddings vs index rows of layers.{1,14}.engram.embed.weight",
        [rows.get(f"layers.{l}.engram.embed.weight", [None])[0] for l in A["engram_layer_ids"]], inf.get("engram_num_embeddings"), "official reads the rows from the tensors")
    # dspark / dflash
    df = gpt["dflash_cfg"]
    cmp("dflash_cfg.dspark_block_size", df["dspark_block_size"], A["dspark_block_size"])
    cmp("dflash_cfg.dspark_target_layer_ids", df["dspark_target_layer_ids"], A["dspark_target_layer_ids"])
    cmp("dflash_cfg.dspark_markov_rank", df["dspark_markov_rank"], A["dspark_markov_rank"])
    cmp("dflash_cfg.dspark_noise_token_id", df["dspark_noise_token_id"], A["dspark_noise_token_id"])
    cmp("dflash_cfg.moe.n_routed_experts / dspark_n_routed_experts", df["moe"]["n_routed_experts"], A["dspark_n_routed_experts"])
    cmp("dflash_cfg.moe.num_experts_per_tok / dspark_n_activated_experts", df["moe"]["num_experts_per_tok"], A["dspark_n_activated_experts"])
    cmp("dflash_cfg.pad_token_id / engine sampling blacklist contains pad", df["pad_token_id"], 2 if 2 in M["tokenizer"]["blacklist"] else None)
    g.info("dflash_cfg.dspark_target_attn_input", f"official={df.get('dspark_target_attn_input')} (no Mac Args field; structural choice in dspark.py)")
    # attention topology
    kinds, ratios = aml["kinds"], aml["ratios"]
    mac_ratios = list(A["compress_ratios"])
    expect_kinds = ["window" if r == 0 else f"compress{r}" for r in mac_ratios[: A["n_layers"]]] + ["HEAD"] + ["dflash"] * A["n_mtp_layers"]
    cmp("attn_module_list kinds from compress_ratios (40 backbone + head + 3 MTP)", kinds, expect_kinds, f"len {aml['len']} vs {len(expect_kinds)}")
    cmp("compress_ratios (HF text_config vs inference config)", text["compress_ratios"], mac_ratios)
    cs = aml["compress_spec_example"]
    cmp("compress spec: compress_rope_theta", cs["compress_rope_theta"], A["compress_rope_theta"])
    cmp("compress spec: index_topk", cs["index_topk"], A["index_topk"])
    cmp("compress spec: attn_index.index_n_heads", cs["attn_index"]["index_n_heads"], A["index_n_heads"])
    cmp("compress spec: attn_index.index_head_dim", cs["attn_index"]["index_head_dim"], A["index_head_dim"])
    cmp("compress spec: attn_index.qk_rope_head_dim", cs["attn_index"]["qk_rope_head_dim"], A["rope_head_dim"])
    cmp("compress spec: rope_variant yarn + params / Mac Rope(original_seq_len, compress_rope_theta, rope_factor, beta_fast, beta_slow) (model.py:353)",
        [cs["rope_variant"], cs["rope_yarn_params"]],
        ["yarn", {"original_max_position_embeddings": A["original_seq_len"], "factor": A["rope_factor"], "beta_fast": A["beta_fast"], "beta_slow": A["beta_slow"], "mscale": False}])
    cmp("window spec: rope_theta, no yarn (Mac Rope(..., 0, rope_theta, ...) model.py:355)", [aml["window_spec"]["rope_theta"], aml["window_spec"].get("rope_variant")], [A["rope_theta"], None])
    cmp("dflash spec rope_theta", aml["dflash_spec"]["rope_theta"], A["rope_theta"])

    def nearest_map(sources, n_layer):  # config.py:62-65
        ids = sorted(sources)
        return {str(layer): max(s for s in ids if s <= layer) for layer in range(ids[0], n_layer)}

    sh = aml["shared"]
    cmp("shared.layer_kv_map / nearest kv_source_layers", sh["layer_kv_map"], nearest_map(A["kv_source_layers"], A["n_layers"]))
    cmp("shared.layer_index_module_map / nearest index_source_layers", sh["layer_index_module_map"], nearest_map(A["index_source_layers"], A["n_layers"]))
    cmp("shared.layer_index_k_map / nearest kv_source_layers (index K owned by KV sources, model.py:340)", sh["layer_index_k_map"], nearest_map(A["kv_source_layers"], A["n_layers"]))
    cmp("shared.layer_candidate_source_map / candidate_source_layer", sh["layer_candidate_source_map"], {str(l): A["candidate_source_layer"] for l in range(A["candidate_source_layer"], A["n_layers"])})
    cmp("shared.candidate_source_cfg / candidate_topk_blocks, candidate_block_size", sh["candidate_source_cfg"],
        {str(A["candidate_source_layer"]): {"candidate_topk_blocks": A["candidate_topk_blocks"], "candidate_block_size": A["candidate_block_size"], "topk_block_score_reduce": "max"}})
    cmp("kv_source_layer_ids / index_source_layer_ids", [text["kv_source_layer_ids"], text["index_source_layer_ids"]], [A["kv_source_layers"], A["index_source_layers"]])
    # vision
    vc, al = hf["vision_config"], oc["top"]["vl_aligner"]["aligner"]["params"]
    cmp("vision_config.num_hidden_layers / vision_n_layers", vc["num_hidden_layers"], A["vision_n_layers"])
    cmp("vision_config.hidden_size / vision_dim (== aligner input_dim)", [vc["hidden_size"], al["input_dim"]], [A["vision_dim"]] * 2)
    cmp("vision_config.num_attention_heads / vision_n_heads", vc["num_attention_heads"], A["vision_n_heads"])
    cmp("vision_config.intermediate_size / vision_inter_dim", vc["intermediate_size"], A["vision_inter_dim"])
    cmp("vision_config.patch_size / vision_patch_size", vc["patch_size"], A["vision_patch_size"])
    cmp("vision_config.downsample_ratio / vision_downsample_ratio (== aligner downsample_ratio)", [vc["downsample_ratio"], al["downsample_ratio"]], [A["vision_downsample_ratio"]] * 2)
    cmp("vision_config.max_image_tokens / vision_max_n_token", vc["max_image_tokens"], A["vision_max_n_token"])
    cmp("vision_config.min_pixels / vision_min_pixels", vc["min_pixels"], A["vision_min_pixels"])
    cmp("vision_config.max_wh_ratio / vision_max_wh_ratio", vc["max_wh_ratio"], A["vision_max_wh_ratio"])
    cmp("vision_config.rope_theta / vision_rope_theta", vc["rope_theta"], A["vision_rope_theta"])
    cmp("aligner hidden_size / dim", al["hidden_size"], A["dim"])
    cmp("image_token_id", hf["image_token_id"], A["image_token_id"])
    cmp("bos/eos/pad ids (sentence_level_mask, dflash pad)", [oc["top"]["sentence_level_mask"]["bos_token_id"], oc["top"]["sentence_level_mask"]["eos_token_id"], df["pad_token_id"]], [0, M["tokenizer"]["EOS_ID"], 2])
    cmp("quantization: mxfp4 experts / inference expert_dtype fp4", gpt["mxfp4"], inf.get("expert_dtype") == "fp4", "Mac re-quantises fp4 experts to affine 4-bit per layout.json (remote.py:100-115)")
    cmp("quantization: fp8_attn/fp8_mlp/mxfp8 / inference dtype fp8", [gpt["fp8_attn"], gpt["fp8_mlp"], gpt["mxfp8"]], [inf.get("dtype") == "fp8"] * 3, "Mac: fp8 32x32 blocks -> mxfp8 bit-exact (remote.py:87-91)")
    g.info("official-only fields (no Mac counterpart, informational)",
           f"max_position_embeddings={gpt['max_position_embeddings']} num_key_value_heads={gpt['num_key_value_heads']} use_fp4_sparse_kv={gpt['use_fp4_sparse_kv']} "
           f"sparse_stage={gpt['sparse_stage']} hc_shifted={gpt['hc_shifted']} hc_norm_group_size={gpt['hc_norm_group_size']} rope_interleaved={gpt['rope_interleaved']}")
    return g.result()


FAMILY_HINTS = {  # index-name family -> substrings the Mac code uses to address it (model.py builds hc_* keys as f"hc_{k}")
    "hc_attn_base": ["attn_base"], "hc_attn_fn": ["attn_fn"], "hc_attn_scale": ["attn_scale"],
    "hc_ffn_base": ["ffn_base"], "hc_ffn_fn": ["ffn_fn"], "hc_ffn_scale": ["ffn_scale"],
}


def family(name: str) -> str:
    f = re.sub(r"^(layers|mtp)\.\d+\.", "", name)
    f = re.sub(r"experts\.\d+\.", "experts.", f)
    f = re.sub(r"vision\.blocks\.\d+\.", "vision.blocks.", f)
    return re.sub(r"\.(weight|scale|bias)$", "", f)


def gate_h3(O: dict, M: dict, args) -> bool:
    g = Gate("H3", "weight coverage (hf_to_model, index == tensor_index, converter task consumption)")
    ref, mac = Path(args.ref), Path(args.mac_repo)
    wm = json.load(open(ref / "model.safetensors.index.json"))["weight_map"]
    ti = json.load(open(mac / "deepinfer_mac" / "convert" / "tensor_index.json"))
    only_index, only_ti = sorted(set(wm) - set(ti)), sorted(set(ti) - set(wm))
    g.check(not only_index and not only_ti and len(wm) == 96085, "tensor names: model.safetensors.index.json == converter tensor_index.json",
            f"{len(wm)} vs {len(ti)} tensors; only-in-index={only_index[:5]} only-in-tensor_index={only_ti[:5]}")
    shard_mism = [n for n in wm if n in ti and ti[n]["shard"] != wm[n]]
    g.check(not shard_mism, "shard assignment agrees for every tensor", f"mismatches={shard_mism[:5] or 'none'} across {len(set(wm.values()))} shards")
    n_scale = sum(1 for n in wm if n.endswith(".scale"))
    g.check(len(wm) - n_scale == 48496, "non-.scale tensor count", f"{len(wm) - n_scale} (scale tensors {n_scale})")
    on = O.get("names")
    if g.check(on is not None, "official names dump present", str(O["errors"].get("names", ""))):
        g.check(on["n_unmapped"] == 0 and on["n_mapped"] == len(wm), f"hf_to_model(name, n_layer={on['n_layer']}) maps every index tensor",
                f"mapped={on['n_mapped']} unmapped={on['n_unmapped']} {on['unmapped'][:5]} mapping sha256={on['mapping_sha256'][:16]}")
        g.check(on["n_targets_with_two_sources"] == 43 and not on["targets_with_more_than_two_sources"],
                "shared_experts w1/w3 -> one linear1 target per layer (40 backbone + 3 MTP)", f"two-source targets={on['n_targets_with_two_sources']} >2={on['targets_with_more_than_two_sources']}")
        g.info("suffixes passed through unchanged by hf_to_model (parameter container decides)", json.dumps(on["passthrough_suffix_patterns"], ensure_ascii=False))
        g.check(on["has_mtp_tied_tensors_in_index"] == [] and on["tied_sources_present"] == ["embed.weight", "head.weight"],
                "tied MTP embed/head: HF stores no mtp.N.embed/head copies; backbone embed.weight/head.weight present",
                f"mtp tied tensors in index={on['has_mtp_tied_tensors_in_index']} sources={on['tied_sources_present']} "
                f"(official aliases MTP targets to the backbone: hugging_face/loading.py:69-83; Mac MTP uses self.model.embed_tokens/head: dspark.py:115,121)")
        g.info("module ids covered by the mapping", ",".join(on["module_ids_covered"]))
    mc = M.get("converter")
    if g.check(mc is not None, "Mac converter task dump present", str(M["errors"].get("converter", ""))):
        g.check(mc["n_ignored"] == 0 and mc["n_consumed"] == mc["n_index"], "every index tensor is read by some converter task (Converter.tasks)",
                f"tasks={mc['n_tasks']} {mc['kind_counts']} consumed={mc['n_consumed']}/{mc['n_index']} ignored={mc['ignored'][:5] or 'none'}")
        g.check(not mc["problems"], "task payload sanity (expert regex, fp8 weight has scale, no orphan scale, embed payload)", f"problems={mc['problems'][:5] or 'none'}")
        g.check(mc["n_shard_mismatch_vs_hf_index"] == 0, "the Converter's loaded index names the same shard as the HF weight_map for every tensor",
                f"mismatches={mc['n_shard_mismatch_vs_hf_index']} {mc['shard_mismatch_examples'] or ''}")
    # informational: does the Mac model code address every converted tensor family?
    src = ""
    for p in [mac / "deepinfer_mac" / "model" / "model.py", mac / "deepinfer_mac" / "model" / "vision.py", mac / "deepinfer_mac" / "model" / "engram.py",
              mac / "deepinfer_mac" / "dspark.py", mac / "deepinfer_mac" / "weights.py", mac / "deepinfer_mac" / "store.py", mac / "deepinfer_mac" / "convert" / "remote.py"]:
        src += p.read_text()
    fams = sorted({family(n) for n in wm})
    unref = []
    for f in fams:
        hints = FAMILY_HINTS.get(f, [f, f.split(".")[-1]])
        if f == "embed":
            hints = ["embed.weight", "embed.bf16"]
        if not any(h in src for h in hints):
            unref.append(f)
    g.info(f"{len(fams)} tensor families; families with no reference in the Mac model/converter sources", str(unref or "none"))
    return g.result()


# expected recipe-vs-golden outcomes (documented contract differences are asserted, not waived)
H4_EXPECT = {
    (1, "thinking_mode=thinking"): ("MATCH", None),
    (2, "thinking_mode=chat"): ("MATCH", None),
    (2, "thinking_mode=thinking"): ("DIFF", "reasoning_effort_header_only"),  # golden 2 is a chat-mode rendering; thinking adds the effort header by design
    (3, "thinking_mode=chat"): ("ConversionError", None),  # recipe (DeepSeek API validator) rejects `tool_calls: []`; encoding.py tolerates it
    (3, "thinking_mode=thinking"): ("ConversionError", None),
    # once the rejected `tool_calls: []` is dropped, the only remaining difference is the tools block: encoding.py takes
    # per-message `tools` (here on the mid-conversation system message) which the OpenAI request schema cannot carry ...
    (3, "thinking_mode=chat, empty tool_calls removed (probe A)"): ("DIFF", "tools_block_missing_only"),
    # ... and request-level tools render after the first system message instead of the mid-conversation one
    (3, "thinking_mode=chat, empty tool_calls removed + per-message tools hoisted to request level (probe B)"): ("DIFF", "tools_block_position_only"),
    (4, "thinking_mode=chat"): ("DIFF", "action_token_only"),  # `task: action` -> <｜action｜> is not expressible in the OpenAI request schema
    (4, "thinking_mode=thinking"): ("DIFF", "reasoning_effort_header_and_action_token"),
    (5, "thinking_mode=thinking, effort=max"): ("MATCH", None),
}


def gate_h4(O: dict, M: dict, inputs: dict) -> bool:
    g = Gate("H4", "prompt-rendering parity (deepseek_recipe vs checkpoint encoding.py via eval/common/encoder.py)")
    oe, mr = O.get("encoder"), M.get("recipe")
    if not g.check(oe is not None and mr is not None, "both renderer dumps present", f"official_err={O['errors'].get('encoder')} mac_err={M['errors'].get('recipe')}"):
        return g.result()
    g.info("official encoder module", oe["module_file"])
    g.check(all(r["reproduces_golden"] for r in oe["goldens"]), "official encoder reproduces its own 5 goldens (encode_case)", str([(r["case"], r["reproduces_golden"]) for r in oe["goldens"]]))
    # a) recipe vs goldens
    seen = set()
    for row in mr["goldens"]:
        key = (row["case"], row["label"])
        seen.add(key)
        exp_status, exp_kind = H4_EXPECT.get(key, ("?", None))
        got = row["status"]
        ok = got == exp_status and (exp_kind is None or row.get("diff_kind") == exp_kind)
        detail = f"got {got}" + (f" diff_kind={row.get('diff_kind')}" if got == "DIFF" else "") + f"; expected {exp_status}" + (f"/{exp_kind}" if exp_kind else "")
        if got == "ConversionError":
            detail += f"; HTTP {row.get('http_status')}: {row.get('message', '')[:120]}"
        elif got in ("MATCH", "DIFF"):
            detail += f"; golden {row['golden_chars']} chars, recipe {row['recipe_chars']} chars, images={row['n_images']}, tail={row['recipe_tail']!r}"
        elif got == "EXC":
            detail += f"; {row.get('message')}"
        g.check(ok, f"fixture {row['case']} [{row['label']}]", detail)
        if got == "DIFF" and row.get("diff_kind") == "other":
            for l in row.get("diff", [])[:8]:
                print("        " + l)
    missing = set(H4_EXPECT) - seen
    g.check(not missing, "all expected fixture/mode rows were produced", str(sorted(missing)) if missing else "")
    g.info("documented contract differences (request schema, not tokenization)",
           "fixture 3: the recipe (DeepSeek API validator) rejects `tool_calls: []`, which encoding.py tolerates; with it dropped, the fixture's per-message `tools` on the "
           "mid-conversation system message is ignored by the OpenAI schema (probe A: identical except the tools block), and request-level tools render after the first "
           "system message instead (probe B: identical block, different position) -- latest_reminder, mid-conversation <｜System｜>, DSML calls and tool_result render identically; "
           "fixture 4: encoding.py's internal `task: action` token <｜action｜> has no OpenAI-request equivalent, so the recipe renders a plain assistant turn; "
           "fixture 2 thinking-mode row only adds the Reasoning Effort header to a chat-mode golden")
    # b) official offline prompts, both renderers, chat + thinking(high)
    for ob, mb in zip(oe["offline"], mr["offline"]):
        for mode in ("chat", "thinking"):
            o, m = ob["renders"][mode], mb["renders"][mode]
            if "error" in m:
                g.check(False, f"offline prompt {ob['index']} [{mode}] mac render", m["error"])
                continue
            same = o["prompt"] == m["prompt"]
            same_ids = o.get("ids_no_special") == m["ids"]
            g.check(same and same_ids and o["n_images"] == m["n_images"] and m["ids"][:1] == [0] and m["ids"].count(0) == 1,
                    f"offline prompt {ob['index']} [{mode}{', effort=' + inputs['reasoning_effort'] if mode == 'thinking' else ''}]: byte-equal render, equal ids",
                    f"chars {o['chars']}/{m['chars']} tokens official={len(o.get('ids_no_special', []))} mac={len(m['ids'])} images={o['n_images']}/{m['n_images']} "
                    f"tail={m['prompt'][-28:]!r}")
            if not same:
                a, b = o["prompt"], m["prompt"]
                k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
                print(f"        first difference at char {k}: official={a[max(0, k - 30):k + 40]!r} mac={b[max(0, k - 30):k + 40]!r}")
    return g.result()


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=os.environ.get(f"ORACLE_GATE_{k.upper()}", v))
    ap.add_argument("--workdir", default=os.environ.get("ORACLE_GATE_WORKDIR"), help="where the JSON dumps/logs go (default: a fresh temp dir)")
    ap.add_argument("--reuse", action="store_true", help="skip the subprocesses and compare existing dumps in --workdir")
    args = ap.parse_args()
    for k in ("official_repo", "mac_repo", "ref"):
        if not Path(getattr(args, k)).is_dir():
            print(f"missing directory for --{k.replace('_', '-')}: {getattr(args, k)}")
            return 2
    for k in ("official_python", "mac_python"):
        if not os.access(getattr(args, k), os.X_OK):
            print(f"missing interpreter for --{k.replace('_', '-')}: {getattr(args, k)}  (see the module docstring to recreate the scratch venv)")
            return 2
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="oracle-gate-"))
    workdir.mkdir(parents=True, exist_ok=True)
    print(f"workdir: {workdir}")

    inputs = build_inputs(args)
    inputs_path = workdir / "inputs.json"
    inputs_path.write_text(json.dumps(inputs, ensure_ascii=False, indent=1))
    print(f"inputs: {len(inputs['offline_prompts'])} official offline prompts (examples/_prompts.py), {len(inputs['goldens'])} golden fixtures, "
          f"{len(inputs['special_tokens'])} special strings, {len(inputs['mm_span_tokens'])} mm-span strings")

    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
           "DEEPINFER_HF_MODEL_DIR": str(Path(args.ref).resolve()), "TOKENIZERS_PARALLELISM": "false",
           "DEEPINFER_USE_NEXTN_DECODE": env_default("DEEPINFER_USE_NEXTN_DECODE", "5"), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
    env.pop("PYTHONPATH", None)
    official_json, mac_json = workdir / "official.json", workdir / "mac.json"
    if not args.reuse:
        rc_o = run_side("official", args.official_python, HERE / "_oracle_official.py", args.official_repo,
                        ["--repo", args.official_repo, "--ref", args.ref, "--inputs", str(inputs_path), "--out", str(official_json)], env, workdir / "official.log")
        rc_m = run_side("mac", args.mac_python, HERE / "_oracle_mac.py", args.mac_repo,
                        ["--repo", args.mac_repo, "--ref", args.ref, "--inputs", str(inputs_path), "--out", str(mac_json), "--workdir", str(workdir)], env, workdir / "mac.log")
        if rc_o or rc_m or not official_json.exists() or not mac_json.exists():
            print("a side failed to produce its dump; see logs above")
            return 1
    O, M = json.load(open(official_json)), json.load(open(mac_json))
    for side, d in (("official", O), ("mac", M)):
        if d["errors"]:
            print(f"[{side}] partial dump, errors: {d['errors']}")

    results = {
        "H1": gate_h1(O, M, inputs),
        "H2": gate_h2(O, M),
        "H3": gate_h3(O, M, args),
        "H4": gate_h4(O, M, inputs),
    }
    print("\n=== summary ===")
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    ok = all(results.values())
    print(f"overall: {'PASS' if ok else 'FAIL'}  (dumps: {official_json}, {mac_json})")
    return 0 if ok else 1


def env_default(name: str, default: str) -> str:
    return os.environ.get(name, default)


if __name__ == "__main__":
    sys.exit(main())
