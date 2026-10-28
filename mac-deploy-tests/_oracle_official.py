"""Official-side oracle dump for gate_official_oracles.py.

Runs under a venv that can import the OFFICIAL deepseek-ai/DeepInfer pure-Python modules
(transformers 4.50 / tokenizers 0.21 / CPU torch).  It never loads weights and never touches the
network: the caller must set HF_HUB_OFFLINE=1 and DEEPINFER_HF_MODEL_DIR=<reference checkpoint dir>
(the runner does), otherwise examples/_config.resolve_model_path() would snapshot_download 510 GB.

Everything it learns is written as one JSON file; the runner compares it against the Mac-side dump.

    python _oracle_official.py --repo <DeepInfer> --ref <ckpt-ref> --inputs inputs.json --out official.json
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.metadata as md
import json
import os
import re
import sys


def _plain(v):
    """AttrDict / tuple / torch.dtype -> JSON-able."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    return str(v)


def _version(name: str) -> str:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return "missing"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="official DeepInfer repo root")
    ap.add_argument("--ref", required=True, help="reference checkpoint dir (config/tokenizer/index/encoding)")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    repo, ref = os.path.realpath(a.repo), os.path.realpath(a.ref)

    # --- safety rails: offline, and the official loader must resolve to the local reference dir.
    if os.environ.get("HF_HUB_OFFLINE") != "1":
        sys.exit("refusing to run: HF_HUB_OFFLINE must be 1")
    if os.path.realpath(os.environ.get("DEEPINFER_HF_MODEL_DIR", "")) != ref:
        sys.exit("refusing to run: DEEPINFER_HF_MODEL_DIR must point at the reference checkpoint dir")

    sys.path.insert(0, repo)
    sys.path.insert(0, os.path.join(repo, "eval"))  # eval scripts import `common.encoder` relative to eval/
    os.chdir(repo)
    inputs = json.load(open(a.inputs))
    out: dict = {
        "python": sys.version.split()[0],
        "versions": {k: _version(k) for k in ("transformers", "tokenizers", "torch", "safetensors", "huggingface_hub")},
        "errors": {},
    }

    # ------------------------------------------------------------------ tokenizer (H1)
    try:
        from deepinfer.hugging_face.tokenizer import get_hf_tokenizer
        from deepinfer.multimodal_tokens import CANDIDATE_IMAGE_PLACEHOLDER_TOKENS, ImageTokenOffset

        tok = get_hf_tokenizer(ref)
        hf_cfg = json.load(open(os.path.join(ref, "config.json")))
        vocab = tok.get_vocab()
        # deepinfer/ve/multimodal_processor.py:69-71 resolves the image token from the candidates, then
        # :84 asserts it equals config.json image_token_id; examples/_prompts.py:38 spells it back.
        resolved = next((tid for t in CANDIDATE_IMAGE_PLACEHOLDER_TOKENS if (tid := vocab.get(t, -1)) != -1), -1)
        image_token_str = tok.convert_ids_to_tokens(hf_cfg["image_token_id"])

        def replace_image_tokens(prompt: str) -> str:  # examples/_prompts.py:42-50
            for placeholder in CANDIDATE_IMAGE_PLACEHOLDER_TOKENS:
                if placeholder != image_token_str:
                    prompt = prompt.replace(placeholder, image_token_str)
            return prompt

        t = {
            "class": type(tok).__name__,
            "len": len(tok),
            "vocab_size_config": hf_cfg["text_config"]["vocab_size"],
            "add_bos_token": bool(tok.add_bos_token),
            "add_eos_token": bool(tok.add_eos_token),
            "bos": [tok.bos_token, tok.bos_token_id],
            "eos": [tok.eos_token, tok.eos_token_id],
            "pad": [tok.pad_token, tok.pad_token_id],
            "unk_id": tok.unk_token_id,
            "encode_empty": tok.encode("", add_special_tokens=True),
            "specials": {s: tok.convert_tokens_to_ids(s) for s in inputs["special_tokens"]},
            "mm_span": {s: tok.convert_tokens_to_ids(s) for s in inputs["mm_span_tokens"]},
            "image_token_offsets": {o.name: hf_cfg["text_config"]["vocab_size"] + int(o) for o in ImageTokenOffset},
            "candidate_image_placeholders": list(CANDIDATE_IMAGE_PLACEHOLDER_TOKENS),
            "image_token_id_config": hf_cfg["image_token_id"],
            "image_token_id_resolved": resolved,
            "image_token_str": image_token_str,
            "raw_prompts": [],
            "goldens": [],
        }
        for p in inputs["offline_prompts"]:
            q = replace_image_tokens(p)
            t["raw_prompts"].append({
                "prompt_after_replace": q,
                "ids_default": tok.encode(q),  # LLM.generate -> llm_engine.tokenizer.encode(prompt): add_bos_token=True
                "ids_no_special": tok.encode(q, add_special_tokens=False),
                "n_image_tokens": tok.encode(q).count(hf_cfg["image_token_id"]),
            })
        for g in inputs["goldens"]:
            t["goldens"].append({
                "ids_no_special": tok.encode(g, add_special_tokens=False),  # eval/run_eval.py:222 policy for rendered text
                "ids_default": tok.encode(g),  # what tokenizer.encode(rendered) would do (BOS doubled)
            })
        out["tokenizer"] = t
    except Exception as e:  # noqa: BLE001
        out["errors"]["tokenizer"] = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ config (H2)
    try:
        from deepinfer.hugging_face.config import load_hf_model_config

        cfg = load_hf_model_config(os.path.join(ref, "config.json"))
        g = cfg.gpt
        gpt = {k: _plain(v) for k, v in g.items() if k != "attn_module_list"}
        aml = g.attn_module_list
        kinds, ratios = [], []
        for spec in aml:
            if spec is None:
                kinds.append("HEAD")
                ratios.append(None)
                continue
            c = spec.cfg
            if "kv_compress_ratio" in c:
                kinds.append(f"compress{c.kv_compress_ratio}")
                ratios.append(int(c.kv_compress_ratio))
            elif c.get("sparse_stage") == 0:
                kinds.append("window")
                ratios.append(0)
            else:
                kinds.append("dflash")
                ratios.append(0)
        n_layer = g.num_hidden_layers
        first_compress = next(s for s in aml if s is not None and "kv_compress_ratio" in s.cfg)
        shared = first_compress.cfg.layer_info_shared_cfg
        out["config"] = {
            "top": {k: _plain(v) for k, v in cfg.items() if k != "gpt"},
            "gpt": gpt,
            "attn_module_list": {
                "len": len(aml),
                "kinds": kinds,
                "ratios": ratios,
                "window_spec": _plain(aml[0].cfg),
                "compress_spec_example": {k: _plain(v) for k, v in first_compress.cfg.items() if k != "layer_info_shared_cfg"},
                "compress_ratio1_spec": _plain({k: v for k, v in aml[g.n_prefill_layer].cfg.items() if k != "layer_info_shared_cfg"}),
                "dflash_spec": _plain(aml[n_layer + 1].cfg) if len(aml) > n_layer + 1 else None,
                "shared": {
                    "layer_kv_map": {str(k): v for k, v in shared.layer_kv_map.items()},
                    "layer_index_module_map": {str(k): v for k, v in shared.layer_index_module_map.items()},
                    "layer_index_k_map": {str(k): v for k, v in shared.layer_index_k_map.items()},
                    "layer_candidate_source_map": {str(k): v for k, v in shared.get("layer_candidate_source_map", {}).items()},
                    "candidate_source_cfg": _plain(shared.get("candidate_source_cfg")),
                },
            },
            "hf_raw": json.load(open(os.path.join(ref, "config.json"))),
        }
    except Exception as e:  # noqa: BLE001
        out["errors"]["config"] = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ names (H3)
    try:
        from deepinfer.hugging_face import names as N

        index = json.load(open(os.path.join(ref, "model.safetensors.index.json")))
        wm = index["weight_map"]
        n_layer = json.load(open(os.path.join(ref, "config.json")))["text_config"]["num_hidden_layers"]
        unmapped, pairs = [], []
        passthrough = collections.Counter()
        module_hist = collections.Counter()
        dup_targets = collections.Counter()
        for name in sorted(wm):
            N.resolve_hf_name(name, wm)  # raises if the model. prefix dance fails
            m = N.hf_to_model(name, n_layer=n_layer)
            if m is None:
                unmapped.append(name)
                continue
            pairs.append(f"{name}\t{m}")
            module_hist[m.split(".")[2] if m.startswith("module.module.") else m.split(".")[0]] += 1
            if not name.endswith(".scale"):
                dup_targets[m] += 1
            plain = name.removeprefix("model.")
            lm = N._HF_LAYER_RE.match(plain)
            mm = N._HF_MTP_RE.match(plain)
            suffix = lm.group(2) if lm else (mm.group(2) if mm else None)
            if suffix is not None and not suffix.endswith(".scale"):
                known = suffix in N._HF_TO_MODEL_SUFFIX or (mm is not None and suffix in N._HF_TO_MTP_EXTRA)
                if not known:
                    passthrough[re.sub(r"\.\d+\.", ".N.", suffix)] += 1
        out["names"] = {
            "n_layer": n_layer,
            "n_index": len(wm),
            "n_scale": sum(1 for n in wm if n.endswith(".scale")),
            "n_unmapped": len(unmapped),
            "unmapped": unmapped[:50],
            "n_mapped": len(pairs),
            "mapping_sha256": hashlib.sha256("\n".join(pairs).encode()).hexdigest(),
            "passthrough_suffix_patterns": dict(passthrough),
            "n_targets_with_two_sources": sum(1 for v in dup_targets.values() if v == 2),
            "targets_with_more_than_two_sources": [k for k, v in dup_targets.items() if v > 2],
            "module_ids_covered": sorted(module_hist, key=lambda s: (not s.isdigit(), int(s) if s.isdigit() else 0, s)),
            "has_mtp_tied_tensors_in_index": sorted(n for n in wm if re.match(r"^(model\.)?mtp\.\d+\.(embed|head)\.weight$", n)),
            "tied_sources_present": [n for n in ("embed.weight", "head.weight") if n in wm],
        }
    except Exception as e:  # noqa: BLE001
        out["errors"]["names"] = f"{type(e).__name__}: {e}"

    # ------------------------------------------------------------------ encoder (H4)
    try:
        from common.encoder import load_encoder  # eval/common/encoder.py

        enc = load_encoder(inputs["official_model_name"])  # resolves via DEEPINFER_HF_MODEL_DIR; HF_HUB_OFFLINE guards the rest
        enc_file = os.path.realpath(enc.__file__)
        if not enc_file.startswith(ref + os.sep):
            raise RuntimeError(f"encoder loaded from outside the reference dir: {enc_file}")
        e: dict = {"module_file": enc_file, "goldens": [], "offline": []}
        tests_dir = os.path.join(ref, "encoding", "tests")
        for i in range(1, 6):
            cases = enc.load_cases(os.path.join(tests_dir, f"test_input_{i}.json"))
            golden = open(os.path.join(tests_dir, f"test_output_{i}.txt")).read()
            prompt, images = enc.encode_case(cases[0], "chat")
            e["goldens"].append({"case": i, "reproduces_golden": prompt == golden, "n_images": len(images), "len": len(prompt)})
        for body in inputs["offline_bodies"]:
            row = {"index": body["index"], "renders": {}}
            for mode in ("chat", "thinking"):
                kwargs = {"thinking_mode": mode}
                if mode == "thinking":
                    kwargs["reasoning_effort"] = inputs["reasoning_effort"]
                prompt, media = enc.encode_messages(body["messages"], return_multi_modal_data=True, **kwargs)
                rec = {"prompt": prompt, "n_images": len(media["images"]), "chars": len(prompt)}
                if "tokenizer" in out:
                    rec["ids_no_special"] = tok.encode(prompt, add_special_tokens=False)
                row["renders"][mode] = rec
            e["offline"].append(row)
        out["encoder"] = e
    except Exception as ex:  # noqa: BLE001
        out["errors"]["encoder"] = f"{type(ex).__name__}: {ex}"

    with open(a.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"[official] wrote {a.out}; errors: {out['errors'] or 'none'}")


if __name__ == "__main__":
    main()
