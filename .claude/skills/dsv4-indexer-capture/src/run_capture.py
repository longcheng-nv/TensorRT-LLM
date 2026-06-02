"""DSv4 indexer-capture driver (Flash / Pro, parameterized).

This script is invoked by `launch_capture.sh`, which sets:
  - PYTHONPATH=<q9j-worktree>:$PYTHONPATH    so `import tensorrt_llm` -> worktree
  - PYTHONSAFEPATH=1                          (suppress cwd auto-injection)
  - PYTHONDONTWRITEBYTECODE=1                 (no .pyc on shared NFS)
  - DSV4_INDEXER_CAPTURE_DIR=<out>            (activates the dsa.py hook)
  - DSV4_INDEXER_CAPTURE_LAYERS="2,4,...,N"   (layer subset; set per model)
  - DSV4_INDEXER_CAPTURE_PHASE=prefill|decode|both
  - DSV4_INDEXER_CAPTURE_LAYOUT=single-file|per-layer
  - DSV4_INDEXER_CAPTURE_FORMAT=pt|npz

The Python driver is responsible for:
  - validating that `import tensorrt_llm` resolves to the q9j worktree
  - resolving model_path for `flash` / `pro` aliases
  - reading config.json for index_topk + num_hidden_layers + compress_ratios
  - loading prompt from raw text or @<jsonl>[#idx]
  - applying DeepseekV4Tokenizer chat template
  - sizing max_num_tokens from (prompt_tokens + osl)
  - calling LLM.generate with cuda_graph_config=None + temperature=0.0
  - writing manifest.json so downstream analyses can reconstruct context

Atexit flush of capture tensors is owned by the dsa.py hook.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


WORKTREE = "/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM-q9j"

MODEL_PATH_CANDIDATES = {
    "flash": [
        "/dev/shm/DeepSeek-V4-Flash",
        f"/raid/data/{os.environ.get('USER', 'loncheng')}-stage/DeepSeek-V4-Flash",
        "/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V4-Flash",
        "/home/scratch.jinshik_gpu/DeepSeek-V4-Flash",
    ],
    "pro": [
        "/dev/shm/DeepSeek-V4-Pro",
        f"/raid/data/{os.environ.get('USER', 'loncheng')}-stage/DeepSeek-V4-Pro",
        "/home/scratch.trt_llm_data_ci/llm-models/DeepSeek-V4-Pro",
    ],
}


def validate_worktree_redirect() -> None:
    import tensorrt_llm
    path = tensorrt_llm.__file__
    if "/TensorRT-LLM-q9j/" not in path:
        raise RuntimeError(
            f"FATAL: tensorrt_llm resolved to {path}\n"
            "  expected substring: /TensorRT-LLM-q9j/\n"
            "  fix: invoke through launch_capture.sh, which sets PYTHONPATH + "
            "PYTHONSAFEPATH=1.")
    print(f"[run_capture] tensorrt_llm = {path}", flush=True)


def validate_capture_env() -> str:
    out = os.environ.get("DSV4_INDEXER_CAPTURE_DIR")
    if not out:
        raise RuntimeError(
            "FATAL: DSV4_INDEXER_CAPTURE_DIR is unset; the dsa.py hook is "
            "env-gated. Invoke through launch_capture.sh.")
    os.makedirs(out, exist_ok=True)
    canary = os.path.join(out, ".capture_canary")
    with open(canary, "w") as f:
        f.write("ok")
    os.remove(canary)
    print(f"[run_capture] capture dir = {out} (writable)", flush=True)
    return out


def resolve_model_path(model_arg: str) -> tuple[str, str]:
    """Return (resolved_absolute_path, variant_tag) where variant_tag is
    one of 'flash' / 'pro' / 'custom'."""
    if model_arg in ("flash", "pro"):
        for cand in MODEL_PATH_CANDIDATES[model_arg]:
            if os.path.isdir(cand):
                return cand, model_arg
        raise FileNotFoundError(
            f"No DSv4 {model_arg.title()} checkpoint found in any of "
            f"{MODEL_PATH_CANDIDATES[model_arg]}")
    if not os.path.isdir(model_arg):
        raise FileNotFoundError(f"--model path does not exist: {model_arg}")
    # Sniff variant from config.json name
    cfg_name = ""
    cfg = Path(model_arg) / "config.json"
    if cfg.is_file():
        cfg_name = json.loads(cfg.read_text()).get("_name_or_path", "")
    if "Flash" in cfg_name or "flash" in model_arg.lower():
        return model_arg, "flash"
    if "Pro" in cfg_name or "pro" in model_arg.lower():
        return model_arg, "pro"
    return model_arg, "custom"


def read_model_config(model_path: str) -> dict:
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.json missing in {model_path}")
    return json.loads(cfg_path.read_text())


def parse_layers_arg(spec: str, num_hidden_layers: int,
                     compress_ratios: list | None) -> tuple[int, ...]:
    if spec == "all":
        return tuple(range(num_hidden_layers))
    if spec == "even":
        # GVR-active layers = those with compress_ratio == 4
        if compress_ratios:
            return tuple(i for i, cr in enumerate(compress_ratios) if cr == 4)
        return tuple(range(2, num_hidden_layers, 2))
    return tuple(int(x) for x in spec.split(",") if x.strip())


def load_prompt(spec: str) -> list[dict]:
    """Parse `--prompt`. Returns a chat-message list.

    Two forms:
      - raw text:     "What is 2+2?"
      - jsonl ref:    "@/path/to/foo.jsonl#3"   (defaults to #0)
    """
    if not spec.startswith("@"):
        return [{"role": "user", "content": spec}]
    ref = spec[1:]
    if "#" in ref:
        path, idx = ref.rsplit("#", 1)
        idx = int(idx)
    else:
        path, idx = ref, 0
    if not os.path.isfile(path):
        raise FileNotFoundError(f"--prompt jsonl not found: {path}")
    with open(path) as f:
        for i, line in enumerate(f):
            if i == idx:
                entry = json.loads(line)
                msgs = []
                sys_text = entry.get("system", "")
                if sys_text:
                    msgs.append({"role": "system", "content": sys_text})
                msgs.append({"role": "user", "content": entry["user"]})
                return msgs
    raise IndexError(f"prompt[{idx}] not found in {path}")


def round_up(x: int, step: int) -> int:
    return ((x + step - 1) // step) * step


def git_sha(worktree: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", worktree, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True,
                   help="'flash' | 'pro' | absolute checkpoint path")
    p.add_argument("--prompt", required=True,
                   help="raw text  OR  @<jsonl-path>[#idx]")
    p.add_argument("--osl", type=int, default=300,
                   help="max_new_tokens (default 300)")
    p.add_argument("--phase", choices=("prefill", "decode", "both"),
                   default="both")
    p.add_argument("--layers", default="even",
                   help="'all' | 'even' (GVR-active) | comma list (default even)")
    p.add_argument("--num-gpus", type=int, default=8,
                   help="TP=EP=N. DSv4 production layout (default 8)")
    p.add_argument("--save-format", choices=("pt", "npz"), default="pt")
    p.add_argument("--layout", choices=("single-file", "per-layer"),
                   default="per-layer")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--index-topk", default="auto",
                   help="'auto' (from config.json) | 512 | 1024")
    p.add_argument("--kv-cache-frac", type=float, default=0.7)
    args = p.parse_args()

    validate_worktree_redirect()
    out_dir = validate_capture_env()
    assert out_dir == args.out_dir, (
        f"DSV4_INDEXER_CAPTURE_DIR={out_dir} != --out-dir={args.out_dir}; "
        f"launch_capture.sh must set the env var to match --out-dir")

    # Resolve model
    model_path, variant = resolve_model_path(args.model)
    print(f"[run_capture] model = {variant} @ {model_path}", flush=True)

    cfg = read_model_config(model_path)
    num_hidden_layers = cfg["num_hidden_layers"]
    cfg_index_topk = cfg.get("index_topk", 512)
    compress_ratios = cfg.get("compress_ratios")
    if args.index_topk == "auto":
        index_topk = cfg_index_topk
    else:
        index_topk = int(args.index_topk)

    layers = parse_layers_arg(args.layers, num_hidden_layers, compress_ratios)
    print(f"[run_capture] layers logged ({len(layers)}): {layers[:5]}...{layers[-3:]}",
          flush=True)

    # Verify env vars set by launcher match CLI flags (defense in depth)
    env_layers = os.environ.get("DSV4_INDEXER_CAPTURE_LAYERS", "")
    expected = ",".join(str(x) for x in layers)
    if env_layers and env_layers != expected:
        raise RuntimeError(
            f"DSV4_INDEXER_CAPTURE_LAYERS={env_layers!r} != "
            f"--layers expansion {expected!r}; launcher and CLI disagree")
    if os.environ.get("DSV4_INDEXER_CAPTURE_PHASE") != args.phase:
        raise RuntimeError("DSV4_INDEXER_CAPTURE_PHASE / --phase mismatch")
    if os.environ.get("DSV4_INDEXER_CAPTURE_LAYOUT") != args.layout:
        raise RuntimeError("DSV4_INDEXER_CAPTURE_LAYOUT / --layout mismatch")
    if os.environ.get("DSV4_INDEXER_CAPTURE_FORMAT") != args.save_format:
        raise RuntimeError("DSV4_INDEXER_CAPTURE_FORMAT / --save-format mismatch")

    # Prompt
    messages = load_prompt(args.prompt)
    from tensorrt_llm.tokenizer.deepseek_v4 import DeepseekV4Tokenizer
    tokenizer = DeepseekV4Tokenizer.from_pretrained(model_path,
                                                    trust_remote_code=True)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False)
    prompt_tokens = len(tokenizer.encode(prompt_text))
    print(f"[run_capture] prompt tokens = {prompt_tokens}", flush=True)

    # max_num_tokens sizing
    max_num_tokens = round_up(prompt_tokens + args.osl + 1024, 1024)
    # Safety floor for short prompts
    max_num_tokens = max(max_num_tokens, 8192)

    # LLM config
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig
    from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig

    sparse_cfg = DeepSeekV4SparseAttentionConfig(
        algorithm="deepseek_v4", enable_heuristic_topk=True)
    kv_cache_cfg = KvCacheConfig(free_gpu_memory_fraction=args.kv_cache_frac)

    print(f"[run_capture] building LLM: TP=EP={args.num_gpus} "
          f"max_num_tokens={max_num_tokens} index_topk={index_topk} "
          f"cuda_graph=disabled GVR=enabled MTP=off", flush=True)
    t0 = time.time()
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.num_gpus,
        moe_expert_parallel_size=args.num_gpus,
        max_batch_size=2,
        max_num_tokens=max_num_tokens,
        max_seq_len=max_num_tokens,
        kv_cache_config=kv_cache_cfg,
        cuda_graph_config=None,
        sparse_attention_config=sparse_cfg,
    )
    print(f"[run_capture] LLM built in {time.time() - t0:.1f}s", flush=True)

    sampling = SamplingParams(temperature=0.0, max_tokens=args.osl)
    print(f"[run_capture] generating: input={prompt_tokens} osl={args.osl} (greedy)",
          flush=True)
    t1 = time.time()
    outputs = llm.generate([prompt_text], sampling_params=sampling)
    elapsed = time.time() - t1
    out_tokens = outputs[0].outputs[0].token_ids
    out_text = outputs[0].outputs[0].text
    print(f"[run_capture] generation done in {elapsed:.1f}s; "
          f"output tokens = {len(out_tokens)}", flush=True)
    print(f"[run_capture] output preview: {out_text[:160]!r}", flush=True)

    manifest = {
        "model_variant": variant,
        "model_path": model_path,
        "prompt_source": args.prompt,
        "prompt_token_count": prompt_tokens,
        "max_new_tokens": args.osl,
        "actual_output_tokens": len(out_tokens),
        "temperature": 0.0,
        "phase": args.phase,
        "layers_logged": list(layers),
        "tp_size": args.num_gpus,
        "ep_size": args.num_gpus,
        "index_topk": index_topk,
        "num_hidden_layers": num_hidden_layers,
        "max_num_tokens": max_num_tokens,
        "kv_cache_frac": args.kv_cache_frac,
        "sparse_attention": "deepseek_v4",
        "enable_heuristic_topk": True,
        "mtp": "off",
        "cuda_graph": "disabled",
        "save_format": args.save_format,
        "layout": args.layout,
        "capture_dir": out_dir,
        "elapsed_seconds": elapsed,
        "hook_version": "v2",
        "worktree": WORKTREE,
        "worktree_sha": git_sha(WORKTREE),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[run_capture] wrote manifest: {os.path.join(out_dir, 'manifest.json')}",
          flush=True)

    # atexit hook in dsa.py flushes the capture buffers from here.
    return 0


if __name__ == "__main__":
    sys.exit(main())
