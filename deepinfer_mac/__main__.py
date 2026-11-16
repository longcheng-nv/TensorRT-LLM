"""deepinfer-mac command line.

  python -m deepinfer_mac convert --out ~/models/dsv41-mac      # stream + convert from the Hub
  python -m deepinfer_mac chat [--think]                         # interactive multi-turn chat
  python -m deepinfer_mac serve [--port 8000]                    # OpenAI/Anthropic-compatible API
  python -m deepinfer_mac bench [--prompt-len 512 --gen 64]      # throughput numbers
  python -m deepinfer_mac doctor                                 # readiness + conversion status

chat / serve / bench decode in the fast mode by default (~10 tok/s at --cache-gb 16 on an M4 Pro): LOSSY
cache-aware expert routing (--sub-rank 2 --spec-k 2), so outputs are not reproducible run to run.
--lossless restores exact routing: deterministic, SSD-bound (~4.5 tok/s).
"""

import argparse
import json
import sys
import time

from .engine import DEFAULT_MODEL_DIR


def decode_mode(a) -> str:
    """Set the decode-time MoE knobs from the CLI flags and return a one-line description of the mode.
    Default = the fast preset (--sub-rank 2 --spec-k 2, lossy); --lossless = exact routing (--spec-k 4).
    An explicit --sub-rank / --spec-k always wins."""
    from .model.model import MoE

    fast = not a.lossless
    spec_k = a.spec_k if a.spec_k is not None else (2 if fast else 4)
    sub_rank = a.sub_rank if a.sub_rank is not None else (2 if fast else None)
    MoE.SKIP_MISS_BELOW = a.skip_miss
    MoE.PREFILL_TOPK = a.prefill_topk
    MoE.LOOKAHEAD, MoE.LOOKAHEAD_K = a.lookahead > 0, a.lookahead
    MoE.SPEC_K = 0 if a.lookahead else spec_k
    MoE.SPEC_K2 = a.spec_k2
    MoE.SUB_RANK = sub_rank
    MoE.SUB_GAP = a.sub_gap
    if a.no_mk:
        from .model.model import Attention, Transformer

        Transformer.MK = MoE.MK = Attention.MK = False
    lossy = [f"--sub-rank {sub_rank}"] if sub_rank is not None else []
    lossy += [f"--sub-gap {a.sub_gap}"] if sub_rank is not None and a.sub_gap != float("inf") else []
    lossy += [f"--moe-topk {a.moe_topk}"] if a.moe_topk else []
    lossy += [f"--skip-miss {a.skip_miss}"] if a.skip_miss else []
    lossy += [f"--prefill-topk {a.prefill_topk}"] if a.prefill_topk else []
    if lossy:
        return (f"decode mode: LOSSY ({' '.join(lossy)}, spec-k {MoE.SPEC_K}); outputs are not reproducible run to run, "
                "use --lossless for exact, deterministic routing")
    return f"decode mode: lossless (exact expert routing, spec-k {MoE.SPEC_K})"


def _llm(a):
    from .engine import LLM

    print(decode_mode(a), file=sys.stderr, flush=True)
    return LLM(a.model, expert_cache_gb=a.cache_gb, speculative=a.spec or None, moe_topk=a.moe_topk)


def chat(a):
    llm = _llm(a)
    from .engine import EOS_ID, SamplingParams

    msgs = []
    print("DeepSeek-V4.1-Flash on DeepInfer-Mac. Empty line = quit, /reset = new conversation, /img <path> <question>.\n")
    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            break
        if not user:
            break
        if user == "/reset":
            msgs = []
            llm.reset()
            continue
        if user.startswith("/img "):  # /img <path-or-url> question
            _, path, *rest = user.split(" ", 2)
            import base64, mimetypes

            if not path.startswith(("http://", "https://")):
                mime = mimetypes.guess_type(path)[0] or "image/png"
                path = f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()
            content = [{"type": "text", "text": rest[0] if rest else "Describe this image."}, {"type": "image_url", "image_url": {"url": path}}]
        else:
            content = user
        msgs.append({"role": "user", "content": content})
        extra = {"reasoning_effort": a.effort} if a.think and a.effort else {}
        ids, images, conv = llm.chat_inputs(msgs, thinking=a.think, **extra)
        sp = SamplingParams(temperature=a.temperature, top_p=1.0, max_tokens=a.max_tokens)
        out, shown = [], ""
        print("ds> ", end="", flush=True)
        for o in llm.stream(ids, sp, images):
            if o.token_id == EOS_ID:
                break
            out.append(o.token_id)
            text = llm.tokenizer.decode(out, skip_special_tokens=False)
            if not text.endswith("�"):
                print(text[len(shown):], end="", flush=True)
                shown = text
        print(f"\n   [{json.dumps(llm.last_stats)}]\n")
        reply = shown.split("</think>")[-1].strip()
        msgs.append({"role": "assistant", "content": reply})


def serve(a):
    import uvicorn

    from .server import create_app

    uvicorn.run(create_app(_llm(a)), host=a.host, port=a.port, log_level="info")


def bench(a):
    from .engine import SamplingParams

    llm = _llm(a)
    text = ("The history of computing is a story of abstraction layered on abstraction. " * 400)
    ids = llm.encode(text)[: a.prompt_len]
    for rnd in range(a.rounds):
        llm.reset()
        t = time.perf_counter()
        for _ in llm.stream(ids, SamplingParams(temperature=0, max_tokens=a.gen, ignore_eos=True)):
            pass
        print(json.dumps({"round": rnd, **llm.last_stats, "wall_s": round(time.perf_counter() - t, 1)}), flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="deepinfer_mac")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("chat", "serve", "bench"):
        p = sub.add_parser(name)
        p.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
        p.add_argument("--cache-gb", type=float, default=None, help="expert cache size (default: free memory - dense weights - 4 GB, keeping >= 9 GB for macOS)")
        p.add_argument("--spec", action="store_true", help="DSpark speculative decoding")
        p.add_argument("--moe-topk", type=int, default=None, help="experts per token (model: 6); fewer = faster, lossy")
        p.add_argument("--skip-miss", type=float, default=0.0, help="skip a non-cached expert under this routing-weight share (e.g. 0.08); lossy")
        p.add_argument("--prefill-topk", type=int, default=0, help="experts per prompt token outside the last 140 (e.g. 4); faster TTFT, lossy")
        p.add_argument("--lookahead", type=int, default=0, help="prefetch the k most confident next-layer experts (0 = off)")
        p.add_argument("--spec-k", type=int, default=None,
                       help="speculation: read the k most likely next-layer experts at low priority, inserted cold "
                            "(0 = off; default 2 in the fast mode, 4 with --lossless)")
        p.add_argument("--spec-k2", type=int, default=0, help="also guess k experts two layers ahead (0 = off)")
        p.add_argument("--sub-rank", type=int, default=None,
                       help="LOSSY cache-aware routing at decode: a selected expert ranked >= N (0-based) that is not cached "
                            "is replaced by the best-scoring cached one instead of read from SSD (2 = keep the top-2 exact; "
                            "default 2, off with --lossless)")
        p.add_argument("--sub-gap", type=float, default=float("inf"), help="... only when the substitute's biased router score is within this gap")
        mode = p.add_mutually_exclusive_group()
        mode.add_argument("--fast", action="store_true",
                          help="the default decode mode (flag kept for compatibility): LOSSY preset for ~10 tok/s, --sub-rank 2 --spec-k 2")
        mode.add_argument("--lossless", action="store_true",
                          help="exact expert routing: deterministic, reproducible outputs (~4.5 tok/s, SSD-bound)")
        p.add_argument("--no-mk", action="store_true", help="disable the fused decode kernels (reference MLX path)")
        if name == "chat":
            p.add_argument("--think", action="store_true")
            p.add_argument("--effort", choices=["high", "max"], default=None, help="reasoning effort in thinking mode (API default: high = 75)")
            p.add_argument("--temperature", type=float, default=1.0, help="model card: 1.0")
            p.add_argument("--max-tokens", type=int, default=2048)
        if name == "serve":
            p.add_argument("--host", default="127.0.0.1")
            p.add_argument("--port", type=int, default=8000)
        if name == "bench":
            p.add_argument("--prompt-len", type=int, default=256)
            p.add_argument("--gen", type=int, default=32)
            p.add_argument("--rounds", type=int, default=2)
    d = sub.add_parser("doctor")
    d.add_argument("--model", default=str(DEFAULT_MODEL_DIR))
    c = sub.add_parser("convert")
    c.add_argument("--out", default=str(DEFAULT_MODEL_DIR))
    c.add_argument("--workers", type=int, default=6)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.cmd == "doctor":
        from pathlib import Path

        from .doctor import main as doctor

        return doctor(Path(a.model).expanduser())
    if a.cmd == "convert":
        from .convert.remote import main as conv

        sys.argv = ["convert", "--out", a.out, "--workers", str(a.workers)]
        return conv()
    {"chat": chat, "serve": serve, "bench": bench}[a.cmd](a)


if __name__ == "__main__":
    main()
