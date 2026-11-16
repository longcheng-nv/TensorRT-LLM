"""Teacher-forced quality on official-API continuations (antirez/ds4 V4.1-Flash vectors), through
the real single-token decode path, so decode-time policies (cache-aware substitution, speculation)
act exactly as when generating. Per position: NLL of the API token, our argmax == API top token.

usage: mk_quality.py OUT.json [--set general] [--max-cases N] [--cache-gb 16]
                     [--sub-rank R] [--sub-gap G] [--spec-k K] [--mk 0|1]
compare two runs: mk_quality.py --compare A.json B.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")

SETS = {
    "general": "/Users/loncheng/Models/data/ds4/gguf-tools/quality-testing/deepseek-v4.1-flash-20260910-general",
    "boundary": "/Users/loncheng/Models/data/ds4/gguf-tools/quality-testing/deepseek-v4.1-flash-20260910",
    "router": "/Users/loncheng/Models/data/ds4/gguf-tools/quality-testing/deepseek-v4.1-flash-20260919-router",
}


def load_cases(d: Path, manifest="manifest.tsv"):
    cases = []
    for line in (d / manifest).read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        cid = line.split("\t")[0]
        prompt = (d / "prompts" / f"{cid}.txt").read_text(encoding="utf-8")
        resp = json.loads((d / "responses" / f"{cid}.json").read_text())
        cases.append((cid, prompt, resp))
    return cases


def _byte_id(tok, b: bytes, s: str):
    i = tok.token_to_id(s)
    if i is not None:
        return i
    enc = tok.encode(b.decode("utf-8", "replace"), add_special_tokens=False).ids
    return enc[0] if len(enc) == 1 else None


def api_tokens(resp, tok):
    ids, lps, tops = [], [], []
    for e in resp["choices"][0]["logprobs"]["content"]:
        ids.append(_byte_id(tok, bytes(e["bytes"]), e["token"]))
        lps.append(e["logprob"])
        top = e["top_logprobs"][0]
        tops.append(_byte_id(tok, bytes(top["bytes"]), top["token"]))
    return ids, lps, tops


def compare(a_path, b_path, n_boot=2000):
    A, B = json.load(open(a_path)), json.load(open(b_path))
    ca = {c["id"]: c for c in A["cases"]}
    cb = {c["id"]: c for c in B["cases"]}
    common = [k for k in ca if k in cb and len(ca[k]["nll"]) == len(cb[k]["nll"])]
    dn = np.array([np.mean(cb[k]["nll"]) - np.mean(ca[k]["nll"]) for k in common])
    da = np.array([np.mean(cb[k]["agree"]) - np.mean(ca[k]["agree"]) for k in common])
    rng = np.random.default_rng(0)
    bs = rng.integers(0, len(common), (n_boot, len(common)))
    ci_n = np.percentile(dn[bs].mean(1), [2.5, 97.5])
    ci_a = np.percentile(da[bs].mean(1), [2.5, 97.5])
    na = np.mean([np.mean(ca[k]["nll"]) for k in common])
    nb = np.mean([np.mean(cb[k]["nll"]) for k in common])
    aa = np.mean([np.mean(ca[k]["agree"]) for k in common])
    ab = np.mean([np.mean(cb[k]["agree"]) for k in common])
    print(f"{len(common)} paired cases | NLL {na:.4f} -> {nb:.4f} (delta {dn.mean():+.4f}, 95% CI [{ci_n[0]:+.4f}, {ci_n[1]:+.4f}]) | "
          f"API top-1 agreement {aa:.2%} -> {ab:.2%} (delta {da.mean():+.2%}, CI [{ci_a[0]:+.2%}, {ci_a[1]:+.2%}])")
    print(f"  A: {A['config']}\n  B: {B['config']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--set", default="general")
    ap.add_argument("--manifest", default="manifest.tsv")
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--first-case", type=int, default=0)
    ap.add_argument("--cache-gb", type=float, default=16.0)
    ap.add_argument("--sub-rank", type=int, default=None)
    ap.add_argument("--sub-gap", type=float, default=float("inf"))
    ap.add_argument("--sub-free", type=int, default=99)
    ap.add_argument("--sub-mode", default="sub")
    ap.add_argument("--spec-k", type=int, default=4)
    ap.add_argument("--mk", type=int, default=1)
    a = ap.parse_args()
    if a.compare:
        compare(*a.compare)
        return
    import mlx.core as mx
    from tokenizers import Tokenizer

    from deepinfer_mac import LLM
    from deepinfer_mac.model import model as M

    M.MoE.SPEC_K = a.spec_k
    M.MoE.SUB_RANK = a.sub_rank
    M.MoE.SUB_GAP = a.sub_gap
    M.MoE.SUB_FREE_RANK = a.sub_free
    M.MoE.SUB_MODE = a.sub_mode
    M.Transformer.MK = bool(a.mk)
    root = Path("/Users/loncheng/Models/dsv41-mac")
    tok = Tokenizer.from_file(str(root / "tokenizer.json"))
    d = Path(SETS.get(a.set, a.set))
    cases = load_cases(d, a.manifest)[a.first_case:]
    if a.max_cases:
        cases = cases[: a.max_cases]
    llm = LLM(root, expert_cache_gb=a.cache_gb)
    store = llm.weights.experts
    config = {"set": a.set, "cache_gb": a.cache_gb, "sub_rank": a.sub_rank, "sub_gap": a.sub_gap, "sub_free": a.sub_free, "sub_mode": a.sub_mode, "spec_k": a.spec_k, "mk": a.mk}
    out = {"config": config, "cases": []}
    t_all = time.time()
    dec_t, dec_n = 0.0, 0
    io0 = dict(store.stats)
    sub0 = dict(M.MoE.sub_stats)
    for cid, prompt, resp in cases:
        ids, lps, tops = api_tokens(resp, tok)
        if any(i is None for i in ids):
            print(f"{cid}: skipped (unmapped API token)", flush=True)
            continue
        pids, _ = llm.chat_prompt([{"role": "user", "content": prompt}], thinking=False)
        llm.reset()
        st, logits, _, _ = llm._prefill(pids, want_hidden=False)
        rows = [np.array(logits[-1], dtype=np.float64)]
        t0 = time.perf_counter()
        for t in ids[:-1]:
            rows.append(np.array(llm.model.forward(st, [t])[-1], dtype=np.float64))
        dec_t += time.perf_counter() - t0
        dec_n += len(ids) - 1
        allg = np.stack(rows)
        mxv = allg.max(1, keepdims=True)
        lse = np.log(np.exp(allg - mxv).sum(1)) + mxv[:, 0]
        nll = -(allg[np.arange(len(ids)), ids] - lse)
        agree = [int(int(np.argmax(allg[j])) == tops[j]) for j in range(len(ids)) if tops[j] is not None]
        out["cases"].append({"id": cid, "prompt_tokens": len(pids), "nll": nll.tolist(), "agree": agree, "api_nll": float(-np.mean(lps))})
        print(f"{cid}: prompt {len(pids):5d} | cont {len(ids):3d} | NLL {nll.mean():.3f} (API {-np.mean(lps):.3f}) | agree {np.mean(agree):.1%}", flush=True)
    io = store.stats
    s = M.MoE.sub_stats
    summ = {
        "cases": len(out["cases"]), "positions": int(sum(len(c["nll"]) for c in out["cases"])),
        "mean_nll": float(np.mean([np.mean(c["nll"]) for c in out["cases"]])),
        "mean_agree": float(np.mean([np.mean(c["agree"]) for c in out["cases"]])),
        "decode_ms_per_tok": round(dec_t / max(1, dec_n) * 1e3, 1),
        "subs_per_tok": round((s["subs"] - sub0["subs"]) / max(1, dec_n), 2),
        "hit_rate_overall": round(store.hit_rate(), 3),
        "GB_read_total": round((io["bytes"] - io0["bytes"]) / 1e9, 1),
        "wall_s": round(time.time() - t_all),
    }
    out["summary"] = summ
    print(json.dumps(summ), flush=True)
    if a.out:
        json.dump(out, open(a.out, "w"))


if __name__ == "__main__":
    main()
