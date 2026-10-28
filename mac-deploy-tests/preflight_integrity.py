"""Pre-flight integrity gate for the DeepInfer-Mac conversion.

1. Confirms the Hugging Face repo sha equals the revision pinned in deepinfer_mac/convert/hf.py.
2. Range-reads the safetensors header of all 48 shards at that pinned revision (direct, no proxy)
   and checks every tensor's dtype/shape/absolute byte offsets against the converter's bundled
   tensor_index.json. Any mismatch means the converter would read garbage: abort.

Usage: .venv/bin/python preflight_integrity.py   (from anywhere; paths are absolute)
"""
from __future__ import annotations

import json
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, "/Users/loncheng/Models/DeepInfer-Mac")
from deepinfer_mac.convert import hf  # noqa: E402

INDEX = "/Users/loncheng/Models/DeepInfer-Mac/deepinfer_mac/convert/tensor_index.json"


def session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False  # bypass the local privoxy (4x slower) and any env proxies
    return s


def get(s: requests.Session, path: str, start: int, end: int) -> bytes:
    r = s.get(hf.url(path), headers={"Range": f"bytes={start}-{end - 1}"}, timeout=60)
    r.raise_for_status()
    assert r.status_code == 206, (path, r.status_code)
    return r.content


def check_shard(shard: str, expected: dict) -> tuple[str, int, int, list[str]]:
    s = session()
    n = struct.unpack("<Q", get(s, shard, 0, 8))[0]
    h = json.loads(get(s, shard, 8, 8 + n))
    h.pop("__metadata__", None)
    problems = []
    seen = 0
    for k, v in h.items():
        a, b = v["data_offsets"]
        got = {"dtype": v["dtype"], "shape": v["shape"], "start": 8 + n + a, "end": 8 + n + b}
        exp = expected.get(k)
        if exp is None:
            problems.append(f"{k}: in shard header but not in tensor_index.json")
            continue
        seen += 1
        for f in ("dtype", "shape", "start", "end"):
            if exp[f] != got[f]:
                problems.append(f"{k}.{f}: index={exp[f]} header={got[f]}")
    missing = set(expected) - set(h)
    for k in sorted(missing)[:5]:
        problems.append(f"{k}: in tensor_index.json but not in shard header")
    if len(missing) > 5:
        problems.append(f"... {len(missing) - 5} more missing")
    return shard, len(h), seen, problems


def main() -> int:
    t0 = time.time()
    s = session()
    api = s.get(f"https://huggingface.co/api/models/{hf.REPO}", timeout=60).json()
    sha = api.get("sha")
    print(f"repo {hf.REPO} sha={sha} lastModified={api.get('lastModified')} pinned={hf.REVISION}")
    if sha != hf.REVISION:
        print("FAIL: repo sha differs from pinned REVISION (repo updated?) — re-verify before converting")
    idx = json.load(open(INDEX))
    by_shard: dict[str, dict] = {}
    for k, v in idx.items():
        by_shard.setdefault(v["shard"], {})[k] = v
    shards = sorted(by_shard)
    print(f"tensor_index.json: {len(idx)} tensors across {len(shards)} shards")
    total_bytes = sum(v["end"] - v["start"] for v in idx.values())
    print(f"total payload bytes: {total_bytes / 1e9:.3f} GB")
    bad = 0
    with ThreadPoolExecutor(8) as ex:
        for shard, n_hdr, n_ok, problems in ex.map(lambda sh: check_shard(sh, by_shard[sh]), shards):
            status = "OK " if not problems else "BAD"
            if problems:
                bad += 1
            print(f"  {status} {shard}: {n_hdr} tensors in header, {n_ok} matched" + (f"; {len(problems)} problems" if problems else ""))
            for p in problems[:8]:
                print("      ", p)
    print(f"done in {time.time() - t0:.1f}s; shards with problems: {bad}/{len(shards)}; sha match: {sha == hf.REVISION}")
    return 0 if (bad == 0 and sha == hf.REVISION) else 1


if __name__ == "__main__":
    raise SystemExit(main())
