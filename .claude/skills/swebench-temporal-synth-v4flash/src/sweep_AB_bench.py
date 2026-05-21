"""Sweep A (seq-len scaling at BS=1) + Sweep B (BS scaling at N=64K) for both
V4 Flash and V4 Pro synth skills, nsys-NVTX-bracketed.

The canonical home of this script is
`.claude/skills/swebench-temporal-synth-v4flash/src/`; it imports both
that skill's `synth_temporal_data.py` and the sibling v4pro skill's
module via `importlib.util` (forcing distinct module objects to avoid
sys.modules aliasing — see Anti-Patterns in AGENTS.md).

Usage (paths are relative to your run dir; --out-dir defaults to ./data):
  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \\
    --capture-range-end=stop --force-overwrite=true -o data/sweep_A \\
    python3 sweep_AB_bench.py --sweep A --out-dir data
  nsys profile -t cuda,nvtx --capture-range=cudaProfilerApi \\
    --capture-range-end=stop --force-overwrite=true -o data/sweep_B \\
    python3 sweep_AB_bench.py --sweep B --out-dir data

Skill resolution order (when this file is NOT under a skill src/ dir):
  1. env var $SKILL_FLASH / $SKILL_PRO  (point at skill root or src/)
  2. sibling layout: <_HERE>/../../<skill-name>/src/
  3. walk up the dir tree looking for .claude/skills/<skill-name>/src/
  4. $TRTLLM_REPO/.claude/skills/<skill-name>/src/
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch
import nvtx
# Loads trtllm ops via the python package — do NOT also call
# torch.ops.load_library on libth_common.so, that triggers a
# double-registration of `trtllm::indexer_topk_decode` and crashes with
# c10::Error.
import tensorrt_llm  # noqa: F401

_HERE = os.path.dirname(os.path.abspath(__file__))


def _resolve_skill(skill_dirname, env_var):
    """Resolve absolute path to <skill>/src/ containing synth_temporal_data.py."""
    target_basename = "synth_temporal_data.py"

    def _ok(p):
        return p and os.path.isfile(os.path.join(p, target_basename))

    # 1. env var (accept either skill root or src/)
    p = os.environ.get(env_var)
    if p:
        for cand in (p, os.path.join(p, "src")):
            if _ok(cand):
                return cand

    # 2. sibling layout: this file lives under .../<flash-skill>/src/,
    #    peer at .../<pro-skill>/src/ (or vice versa)
    sibling = os.path.normpath(os.path.join(_HERE, "..", "..", skill_dirname, "src"))
    if _ok(sibling):
        return sibling

    # 3. walk up looking for .claude/skills/<name>/src
    cur = _HERE
    for _ in range(8):
        cand = os.path.join(cur, ".claude", "skills", skill_dirname, "src")
        if _ok(cand):
            return cand
        nxt = os.path.dirname(cur)
        if nxt == cur:
            break
        cur = nxt

    # 4. explicit TRTLLM_REPO env var
    repo = os.environ.get("TRTLLM_REPO")
    if repo:
        cand = os.path.join(repo, ".claude", "skills", skill_dirname, "src")
        if _ok(cand):
            return cand

    raise FileNotFoundError(
        f"Cannot locate {skill_dirname}/src/{target_basename}. "
        f"Set {env_var}=<path-to-skill-or-skill-src> or TRTLLM_REPO=<repo-root>.")


SKILL_FLASH = _resolve_skill("swebench-temporal-synth-v4flash", "SKILL_FLASH")
SKILL_PRO = _resolve_skill("swebench-temporal-synth-v4pro", "SKILL_PRO")

# Load each skill's synth_temporal_data.py as a DISTINCT module — else
# sys.modules['synth_temporal_data'] caches the first import and the
# second import returns the same object, silently using Flash's K/BETA_CFGS
# for the Pro family.
import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(path, "synth_temporal_data.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
synth_flash = _load("synth_flash", SKILL_FLASH)
synth_pro = _load("synth_pro", SKILL_PRO)
assert synth_flash.K_DEFAULT == 512, synth_flash.K_DEFAULT
assert synth_pro.K_DEFAULT == 1024, synth_pro.K_DEFAULT

SKILLS = {"flash": synth_flash, "pro": synth_pro}
RADIX_AUX_BLOCKS_MAX = 32
L2_FLUSH_BYTES = 128 * 1024 * 1024

DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def time_one(call, *, warmup=3, reps=10, flush):
    for _ in range(warmup):
        flush.zero_(); torch.cuda.synchronize()
        call(); torch.cuda.synchronize()
    for _ in range(reps):
        flush.zero_(); torch.cuda.synchronize()
        call(); torch.cuda.synchronize()


def make_bench_args(bundle_bs1, BS, dtype, K, compress_ratio):
    """Replicate a BS=1 synth bundle to BS rows for the given dtype.
    radix_aux_logits is always fp32."""
    logits_bs1 = bundle_bs1["logits"]  # [1, N_padded]
    preIdx_bs1 = bundle_bs1["preIdx"]  # [1, K]
    N_padded = logits_bs1.shape[1]
    seq_lens_val = bundle_bs1["meta"]["seq_lens_val"]
    if dtype == torch.float32:
        logits = logits_bs1.float().expand(BS, -1).contiguous()
    else:
        logits = logits_bs1.float().to(dtype).expand(BS, -1).contiguous()
    seq_lens = torch.full((BS,), seq_lens_val, dtype=torch.int32, device="cuda")
    preIdx = preIdx_bs1.expand(BS, -1).contiguous()
    indices = torch.empty((BS, K), dtype=torch.int32, device="cuda")
    scratch = torch.empty((BS * K,), dtype=dtype, device="cuda")
    radix_aux_i = torch.empty(
        (BS * RADIX_AUX_BLOCKS_MAX * K,), dtype=torch.int32, device="cuda")
    radix_aux_l = torch.empty(
        (BS * RADIX_AUX_BLOCKS_MAX * K,), dtype=torch.float32, device="cuda")
    return dict(
        logits=logits, seq_lens=seq_lens, indices=indices, preIdx=preIdx,
        scratch=scratch, radix_aux_i=radix_aux_i, radix_aux_l=radix_aux_l,
        K=K, compress_ratio=compress_ratio,
    )


def run_sweep(family, sweep, out_dir):
    """sweep = 'A' (seq-len, BS=1) or 'B' (BS-scaling, N=65536)."""
    skill = SKILLS[family]
    cfgs = list(skill.BETA_CFGS)  # 3 cfgs
    K = skill.K_DEFAULT
    cr = skill.COMPRESS_RATIO_DEFAULT
    flush_buf = torch.empty(L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")

    if sweep == "A":
        Ns = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
        BSs = [1]
    else:
        Ns = [65536]
        BSs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    print(f"[{family}/{sweep}] cfgs={cfgs} Ns={Ns} BSs={BSs} K={K} cr={cr}", flush=True)

    # Pre-synth BS=1 bundles per (cfg, N).
    bundles = {}
    for cfg in cfgs:
        for N in Ns:
            if N <= 2 * K:
                print(f"  [{family}] skip {cfg} N={N} (≤ 2·K={2*K})", flush=True)
                continue
            b = skill.synthesize(N=N, BS=1, cfg_name=cfg, K=K,
                                 compress_ratio=cr, dtype=torch.float32)
            bundles[(cfg, N)] = b
            print(f"  synth {family} {cfg} N={N}: hr={b['meta']['calibration_realised_hr']:.3f}", flush=True)

    rows = []
    # Warmup ALL combos outside the nsys-tagged region.
    for (cfg, N), bundle in bundles.items():
        for BS in BSs:
            args_template_fp32 = make_bench_args(bundle, BS, torch.float32, K, cr)
            for _ in range(2):
                flush_buf.zero_(); torch.cuda.synchronize()
                torch.ops.trtllm.indexer_topk_decode(
                    args_template_fp32["logits"], args_template_fp32["seq_lens"],
                    args_template_fp32["indices"], 1, K,
                    pre_idx=None, heuristic_scratch=None, compress_ratio=cr,
                    radix_aux_indices=args_template_fp32["radix_aux_i"],
                    radix_aux_logits=args_template_fp32["radix_aux_l"])
                torch.cuda.synchronize()

    # bf16/fp16 path does not support numColumns >= kDefaultSplitWorkThreshold
    # (kernel asserts at indexerTopK.cu:1115). Skip those dtypes when N hits
    # the threshold; fp32 + radix-fp32 still run.
    K_SPLIT_THRESHOLD = 200000

    REPS = 10
    for (cfg, N), bundle in bundles.items():
        for BS in BSs:
            # Per-dtype heuristic
            for dt_name in ("fp32", "bf16", "fp16"):
                if N >= K_SPLIT_THRESHOLD and dt_name != "fp32":
                    continue  # bf16/fp16 entry rejects N >= splitWorkThreshold
                dt = DTYPE_MAP[dt_name]
                a = make_bench_args(bundle, BS, dt, K, cr)
                for rep in range(REPS):
                    flush_buf.zero_(); torch.cuda.synchronize()
                    label = f"GVR_{dt_name}|{family}|{cfg}|N{N}|BS{BS}|rep{rep}"
                    with nvtx.annotate(label, color="green"):
                        torch.ops.trtllm.indexer_topk_decode(
                            a["logits"], a["seq_lens"], a["indices"], 1, K,
                            pre_idx=a["preIdx"], heuristic_scratch=a["scratch"],
                            compress_ratio=cr,
                            radix_aux_indices=a["radix_aux_i"],
                            radix_aux_logits=a["radix_aux_l"])
                        torch.cuda.synchronize()
            # Radix fp32 baseline
            a = make_bench_args(bundle, BS, torch.float32, K, cr)
            for rep in range(REPS):
                flush_buf.zero_(); torch.cuda.synchronize()
                label = f"RADIX_fp32|{family}|{cfg}|N{N}|BS{BS}|rep{rep}"
                with nvtx.annotate(label, color="red"):
                    torch.ops.trtllm.indexer_topk_decode(
                        a["logits"], a["seq_lens"], a["indices"], 1, K,
                        pre_idx=None, heuristic_scratch=None,
                        compress_ratio=cr,
                        radix_aux_indices=a["radix_aux_i"],
                        radix_aux_logits=a["radix_aux_l"])
                    torch.cuda.synchronize()
    print(f"[{family}/{sweep}] DONE", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True, choices=["A", "B"])
    p.add_argument("--families", default="flash,pro")
    p.add_argument("--out-dir", default=os.environ.get("OUT_DIR", "data"),
                   help="Output dir for sweep logs / aux artefacts (default: ./data or $OUT_DIR).")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    fams = [f.strip() for f in args.families.split(",")]
    # Single nsys capture window around BOTH families (so --capture-range=
    # cudaProfilerApi captures the whole sweep, not just the first family).
    torch.cuda.cudart().cudaProfilerStart()
    try:
        for fam in fams:
            run_sweep(fam, args.sweep, args.out_dir)
    finally:
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
