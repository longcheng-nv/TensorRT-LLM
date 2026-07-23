#!/usr/bin/env python3
"""Phase-4 exactness gate (OmniKernel): criterion by kernel class x three data tracks.

Criteria:
  dense   -- atol/rtol vs reference (fp32 1e-5, bf16 1e-2, fp16 1e-3)
  select  -- tie-aware value-multiset for selection/search kernels:
             sorted(values(out)) == sorted(values(ref)) AND cardinality correct.
             NOT index equality (tie order is nondeterministic under atomics);
             NOT set equality (fails on duplicates at low precision).

Tracks (all provided tracks must pass; a gate is a GATE, not a report):
  synth        -- get_inputs()            (validated generator; torch.randn is
                                           BANNED for low-precision selection
                                           inputs; use a seed policy per cell)
  real         -- get_real_inputs()       (optional; synth-exact != real-exact:
                                           two gate escapes in the source record
                                           were real-data-only)
  adversarial  -- get_adversarial_inputs() (optional; near-tie clusters, boundary
                                           padding, varlen; found a 0/72 escape)

Impl module contract: kernel_fn, reference_fn, get_inputs, and optionally
get_real_inputs / get_adversarial_inputs (each returns a list of input-lists,
or a single input-list).

Usage:
    python scripts/verify_exact.py --impl src/candidate.py --mode select --trials 5
    python scripts/verify_exact.py --impl src/candidate.py --mode dense --dtype bf16
"""
import argparse
import importlib.util
import sys

import torch

TOL = {"fp32": (1e-5, 1e-5), "bf16": (1e-2, 1e-2), "fp16": (1e-3, 1e-3)}


def load_module(path):
    spec = importlib.util.spec_from_file_location("impl", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_dense(out, ref, atol, rtol):
    ok = torch.allclose(out.float(), ref.float(), atol=atol, rtol=rtol)
    return ok, "" if ok else f"max|diff|={float((out.float()-ref.float()).abs().max()):.3e}"


def check_select(out, ref, values=None):
    """Tie-aware value-multiset. If out/ref are index tensors, pass the source
    row via `values` so indices are compared through their gathered values."""
    o = values.flatten()[out.long().flatten()] if values is not None else out.flatten()
    r = values.flatten()[ref.long().flatten()] if values is not None else ref.flatten()
    if o.numel() != r.numel():
        return False, f"cardinality {o.numel()} != {r.numel()}"
    so, sr = torch.sort(o.float()).values, torch.sort(r.float()).values
    if not torch.equal(so, sr):
        bad = int((so != sr).sum())
        return False, f"value-multiset mismatch on {bad}/{so.numel()} entries (vdiff!=0)"
    if values is not None:  # index outputs must also be duplicate-free
        u = out.long().flatten().unique().numel()
        if u != out.numel():
            return False, f"duplicate indices: uniq {u} != K {out.numel()}"
    return True, ""


def normalize(track_inputs):
    if not track_inputs:
        return []
    return track_inputs if isinstance(track_inputs[0], (list, tuple)) else [track_inputs]


def run_track(mod, name, getter, mode, atol, rtol, trials):
    if not hasattr(mod, getter):
        return None
    cases = []
    for _ in range(trials if getter == "get_inputs" else 1):
        cases += normalize(getattr(mod, getter)())
    n_pass = 0
    for i, inputs in enumerate(cases):
        out, ref = mod.kernel_fn(*inputs), mod.reference_fn(*inputs)
        torch.cuda.synchronize()
        if mode == "dense":
            ok, msg = check_dense(out, ref, atol, rtol)
        else:
            vals = inputs[0] if out.dtype in (torch.int32, torch.int64) else None
            ok, msg = check_select(out, ref, values=vals)
        n_pass += ok
        if not ok:
            print(f"  [{name}] case {i}: FAIL — {msg}")
    print(f"  [{name}] {n_pass}/{len(cases)} pass")
    return n_pass == len(cases)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impl", required=True)
    ap.add_argument("--mode", choices=["dense", "select"], required=True)
    ap.add_argument("--dtype", choices=list(TOL), default="fp32")
    ap.add_argument("--trials", type=int, default=5)
    args = ap.parse_args()

    mod = load_module(args.impl)
    atol, rtol = TOL[args.dtype]
    print(f"exactness gate: mode={args.mode} dtype={args.dtype} impl={args.impl}")
    results = {t: run_track(mod, t, g, args.mode, atol, rtol, args.trials)
               for t, g in [("synth", "get_inputs"),
                            ("real", "get_real_inputs"),
                            ("adversarial", "get_adversarial_inputs")]}
    ran = {t: r for t, r in results.items() if r is not None}
    missing = [t for t, r in results.items() if r is None and t != "synth"]
    if missing:
        print(f"  note: tracks not provided by impl: {missing} — the gate is weaker; "
              f"real/adversarial tracks caught the only two escapes in the source record.")
    if all(ran.values()):
        print("GATE: PASS")
    else:
        print("GATE: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
