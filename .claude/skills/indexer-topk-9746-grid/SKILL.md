---
name: indexer-topk-9746-grid
description: >
  Run the canonical 886-cell x 11-BS = 9,746-case DSv4/DSv3.2 indexer top-K
  operator campaign on real decode captures: shard across B200 GPUs under
  nsys, cold-L2 same-rep pairing, per-case tie-aware exactness, per-repetition
  raw times, then a bucketed verdict. Use when asked to "benchmark a top-K arm
  on the full grid", "compare against production radix / SGLang v2 /
  FlashInfer", "re-measure after a kernel change", or to add a new arm to the
  DATA-12 style evidence set.
tags: [top-k, indexer, dsv4, dsv3.2, b200, nsys, benchmark, gvr, sglang, flashinfer]
license: LicenseRef-NvidiaProprietary
metadata:
  author: NVIDIA Corporation
  established: 2026-07-30
  provenance: op43_bsx_cutedsl radix + external campaigns (PR #16877)
---

# Indexer top-K 9,746-case grid campaign

The reusable protocol first used for the PR #16877 evidence set. One command per
host shards the grid across GPUs; everything is idempotent, resumable and
splittable across machines. The **protocol, data grid, and reporting rules are
head-independent**; only the per-arm entry points (§2 "Current entry points")
must be re-pointed to the kernel head you are measuring.

## Example prompt to invoke this skill

> Run the indexer-topk-9746-grid campaign on the B200 host. Benchmark the current
> in-tree GVR V2 self-sampling top-K (`selfsampling_topk_run_varlen`) against the
> production CUDA radix (`torch.ops.trtllm.indexer_topk_decode`) over the full
> 886-cell × 11-BS = 9,746-case real decode-capture grid. Point the driver's
> worktree (`WT`) at my built head at `<abs/path/to/worktree>`, shard cells across
> GPUs 0–6 under nsys with cold-L2 same-rep pairing, check tie-aware value-multiset
> exactness vs `torch.topk` per case, then emit the bucketed verdict **split at
> N=4096** with per-segment (flash/pro/v3.2) geomean, floor, and slow-case census.
> Confirm every case is exact before reporting any speedup.

Narrower variants: "just the V3.2 segment (427 cells) vs SGLang v2 + FlashInfer",
"re-measure the radix arm after my kernel change and diff against the last run's
`radix_cases.csv`", or "add a new arm `<name>` calling `<entry>` and fold it into
the existing evidence set".

## 0. Coordinates

```
BENCH=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench
OP43=$BENCH/op43_bsx_cutedsl          # drivers, shard scripts, results
CAP=/home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/E2E_exp/indexer_decode_capture/data
```

## 1. The grid — 886 cells x 11 BS = 9,746 cases

| segment | source | cells | K | cr | cases |
|---|---|---|---|---|---|
| v4 flash | `$CAP/flash/ISL_*/layer_*` | 9 ISL x 21 layers = 189 | 512 | 4 | 2,079 |
| v4 pro | `$CAP/pro/ISL_*/layer_*` | 9 ISL x 30 layers = 270 | 1024 | 4 | 2,970 |
| v3.2 | `$CAP/v32/ISL_*/layer_*` | 7 ISL x 61 layers = 427 | 2048 | 1 | 4,697 |

- BS list: `1,2,4,8,16,32,64,128,256,512,1024` (11 values), dtype fp32.
- ISL rungs: v4 `4k…1024k` (9), v3.2 `4k…256k` (7). `N` is the *compressed*
  width: v4 divides tokens by cr=4, v3.2 does not.
- Cell naming used by the drivers: `v4_<model>_<isl>_L<NN>` and `v32_<isl>_L<NN>`.
  The op42-era grid (865 cells) uses `<model>_<isl>_L<NN>` and is a strict
  SUBSET (it lacks v3.2 L00–L02); strip the `v4_` prefix to cross-reference.
- Cell list generator + committed list: `$OP43/results/radix_cells.txt` (886),
  `$OP43/results/ext_cells.txt` (v3.2-only, 427).
- Loaders: `$BENCH/harness/real_data_v4cap.py`, `real_data_v32.py`. Both take
  `get_bundle(model, isl, layer, "fp32")` and return identical tensors for
  every driver, so different campaigns are comparable by construction.

**Before trusting any run**: confirm each ISL dir has `manifest.json` and
`_COMPLETE`, and check the capture mtime. V3.2 was re-captured 2026-07-29
(L0–L2 added AND L3–L60 values changed) — measurements older than that are on
different inputs and CANNOT be paired with newer ones.

## 2. Run a campaign

Drivers (copy the closest one and **re-point the arm to the head under test** —
see "Current entry points" below; the committed `ab_radix.py`/`ab_ext.py` were
pinned to the PR #16877 head and its interfaces have since changed):

| driver | arms | note |
|---|---|---|
| `$OP43/scripts/ab_radix.py` | `gvrsh` (GVR production entry) + `radix` | production CUDA radix via `torch.ops.trtllm.indexer_topk_decode` |
| `$OP43/scripts/ab_ext.py` | `gvrsh` + `sg2` + `fi` | `--arms` selects a subset; SGLang v2 plan+transform, FlashInfer `top_k` |

### Current entry points (verify against the head you build; adapt the arm)

The op interfaces evolved after the two-level-dispatch / CUDA-heuristic-removal
work (PRs #18446/#18702). As of that lineage:

- **radix arm** — `torch.ops.trtllm.indexer_topk_decode(logits, seq_lens, indices,
  next_n, index_topk, compress_ratio=cr, radix_aux_indices=aux_i,
  radix_aux_logits=aux_l)`. The old `pre_idx=` / `heuristic_scratch=` kwargs were
  **removed** (they no longer exist — passing them raises). Everything else is
  unchanged; supply worst-case aux buffers (`kMaxBlocksPerRowDecode = 10`) so the
  split-work tier is not rejected.
- **GVR V2 arm (`gvrsh`)** — hint-free self-sampling, the current production decode
  entry:
  `from tensorrt_llm._torch.cute_dsl_kernels.blackwell.top_k import selfsampling_topk_run_varlen`
  then `selfsampling_topk_run_varlen(logits, seq_lens, out, next_n=1,
  compress_ratio=cr, max_seq_len=N*cr)`. It is hint-free — **drop the `preIdx`
  argument** the old `gvr_framework.gvr_topk` took. `seq_lens` and `max_seq_len`
  are in **KV-token space** (multiply the compressed width by `cr`). The engine
  gates on fp32 + `stride(0)%4==0` + 16-byte alignment, which the fp32 captures
  already satisfy; on a non-fp4/unaligned buffer it silently falls back to radix,
  so assert the gate in the driver.
- The old `gvr_framework` / `gvr_topk_decode_bsx_dispatch` modules may not exist
  on the current head — replace those imports with the `selfsampling_topk_run_varlen`
  entry above. For V1 (temporal-hint) arms, use `enable_heuristic_topk=True,
  use_self_sampling_topk=False` and the corresponding hinted entry.

Shard + launch (per host):

```bash
cd $OP43
bash scripts/launch_radix_host.sh 0 1 2 3 4 5 6      # or launch_ext_host.sh
# progress
ls results/nsys/ | grep -c '^radix_.*\.done$'         # target 886
grep -h FAIL logs/radix_$(hostname -s)_g*.log
```

- Round-robins cells across the listed GPUs, one `nsys profile` per `CHUNK`
  (default 6) cells, `setsid nohup` per GPU.
- Done-markers `results/nsys/<TAG>_<CELL>.done` live on shared NFS ⇒ re-running
  the launcher only picks up what is missing, and a second machine can join by
  running the same command.
- Re-run the launcher after failures; it is safe and skips completed cells.

Wall-clock reference: 886 cells x 2 arms ≈ 2h20m on 7 B200s; 427 cells x 2
arms ≈ 38 min.

## 3. Measurement contract (do not silently change)

- `sweep_nsys.measure_cell` emits NVTX ranges `{c,w}|<arm>|<cell>|BS<n>`.
- **cold-L2**: 512 MB evict *outside* the range before each timed call —
  this is the canonical number. **warm-L2**: no evict. warmup 10 (untimed),
  then 10 cold + 5 warm reps.
- Host prep (tensor slicing, seq_lens fill, closure binding) happens ONCE per
  (cell, BS) outside the timed region; every arm gets the identical tensors.
- Pre-warm every arm's JIT/compiled variant for the shape before timing.
- Exactness per (cell, BS, arm): tie-aware **value-multiset** equality against
  `torch.topk` — never index equality (ties make index sets non-unique).

## 4. Parse + verdict

```bash
# parallel parse (nsys stats is the bottleneck; 6 shards is a good default)
ls results/nsys/radix_*.nsys-rep | split -n r/6 -d - /tmp/shard_
for i in 0 1 2 3 4 5; do
  python3 scripts/parse_radix.py --rep $(tr '\n' ' ' < /tmp/shard_0$i) \
    --cases results/cases_s$i.csv --reps results/reps_s$i.csv &
done; wait
# merge (header once) then verdict
python3 scripts/radix_verdict.py        # or ext_verdict.py
```

Two tables per campaign:

- `*_cases.csv` — one row per (cell, BS): absolute us per arm (cold+warm),
  speedup, exactness per arm, dispatch route, shape metadata (`K/N/Npad/cr/hit`).
  Mean us = `nvtx_kern_sum` (summed kernel ns / instances).
- `*_reps.csv` — one row per (cell, BS, arm, mode, rep) from
  `nsys stats --report nvtx_gpu_proj_trace`. **Always keep this**: papers and
  reviewers ask for per-repetition raw times, and within-pair RSD (~0.8% here)
  is what makes the aggregate auditable.

`ext_verdict.py` also emits bucketed distributions
(`<0.50 / 0.50–0.70 / … / ≥3.00`) — report these alongside gm, never gm alone.

## 5. Reporting rules learned the hard way

- **Split by provenance.** Same-host/same-day/same-input segments and cross-run
  segments go in separate tables. Never merge them into one "9,746-case" claim
  without saying which is which.
- **Never multiply ratios across reports.** If arm A was measured against
  baseline X in campaign 1 and arm B against X in campaign 2, `A/X ÷ B/X` is an
  inference, not a measurement — label it as such or re-measure.
- **Always report the floor and the slow-case census**, not just gm.
- **N < 4096 is outside production GVR.** `kSeqSmallDefaultForK()` in
  `cpp/tensorrt_llm/kernels/indexerTopK.cu` returns 4096; the dispatcher never
  routes shorter rows to GVR (measured 0.55–0.84x there). A DSL-entry campaign
  calls GVR unconditionally, so split the verdict at N=4096 and say so.
- Environment manifest (`radix_env_manifest.json` pattern): GPU/driver/CUDA/
  nsys/torch versions, each arm's entry point, the built `.so` path + sha256 +
  mtime, the build-tree SHA, and a diff check that the kernel source in the
  build tree matches the head under test.

## 6. Gotchas

- **nsys sqlite/rep files embed environment variables including tokens.** Run
  with `env -u GITHUB_TOKEN -u HF_TOKEN`, and gitignore `results/nsys/` BEFORE
  the first commit.
- **Slim-cache staleness.** `real_data_v32.py` keys its cache on the capture
  generation + layer set; if you add a loader, do the same or a recapture will
  be silently served from the old cache.
- **A GPU can be held by an invisible tenant.** `nvidia-smi` may show 0 MiB/0%
  while `torch.zeros(8, device='cuda')` raises `cudaErrorDevicesUnavailable`
  (other-namespace process). Probe every GPU with that one-liner before
  launching, and exclude the bad ones; transient `device busy` chunk failures
  are recovered by re-running the launcher.
- **Don't `pkill -f` broad patterns** — monitor loops matching `run_*.sh` will
  match themselves. Use `pgrep -f 'nsys profile.*ab_<x>'` to test liveness.
- FlashInfer scans the whole row (no `seq_lens`), so its input needs a fresh
  `-inf`-padded `[BS, Npad]` copy built outside the timed region. SGLang v2's
  complete op is plan+transform with plan timed every rep (no cross-layer
  amortization).

## 7. Reference results — historical baseline (2026-07-30, umbriel-b200-027, B200, PR #16877 head)

These are the *original* #16877-head numbers, kept for provenance and as a sanity
anchor for the protocol. They are **not** current-head measurements — the GVR
entry point and thop schema have since changed (§2); re-measure against your head
before quoting a speedup.

| comparison | n | gm | floor | GVR faster |
|---|---|---|---|---|
| vs production CUDA radix | 9,746 | 3.394x | 0.647x | 98.4% |
| … restricted to N>=4096 (production GVR region) | 8,624 | 3.856x | 1.241x | 100% |
| vs SGLang v2 plan+transform (V3.2, same day) | 4,697 | 1.231x | 0.237x | 72.6% |
| vs FlashInfer 0.6.14 (V3.2, same day) | 4,697 | 1.540x | 0.222x | 89.0% |

All arms exact on every measured case. Full artifacts under `$OP43/results/`:
`radix_cases.csv`, `radix_reps.csv.gz`, `radix_verdict.{txt,json}`,
`radix_env_manifest.json`, `ext_cases.csv`, `ext_reps.csv.gz`,
`ext_verdict.{txt,json}`, `rival_matched_summary.json`.
