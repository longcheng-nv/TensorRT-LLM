# op21 B300 cross-check RELAUNCH — for ANY B300 host (v2, 2026-07-06)

Context: the first B300 run (umb-b300-dp-185) completed the fp32 grid
(verdict already computed: gm 1.268, 17/17) and died 11/34 into the 16-bit
sweep. Its partial reps are being archived; this relaunch runs the FULL
51-cell grid fresh so all three dtypes share ONE measurement axis on the
new node. Everything lives on shared NFS — nothing else to stage.

## Option A — bare shell (no Claude session needed on the B300 host)

Run steps 1–6; a B200-side session can poll NFS and do verdicts + writeup.

```bash
# 1. checkout + env
cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/op21_gvr_prod
git log --oneline -1        # expect c989bad434 "[op21 handoff]" or later
python3 -c "import torch, cutlass; print(torch.cuda.get_device_name(0))"
which nsys

# 2. GPU preflight — pick a GPU idling <50C (broken-cooling lesson)
nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used --format=csv

# 3. exactness sanity on this silicon (~1 min; expect 9/9 OK, any FAIL => stop)
python3 src/gvr_msc_op.py 4

# 4. archive the dp-185 partial grid (REQUIRED — drivers skip existing files;
#    leaving them would silently mix two nodes' axes)
mkdir -p results/nsys/iter10_b300_dp185_partial
mv results/nsys/msa_*.nsys-rep results/nsys/msa_*.sqlite \
   results/nsys/iter10_b300_dp185_partial/ 2>/dev/null; true

# 5. launch both sweeps detached (survives ssh/session death — proven on dp-185).
#    Replace GPU=0 with the healthy index from step 2.
GPU=0 nohup bash -c \
  'bash scripts/drive_nsys_iter2.sh && bash scripts/drive_nsys_16bit.sh' \
  > /tmp/op21_b300_sweep.log 2>&1 &

# 6. monitor (51 reps total: 17 fp32 + 17 bf16 + 17 fp16; ~30-45 min incl. JIT)
watch -n 60 'ls results/nsys/msa_*.nsys-rep 2>/dev/null | wc -l; tail -2 /tmp/op21_b300_sweep.log'

# 7. verdicts (any host with nsys + the NFS mount can run these)
python3 scripts/nsys_verdict.py msa fp32 B300
python3 scripts/nsys_verdict.py msa bf16 B300
python3 scripts/nsys_verdict.py msa fp16 B300
```

Notes:
- The drivers already run `env -u GITHUB_TOKEN -u HF_TOKEN` (nsys embeds
  process env; *.nsys-rep/*.sqlite are gitignored — NEVER commit them).
- Co-tenancy check = output-file growth, not nvidia-smi (namespace-blind).
- Cross-node consistency: the fresh fp32 verdict should reproduce dp-185's
  per-cell ratios (gm 1.268, 17/17) within ~3%; a bigger shift = different
  axis silicon — note it in the results doc, ratios remain canonical.
- MEASUREMENT-ONLY: no kernel or dispatch edits on this host.

## Option B — full Claude session on the B300 host

Paste `op21_gvr_prod/B300_PROMPT.md` into a fresh Claude Code session on the
B300 host, with ONE amendment prepended:

> Amendment (2026-07-06): the dp-185 partial grid must first be archived to
> `results/nsys/iter10_b300_dp185_partial/` (fp32 17 reps+sqlite, bf16 11
> reps). The B200 grids are ALREADY archived (iter7_msa_b200/,
> iter9_16bit_b200/) — skip that step. Then run BOTH drivers (fp32 too) so
> all dtypes share this node's axis, compute the three verdicts, write
> B300_RESULTS.md (fp32 dp-185 reference: gm 1.268 win 17/17; per-cell
> table in the B200 session's transcript or recompute from the archive via
> `OP21_NSYS_DIR=results/nsys/iter10_b300_dp185_partial python3
> scripts/nsys_verdict.py msa fp32 B300`), commit `[op21 B300]` per the
> prompt's step 7.

## Division of labor if a B200-side session is still alive

The B200 session only needs the reps to appear on NFS — it can run step 7's
verdicts, write B300_RESULTS.md, and commit. In that case the B300 host only
runs steps 1–6 (Option A) and nothing else.
