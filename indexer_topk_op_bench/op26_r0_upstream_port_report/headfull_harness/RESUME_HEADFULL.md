# RESUME — HEAD full-coverage sweep (migration 027 → umbriel-b200-019)

> Written 2026-07-20 ~07:2x UTC. Mission: full REPORT.html-coverage perf
> test of the pushed PR#16457 head @e6fdbfac3d (incl. p4tt, shipped
> K-gate defaults), then per-cell comparison vs REPORT numbers.

## State

- Sweep = refresh protocol, 54 batches (batches_refresh.py), arms
  gvr_base / gvr_pr(=head, via p4f1_harness/gvrpkgprod2 — md5 3396037c ==
  branch file) / op26_r0auto anchor. Harness: THIS dir (ops_refresh.py
  already points at gvrpkgprod2; sweep/driver/parse from refresh clone).
- b200-027 completed 16/54 batches; they are staged with reps in
  `migrate_027/` (see MIGRATE_NOTE.md for exact restore commands).
  8 incomplete batches were deliberately dropped (whole-batch redo —
  partial-jsonl + fresh rep loses NVTX ranges; known trap).
- 027 processes fully killed (GPU compute-apps 0, row counts frozen).

## Steps on 019

1. Env farm (node-local): RESUME_P4TT.md recipe (cutlass450 symlinks,
   PYTHONNOUSERSITE=1, verify cutlass 4.5.0 + make_fragment).
2. **GPU0 has broken cooling on 019 (70C idle — long-standing memory).
   Use GPUs 1-7 only**: run the driver 7-way with W∈0..6 →
   `bash headfull_harness/drive_refresh_shard.sh $W $((W+1)) 7`
   (args: WORKER GPU NWORKERS; the batch split key is W%NW so 7-way is
   self-consistent; .done markers from 027 are skipped automatically).
3. Restore staged results per migrate_027/MIGRATE_NOTE.md BEFORE launching.
4. Smoke first (1 cell, any GPU 1-7): headfull_harness ops import + one
   build_call_rival('gvr_pr', ...) — see the smoke snippet pattern in the
   session transcript / p4f1_harness batteries.
5. After 54/54: `python3 headfull_harness/parse_refresh.py
   /tmp/gvrheadfull/refresh_results` then
   `python3 headfull_harness/compare_headfull.py`.
   Cross-node note: batches are internally paired (3 arms same process),
   so pr/base and anchor ratios are node-clean per batch; the global
   anchor med in compare output mixes 027+019 batches — if p95 splits
   bimodally, split the anchor stats by batch origin (027 tags are the 16
   in migrate_027/) before quoting absolute-time conclusions.
6. Deliverable: per-axis comparison conclusion vs REPORT (§3 synth seqlen,
   §7 synth/real BS grids fp32+16bit, §4 real seqlen) — the user wants the
   comparison verdict FIRST, before any further action.

## Watchouts

- /tmp disk: keep an eye (027 was 96% full; reps ~1.5GB total expected).
- Anchor gate ≤1.15 (refresh convention); per-batch anchor lives in every
  batch via op26 rows vs rival_long.csv.
- Never commit *.nsys-rep (env tokens); migrate_027/nsys_reps is gitignored.
- pkill -f matches your own shell (exit 144) — kill by PID/PGID list.
