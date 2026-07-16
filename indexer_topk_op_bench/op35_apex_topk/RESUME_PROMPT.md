# op35 APEX top-K — RESUME (updated 2026-07-16, post iter-9)

## 1-minute context
Campaign: beat the 6-arm composite frontier (rival_long.csv, op26 report) by
~1.5x geomean with ONE new algorithm (no per-case dispatch). H0 = APEX-FR:
stratified-sample -> threshold band -> single filtered pass -> tiny exact tail.
Feasibility PROVEN: frontier = 3.17x above info floor (notes/floor_map.txt);
filter pass now at tax 1.0-1.17 vs pure read (except BS1024-smallN 1.31).
Current frontier/filter-pass margins: BS1 3.1-4.25x · BS32 1.99x · BS256 1.69x
· BS1024/65k 1.31x. Band math: stratified-jittered sampling 0 miss/3312 real
trials, admit p95 <= 2.5K (results/rung0_band.csv).

## Preflight
- node b200-038 (8 GPU, all cool); env: PYTHONNOUSERSITE=1 (nvshmem fix)
- build: torch cpp_extension, BUILD_DIR=/tmp/op35_build (mkdir first!)
- git log -1 must show iter9 commit; kernels in src/floor_probe.cu (v10 = best)
- rival frontier numbers: ../op26_r0_upstream_port_report/rival_long.csv
- real loaders: ../harness/real_data_v4cap.py / real_data_v32.py (slim caches warm)
- nsys artifacts to /tmp only; env -u GITHUB_TOKEN -u HF_TOKEN for profilers

## Next work items (in order)
1. APEX kernel v0 E2E: sample+threshold phase (stratified-jittered, redundant
   per-CTA broadcast at small BS; per-row at cpr=1) + v10 filter + last-CTA tail
   (select K among <=segcap*nseg candidates from per-warp segments, using
   n_hi(count>=t_hi) to shortcut; write final indices) + miss/overflow retry pass.
   Grid policy: cpr by (BS,N) ~ {BS=1: 148; BS<=8: 148/BS; else 1}, NT=1024 for
   BS>=32 else 512.
2. Exactness gate 3-track: synth (seed-policied), real captures (all 25 shapes),
   adversarial (1-2ulp tie bands, bf16 plateaus, degenerate hint) — tie-aware
   value-multiset vs torch.topk.
3. BS1024-smallN lever: persistent multi-row-per-CTA (wave quantization).
4. 16-bit dtypes: reuse fp32 pipeline w/ half2 loads (2 rounds not needed — same
   sampling band works; measure).
5. Full-grid L1 sweep vs frontier (reuse op26 rival_harness protocol) -> L2 nsys
   ship verdicts (single-GPU paired, x3 batches) -> REPORT.
6. Hint fusion (optional, after sampling-only version ships): hint quantiles as
   extra rungs; must NOT dispatch on hit-rate (memory: not observable at inference).

## Gotchas
- ballot/any_sync in ragged tails: keep uniform iters + predication (illegal
  access incident iter-0).
- static smem cap 48KB (v8 64KB compile fail).
- NVTX host ranges close before async kernels: match nsys kernels by NAME+order.
- counts layout v10: [row * (2+nseg)]; cnt[0]=c_hi (atomic), cnt[2+seg]=per-seg
  count (>segcap = overflow -> retry).
- iter scripts: scripts/iter{1..9}_nsys.py; extractor pattern in each.
