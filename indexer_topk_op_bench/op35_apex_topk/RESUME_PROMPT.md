# op35 APEX top-K — RESUME (updated 2026-07-16, post iter-13, b200-072)

## 1-minute context
Campaign: beat 6-arm composite frontier (rival_long.csv) ~1.5x geomean with ONE
new algorithm. H0 APEX-FR is BUILT and EXACT: v3 fused/split kernels in
src/apex_topk.cu + src/apex_op.py (pick_config policy). Exactness: 347/347
envelope cells (incl. all real captures) + 32/32 screen (oddN/bf16-plateau/
const-row/self-clean). Full-envelope position (iter13, fp32):
**geomean frontier/apex 0.468** — kernel loses ~2x overall. Wins only at
BS128-1024 x N131-262k (up to 0.94; earlier variants hit 1.16-1.25 before
safety margins widened). N<=16k is 0.30-0.42 everywhere. Ship bar (1.5x) is
FAR; see ITERATIONS iter12-13 honest assessment.

## Architecture (v3)
- fused<NT=512> (BS<=16, cpr=max(1,148//BS)): phase A (reg-resident stratified
  float4 samples -> one-pass 2048-bin window hist -> t_lo) -> v10-style filter
  staging admits {bits,idx} int2 into per-warp smem regions (96KB dyn) ->
  end-of-CTA flush (1 global atomic reserve on cnt[1]) with graceful spill on
  region overflow -> last-CTA (ticket) tail: coalesced gather -> smem radix
  (byte-skip via kmin/kmax) -> tie-aware ballot emission.
- split (BS>=32): k_thr (NT512, grid rows) / k_filter<1024> (cpr=max(1,256//BS),
  v10 registers, 2 CTA/SM) / k_tail (NT512, dyn smem = tail_cap*8).
- fallback ladder: M<=tail_cap staged; tail_cap<M<=GCAP(32768) big-M path
  (radix/emit direct from global cand, ~M cost); M<K or M>GCAP full-row radix
  (exact, rare). Band: i_lo = ceil(r0+6*(2*sig_iid))-1 (x2 = float4-quad
  spatial correlation, VALIDATED on op26 synth), s = clamp(pow2(lam*N/K),2048,
  8192), lam={512:4,1024:8,2048:16}.

## Preflight
- b200-072 (8 GPU all cool; read anchors == 038: 3.52/3.65/4.66/10.1/54.8/54.0)
- PYTHONNOUSERSITE=1; BUILD_DIR=/tmp/op35_build (mkdir first)
- git log -1 ~ "[op35 iter13] full-envelope position"
- quick gates: scripts/iter10_screen.py (exactness), scripts/iter10_nsys.py +
  parse_iter10.py (6 anchors), scripts/iter13_sweep.py + iter13_report.py
  (full envelope; nsys -t cuda,nvtx; ~2 min)
- probes: iter10_phase_probe.py (mode 1/2/3 + in-kernel globaltimer dbg[BS,8]);
  iter11_ncu_target.py (single-shot NCU)
- nsys/ncu artifacts: /tmp or results/ (env -u GITHUB_TOKEN -u HF_TOKEN);
  results/iter13/*.nsys-rep are gitignored? CHECK before commit (token leak!)

## Next work items (leverage order)
1. small-N full-scan mode (N<=16-32k, 68 cells at 0.30-0.42): s=N conceptually —
   window hist over the FULL row (2 passes over L2-resident row), exact counts,
   no miss, M~K+bin-width. Route in pick_config; same kernel family.
2. tail select: replace 256-bin byte rounds with one-pass 2048-bin window
   (kmin/kmax known from gather) + one refine pass -> exact u_kth in 2 passes;
   tail radix currently 3-9us for M~3K (pass/barrier-bound, NOT flops).
3. k_thr: drop minmax dependency (fixed top-12-bit window pass1 + refine pass2)
   or fuse minmax into sampling load pass better; thr still ~8us/CTA-wave.
4. sig x2 is blunt: consider per-quad-decimated order stats (use quad max +
   corrected quantile math) to shrink M without miss risk.
5. THEN re-decide: if overall lands <1.0, the 1.5x bar needs either 16-bit
   envelope (frontier weaker there? measure 3 anchor cells bf16 first) or a
   structurally different BS1/small-N attack (single-wave cooperative smem-row
   cache) — op34 falsified BS1 miracles, keep expectations bounded.
6. Exactness gate track-3 (adversarial: 1-2ulp tie bands, denormals, inf rows)
   still pending; tracks 1-2 green via iter13 (347/347) + screen.

## Gotchas (new since iter9)
- match_any/ballot/any_sync: EVERY call site must be warp-uniform; predicated
  single-call-site pattern (hist_add(bin, sel)) — a divergent if/else pair of
  hist_add calls deadlocks/corrupts.
- workspace cand row stride == GCAP (NOT PAIR_CAP) — the iter13 exactness bug.
- counts self-clean happens in TAIL; mode=1/2 probes leave counts dirty —
  probe scripts must zero counts+tickets between calls.
- __launch_bounds__(NT,2048/NT) on the fat fused kernel FALSIFIED (hot-loop
  spills, BS32 A+B 58->84us). Register relief must come from kernel SPLITTING.
- op26 synth data is spatially clustered: float4-quad samples correlated ->
  band width x2-3 vs IID math; any sampling change must re-validate miss rate
  on op26 synth worst + real (not torch.rand!).
- event-axis probes carry ~8.4us launch floor; nsys span is the only ship axis.
- effective SM clock during micro-kernels ~0.5-1GHz (not DVFS-idle artifact;
  hot-loop test 55 vs 61us) — pass/barrier counts cost ~3x what boost math says.
EOF
git add RESUME_PROMPT.md && git commit -q -m "[op35] RESUME updated to post-iter13 state" && git log --oneline -1