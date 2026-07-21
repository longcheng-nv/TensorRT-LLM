# L-J escape-only bracket (tb_escape) — implementation design

Date: 2026-07-21, b200-069. Follows LJ_TAX_DIAGNOSIS.md verdict (c):
warm cells must never pay the ladder tax; the bracket machinery fires
ONLY on base-admission miss (the cold/fat-admission regime, e.g.
flash-512k hit .057, today ~3.5 full-N pass equivalents of refine).

## Shape

New ctor flag `tb_escape=True` (implies the tight_bracket band P3/P4
TRACING but NOT the wide first-pass ladder):

1. P1b + P2 first pass = EXACTLY the base narrow ladder
   (K512 (0.85,)+vseed, K1024 (0.85,0.35)+vseed). Zero warm delta.
2. Base admission accepts (bc >= 0) -> s_lj_i[0]=0, normal single-thr
   P3/P4. Warm cells end here: zero new work, zero new syncs.
3. Base admission MISSES (bc < 0) -> escape sequence replaces the
   fb_fix log-falsi chain:
   a. Re-walk the still-live smem_hist (P1b hist is untouched between
      P1b and band-P3's scratch reuse) with DENSE escape qfracs
      (M_esc rungs, clamped inside the measured (bhi, blo) bracket),
      warp-0 extraction as in phase1b_hspace_rungs (~1us).
   b. ONE block_count_ge_multi pass with M=M_esc (new M override arg),
      caching per-thread columns into the escape smem_ptcnt_multi
      region; cluster-merge as usual (one extra sync round, miss-only).
   c. Bracket-admit over the M_esc counts (same lo/hi rule as tb=on,
      band <= kC) -> fire band P3 (s_lj_i set) + band P4.
   d. If STILL no bracket (plateau/degenerate): seed fb_fix falsi with
      the tightened (now M_esc-point) bracket — strictly better seeds
      than today.

## Decisions — SETTLED (2026-07-21, ablation + tb_debug probes, b200-069)

- D1 RESOLVED — unroll port REJECTED: off_nou3 P3 cyc 0.93-1.08x warm,
  1.12x on the BW cell; the band P3 tax is the dual-threshold classify,
  and the thin ladder sidesteps most of it anyway (tb_thin P3 1.01x on
  the win cell). v1 ships without the unroll port.
- D2 RESOLVED — thin escape ladder: tb_thin kept the FULL cold-cell
  refine kill (P2 0.55x) and beat tb_on net (1.408 vs 1.217).
  tb_esc_qfracs default = (0.85, 0.35, 0.05).
- D3 RESOLVED — the cold win is SEEDED FALSI, not the band: tb_debug on
  flash/512k shows fired=0 for tb_on/tb_thin/esc alike (band never fires
  cold); the rung counts tighten the falsi seed to ~1 pass. The band
  fires on WARM cells (pro/128k fired=1 band=686; flash/128k fired=1
  band=123) — exactly where it is a net tax under always-on, and exactly
  what escape-only avoids (warm rows take the base accept, escape
  dormant — confirmed by zero [esc] prints on warm cells).

## Implementation state (same day)

- gvrpkg37 spliced: ctor flags tb_escape/tb_esc_qfracs, tb_band gates,
  block_count_ge_multi M_ov/vseed_ov, phase1b_escape_rungs (hist
  re-walk), escape admission + M_esc-seeded falsi copy in the miss path,
  band P3 escape column mapping, smem sized max(M_thr, M_esc).
- First silicon: probe 3/3 exact (cold fires escape->falsi; warm never
  enters). PTX default-off proof PASSED (md5 2a25f8fc == pristine, same
  as the lj-era proof). battery_esc.py (E1-E6, forced-miss levers
  r0_qfracs=(0.999,)/r0_vseed=False) running.
- GOTCHA logged: probe's first "OOB crash" was a 2D-logits [1,L] slicing
  bug in the probe itself (base arm crashed identically); kernel was
  never wrong.

## Superseded open questions (pre-ablation text)

- D1 (P3 unroll port): first-cell off_nou3 shows unroll contributes ~0
  on warm cs=4 (P3 cyc 0.93x, wall unchanged) — hypothesis REJECTED
  there; final call on the BW-rich flash/512k BS512 off_nou3 row. If
  rejected there too, band-P3's 1.2-1.6x cyc is the dual-threshold
  classify itself -> accept it (miss-only cost) and skip the port.
- D2 (escape ladder width M_esc): tb_thin flash/512k row answers
  whether a thin (M=4) ladder keeps the cold win. BW arithmetic
  @BS512/131k (~45us/pass BW floor): base miss chain 159.5us; tb=on
  wide-pass P2 96.5us; escape = narrow pass (~50) + M_esc pass
  (M4 ~55 / M10 ~96) -> M_esc must stay thin-ish for the win to beat
  tb=on's. If tb_thin does NOT fire the bracket on the cold cell,
  widen from the hist tail side only.
- D3 (win mechanism audit): battery (43a92cc70c) says the bracket did
  NOT fire on cold flash/512k (fell to refine) yet nsys+phase shows
  tb=on P2 halves AND band P3 runs there. Contradiction to resolve
  with a tb_debug run on the win cell BEFORE coding: if the tb win is
  really "wide rung counts tighten the falsi seed", the escape's step
  (c) band-fire is optional and step (d) seeded-falsi is the payload —
  the design still stands, only the P3/P4 band reuse becomes
  secondary.

## SMEM & ship-safety

- Escape per-thread column cache: M_esc * num_threads * 4B (M4/T1024 =
  16KB, M10/T1024 = 40KB). fp32 (op37 axis) has headroom (tb=on showed
  no occupancy cliff); 16-bit configs have a known cliff precedent
  (vseed round-1 note @L5217) -> gate tb_escape OFF for 16-bit until
  separately re-measured.
- Default-off PTX must stay byte-identical (same discipline as
  tight_bracket; verify md5 vs pristine).
- cs>1: escape adds ONE cluster sync round, miss-only; the fb_fix
  chain it replaces paid one per falsi iteration.

## Verdict plan

battery (S1 grid + tie fixtures + real cells) -> PTX proof ->
132-cell nsys A/B (gvr_pr vs gvr_esc [vs sglang_v2 anchor]) on the
op26 §7b real axis; zero-regression bar on the BS32-128 win block;
target = flash-512k BS>=256 lifted toward tb=on's 1.24-1.33x vs pr at
<=1% everywhere else.
