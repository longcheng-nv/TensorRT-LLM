# RESUME_PROMPT.md — op40_omni_gvr

## 1-minute context
omni-kernel v2 campaign to further optimize GVR top-K on B200, starting from
PR #16457 head pinned @ e612fc2f38 (vendored src/baseline/, sha1 db7da478/f928b244).
Envelope: 865-cell real-capture grid (§7b of op26_r0 REPORT: Pro K1024 / Flash
K512 / V3.2 K2048, all layers, ISL 4K–1M, BS=1 fp32). Stretch goal gm 1.60× vs
fresh same-node baseline; zero-regression band ratio ≥ 0.97; exact
(value-multiset). KF firewall: never read kf_campaign/, op37_bs_scaling/,
op38_r3v11_bs/, op39_gvr_bsx/. Non-KF op-series ledgers ARE allowed.

## Preflight
- [ ] `git log -1` shows latest iter commit
- [ ] node umb-b200-239, pick an idle GPU (all 8 healthy at kickoff, 30-37 °C)
- [ ] no co-resident driver (poll output-file growth, not ps)
- [ ] progress markers in results/ match ITERATIONS.md
- [ ] `env -u GITHUB_TOKEN -u HF_TOKEN` on every nsys/ncu invocation

## Status
Phase 0 scaffold in progress. Next: harness (task #2) → fresh baseline (task #3)
→ characterization (task #4) → iteration loop.

## Gotchas
- REPORT.html absolute numbers are stale-node — never compare against them.
- PR branch is read-only on this machine (test-slim session elsewhere).
- Baseline files are byte-frozen; variants only in src/variant/.
