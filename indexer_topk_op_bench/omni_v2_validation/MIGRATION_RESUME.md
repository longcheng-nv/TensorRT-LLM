# MIGRATION_RESUME — omni-kernel v2 skill validation (node handoff)

> **⛔ OBSOLETE — CAMPAIGN COMPLETE 2026-07-13 @f5364a2d6d (umbriel-b200-035).**
> Everything below was executed and closed out. Do NOT re-run the resume flow.
> - Tier B: SHIP at iter 3/5 (commits 566fadda5d / 8b7849a08b / d68bb16b47);
>   re-anchor done (21.82@027 → 21.17@035).
> - Tier C: audit 22 PASS / 2 DEGRADED / 1 N-A; P1-P5 adjudicated.
> - Authoritative record: `VALIDATION_REPORT.md` (same directory).
> - Open follow-ups (new campaigns, not resumes): P3 human-pivot control arm;
>   P6 parallel orchestration; split-row small-T attack; TileIR/TMA vs WALLS #3;
>   v3 hooks (gate-per-impl-file, RESUME+COST per commit).

> Written 2026-07-13 ~02:15 on umbriel-b200-027 (session timing out).
> Workspace is NFS scratch — visible from any computelab node. All durable
> state is in git; this file is the five-part relay per the v2 skill protocol.

## 1. Context (1 minute)

Task: validate the omni-kernel **v2 skill** (`indexer_topk_op_bench/
gvr_agent_retrospective/skill_v2_draft/`, live via `.claude/skills/omni-kernel`
symlink) against OMNI_KERNEL_V2_GAPS.md §二 (P1-P5). Plan + progress:
`omni_v2_validation/VALIDATION_REPORT.md`.

- **Tier A (P4 scripts) DONE** @c5dad99cb6 — 9/9, 4 script bugs found & fixed
  (nsys csv comma-split, ncu heredoc dead-code, set -u, first-kernel footgun).
- **Tier B (P1/P2/P5) IN FLIGHT** — mini-campaign `tierB/rmsnorm_campaign/`
  vs flashinfer.norm.rmsnorm per `tierB/KICKOFF.md` (budget ≤5 iters).
  - iter 0 committed @74c717613b: incumbent is single-HBM-pass (traffic levers
    VOID, NCU-locked); win region = T≤256 latency regime (headroom 1.73×/1.31×/
    1.11× at T=1/16/256); T≥4096 = saturated margin defense; anchor:
    incumbent T=4096 nsys = 21.82 µs ± 3% (VALID ON 027 ONLY — re-anchor!).
  - iter 1 INTERRUPTED mid-flight (campaign agent killed for migration).
    `src/candidate_triton.py` exists (committed at checkpoint @HEAD).
    Last unverified partial from the agent: **T=4096 candidate 0.933×**
    (loses 6.7%); small-T cells NOT yet measured. Treat as hypothesis, not
    result — it is not in ITERATIONS.md and carries no nsys record on disk.
- **Tier C (audit + final report) NOT STARTED** — rubric ready:
  `tierB/AUDIT_CHECKLIST.md` (25 items, evidence-based scoring).

## 2. Preflight checklist (new node)

- [ ] `git log --oneline -3` shows the migration checkpoint commit at HEAD
      (message starts `[omni-v2-validate] MIGRATION checkpoint`).
- [ ] Old node dead: this handoff assumed umbriel-b200-027 timed out. If in
      doubt, watch `tierB/rmsnorm_campaign/` for file growth for ~2 min before
      touching it (ps/nvidia-smi are namespace-blind) — dual-driver = corruption.
- [ ] Env probe: `python3 -c "import torch,triton,flashinfer; print(torch.__version__, triton.__version__, flashinfer.__version__, torch.cuda.get_device_capability())"`
      → expect torch 2.11.x / triton 3.6 / flashinfer 0.6.x / (10,0) B200.
      `which nsys ncu`. If flashinfer missing, the campaign cannot continue —
      fix env first (incumbent must be the real production kernel).
- [ ] GPU pick: idle GPU with idle temp <50 °C (`nvidia-smi -q -d TEMPERATURE`);
      memory blacklist: 019-GPU0, 035-GPU0, 036-GPU1 have broken cooling.
      Export CUDA_VISIBLE_DEVICES=<picked> consistently.
- [ ] **Re-anchor (MANDATORY)**: absolute µs never transfer across nodes.
      Re-run the anchor on the new node and REWRITE the expected value in
      `rmsnorm_campaign/RESUME_PROMPT.md` §2 + PLAN.md:
      ```bash
      cd indexer_topk_op_bench/omni_v2_validation/tierB/rmsnorm_campaign
      CUDA_VISIBLE_DEVICES=<g> TOKENS=4096 python3 ../../../gvr_agent_retrospective/skill_v2_draft/scripts/nsys_verdict.py \
        --impl src/incumbent.py --kernel-regex 'RMSNormKernel|rmsnorm' --batches 3 --launches 20
      ```
      All iter-0 ratio conclusions (headroom table) were same-batch ratios —
      they survive the move; only the absolute anchor value must be replaced.
      The copy-ceiling table is cheap to re-derive via src/probe_copy.py if the
      new arch differs (B300: expect different saturation points).

## 3. Work split

Single session, sequential: finish Tier B → Tier C audit → final report.
No sharding needed at this scale.

## 4. Resume commands (byte-exact)

New Claude Code session opening prompt — paste this:

```
继续 omni-kernel v2 skill 验证任务。先读:
1. indexer_topk_op_bench/omni_v2_validation/MIGRATION_RESUME.md  (本文件, 做完 §2 preflight)
2. indexer_topk_op_bench/omni_v2_validation/VALIDATION_REPORT.md (总计划+Tier A 结论)
3. indexer_topk_op_bench/omni_v2_validation/tierB/rmsnorm_campaign/{RESUME_PROMPT,ITERATIONS,PLAN}.md

然后:
(a) 重跑 anchor 并改写 RESUME_PROMPT.md 里的期望值 (跨节点绝对 µs 不可沿用);
(b) 以 general-purpose subagent 恢复 Tier B 战役 — prompt 模板在
    MIGRATION_RESUME.md §5, 从 iter 1 继续, 预算剩 ≤4 iterations;
(c) 战役收尾后按 tierB/AUDIT_CHECKLIST.md 用磁盘证据逐项审计 (25 项,
    PASS/DEGRADED/SKIPPED), 把 Tier B 结果 + P1-P5 判决写回
    VALIDATION_REPORT.md, commit (git commit -s, 加 Claude 附署 trailer)。
KICKOFF.md 的 objective triple 与 pre-authorized negative conclusion 不变:
flashinfer 若保持最优, 明说即是成功结局。
```

## 5. Campaign-agent relaunch prompt (for step b)

Reuse the original Tier-B agent prompt verbatim (it is reproduced in the git
history / VALIDATION_REPORT context) with these deltas:
- Prepend: "RESUME, not fresh start: read rmsnorm_campaign/RESUME_PROMPT.md,
  ITERATIONS.md, FALSIFIED.md, WALLS.md first. iter 0 is done; continue at
  iter 1. An unverified partial exists: candidate T=4096 ≈ 0.933× vs
  incumbent — re-measure via nsys_verdict before trusting it."
- Replace CUDA_VISIBLE_DEVICES=2 / node name with the new node's pick.
- Replace the anchor expectation with the §2-re-anchored value.
- Budget: ≤4 remaining iterations.

## Known gotchas (carried from both tiers)

- ncu -k needs `regex:` prefix → KERNEL_REGEX="regex:RMSNormKernel".
- tensor.copy_() lowers to DtoD memcpy — invisible in cuda_gpu_kern_sum;
  probe ceilings with torch.mul(out=).
- impl modules read the cell from the TOKENS env var (default 4096).
- L1 graph timing inflates small-T cells ~7-18 µs launch bias; small-T claims
  only via nsys (this is skill M1 escalation, validated in iter 0).
- nsys sqlite/nsys-rep embed env tokens — already gitignored; never commit.
- The skill scripts were FIXED in Tier A @c5dad99cb6 — if the new node's
  checkout predates it, `git pull` first; running the old nsys_verdict.py
  silently inflates every number by the evictor kernel.
