# RESUME_PROMPT — rmsnorm_campaign (refresh at EVERY commit)

<!-- PASTE-READY PROMPT (copy from here) -->

## 1. Context (1 minute)
Campaign: rmsnorm_campaign (omni-kernel v2 Tier-B trial). Objective: beat/tie
flashinfer.norm.rmsnorm on hidden=7168 bf16, T grid {1,16,256,4096,16384}, B200.
State: CAMPAIGN CONVERGED — SHIP at iter 3 (of 5). Final artifact
src/candidate_dispatch.py (1-rule dispatch: T<=512 triton arm / T>512 flashinfer).
Final grid (nsys): 1.0372/0.9957/1.0048/1.0013/0.9986, geomean 1.0074, worst 0.9957
— ship rule met. See ITERATIONS.md "FINAL CAMPAIGN VERDICT". Nothing to resume;
any follow-up campaign starts from the un-spent levers listed in iter 3 (split-row
small-T attack; TileIR/TMA vs WALLS #3).
Read PLAN.md + ITERATIONS.md tail + FALSIFIED.md before proposing anything.

## 2. Preflight checklist
- [ ] `git log -1 -- .` shows the latest `[rmsnorm-campaign iter N]` commit
- [ ] env: umbriel-b200-035, /usr/bin/python3.12, torch 2.11.0a0+nv26.02, triton 3.6.0,
      flashinfer 0.6.11; ALWAYS `CUDA_VISIBLE_DEVICES=1`
- [ ] GPU thermal: GPU1 idles 37 °C (OK); GPU0 on 035 has a thermal-throttle history — never use it
- [ ] no co-resident driver on GPU1: check output-file growth (ps/nvidia-smi namespace-blind)
- [ ] anchor cell re-run: incumbent T=4096 nsys = 21.17 µs ± 3% (drift > 3% ⇒ re-baseline)
      (re-anchored 2026-07-13 on 035/GPU1; the 027-era value was 21.82 — do not reuse)

## 3. Work split
Single node, single session. No sharding.

## 4. Launch commands (byte-exact; scripts live in ../../../gvr_agent_retrospective/skill_v2_draft/scripts/)
```bash
cd /home/scratch.loncheng_gpu/workspace/perf/workloads/DSV4/TensorRT-LLM/indexer_topk_op_bench/omni_v2_validation/tierB/rmsnorm_campaign
SKILL=../../../gvr_agent_retrospective/skill_v2_draft
CUDA_VISIBLE_DEVICES=1 TOKENS=4096 python3 $SKILL/scripts/verify_exact.py --impl src/<cand>.py --mode dense --dtype bf16
CUDA_VISIBLE_DEVICES=1 TOKENS=4096 python3 $SKILL/scripts/bench_cold.py --impl src/<cand>.py --baseline src/incumbent.py
CUDA_VISIBLE_DEVICES=1 TOKENS=4096 python3 $SKILL/scripts/nsys_verdict.py --impl src/<cand>.py --baseline src/incumbent.py --kernel-regex 'RMSNormKernel|rmsnorm' --anchor-impl src/incumbent.py --anchor-expected 21.17
```

## 5. Known gotchas
- profilers embed env → nsys_verdict.py already wraps `env -u GITHUB_TOKEN -u HF_TOKEN`
- ncu -k needs `regex:` prefix: KERNEL_REGEX="regex:RMSNormKernel" for ncu_attrib.sh
- tensor.copy_() lowers to DtoD memcpy (invisible to cuda_gpu_kern_sum) — probe with torch.mul(out=)
- L1 graph timing inflates small-T cells ~7-18 µs (launch bias); ship claims only via nsys
- impl modules read the cell from the TOKENS env var (default 4096); seed = f(TOKENS)

<!-- END PASTE -->
