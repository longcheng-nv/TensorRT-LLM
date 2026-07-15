# op34 RESUME — GVR-skeleton top-K vs sglang_v2 (real v4cap, BS=1) — CONVERGED

Node-agnostic (NFS-shared). Workspace: indexer_topk_op_bench/op34_gvr_beat_sglang/

## STATUS: CONVERGED — STOP (double-locked INFEASIBLE, pre-authorized negative conclusion)
Beating sglang_v2 by 30% on BS=1 within the GVR threshold skeleton is infeasible on real V4
decode data. The UB probe (oracle-threshold multi-CTA collect-only, C=64, no tail) merely EQUALS
sglang's entire kernel ⇒ no room for the mandatory rank tail. Real hint arm = 4–8× slower.
Full detail: analysis/DOUBLE_LOCK_048.md. Report: report/op34_report.html (bilingual, CSS toggle).

## 1-min context
- LATENCY-bound (both arms <1% DRAM AND <1% SM peak); sglang wins via 8-CTA MLP (NCU_CRUX_048).
- Cold-L2: both do 1 HBM read; pass count NOT the wall; hint saves only an L2-hot pass.
- Multi-CTA (C>8) scan scales (3.7× @large-N, CRUX-A) but hint can't place an exact-safe tight
  threshold ⇒ candidate set huge on real data ⇒ collect+tail loses.
- REAL nsys sglang = 12–19µs @large N (NCU's 28–39µs was replay inflation — ship-verdict on nsys).

## preflight (30s)
cd indexer_topk_op_bench/op34_gvr_beat_sglang
git rev-parse --short HEAD
python3 -c "import torch,triton;print(torch.cuda.get_device_name())"   # B200
python3 scripts/env_anchor.py    # anchor pro/256k L32: sglang~21.6 r0~31.2 wall (048); r0/sgl~1.45
nvidia-smi --query-gpu=index,temperature.gpu,memory.used --format=csv,noheader  # idle<50C

## artifacts (all committed except *.nsys-rep/*.sqlite/*.kern.json = gitignored)
- src/op34_mcta_op.py            multi-CTA single-pass GVR (exact; the falsified arm)
- scripts/{env_anchor,ncu_crux,crux_a_mlp,crux_c_proxy,nsys_op34,parse_op34}.py + drive_nsys_op34.sh
- analysis/{ANCHOR,NCU_CRUX,CRUX_A_MLP,CRUX_C_PROXY,DOUBLE_LOCK}_048.md + PHASE1/FEASIBILITY
- results/{harvest_pro,decomp2,grid}/results.jsonl  (nsys cold pure-kernel)
- report/{gen_report_op34.py,template.html,op34_report.html}

## to refresh the report after any re-measure
python3 report/gen_report_op34.py    # reads results/{harvest_pro,decomp2,grid}, idempotent

## gotchas
- GVR seq_lens = N*cr (uncompressed); passing N → scans N/cr → recall 0 (env_anchor caught this).
- NEVER commit *.nsys-rep/*.sqlite (env tokens; gitignored). setsid/nohup& is sandbox-killed here
  (exit 144) → run nsys drive foreground or via the tool's run_in_background.
- collect_only arms write sentinel out[0]=arange(K) so the harness exactness probe can't OOB.
- If re-opening: the ONLY unexplored axis is BS>1 (different MLP calculus) — out of this campaign's scope.
