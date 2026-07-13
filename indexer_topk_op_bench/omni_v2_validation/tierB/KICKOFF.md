# Tier B mini-campaign kickoff — omni-kernel v2 trial run (dense class)

Human-supplied objective triple (the agent MAY NOT relax it):

```yaml
objective:
  incumbent: flashinfer.norm.rmsnorm (flashinfer 0.6.11)   # TRT-LLM production default
  rivals: [eager torch RMSNorm (fp32 upcast), torch.compile RMSNorm]
  envelope: {hidden: 7168, tokens: [1, 16, 256, 4096, 16384], dtype: [bf16], BS: n/a}
  verdict_axes: [worst, geomean, best]   # over the token grid; no real-capture axis for this op
  ship_rule: "geomean >= 1.00 vs incumbent AND no cell < 0.98 AND exactness green
              (dense bf16 atol/rtol 1e-2) AND dispatch rules <= 3"
  hard_constraints: [CUDA-graph compatible, out-of-place, no incumbent source edits]
budget:
  iterations_max: 5
  wallclock_max: 2h
  gpu: single (CUDA_VISIBLE_DEVICES=2 on umbriel-b200-027; idle, 33-39C)
pre_authorized_negative_conclusion: >
  If flashinfer.norm.rmsnorm remains best on the envelope, say so plainly with
  numbers. A clean FALSIFIED/INFEASIBLE verdict is a fully successful outcome.
```

Envelope note: hidden=7168 is the DSv4 hidden size; the token grid spans the
decode (1-256) and prefill (4096-16384) regimes the production path sees.
