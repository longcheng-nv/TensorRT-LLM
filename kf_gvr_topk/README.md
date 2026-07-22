# `topk_compB` — exact fp32 top-K index selection for the DSv4/V3.2 indexer (B200)

Second-generation KernelFactory result (campaign `e5q1zgrfhs0z57dj6850kc444r`,
"gvr-topk-r3", baseline = campaign-1 champion `c74f_sbx` on this branch's
parent commit). Composite of round-2/3 winners `becdc5c7` + `aef33fac` +
`30e79029` with two engineer dispatch grafts.

## Contract

```
run(logits: fp32[1, npad] (CUDA), pre_idx: i32[1, k] (unused), n_valid: i64, indices_out: i32[1, k])
```

Exact top-k **indices** of `logits[0, :n_valid]` (any order; tie-robust set
semantics: all strictly-greater-than-kth present, remainder from the tie set).
Hint-free: `pre_idx` ignored (re-falsified again in R3: a provably-exact
warm-hint admission filter measured 1.0001 in its own activation zone).

## Algorithm (top-level dispatch on (n, k) only)

- `n-k==3, n<=1536` -> `bottom3_kernel` (complement selection).
- `n<=16896` -> single-CTA `topk_small<KPT>` byte-radix rungs (unchanged
  from campaign 1), except tail-selection cells `4*k<=n`, `8195<=n<=16387`
  -> `topk_mid<3|4>` @1024 threads (single-CTA two-level histogram over
  register-cached keys; heals the N=16387 weak band +19%). The `topk_mid<1>`
  rung (n~4099) measured a regression and is gated out.
- `k==2048 && 16896<n<=140000` -> `v30::topk_coop`: contiguous-slice
  3-pass 11/11-12/10-9 histogram ladder (K=2048 boundary buckets are large;
  the fast-tail path below does not pay there).
- else -> `aefm::topk_fast`: REGULAR launch (no cudaLaunchCooperativeKernel),
  grid sized to co-residency, register-cached row (1 float4 + tail scalar per
  thread, zero global re-reads), one 11-bit MSB histogram pass merged with
  packed 64-bit atomics, then a sense-token spin grid barrier and a 3-way
  adaptive finish on boundary-bucket size T:
  (a) whole bucket needed -> direct classify+write, done (1 barrier total);
  (b) T<=4096 -> smem-staged compaction to a global tie buffer, non-spinning
      rendezvous (arrive-and-exit), LAST arriver alone refines the low 21 bits
      in shared memory (2048-bin pass + warp finisher; 1 barrier, no drain);
  (c) large bucket -> classic 11/11/10 ladder over the register-cached keys.

### Memory-ordering note (production-port review item)

The spin barrier uses relaxed atomics with NO fence, deliberately: every
formally-ordered variant (`__threadfence`, scoped `atom.acq_rel`/
`ld.acquire.gpu`, relaxed-spin + trailing acquire) measured 8-11% slower —
release semantics force the arriving block to drain its L2-pending writes on
the critical path. Safety on sm_100 relies on: (1) merged-histogram lines are
first plain-touched only AFTER the barrier within a launch, (2) L1 is
invalidated at kernel-launch boundaries, (3) pre-barrier writes are L2
atomics. Any edit that plain-reads merged scratch before the barrier, or an
architecture with launch-persistent L1, breaks this contract.

## Measured (B200, nsys cold-L2, paired same-GPU, 865 real decode cells BS=1)

- geomean **1.8267x** vs TensorRT-LLM GVR PR#16457 head @b14ec40e1b
  (865/865 exact, zero regressions, min cell 1.140; anchors med 1.006).
- +8.1% over the campaign-1 champion `c74f_sbx` on the same protocol.
- PR-arm-normalized rivals: vs SGLang indexer top-K v2 **1.215x**,
  vs in-tree `radix_cutedsl` **1.760x**, vs FlashInfer top-k **1.585x**.
- Campaign cost $764.66 (4 rounds, 27 agents, cancelled at operator
  close-out with round-4 best below the harvested composite).

Process log: `indexer_topk_op_bench/op26_r0_upstream_port_report/kf_campaign/KF_PROCESS_LOG.html` §7 (repo-local, not on GitHub).
