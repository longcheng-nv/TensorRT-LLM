# Making LLMs Think Faster: A Smarter Top-K for Sparse Attention

By NVIDIA TensorRT LLM team

## The Problem: Finding Needles in a Haystack, Every Single Step

Modern AI agents — systems that browse the web, write code, and call tools autonomously — need to process *long* conversations, sometimes 64K–128K tokens or more. To avoid the quadratic cost of full attention, a growing family of **sparse attention** methods — including DeepSeek [DSA](https://api-docs.deepseek.com/news/news251201) and [NSA](https://arxiv.org/abs/2502.11089), Moonshot's [MoBA](https://arxiv.org/abs/2502.13189), NVIDIA's [RocketKV](https://arxiv.org/abs/2502.15579), MIT's [Quest](https://proceedings.mlr.press/v235/tang24l.html), and [SAGE-KV](https://arxiv.org/abs/2503.08879) — all rely on a common primitive: **Top-K selection** to pick the most important key-value entries at token, page, or block granularity.

The catch? **Picking the top entries from tens or hundreds of thousands of candidates** becomes a real bottleneck as sequences grow. In DeepSeek-V3.2's Sparse Attention (DSA), the Top-K step selects 2048 tokens from up to 128K+ indexer scores each decode step — consuming a substantial fraction of the sparse attention module's latency at long sequences.

<div align="center">
<figure>
  <img src="../media/tech_blog19_indexer_topk.png" alt="DSA Indexer Top-K" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>The DeepSeek Sparse Attention pipeline: a lightning indexer scores all tokens, then a Top-K selector picks the 2048 most important ones for attention computation. (Illustrated with K=5, N=10; actual K=2048, N up to 128K+.)</em></sub></p>

## The Insight: Yesterday's Answer Is Today's Cheat Sheet

Here's the key observation: when an LLM generates text token by token, the set of "important" past tokens barely changes from one step to the next. If tokens #500, #1200, and #3000 were important for generating the word "the", they're very likely still important for generating the next word "cat".

The following chart shows our measurements on real DeepSeek-V3.2 workloads (SWE-Bench-64K) — **about 35–50% of the top-2048 tokens remain the same** between consecutive decoding steps (more in deeper layers):

<div align="center">
<figure>
  <img src="../media/tech_blog19_hit_ratio.png" alt="Top-K Hit Ratio" width="850" height="auto">
</figure>
</div>
<p align="center"><sub><em>Top-K overlap (hit ratio) between consecutive decoding steps. Deeper layers (L20–L60) exhibit 35–50% average overlap, while shallow layers (L0, L1) are near zero.</em></sub></p>

This isn't a coincidence — it's a direct consequence of how positional encoding (RoPE/YaRN) works. The attention scores depend on *relative distance* between tokens, and advancing by one position only shifts each distance by one — a tiny perturbation in a smooth function.

**This means we can use the previous step's Top-K result as a "cheat sheet" to speed up the current step.**

## The Algorithm: Guess, Verify, Refine

Instead of exhaustively scanning all data 3–4 times (like the existing radix-select method), our heuristic approach exploits the "cheat sheet" to find the answer in just 1–2 scans:

<div align="center">
<figure>
  <img src="../media/tech_blog19_algorithm_flow.png" alt="Four-Phase Algorithm Flow" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>Heuristic-guided Top-K: four phases execute sequentially within a single CTA (thread block). Per-phase complexity is annotated below each phase.</em></sub></p>

Each phase in one sentence:

| Phase | What it does | Why it's fast |
|:-----:|-------------|---------------|
| **Guess** | Compute a score threshold from last step's top-2048 | Reads only 2048 scattered values, not all N |
| **Search** | Secant interpolation to adjust threshold until 2048–6144 candidates | Good initial guess → only 1–2 full scans needed |
| **Collect** | Gather all above-threshold candidates into on-chip SRAM | No ballot sync, no atomic contention, full memory pipeline |
| **Refine** | 2048-bin histogram to select exactly 2048 | Pure on-chip work, converges in 1–3 iterations |

**The core advantage**: the existing radix-select method scans the entire score array 3–4 times regardless of the data. We exploit the "cheat sheet" for an accurate initial guess — **fewer memory passes = less time reading data = faster**.

## The Results

### Real DeepSeek-V3.2 Decoding on NVIDIA B200

On actual decode-stage data from SWE-Bench-64K evaluation (9 layers × 17 sampled decode steps), the heuristic kernel **outperforms the baseline on every single layer**:

<div align="center">
<figure>
  <img src="../media/tech_blog19_real_data_bars.png" alt="Per-Layer Latency and Speedup" width="800" height="auto">
</figure>
</div>
<p align="center"><sub><em>Per-layer kernel latency at N=70,690 on B200. The heuristic kernel achieves 1.32×–2.11× speedup across all 9 layers.</em></sub></p>

| Metric | Value |
|:------:|:-----:|
| **Overall average speedup** | **1.81×** |
| **Best layer per-step** | up to **2.36×** |
| **Layers beating baseline** | **9 / 9** (all layers win) |

### It Works Across Data Distributions

This is a **data-aware** algorithm — it works best when the "cheat sheet" is accurate. We analyzed the score distributions across layers and found stable speedups across all distribution types:

| Distribution Type | Examples | Speedup Range |
|:-----------------:|:--------:|:-------------:|
| Beta (bounded, peaked) | L21, L40, L41 | **1.80–2.11×** |
| Weibull (right-skewed) | L22, L60 | **1.72–1.92×** |
| Logistic/t (heavy-tailed) | L1 | **1.74×** |
| Lognormal (heterogeneous) | L0 | **1.32×** |
| Synthetic random (N≥16K) | — | **1.01–1.75×** |

Even the trickiest layer (L0, lognormal) still consistently beats the baseline — the algorithm is **robust**.

### No Accuracy Loss

The heuristic Top-K produces the exact same set of top-2048 indices as `torch.topk` — verified across sequence lengths from 8K to 131K. End-to-end model accuracy is unchanged across five benchmarks (MMLU, GSM8K, GPQA, LongBench V1/V2), with all deltas within run-to-run variance.

## How It Fits In

The heuristic kernel is integrated into TensorRT-LLM as a **configurable fast path**. Enable it via YAML:

```yaml
sparse_attention_config:
    algorithm: dsa
    enable_heuristic_topk: true
```

When enabled, the system automatically uses the heuristic kernel when previous-step Top-K indices are available and conditions are right; otherwise, it falls back to the proven radix-select method. The kernel has been heavily tuned for **batch=1 minimum-latency** scenarios; multi-batch decode is functionally supported. The heuristic path is only active on sm\_100+ (Blackwell) GPUs.

## The Bigger Picture

This work illustrates an important principle: **data-aware GPU kernel design**. Instead of building algorithms that treat all inputs identically, we exploit the statistical structure of specific workloads — in this case, the temporal correlation inherent in autoregressive LLM decoding.

While demonstrated on DeepSeek DSA, the approach generalizes to **any sparse attention method whose decode-phase Top-K exhibits temporal correlation** — including NSA, MoBA, RocketKV, and others. As context lengths continue to grow in the era of agentic AI, such workload-specific optimizations will be essential for keeping inference fast and affordable.

TensorRT-LLM is open source — we welcome the community to extend this approach, whether through new prediction strategies, cross-model validation, or next-generation architecture support. Join us in building a faster GPU inference ecosystem for the agentic era.

## Learn More

- **Full technical report**: [Temporal Correlation Meets Sparse Attention](blog19_Temporal_Correlation_Meets_Sparse_Attention.md)
- **DeepSeek-V3.2 optimizations**: [Tech Blog 15](blog15_Optimizing_DeepSeek_V32_on_NVIDIA_Blackwell_GPUs.md)
- **Sparse attention framework**: [Tech Blog 17](blog17_Sparse_Attention_in_TensorRT-LLM.md)
