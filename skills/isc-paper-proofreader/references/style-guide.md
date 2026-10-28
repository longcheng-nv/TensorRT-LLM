# ISC expert-prose style guide

## Editorial target

Write for an HPC reviewer who wants to decide quickly whether the problem matters, the mechanism is new, and the evidence supports the claim. Natural expert prose is a consequence of technical control: the author knows which noun is acting, which comparison is valid, and where the result stops generalizing.

Do not optimize for an AI detector. Optimize for traceable reasoning, precise claims, and readable technical density.

## Twelve general rules

### 1. Start from the technical object

Name the operation, runtime behavior, resource constraint, numerical failure, or scaling problem in the first useful sentence. Avoid a generic tour of “the rapidly evolving HPC landscape.”

Weak:

> Modern AI workloads are transforming computing and creating unprecedented challenges.

Better:

> Sparse-attention inference must select the exact top-k keys for every query while reading only a small fraction of the key cache.

The better sentence gives the reader an operation and a constraint. It does not claim importance before explaining the problem.

### 2. Give each paragraph one job

A paragraph should establish a problem, explain a mechanism, interpret evidence, delimit a claim, or connect the work to prior art. Split paragraphs that do several jobs without a clear order. Merge one-sentence fragments that merely restate the heading.

Use the progression `observation -> cause -> consequence -> response` when explaining motivation. Use `question -> setup -> result -> interpretation -> boundary` when discussing evaluation.

### 3. Put the mechanism before the praise

Words such as efficient, scalable, robust, portable, comprehensive, and seamless are conclusions. Explain the design choice that could produce the property.

Weak:

> GVR provides a robust and highly efficient solution across diverse architectures.

Better:

> GVR samples candidates from the input distribution, verifies them against an exact stopping condition, and uses architecture-specific kernels without changing the selection semantics.

Follow with measured evidence before calling the result efficient or portable.

### 4. Bind every result to a denominator and conditions

State what changed, relative to what, on which workload and platform, with which metric. “Up to 2.1x” is incomplete without the baseline and configuration. Distinguish kernel latency, operator latency, model throughput, and service-level end-to-end throughput.

Prefer:

> On [platform] with [shape and precision], GVR V2 reduces indexer top-k latency by [x%] relative to [baseline]; the gain falls to [y%] when [condition].

Leave placeholders when evidence is unavailable. Do not manufacture completeness.

### 5. Scope priority claims

Replace universal “state of the art” wording with the actual comparison set. A defensible claim names the task, exactness requirement, platform, baseline set, and evidence date or citation.

Prefer:

> Among the exact top-k GPU implementations evaluated on [platforms], GVR V2 achieves the lowest latency for [shape range].

If the comparison is incomplete, say “outperforms the evaluated baselines.”

### 6. Use first person for agency

`We design`, `we measure`, and `we find` are direct and normal. Do not convert them mechanically to passive voice. Use passive voice for a standard procedure or when the actor does not matter.

Good active use:

> We measure operator latency with CUDA Graph replay after 200 warm-up iterations.

Good passive use:

> Latency is reported as the median of 1,000 iterations.

### 7. Place caveats next to claims

Do not hide all limitations in the final paragraph. State a limitation when it changes how a figure or result should be read. Strong ISC papers report performance cliffs, unsupported cases, setup overhead, and portability gaps.

Prefer:

> The kernel scales linearly through eight GPUs; beyond eight, host-side dispatch dominates, so this experiment does not establish multi-node scaling.

### 8. Use transitions that carry logic

Transitions should say how two statements relate. `However` is useful when the second sentence contradicts or limits the first. `Therefore` is useful only when the conclusion follows. Delete `Furthermore`, `Moreover`, `Importantly`, and `Notably` when they merely decorate the next sentence.

Weak:

> Furthermore, we evaluate Blackwell. Importantly, the method is deterministic.

Better:

> Blackwell changes the available memory and instruction throughput, so we retune the kernel configuration while keeping the exact stopping rule. Deterministic mode fixes candidate ordering and tie handling.

### 9. Keep technical nouns stable

Use one term for one concept. Do not rotate among method, framework, approach, solution, and system for variety. Define abbreviations once. Keep exact versus approximate and deterministic versus bitwise reproducible distinct throughout the paper.

### 10. Let figures and equations carry work

Introduce a figure with the question it answers. After the figure, state the main pattern, an exception, and the implication. Do not narrate every bar.

Introduce notation immediately before the equation. Define variables and state what the equation permits the system to decide. Remove equations that only restate prose without enabling analysis or implementation.

### 11. Make related work comparative

Group prior work by the design choice that matters to the paper. For each group, state the shared approach and the remaining gap. Avoid one sentence per citation and avoid praising or dismissing whole fields.

Useful pattern:

> Sampling-based selectors reduce the number of inspected elements [citations], but the evaluated variants either permit approximation or rely on a fixed sample budget. GVR instead verifies an input-dependent stopping condition to preserve exact top-k semantics.

### 12. End with the established result

The conclusion should answer the paper's question using the evaluated scope. Retain a material limitation or next technical obstacle. Avoid claims that the work “paves the way,” “ushers in a new era,” or “has transformative potential.”

## Common synthetic patterns and repairs

| Pattern | Why it reads as synthetic | Repair |
|---|---|---|
| “It is important/interesting/worth noting that …” | Announces importance instead of supplying it. | Delete the frame and state the fact. |
| “In today's rapidly evolving landscape …” | Generic opening that fits many papers. | Name the operation, resource, or failure mode. |
| “This paper delves into …” | Describes the act of writing. | State the research question or action. |
| “seamlessly leverages X to enable Y” | Hides the mechanism and likely overclaims integration quality. | Use the actual data movement, API, algorithm, or mapping. |
| “not only X but also Y” repeated | Adds rhetorical rhythm without clarifying priority. | State X and Y directly, or explain their causal relation. |
| “This result underscores/highlights …” | Adds a second interpretation sentence with no new content. | State the implication or delete the sentence. |
| Repeated `First`, `Moreover`, `Furthermore`, `Finally` | Imposes list rhythm on connected reasoning. | Use cause, contrast, condition, or no transition. |
| Three-adjective noun phrases | Simulates confidence through modifiers. | Keep the property that is defined or measured. |
| Every paragraph ending with a mini-summary | Repeats claims and produces a generated cadence. | End after the evidence or implication. |
| Consecutive sentences beginning with “We” | Makes prose read like a procedure list. | Combine related actions or make the artifact/result the subject where natural. |
| Em-dashes used for every qualification | Gives prose a scripted cadence and hides hierarchy. | Use parentheses for local detail and separate sentences for material caveats. |
| “Clearly,” “obviously,” “undeniably” | Substitutes confidence for evidence. | Provide the evidence or remove the adverb. |

## Section-specific checklist

### Abstract

- Concrete problem appears before broad impact.
- Gap is a technical limitation, not “little work has been done.”
- Mechanism is named with at least one operative detail.
- Evaluation scope is recoverable.
- Main number has a baseline or physical meaning.
- Exactness, determinism, portability, and end-to-end claims are separately scoped.
- Abstract fits the target year's limit.

### Introduction and contributions

- The first page reaches the paper's actual bottleneck.
- Terms and system boundaries are defined before contributions.
- Each contribution says what was built, derived, discovered, or measured.
- Contribution bullets are non-overlapping and supported later in the paper.
- The introduction does not promise results absent from evaluation.

### Evaluation

- Hardware, software versions, precision, shapes, workloads, and timing method are stated.
- Baseline implementation and tuning effort are fair and identifiable.
- Aggregation and variability are reported where material.
- Strong and weak scaling are named correctly.
- Performance portability uses a cross-platform metric or clearly scoped per-platform comparisons.
- Determinism includes tie handling, launch/order assumptions, and repeat count.
- End-to-end gains identify the service configuration and bottleneck shift.
- Negative cases and sensitivity are reported.

### Final line edit

- Pronouns have an unambiguous referent.
- One term names each concept.
- Adjectives correspond to a definition or measurement.
- Citation placement makes the supported claim clear.
- Sentence length varies because ideas vary, not to satisfy a fixed target.
- Paragraph openings are not a repeated template.
- No generic transition or summary sentence remains after its content is removed.

## Suggested proofreading report

Use a compact report after editing:

1. **Argument fixes:** missing gap, reordered mechanism, scoped claim, restored caveat.
2. **Prose fixes:** removed throat-clearing, stabilized terminology, shortened overloaded sentences, varied paragraph rhythm.
3. **Author decisions:** unsupported numbers, ambiguous baselines, missing citations, or claims requiring domain judgment.
4. **Verification:** compile, reference/citation check, page count, and optional diagnostic scan.
