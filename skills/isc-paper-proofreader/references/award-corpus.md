# Evidence corpus: ten pre-2026 Hans Meuer Award papers

## Selection method

The corpus contains the ten papers listed by ISC as Hans Meuer Award winners from 2017 through 2025, stopping before 2026. ISC lists two winners for 2020, so this interval yields exactly ten papers. The [official ISC history](https://isc-hpc.com/history/) establishes award status. Paper pages or author manuscripts below were used to inspect the prose.

This is a deliberately selective corpus of recognized papers, not a statistical sample of every accepted ISC paper. It spans both the earlier Springer LNCS format and the newer IEEE-style proceedings. Use its recurring patterns as editorial evidence rather than rigid venue rules.

## Papers and useful writing observations

| Year | Award paper and full-text source | Writing evidence used in the guide |
|---|---|---|
| 2025 | [Towards a Unified Architectural Representation in HPCQC: Extending sys-sage for Quantum Technologies](https://inspirehep.net/files/5e8894d3bc20a1488a9a7ff929c4254a) | Frames the work around a concrete representation problem, then names the extended library, data structures, parsers, and topology metrics. Its abstract is an outlier in this corpus because it gives no numeric performance result; this is evidence against forcing every abstract into one template. |
| 2024 | [Evaluation of the Classical Hardware Requirements for Large-Scale Quantum Computations](https://inspirehep.net/files/4c95b2e0e5c45576d996a4adc5f1de69) | States the model and assumptions first, then reports clock rate, syndrome bandwidth, and compute requirements with ranges. The abstract closes with unresolved quantum and classical hardware constraints. The discussion revisits assumptions instead of treating estimates as universal facts. |
| 2023 | [Expression Isolation of Compiler-Induced Numerical Inconsistencies in Heterogeneous Code](https://web.cs.ucdavis.edu/~rubio/includes/isc23.pdf) | Gives the tool's search mechanism before its results, names synthetic, benchmark, and real workloads, and attaches percentages to specific comparison tasks. The evaluation is organized by research questions and includes threats and cases that could not be reproduced. |
| 2022 | [Remote OpenMP Offloading](https://doi.org/10.1007/978-3-031-07312-0_16) ([author preprint](https://drive.google.com/file/d/1yjmuDT8LQiG1tThhjPzyX4gLixXdhXZp/view)) | Repeats a concrete boundary condition—remote execution without source, compiler, or language changes—rather than using broad portability language. Scaling claims name the systems and depend on compute-to-transfer ratio. A dedicated limitations section explains what OpenMP cannot express efficiently. |
| 2021 | [A Performance Analysis of Modern Parallel Programming Models Using a Compute-Bound Application](https://uob-hpc.github.io/assets/ISC-Bristol-Hans-Meueur-award-2021.pdf) | Defines a realistic mini-app and compares several models across CPUs and GPUs. The conclusion reports the negative result that no implementation wins across all platforms and quantifies the effect of platform-specific tuning. A reproducibility section links code and scripts. |
| 2020 | [Load-Balancing Parallel Relational Algebra](https://pmc.ncbi.nlm.nih.gov/articles/PMC7295339/) | Turns an ambiguous problem into a two-part taxonomy: spatial and temporal imbalance. Each mechanism maps to one part of that taxonomy. The introduction gives scale and corner cases; the abstract emphasizes conceptual structure rather than forcing a headline number. |
| 2020 | [Time Series Mining at Petascale Performance](https://pmc.ncbi.nlm.nih.gov/articles/PMC7295345/) | Moves from the computational obstacle to an MPI design, analytical model, and experiment. The main claim is anchored by dimensionality, record count, number of sorts, sustained performance, and named machine. The performance discussion identifies setup overhead rather than hiding it. |
| 2019 | [GPUMixer: Performance-Driven Floating-Point Tuning for GPU Scientific Applications](https://engineering.purdue.edu/dcsl/publications/papers/2019/gpumixer_isc19_cameraready.pdf) | Uses a compact prior-work contrast, names two mechanisms, and reports one direct comparison in the abstract. A dedicated limitations subsection discusses the proxy cost model and register pressure at the point where they matter. |
| 2018 | [Compiler-Assisted Source-to-Source Skeletonization of Application Models for System Simulation](https://www.osti.gov/servlets/purl/1513073) | Explains the tradeoff between trace replay and hand-written skeletons before introducing its pragma language and compiler. Validation covers both fidelity and scaling beyond 100,000 endpoints. The conclusion stays on the network-design use case established at the start. |
| 2017 | [Designing Dynamic and Adaptive MPI Point-to-Point Communication Protocols for Efficient Overlap of Computation and Communication](https://souravc.com/assets/papers/isc_17.pdf) | Defines the eager/rendezvous choice and the performance-memory-overlap tradeoff before presenting per-peer adaptation. Results name applications, process counts, and memory effects. Its long abstract shows that older award papers should not be copied as a current length model. |

## Corpus-level observations

The following values are approximate because they come from PDF text extraction, but they are useful guardrails:

- Abstracts range from about 123 to 344 words, with a median near 211. The 2017 and 2021 papers are longer than a modern 250-word limit would allow.
- Median sentence length by paper ranges from roughly 16 to 24 words; the 90th percentile ranges from about 32 to 45. The prose mixes short claims with longer qualified explanations.
- First-person `we` and `our` occur in every paper, at roughly 6–20 occurrences per 1,000 words. First person is normal ISC prose when it clarifies agency.
- Nine papers explicitly announce contributions. The remaining paper still makes the artifact, validation, and use case easy to recover from the introduction.
- All ten papers state boundaries through limitations, negative findings, future work, assumptions, or workload restrictions. Several place the boundary next to the result rather than postponing it to the conclusion.
- `delve`, `pivotal`, `underscore`, `transformative`, `groundbreaking`, `revolutionary`, and `notably` do not occur in the extracted corpus. Individual words such as `robust`, `comprehensive`, `paradigm`, and `seamless` do occur, usually in domain-specific contexts. This supports contextual review rather than a banned-word rule.

## What not to infer

Winning an award does not make every sentence an editing model. Some abstracts repeat material, some opening claims are broad, and paper formats changed across the interval. The guide retains patterns that recur across topics and years and rejects quirks that appear in only one paper.
