# Human-Expert Style Baseline: Pre-2022 Systems Best Papers

Corpus: six PPoPP Best Papers written before the LLM era — Tapir (2017,
~13.7k words), TxSampler (2019, ~11.9k), Non-blocking IST (2020, ~11.8k),
SCCL (2021, ~9.3k), Lock-Free Locks Revisited "LFL" (2022, ~13.7k), Software
Combining in Persistence "SWC" (2022, ~13.8k). Total ~74k words. All counts
are grep-based over extracted full text. These norms generalize to
PPoPP/ASPLOS/ISCA/MICRO/SOSP/OSDI-class systems venues.

## 1. Abstract structure

- 158–309 words; multi-paragraph is normal (3 paragraphs common).
- Shape: context (2–4 sentences) → the artifact, named and defined → 1–2
  mechanism sentences → concrete quantified results.
- The system name appears early and does grammatical work as a subject:
  "Tapir is a compiler IR that represents logically parallel tasks
  asymmetrically."
- Numbers are specific, conditioned, unglamorous, and bound to their
  condition in the same sentence: "improvements of up to 15% under high
  update rates, and of up to 50% under moderate update rates"; "we added or
  modified about 6000 lines of LLVM's 4-million-line codebase"; "incurs ~4%
  runtime overhead".
- Vague quantifiers only where one number genuinely can't summarize:
  "obtain nontrivial speedups", "many times faster".
- Honest scoping appears in abstracts: "which should be of independent
  interest"; "surprisingly robust to distributional skew, which suggests…".
- No abstract follows the rigid "Problem. / We propose X. / Extensive
  experiments show…" template.

## 2. Introduction and contribution lists

- Motivation is concrete: a single worked example or exact number within the
  first page. Tapir opens on a specific `cilk_for` loop that "GCC 5.3.0, ICC
  16.0.3, and Cilk Plus/LLVM 3.9.0" all fail to optimize, with the
  Θ(n²)→Θ(n) consequence. SCCL cites "30% of the training time for the 8.3
  billion parameter Megatron language model… is spent inside Allreduce".
  Never a generic "X has become increasingly important in recent years"
  paragraph.
- Related-work triage happens inside the intro with named prior tools and
  concrete defects: "TSXProf… incurs non-trivial overhead (~3×) in the
  replay stage".
- Contribution lists: plain lead-in ("Our contributions are as follows:",
  "We make the following five contributions in this paper:") + terse,
  **unequal-length**, head-less bullets, often verb-first or bare noun
  phrases, with §-references and numbers embedded: "We prove the
  correctness, non-blocking and complexity properties of the C-IST
  (Section 4)."; "A reduction from the synthesis problem for combining
  collectives to that for non-combining collectives." No bolded bullet
  heads, no "**Contribution 1:**", no parallel triads of equal length.

## 3. Sentence-level statistics

- Mean sentence length 19.7–22.2 words, median 18–19.5, stdev 10.4–14.1.
  **High variance is the signature**: "Another issue is performance." sits
  next to 40+-word sentences with mid-sentence citations.
- Em-dashes: 9/4/0/0/10/0 per paper — author-idiosyncratic, not house
  style. Three of six papers contain zero. Where used, ONE consistent
  function per author (spaced appositive definition, or unspaced "—i.e."
  gloss). Frequent rhetorical "not X—but Y" matches none of these papers.
- Semicolons: real but modest clause-joining in three papers; near-zero in
  the other three.
- Hedging is empirical, not rhetorical: "appears to be", "we believe",
  "which suggests" — attached ONLY to claims the authors cannot measure.
  Measured claims are stated flat.

## 4. Vocabulary frequencies (total across ~74k words)

Essentially ABSENT (0 hits): principled, crucially, "key insight",
"extensive experiments", "we systematically", "in conclusion", "in essence",
delve, pivotal, seamless, holistic, meticulous, paradigm, landscape, realm,
myriad, plethora, versatile, cutting-edge, synerg-, orchestrat-,
substantially.

Rare (≤3 total, usually literal or citing): comprehensive (3, one in a cited
title), carefully (3, literal), systematically (2), notably (3),
importantly (2), interestingly (2), surprisingly (1, reporting a genuine
unexpected finding).

Context-dependent: novel (9 total, 7 in SCCL — a synthesis paper literally
claiming new algorithms; 0–1 elsewhere). leverag- (9, paper-local).
"state-of-the-art" (16) — almost always naming the **competitor** being
beaten, never self-description.

Connectives, moderate: however 52 (~1/1.4k words), thus 42, therefore 18,
moreover 17 (0 in two papers), "in particular" 17, furthermore 8 (0 in four
papers), "in summary" 3, overall 2. "Note that" 26 — humans use it freely.
"In this paper" 17 — standard.

First person is heavy: "we" 63–115 per paper (5–11 per 1000 words), "our"
18–65. "We believe" (2), "as far as we know", "to the best of our
knowledge" (1).

## 5. Experiments and results prose

- Setup sections are dense hardware liturgy: exact SKUs, clocks, cache
  sizes, OS, compiler+flags, allocator, pinning. Methodology quirks are
  disclosed: "The system was 'quiesced' … by turning off Turbo Boost, dvfs,
  hyperthreading, extraneous interrupts, etc."
- Speedups always carry a multiplier/percent WITH condition: "improvements
  ranging from 15-50% compared to the (a,b)-tree… depending on the ratio of
  updates and lookups". Approximation marks (~, ≈) used honestly.
- **Losses reported in the same register as wins, with diagnosis**: "The
  biggest slowdown … occurs on Cholesky, for which the executable produced
  by Tapir/LLVM has 4% more work"; "PBheap has good performance when the
  size of the heap is not very large" (stated twice, including the
  abstract). No "while X, our method still…" damage control.
- Results narration points at figures directly and mixes tense freely: "We
  see this happen in the leaftree in Figure 4."

## 6. Captions

Bimodal, never uniform: ultra-short fragments ("Figure 1. Data Types")
coexist with 100–150-word self-contained caption essays. Fragments and full
sentences mix within one caption. A manuscript where every caption is a
polished paragraph of similar length is off-pattern.

## 7. Terminology discipline

- **One branded name per paper** — the artifact itself. Everything else is
  lowercase technical English: "parallel rebuilding technique", "lock-free
  locks", "the combiner", "helping mode", "serial elision". Competitors get
  flat acronyms.
- No capitalized multi-word concept branding anywhere in the corpus.

## 8. Paragraph rhythm

- Paragraphs 4–10 sentences (80–180 words), single-topic. Two-sentence
  paragraphs rare and deliberate.
- "First,… Second,…" only for genuine enumerations of alternatives/problems
  (content taxonomy), never as the paper's own argument scaffolding. Three
  of six papers never use it.
- Run-in bold/italic mini-headings replace enumeration in denser papers:
  "Setup.", "Workloads.", "Previous approaches", "Ease of implementation".
  This is a strong human-systems marker.

## 9. Human quirks (hardest to fake — do NOT fabricate these)

- Idiom and jokes: LFL's introduction opens "To be or not to be lock free,
  that is the question."; "not necessarily a cakewalk"; "come to a grinding
  halt"; "for peace of mind in general".
- Confessional engineering detail: "we placed the initial lowering pass as
  early as we could muster while still ensuring that Reference could compile
  all benchmarks correctly".
- Blunt assertions without cushioning: "repeating this manual effort… is
  simply infeasible".
- Roughness survives: occasional typos, inconsistent hyphenation. (Never
  inject these deliberately — the lesson is that friction is tolerated, not
  that it should be manufactured.)
- Unforced modesty: "has not yet shown itself to be competitive".

## 10. Norms-vs-tells summary table

| Dimension | Human norm | LLM tell |
|---|---|---|
| Marker words (principled/crucially/seamless/…) | 0 in 74k words | any hit |
| comprehensive/novel/carefully (self-praise) | ≤1/paper | attached to every component |
| "key insight" | 0 — say what the idea IS | "Our key insight is that…" |
| furthermore/moreover | 0–5/paper, 0 in several | paragraph glue |
| Em-dash | 0 in half; one function where used | rhetorical "not X—but Y" everywhere |
| notably/importantly/interestingly | ≤2/paper | recurring sentence-initial |
| Contribution list | plain lead-in, unequal head-less bullets, §-refs | bolded parallel equal-length bullets |
| Abstract numbers | exact + conditioned | unconditioned superlatives |
| Negative results | flat + diagnosed | buried or damage-controlled |
| Named concepts | one system name, rest lowercase | multiple Title-Case brands |
| First person | 5–11 "we"/1000 words, epistemic uses | system as sole actor |
| Captions | bimodal lengths | uniform polished paragraphs |
| Paragraphs | 4–10 sentences, run-in bold heads | uniform short blocks, First/Second scaffolding |
| state-of-the-art | names the competitor | describes own system |
| Hedging | unmeasured claims only | every sentence qualified / hedge+boast combos |
| Repetition | invariant stated once + theorem | mantra restated in every section and caption |
| Antithesis | functional only | "X, not Y" aphorisms throughout |
| Quirks | idiom, confession, bluntness | frictionless prose |

**One-line summary:** human expert papers are concrete-first (worked example
or exact number within ~150 words of every major unit), first-person,
unevenly formatted, connective-light, superlative-free, and honest about
losses. The LLM signature is the inverse — uniform geometry (paragraphs,
bullets, captions), marker adverbs, self-praising adjectives, unconditioned
claims, Title-Case branding, mantra repetition, and zero friction.

## 11. Case-study calibration (GVR paper, 2026-08)

A real manuscript scored **0 on all Family A lexical markers** yet still
read AI-flavored via Family B: invariant mantra ×10+ (`exact`×96,
`authoriz-`×16), "X, not Y" antitheses ×17, "… are supplementary" template
×14, we/our only 19 total (~2.4/1000 words), caption word counts
[96,67,47,19,68,68,126,64,123] (uniform essays), italic-headed parallel
contribution bullets, ~10 Title-Case coinages, generic opener with zero
worked examples. Lesson: **run Family B always; Family A alone clears
nothing.**
