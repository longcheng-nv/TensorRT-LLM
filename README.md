# SWE-bench Agentic Synthetic Datasets

This branch contains only the SWE-bench prompt datasets used for long-context
inference comparison. It intentionally excludes TensorRT-LLM source files so the
branch can be consumed as a lightweight dataset artifact.

## Directory Layout

```text
longseqtasks/
generated_longseqtasks/
```

`longseqtasks/` contains the original single-turn SWE-bench long-context prompt
files from the GVR Top-K supplementary materials:

```text
swe_bench_16k.jsonl
swe_bench_32k.jsonl
swe_bench_64k.jsonl
swe_bench_100k.jsonl
```

Each file has 5 JSONL entries. Each entry uses chat-style fields:

```json
{"system": "", "user": "..."}
```

`generated_longseqtasks/` contains derived two-turn agentic workflow datasets.
Each generated file corresponds to one original length bucket and keeps the same
entry order as its source file.

## Generated Dataset Format

Each generated JSONL row contains:

- `source`: original file, entry index, issue title, issue excerpt, and source
  paths extracted from the prompt.
- `lengths`: character and token length metadata for each turn and the combined
  prompt.
- `conversation`: four chat messages in this order:
  `system`, `user` turn 1, `assistant` turn 1, `user` turn 2.
- `combined_text`: the concatenation used for the two-turn prompt context:
  first user prompt, first assistant output, and second user prompt.

The second user prompt is synthesized from the first assistant output and is
designed to continue the same coding-agent workflow. It asks the agent to refine
the draft answer into concrete `edit_file` commands, add regression coverage,
preserve compatibility-sensitive behavior, and provide a validation plan.

## File Name Lengths

Generated file names include token-length ranges measured with the
`deepseek-ai/DeepSeek-V3.2-Exp` tokenizer:

```text
swe_bench_<bucket>_2turn_isl1tok<range>_osl1tok<range>_isl2tok<range>_ctxtok<range>.jsonl
```

Fields:

- `isl1tok`: token range of the original first-turn user prompt.
- `osl1tok`: token range of the synthesized first-turn assistant output.
- `isl2tok`: token range of the synthesized second-turn user prompt.
- `ctxtok`: token range of the full two-turn concatenated context.

The token ranges are computed over the stored text fields using the DeepSeek
V3.2 tokenizer JSON. They are intended for quick dataset selection and
comparison; inference runners may still apply their own chat template when
building model-ready input IDs.

## Generation Method

The generated dataset was constructed from each original `longseqtasks` entry:

1. Read one original SWE-bench entry as the first-turn user prompt.
2. Extract issue metadata such as the title, issue excerpt, and source file
   paths from the prompt.
3. Synthesize a first-turn assistant response in the same command-oriented
   coding-agent style expected by SWE-bench prompts.
4. Build a second-turn user prompt from that assistant output, asking for a
   merge-ready refinement and validation plan.
5. Concatenate turn 1 user input, turn 1 assistant output, and turn 2 user input
   into `combined_text`.
6. Compute token-length metadata with the DeepSeek V3.2 tokenizer and write the
   output JSONL files.

The generated datasets preserve the 5-entry cardinality and ordering of the
original files so inference scripts can pair rows by bucket and entry index.
