#!/usr/bin/env python3
"""Surface passages that deserve human review in an ISC/HPC manuscript.

This is a diagnostic linter, not an AI detector and not an automatic rewriter.
It intentionally reports context so an editor can keep legitimate domain usage.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Finding:
    path: str
    line: int
    rule: str
    message: str
    excerpt: str


STOCK_PATTERNS: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    ("throat-clearing", "State the fact directly.", re.compile(
        r"\b(?:it is (?:important|interesting|worthwhile|worth noting) to note that|"
        r"it should be noted that|needless to say)\b", re.I)),
    ("generic-landscape", "Replace the generic field opener with the concrete object or bottleneck.", re.compile(
        r"\b(?:in (?:today['’]s|the) (?:rapidly |ever[- ]?)?(?:evolving )?landscape|"
        r"in an era of|as technology continues to evolve)\b", re.I)),
    ("writing-meta", "Describe the research action or finding, not the act of writing.", re.compile(
        r"\b(?:this (?:paper|work) (?:delves? into|aims to explore|seeks to|provides a comprehensive overview of))\b", re.I)),
    ("promotional-phrase", "Replace promotional wording with the mechanism or measured effect.", re.compile(
        r"\b(?:seamlessly leverag(?:e|es|ed|ing)|paves? the way|ushers? in a new era|"
        r"transformative potential|game[- ]chang(?:ing|er)|paradigm[- ]shifting|"
        r"groundbreaking|revolutionary)\b", re.I)),
    ("empty-interpretation", "State the concrete implication or remove the sentence.", re.compile(
        r"\b(?:this (?:result|finding|observation) (?:underscores?|highlights?|demonstrates?) the (?:importance|need|potential))\b", re.I)),
    ("unsupported-certainty", "Replace certainty language with evidence or a scoped claim.", re.compile(
        r"\b(?:clearly|obviously|undeniably|without a doubt)\b", re.I)),
)

REVIEW_WORDS = re.compile(
    r"\b(?:novel|significant(?:ly)?|important(?:ly)?|notably|robust|comprehensive|"
    r"seamless(?:ly)?|efficient(?:ly)?|scalable|state[- ]of[- ]the[- ]art)\b", re.I
)

COMMAND_WITH_TEXT = re.compile(
    r"\\(?:section|subsection|subsubsection|paragraph|caption|title|author)\*?\{([^{}]*)\}"
)


def latex_to_prose(text: str) -> str:
    """Remove common non-prose LaTeX while retaining visible sentence text."""
    text = re.sub(r"(?<!\\)%.*", "", text)

    def blank_block(match: re.Match[str]) -> str:
        return "\n" * match.group(0).count("\n")

    text = re.sub(
        r"\\begin\{(?P<env>equation\*?|align\*?|gather\*?|verbatim|lstlisting|algorithm\*?|"
        r"figure\*?|table\*?|tikzpicture|tabular\*?)\}.*?\\end\{(?P=env)\}",
        blank_block, text, flags=re.S,
    )
    text = re.sub(r"\$\$.*?\$\$|\$[^$]*\$|\\\[.*?\\\]", " [MATH] ", text, flags=re.S)
    text = COMMAND_WITH_TEXT.sub(lambda match: match.group(1), text)
    text = re.sub(
        r"\\(?:cite|ref|eqref|autoref|label|url|href)\w*\{[^{}]*\}(?:\{[^{}]*\})?",
        " [REF] ", text,
    )
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^\]]*\])?", " ", text)
    return text.replace("{", "").replace("}", "")


def line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def excerpt(text: str, start: int, end: int, radius: int = 90) -> str:
    low = max(0, start - radius)
    high = min(len(text), end + radius)
    value = re.sub(r"\s+", " ", text[low:high]).strip()
    return value[:260]


def sentence_spans(text: str):
    pattern = re.compile(r"(?:^|(?<=[.!?]))\s+(?=[A-Z0-9])", re.M)
    starts = [0] + [match.end() for match in pattern.finditer(text)]
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(text)
        sentence = text[start:end].strip()
        if sentence:
            yield start, end, sentence


def scan(path: Path, long_sentence_words: int) -> list[Finding]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    text = latex_to_prose(raw) if path.suffix.lower() in {".tex", ".ltx"} else raw
    findings: list[Finding] = []

    for rule, message, pattern in STOCK_PATTERNS:
        for match in pattern.finditer(text):
            findings.append(Finding(
                str(path), line_number(text, match.start()), rule, message,
                excerpt(text, match.start(), match.end()),
            ))

    for match in REVIEW_WORDS.finditer(text):
        findings.append(Finding(
            str(path), line_number(text, match.start()), "evidence-word",
            f"Check whether ‘{match.group(0)}’ is defined, measured, or scoped in this passage.",
            excerpt(text, match.start(), match.end()),
        ))

    openers: list[tuple[str, int, int]] = []
    for start, end, sentence in sentence_spans(text):
        words = re.findall(r"[A-Za-z]+(?:[-’'][A-Za-z]+)*", sentence)
        if len(words) > long_sentence_words:
            findings.append(Finding(
                str(path), line_number(text, start), "long-sentence",
                f"Review this {len(words)}-word sentence for multiple logical jobs.",
                excerpt(text, start, end),
            ))
        if words:
            openers.append((" ".join(word.lower() for word in words[:2]), start, end))

    counts = Counter(item[0] for item in openers)
    tracked_openers = {"we present", "we propose", "we show", "we evaluate", "this paper", "in this"}
    for key, start, end in openers:
        if counts[key] >= 5 and key in tracked_openers:
            findings.append(Finding(
                str(path), line_number(text, start), "repeated-opener",
                f"The opener ‘{key}’ occurs {counts[key]} times; vary it only where repetition creates list-like cadence.",
                excerpt(text, start, end),
            ))

    for match in re.finditer(r"—", text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        line_end = len(text) if line_end < 0 else line_end
        if text[line_start:line_end].count("—") >= 4:
            findings.append(Finding(
                str(path), line_number(text, match.start()), "dash-density",
                "Review whether material qualifications should be separate sentences.",
                excerpt(text, line_start, line_end),
            ))

    unique = {}
    for item in findings:
        unique[(item.path, item.line, item.rule, item.excerpt)] = item
    return sorted(unique.values(), key=lambda item: (item.path, item.line, item.rule))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help=".tex, .md, or .txt files to inspect")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a readable report")
    parser.add_argument(
        "--long-sentence-words", type=int, default=50,
        help="report sentences longer than this many words (default: 50)",
    )
    args = parser.parse_args()

    missing = [str(path) for path in args.paths if not path.is_file()]
    if missing:
        print("Missing file(s): " + ", ".join(missing), file=sys.stderr)
        return 2

    findings = []
    for path in args.paths:
        findings.extend(scan(path, args.long_sentence_words))

    if args.json:
        print(json.dumps([asdict(item) for item in findings], indent=2, ensure_ascii=False))
    elif not findings:
        print("No heuristic findings. This does not replace a technical or editorial review.")
    else:
        for item in findings:
            print(f"{item.path}:{item.line}: [{item.rule}] {item.message}")
            print(f"  {item.excerpt}")
        print(f"\n{len(findings)} finding(s); review context before editing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
