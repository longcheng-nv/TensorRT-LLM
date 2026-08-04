#!/usr/bin/env bash
# de-ai-flavor-paper: grep battery + scorecard for a LaTeX/Markdown manuscript.
# Usage: detect_ai_flavor.sh <file.tex> [more files...]
# Targets are calibrated against six pre-2022 PPoPP Best Papers (~74k words).
set -u

[ $# -ge 1 ] || { echo "usage: $0 <file.tex> [...]" >&2; exit 1; }

for f in "$@"; do
  [ -r "$f" ] || { echo "cannot read: $f" >&2; continue; }
  echo "==================================================================="
  echo "FILE: $f"
  words=$(wc -w < "$f")
  echo "word count (raw, incl. markup): $words"
  kw=$(( words / 1000 )); [ "$kw" -lt 1 ] && kw=1

  echo
  echo "--- Family A: classic LLM lexical markers (target: 0, except noted) ---"
  for w in comprehensive novel principled crucially notably importantly \
           furthermore moreover systematically extensive leverage delve \
           seamless holistic meticulous pivotal paradigm landscape realm \
           myriad plethora versatile cutting-edge substantially; do
    c=$(grep -oiw "$w" "$f" | wc -l)
    [ "$c" -gt 0 ] && printf '  %-18s %3d  <-- check each use\n' "$w" "$c"
  done
  for p in 'key insight' 'extensive experiments' 'in conclusion' \
           'state-of-the-art' 'we systematically'; do
    c=$(grep -oiE "$p" "$f" | wc -l)
    [ "$c" -gt 0 ] && printf '  %-22s %3d  <-- self-descriptive use is a flag\n' "$p" "$c"
  done
  echo "  (unlisted = 0 hits)"

  echo
  echo "--- Family B: structural tells ---"

  # B2 antithesis
  b2a=$(grep -cE ', not ' "$f"); b2b=$(grep -cE 'but (not|never)' "$f")
  echo "  B2 antithesis  ', not '=$b2a  'but not/never'=$b2b   (target: <=5 total; keep <=3 load-bearing)"

  # B3 template tails
  echo "  B3 repeated sentence templates (top repeated 3-word sentence tails):"
  grep -oE '[a-z-]+ [a-z-]+ (is|are) (supplementary|omitted|deferred|future work)' "$f" \
    | sort | uniq -c | sort -rn | head -5 | sed 's/^/      /'

  # B4 first person
  b4=$(grep -oiE '\bwe\b|\bour\b' "$f" | wc -l)
  echo "  B4 we/our total=$b4  (~$(( b4 / kw ))/1000w; human norm: >=8/1000w combined, with epistemic uses)"

  # B5 caption uniformity
  if grep -q '\\caption{' "$f"; then
    echo -n "  B5 caption word counts: "
    python3 - "$f" <<'EOF'
import re,sys
t=open(sys.argv[1]).read()
caps=[len(c.split()) for c in re.findall(r'\\caption\{(.*?)\}\s*\n', t, re.S)]
print(caps, " (human norm: bimodal; uniform 50-130w essays = flag)")
EOF
  fi

  # B1 mantra candidates: most repeated content words
  echo "  B1 top repeated technical unigrams (manual check for mantra repetition):"
  tr -cs 'A-Za-z-' '\n' < "$f" | tr 'A-Z' 'a-z' \
    | grep -vwE 'the|a|an|and|or|of|to|in|for|is|are|with|by|on|we|our|that|this|it|as|at|be|from|its|not|all|per|one|two|use|uses|used|only|but|section|figure|table|equation|begin|end|item|label|ref|cite|citep|textbf|emph|centering|caption|description|tabular|toprule|midrule|bottomrule|small|times|left|right|sum|frac|mathcal|mathbf|rm|max|min|log' \
    | awk 'length($0)>3' | sort | uniq -c | sort -rn | head -8 | sed 's/^/      /'

  # B7 Title-Case coinage candidates
  echo "  B7 capitalized multi-word coinage candidates (human norm: ONE brand):"
  grep -oE '\b([A-Z][a-z]+-)?[A-Z][A-Za-z]+-[A-Z][A-Za-z]+\b|\\textsc\{[^}]*\}' "$f" \
    | sort | uniq -c | sort -rn | head -8 | sed 's/^/      /'

  # B10 connectives
  b10=$(grep -ciE '^ *(However|Moreover|Furthermore|Additionally|Notably|Importantly),' "$f")
  echo "  B10 sentence/line-initial connectives: $b10  (human norm: sparse)"

  # B12 em-dashes
  b12=$(grep -o -- '---' "$f" | wc -l)
  echo "  B12 em-dashes (---): $b12  (0 in half the human corpus; check for rhetorical use)"

  echo
  echo "--- Manual-review items (cannot be fully grepped) ---"
  echo "  B1  mantra: is any invariant/slogan restated in abstract+intro+captions+conclusion?"
  echo "  B6  contribution bullets: bold/italic heads? equal length? parallel syntax?"
  echo "  B8  first page: worked example or exact number in first ~150 words? generic opener?"
  echo "  B9  results sections: scope qualifiers on measured claims (should be flat)?"
  echo "  B11 paragraph geometry: uniform 3-4-sentence blocks? First/Second scaffolding?"
  echo "  A-x losses/negative results reported flat with diagnosis? (GOOD if yes - keep)"
  echo "  policy: venue-required generative-AI disclosure present in Acknowledgments?"
done
