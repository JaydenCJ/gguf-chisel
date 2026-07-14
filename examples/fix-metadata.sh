#!/usr/bin/env bash
# Walkthrough: the typical "fix a published quant" workflow, end to end,
# against a generated sample model. Offline and idempotent.
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v gguf-chisel >/dev/null 2>&1; then
  BIN=(gguf-chisel)
else
  BIN=(cargo run --quiet --)
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/gguf-chisel-example.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
MODEL="$WORK/model.gguf"

echo "== 1. generate a sample model and inspect it"
"${BIN[@]}" sample "$MODEL"
"${BIN[@]}" show "$MODEL"

echo
echo "== 2. the classic one-liner: fix a wrong context length, in place"
"${BIN[@]}" set "$MODEL" sample.context_length=32768

echo
echo "== 3. install a better chat template (grows the head: rewrite once,"
echo "==    reserving 4 KiB so every future edit stays in place)"
"${BIN[@]}" template set "$MODEL" --file examples/system-default.jinja --rewrite --reserve 4K
"${BIN[@]}" template check "$MODEL"

echo
echo "== 4. batch fixes from a JSON patch document (atomic)"
"${BIN[@]}" apply "$MODEL" examples/patch.json

echo
echo "== 5. these edits now fit into the reserved headroom — no rewrite"
"${BIN[@]}" set "$MODEL" "general.name=Sample (fixed)" general.url=https://example.test/model

echo
echo "== 6. final state"
"${BIN[@]}" verify "$MODEL"
"${BIN[@]}" get "$MODEL" sample.context_length --json
