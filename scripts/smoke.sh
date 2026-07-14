#!/usr/bin/env bash
# Smoke test: builds gguf-chisel, then runs the whole CLI surface against a
# generated sample model — in-place patching (with byte-level proof that the
# tensor region never changes), growth/rewrite/reserve, JSON dump + apply,
# chat-template lint/set/show, dry-run exit codes and structural verify.
# Self-contained: temp dirs only, no network, idempotent.
set -euo pipefail

cd "$(dirname "$0")/.."

fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

echo "[smoke] building..."
cargo build --quiet
BIN="$PWD/target/debug/gguf-chisel"

WORK=$(mktemp -d "${TMPDIR:-/tmp}/gguf-chisel-smoke.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
MODEL="$WORK/model.gguf"

# --- 1. version/help sanity ---------------------------------------------------
"$BIN" --version | grep -q '^gguf-chisel 0\.1\.0$' || fail "--version mismatch"
"$BIN" --help | grep -q 'COMMANDS:' || fail "--help missing sections"
echo "[smoke] version/help OK"

# --- 2. sample -> show -> verify ------------------------------------------------
"$BIN" sample "$MODEL" | grep -q '2 tensors, 9 metadata keys' || fail "sample summary"
"$BIN" show "$MODEL" --tensors > "$WORK/show.out"
grep -q 'gguf:       v3 little-endian' "$WORK/show.out" || fail "show missing version"
grep -q 'sample.context_length' "$WORK/show.out" || fail "show missing key"
grep -q 'token_embd.weight' "$WORK/show.out" || fail "show missing tensor"
"$BIN" verify "$MODEL" | grep -q ': OK' || fail "fresh sample must verify OK"
echo "[smoke] sample/show/verify OK"

# --- 3. get: bare, json, raw ---------------------------------------------------
[ "$("$BIN" get "$MODEL" sample.context_length)" = "4096" ] || fail "get bare value"
"$BIN" get "$MODEL" sample.context_length --json \
  | grep -q '"type":"u32","value":4096' || fail "get --json"
"$BIN" get "$MODEL" tokenizer.chat_template --raw > "$WORK/tpl.jinja"
grep -q '<|im_start|>' "$WORK/tpl.jinja" || fail "get --raw template"
echo "[smoke] get OK"

# --- 4. in-place set: tensor region must be bit-identical ----------------------
DATA_START=$("$BIN" dump "$MODEL" | grep -o '"data_start": [0-9]*' | grep -o '[0-9]*$')
[ -n "$DATA_START" ] || fail "dump missing data_start"
SIZE_BEFORE=$(wc -c < "$MODEL")
tail -c +$((DATA_START + 1)) "$MODEL" > "$WORK/data.before"
"$BIN" set "$MODEL" sample.context_length=32768 > "$WORK/set.out"
grep -q 'u32 4096 -> u32 32768' "$WORK/set.out" || fail "set summary"
grep -q 'in place' "$WORK/set.out" || fail "same-size set must be in place"
[ "$(wc -c < "$MODEL")" = "$SIZE_BEFORE" ] || fail "in-place set changed file size"
tail -c +$((DATA_START + 1)) "$MODEL" > "$WORK/data.after"
cmp -s "$WORK/data.before" "$WORK/data.after" || fail "tensor bytes changed on in-place set"
echo "[smoke] in-place set OK (tensor region bit-identical)"

# --- 5. growth: refused without --rewrite, streamed with it --------------------
if "$BIN" set "$MODEL" "general.description=$(printf 'x%.0s' $(seq 1 2000))" \
    > /dev/null 2> "$WORK/grow.err"; then
  fail "oversized growth accepted without --rewrite"
fi
grep -q -- '--rewrite' "$WORK/grow.err" || fail "growth error must suggest --rewrite"
"$BIN" template set "$MODEL" --preset llama3 --rewrite --reserve 4K > "$WORK/rw.out"
grep -q 'rewrote' "$WORK/rw.out" || fail "rewrite not reported"
grep -q '+4096 reserved' "$WORK/rw.out" || fail "reserve not reported"
tail -c 160 "$MODEL" > "$WORK/data.rewritten"
cmp -s "$WORK/data.before" "$WORK/data.rewritten" || fail "rewrite altered tensor bytes"
# the reserved headroom must absorb the next edit in place
"$BIN" set "$MODEL" "general.name=Sample 32k" | grep -q 'in place' \
  || fail "reserved headroom not used"
echo "[smoke] growth/rewrite/reserve OK"

# --- 6. rm + rename + dry-run exit codes ---------------------------------------
"$BIN" rm "$MODEL" sample.block_count > /dev/null || fail "rm failed"
"$BIN" rename "$MODEL" general.file_type general.quant_type > /dev/null || fail "rename"
"$BIN" show "$MODEL" | grep -q 'general.quant_type' || fail "rename not visible"
"$BIN" set "$MODEL" sample.context_length=65536 --dry-run > /dev/null \
  || fail "dry-run fit should exit 0"
set +e
"$BIN" set "$MODEL" "big.blob=$(printf 'y%.0s' $(seq 1 9000))" --dry-run > /dev/null
CODE=$?
set -e
[ "$CODE" = "3" ] || fail "dry-run overflow should exit 3, got $CODE"
echo "[smoke] rm/rename/dry-run OK"

# --- 7. apply a JSON patch document --------------------------------------------
cat > "$WORK/patch.json" <<'EOF'
{
  "set": {
    "sample.context_length": 131072,
    "general.tag": {"type": "str", "value": "smoke-fixed"}
  }
}
EOF
"$BIN" apply "$MODEL" "$WORK/patch.json" > /dev/null || fail "apply failed"
[ "$("$BIN" get "$MODEL" general.tag)" = "smoke-fixed" ] || fail "apply not persisted"
echo "[smoke] apply OK"

# --- 8. template lint: broken templates are caught and never installed ----------
printf '{%% for m in messages %%}{{ m }} add_generation_prompt' > "$WORK/broken.jinja"
if "$BIN" template check --file "$WORK/broken.jinja" > "$WORK/lint.out"; then
  fail "broken template passed lint"
fi
grep -q "unclosed '{% for %}'" "$WORK/lint.out" || fail "lint missing diagnosis"
if "$BIN" template set "$MODEL" --file "$WORK/broken.jinja" --rewrite > /dev/null 2>&1; then
  fail "broken template was installed"
fi
"$BIN" template show "$MODEL" | grep -q '<|start_header_id|>' || fail "template show"
"$BIN" template check "$MODEL" | grep -q 'template OK' || fail "installed template lints"
echo "[smoke] template lint/set/show OK"

# --- 9. verify catches corruption ----------------------------------------------
head -c $(( $(wc -c < "$MODEL") - 100 )) "$MODEL" > "$WORK/cut.gguf"
if "$BIN" verify "$WORK/cut.gguf" > "$WORK/verify.out"; then
  fail "truncated file verified OK"
fi
grep -q 'error:' "$WORK/verify.out" || fail "verify missing error line"
"$BIN" verify "$MODEL" | grep -q ': OK' || fail "final model must verify OK"
echo "[smoke] verify OK"

echo "SMOKE OK"
