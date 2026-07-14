# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-13

### Added

- GGUF head parser for v2/v3 little-endian files: all 13 metadata value types (including nested arrays), tensor descriptors, effective-alignment handling, and sanity caps that turn corrupt length fields into readable errors instead of OOMs. Big-endian and v1 files are detected and rejected with clear messages.
- In-place fit planner: edited heads are landed exactly on the original data offset by combining alignment slack with a managed, byte-exactly sized `chisel.pad` key — so metadata edits never move or rewrite tensor data.
- `set` / `rm` / `rename` with `KEY=VALUE` and `KEY=TYPE:VALUE` syntax: existing keys keep their wire type, new keys use documented inference, range checks name the violated bounds, and every multi-op invocation applies atomically.
- Growth handling: edits that outgrow the head space are refused with a clear message unless `--rewrite` is given, which streams tensor data verbatim (never re-encoded); `--reserve N` leaves headroom so future edits stay in place. `-o/--output` writes to a new file instead.
- `--dry-run` fit probing with scriptable exit codes: 0 = fits in place, 3 = needs a rewrite.
- Chat-template toolkit: `template show` / `set` / `check` / `presets` with six built-in presets and a Jinja-subset linter (delimiter balance, block matching, raw blocks, quote-aware scanning, `messages` / `add_generation_prompt` heuristics). Broken templates are refused before any write.
- `dump` (pretty JSON with exact u64 integers) and `apply` (JSON patch documents with delete/rename/set sections, from file or stdin).
- `verify`: duplicate keys/tensors, key charset, alignment sanity, tensor offset alignment, extent and overlap checks against a built-in ggml type-size table, truncation detection, and template lint surfacing.
- `sample`: a deterministic ~1 KiB GGUF generator for pipeline tests and CI.
- Zero runtime dependencies: GGUF codec, JSON parser/encoder and template linter are all implemented against std.
- Test suite: 75 unit tests, 15 CLI integration tests (including byte-identity proofs for the tensor region), and `scripts/smoke.sh`.

[0.1.0]: https://github.com/JaydenCJ/gguf-chisel/releases/tag/v0.1.0
