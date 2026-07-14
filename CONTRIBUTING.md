# Contributing to gguf-chisel

Thanks for your interest in improving gguf-chisel. Issues, discussions and pull requests are all welcome.

## Getting started

Prerequisites: Rust 1.75 or newer (stable toolchain).

```bash
git clone https://github.com/JaydenCJ/gguf-chisel.git
cd gguf-chisel
cargo build
cargo test
bash scripts/smoke.sh
```

`scripts/smoke.sh` exercises the whole CLI against a generated sample model — including a byte-level check that in-place edits never touch the tensor region. It finishes in well under a minute and must print `SMOKE OK`.

## Before you open a pull request

1. `cargo fmt` — formatting is enforced.
2. `cargo clippy --all-targets -- -D warnings` — clippy must be clean.
3. `cargo test` — the 75 unit tests and 15 CLI integration tests must pass.
4. `bash scripts/smoke.sh` — the smoke test must print `SMOKE OK`.
5. Add tests for behavior changes. All format logic lives in pure modules (`types`, `reader`, `writer`, `patch`, `template`, `json`, `verify`) that are easy to unit-test; please keep it that way.

## Ground rules

- Keep dependencies at zero. gguf-chisel is std-only by design — the GGUF codec, JSON codec and template linter are all in-tree. Adding a dependency needs a very strong justification in the PR description.
- No network calls, ever, and no telemetry. The tool reads and writes local files, nothing else.
- Never risk tensor data. Any change to `writer` must preserve the invariant that an in-place head ends exactly at the original data offset; the planner tests and the smoke test's `cmp` check both guard it.
- Code comments and doc comments are written in English.
- Compatibility first: files written by gguf-chisel must stay loadable by mainstream GGUF runtimes. Tool-specific state is confined to the single managed `chisel.pad` key.

## Reporting bugs

Please include the `gguf-chisel --version` output, the `show` and `verify` output for the affected file (or `dump` if metadata content matters), and the exact command you ran. For fit-planner bugs, `--dry-run` output plus the file's `data at:` / `headroom` lines from `show` usually pinpoint the problem.

## Security

If you find a security issue (e.g. a crafted file that causes out-of-bounds behavior), please do not open a public issue. Use GitHub's private vulnerability reporting on this repository instead.
