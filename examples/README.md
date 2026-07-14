# Examples

Everything here runs offline against a generated sample model — no downloads.

| File | What it shows |
|---|---|
| `fix-metadata.sh` | The full workflow: generate a sample, patch a context length in place, install a chat template with reserved headroom, apply a JSON patch, verify. |
| `patch.json` | A JSON patch document for `gguf-chisel apply`: delete, rename and set in one atomic call. |
| `system-default.jinja` | A ChatML-style template that injects a default system prompt when the chat has none — install it with `template set --file`. |

Run the walkthrough:

```bash
bash examples/fix-metadata.sh
```

The script uses `gguf-chisel` from your `PATH` if installed, and falls back to
`cargo run` inside this repository otherwise.

Lint the example template without touching any model file:

```bash
gguf-chisel template check --file examples/system-default.jinja
```
