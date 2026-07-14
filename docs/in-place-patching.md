# How in-place patching works

This note explains why gguf-chisel can edit the metadata of a 40 GB GGUF file
by writing only a few kilobytes, and exactly when it cannot.

## The layout fact everything rests on

A GGUF file is a *head* followed by a *tensor-data section*:

| Region | Contents |
|---|---|
| header | magic `GGUF`, version (u32), tensor count (u64), metadata count (u64) |
| metadata | key/value pairs: string key, u32 type code, value |
| tensor descriptors | name, dims, ggml type, **offset** |
| padding | zero bytes up to the next multiple of `general.alignment` (default 32) |
| tensor data | the actual weights |

The crucial detail: every tensor descriptor's `offset` is **relative to the
start of the tensor-data section**, not to the start of the file. The data
section itself starts at `align_up(end_of_descriptors, alignment)`.

So a runtime finds tensor bytes by computing `data_start + offset`. If an
edited head still *ends at exactly the same aligned position*, every stored
offset stays valid and the tensor bytes never have to move — regardless of
how much the metadata inside the head changed.

## The fit planner

After applying your edits in memory, gguf-chisel serializes the new head and
compares its length `end` against the file's original `data_start`:

1. **`end == data_start`** — perfect fit, write the head, done.
2. **`data_start - end < alignment`** — the difference is legal alignment
   padding; write the head plus zero fill.
3. **`data_start - end >= 30`** — too big for padding alone. The planner
   appends a managed string key, `chisel.pad`, whose value length is tuned
   byte-exactly so the head ends at `data_start` again. The pad key costs
   30 bytes of overhead (8-byte key length + 10 key bytes + 4-byte type +
   8-byte string length), which is why gaps of at least 30 bytes are always
   fillable when the alignment is the default 32.
4. **`end > data_start`** — the head has outgrown its space. In-place editing
   is impossible without corrupting the first tensor, so gguf-chisel refuses
   and asks for `--rewrite`.

`chisel.pad` is stripped before every plan, so shrinking edits automatically
reclaim previously reserved space, and repeated edits never accumulate pads.
Runtimes ignore unknown keys, so the pad is inert. Because the pad value can
be any length, it doubles as **reserved headroom**: `--reserve 4K` on a
rewrite leaves 4096 spare bytes, and every later edit that fits in them is a
pure in-place head write.

There is one pathological corner: a custom alignment smaller than 30 can
produce a gap too large for alignment slack but too small for the pad key.
The planner never guesses — it reports a rewrite. With the default alignment
of 32 this cannot happen.

## When a rewrite does happen

`--rewrite` serializes the new head (plus any `--reserve` pad), zero-pads to
alignment, then streams the entire old data section through unchanged with a
buffered copy. Offsets are relative, so descriptors need no adjustment and
tensor bytes are **copied verbatim, never re-encoded or re-quantized**. With
no `-o`, the rewrite goes to a temporary file in the same directory and is
renamed over the original.

`--dry-run` runs only the planner and encodes the verdict in the exit code —
`0` fits in place, `3` needs a rewrite — so scripts can decide whether to
schedule the expensive path before touching anything.

## Safety properties

- In-place writes cover bytes `0..data_start` only; the tensor region is
  never opened for writing. The integration tests and `scripts/smoke.sh`
  assert bit-identity of the data region after edits.
- All multi-key operations validate against a working copy first and apply
  atomically — a bad key in position three leaves the file untouched.
- `general.alignment` is refused as an edit target in 0.1.0: changing it
  moves `data_start` and would require relocating every tensor.
- Parser sanity caps (string/array/count limits) turn corrupt length fields
  into readable errors instead of multi-gigabyte allocations.
