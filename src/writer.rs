//! Head serialization and the in-place fit planner.
//!
//! The whole point of gguf-chisel lives here: tensor offsets in a GGUF file
//! are relative to the start of the tensor-data section, so as long as an
//! edited head still ends at the *same* aligned data offset, the tensor data
//! never has to move. The planner makes edited heads land exactly there by
//! combining two mechanisms:
//!
//! 1. the format's own alignment slack (a gap smaller than `general.alignment`
//!    is implied padding and costs nothing), and
//! 2. a managed `chisel.pad` string key whose length is tuned byte-exactly to
//!    absorb any larger gap — and which doubles as reserved headroom for
//!    future edits.
//!
//! Only when the edited head *grows past* the old data offset is a rewrite
//! unavoidable; even then the tensor bytes are streamed through verbatim,
//! never re-encoded.

use crate::types::{align_up, GgufValue, KvPair, TensorInfo, MAGIC};
use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::Path;

/// The managed padding/headroom key. It is stripped and re-created on every
/// write, so user edits are never allowed to touch it directly.
pub const PAD_KEY: &str = "chisel.pad";

/// Wire overhead of the pad key with an empty value:
/// key length (8) + key bytes + value type (4) + string length (8).
pub fn pad_overhead() -> u64 {
    8 + PAD_KEY.len() as u64 + 4 + 8
}

fn put_u32(buf: &mut Vec<u8>, v: u32) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_u64(buf: &mut Vec<u8>, v: u64) {
    buf.extend_from_slice(&v.to_le_bytes());
}

fn put_str(buf: &mut Vec<u8>, s: &str) {
    put_u64(buf, s.len() as u64);
    buf.extend_from_slice(s.as_bytes());
}

fn put_value(buf: &mut Vec<u8>, v: &GgufValue) {
    use GgufValue::*;
    match v {
        U8(x) => buf.push(*x),
        I8(x) => buf.push(*x as u8),
        U16(x) => buf.extend_from_slice(&x.to_le_bytes()),
        I16(x) => buf.extend_from_slice(&x.to_le_bytes()),
        U32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        I32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        F32(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Bool(x) => buf.push(*x as u8),
        Str(s) => put_str(buf, s),
        U64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        I64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        F64(x) => buf.extend_from_slice(&x.to_le_bytes()),
        Array(elem, items) => {
            put_u32(buf, elem.code());
            put_u64(buf, items.len() as u64);
            for item in items {
                put_value(buf, item);
            }
        }
    }
}

/// Serialize a complete head (magic through the last tensor descriptor),
/// with no trailing padding. Deterministic: same input, same bytes.
pub fn serialize_head(version: u32, kvs: &[KvPair], tensors: &[TensorInfo]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(4096);
    buf.extend_from_slice(&MAGIC);
    put_u32(&mut buf, version);
    put_u64(&mut buf, tensors.len() as u64);
    put_u64(&mut buf, kvs.len() as u64);
    for kv in kvs {
        put_str(&mut buf, &kv.key);
        put_u32(&mut buf, kv.value.ty().code());
        put_value(&mut buf, &kv.value);
    }
    for t in tensors {
        put_str(&mut buf, &t.name);
        put_u32(&mut buf, t.dims.len() as u32);
        for d in &t.dims {
            put_u64(&mut buf, *d);
        }
        put_u32(&mut buf, t.type_code);
        put_u64(&mut buf, t.offset);
    }
    buf
}

/// Return `kvs` with any managed pad key removed.
pub fn strip_pad(kvs: &[KvPair]) -> Vec<KvPair> {
    kvs.iter().filter(|kv| kv.key != PAD_KEY).cloned().collect()
}

fn pad_kv(value_len: u64) -> KvPair {
    KvPair {
        key: PAD_KEY.to_string(),
        value: GgufValue::Str(" ".repeat(value_len as usize)),
    }
}

/// The planner's verdict for an edited head against a fixed data offset.
#[derive(Debug)]
pub enum FitPlan {
    /// The head fits: write `head` at offset 0, then `zero_fill` zero bytes.
    /// `head.len() + zero_fill == data_start`, so tensor data is untouched.
    InPlace { head: Vec<u8>, zero_fill: u64 },
    /// The head no longer fits before the tensor data; a rewrite is needed.
    NeedsRewrite { needed: u64, available: u64 },
}

/// Plan an in-place write of the edited metadata against the *existing*
/// `data_start`. Any pre-existing pad key is stripped first, so shrinking
/// edits reclaim previously reserved space automatically.
pub fn plan_in_place(
    version: u32,
    kvs: &[KvPair],
    tensors: &[TensorInfo],
    data_start: u64,
    alignment: u64,
) -> FitPlan {
    let mut kvs = strip_pad(kvs);
    let base = serialize_head(version, &kvs, tensors);
    let end0 = base.len() as u64;

    if end0 <= data_start {
        let gap = data_start - end0;
        if gap < alignment.max(1) {
            // Alignment slack alone absorbs the difference.
            return FitPlan::InPlace {
                head: base,
                zero_fill: gap,
            };
        }
        let overhead = pad_overhead();
        if gap >= overhead {
            // Tune the pad value so the head ends exactly at data_start.
            kvs.push(pad_kv(gap - overhead));
            let head = serialize_head(version, &kvs, tensors);
            debug_assert_eq!(head.len() as u64, data_start);
            return FitPlan::InPlace { head, zero_fill: 0 };
        }
        // Rare: alignment < 30 with a gap too big for slack but too small for
        // the pad key. Punting to a rewrite keeps the invariant simple.
    }

    FitPlan::NeedsRewrite {
        needed: end0,
        available: data_start,
    }
}

/// Overwrite only the head region of `path`. The caller guarantees (via
/// [`plan_in_place`]) that `head.len() + zero_fill` equals the old data
/// offset, so every byte from the tensor data onward is untouched.
pub fn write_in_place(path: &Path, head: &[u8], zero_fill: u64) -> io::Result<u64> {
    let mut f = OpenOptions::new().write(true).open(path)?;
    f.write_all(head)?;
    if zero_fill > 0 {
        f.write_all(&vec![0u8; zero_fill as usize])?;
    }
    f.sync_all()?;
    Ok(head.len() as u64 + zero_fill)
}

/// Build a complete file image in memory: head (+ optional reserved pad),
/// zero padding up to `alignment`, then `data`. Used by the sample builder
/// and by tests; the streaming path for big files is [`rewrite_file`].
pub fn build_full(
    version: u32,
    kvs: &[KvPair],
    tensors: &[TensorInfo],
    data: &[u8],
    alignment: u64,
    reserve: u64,
) -> Vec<u8> {
    let mut kvs = strip_pad(kvs);
    if reserve > 0 {
        kvs.push(pad_kv(reserve));
    }
    let mut buf = serialize_head(version, &kvs, tensors);
    let data_start = align_up(buf.len() as u64, alignment);
    buf.resize(data_start as usize, 0);
    buf.extend_from_slice(data);
    buf
}

/// Rewrite `src` into `dst` with edited metadata: serialize the new head
/// (reserving `reserve` bytes of headroom via the pad key), pad to alignment,
/// then stream the tensor-data section through verbatim. Tensor offsets are
/// relative to the data section, so they need no adjustment.
///
/// Returns `(head_len, copied_data_bytes)`.
pub fn rewrite_file(
    src: &Path,
    dst: &Path,
    g: &crate::reader::Gguf,
    reserve: u64,
) -> io::Result<(u64, u64)> {
    let mut kvs = strip_pad(&g.kvs);
    if reserve > 0 {
        kvs.push(pad_kv(reserve));
    }
    let head = serialize_head(g.version, &kvs, &g.tensors);
    let new_start = align_up(head.len() as u64, g.alignment);

    let mut input = File::open(src)?;
    input.seek(SeekFrom::Start(g.data_start))?;

    let out_file = File::create(dst)?;
    let mut out = BufWriter::new(out_file);
    out.write_all(&head)?;
    let fill = new_start - head.len() as u64;
    if fill > 0 {
        out.write_all(&vec![0u8; fill as usize])?;
    }
    let copied = io::copy(&mut input, &mut out)?;
    out.flush()?;
    out.get_ref().sync_all()?;
    Ok((head.len() as u64, copied))
}

/// Deterministic tiny GGUF used by `gguf-chisel sample`, the smoke test and
/// the examples: nine metadata keys (including a ChatML chat template) and
/// two small F32 tensors with a recognizable byte pattern.
pub fn sample_bytes(reserve: u64) -> Vec<u8> {
    use crate::types::kv;
    let kvs = vec![
        kv("general.architecture", GgufValue::Str("sample".into())),
        kv(
            "general.name",
            GgufValue::Str("gguf-chisel sample model".into()),
        ),
        kv("general.alignment", GgufValue::U32(32)),
        kv("general.file_type", GgufValue::U32(0)),
        kv("sample.context_length", GgufValue::U32(4096)),
        kv("sample.embedding_length", GgufValue::U32(8)),
        kv("sample.block_count", GgufValue::U32(1)),
        kv("tokenizer.ggml.model", GgufValue::Str("none".into())),
        kv(
            "tokenizer.chat_template",
            GgufValue::Str(
                crate::template::preset("chatml")
                    .expect("chatml preset exists")
                    .to_string(),
            ),
        ),
    ];
    let tensors = vec![
        TensorInfo {
            name: "token_embd.weight".into(),
            dims: vec![8, 4],
            type_code: 0, // F32: 128 bytes
            offset: 0,
        },
        TensorInfo {
            name: "output.weight".into(),
            dims: vec![8],
            type_code: 0, // F32: 32 bytes
            offset: 128,
        },
    ];
    let data: Vec<u8> = (0..160u32).map(|i| ((i * 7 + 3) % 256) as u8).collect();
    build_full(3, &kvs, &tensors, &data, 32, reserve)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::parse_head;
    use crate::types::{kv, GgufType};
    use std::fs;
    use std::path::PathBuf;

    fn tempdir(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("gguf-chisel-writer-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn demo_kvs() -> Vec<KvPair> {
        vec![
            kv("general.architecture", GgufValue::Str("demo".into())),
            kv("demo.context_length", GgufValue::U32(2048)),
            kv("demo.rope.freq_base", GgufValue::F32(10000.0)),
            kv("demo.flag", GgufValue::Bool(true)),
            kv(
                "demo.stops",
                GgufValue::Array(
                    GgufType::Str,
                    vec![
                        GgufValue::Str("</s>".into()),
                        GgufValue::Str("<eot>".into()),
                    ],
                ),
            ),
            kv("demo.big", GgufValue::U64(1 << 40)),
            kv("demo.neg", GgufValue::I64(-9)),
            kv("demo.pi", GgufValue::F64(std::f64::consts::PI)),
        ]
    }

    fn demo_tensor() -> TensorInfo {
        TensorInfo {
            name: "blk.0.attn.weight".into(),
            dims: vec![8, 4],
            type_code: 0,
            offset: 0,
        }
    }

    #[test]
    fn serialize_then_parse_roundtrips_deterministically() {
        let kvs = demo_kvs();
        let tensors = vec![demo_tensor()];
        let head = serialize_head(3, &kvs, &tensors);
        let g = parse_head(&head[..], head.len() as u64).expect("own output parses");
        assert_eq!(g.kvs, kvs);
        assert_eq!(g.tensors, tensors);

        let a = serialize_head(3, &demo_kvs(), &[demo_tensor()]);
        let b = serialize_head(3, &demo_kvs(), &[demo_tensor()]);
        assert_eq!(a, b);
    }

    #[test]
    fn build_full_aligns_data_and_preserves_payload() {
        let data = [0xAAu8; 64];
        let img = build_full(3, &demo_kvs(), &[demo_tensor()], &data, 32, 0);
        let g = parse_head(&img[..], img.len() as u64).unwrap();
        assert_eq!(g.data_start % 32, 0);
        assert_eq!(&img[g.data_start as usize..], &data[..]);
    }

    #[test]
    fn planner_handles_no_change_small_shrink_and_growth() {
        let img = build_full(3, &demo_kvs(), &[demo_tensor()], &[0u8; 32], 32, 0);
        let g = parse_head(&img[..], img.len() as u64).unwrap();
        match plan_in_place(3, &g.kvs, &g.tensors, g.data_start, g.alignment) {
            FitPlan::InPlace { head, zero_fill } => {
                assert_eq!(head.len() as u64 + zero_fill, g.data_start);
            }
            other => panic!("expected InPlace, got {other:?}"),
        }

        // Removing "demo.flag" (bool) shrinks the head by 8+9+4+1 = 22 bytes;
        // combined slack can exceed 32, so accept either mechanism but insist
        // the head still ends exactly at data_start.
        let img = build_full(3, &demo_kvs(), &[demo_tensor()], &[0u8; 32], 32, 0);
        let g = parse_head(&img[..], img.len() as u64).unwrap();
        let kvs: Vec<KvPair> = g
            .kvs
            .iter()
            .filter(|kv| kv.key != "demo.flag")
            .cloned()
            .collect();
        match plan_in_place(3, &kvs, &g.tensors, g.data_start, g.alignment) {
            FitPlan::InPlace { head, zero_fill } => {
                assert_eq!(head.len() as u64 + zero_fill, g.data_start);
                assert!(zero_fill < 32, "zero fill stays below alignment");
            }
            other => panic!("expected InPlace, got {other:?}"),
        }
    }

    #[test]
    fn large_shrink_inserts_a_byte_exact_pad_key() {
        // Dropping the chat-template-sized string frees far more than one
        // alignment unit, so the planner must synthesize chisel.pad.
        let mut kvs = demo_kvs();
        kvs.push(kv("demo.blob", GgufValue::Str("x".repeat(500))));
        let img = build_full(3, &kvs, &[demo_tensor()], &[0u8; 32], 32, 0);
        let g = parse_head(&img[..], img.len() as u64).unwrap();
        let shrunk: Vec<KvPair> = g
            .kvs
            .iter()
            .filter(|kv| kv.key != "demo.blob")
            .cloned()
            .collect();
        match plan_in_place(3, &shrunk, &g.tensors, g.data_start, g.alignment) {
            FitPlan::InPlace { head, zero_fill } => {
                assert_eq!(zero_fill, 0, "pad key lands the head exactly");
                assert_eq!(head.len() as u64, g.data_start);
                let reparsed = parse_head(&head[..], head.len() as u64).unwrap();
                assert!(reparsed.get(PAD_KEY).is_some(), "pad key present");
            }
            other => panic!("expected InPlace, got {other:?}"),
        }

        let img = build_full(3, &demo_kvs(), &[demo_tensor()], &[0u8; 32], 32, 0);
        let g = parse_head(&img[..], img.len() as u64).unwrap();
        let mut kvs = g.kvs.clone();
        kvs.push(kv("demo.long", GgufValue::Str("y".repeat(4096))));
        match plan_in_place(3, &kvs, &g.tensors, g.data_start, g.alignment) {
            FitPlan::NeedsRewrite { needed, available } => {
                assert!(needed > available);
                assert_eq!(available, g.data_start);
            }
            other => panic!("expected NeedsRewrite, got {other:?}"),
        }
    }

    #[test]
    fn existing_pad_is_reclaimed_before_planning() {
        // A file built with 200 reserved bytes can absorb a 100-byte growth
        // in place: the pad shrinks instead of the data moving.
        let img = build_full(3, &demo_kvs(), &[demo_tensor()], &[0u8; 32], 32, 200);
        let g = parse_head(&img[..], img.len() as u64).unwrap();
        assert!(g.get(PAD_KEY).is_some(), "reserve materialized as pad");
        let mut kvs = g.kvs.clone();
        kvs.push(kv("demo.extra", GgufValue::Str("z".repeat(100))));
        match plan_in_place(3, &kvs, &g.tensors, g.data_start, g.alignment) {
            FitPlan::InPlace { head, zero_fill } => {
                assert_eq!(head.len() as u64 + zero_fill, g.data_start);
                let reparsed = parse_head(&head[..], head.len() as u64).unwrap();
                let pad = reparsed.get(PAD_KEY).expect("pad still present");
                if let GgufValue::Str(s) = pad {
                    assert!(s.len() < 200, "pad shrank to absorb the growth");
                } else {
                    panic!("pad is not a string");
                }
            }
            other => panic!("expected InPlace, got {other:?}"),
        }
    }

    #[test]
    fn narrow_window_with_tiny_alignment_falls_back_to_rewrite() {
        // alignment 8: a 16-byte gap is too big for slack (< 8 required) and
        // too small for the 30-byte pad key overhead. The planner must not
        // guess — it reports a rewrite.
        let mut kvs = vec![kv("general.alignment", GgufValue::U32(8))];
        kvs.extend(demo_kvs());
        let base_len = serialize_head(3, &kvs, &[]).len() as u64;
        let data_start = base_len + 16;
        match plan_in_place(3, &kvs, &[], data_start, 8) {
            FitPlan::NeedsRewrite { needed, available } => {
                assert_eq!(needed, base_len);
                assert_eq!(available, data_start);
            }
            other => panic!("expected NeedsRewrite, got {other:?}"),
        }
    }

    #[test]
    fn reserve_headroom_survives_roundtrip_and_absorbs_growth() {
        let img = build_full(3, &demo_kvs(), &[demo_tensor()], &[0u8; 32], 32, 128);
        let g = parse_head(&img[..], img.len() as u64).unwrap();
        let mut kvs = g.kvs.clone();
        kvs.push(kv(
            "demo.note",
            GgufValue::Str("fits in the headroom".into()),
        ));
        assert!(matches!(
            plan_in_place(3, &kvs, &g.tensors, g.data_start, g.alignment),
            FitPlan::InPlace { .. }
        ));
    }

    #[test]
    fn write_in_place_touches_only_the_head_region() {
        let dir = tempdir("inplace");
        let path = dir.join("m.gguf");
        let img = build_full(3, &demo_kvs(), &[demo_tensor()], &[0x5Au8; 64], 32, 64);
        fs::write(&path, &img).unwrap();
        let g = parse_head(&img[..], img.len() as u64).unwrap();

        let mut kvs = g.kvs.clone();
        for kv in &mut kvs {
            if kv.key == "demo.context_length" {
                kv.value = GgufValue::U32(65536);
            }
        }
        let plan = plan_in_place(3, &kvs, &g.tensors, g.data_start, g.alignment);
        let FitPlan::InPlace { head, zero_fill } = plan else {
            panic!("same-size edit must fit");
        };
        write_in_place(&path, &head, zero_fill).unwrap();

        let after = fs::read(&path).unwrap();
        assert_eq!(after.len(), img.len(), "file size unchanged");
        assert_eq!(
            &after[g.data_start as usize..],
            &img[g.data_start as usize..],
            "tensor data bytes are bit-identical"
        );
        let reparsed = parse_head(&after[..], after.len() as u64).unwrap();
        assert_eq!(
            reparsed.get("demo.context_length"),
            Some(&GgufValue::U32(65536))
        );
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rewrite_streams_tensor_data_verbatim_with_new_headroom() {
        let dir = tempdir("rewrite");
        let src = dir.join("src.gguf");
        let dst = dir.join("dst.gguf");
        let payload: Vec<u8> = (0..96u8).collect();
        let img = build_full(3, &demo_kvs(), &[demo_tensor()], &payload, 32, 0);
        fs::write(&src, &img).unwrap();
        let g = parse_head(&img[..], img.len() as u64).unwrap();

        let mut edited = g.clone();
        edited
            .kvs
            .push(kv("demo.grown", GgufValue::Str("g".repeat(1000))));
        let (head_len, copied) = rewrite_file(&src, &dst, &edited, 256).unwrap();
        assert_eq!(copied, payload.len() as u64);

        let out = fs::read(&dst).unwrap();
        let ng = parse_head(&out[..], out.len() as u64).unwrap();
        assert_eq!(head_len, ng.head_end);
        assert_eq!(
            ng.get("demo.grown"),
            Some(&GgufValue::Str("g".repeat(1000)))
        );
        assert!(ng.get(PAD_KEY).is_some(), "reserve pad written");
        assert_eq!(ng.tensors, g.tensors, "tensor descriptors unchanged");
        assert_eq!(&out[ng.data_start as usize..], &payload[..]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn sample_bytes_parse_cleanly_and_carry_a_chat_template() {
        let img = sample_bytes(0);
        let g = parse_head(&img[..], img.len() as u64).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.tensors.len(), 2);
        assert_eq!(g.kvs.len(), 9);
        assert!(matches!(
            g.get("tokenizer.chat_template"),
            Some(GgufValue::Str(_))
        ));
        assert_eq!(g.data_len(), 160);
        // Reserve variant grows the head but keeps the same payload.
        let img2 = sample_bytes(512);
        let g2 = parse_head(&img2[..], img2.len() as u64).unwrap();
        assert!(g2.get(PAD_KEY).is_some());
        assert_eq!(g2.data_len(), 160);
    }
}
