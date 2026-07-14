//! Rendering: the pretty JSON `dump`, the human-readable `show` view, and
//! the shared size formatting helpers. Everything returns strings so it is
//! trivially unit-testable.

use crate::json::{encode_pretty, Json};
use crate::reader::Gguf;
use crate::types::{ggml_type_name, GgufValue};
use crate::writer::{serialize_head, strip_pad, PAD_KEY};

/// Convert a metadata value into JSON. Integers stay exact (`i128` carrier);
/// arrays convert element-wise.
pub fn value_json(v: &GgufValue) -> Json {
    use GgufValue::*;
    match v {
        U8(x) => Json::Int(*x as i128),
        I8(x) => Json::Int(*x as i128),
        U16(x) => Json::Int(*x as i128),
        I16(x) => Json::Int(*x as i128),
        U32(x) => Json::Int(*x as i128),
        I32(x) => Json::Int(*x as i128),
        U64(x) => Json::Int(*x as i128),
        I64(x) => Json::Int(*x as i128),
        F32(x) => Json::Float(*x as f64),
        F64(x) => Json::Float(*x),
        Bool(x) => Json::Bool(*x),
        Str(s) => Json::Str(s.clone()),
        Array(_, items) => Json::Arr(items.iter().map(value_json).collect()),
    }
}

/// The complete head as a pretty-printed JSON document.
pub fn dump_json(g: &Gguf) -> String {
    let metadata: Vec<(String, Json)> = g
        .kvs
        .iter()
        .map(|kv| {
            (
                kv.key.clone(),
                Json::Obj(vec![
                    ("type".into(), Json::Str(kv.value.type_label())),
                    ("value".into(), value_json(&kv.value)),
                ]),
            )
        })
        .collect();

    let tensors: Vec<Json> = g
        .tensors
        .iter()
        .map(|t| {
            Json::Obj(vec![
                ("name".into(), Json::Str(t.name.clone())),
                ("type".into(), Json::Str(ggml_type_name(t.type_code))),
                (
                    "dims".into(),
                    Json::Arr(t.dims.iter().map(|d| Json::Int(*d as i128)).collect()),
                ),
                ("offset".into(), Json::Int(t.offset as i128)),
                (
                    "bytes".into(),
                    t.byte_size()
                        .map(|b| Json::Int(b as i128))
                        .unwrap_or(Json::Null),
                ),
            ])
        })
        .collect();

    let doc = Json::Obj(vec![
        ("gguf_version".into(), Json::Int(g.version as i128)),
        ("alignment".into(), Json::Int(g.alignment as i128)),
        ("data_start".into(), Json::Int(g.data_start as i128)),
        ("data_bytes".into(), Json::Int(g.data_len() as i128)),
        ("metadata".into(), Json::Obj(metadata)),
        ("tensors".into(), Json::Arr(tensors)),
    ]);
    encode_pretty(&doc)
}

/// `1234` → `"1.2 KiB"`, etc. Bytes below 1 KiB are printed exactly.
pub fn fmt_size(bytes: u64) -> String {
    const UNITS: &[(u64, &str)] = &[
        (1 << 40, "TiB"),
        (1 << 30, "GiB"),
        (1 << 20, "MiB"),
        (1 << 10, "KiB"),
    ];
    for (scale, unit) in UNITS {
        if bytes >= *scale {
            return format!("{:.1} {unit}", bytes as f64 / *scale as f64);
        }
    }
    fmt_bytes(bytes)
}

/// Exact byte count with a grammatical unit: `1 byte`, `42 bytes`.
pub fn fmt_bytes(bytes: u64) -> String {
    if bytes == 1 {
        "1 byte".to_string()
    } else {
        format!("{bytes} bytes")
    }
}

/// How many bytes the metadata could still grow while staying in place:
/// the distance between the pad-stripped head end and the data offset.
pub fn headroom(g: &Gguf) -> u64 {
    let end0 = serialize_head(g.version, &strip_pad(&g.kvs), &g.tensors).len() as u64;
    g.data_start.saturating_sub(end0)
}

/// The human-readable `show` view.
pub fn render_show(g: &Gguf, name: &str, with_tensors: bool) -> String {
    let mut out = String::new();
    out.push_str(&format!("file:       {name}\n"));
    out.push_str(&format!("size:       {}\n", fmt_size(g.file_len)));
    out.push_str(&format!("gguf:       v{} little-endian\n", g.version));
    out.push_str(&format!("alignment:  {}\n", g.alignment));
    out.push_str(&format!(
        "tensors:    {} ({} of tensor data)\n",
        g.tensors.len(),
        fmt_size(g.data_len())
    ));
    out.push_str(&format!(
        "data at:    0x{:x} (headroom for in-place edits: {})\n",
        g.data_start,
        fmt_bytes(headroom(g))
    ));
    out.push_str(&format!(
        "metadata:   {} {}\n",
        g.kvs.len(),
        if g.kvs.len() == 1 { "key" } else { "keys" }
    ));
    for kv in &g.kvs {
        if kv.key == PAD_KEY {
            if let GgufValue::Str(s) = &kv.value {
                out.push_str(&format!(
                    "  {:<36} {:<13} ({} of managed headroom)\n",
                    kv.key,
                    "string",
                    fmt_bytes(s.len() as u64)
                ));
                continue;
            }
        }
        out.push_str(&format!(
            "  {:<36} {:<13} {}\n",
            kv.key,
            kv.value.type_label(),
            kv.value.preview(60)
        ));
    }
    if with_tensors {
        out.push_str("tensor list:\n");
        for t in &g.tensors {
            let dims: Vec<String> = t.dims.iter().map(|d| d.to_string()).collect();
            out.push_str(&format!(
                "  {:<28} {:<8} [{}]  offset 0x{:x}\n",
                t.name,
                ggml_type_name(t.type_code),
                dims.join(", "),
                t.offset
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;
    use crate::reader::parse_head;
    use crate::types::{GgufType, GgufValue};
    use crate::writer::sample_bytes;

    fn sample() -> Gguf {
        let img = sample_bytes(0);
        parse_head(&img[..], img.len() as u64).unwrap()
    }

    #[test]
    fn dump_output_is_valid_json_that_reparses() {
        let doc = parse(&dump_json(&sample())).expect("dump output parses");
        assert_eq!(doc.get("gguf_version"), Some(&Json::Int(3)));
        assert_eq!(doc.get("alignment"), Some(&Json::Int(32)));
        let meta = doc.get("metadata").unwrap();
        let ctx = meta.get("sample.context_length").unwrap();
        assert_eq!(ctx.get("type"), Some(&Json::Str("u32".into())));
        assert_eq!(ctx.get("value"), Some(&Json::Int(4096)));

        let doc = parse(&dump_json(&sample())).unwrap();
        let Json::Arr(tensors) = doc.get("tensors").unwrap() else {
            panic!()
        };
        assert_eq!(tensors.len(), 2);
        assert_eq!(
            tensors[0].get("name"),
            Some(&Json::Str("token_embd.weight".into()))
        );
        assert_eq!(tensors[0].get("type"), Some(&Json::Str("F32".into())));
        assert_eq!(tensors[0].get("bytes"), Some(&Json::Int(128)));
    }

    #[test]
    fn value_json_keeps_u64_exact_and_arrays_elementwise() {
        assert_eq!(
            value_json(&GgufValue::U64(u64::MAX)),
            Json::Int(u64::MAX as i128)
        );
        let arr = GgufValue::Array(
            GgufType::Str,
            vec![GgufValue::Str("a".into()), GgufValue::Str("b".into())],
        );
        assert_eq!(
            value_json(&arr),
            Json::Arr(vec![Json::Str("a".into()), Json::Str("b".into())])
        );
    }

    #[test]
    fn fmt_size_picks_sane_units() {
        assert_eq!(fmt_size(0), "0 bytes");
        assert_eq!(fmt_size(1), "1 byte", "singular, never '1 bytes'");
        assert_eq!(fmt_size(999), "999 bytes");
        assert_eq!(fmt_size(1536), "1.5 KiB");
        assert_eq!(fmt_size(40 * (1 << 30)), "40.0 GiB");
    }

    #[test]
    fn show_reports_layout_facts_and_every_key() {
        let g = sample();
        let text = render_show(&g, "sample.gguf", true);
        assert!(text.contains("gguf:       v3 little-endian"), "{text}");
        assert!(text.contains("tensors:    2"), "{text}");
        assert!(text.contains("sample.context_length"), "{text}");
        assert!(text.contains("headroom"), "{text}");
        assert!(text.contains("token_embd.weight"), "{text}");
        assert!(text.contains("[8, 4]"), "{text}");
    }

    #[test]
    fn show_renders_managed_pad_as_headroom_not_spaces() {
        let img = sample_bytes(256);
        let g = parse_head(&img[..], img.len() as u64).unwrap();
        let text = render_show(&g, "s.gguf", false);
        assert!(text.contains("256 bytes of managed headroom"), "{text}");
        assert!(!text.contains("\"    "), "no raw runs of spaces: {text}");
    }

    #[test]
    fn headroom_counts_pad_and_alignment_slack() {
        let img = sample_bytes(300);
        let g = parse_head(&img[..], img.len() as u64).unwrap();
        // Stripping the pad frees its 30-byte overhead + 300-byte value,
        // plus whatever alignment slack the layout already had.
        assert!(headroom(&g) >= 330, "headroom = {}", headroom(&g));
        assert!(headroom(&g) < 330 + 32, "headroom = {}", headroom(&g));
    }
}
