//! Structural verification of a parsed GGUF head: duplicate or malformed
//! keys, alignment sanity, tensor offset/extent/overlap checks, and a lint
//! pass over the embedded chat template. Errors mean a runtime will likely
//! reject or misread the file; warnings mean something is unusual enough to
//! look at before publishing.

use crate::reader::Gguf;
use crate::template;
use crate::types::{ggml_type_name, GgufValue, CHAT_TEMPLATE_KEY};

/// The verifier's findings, in a stable order.
#[derive(Debug, Default)]
pub struct Report {
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
}

impl Report {
    pub fn ok(&self) -> bool {
        self.errors.is_empty()
    }
}

fn key_charset_ok(key: &str) -> bool {
    !key.is_empty()
        && key
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
}

/// Run every check against a parsed head.
pub fn verify(g: &Gguf) -> Report {
    let mut r = Report::default();

    // --- metadata keys -------------------------------------------------
    for (i, kv) in g.kvs.iter().enumerate() {
        if kv.key.is_empty() {
            r.errors.push(format!("metadata key #{i} is empty"));
        } else if !key_charset_ok(&kv.key) {
            r.warnings.push(format!(
                "key '{}' contains characters outside [A-Za-z0-9._-]",
                crate::types::escape_short(&kv.key)
            ));
        }
        for later in &g.kvs[i + 1..] {
            if later.key == kv.key {
                r.errors
                    .push(format!("duplicate metadata key '{}'", kv.key));
            }
        }
    }

    // --- alignment ------------------------------------------------------
    if !g.alignment.is_power_of_two() {
        r.warnings.push(format!(
            "general.alignment {} is not a power of two",
            g.alignment
        ));
    }

    if g.get("general.architecture").is_none() {
        r.warnings
            .push("no general.architecture key; most runtimes require it".into());
    }

    // --- tensors ----------------------------------------------------------
    if !g.tensors.is_empty() && g.data_start > g.file_len {
        r.errors.push(format!(
            "tensor data should start at byte {} but the file is only {} bytes (truncated?)",
            g.data_start, g.file_len
        ));
    }
    let data_len = g.data_len();

    for (i, t) in g.tensors.iter().enumerate() {
        for later in &g.tensors[i + 1..] {
            if later.name == t.name {
                r.errors.push(format!("duplicate tensor name '{}'", t.name));
            }
        }
        if t.dims.contains(&0) {
            r.errors
                .push(format!("tensor '{}' has a zero-sized dimension", t.name));
        }
        if t.offset % g.alignment.max(1) != 0 {
            r.errors.push(format!(
                "tensor '{}' offset {} is not a multiple of the alignment ({})",
                t.name, t.offset, g.alignment
            ));
        }
        if crate::types::ggml_type_size(t.type_code).is_none() {
            if ggml_type_name(t.type_code).starts_with("type#") {
                r.warnings.push(format!(
                    "tensor '{}' has unknown ggml type {} (size checks skipped)",
                    t.name, t.type_code
                ));
            }
            continue;
        }
        match t.byte_size() {
            Some(size) => {
                if t.offset.checked_add(size).is_none() || t.offset + size > data_len {
                    r.errors.push(format!(
                        "tensor '{}' ({} bytes at offset {}) extends past the data section \
                         ({} bytes)",
                        t.name, size, t.offset, data_len
                    ));
                }
            }
            None => r.warnings.push(format!(
                "tensor '{}' ({}, dims {:?}) is not a whole number of blocks; size check skipped",
                t.name,
                ggml_type_name(t.type_code),
                t.dims
            )),
        }
    }

    // Overlap detection among tensors with known sizes.
    let mut extents: Vec<(&str, u64, u64)> = g
        .tensors
        .iter()
        .filter_map(|t| t.byte_size().map(|s| (t.name.as_str(), t.offset, s)))
        .collect();
    extents.sort_by_key(|(_, off, _)| *off);
    for pair in extents.windows(2) {
        let (a_name, a_off, a_size) = pair[0];
        let (b_name, b_off, _) = pair[1];
        if a_off + a_size > b_off {
            r.errors.push(format!(
                "tensors '{a_name}' and '{b_name}' overlap ({a_off}+{a_size} > {b_off})"
            ));
        }
    }

    // --- chat template ------------------------------------------------------
    if let Some(GgufValue::Str(tpl)) = g.get(CHAT_TEMPLATE_KEY) {
        for issue in template::lint(tpl) {
            if issue.severity == template::Severity::Error {
                let at = issue
                    .pos
                    .map(|(l, c)| format!(" at {l}:{c}"))
                    .unwrap_or_default();
                r.warnings.push(format!(
                    "{CHAT_TEMPLATE_KEY}: {}{at} (run 'template check' for details)",
                    issue.message
                ));
            }
        }
    } else if g.get(CHAT_TEMPLATE_KEY).is_some() {
        r.errors
            .push(format!("{CHAT_TEMPLATE_KEY} is not a string value"));
    }

    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::parse_head;
    use crate::types::{kv, GgufValue, KvPair, TensorInfo};
    use crate::writer::build_full;

    fn parse(img: &[u8]) -> Gguf {
        parse_head(img, img.len() as u64).expect("fixture parses")
    }

    fn arch() -> KvPair {
        kv("general.architecture", GgufValue::Str("demo".into()))
    }

    fn f32_tensor(name: &str, ne0: u64, offset: u64) -> TensorInfo {
        TensorInfo {
            name: name.into(),
            dims: vec![ne0],
            type_code: 0,
            offset,
        }
    }

    #[test]
    fn the_builtin_sample_verifies_clean() {
        let img = crate::writer::sample_bytes(0);
        let r = verify(&parse(&img));
        assert!(r.ok(), "errors: {:?}", r.errors);
        assert!(r.warnings.is_empty(), "warnings: {:?}", r.warnings);
    }

    #[test]
    fn duplicate_keys_error_and_odd_charset_warns() {
        let kvs = vec![
            arch(),
            kv("x.y", GgufValue::U32(1)),
            kv("x.y", GgufValue::U32(2)),
        ];
        let img = build_full(3, &kvs, &[], &[], 32, 0);
        let r = verify(&parse(&img));
        assert!(
            r.errors
                .iter()
                .any(|e| e.contains("duplicate metadata key 'x.y'")),
            "{r:?}"
        );

        let kvs = vec![arch(), kv("weird key!", GgufValue::U32(1))];
        let img = build_full(3, &kvs, &[], &[], 32, 0);
        let r = verify(&parse(&img));
        assert!(r.ok());
        assert!(r.warnings.iter().any(|w| w.contains("weird key!")), "{r:?}");
    }

    #[test]
    fn unaligned_tensor_offset_is_an_error() {
        let t = f32_tensor("t", 8, 7); // 7 is not a multiple of 32
        let img = build_full(3, &[arch()], &[t], &[0u8; 64], 32, 0);
        let r = verify(&parse(&img));
        assert!(
            r.errors.iter().any(|e| e.contains("not a multiple")),
            "{r:?}"
        );
    }

    #[test]
    fn tensor_past_the_end_of_the_data_section_is_an_error() {
        let t = f32_tensor("t", 64, 0); // needs 256 bytes, data has 64
        let img = build_full(3, &[arch()], &[t], &[0u8; 64], 32, 0);
        let r = verify(&parse(&img));
        assert!(r.errors.iter().any(|e| e.contains("extends past")), "{r:?}");
    }

    #[test]
    fn overlapping_tensors_and_duplicate_names_are_errors() {
        let a = f32_tensor("a", 32, 0); // 128 bytes at 0
        let b = f32_tensor("b", 8, 64); // 32 bytes at 64 — inside a
        let img = build_full(3, &[arch()], &[a, b], &[0u8; 128], 32, 0);
        let r = verify(&parse(&img));
        assert!(
            r.errors
                .iter()
                .any(|e| e.contains("'a'") && e.contains("'b'") && e.contains("overlap")),
            "{r:?}"
        );

        let a = f32_tensor("same", 8, 0);
        let b = f32_tensor("same", 8, 32);
        let img = build_full(3, &[arch()], &[a, b], &[0u8; 64], 32, 0);
        let r = verify(&parse(&img));
        assert!(
            r.errors.iter().any(|e| e.contains("duplicate tensor name")),
            "{r:?}"
        );
    }

    #[test]
    fn unknown_ggml_type_warns_and_skips_size_math() {
        let t = TensorInfo {
            name: "mystery".into(),
            dims: vec![8],
            type_code: 200,
            offset: 0,
        };
        let img = build_full(3, &[arch()], &[t], &[0u8; 32], 32, 0);
        let r = verify(&parse(&img));
        assert!(r.ok(), "{r:?}");
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("unknown ggml type 200")),
            "{r:?}"
        );
    }

    #[test]
    fn missing_architecture_and_odd_alignment_warn() {
        let kvs = vec![kv("general.alignment", GgufValue::U32(24))];
        let img = build_full(3, &kvs, &[], &[], 24, 0);
        let r = verify(&parse(&img));
        assert!(r.ok());
        assert!(
            r.warnings.iter().any(|w| w.contains("power of two")),
            "{r:?}"
        );
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("general.architecture")),
            "{r:?}"
        );
    }

    #[test]
    fn chat_template_problems_warn_and_wrong_type_errors() {
        let kvs = vec![
            arch(),
            kv(
                "tokenizer.chat_template",
                GgufValue::Str("{% for m in messages %}{{ m }} add_generation_prompt".into()),
            ),
        ];
        let img = build_full(3, &kvs, &[], &[], 32, 0);
        let r = verify(&parse(&img));
        assert!(r.ok(), "template problems are warnings, not errors: {r:?}");
        assert!(
            r.warnings
                .iter()
                .any(|w| w.contains("tokenizer.chat_template")),
            "{r:?}"
        );

        let kvs = vec![arch(), kv("tokenizer.chat_template", GgufValue::U32(1))];
        let img = build_full(3, &kvs, &[], &[], 32, 0);
        let r = verify(&parse(&img));
        assert!(r.errors.iter().any(|e| e.contains("not a string")), "{r:?}");
    }

    #[test]
    fn truncated_data_section_is_an_error() {
        let t = f32_tensor("t", 8, 0);
        let mut img = build_full(3, &[arch()], &[t], &[0u8; 32], 32, 0);
        // Chop the file inside the head padding: data_start > file_len.
        let g = parse(&img);
        img.truncate(g.data_start as usize - 1);
        let cut = parse_head(&img[..], img.len() as u64).expect("head still parses");
        let r = verify(&cut);
        assert!(r.errors.iter().any(|e| e.contains("truncated")), "{r:?}");
    }
}
