//! Edit operations on the metadata list: `set`, `rm` and `rename`, plus the
//! value syntax used on the command line (`KEY=VALUE` / `KEY=TYPE:VALUE`) and
//! in JSON patch documents. Application is atomic: every operation is
//! validated against a working copy, and the original list is only replaced
//! when all of them succeed.

use crate::json::Json;
use crate::types::{GgufType, GgufValue, KvPair, ALIGNMENT_KEY};
use crate::writer::PAD_KEY;
use std::fmt;

/// Keys that cannot be edited because moving them breaks the offset math
/// that in-place patching depends on.
pub const PROTECTED_KEYS: &[&str] = &[ALIGNMENT_KEY];

#[derive(Debug)]
pub struct PatchError(pub String);

impl fmt::Display for PatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for PatchError {}

fn fail<T>(msg: impl Into<String>) -> Result<T, PatchError> {
    Err(PatchError(msg.into()))
}

/// A value as written by the user: fully typed, or raw text to be coerced
/// against the key's existing type (or inferred for a new key).
#[derive(Clone, Debug, PartialEq)]
pub enum ValueSpec {
    Typed(GgufValue),
    Inferred(String),
}

/// One edit operation.
#[derive(Clone, Debug, PartialEq)]
pub enum EditOp {
    Set { key: String, spec: ValueSpec },
    Remove { key: String },
    Rename { from: String, to: String },
}

/// Parse one `KEY=VALUE` / `KEY=TYPE:VALUE` argument. The prefix before the
/// first `:` is only treated as a type when it names one, so plain values
/// containing colons (URLs, templates) need no escaping.
pub fn parse_set_arg(arg: &str) -> Result<EditOp, PatchError> {
    let Some((key, rest)) = arg.split_once('=') else {
        return fail(format!("'{arg}' is not KEY=VALUE (missing '=')"));
    };
    if key.is_empty() {
        return fail(format!("'{arg}' has an empty key"));
    }
    let spec = match rest.split_once(':') {
        Some((ty, raw)) if GgufType::from_name(ty).is_some() => {
            let ty = GgufType::from_name(ty).unwrap();
            ValueSpec::Typed(coerce(ty, raw)?)
        }
        _ => ValueSpec::Inferred(rest.to_string()),
    };
    Ok(EditOp::Set {
        key: key.to_string(),
        spec,
    })
}

fn parse_int(raw: &str) -> Result<i128, PatchError> {
    raw.parse::<i128>()
        .map_err(|_| PatchError(format!("'{raw}' is not an integer")))
}

fn int_in<T>(raw: &str, min: i128, max: i128, ty: &str) -> Result<T, PatchError>
where
    T: TryFrom<i128>,
{
    let v = parse_int(raw)?;
    if v < min || v > max {
        return fail(format!("{v} does not fit {ty} ({min}..={max})"));
    }
    T::try_from(v).map_err(|_| PatchError(format!("{v} does not fit {ty}")))
}

fn parse_float(raw: &str) -> Result<f64, PatchError> {
    raw.parse::<f64>()
        .map_err(|_| PatchError(format!("'{raw}' is not a number")))
}

/// Coerce raw text into a specific GGUF type, with range checks.
pub fn coerce(ty: GgufType, raw: &str) -> Result<GgufValue, PatchError> {
    use GgufType::*;
    Ok(match ty {
        U8 => GgufValue::U8(int_in(raw, 0, u8::MAX as i128, "u8")?),
        I8 => GgufValue::I8(int_in(raw, i8::MIN as i128, i8::MAX as i128, "i8")?),
        U16 => GgufValue::U16(int_in(raw, 0, u16::MAX as i128, "u16")?),
        I16 => GgufValue::I16(int_in(raw, i16::MIN as i128, i16::MAX as i128, "i16")?),
        U32 => GgufValue::U32(int_in(raw, 0, u32::MAX as i128, "u32")?),
        I32 => GgufValue::I32(int_in(raw, i32::MIN as i128, i32::MAX as i128, "i32")?),
        U64 => GgufValue::U64(int_in(raw, 0, u64::MAX as i128, "u64")?),
        I64 => GgufValue::I64(int_in(raw, i64::MIN as i128, i64::MAX as i128, "i64")?),
        F32 => GgufValue::F32(parse_float(raw)? as f32),
        F64 => GgufValue::F64(parse_float(raw)?),
        Bool => match raw {
            "true" => GgufValue::Bool(true),
            "false" => GgufValue::Bool(false),
            _ => return fail(format!("'{raw}' is not a bool (use 'true' or 'false')")),
        },
        Str => GgufValue::Str(raw.to_string()),
        Array => return fail("array values cannot be written in 0.1.0".to_string()),
    })
}

/// Infer a type for a brand-new key: bool, then unsigned/signed integer
/// (u32 preferred — the width GGUF convention uses for counts and lengths),
/// then f32, falling back to string.
pub fn infer(raw: &str) -> GgufValue {
    match raw {
        "true" => return GgufValue::Bool(true),
        "false" => return GgufValue::Bool(false),
        _ => {}
    }
    if let Ok(v) = raw.parse::<i128>() {
        if (0..=u32::MAX as i128).contains(&v) {
            return GgufValue::U32(v as u32);
        }
        if (i32::MIN as i128..0).contains(&v) {
            return GgufValue::I32(v as i32);
        }
        if v >= 0 && v <= u64::MAX as i128 {
            return GgufValue::U64(v as u64);
        }
        if v >= i64::MIN as i128 && v < 0 {
            return GgufValue::I64(v as i64);
        }
    }
    // Only strings that *look* numeric become floats; "1.2.3" stays a string.
    if raw.parse::<f64>().is_ok() && raw.chars().all(|c| "0123456789+-.eE_".contains(c)) {
        if let Ok(v) = raw.parse::<f64>() {
            return GgufValue::F32(v as f32);
        }
    }
    GgufValue::Str(raw.to_string())
}

fn check_editable(key: &str, action: &str) -> Result<(), PatchError> {
    if PROTECTED_KEYS.contains(&key) {
        return fail(format!(
            "refusing to {action} '{key}': changing the alignment would move every tensor; \
             this is not supported in 0.1.0"
        ));
    }
    if key == PAD_KEY {
        return fail(format!(
            "'{PAD_KEY}' is managed automatically (it is resized on every write); \
             use --reserve to control headroom"
        ));
    }
    Ok(())
}

/// Human-readable description of what changed, one line per operation.
#[derive(Debug, Default)]
pub struct PatchSummary {
    pub lines: Vec<String>,
}

/// Apply all operations, atomically: on any error the input is unchanged.
pub fn apply_ops(kvs: &mut Vec<KvPair>, ops: &[EditOp]) -> Result<PatchSummary, PatchError> {
    let mut work = kvs.clone();
    let mut summary = PatchSummary::default();

    for op in ops {
        match op {
            EditOp::Set { key, spec } => {
                check_editable(key, "edit")?;
                let existing = work.iter().position(|kv| kv.key == *key);
                let value = match (spec, existing) {
                    (ValueSpec::Typed(v), _) => v.clone(),
                    (ValueSpec::Inferred(raw), Some(i)) => {
                        let ty = work[i].value.ty();
                        if ty == GgufType::Array {
                            return fail(format!(
                                "'{key}' holds an array; array writes are not supported in \
                                 0.1.0 (rm the key and set a scalar if you really mean it)"
                            ));
                        }
                        coerce(ty, raw).map_err(|e| {
                            PatchError(format!("{key}: {e} (existing type is {})", ty.name()))
                        })?
                    }
                    (ValueSpec::Inferred(raw), None) => infer(raw),
                };
                match existing {
                    Some(i) => {
                        let old = &work[i].value;
                        summary.lines.push(format!(
                            "set {key}: {} {} -> {} {}",
                            old.type_label(),
                            old.preview(40),
                            value.type_label(),
                            value.preview(40)
                        ));
                        work[i].value = value;
                    }
                    None => {
                        summary.lines.push(format!(
                            "add {key}: {} {}",
                            value.type_label(),
                            value.preview(40)
                        ));
                        work.push(KvPair {
                            key: key.clone(),
                            value,
                        });
                    }
                }
            }
            EditOp::Remove { key } => {
                check_editable(key, "remove")?;
                let Some(i) = work.iter().position(|kv| kv.key == *key) else {
                    return fail(format!("no such key '{key}'"));
                };
                let old = work.remove(i);
                summary.lines.push(format!(
                    "rm  {key} ({}, was {})",
                    old.value.type_label(),
                    old.value.preview(40)
                ));
            }
            EditOp::Rename { from, to } => {
                check_editable(from, "rename")?;
                check_editable(to, "rename to")?;
                if work.iter().any(|kv| kv.key == *to) {
                    return fail(format!("cannot rename '{from}': '{to}' already exists"));
                }
                let Some(i) = work.iter().position(|kv| kv.key == *from) else {
                    return fail(format!("no such key '{from}'"));
                };
                work[i].key = to.clone();
                summary.lines.push(format!("rename {from} -> {to}"));
            }
        }
    }

    *kvs = work;
    Ok(summary)
}

/// Convert an explicit `{"type": ..., "value": ...}` object.
fn json_typed(key: &str, ty_name: &str, value: &Json) -> Result<GgufValue, PatchError> {
    let Some(ty) = GgufType::from_name(ty_name) else {
        return fail(format!("{key}: unknown type '{ty_name}'"));
    };
    let raw = match value {
        Json::Str(s) => s.clone(),
        Json::Int(i) => i.to_string(),
        Json::Float(f) => format!("{f:?}"),
        Json::Bool(b) => b.to_string(),
        Json::Null => return fail(format!("{key}: null has no GGUF representation")),
        Json::Arr(_) | Json::Obj(_) => {
            return fail(format!(
                "{key}: array/object values are not supported in 0.1.0"
            ))
        }
    };
    coerce(ty, &raw).map_err(|e| PatchError(format!("{key}: {e}")))
}

/// Build the operation list from a JSON patch document:
///
/// ```json
/// { "delete": ["a.key"],
///   "rename": {"old.key": "new.key"},
///   "set":    {"k1": 4096, "k2": {"type": "u16", "value": 8}} }
/// ```
///
/// Operations apply in a fixed order — delete, rename, set — regardless of
/// the order of the sections in the document.
pub fn ops_from_json(doc: &Json) -> Result<Vec<EditOp>, PatchError> {
    let Json::Obj(sections) = doc else {
        return fail("patch document must be a JSON object");
    };
    let mut deletes = Vec::new();
    let mut renames = Vec::new();
    let mut sets = Vec::new();

    for (section, body) in sections {
        match section.as_str() {
            "delete" => {
                let Json::Arr(items) = body else {
                    return fail("\"delete\" must be an array of key names");
                };
                for item in items {
                    let Json::Str(key) = item else {
                        return fail("\"delete\" entries must be strings");
                    };
                    deletes.push(EditOp::Remove { key: key.clone() });
                }
            }
            "rename" => {
                let Json::Obj(pairs) = body else {
                    return fail("\"rename\" must be an object of old: new pairs");
                };
                for (from, to) in pairs {
                    let Json::Str(to) = to else {
                        return fail(format!("rename target for '{from}' must be a string"));
                    };
                    renames.push(EditOp::Rename {
                        from: from.clone(),
                        to: to.clone(),
                    });
                }
            }
            "set" => {
                let Json::Obj(pairs) = body else {
                    return fail("\"set\" must be an object of key: value pairs");
                };
                for (key, value) in pairs {
                    let spec = match value {
                        Json::Obj(_) => {
                            let (Some(Json::Str(ty)), Some(v)) =
                                (value.get("type"), value.get("value"))
                            else {
                                return fail(format!(
                                    "{key}: object values must be {{\"type\": ..., \"value\": ...}}"
                                ));
                            };
                            ValueSpec::Typed(json_typed(key, ty, v)?)
                        }
                        Json::Str(s) => ValueSpec::Typed(GgufValue::Str(s.clone())),
                        Json::Int(i) => ValueSpec::Inferred(i.to_string()),
                        Json::Float(f) => ValueSpec::Inferred(format!("{f:?}")),
                        Json::Bool(b) => ValueSpec::Inferred(b.to_string()),
                        Json::Null => {
                            return fail(format!(
                                "{key}: null has no GGUF representation (use \"delete\")"
                            ))
                        }
                        Json::Arr(_) => {
                            return fail(format!("{key}: array writes are not supported in 0.1.0"))
                        }
                    };
                    sets.push(EditOp::Set {
                        key: key.clone(),
                        spec,
                    });
                }
            }
            other => {
                return fail(format!(
                    "unknown patch section \"{other}\" (expected delete/rename/set)"
                ))
            }
        }
    }

    let mut ops = deletes;
    ops.extend(renames);
    ops.extend(sets);
    if ops.is_empty() {
        return fail("patch document contains no operations");
    }
    Ok(ops)
}

/// Parse a byte count with optional binary suffix: `4096`, `16K`, `2M`, `1G`.
pub fn parse_size(raw: &str) -> Result<u64, PatchError> {
    let (digits, mult) = match raw.chars().last() {
        Some('K') | Some('k') => (&raw[..raw.len() - 1], 1024u64),
        Some('M') | Some('m') => (&raw[..raw.len() - 1], 1024 * 1024),
        Some('G') | Some('g') => (&raw[..raw.len() - 1], 1024 * 1024 * 1024),
        _ => (raw, 1),
    };
    let n: u64 = digits
        .parse()
        .map_err(|_| PatchError(format!("'{raw}' is not a size (use N, NK, NM or NG)")))?;
    n.checked_mul(mult)
        .ok_or_else(|| PatchError(format!("'{raw}' overflows")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::kv;

    fn base_kvs() -> Vec<KvPair> {
        vec![
            kv("general.architecture", GgufValue::Str("demo".into())),
            kv("demo.context_length", GgufValue::U32(4096)),
            kv("demo.small", GgufValue::U16(8)),
            kv("demo.temp", GgufValue::F32(0.7)),
            kv("demo.flag", GgufValue::Bool(false)),
            kv(
                "demo.stops",
                GgufValue::Array(GgufType::Str, vec![GgufValue::Str("</s>".into())]),
            ),
        ]
    }

    #[test]
    fn parse_set_arg_covers_inferred_typed_and_error_forms() {
        let op = parse_set_arg("general.name=My Model").unwrap();
        assert_eq!(
            op,
            EditOp::Set {
                key: "general.name".into(),
                spec: ValueSpec::Inferred("My Model".into())
            }
        );

        let op = parse_set_arg("demo.n=u16:9").unwrap();
        assert_eq!(
            op,
            EditOp::Set {
                key: "demo.n".into(),
                spec: ValueSpec::Typed(GgufValue::U16(9))
            }
        );

        // "https" is not a type name, so the whole value stays inferred text.
        let op = parse_set_arg("general.url=https://example.test/m").unwrap();
        let EditOp::Set { spec, .. } = op else {
            panic!()
        };
        assert_eq!(spec, ValueSpec::Inferred("https://example.test/m".into()));

        assert!(parse_set_arg("no-equals").is_err());
        assert!(parse_set_arg("=5").is_err());
    }

    #[test]
    fn coerce_covers_every_scalar_type() {
        assert_eq!(coerce(GgufType::U8, "255").unwrap(), GgufValue::U8(255));
        assert_eq!(coerce(GgufType::I8, "-128").unwrap(), GgufValue::I8(-128));
        assert_eq!(coerce(GgufType::U16, "9").unwrap(), GgufValue::U16(9));
        assert_eq!(coerce(GgufType::I16, "-9").unwrap(), GgufValue::I16(-9));
        assert_eq!(
            coerce(GgufType::U32, "70000").unwrap(),
            GgufValue::U32(70000)
        );
        assert_eq!(
            coerce(GgufType::I32, "-70000").unwrap(),
            GgufValue::I32(-70000)
        );
        assert_eq!(
            coerce(GgufType::U64, &u64::MAX.to_string()).unwrap(),
            GgufValue::U64(u64::MAX)
        );
        assert_eq!(coerce(GgufType::I64, "-1").unwrap(), GgufValue::I64(-1));
        assert_eq!(coerce(GgufType::F32, "0.5").unwrap(), GgufValue::F32(0.5));
        assert_eq!(coerce(GgufType::F64, "2.5").unwrap(), GgufValue::F64(2.5));
        assert_eq!(
            coerce(GgufType::Bool, "true").unwrap(),
            GgufValue::Bool(true)
        );
        assert_eq!(
            coerce(GgufType::Str, "123").unwrap(),
            GgufValue::Str("123".into())
        );
    }

    #[test]
    fn coerce_rejects_out_of_range_bad_bool_and_array_values() {
        let e = coerce(GgufType::U8, "256").unwrap_err();
        assert!(e.0.contains("does not fit u8"), "{e}");
        assert!(e.0.contains("0..=255"), "{e}");
        assert!(coerce(GgufType::U32, "-1").is_err());
        assert!(coerce(GgufType::I16, "40000").is_err());

        assert!(coerce(GgufType::Bool, "TRUE").is_err());
        assert!(coerce(GgufType::Bool, "1").is_err());

        let e = coerce(GgufType::Array, "x").unwrap_err();
        assert!(e.0.contains("0.1.0"), "{e}");
    }

    #[test]
    fn infer_picks_bool_u32_i32_f32_then_string() {
        assert_eq!(infer("true"), GgufValue::Bool(true));
        assert_eq!(infer("32768"), GgufValue::U32(32768));
        assert_eq!(infer("-40"), GgufValue::I32(-40));
        assert_eq!(infer("5000000000"), GgufValue::U64(5_000_000_000));
        assert_eq!(infer("-5000000000"), GgufValue::I64(-5_000_000_000));
        assert_eq!(infer("0.7"), GgufValue::F32(0.7));
        assert_eq!(infer("hello world"), GgufValue::Str("hello world".into()));
        assert_eq!(infer("1.2.3"), GgufValue::Str("1.2.3".into()));
    }

    #[test]
    fn set_respects_existing_types_positions_and_range_checks() {
        let mut kvs = base_kvs();
        let ops = [parse_set_arg("demo.context_length=32768").unwrap()];
        let summary = apply_ops(&mut kvs, &ops).unwrap();
        assert_eq!(
            kvs.iter()
                .find(|kv| kv.key == "demo.context_length")
                .unwrap()
                .value,
            GgufValue::U32(32768)
        );
        assert!(
            summary.lines[0].contains("u32 4096 -> u32 32768"),
            "{:?}",
            summary.lines
        );

        let mut kvs = base_kvs();
        let ops = [parse_set_arg("demo.small=70000").unwrap()];
        let e = apply_ops(&mut kvs, &ops).unwrap_err();
        assert!(e.0.contains("u16"), "{e}");

        let mut kvs = base_kvs();
        let ops = [parse_set_arg("general.architecture=123").unwrap()];
        apply_ops(&mut kvs, &ops).unwrap();
        assert_eq!(kvs[0].value, GgufValue::Str("123".into()));

        let mut kvs = base_kvs();
        let ops = [
            parse_set_arg("demo.temp=0.9").unwrap(),
            parse_set_arg("demo.new_key=hello").unwrap(),
        ];
        apply_ops(&mut kvs, &ops).unwrap();
        assert_eq!(kvs[3].key, "demo.temp", "edited key keeps its slot");
        assert_eq!(kvs.last().unwrap().key, "demo.new_key");
    }

    #[test]
    fn set_on_array_key_without_explicit_type_is_refused() {
        let mut kvs = base_kvs();
        let ops = [parse_set_arg("demo.stops=</s>").unwrap()];
        let e = apply_ops(&mut kvs, &ops).unwrap_err();
        assert!(e.0.contains("array"), "{e}");
    }

    #[test]
    fn remove_is_atomic_on_missing_keys() {
        let mut kvs = base_kvs();
        let before = kvs.clone();
        let ops = [
            EditOp::Remove {
                key: "demo.flag".into(),
            },
            EditOp::Remove {
                key: "demo.nope".into(),
            },
        ];
        let e = apply_ops(&mut kvs, &ops).unwrap_err();
        assert!(e.0.contains("demo.nope"), "{e}");
        assert_eq!(kvs, before, "nothing applied when any op fails");
    }

    #[test]
    fn rename_moves_the_key_and_rejects_collisions() {
        let mut kvs = base_kvs();
        let ops = [EditOp::Rename {
            from: "demo.flag".into(),
            to: "demo.enabled".into(),
        }];
        apply_ops(&mut kvs, &ops).unwrap();
        assert!(kvs.iter().any(|kv| kv.key == "demo.enabled"));

        let collide = [EditOp::Rename {
            from: "demo.enabled".into(),
            to: "demo.temp".into(),
        }];
        let e = apply_ops(&mut kvs, &collide).unwrap_err();
        assert!(e.0.contains("already exists"), "{e}");
    }

    #[test]
    fn protected_and_managed_keys_are_refused() {
        let mut kvs = base_kvs();
        let e = apply_ops(&mut kvs, &[parse_set_arg("general.alignment=64").unwrap()]).unwrap_err();
        assert!(e.0.contains("alignment"), "{e}");
        let e = apply_ops(
            &mut kvs,
            &[EditOp::Remove {
                key: PAD_KEY.into(),
            }],
        )
        .unwrap_err();
        assert!(e.0.contains("managed"), "{e}");
    }

    #[test]
    fn ops_from_json_builds_delete_rename_set_in_fixed_order() {
        let doc = crate::json::parse(
            r#"{"set": {"a": 1, "b": {"type": "u16", "value": 2}},
                "delete": ["c"],
                "rename": {"d": "e"}}"#,
        )
        .unwrap();
        let ops = ops_from_json(&doc).unwrap();
        assert_eq!(ops.len(), 4);
        assert!(matches!(ops[0], EditOp::Remove { .. }), "delete first");
        assert!(matches!(ops[1], EditOp::Rename { .. }), "rename second");
        assert_eq!(
            ops[3],
            EditOp::Set {
                key: "b".into(),
                spec: ValueSpec::Typed(GgufValue::U16(2))
            }
        );

        let doc =
            crate::json::parse(r#"{"set": {"s": "text", "i": 7, "f": 2.0, "b": true}}"#).unwrap();
        let ops = ops_from_json(&doc).unwrap();
        let specs: Vec<&ValueSpec> = ops
            .iter()
            .map(|op| match op {
                EditOp::Set { spec, .. } => spec,
                _ => panic!(),
            })
            .collect();
        assert_eq!(*specs[0], ValueSpec::Typed(GgufValue::Str("text".into())));
        assert_eq!(*specs[1], ValueSpec::Inferred("7".into()));
        // Floats keep a decimal point so they never coerce into integer keys.
        assert_eq!(*specs[2], ValueSpec::Inferred("2.0".into()));
        assert_eq!(*specs[3], ValueSpec::Inferred("true".into()));
    }

    #[test]
    fn ops_from_json_rejects_unknown_sections_null_and_arrays() {
        let bad = crate::json::parse(r#"{"sett": {}}"#).unwrap();
        assert!(ops_from_json(&bad).unwrap_err().0.contains("sett"));
        let null = crate::json::parse(r#"{"set": {"k": null}}"#).unwrap();
        assert!(ops_from_json(&null).unwrap_err().0.contains("delete"));
        let arr = crate::json::parse(r#"{"set": {"k": [1]}}"#).unwrap();
        assert!(ops_from_json(&arr).unwrap_err().0.contains("0.1.0"));
        let empty = crate::json::parse("{}").unwrap();
        assert!(ops_from_json(&empty)
            .unwrap_err()
            .0
            .contains("no operations"));
    }

    #[test]
    fn parse_size_accepts_binary_suffixes() {
        assert_eq!(parse_size("4096").unwrap(), 4096);
        assert_eq!(parse_size("16K").unwrap(), 16 * 1024);
        assert_eq!(parse_size("2m").unwrap(), 2 * 1024 * 1024);
        assert_eq!(parse_size("1G").unwrap(), 1 << 30);
        assert!(parse_size("lots").is_err());
        assert!(parse_size("").is_err());
    }
}
