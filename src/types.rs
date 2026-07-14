//! Core data model for GGUF files: metadata value types and values, tensor
//! descriptors, and the ggml tensor-type table used for size math and
//! human-readable display. Everything here is pure and I/O free.

/// The four magic bytes at the start of every GGUF file.
pub const MAGIC: [u8; 4] = *b"GGUF";
/// Default tensor-data alignment when `general.alignment` is absent.
pub const DEFAULT_ALIGNMENT: u64 = 32;
/// The metadata key that stores the alignment override.
pub const ALIGNMENT_KEY: &str = "general.alignment";
/// The metadata key that stores the chat template.
pub const CHAT_TEMPLATE_KEY: &str = "tokenizer.chat_template";

/// Round `x` up to the next multiple of `align` (treats `align == 0` as 1).
pub fn align_up(x: u64, align: u64) -> u64 {
    let a = align.max(1);
    x.div_ceil(a) * a
}

/// The thirteen GGUF metadata value types, in wire-code order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    Str,
    Array,
    U64,
    I64,
    F64,
}

impl GgufType {
    /// Decode a wire type code (0..=12).
    pub fn from_code(code: u32) -> Option<GgufType> {
        use GgufType::*;
        Some(match code {
            0 => U8,
            1 => I8,
            2 => U16,
            3 => I16,
            4 => U32,
            5 => I32,
            6 => F32,
            7 => Bool,
            8 => Str,
            9 => Array,
            10 => U64,
            11 => I64,
            12 => F64,
            _ => return None,
        })
    }

    /// The wire type code written into the file.
    pub fn code(self) -> u32 {
        use GgufType::*;
        match self {
            U8 => 0,
            I8 => 1,
            U16 => 2,
            I16 => 3,
            U32 => 4,
            I32 => 5,
            F32 => 6,
            Bool => 7,
            Str => 8,
            Array => 9,
            U64 => 10,
            I64 => 11,
            F64 => 12,
        }
    }

    /// Short human name, as used in CLI type prefixes and JSON dumps.
    pub fn name(self) -> &'static str {
        use GgufType::*;
        match self {
            U8 => "u8",
            I8 => "i8",
            U16 => "u16",
            I16 => "i16",
            U32 => "u32",
            I32 => "i32",
            F32 => "f32",
            Bool => "bool",
            Str => "string",
            Array => "array",
            U64 => "u64",
            I64 => "i64",
            F64 => "f64",
        }
    }

    /// Parse a type name as accepted on the command line (`str` and `string`
    /// are aliases; everything else matches [`GgufType::name`]).
    pub fn from_name(name: &str) -> Option<GgufType> {
        use GgufType::*;
        Some(match name {
            "u8" => U8,
            "i8" => I8,
            "u16" => U16,
            "i16" => I16,
            "u32" => U32,
            "i32" => I32,
            "f32" => F32,
            "bool" => Bool,
            "str" | "string" => Str,
            "array" => Array,
            "u64" => U64,
            "i64" => I64,
            "f64" => F64,
            _ => return None,
        })
    }
}

/// A decoded GGUF metadata value. Arrays carry their element type so that
/// re-serialization is byte-exact.
#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    Array(GgufType, Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    /// The value's GGUF type.
    pub fn ty(&self) -> GgufType {
        use GgufValue::*;
        match self {
            U8(_) => GgufType::U8,
            I8(_) => GgufType::I8,
            U16(_) => GgufType::U16,
            I16(_) => GgufType::I16,
            U32(_) => GgufType::U32,
            I32(_) => GgufType::I32,
            F32(_) => GgufType::F32,
            Bool(_) => GgufType::Bool,
            Str(_) => GgufType::Str,
            Array(_, _) => GgufType::Array,
            U64(_) => GgufType::U64,
            I64(_) => GgufType::I64,
            F64(_) => GgufType::F64,
        }
    }

    /// Human type label: scalar names, or `array<elem>` for arrays.
    pub fn type_label(&self) -> String {
        match self {
            GgufValue::Array(elem, _) => format!("array<{}>", elem.name()),
            other => other.ty().name().to_string(),
        }
    }

    /// Widen any non-negative integer value to `u64` (used for
    /// `general.alignment`, context lengths, etc.).
    pub fn as_u64(&self) -> Option<u64> {
        use GgufValue::*;
        match *self {
            U8(v) => Some(v as u64),
            U16(v) => Some(v as u64),
            U32(v) => Some(v as u64),
            U64(v) => Some(v),
            I8(v) if v >= 0 => Some(v as u64),
            I16(v) if v >= 0 => Some(v as u64),
            I32(v) if v >= 0 => Some(v as u64),
            I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }

    /// One-line preview for tables: strings are escaped and truncated to
    /// `max_chars`, arrays show their length plus the first few elements.
    pub fn preview(&self, max_chars: usize) -> String {
        use GgufValue::*;
        match self {
            Str(s) => {
                let esc = escape_short(s);
                if esc.chars().count() > max_chars {
                    let head: String = esc.chars().take(max_chars).collect();
                    format!("\"{head}…\" ({} bytes)", s.len())
                } else {
                    format!("\"{esc}\"")
                }
            }
            Array(elem, items) => {
                let mut out = format!("array<{}>[{}]", elem.name(), items.len());
                if !items.is_empty() {
                    let shown: Vec<String> = items.iter().take(3).map(|v| v.preview(24)).collect();
                    out.push_str(" {");
                    out.push_str(&shown.join(", "));
                    if items.len() > 3 {
                        out.push_str(", …");
                    }
                    out.push('}');
                }
                out
            }
            U8(v) => v.to_string(),
            I8(v) => v.to_string(),
            U16(v) => v.to_string(),
            I16(v) => v.to_string(),
            U32(v) => v.to_string(),
            I32(v) => v.to_string(),
            U64(v) => v.to_string(),
            I64(v) => v.to_string(),
            F32(v) => format!("{v}"),
            F64(v) => format!("{v}"),
            Bool(v) => v.to_string(),
        }
    }
}

/// Escape control characters for one-line display (`\n` → `\\n`, etc.).
pub fn escape_short(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '"' => out.push_str("\\\""),
            c if c.is_control() => out.push_str(&format!("\\u{{{:x}}}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// One metadata key/value pair, in file order.
#[derive(Clone, Debug, PartialEq)]
pub struct KvPair {
    pub key: String,
    pub value: GgufValue,
}

/// Convenience constructor used by the sample builder and tests.
pub fn kv(key: &str, value: GgufValue) -> KvPair {
    KvPair {
        key: key.to_string(),
        value,
    }
}

/// One tensor descriptor from the head. `offset` is relative to the start of
/// the tensor-data section — which is exactly why metadata edits can leave
/// tensor data untouched.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub type_code: u32,
    pub offset: u64,
}

impl TensorInfo {
    /// Total element count, or `None` on overflow / zero dims.
    pub fn n_elements(&self) -> Option<u64> {
        if self.dims.is_empty() {
            return None;
        }
        let mut total: u128 = 1;
        for d in &self.dims {
            total = total.checked_mul(*d as u128)?;
        }
        u64::try_from(total).ok()
    }

    /// Byte size on disk, when the ggml type's block layout is known and the
    /// first dimension is a whole number of blocks. `None` means "cannot
    /// compute" — the verifier then skips extent checks for this tensor.
    pub fn byte_size(&self) -> Option<u64> {
        let (block_elems, block_bytes) = ggml_type_size(self.type_code)?;
        let ne0 = *self.dims.first()?;
        if ne0 % block_elems != 0 {
            return None;
        }
        let mut total: u128 = (ne0 / block_elems) as u128 * block_bytes as u128;
        for d in &self.dims[1..] {
            total = total.checked_mul(*d as u128)?;
        }
        u64::try_from(total).ok()
    }
}

/// (code, name, block_elems, block_bytes). `block_bytes == 0` marks types we
/// can name but whose exact block size we do not track; size checks are
/// skipped for those rather than risking false verification errors.
const GGML_TYPES: &[(u32, &str, u64, u64)] = &[
    (0, "F32", 1, 4),
    (1, "F16", 1, 2),
    (2, "Q4_0", 32, 18),
    (3, "Q4_1", 32, 20),
    (6, "Q5_0", 32, 22),
    (7, "Q5_1", 32, 24),
    (8, "Q8_0", 32, 34),
    (9, "Q8_1", 32, 36),
    (10, "Q2_K", 256, 84),
    (11, "Q3_K", 256, 110),
    (12, "Q4_K", 256, 144),
    (13, "Q5_K", 256, 176),
    (14, "Q6_K", 256, 210),
    (15, "Q8_K", 256, 292),
    (16, "IQ2_XXS", 256, 0),
    (17, "IQ2_XS", 256, 0),
    (18, "IQ3_XXS", 256, 0),
    (19, "IQ1_S", 256, 0),
    (20, "IQ4_NL", 32, 18),
    (21, "IQ3_S", 256, 0),
    (22, "IQ2_S", 256, 0),
    (23, "IQ4_XS", 256, 136),
    (24, "I8", 1, 1),
    (25, "I16", 1, 2),
    (26, "I32", 1, 4),
    (27, "I64", 1, 8),
    (28, "F64", 1, 8),
    (29, "IQ1_M", 256, 0),
    (30, "BF16", 1, 2),
];

/// Human name for a ggml tensor type code (`type#N` when unknown).
pub fn ggml_type_name(code: u32) -> String {
    GGML_TYPES
        .iter()
        .find(|(c, _, _, _)| *c == code)
        .map(|(_, name, _, _)| name.to_string())
        .unwrap_or_else(|| format!("type#{code}"))
}

/// Block layout `(elements, bytes)` for a ggml type, when tracked.
pub fn ggml_type_size(code: u32) -> Option<(u64, u64)> {
    GGML_TYPES
        .iter()
        .find(|(c, _, _, bytes)| *c == code && *bytes > 0)
        .map(|(_, _, elems, bytes)| (*elems, *bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_codes_and_names_roundtrip_for_all_thirteen_types() {
        for code in 0u32..=12 {
            let ty = GgufType::from_code(code).expect("code in range");
            assert_eq!(ty.code(), code);
            assert_eq!(GgufType::from_name(ty.name()), Some(ty));
        }
        assert_eq!(GgufType::from_code(13), None);

        assert_eq!(GgufType::from_name("str"), Some(GgufType::Str));
        assert_eq!(GgufType::from_name("string"), Some(GgufType::Str));
        assert_eq!(GgufType::from_name("float"), None);
        assert_eq!(GgufType::from_name("U32"), None, "type names are lowercase");
    }

    #[test]
    fn type_label_distinguishes_scalars_and_arrays() {
        assert_eq!(GgufValue::U32(1).type_label(), "u32");
        let arr = GgufValue::Array(GgufType::Str, vec![]);
        assert_eq!(arr.type_label(), "array<string>");
        let nested = GgufValue::Array(GgufType::Array, vec![arr]);
        assert_eq!(nested.type_label(), "array<array>");
    }

    #[test]
    fn as_u64_widens_unsigned_and_non_negative_signed_only() {
        assert_eq!(GgufValue::U8(7).as_u64(), Some(7));
        assert_eq!(GgufValue::U64(u64::MAX).as_u64(), Some(u64::MAX));
        assert_eq!(GgufValue::I32(32).as_u64(), Some(32));
        assert_eq!(GgufValue::I32(-1).as_u64(), None);
        assert_eq!(GgufValue::F32(1.0).as_u64(), None);
        assert_eq!(GgufValue::Str("32".into()).as_u64(), None);
    }

    #[test]
    fn preview_truncates_strings_and_summarizes_arrays() {
        let long = GgufValue::Str("line one\nline two and much more text".into());
        let p = long.preview(12);
        assert!(p.starts_with('"'), "preview is quoted: {p}");
        assert!(p.contains("\\n"), "newline escaped: {p}");
        assert!(p.contains("(36 bytes)"), "byte length reported: {p}");
        let short = GgufValue::Str("hi".into());
        assert_eq!(short.preview(12), "\"hi\"");

        let arr = GgufValue::Array(GgufType::U32, (0..5).map(GgufValue::U32).collect());
        let p = arr.preview(60);
        assert!(p.starts_with("array<u32>[5]"), "{p}");
        assert!(p.contains("{0, 1, 2, …}"), "{p}");
    }

    #[test]
    fn ggml_type_table_names_types_and_block_size_math() {
        assert_eq!(ggml_type_name(0), "F32");
        assert_eq!(ggml_type_name(12), "Q4_K");
        assert_eq!(ggml_type_name(30), "BF16");
        assert_eq!(ggml_type_name(99), "type#99");

        // 8x4 F32 = 32 elements * 4 bytes.
        let t = TensorInfo {
            name: "t".into(),
            dims: vec![8, 4],
            type_code: 0,
            offset: 0,
        };
        assert_eq!(t.byte_size(), Some(128));
        // Q4_K: 256-element blocks of 144 bytes; [256, 2] = 2 blocks * 2 rows.
        let q = TensorInfo {
            name: "q".into(),
            dims: vec![256, 2],
            type_code: 12,
            offset: 0,
        };
        assert_eq!(q.byte_size(), Some(288));
    }

    #[test]
    fn size_math_defends_against_partial_blocks_unknown_types_and_overflow() {
        let ragged = TensorInfo {
            name: "r".into(),
            dims: vec![100],
            type_code: 2, // Q4_0 has 32-element blocks; 100 is not a multiple
            offset: 0,
        };
        assert_eq!(ragged.byte_size(), None);
        let unknown = TensorInfo {
            name: "u".into(),
            dims: vec![32],
            type_code: 16, // IQ2_XXS: named, size deliberately untracked
            offset: 0,
        };
        assert_eq!(unknown.byte_size(), None);

        let t = TensorInfo {
            name: "huge".into(),
            dims: vec![u64::MAX, u64::MAX, 2],
            type_code: 0,
            offset: 0,
        };
        assert_eq!(t.n_elements(), None);
        assert_eq!(t.byte_size(), None);
    }

    #[test]
    fn align_up_handles_boundaries_and_zero_alignment() {
        assert_eq!(align_up(0, 32), 0);
        assert_eq!(align_up(1, 32), 32);
        assert_eq!(align_up(32, 32), 32);
        assert_eq!(align_up(33, 32), 64);
        assert_eq!(align_up(7, 0), 7, "zero alignment treated as 1");
    }
}
