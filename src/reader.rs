//! Streaming parser for the GGUF *head*: magic, version, metadata key/value
//! pairs and tensor descriptors. The parser never touches tensor data, so
//! opening a 40 GB model file reads only the first few megabytes. All sizes
//! read from the file are checked against sanity caps before allocation, so a
//! corrupt length field produces a clear error instead of an OOM.

use crate::types::{
    align_up, GgufType, GgufValue, KvPair, TensorInfo, ALIGNMENT_KEY, DEFAULT_ALIGNMENT, MAGIC,
};
use std::fmt;
use std::fs::File;
use std::io::{BufReader, ErrorKind, Read};
use std::path::Path;

/// Sanity caps: a well-formed model head stays far below all of these; a
/// corrupt or hostile length field trips them with a readable error.
pub const MAX_KV_COUNT: u64 = 1 << 20;
pub const MAX_TENSOR_COUNT: u64 = 1 << 20;
pub const MAX_STRING_LEN: u64 = 1 << 28;
pub const MAX_ARRAY_LEN: u64 = 1 << 28;
pub const MAX_DIMS: u32 = 4;
const MAX_ARRAY_DEPTH: u32 = 4;

/// A parse error with the byte offset where it was detected.
#[derive(Debug)]
pub struct GgufError {
    pub message: String,
    pub offset: u64,
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (at byte offset {})", self.message, self.offset)
    }
}

impl std::error::Error for GgufError {}

pub type Result<T> = std::result::Result<T, GgufError>;

fn err<T>(offset: u64, message: impl Into<String>) -> Result<T> {
    Err(GgufError {
        message: message.into(),
        offset,
    })
}

/// The parsed head of a GGUF file plus the layout facts editing needs.
#[derive(Clone, Debug)]
pub struct Gguf {
    pub version: u32,
    pub kvs: Vec<KvPair>,
    pub tensors: Vec<TensorInfo>,
    /// Effective alignment (from `general.alignment`, default 32).
    pub alignment: u64,
    /// Offset right after the last tensor descriptor, before padding.
    pub head_end: u64,
    /// Absolute offset where the tensor-data section begins.
    pub data_start: u64,
    /// Total file length in bytes.
    pub file_len: u64,
}

impl Gguf {
    /// First value for `key`, in file order.
    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.kvs.iter().find(|kv| kv.key == key).map(|kv| &kv.value)
    }

    /// Length of the tensor-data section (zero for metadata-only files).
    pub fn data_len(&self) -> u64 {
        self.file_len.saturating_sub(self.data_start)
    }
}

/// A `Read` wrapper that tracks the byte position for error reporting.
struct Src<R: Read> {
    r: R,
    pos: u64,
}

impl<R: Read> Src<R> {
    fn fill(&mut self, buf: &mut [u8]) -> Result<()> {
        match self.r.read_exact(buf) {
            Ok(()) => {
                self.pos += buf.len() as u64;
                Ok(())
            }
            Err(e) if e.kind() == ErrorKind::UnexpectedEof => err(
                self.pos,
                "unexpected end of file while reading the head (truncated file?)",
            ),
            Err(e) => err(self.pos, format!("read error: {e}")),
        }
    }

    fn u8(&mut self) -> Result<u8> {
        let mut b = [0u8; 1];
        self.fill(&mut b)?;
        Ok(b[0])
    }

    fn u32(&mut self) -> Result<u32> {
        let mut b = [0u8; 4];
        self.fill(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn u64(&mut self) -> Result<u64> {
        let mut b = [0u8; 8];
        self.fill(&mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    fn string(&mut self, what: &str) -> Result<String> {
        let at = self.pos;
        let len = self.u64()?;
        if len > MAX_STRING_LEN {
            return err(
                at,
                format!("{what} length {len} exceeds the {MAX_STRING_LEN}-byte sanity cap"),
            );
        }
        let mut buf = vec![0u8; len as usize];
        self.fill(&mut buf)?;
        String::from_utf8(buf).map_err(|_| GgufError {
            message: format!("{what} is not valid UTF-8"),
            offset: at,
        })
    }

    fn value(&mut self, ty: GgufType, depth: u32) -> Result<GgufValue> {
        use GgufType::*;
        Ok(match ty {
            U8 => GgufValue::U8(self.u8()?),
            I8 => GgufValue::I8(self.u8()? as i8),
            U16 => {
                let mut b = [0u8; 2];
                self.fill(&mut b)?;
                GgufValue::U16(u16::from_le_bytes(b))
            }
            I16 => {
                let mut b = [0u8; 2];
                self.fill(&mut b)?;
                GgufValue::I16(i16::from_le_bytes(b))
            }
            U32 => GgufValue::U32(self.u32()?),
            I32 => GgufValue::I32(self.u32()? as i32),
            F32 => GgufValue::F32(f32::from_le_bytes({
                let mut b = [0u8; 4];
                self.fill(&mut b)?;
                b
            })),
            Bool => GgufValue::Bool(self.u8()? != 0),
            Str => GgufValue::Str(self.string("string value")?),
            U64 => GgufValue::U64(self.u64()?),
            I64 => GgufValue::I64(self.u64()? as i64),
            F64 => GgufValue::F64(f64::from_le_bytes({
                let mut b = [0u8; 8];
                self.fill(&mut b)?;
                b
            })),
            Array => {
                if depth >= MAX_ARRAY_DEPTH {
                    return err(
                        self.pos,
                        format!("arrays nested deeper than {MAX_ARRAY_DEPTH} levels"),
                    );
                }
                let at = self.pos;
                let code = self.u32()?;
                let elem = GgufType::from_code(code).ok_or_else(|| GgufError {
                    message: format!("unknown array element type code {code}"),
                    offset: at,
                })?;
                let count = self.u64()?;
                if count > MAX_ARRAY_LEN {
                    return err(
                        at,
                        format!(
                            "array length {count} exceeds the {MAX_ARRAY_LEN}-element sanity cap"
                        ),
                    );
                }
                let mut items = Vec::with_capacity(count.min(1 << 16) as usize);
                for _ in 0..count {
                    items.push(self.value(elem, depth + 1)?);
                }
                GgufValue::Array(elem, items)
            }
        })
    }
}

/// Determine the effective alignment from parsed metadata.
fn effective_alignment(kvs: &[KvPair], head_end: u64) -> Result<u64> {
    match kvs.iter().find(|kv| kv.key == ALIGNMENT_KEY) {
        None => Ok(DEFAULT_ALIGNMENT),
        Some(kv) => match kv.value.as_u64() {
            Some(0) => err(head_end, "general.alignment is 0, which is invalid"),
            Some(a) => Ok(a),
            None => err(
                head_end,
                format!(
                    "general.alignment must be an unsigned integer, found {}",
                    kv.value.type_label()
                ),
            ),
        },
    }
}

/// Parse a GGUF head from any reader. `file_len` is the total file size and
/// is only used to compute the data-section extent, never to read ahead.
pub fn parse_head<R: Read>(reader: R, file_len: u64) -> Result<Gguf> {
    let mut s = Src { r: reader, pos: 0 };

    let mut magic = [0u8; 4];
    s.fill(&mut magic)?;
    if magic != MAGIC {
        return err(
            0,
            format!("not a GGUF file (magic {magic:02x?}, expected \"GGUF\")"),
        );
    }

    let version = s.u32()?;
    match version {
        2 | 3 => {}
        1 => {
            return err(
                4,
                "GGUF v1 (32-bit lengths) is not supported; convert it to v3 first",
            )
        }
        v if (1..=3).contains(&v.swap_bytes()) => {
            return err(
                4,
                "big-endian GGUF is not supported (the version field is byte-swapped)",
            )
        }
        v => return err(4, format!("unsupported GGUF version {v}")),
    }

    let tensor_count = s.u64()?;
    if tensor_count > MAX_TENSOR_COUNT {
        return err(
            8,
            format!("tensor count {tensor_count} exceeds the sanity cap"),
        );
    }
    let kv_count = s.u64()?;
    if kv_count > MAX_KV_COUNT {
        return err(
            16,
            format!("metadata count {kv_count} exceeds the sanity cap"),
        );
    }

    let mut kvs = Vec::with_capacity(kv_count.min(1 << 12) as usize);
    for _ in 0..kv_count {
        let key = s.string("metadata key")?;
        let at = s.pos;
        let code = s.u32()?;
        let ty = GgufType::from_code(code).ok_or_else(|| GgufError {
            message: format!("unknown metadata value type code {code} for key '{key}'"),
            offset: at,
        })?;
        let value = s.value(ty, 0)?;
        kvs.push(KvPair { key, value });
    }

    let mut tensors = Vec::with_capacity(tensor_count.min(1 << 12) as usize);
    for _ in 0..tensor_count {
        let name = s.string("tensor name")?;
        let at = s.pos;
        let n_dims = s.u32()?;
        if n_dims == 0 || n_dims > MAX_DIMS {
            return err(
                at,
                format!("tensor '{name}' has {n_dims} dimensions (GGUF allows 1..={MAX_DIMS})"),
            );
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(s.u64()?);
        }
        let type_code = s.u32()?;
        let offset = s.u64()?;
        tensors.push(TensorInfo {
            name,
            dims,
            type_code,
            offset,
        });
    }

    let head_end = s.pos;
    let alignment = effective_alignment(&kvs, head_end)?;
    let data_start = align_up(head_end, alignment);

    Ok(Gguf {
        version,
        kvs,
        tensors,
        alignment,
        head_end,
        data_start,
        file_len,
    })
}

/// Open `path` and parse its head through a buffered reader.
pub fn read_head(path: &Path) -> Result<Gguf> {
    let file = File::open(path).map_err(|e| GgufError {
        message: format!("cannot open {}: {e}", path.display()),
        offset: 0,
    })?;
    let file_len = file
        .metadata()
        .map_err(|e| GgufError {
            message: format!("cannot stat {}: {e}", path.display()),
            offset: 0,
        })?
        .len();
    parse_head(BufReader::new(file), file_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-rolled byte builder, deliberately independent from `writer` so a
    /// shared encoding bug cannot cancel itself out across the two modules.
    struct Raw(Vec<u8>);

    impl Raw {
        fn new(version: u32, tensor_count: u64, kv_count: u64) -> Raw {
            let mut b = Vec::new();
            b.extend_from_slice(b"GGUF");
            b.extend_from_slice(&version.to_le_bytes());
            b.extend_from_slice(&tensor_count.to_le_bytes());
            b.extend_from_slice(&kv_count.to_le_bytes());
            Raw(b)
        }
        fn str(&mut self, s: &str) -> &mut Self {
            self.0.extend_from_slice(&(s.len() as u64).to_le_bytes());
            self.0.extend_from_slice(s.as_bytes());
            self
        }
        fn u32(&mut self, v: u32) -> &mut Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn u64(&mut self, v: u64) -> &mut Self {
            self.0.extend_from_slice(&v.to_le_bytes());
            self
        }
        fn bytes(&mut self, v: &[u8]) -> &mut Self {
            self.0.extend_from_slice(v);
            self
        }
        fn parse(&self) -> Result<Gguf> {
            parse_head(&self.0[..], self.0.len() as u64)
        }
    }

    #[test]
    fn parses_minimal_v3_and_v2_files() {
        let g = Raw::new(3, 0, 0).parse().expect("minimal file parses");
        assert_eq!(g.version, 3);
        assert!(g.kvs.is_empty());
        assert!(g.tensors.is_empty());
        assert_eq!(g.head_end, 24);
        assert_eq!(g.alignment, 32);
        assert_eq!(g.data_start, 32, "24 rounded up to the default alignment");

        let g = Raw::new(2, 0, 0).parse().expect("v2 parses");
        assert_eq!(g.version, 2);
    }

    #[test]
    fn parses_scalar_metadata_of_every_type() {
        let mut r = Raw::new(3, 0, 4);
        r.str("a.u8").u32(0).bytes(&[7]);
        r.str("a.i16").u32(3).bytes(&(-2i16).to_le_bytes());
        r.str("a.f32").u32(6).bytes(&1.5f32.to_le_bytes());
        r.str("a.bool").u32(7).bytes(&[1]);
        let g = r.parse().expect("scalars parse");
        assert_eq!(g.get("a.u8"), Some(&GgufValue::U8(7)));
        assert_eq!(g.get("a.i16"), Some(&GgufValue::I16(-2)));
        assert_eq!(g.get("a.f32"), Some(&GgufValue::F32(1.5)));
        assert_eq!(g.get("a.bool"), Some(&GgufValue::Bool(true)));
    }

    #[test]
    fn parses_string_and_string_array_values() {
        let mut r = Raw::new(3, 0, 2);
        r.str("general.name").u32(8).str("demo");
        r.str("tokenizer.ggml.tokens").u32(9).u32(8).u64(2);
        r.str("<s>").str("</s>");
        let g = r.parse().expect("strings parse");
        assert_eq!(g.get("general.name"), Some(&GgufValue::Str("demo".into())));
        assert_eq!(
            g.get("tokenizer.ggml.tokens"),
            Some(&GgufValue::Array(
                GgufType::Str,
                vec![GgufValue::Str("<s>".into()), GgufValue::Str("</s>".into())]
            ))
        );
    }

    #[test]
    fn parses_nested_arrays() {
        // array<array>[1] whose single element is array<u32>[2] {5, 6}
        let mut r = Raw::new(3, 0, 1);
        r.str("nested")
            .u32(9)
            .u32(9)
            .u64(1)
            .u32(4)
            .u64(2)
            .u32(5)
            .u32(6);
        let g = r.parse().expect("nested arrays parse");
        let expected = GgufValue::Array(
            GgufType::Array,
            vec![GgufValue::Array(
                GgufType::U32,
                vec![GgufValue::U32(5), GgufValue::U32(6)],
            )],
        );
        assert_eq!(g.get("nested"), Some(&expected));
    }

    #[test]
    fn parses_tensor_descriptors_and_computes_data_start() {
        let mut r = Raw::new(3, 1, 0);
        r.str("token_embd.weight")
            .u32(2)
            .u64(8)
            .u64(4)
            .u32(0)
            .u64(0);
        let g = r.parse().expect("tensor parses");
        assert_eq!(g.tensors.len(), 1);
        assert_eq!(g.tensors[0].dims, vec![8, 4]);
        assert_eq!(g.tensors[0].type_code, 0);
        assert_eq!(g.data_start, align_up(g.head_end, 32));
    }

    #[test]
    fn alignment_key_is_honored_and_validated() {
        let mut r = Raw::new(3, 0, 1);
        r.str("general.alignment").u32(4).u32(64);
        let g = r.parse().expect("alignment parses");
        assert_eq!(g.alignment, 64);
        assert_eq!(g.data_start % 64, 0);

        let mut zero = Raw::new(3, 0, 1);
        zero.str("general.alignment").u32(4).u32(0);
        let e = zero.parse().unwrap_err();
        assert!(e.message.contains("alignment is 0"), "{e}");

        let mut stringy = Raw::new(3, 0, 1);
        stringy.str("general.alignment").u32(8).str("32");
        let e = stringy.parse().unwrap_err();
        assert!(e.message.contains("unsigned integer"), "{e}");
    }

    #[test]
    fn rejects_bad_magic_with_hex_context() {
        let mut b = Raw::new(3, 0, 0).0;
        b[0] = b'X';
        let e = parse_head(&b[..], b.len() as u64).unwrap_err();
        assert!(e.message.contains("not a GGUF file"), "{e}");
        assert_eq!(e.offset, 0);
    }

    #[test]
    fn rejects_v1_big_endian_and_future_versions() {
        let e = Raw::new(1, 0, 0).parse().unwrap_err();
        assert!(e.message.contains("v1"), "{e}");
        assert!(e.message.contains("not supported"), "{e}");

        let e = Raw::new(3u32.swap_bytes(), 0, 0).parse().unwrap_err();
        assert!(e.message.contains("big-endian"), "{e}");

        let e = Raw::new(99, 0, 0).parse().unwrap_err();
        assert!(e.message.contains("unsupported GGUF version 99"), "{e}");
    }

    #[test]
    fn rejects_truncated_oversized_and_non_utf8_strings() {
        let mut r = Raw::new(3, 0, 1);
        r.str("k").u32(8).u64(1000); // declares 1000 bytes, provides none
        let e = r.parse().unwrap_err();
        assert!(e.message.contains("unexpected end of file"), "{e}");

        let mut r = Raw::new(3, 0, 1);
        r.str("k").u32(8).u64(u64::MAX);
        let e = r.parse().unwrap_err();
        assert!(e.message.contains("sanity cap"), "{e}");

        let mut r = Raw::new(3, 0, 1);
        r.str("k").u32(8).u64(2).bytes(&[0xff, 0xfe]);
        let e = r.parse().unwrap_err();
        assert!(e.message.contains("not valid UTF-8"), "{e}");
    }

    #[test]
    fn rejects_unknown_value_type_code_naming_the_key() {
        let mut r = Raw::new(3, 0, 1);
        r.str("weird.key").u32(13).u32(0);
        let e = r.parse().unwrap_err();
        assert!(e.message.contains("type code 13"), "{e}");
        assert!(e.message.contains("weird.key"), "{e}");
    }

    #[test]
    fn rejects_tensors_with_zero_or_too_many_dims() {
        let mut zero = Raw::new(3, 1, 0);
        zero.str("t").u32(0).u32(0).u64(0);
        let e = zero.parse().unwrap_err();
        assert!(e.message.contains("0 dimensions"), "{e}");

        let mut five = Raw::new(3, 1, 0);
        five.str("t").u32(5);
        for _ in 0..5 {
            five.u64(1);
        }
        five.u32(0).u64(0);
        let e = five.parse().unwrap_err();
        assert!(e.message.contains("5 dimensions"), "{e}");
    }

    #[test]
    fn data_len_is_zero_when_file_ends_at_the_head() {
        let g = Raw::new(3, 0, 0).parse().unwrap();
        // file_len (24) < data_start (32): no data section, not an error here.
        assert_eq!(g.data_len(), 0);
    }
}
