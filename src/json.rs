//! Dependency-free JSON parser and encoder, sized for gguf-chisel's needs:
//! `dump` output and `apply` patch documents. Integers are kept as `i128` so
//! GGUF `u64`/`i64` metadata values round-trip without floating-point loss.
//! Objects preserve insertion order (a `Vec`, not a map) so dumps are stable
//! and patch documents apply in a predictable order.

use std::fmt;

/// A parsed JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i128),
    Float(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Look up a key in an object (first match, file order).
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
}

/// A parse error with 1-based line and column.
#[derive(Debug)]
pub struct JsonError {
    pub message: String,
    pub line: usize,
    pub col: usize,
}

impl fmt::Display for JsonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} at {}:{}", self.message, self.line, self.col)
    }
}

impl std::error::Error for JsonError {}

const MAX_DEPTH: usize = 128;

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn line_col(&self) -> (usize, usize) {
        let mut line = 1;
        let mut col = 1;
        for &b in &self.bytes[..self.pos.min(self.bytes.len())] {
            if b == b'\n' {
                line += 1;
                col = 1;
            } else {
                col += 1;
            }
        }
        (line, col)
    }

    fn fail<T>(&self, message: impl Into<String>) -> Result<T, JsonError> {
        let (line, col) = self.line_col();
        Err(JsonError {
            message: message.into(),
            line,
            col,
        })
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn expect(&mut self, b: u8) -> Result<(), JsonError> {
        if self.peek() == Some(b) {
            self.pos += 1;
            Ok(())
        } else {
            self.fail(format!("expected '{}'", b as char))
        }
    }

    fn value(&mut self, depth: usize) -> Result<Json, JsonError> {
        if depth > MAX_DEPTH {
            return self.fail(format!("nesting deeper than {MAX_DEPTH} levels"));
        }
        self.skip_ws();
        match self.peek() {
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Json::Str(self.string()?)),
            Some(b't') => self.literal("true", Json::Bool(true)),
            Some(b'f') => self.literal("false", Json::Bool(false)),
            Some(b'n') => self.literal("null", Json::Null),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(c) => self.fail(format!("unexpected character '{}'", c as char)),
            None => self.fail("unexpected end of input"),
        }
    }

    fn literal(&mut self, word: &str, value: Json) -> Result<Json, JsonError> {
        if self.bytes[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(value)
        } else {
            self.fail(format!("invalid literal (expected '{word}')"))
        }
    }

    fn object(&mut self, depth: usize) -> Result<Json, JsonError> {
        self.expect(b'{')?;
        let mut pairs = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Json::Obj(pairs));
        }
        loop {
            self.skip_ws();
            if self.peek() != Some(b'"') {
                return self.fail("expected a string key");
            }
            let key = self.string()?;
            self.skip_ws();
            self.expect(b':')?;
            let value = self.value(depth + 1)?;
            pairs.push((key, value));
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(pairs));
                }
                _ => return self.fail("expected ',' or '}'"),
            }
        }
    }

    fn array(&mut self, depth: usize) -> Result<Json, JsonError> {
        self.expect(b'[')?;
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.skip_ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return self.fail("expected ',' or ']'"),
            }
        }
    }

    fn hex4(&mut self) -> Result<u16, JsonError> {
        if self.pos + 4 > self.bytes.len() {
            return self.fail("truncated \\u escape");
        }
        let s =
            std::str::from_utf8(&self.bytes[self.pos..self.pos + 4]).map_err(|_| JsonError {
                message: "invalid \\u escape".into(),
                line: 0,
                col: 0,
            })?;
        let v = u16::from_str_radix(s, 16);
        match v {
            Ok(v) => {
                self.pos += 4;
                Ok(v)
            }
            Err(_) => self.fail("invalid \\u escape"),
        }
    }

    fn string(&mut self) -> Result<String, JsonError> {
        self.expect(b'"')?;
        let mut out = String::new();
        loop {
            match self.peek() {
                None => return self.fail("unterminated string"),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    match self.peek() {
                        Some(b'"') => out.push('"'),
                        Some(b'\\') => out.push('\\'),
                        Some(b'/') => out.push('/'),
                        Some(b'b') => out.push('\u{8}'),
                        Some(b'f') => out.push('\u{c}'),
                        Some(b'n') => out.push('\n'),
                        Some(b'r') => out.push('\r'),
                        Some(b't') => out.push('\t'),
                        Some(b'u') => {
                            self.pos += 1;
                            let hi = self.hex4()?;
                            let c = if (0xD800..0xDC00).contains(&hi) {
                                // Surrogate pair: require \uXXXX low half.
                                if self.peek() != Some(b'\\') {
                                    return self.fail("unpaired surrogate");
                                }
                                self.pos += 1;
                                if self.peek() != Some(b'u') {
                                    return self.fail("unpaired surrogate");
                                }
                                self.pos += 1;
                                let lo = self.hex4()?;
                                if !(0xDC00..0xE000).contains(&lo) {
                                    return self.fail("invalid low surrogate");
                                }
                                let cp =
                                    0x10000 + ((hi as u32 - 0xD800) << 10) + (lo as u32 - 0xDC00);
                                char::from_u32(cp)
                            } else {
                                char::from_u32(hi as u32)
                            };
                            match c {
                                Some(c) => {
                                    out.push(c);
                                    continue;
                                }
                                None => return self.fail("invalid unicode escape"),
                            }
                        }
                        _ => return self.fail("invalid escape sequence"),
                    }
                    self.pos += 1;
                }
                Some(b) if b < 0x20 => return self.fail("raw control character in string"),
                Some(_) => {
                    // Copy one UTF-8 scalar (input is valid UTF-8 by contract).
                    let start = self.pos;
                    self.pos += 1;
                    while self.pos < self.bytes.len() && (self.bytes[self.pos] & 0xC0) == 0x80 {
                        self.pos += 1;
                    }
                    out.push_str(std::str::from_utf8(&self.bytes[start..self.pos]).map_err(
                        |_| JsonError {
                            message: "invalid UTF-8 in string".into(),
                            line: 0,
                            col: 0,
                        },
                    )?);
                }
            }
        }
    }

    fn number(&mut self) -> Result<Json, JsonError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        // JSON forbids leading zeros: "0" alone or a 1-9 start.
        match self.peek() {
            Some(b'0') => {
                self.pos += 1;
                if matches!(self.peek(), Some(b'0'..=b'9')) {
                    return self.fail("leading zeros are not allowed");
                }
            }
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return self.fail("invalid number"),
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return self.fail("digit expected after decimal point");
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return self.fail("digit expected in exponent");
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap();
        if is_float {
            text.parse::<f64>()
                .map(Json::Float)
                .or_else(|_| self.fail("invalid number"))
        } else {
            match text.parse::<i128>() {
                Ok(v) => Ok(Json::Int(v)),
                // Out-of-i128-range integers degrade to float rather than fail.
                Err(_) => text
                    .parse::<f64>()
                    .map(Json::Float)
                    .or_else(|_| self.fail("invalid number")),
            }
        }
    }
}

/// Parse a complete JSON document; trailing non-whitespace is an error.
pub fn parse(input: &str) -> Result<Json, JsonError> {
    let mut p = Parser {
        bytes: input.as_bytes(),
        pos: 0,
    };
    let v = p.value(0)?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        return p.fail("trailing characters after the JSON value");
    }
    Ok(v)
}

/// Escape a string for embedding in JSON output (without the quotes).
pub fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn encode_into(v: &Json, out: &mut String, indent: usize, pretty: bool) {
    let pad = |out: &mut String, n: usize| {
        if pretty {
            out.push('\n');
            for _ in 0..n {
                out.push_str("  ");
            }
        }
    };
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Int(i) => out.push_str(&i.to_string()),
        Json::Float(f) => {
            if f.is_finite() {
                out.push_str(&format!("{f:?}"));
            } else {
                // NaN/Inf have no JSON spelling; null keeps the output valid.
                out.push_str("null");
            }
        }
        Json::Str(s) => {
            out.push('"');
            out.push_str(&escape(s));
            out.push('"');
        }
        Json::Arr(items) => {
            if items.is_empty() {
                out.push_str("[]");
                return;
            }
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                pad(out, indent + 1);
                encode_into(item, out, indent + 1, pretty);
            }
            pad(out, indent);
            out.push(']');
        }
        Json::Obj(pairs) => {
            if pairs.is_empty() {
                out.push_str("{}");
                return;
            }
            out.push('{');
            for (i, (k, val)) in pairs.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                pad(out, indent + 1);
                out.push('"');
                out.push_str(&escape(k));
                out.push_str(if pretty { "\": " } else { "\":" });
                encode_into(val, out, indent + 1, pretty);
            }
            pad(out, indent);
            out.push('}');
        }
    }
}

/// Compact single-line encoding.
pub fn encode(v: &Json) -> String {
    let mut out = String::new();
    encode_into(v, &mut out, 0, false);
    out
}

/// Pretty encoding with two-space indentation, as printed by `dump`.
pub fn encode_pretty(v: &Json) -> String {
    let mut out = String::new();
    encode_into(v, &mut out, 0, true);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_scalars_including_exact_u64() {
        assert_eq!(parse("null").unwrap(), Json::Null);
        assert_eq!(parse("true").unwrap(), Json::Bool(true));
        assert_eq!(parse("false").unwrap(), Json::Bool(false));
        assert_eq!(parse("42").unwrap(), Json::Int(42));
        assert_eq!(parse("-7").unwrap(), Json::Int(-7));
        assert_eq!(parse("1.5").unwrap(), Json::Float(1.5));
        assert_eq!(parse("-0.5e3").unwrap(), Json::Float(-500.0));
        assert_eq!(parse("\"hi\"").unwrap(), Json::Str("hi".into()));

        // This is the whole reason Int is i128: GGUF u64 values must not be
        // squeezed through f64 (2^53) on their way in or out.
        let v = parse("18446744073709551615").unwrap();
        assert_eq!(v, Json::Int(u64::MAX as i128));
        assert_eq!(encode(&v), "18446744073709551615");
    }

    #[test]
    fn string_escapes_roundtrip_including_surrogate_pairs() {
        let v = parse(r#""a\n\t\"\\ é 😀""#).unwrap();
        assert_eq!(v, Json::Str("a\n\t\"\\ é 😀".into()));
        let encoded = encode(&v);
        assert_eq!(parse(&encoded).unwrap(), v);
    }

    #[test]
    fn object_key_order_is_preserved() {
        let v = parse(r#"{"z":1,"a":2,"m":3}"#).unwrap();
        let Json::Obj(pairs) = &v else {
            panic!("not an object")
        };
        let keys: Vec<&str> = pairs.iter().map(|(k, _)| k.as_str()).collect();
        assert_eq!(keys, ["z", "a", "m"]);
        assert_eq!(v.get("a"), Some(&Json::Int(2)));

        let src = r#"{"set":{"k":{"type":"u32","value":4096}},"delete":["a","b"]}"#;
        let v = parse(src).unwrap();
        assert_eq!(encode(&v), src);
    }

    #[test]
    fn pretty_encoding_uses_two_space_indent() {
        let v = parse(r#"{"a":[1,2]}"#).unwrap();
        let pretty = encode_pretty(&v);
        assert_eq!(pretty, "{\n  \"a\": [\n    1,\n    2\n  ]\n}");
        assert_eq!(parse(&pretty).unwrap(), v);
    }

    #[test]
    fn rejects_malformed_documents() {
        let e = parse("{} extra").unwrap_err();
        assert!(e.message.contains("trailing"), "{e}");

        assert!(parse("012").is_err());
        assert!(parse("tru").is_err());
        assert!(parse("NaN").is_err());

        assert!(parse("\"abc").is_err());
        assert!(parse(r#""\q""#).is_err());
        assert!(parse(r#""\ud800""#).is_err(), "lone high surrogate");
    }

    #[test]
    fn error_carries_line_and_column() {
        let e = parse("{\n  \"a\": ,\n}").unwrap_err();
        assert_eq!(e.line, 2);
        assert!(e.col > 1);
    }

    #[test]
    fn depth_cap_prevents_stack_exhaustion() {
        let deep = "[".repeat(200) + &"]".repeat(200);
        let e = parse(&deep).unwrap_err();
        assert!(e.message.contains("nesting"), "{e}");
    }

    #[test]
    fn encode_handles_nonfinite_floats_and_control_characters() {
        assert_eq!(encode(&Json::Float(f64::NAN)), "null");
        assert_eq!(encode(&Json::Float(f64::INFINITY)), "null");

        let s = Json::Str("a\u{1}b".into());
        assert_eq!(encode(&s), "\"a\\u0001b\"");
        assert_eq!(parse(&encode(&s)).unwrap(), s);
    }
}
