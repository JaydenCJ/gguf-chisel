//! Command-line interface: argument parsing, help texts and command
//! dispatch. Kept dependency-free on purpose; every command is a thin shell
//! around the pure modules so behavior stays unit-testable there.

use crate::dump::{dump_json, fmt_size, render_show};
use crate::patch::{self, EditOp};
use crate::reader::{read_head, Gguf};
use crate::template;
use crate::types::{GgufValue, CHAT_TEMPLATE_KEY};
use crate::verify::verify;
use crate::writer::{self, FitPlan};
use std::io::Read;
use std::path::{Path, PathBuf};

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

const HELP: &str = "\
gguf-chisel — surgical GGUF metadata editing: patch keys, fix chat templates, no tensor rewrite

USAGE:
    gguf-chisel <COMMAND> [ARGS] [OPTIONS]

COMMANDS:
    show      Summarize header, metadata keys and tensors
    get       Print a single metadata value
    set       Patch metadata keys (in place whenever the head fits)
    rm        Remove metadata keys
    rename    Rename a metadata key
    dump      Dump the whole head as pretty JSON
    apply     Apply a JSON patch document (delete/rename/set)
    template  Show, set, lint or list chat templates
    verify    Check structure, offsets and alignment
    sample    Write a tiny valid GGUF for pipeline tests

OPTIONS:
    -h, --help       Print this help
    -V, --version    Print version

Write commands accept --dry-run, --rewrite, -o/--output and --reserve;
run 'gguf-chisel <COMMAND> --help' for details.";

const WRITE_OPTS: &str = "\
        --dry-run          Plan only: exit 0 if the edit fits in place, 3 if not
        --rewrite          Allow a full rewrite when the head has outgrown its space
    -o, --output <FILE>    Write the result to FILE instead of editing in place
        --reserve <N>      On rewrite, reserve N bytes of headroom (accepts K/M/G)";

const SHOW_HELP: &str = "\
gguf-chisel show — summarize header, metadata keys and tensors

USAGE:
    gguf-chisel show <FILE> [OPTIONS]

OPTIONS:
        --tensors    Also list every tensor (name, type, dims, offset)
    -h, --help       Print this help";

const GET_HELP: &str = "\
gguf-chisel get — print a single metadata value

USAGE:
    gguf-chisel get <FILE> <KEY> [OPTIONS]

OPTIONS:
        --json    Print {\"key\", \"type\", \"value\"} instead of the bare value
        --raw     Print string values byte-exact, with no trailing newline
    -h, --help    Print this help";

const SET_HELP: &str = "\
gguf-chisel set — patch metadata keys (in place whenever the head fits)

USAGE:
    gguf-chisel set <FILE> <KEY=VALUE>... [OPTIONS]

VALUES:
    KEY=VALUE          Existing keys keep their wire type; new keys infer
                       bool / u32 / i32 / u64 / i64 / f32 / string
    KEY=TYPE:VALUE     Force a type: u8 i8 u16 i16 u32 i32 u64 i64 f32 f64
                       bool str (e.g. ctx.len=u32:32768, note=str:42)

OPTIONS:
{WRITE_OPTS}
    -h, --help             Print this help";

const RM_HELP: &str = "\
gguf-chisel rm — remove metadata keys

USAGE:
    gguf-chisel rm <FILE> <KEY>... [OPTIONS]

OPTIONS:
{WRITE_OPTS}
    -h, --help             Print this help";

const RENAME_HELP: &str = "\
gguf-chisel rename — rename a metadata key, keeping its value and position

USAGE:
    gguf-chisel rename <FILE> <OLD_KEY> <NEW_KEY> [OPTIONS]

OPTIONS:
{WRITE_OPTS}
    -h, --help             Print this help";

const DUMP_HELP: &str = "\
gguf-chisel dump — dump the whole head as pretty JSON

USAGE:
    gguf-chisel dump <FILE>

OPTIONS:
    -h, --help    Print this help";

const APPLY_HELP: &str = "\
gguf-chisel apply — apply a JSON patch document

USAGE:
    gguf-chisel apply <FILE> <PATCH.json> [OPTIONS]

The patch document has up to three sections, applied in this order:
    {\"delete\": [\"key\", ...],
     \"rename\": {\"old\": \"new\", ...},
     \"set\":    {\"key\": value-or-{\"type\":...,\"value\":...}, ...}}
Pass '-' as PATCH.json to read the document from stdin.

OPTIONS:
{WRITE_OPTS}
    -h, --help             Print this help";

const TEMPLATE_HELP: &str = "\
gguf-chisel template — show, set, lint or list chat templates

USAGE:
    gguf-chisel template show <FILE>
    gguf-chisel template set <FILE> (--preset <NAME> | --file <TPL>) [OPTIONS]
    gguf-chisel template check (<FILE> | --file <TPL>) [--strict]
    gguf-chisel template presets

OPTIONS:
        --preset <NAME>    Install a built-in template (see 'template presets')
        --file <TPL>       Read the template from a file
        --strict           check: treat warnings as failures
{WRITE_OPTS}
    -h, --help             Print this help";

const VERIFY_HELP: &str = "\
gguf-chisel verify — check structure, offsets and alignment

USAGE:
    gguf-chisel verify <FILE>

Exit code 0 when the file is structurally sound (warnings allowed), 1 when
any error is found.

OPTIONS:
    -h, --help    Print this help";

const SAMPLE_HELP: &str = "\
gguf-chisel sample — write a tiny valid GGUF for pipeline tests

USAGE:
    gguf-chisel sample <FILE> [OPTIONS]

The sample is deterministic: nine metadata keys (including a ChatML chat
template) and two small F32 tensors, ~1 KiB total.

OPTIONS:
        --reserve <N>    Reserve N bytes of in-place headroom (accepts K/M/G)
    -h, --help           Print this help";

/// Write to stdout, exiting quietly when the pipe has been closed. Rust
/// ignores SIGPIPE, so without this a downstream `head` or `grep -q` (which
/// stops reading early) would turn every print into a panic mid-pipeline; a
/// CLI should behave like any Unix filter instead.
macro_rules! out {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        if write!(std::io::stdout(), $($arg)*).is_err() {
            std::process::exit(0);
        }
    }};
}

/// Line-terminated [`out!`].
macro_rules! outln {
    ($($arg:tt)*) => {{
        use std::io::Write as _;
        if writeln!(std::io::stdout(), $($arg)*).is_err() {
            std::process::exit(0);
        }
    }};
}

/// An error carrying the intended process exit code.
pub struct CliError {
    pub message: String,
    pub code: i32,
}

fn usage(message: impl Into<String>) -> CliError {
    CliError {
        message: message.into(),
        code: 2,
    }
}

fn failure(message: impl Into<String>) -> CliError {
    CliError {
        message: message.into(),
        code: 1,
    }
}

type CmdResult = Result<i32, CliError>;

/// `1 error` / `2 errors` — counts read like prose, never "1 errors".
fn plural(n: usize, word: &str) -> String {
    if n == 1 {
        format!("{n} {word}")
    } else {
        format!("{n} {word}s")
    }
}

/// Options shared by every command that writes.
#[derive(Default)]
struct WriteOpts {
    dry_run: bool,
    rewrite: bool,
    output: Option<PathBuf>,
    reserve: u64,
}

/// Split `args` into positionals and recognized flags; `write_opts == None`
/// means the command is read-only and write flags are rejected like any
/// other unknown option.
fn parse_args(
    args: &[String],
    help: &str,
    write_opts: Option<&mut WriteOpts>,
    flags: &mut [(&str, &mut bool)],
) -> Result<Option<Vec<String>>, CliError> {
    let mut positionals = Vec::new();
    let mut wo = write_opts;
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let mut matched = false;
        for (name, slot) in flags.iter_mut() {
            if arg == name {
                **slot = true;
                matched = true;
                break;
            }
        }
        if matched {
            i += 1;
            continue;
        }
        match arg.as_str() {
            "-h" | "--help" => {
                outln!("{}", help.replace("{WRITE_OPTS}", WRITE_OPTS));
                return Ok(None);
            }
            "--dry-run" if wo.is_some() => {
                wo.as_mut().unwrap().dry_run = true;
            }
            "--rewrite" if wo.is_some() => {
                wo.as_mut().unwrap().rewrite = true;
            }
            "-o" | "--output" if wo.is_some() => {
                i += 1;
                let value = args
                    .get(i)
                    .ok_or_else(|| usage("-o/--output needs a path"))?;
                wo.as_mut().unwrap().output = Some(PathBuf::from(value));
            }
            "--reserve" if wo.is_some() => {
                i += 1;
                let value = args.get(i).ok_or_else(|| usage("--reserve needs a size"))?;
                wo.as_mut().unwrap().reserve =
                    patch::parse_size(value).map_err(|e| usage(e.to_string()))?;
            }
            other if other.starts_with('-') && other.len() > 1 => {
                return Err(usage(format!("unknown option '{other}' (see --help)")));
            }
            _ => positionals.push(arg.clone()),
        }
        i += 1;
    }
    Ok(Some(positionals))
}

fn load(path: &Path) -> Result<Gguf, CliError> {
    read_head(path).map_err(|e| failure(e.to_string()))
}

/// Write the edited head back, in place when it fits, honoring the write
/// options. This is the single choke point every mutating command goes
/// through.
fn commit(path: &Path, g: &Gguf, opts: &WriteOpts) -> CmdResult {
    let plan = writer::plan_in_place(g.version, &g.kvs, &g.tensors, g.data_start, g.alignment);

    if opts.dry_run {
        return match plan {
            FitPlan::InPlace { head, zero_fill } => {
                outln!(
                    "dry-run: fits in place ({} head bytes + {} bytes of alignment padding; \
                     tensor data untouched)",
                    head.len(),
                    zero_fill
                );
                Ok(0)
            }
            FitPlan::NeedsRewrite { needed, available } => {
                outln!(
                    "dry-run: needs a rewrite (head needs {needed} bytes, \
                     {available} available before tensor data)"
                );
                Ok(3)
            }
        };
    }

    match (plan, &opts.output) {
        (FitPlan::InPlace { head, zero_fill }, None) => {
            let written = writer::write_in_place(path, &head, zero_fill)
                .map_err(|e| failure(format!("write failed: {e}")))?;
            if opts.reserve > 0 {
                outln!("note: --reserve only applies to rewrites; existing headroom kept");
            }
            outln!(
                "patched {} in place: {written} head bytes written, tensor data untouched",
                path.display()
            );
            Ok(0)
        }
        (FitPlan::NeedsRewrite { needed, available }, None) if !opts.rewrite => {
            Err(failure(format!(
                "metadata head grew beyond the available space (need {needed} bytes, have \
             {available}); re-run with --rewrite (optionally -o NEW.gguf), and consider \
             --reserve to leave headroom for future in-place edits"
            )))
        }
        (_, output) => {
            // Rewrite: either forced by -o, or required and allowed by --rewrite.
            let (dst, in_place) = match output {
                Some(out) => (out.clone(), false),
                None => (path.with_extension("gguf.tmp-chisel"), true),
            };
            let (head_len, copied) =
                writer::rewrite_file(path, &dst, g, opts.reserve).map_err(|e| {
                    // Never leave a half-written temp file behind.
                    if in_place {
                        let _ = std::fs::remove_file(&dst);
                    }
                    failure(format!("rewrite failed: {e}"))
                })?;
            let final_path = if in_place {
                std::fs::rename(&dst, path).map_err(|e| {
                    let _ = std::fs::remove_file(&dst);
                    failure(format!("rename failed: {e}"))
                })?;
                path.to_path_buf()
            } else {
                dst
            };
            let reserved = if opts.reserve > 0 {
                format!(" (+{} reserved)", opts.reserve)
            } else {
                String::new()
            };
            outln!(
                "rewrote {}: {head_len} head bytes{reserved}, copied {} of tensor data",
                final_path.display(),
                fmt_size(copied)
            );
            Ok(0)
        }
    }
}

fn edit_command(file: &str, ops: &[EditOp], opts: &WriteOpts) -> CmdResult {
    let path = Path::new(file);
    let mut g = load(path)?;
    let summary = patch::apply_ops(&mut g.kvs, ops).map_err(|e| failure(e.to_string()))?;
    for line in &summary.lines {
        outln!("{line}");
    }
    commit(path, &g, opts)
}

fn cmd_show(args: &[String]) -> CmdResult {
    let mut tensors = false;
    let Some(pos) = parse_args(args, SHOW_HELP, None, &mut [("--tensors", &mut tensors)])? else {
        return Ok(0);
    };
    let [file] = &pos[..] else {
        return Err(usage("show takes exactly one FILE (see 'show --help')"));
    };
    let g = load(Path::new(file))?;
    out!("{}", render_show(&g, file, tensors));
    Ok(0)
}

fn cmd_get(args: &[String]) -> CmdResult {
    let mut json = false;
    let mut raw = false;
    let Some(pos) = parse_args(
        args,
        GET_HELP,
        None,
        &mut [("--json", &mut json), ("--raw", &mut raw)],
    )?
    else {
        return Ok(0);
    };
    let [file, key] = &pos[..] else {
        return Err(usage("get takes FILE and KEY (see 'get --help')"));
    };
    let g = load(Path::new(file))?;
    let Some(value) = g.get(key) else {
        return Err(failure(format!(
            "no such key '{key}' (run 'gguf-chisel show {file}' to list keys)"
        )));
    };
    if json {
        let doc = crate::json::Json::Obj(vec![
            ("key".into(), crate::json::Json::Str(key.clone())),
            ("type".into(), crate::json::Json::Str(value.type_label())),
            ("value".into(), crate::dump::value_json(value)),
        ]);
        outln!("{}", crate::json::encode(&doc));
        return Ok(0);
    }
    match value {
        GgufValue::Str(s) if raw => {
            out!("{s}");
        }
        GgufValue::Str(s) => outln!("{s}"),
        GgufValue::Array(_, _) => {
            outln!("{}", crate::json::encode(&crate::dump::value_json(value)))
        }
        scalar => outln!("{}", scalar.preview(usize::MAX)),
    }
    Ok(0)
}

fn cmd_set(args: &[String]) -> CmdResult {
    let mut opts = WriteOpts::default();
    let Some(pos) = parse_args(args, SET_HELP, Some(&mut opts), &mut [])? else {
        return Ok(0);
    };
    let [file, pairs @ ..] = &pos[..] else {
        return Err(usage("set takes FILE and at least one KEY=VALUE"));
    };
    if pairs.is_empty() {
        return Err(usage("set takes FILE and at least one KEY=VALUE"));
    }
    let ops = pairs
        .iter()
        .map(|p| patch::parse_set_arg(p))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| usage(e.to_string()))?;
    edit_command(file, &ops, &opts)
}

fn cmd_rm(args: &[String]) -> CmdResult {
    let mut opts = WriteOpts::default();
    let Some(pos) = parse_args(args, RM_HELP, Some(&mut opts), &mut [])? else {
        return Ok(0);
    };
    let [file, keys @ ..] = &pos[..] else {
        return Err(usage("rm takes FILE and at least one KEY"));
    };
    if keys.is_empty() {
        return Err(usage("rm takes FILE and at least one KEY"));
    }
    let ops: Vec<EditOp> = keys
        .iter()
        .map(|k| EditOp::Remove { key: k.clone() })
        .collect();
    edit_command(file, &ops, &opts)
}

fn cmd_rename(args: &[String]) -> CmdResult {
    let mut opts = WriteOpts::default();
    let Some(pos) = parse_args(args, RENAME_HELP, Some(&mut opts), &mut [])? else {
        return Ok(0);
    };
    let [file, from, to] = &pos[..] else {
        return Err(usage("rename takes FILE, OLD_KEY and NEW_KEY"));
    };
    let ops = [EditOp::Rename {
        from: from.clone(),
        to: to.clone(),
    }];
    edit_command(file, &ops, &opts)
}

fn cmd_dump(args: &[String]) -> CmdResult {
    let Some(pos) = parse_args(args, DUMP_HELP, None, &mut [])? else {
        return Ok(0);
    };
    let [file] = &pos[..] else {
        return Err(usage("dump takes exactly one FILE"));
    };
    let g = load(Path::new(file))?;
    outln!("{}", dump_json(&g));
    Ok(0)
}

fn cmd_apply(args: &[String]) -> CmdResult {
    let mut opts = WriteOpts::default();
    let Some(pos) = parse_args(args, APPLY_HELP, Some(&mut opts), &mut [])? else {
        return Ok(0);
    };
    let [file, patch_src] = &pos[..] else {
        return Err(usage("apply takes FILE and PATCH.json (or '-' for stdin)"));
    };
    let text = if patch_src == "-" {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .map_err(|e| failure(format!("cannot read stdin: {e}")))?;
        buf
    } else {
        std::fs::read_to_string(patch_src)
            .map_err(|e| failure(format!("cannot read {patch_src}: {e}")))?
    };
    let doc = crate::json::parse(&text).map_err(|e| failure(format!("{patch_src}: {e}")))?;
    let ops = patch::ops_from_json(&doc).map_err(|e| failure(format!("{patch_src}: {e}")))?;
    edit_command(file, &ops, &opts)
}

fn print_lint(source: &str, issues: &[template::Issue], strict: bool) -> i32 {
    let errors = template::error_count(issues);
    let warnings = issues.len() - errors;
    if errors == 0 && warnings == 0 {
        outln!("{source}: template OK");
    } else if errors == 0 {
        outln!("{source}: template OK ({})", plural(warnings, "warning"));
    } else {
        outln!(
            "{source}: {}, {}",
            plural(errors, "error"),
            plural(warnings, "warning")
        );
    }
    for issue in issues {
        let label = match issue.severity {
            template::Severity::Error => "error",
            template::Severity::Warning => "warning",
        };
        match issue.pos {
            Some((l, c)) => outln!("  {label} at {l}:{c}: {}", issue.message),
            None => outln!("  {label}: {}", issue.message),
        }
    }
    if errors > 0 || (strict && warnings > 0) {
        1
    } else {
        0
    }
}

fn template_source(g: &Gguf, file: &str) -> Result<String, CliError> {
    match g.get(CHAT_TEMPLATE_KEY) {
        Some(GgufValue::Str(s)) => Ok(s.clone()),
        Some(v) => Err(failure(format!(
            "{CHAT_TEMPLATE_KEY} is {} rather than a string",
            v.type_label()
        ))),
        None => Err(failure(format!(
            "{file} has no {CHAT_TEMPLATE_KEY}; set one with 'template set'"
        ))),
    }
}

fn cmd_template(args: &[String]) -> CmdResult {
    let Some(sub) = args.first() else {
        return Err(usage(
            "template needs a subcommand: show, set, check or presets",
        ));
    };
    let rest = &args[1..];
    match sub.as_str() {
        "presets" => {
            for (name, desc) in template::preset_names() {
                outln!("{name:<10} {desc}");
            }
            Ok(0)
        }
        "show" => {
            let Some(pos) = parse_args(rest, TEMPLATE_HELP, None, &mut [])? else {
                return Ok(0);
            };
            let [file] = &pos[..] else {
                return Err(usage("template show takes exactly one FILE"));
            };
            let g = load(Path::new(file))?;
            outln!("{}", template_source(&g, file)?);
            Ok(0)
        }
        "check" => {
            let mut strict = false;
            // --file is handled as a positional-style option here.
            let mut tpl_file: Option<String> = None;
            let mut pos = Vec::new();
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--strict" => strict = true,
                    "--file" => {
                        i += 1;
                        tpl_file = Some(
                            rest.get(i)
                                .ok_or_else(|| usage("--file needs a path"))?
                                .clone(),
                        );
                    }
                    "-h" | "--help" => {
                        outln!("{}", TEMPLATE_HELP.replace("{WRITE_OPTS}", WRITE_OPTS));
                        return Ok(0);
                    }
                    other if other.starts_with('-') => {
                        return Err(usage(format!("unknown option '{other}'")));
                    }
                    other => pos.push(other.to_string()),
                }
                i += 1;
            }
            let (source, text) = match (tpl_file, &pos[..]) {
                (Some(path), []) => {
                    let text = std::fs::read_to_string(&path)
                        .map_err(|e| failure(format!("cannot read {path}: {e}")))?;
                    (path, text)
                }
                (None, [file]) => {
                    let g = load(Path::new(file))?;
                    (
                        format!("{file}:{CHAT_TEMPLATE_KEY}"),
                        template_source(&g, file)?,
                    )
                }
                _ => return Err(usage("template check takes FILE or --file TPL")),
            };
            let issues = template::lint(&text);
            Ok(print_lint(&source, &issues, strict))
        }
        "set" => {
            let mut opts = WriteOpts::default();
            let mut preset_name: Option<String> = None;
            let mut tpl_file: Option<String> = None;
            let mut pos = Vec::new();
            let mut i = 0;
            while i < rest.len() {
                match rest[i].as_str() {
                    "--preset" => {
                        i += 1;
                        preset_name = Some(
                            rest.get(i)
                                .ok_or_else(|| usage("--preset needs a name"))?
                                .clone(),
                        );
                    }
                    "--file" => {
                        i += 1;
                        tpl_file = Some(
                            rest.get(i)
                                .ok_or_else(|| usage("--file needs a path"))?
                                .clone(),
                        );
                    }
                    "--dry-run" => opts.dry_run = true,
                    "--rewrite" => opts.rewrite = true,
                    "-o" | "--output" => {
                        i += 1;
                        opts.output = Some(PathBuf::from(
                            rest.get(i).ok_or_else(|| usage("-o needs a path"))?,
                        ));
                    }
                    "--reserve" => {
                        i += 1;
                        let v = rest.get(i).ok_or_else(|| usage("--reserve needs a size"))?;
                        opts.reserve = patch::parse_size(v).map_err(|e| usage(e.to_string()))?;
                    }
                    "-h" | "--help" => {
                        outln!("{}", TEMPLATE_HELP.replace("{WRITE_OPTS}", WRITE_OPTS));
                        return Ok(0);
                    }
                    other if other.starts_with('-') => {
                        return Err(usage(format!("unknown option '{other}'")));
                    }
                    other => pos.push(other.to_string()),
                }
                i += 1;
            }
            let [file] = &pos[..] else {
                return Err(usage("template set takes FILE plus --preset or --file"));
            };
            let text = match (preset_name, tpl_file) {
                (Some(name), None) => template::preset(&name)
                    .ok_or_else(|| {
                        failure(format!(
                            "unknown preset '{name}' (run 'gguf-chisel template presets')"
                        ))
                    })?
                    .to_string(),
                (None, Some(path)) => std::fs::read_to_string(&path)
                    .map_err(|e| failure(format!("cannot read {path}: {e}")))?,
                _ => {
                    return Err(usage(
                        "template set needs exactly one of --preset or --file",
                    ))
                }
            };
            let issues = template::lint(&text);
            if template::error_count(&issues) > 0 {
                print_lint("new template", &issues, false);
                return Err(failure(
                    "refusing to install a template with lint errors (fix it, or check \
                     with 'template check --file')",
                ));
            }
            let ops = [EditOp::Set {
                key: CHAT_TEMPLATE_KEY.into(),
                spec: patch::ValueSpec::Typed(GgufValue::Str(text)),
            }];
            edit_command(file, &ops, &opts)
        }
        other => Err(usage(format!(
            "unknown template subcommand '{other}' (show, set, check, presets)"
        ))),
    }
}

fn cmd_verify(args: &[String]) -> CmdResult {
    let Some(pos) = parse_args(args, VERIFY_HELP, None, &mut [])? else {
        return Ok(0);
    };
    let [file] = &pos[..] else {
        return Err(usage("verify takes exactly one FILE"));
    };
    let g = load(Path::new(file))?;
    let report = verify(&g);
    if report.ok() {
        match report.warnings.len() {
            0 => outln!("{file}: OK"),
            n => outln!("{file}: OK ({})", plural(n, "warning")),
        }
    } else {
        outln!(
            "{file}: {}, {}",
            plural(report.errors.len(), "error"),
            plural(report.warnings.len(), "warning")
        );
    }
    outln!(
        "  gguf v{}, {} metadata {}, {}, data section {}",
        g.version,
        g.kvs.len(),
        if g.kvs.len() == 1 { "key" } else { "keys" },
        plural(g.tensors.len(), "tensor"),
        fmt_size(g.data_len())
    );
    for e in &report.errors {
        outln!("  error: {e}");
    }
    for w in &report.warnings {
        outln!("  warning: {w}");
    }
    Ok(if report.ok() { 0 } else { 1 })
}

fn cmd_sample(args: &[String]) -> CmdResult {
    let mut opts = WriteOpts::default();
    let Some(pos) = parse_args(args, SAMPLE_HELP, Some(&mut opts), &mut [])? else {
        return Ok(0);
    };
    let [file] = &pos[..] else {
        return Err(usage("sample takes exactly one FILE"));
    };
    if opts.dry_run || opts.rewrite || opts.output.is_some() {
        return Err(usage("sample only accepts --reserve (see 'sample --help')"));
    }
    let bytes = writer::sample_bytes(opts.reserve);
    std::fs::write(file, &bytes).map_err(|e| failure(format!("cannot write {file}: {e}")))?;
    outln!(
        "wrote sample GGUF: {file} ({} bytes, 2 tensors, 9 metadata keys)",
        bytes.len()
    );
    Ok(0)
}

/// Entry point used by `main`: dispatch and translate errors to exit codes.
pub fn run(args: Vec<String>) -> i32 {
    let Some(command) = args.first() else {
        outln!("{HELP}");
        return 0;
    };
    let rest = &args[1..];
    let result = match command.as_str() {
        "-h" | "--help" | "help" => {
            outln!("{HELP}");
            Ok(0)
        }
        "-V" | "--version" => {
            outln!("gguf-chisel {VERSION}");
            Ok(0)
        }
        "show" => cmd_show(rest),
        "get" => cmd_get(rest),
        "set" => cmd_set(rest),
        "rm" => cmd_rm(rest),
        "rename" => cmd_rename(rest),
        "dump" => cmd_dump(rest),
        "apply" => cmd_apply(rest),
        "template" => cmd_template(rest),
        "verify" => cmd_verify(rest),
        "sample" => cmd_sample(rest),
        other => Err(usage(format!("unknown command '{other}' (see --help)"))),
    };
    match result {
        Ok(code) => code,
        Err(e) => {
            eprintln!("gguf-chisel: {}", e.message);
            e.code
        }
    }
}
