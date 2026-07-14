//! End-to-end tests that exercise the compiled `gguf-chisel` binary: show,
//! get, set, rm, rename, dump, apply, template, verify and sample, plus exit
//! codes and — the core promise — proof that in-place edits leave every
//! tensor byte and the file length untouched. Each test builds its own
//! fixture under a temporary directory: offline, deterministic, no shared
//! state.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_gguf-chisel")
}

fn run(args: &[&str]) -> Output {
    Command::new(bin())
        .args(args)
        .output()
        .expect("failed to run gguf-chisel binary")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn tempdir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gguf-chisel-cli-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// Create a sample model in `dir` via the real CLI and return its path.
fn sample_in(dir: &Path) -> PathBuf {
    let path = dir.join("model.gguf");
    let out = run(&["sample", path.to_str().unwrap()]);
    assert!(out.status.success(), "sample failed: {}", stderr(&out));
    path
}

/// The tensor-data region of a file, located by parsing the head with the
/// library (integration tests may use the crate directly).
fn data_region(path: &Path) -> (u64, Vec<u8>) {
    let bytes = fs::read(path).unwrap();
    let g = gguf_chisel::reader::parse_head(&bytes[..], bytes.len() as u64).unwrap();
    (g.data_start, bytes[g.data_start as usize..].to_vec())
}

#[test]
fn version_and_help_print_and_exit_zero() {
    let version = run(&["--version"]);
    assert!(version.status.success());
    assert_eq!(
        stdout(&version).trim(),
        format!("gguf-chisel {}", env!("CARGO_PKG_VERSION"))
    );

    let help = run(&["--help"]);
    assert!(help.status.success());
    let text = stdout(&help);
    assert!(text.contains("COMMANDS:"));
    assert!(text.contains("OPTIONS:"));
    for cmd in [
        "show", "get", "set", "rm", "rename", "dump", "apply", "template", "verify", "sample",
    ] {
        assert!(text.contains(cmd), "help must mention '{cmd}'");
    }
}

#[test]
fn unknown_commands_and_options_are_usage_errors_with_exit_2() {
    let cmd = run(&["frobnicate"]);
    assert_eq!(cmd.status.code(), Some(2));
    assert!(stderr(&cmd).contains("unknown command"));

    let dir = tempdir("badopt");
    let model = sample_in(&dir);
    let opt = run(&["show", model.to_str().unwrap(), "--frobnicate"]);
    assert_eq!(opt.status.code(), Some(2));
    assert!(stderr(&opt).contains("unknown option"));

    // Write flags that a command cannot honor are rejected, not ignored:
    // sample generates a fresh file, so only --reserve makes sense.
    let noop = run(&["sample", dir.join("x.gguf").to_str().unwrap(), "--dry-run"]);
    assert_eq!(noop.status.code(), Some(2));
    assert!(stderr(&noop).contains("--reserve"), "{}", stderr(&noop));
    assert!(
        !dir.join("x.gguf").exists(),
        "rejected sample writes nothing"
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn sample_then_show_reports_layout_and_keys() {
    let dir = tempdir("show");
    let model = sample_in(&dir);
    let out = run(&["show", model.to_str().unwrap(), "--tensors"]);
    assert!(out.status.success());
    let text = stdout(&out);
    assert!(text.contains("gguf:       v3 little-endian"), "{text}");
    assert!(text.contains("metadata:   9 keys"), "{text}");
    assert!(text.contains("sample.context_length"), "{text}");
    assert!(text.contains("token_embd.weight"), "{text}");
    assert!(text.contains("headroom"), "{text}");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn get_prints_bare_values_json_and_raw_bytes() {
    let dir = tempdir("get");
    let model = sample_in(&dir);
    let m = model.to_str().unwrap();

    let bare = run(&["get", m, "sample.context_length"]);
    assert_eq!(stdout(&bare).trim(), "4096");

    let json = run(&["get", m, "sample.context_length", "--json"]);
    let text = stdout(&json);
    assert!(text.contains("\"type\":\"u32\""), "{text}");
    assert!(text.contains("\"value\":4096"), "{text}");

    // --raw emits exact string bytes with no trailing newline, so templates
    // can be piped to a file and re-installed byte-identically.
    let raw = run(&["get", m, "tokenizer.chat_template", "--raw"]);
    let raw_bytes = raw.stdout.clone();
    assert!(!raw_bytes.ends_with(b"\n"));
    assert!(String::from_utf8(raw_bytes)
        .unwrap()
        .contains("<|im_start|>"));

    let missing = run(&["get", m, "no.such.key"]);
    assert_eq!(missing.status.code(), Some(1));
    assert!(
        stderr(&missing).contains("no such key"),
        "{}",
        stderr(&missing)
    );
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn set_patches_in_place_without_touching_tensor_bytes() {
    let dir = tempdir("set");
    let model = sample_in(&dir);
    let m = model.to_str().unwrap();
    let before_len = fs::metadata(&model).unwrap().len();
    let (data_start_before, data_before) = data_region(&model);

    let out = run(&[
        "set",
        m,
        "sample.context_length=32768",
        "general.name=Renamed Model",
    ]);
    assert!(out.status.success(), "{}", stderr(&out));
    let text = stdout(&out);
    assert!(
        text.contains("set sample.context_length: u32 4096 -> u32 32768"),
        "{text}"
    );
    assert!(text.contains("patched"), "{text}");
    assert!(text.contains("tensor data untouched"), "{text}");

    assert_eq!(
        fs::metadata(&model).unwrap().len(),
        before_len,
        "file size unchanged"
    );
    let (data_start_after, data_after) = data_region(&model);
    assert_eq!(data_start_before, data_start_after, "data offset unchanged");
    assert_eq!(data_before, data_after, "tensor bytes bit-identical");

    let get = run(&["get", m, "sample.context_length"]);
    assert_eq!(stdout(&get).trim(), "32768");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn growth_without_rewrite_fails_and_with_rewrite_succeeds() {
    let dir = tempdir("grow");
    let model = sample_in(&dir);
    let m = model.to_str().unwrap();
    let big = "x".repeat(2000);

    let refused = run(&["set", m, &format!("general.description={big}")]);
    assert_eq!(refused.status.code(), Some(1));
    assert!(
        stderr(&refused).contains("--rewrite"),
        "{}",
        stderr(&refused)
    );

    let (_, data_before) = data_region(&model);
    let ok = run(&[
        "set",
        m,
        &format!("general.description={big}"),
        "--rewrite",
        "--reserve",
        "1K",
    ]);
    assert!(ok.status.success(), "{}", stderr(&ok));
    assert!(stdout(&ok).contains("rewrote"), "{}", stdout(&ok));

    let (_, data_after) = data_region(&model);
    assert_eq!(
        data_before, data_after,
        "rewrite streams tensor bytes verbatim"
    );

    // The reserved headroom now absorbs further growth in place.
    let more = run(&["set", m, "general.license=apache-2.0"]);
    assert!(more.status.success(), "{}", stderr(&more));
    assert!(stdout(&more).contains("in place"), "{}", stdout(&more));

    let verify = run(&["verify", m]);
    assert!(verify.status.success(), "{}", stdout(&verify));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn dry_run_reports_fit_with_exit_0_and_rewrite_need_with_exit_3() {
    let dir = tempdir("dryrun");
    let model = sample_in(&dir);
    let m = model.to_str().unwrap();
    let original = fs::read(&model).unwrap();

    let fits = run(&["set", m, "sample.context_length=8192", "--dry-run"]);
    assert_eq!(fits.status.code(), Some(0));
    assert!(stdout(&fits).contains("fits in place"), "{}", stdout(&fits));

    let big = format!("general.notes={}", "y".repeat(4000));
    let needs = run(&["set", m, &big, "--dry-run"]);
    assert_eq!(needs.status.code(), Some(3));
    assert!(
        stdout(&needs).contains("needs a rewrite"),
        "{}",
        stdout(&needs)
    );

    assert_eq!(fs::read(&model).unwrap(), original, "dry-run never writes");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn rm_and_rename_edit_keys_in_place() {
    let dir = tempdir("rmrename");
    let model = sample_in(&dir);
    let m = model.to_str().unwrap();
    let (_, data_before) = data_region(&model);

    let rm = run(&["rm", m, "sample.block_count"]);
    assert!(rm.status.success(), "{}", stderr(&rm));
    assert!(
        stdout(&rm).contains("rm  sample.block_count"),
        "{}",
        stdout(&rm)
    );

    let rename = run(&["rename", m, "general.file_type", "general.quant_type"]);
    assert!(rename.status.success(), "{}", stderr(&rename));

    let show = stdout(&run(&["show", m]));
    assert!(!show.contains("sample.block_count"), "{show}");
    assert!(show.contains("general.quant_type"), "{show}");

    let (_, data_after) = data_region(&model);
    assert_eq!(data_before, data_after);

    let missing = run(&["rm", m, "not.a.key"]);
    assert_eq!(missing.status.code(), Some(1));
    assert!(stderr(&missing).contains("no such key"));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn output_flag_writes_a_new_file_and_leaves_the_source_alone() {
    let dir = tempdir("output");
    let model = sample_in(&dir);
    let m = model.to_str().unwrap();
    let copy = dir.join("patched.gguf");
    let original = fs::read(&model).unwrap();

    let out = run(&[
        "set",
        m,
        "sample.context_length=16384",
        "-o",
        copy.to_str().unwrap(),
        "--reserve",
        "512",
    ]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(
        fs::read(&model).unwrap(),
        original,
        "source untouched with -o"
    );

    let get = run(&["get", copy.to_str().unwrap(), "sample.context_length"]);
    assert_eq!(stdout(&get).trim(), "16384");
    let verify = run(&["verify", copy.to_str().unwrap()]);
    assert!(verify.status.success(), "{}", stdout(&verify));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn dump_emits_json_the_shipped_parser_accepts() {
    let dir = tempdir("dump");
    let model = sample_in(&dir);
    let out = run(&["dump", model.to_str().unwrap()]);
    assert!(out.status.success());
    let doc = gguf_chisel::json::parse(&stdout(&out)).expect("dump is valid JSON");
    assert_eq!(
        doc.get("gguf_version"),
        Some(&gguf_chisel::json::Json::Int(3))
    );
    let tensors = doc.get("tensors").unwrap();
    let gguf_chisel::json::Json::Arr(items) = tensors else {
        panic!()
    };
    assert_eq!(items.len(), 2);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn apply_runs_a_json_patch_document_from_file_and_stdin() {
    let dir = tempdir("apply");
    let model = sample_in(&dir);
    let m = model.to_str().unwrap();
    let patch = dir.join("patch.json");
    fs::write(
        &patch,
        r#"{
  "delete": ["sample.block_count"],
  "rename": {"general.file_type": "general.quant_type"},
  "set": {
    "sample.context_length": 131072,
    "general.tag": {"type": "str", "value": "v2-fixed"}
  }
}"#,
    )
    .unwrap();

    let out = run(&["apply", m, patch.to_str().unwrap()]);
    assert!(out.status.success(), "{}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("rm  sample.block_count"), "{text}");
    assert!(
        text.contains("rename general.file_type -> general.quant_type"),
        "{text}"
    );
    assert!(
        text.contains("set sample.context_length: u32 4096 -> u32 131072"),
        "{text}"
    );

    // Same document via stdin; the delete now fails (already deleted), and
    // atomicity means *nothing* else is applied either.
    let mut child = Command::new(bin())
        .args(["apply", m, "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    use std::io::Write as _;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(fs::read(&patch).unwrap().as_slice())
        .unwrap();
    let rerun = child.wait_with_output().unwrap();
    assert_eq!(rerun.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&rerun.stderr).contains("no such key"));
    let get = run(&["get", m, "general.tag"]);
    assert_eq!(stdout(&get).trim(), "v2-fixed", "first apply persisted");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn template_set_show_roundtrip_with_a_preset() {
    let dir = tempdir("template");
    let model = sample_in(&dir);
    let m = model.to_str().unwrap();

    let presets = run(&["template", "presets"]);
    assert!(presets.status.success());
    for name in ["chatml", "llama3", "mistral", "zephyr", "alpaca"] {
        assert!(stdout(&presets).contains(name), "{}", stdout(&presets));
    }

    // llama3 is longer than the sample's chatml: needs --rewrite once.
    let refused = run(&["template", "set", m, "--preset", "llama3"]);
    assert_eq!(refused.status.code(), Some(1));
    let ok = run(&[
        "template",
        "set",
        m,
        "--preset",
        "llama3",
        "--rewrite",
        "--reserve",
        "2K",
    ]);
    assert!(ok.status.success(), "{}", stderr(&ok));

    let show = run(&["template", "show", m]);
    assert!(
        stdout(&show).contains("<|start_header_id|>"),
        "{}",
        stdout(&show)
    );

    let unknown = run(&["template", "set", m, "--preset", "vicuna"]);
    assert_eq!(unknown.status.code(), Some(1));
    assert!(stderr(&unknown).contains("unknown preset"));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn template_check_lints_files_and_refuses_broken_installs() {
    let dir = tempdir("lint");
    let model = sample_in(&dir);
    let m = model.to_str().unwrap();

    let good = run(&["template", "check", m]);
    assert!(good.status.success(), "{}", stdout(&good));
    assert!(stdout(&good).contains("template OK"), "{}", stdout(&good));

    let broken = dir.join("broken.jinja");
    fs::write(
        &broken,
        "{% for m in messages %}{{ m.content }} add_generation_prompt",
    )
    .unwrap();
    let check = run(&["template", "check", "--file", broken.to_str().unwrap()]);
    assert_eq!(check.status.code(), Some(1));
    assert!(
        stdout(&check).contains("unclosed '{% for %}'"),
        "{}",
        stdout(&check)
    );

    // Installing a broken template must be refused before any write.
    let before = fs::read(&model).unwrap();
    let install = run(&[
        "template",
        "set",
        m,
        "--file",
        broken.to_str().unwrap(),
        "--rewrite",
    ]);
    assert_eq!(install.status.code(), Some(1));
    assert!(
        stderr(&install).contains("lint errors"),
        "{}",
        stderr(&install)
    );
    assert_eq!(fs::read(&model).unwrap(), before);

    // --strict turns warnings into failures.
    let warny = dir.join("warny.jinja");
    fs::write(&warny, "{{ bos_token }} messages").unwrap();
    let lax = run(&["template", "check", "--file", warny.to_str().unwrap()]);
    assert!(lax.status.success());
    let strict = run(&[
        "template",
        "check",
        "--file",
        warny.to_str().unwrap(),
        "--strict",
    ]);
    assert_eq!(strict.status.code(), Some(1));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn verify_passes_the_sample_and_fails_corrupted_files() {
    let dir = tempdir("verify");
    let model = sample_in(&dir);
    let m = model.to_str().unwrap();

    let ok = run(&["verify", m]);
    assert!(ok.status.success());
    assert!(stdout(&ok).contains(": OK"), "{}", stdout(&ok));
    assert!(stdout(&ok).contains("2 tensors"), "{}", stdout(&ok));

    // Corrupt the magic: not even parseable.
    let mut bytes = fs::read(&model).unwrap();
    bytes[0] = b'X';
    let bad = dir.join("bad.gguf");
    fs::write(&bad, &bytes).unwrap();
    let broken = run(&["verify", bad.to_str().unwrap()]);
    assert_eq!(broken.status.code(), Some(1));
    assert!(
        stderr(&broken).contains("not a GGUF file"),
        "{}",
        stderr(&broken)
    );

    // Truncate the data section: parses, but verification must fail.
    let cut = dir.join("cut.gguf");
    let original = fs::read(&model).unwrap();
    fs::write(&cut, &original[..original.len() - 100]).unwrap();
    let short = run(&["verify", cut.to_str().unwrap()]);
    assert_eq!(short.status.code(), Some(1));
    assert!(stdout(&short).contains("error:"), "{}", stdout(&short));
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn protected_alignment_key_is_refused_with_a_clear_reason() {
    let dir = tempdir("protected");
    let model = sample_in(&dir);
    let m = model.to_str().unwrap();
    let before = fs::read(&model).unwrap();

    let out = run(&["set", m, "general.alignment=64"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).contains("alignment"), "{}", stderr(&out));
    assert_eq!(
        fs::read(&model).unwrap(),
        before,
        "refused edits write nothing"
    );
    fs::remove_dir_all(&dir).unwrap();
}
