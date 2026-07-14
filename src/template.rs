//! Chat-template support: a small library of well-known template presets and
//! a linter for the Jinja subset that chat templates actually use. The linter
//! is not a Jinja engine — it checks the things that break model runtimes in
//! practice: unbalanced `{{ }}` / `{% %}` delimiters, mismatched or unclosed
//! block statements, and templates that forget the `messages` loop.

/// A named preset: `(name, one-line description, template source)`.
/// These are the community-standard variants of each format; always compare
/// against your model card before shipping a re-templated file.
const PRESETS: &[(&str, &str, &str)] = &[
    (
        "chatml",
        "ChatML <|im_start|> markers (Qwen and many fine-tunes)",
        "{% for message in messages %}{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}",
    ),
    (
        "llama2",
        "[INST] blocks with a <<SYS>> system section",
        "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'system' %}{{ '<<SYS>>\\n' + message['content'] + '\\n<</SYS>>\\n\\n' }}{% elif message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'] + ' ' + eos_token }}{% endif %}{% endfor %}",
    ),
    (
        "llama3",
        "<|start_header_id|> framing with <|eot_id|> turn ends",
        "{{ bos_token }}{% for message in messages %}{{ '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] + '<|eot_id|>' }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{% endif %}",
    ),
    (
        "mistral",
        "[INST] blocks, no system section",
        "{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token }}{% endif %}{% endfor %}",
    ),
    (
        "zephyr",
        "<|role|> markers with eos after every turn",
        "{% for message in messages %}{{ '<|' + message['role'] + '|>\\n' + message['content'] + eos_token + '\\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}",
    ),
    (
        "alpaca",
        "### Instruction / ### Response sections",
        "{% for message in messages %}{% if message['role'] == 'system' %}{{ message['content'] + '\\n\\n' }}{% elif message['role'] == 'user' %}{{ '### Instruction:\\n' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ '### Response:\\n' + message['content'] + '\\n\\n' }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '### Response:\\n' }}{% endif %}",
    ),
];

/// Template source for a preset name.
pub fn preset(name: &str) -> Option<&'static str> {
    PRESETS
        .iter()
        .find(|(n, _, _)| *n == name)
        .map(|(_, _, src)| *src)
}

/// All preset names with their one-line descriptions.
pub fn preset_names() -> Vec<(&'static str, &'static str)> {
    PRESETS.iter().map(|(n, d, _)| (*n, *d)).collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
}

/// One lint finding. `pos` is the 1-based (line, column) of the offending
/// token, or `None` for whole-template findings.
#[derive(Debug)]
pub struct Issue {
    pub severity: Severity,
    pub pos: Option<(usize, usize)>,
    pub message: String,
}

/// Count of error-severity issues.
pub fn error_count(issues: &[Issue]) -> usize {
    issues
        .iter()
        .filter(|i| i.severity == Severity::Error)
        .count()
}

/// Statement tags that open a block, with their closers.
const BLOCK_TAGS: &[(&str, &str)] = &[
    ("for", "endfor"),
    ("if", "endif"),
    ("macro", "endmacro"),
    ("filter", "endfilter"),
    ("block", "endblock"),
    ("generation", "endgeneration"),
    ("set", "endset"), // block form only; inline `set x = ...` has no closer
];

/// Tags that are complete on their own.
const INLINE_TAGS: &[&str] = &["include", "import", "from", "do", "break", "continue"];

fn line_col(src: &str, byte: usize) -> (usize, usize) {
    let mut line = 1;
    let mut col = 1;
    for (i, c) in src.char_indices() {
        if i >= byte {
            break;
        }
        if c == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Scan from `start` (just past an opener) to the matching `closer`,
/// skipping over quoted strings — chat templates routinely embed `{` and `}`
/// inside string literals (tool-call JSON, for one). Returns the byte index
/// *after* the closer, or `None` at end of input.
fn scan_to_closer(bytes: &[u8], start: usize, closer: &[u8]) -> Option<usize> {
    let mut i = start;
    while i < bytes.len() {
        match bytes[i] {
            b'\'' | b'"' => {
                let quote = bytes[i];
                i += 1;
                while i < bytes.len() && bytes[i] != quote {
                    if bytes[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                i += 1;
            }
            _ if bytes[i..].starts_with(closer) => return Some(i + closer.len()),
            _ => i += 1,
        }
    }
    None
}

/// The first identifier of a statement body, with `-`/`+` whitespace-control
/// markers stripped.
fn stmt_tag(body: &str) -> String {
    body.trim_start_matches(['-', '+'])
        .trim()
        .chars()
        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
        .collect()
}

/// Lint a chat template. Errors are structural problems that will break a
/// runtime's template engine; warnings are strong hints something is off.
pub fn lint(src: &str) -> Vec<Issue> {
    let mut issues = Vec::new();
    if src.trim().is_empty() {
        issues.push(Issue {
            severity: Severity::Error,
            pos: None,
            message: "template is empty".into(),
        });
        return issues;
    }

    let bytes = src.as_bytes();
    let mut stack: Vec<(String, usize)> = Vec::new(); // (tag, byte offset)
    let mut i = 0;

    while i < bytes.len() {
        if bytes[i..].starts_with(b"{{") {
            let open = i;
            match scan_to_closer(bytes, i + 2, b"}}") {
                Some(end) => {
                    let inner = &src[i + 2..end - 2];
                    if inner.contains("{{") || inner.contains("{%") {
                        let (l, c) = line_col(src, open);
                        issues.push(Issue {
                            severity: Severity::Error,
                            pos: Some((l, c)),
                            message: "'{{' opened again inside an expression".into(),
                        });
                    }
                    i = end;
                }
                None => {
                    let (l, c) = line_col(src, open);
                    issues.push(Issue {
                        severity: Severity::Error,
                        pos: Some((l, c)),
                        message: "unclosed '{{' expression".into(),
                    });
                    break;
                }
            }
        } else if bytes[i..].starts_with(b"{%") {
            let open = i;
            let Some(end) = scan_to_closer(bytes, i + 2, b"%}") else {
                let (l, c) = line_col(src, open);
                issues.push(Issue {
                    severity: Severity::Error,
                    pos: Some((l, c)),
                    message: "unclosed '{%' statement".into(),
                });
                break;
            };
            let body = &src[i + 2..end - 2];
            let tag = stmt_tag(body);
            let (l, c) = line_col(src, open);
            i = end;

            if tag == "raw" {
                // Skip everything until {% endraw %} without linting it.
                let mut j = i;
                let mut closed = false;
                while j < bytes.len() {
                    if bytes[j..].starts_with(b"{%") {
                        if let Some(e2) = scan_to_closer(bytes, j + 2, b"%}") {
                            if stmt_tag(&src[j + 2..e2 - 2]) == "endraw" {
                                i = e2;
                                closed = true;
                                break;
                            }
                        }
                        // Anything else inside raw is literal text; step past
                        // the '{%' so a later '{% endraw %}' is still found.
                        j += 2;
                        continue;
                    }
                    j += 1;
                }
                if !closed {
                    issues.push(Issue {
                        severity: Severity::Error,
                        pos: Some((l, c)),
                        message: "'{% raw %}' without '{% endraw %}'".into(),
                    });
                    break;
                }
                continue;
            }

            if tag == "elif" || tag == "else" {
                let ok = match stack.last() {
                    Some((open_tag, _)) if tag == "elif" => open_tag == "if",
                    Some((open_tag, _)) => open_tag == "if" || open_tag == "for",
                    None => false,
                };
                if !ok {
                    issues.push(Issue {
                        severity: Severity::Error,
                        pos: Some((l, c)),
                        message: format!(
                            "'{{% {tag} %}}' outside {}",
                            if tag == "elif" {
                                "'{% if %}'"
                            } else {
                                "'{% if %}'/'{% for %}'"
                            }
                        ),
                    });
                }
            } else if let Some((opener, _)) = BLOCK_TAGS.iter().find(|(_, e)| *e == tag) {
                match stack.pop() {
                    Some((open_tag, _)) if open_tag == *opener => {}
                    Some((open_tag, open_at)) => {
                        let (ol, oc) = line_col(src, open_at);
                        issues.push(Issue {
                            severity: Severity::Error,
                            pos: Some((l, c)),
                            message: format!(
                                "'{{% {tag} %}}' closes '{{% {open_tag} %}}' opened at {ol}:{oc}"
                            ),
                        });
                    }
                    None => {
                        issues.push(Issue {
                            severity: Severity::Error,
                            pos: Some((l, c)),
                            message: format!(
                                "'{{% {tag} %}}' without a matching '{{% {opener} %}}'"
                            ),
                        });
                    }
                }
            } else if BLOCK_TAGS.iter().any(|(o, _)| *o == tag) {
                let is_inline_set = tag == "set" && body.contains('=');
                if !is_inline_set {
                    stack.push((tag.clone(), open));
                }
            } else if tag.is_empty() {
                issues.push(Issue {
                    severity: Severity::Error,
                    pos: Some((l, c)),
                    message: "empty '{% %}' statement".into(),
                });
            } else if !INLINE_TAGS.contains(&tag.as_str()) {
                issues.push(Issue {
                    severity: Severity::Warning,
                    pos: Some((l, c)),
                    message: format!("unknown statement tag '{tag}'"),
                });
            }
        } else if bytes[i..].starts_with(b"{#") {
            let open = i;
            match scan_to_closer(bytes, i + 2, b"#}") {
                Some(end) => i = end,
                None => {
                    let (l, c) = line_col(src, open);
                    issues.push(Issue {
                        severity: Severity::Error,
                        pos: Some((l, c)),
                        message: "unclosed '{#' comment".into(),
                    });
                    break;
                }
            }
        } else if bytes[i..].starts_with(b"}}") || bytes[i..].starts_with(b"%}") {
            let (l, c) = line_col(src, i);
            issues.push(Issue {
                severity: Severity::Warning,
                pos: Some((l, c)),
                message: format!("stray '{}' outside any expression", &src[i..i + 2]),
            });
            i += 2;
        } else {
            i += 1;
        }
    }

    for (tag, at) in stack {
        let (l, c) = line_col(src, at);
        issues.push(Issue {
            severity: Severity::Error,
            pos: Some((l, c)),
            message: format!("unclosed '{{% {tag} %}}'"),
        });
    }

    if !src.contains("messages") {
        issues.push(Issue {
            severity: Severity::Warning,
            pos: None,
            message: "template never references 'messages' — runtimes pass the chat there".into(),
        });
    }
    if !src.contains("add_generation_prompt") {
        issues.push(Issue {
            severity: Severity::Warning,
            pos: None,
            message: "template never checks 'add_generation_prompt'; \
                      bare generation prompts may be impossible"
                .into(),
        });
    }

    issues
}

#[cfg(test)]
mod tests {
    use super::*;

    fn errors(src: &str) -> Vec<String> {
        lint(src)
            .into_iter()
            .filter(|i| i.severity == Severity::Error)
            .map(|i| i.message)
            .collect()
    }

    #[test]
    fn all_six_presets_lint_without_errors() {
        let names = preset_names();
        assert_eq!(names.len(), 6);
        for (name, _) in names {
            let src = preset(name).unwrap();
            let errs = errors(src);
            assert!(errs.is_empty(), "preset '{name}' has lint errors: {errs:?}");
        }

        assert_eq!(preset("vicuna"), None);
    }

    #[test]
    fn a_clean_template_produces_no_issues_at_all() {
        let src = "{% for message in messages %}{{ message['content'] }}{% endfor %}\
                   {% if add_generation_prompt %}{{ 'go:' }}{% endif %}";
        assert!(lint(src).is_empty(), "{:?}", lint(src));
    }

    #[test]
    fn unclosed_expression_reports_its_position() {
        let issues = lint("hello\n{{ messages[0] add_generation_prompt");
        let err = issues
            .iter()
            .find(|i| i.severity == Severity::Error)
            .unwrap();
        assert_eq!(err.pos, Some((2, 1)));
        assert!(err.message.contains("unclosed '{{'"), "{}", err.message);
    }

    #[test]
    fn mismatched_and_dangling_closers_are_errors() {
        let errs = errors(
            "{% for m in messages %}{% if x %}{% endfor %} add_generation_prompt{% endif %}",
        );
        assert!(
            errs.iter()
                .any(|e| e.contains("endfor") && e.contains("{% if %}")),
            "{errs:?}"
        );

        let errs = errors("messages add_generation_prompt {% endfor %}");
        assert!(
            errs.iter().any(|e| e.contains("without a matching")),
            "{errs:?}"
        );
    }

    #[test]
    fn unclosed_block_reports_the_opening_line() {
        let issues = lint("{% for m in messages %}{{ m }} add_generation_prompt");
        let err = issues
            .iter()
            .find(|i| i.severity == Severity::Error)
            .unwrap();
        assert!(
            err.message.contains("unclosed '{% for %}'"),
            "{}",
            err.message
        );
        assert_eq!(err.pos, Some((1, 1)));
    }

    #[test]
    fn elif_outside_if_is_an_error_but_else_in_for_is_fine() {
        let errs = errors("{% elif x %} messages add_generation_prompt");
        assert!(errs.iter().any(|e| e.contains("elif")), "{errs:?}");
        let ok = "{% for m in messages %}{{ m }}{% else %}none{% endfor %} add_generation_prompt";
        assert!(errors(ok).is_empty(), "{:?}", errors(ok));
    }

    #[test]
    fn braces_inside_string_literals_are_not_delimiters() {
        // Tool-call templates emit literal JSON — the '}}' inside the quoted
        // string must not terminate the expression early.
        let src =
            "{% for m in messages %}{{ '{\"end\": \"}}\"}' }}{% endfor %} add_generation_prompt";
        assert!(errors(src).is_empty(), "{:?}", errors(src));
    }

    #[test]
    fn raw_blocks_are_skipped_entirely() {
        let src = "{% raw %}{{ unbalanced {% nonsense {% endraw %}\
                   {% for m in messages %}{{ m }}{% endfor %} add_generation_prompt";
        assert!(errors(src).is_empty(), "{:?}", errors(src));
        let unclosed = "{% raw %}{{ oops";
        assert!(
            errors(unclosed).iter().any(|e| e.contains("endraw")),
            "{:?}",
            errors(unclosed)
        );
    }

    #[test]
    fn inline_set_needs_no_closer_but_block_set_does() {
        let inline = "{% set sep = '\\n' %} messages add_generation_prompt";
        assert!(errors(inline).is_empty(), "{:?}", errors(inline));
        let block = "{% set body %}x{% endset %} messages add_generation_prompt";
        assert!(errors(block).is_empty(), "{:?}", errors(block));
        let unclosed = "{% set body %}x messages add_generation_prompt";
        assert!(
            errors(unclosed).iter().any(|e| e.contains("set")),
            "{:?}",
            errors(unclosed)
        );
    }

    #[test]
    fn unknown_tags_warn_but_do_not_error() {
        let issues = lint("{% frobnicate %} messages add_generation_prompt");
        assert_eq!(error_count(&issues), 0);
        assert!(issues
            .iter()
            .any(|i| i.severity == Severity::Warning && i.message.contains("frobnicate")));
    }

    #[test]
    fn whole_template_findings_missing_vars_empty_input_stray_closers() {
        let issues = lint("{{ bos_token }}");
        assert_eq!(error_count(&issues), 0);
        let warnings: Vec<&str> = issues.iter().map(|i| i.message.as_str()).collect();
        assert!(
            warnings.iter().any(|w| w.contains("'messages'")),
            "{warnings:?}"
        );
        assert!(
            warnings.iter().any(|w| w.contains("add_generation_prompt")),
            "{warnings:?}"
        );

        let issues = lint("   ");
        assert_eq!(error_count(&issues), 1);
        assert!(issues[0].message.contains("empty"));

        let issues = lint("role }} messages add_generation_prompt");
        let w = issues.iter().find(|i| i.message.contains("stray")).unwrap();
        assert_eq!(w.severity, Severity::Warning);
        assert_eq!(w.pos, Some((1, 6)));
    }
}
