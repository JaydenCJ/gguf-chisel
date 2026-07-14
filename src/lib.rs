//! gguf-chisel — surgical metadata editing for GGUF model files.
//!
//! The crate is a set of small, pure modules glued together by a thin CLI:
//!
//! - [`types`] — the GGUF data model (value types, tensors, ggml type table)
//! - [`reader`] — streaming parser for the file head (never reads tensor data)
//! - [`writer`] — head serializer, in-place fit planner, rewriter, sample builder
//! - [`patch`] — edit operations: `set` / `rm` / `rename` and value coercion
//! - [`template`] — chat-template presets and the Jinja-subset linter
//! - [`json`] — dependency-free JSON parser and encoder for `dump` / `apply`
//! - [`verify`] — structural verification of offsets, alignment and keys
//! - [`cli`] — argument parsing and command dispatch

pub mod cli;
pub mod dump;
pub mod json;
pub mod patch;
pub mod reader;
pub mod template;
pub mod types;
pub mod verify;
pub mod writer;
