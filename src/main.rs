//! Binary entry point: hand everything to the CLI module.

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(gguf_chisel::cli::run(args));
}
