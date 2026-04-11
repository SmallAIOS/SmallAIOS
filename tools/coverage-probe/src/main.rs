// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CLI entry point for the SmallAIOS ONNX coverage probe.

use smallaios_coverage_probe::{walk_bytes, Inventory, Verdict};
use std::path::PathBuf;
use std::process::ExitCode;

#[derive(Debug, Default)]
struct Args {
    model: Option<PathBuf>,
    inventory: Option<PathBuf>,
    json: bool,
    detailed: bool,
    quiet: bool,
    strict: bool,
    show_help: bool,
}

fn parse_args(argv: Vec<String>) -> Result<Args, String> {
    let mut args = Args::default();
    let mut iter = argv.into_iter().skip(1);
    while let Some(a) = iter.next() {
        match a.as_str() {
            "--json" => args.json = true,
            "--detailed" => args.detailed = true,
            "--quiet" => args.quiet = true,
            "--strict" => args.strict = true,
            "-h" | "--help" => args.show_help = true,
            "--inventory" => {
                let v = iter
                    .next()
                    .ok_or_else(|| "--inventory requires a path argument".to_string())?;
                args.inventory = Some(PathBuf::from(v));
            }
            s if s.starts_with("--") => {
                return Err(format!("unknown flag: {s}"));
            }
            _ => {
                if args.model.is_some() {
                    return Err(format!("unexpected positional argument: {a}"));
                }
                args.model = Some(PathBuf::from(a));
            }
        }
    }
    Ok(args)
}

const HELP: &str = "\
smallaios-coverage-probe — ONNX model operator coverage probe

USAGE:
    smallaios-coverage-probe [FLAGS] <model.onnx>

FLAGS:
    --json                 Emit JSON instead of Markdown/text
    --detailed             Show per-node attribute concerns
    --quiet                Only summary, no per-op breakdown
    --strict               Exit nonzero if any op is unrecognized or
                           otherwise unsupported
    --inventory <path>     Override the roadmap file location
                           (default: docs/onnx-coverage-roadmap.md)
    -h, --help             Print this help

EXIT CODES:
    0   success
    1   CLI or IO error
    2   --strict and at least one unrecognized or unsupported op
";

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse_args(argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}\n");
            eprintln!("{HELP}");
            return ExitCode::from(1);
        }
    };
    if args.show_help {
        println!("{HELP}");
        return ExitCode::SUCCESS;
    }
    let model_path = match &args.model {
        Some(p) => p.clone(),
        None => {
            eprintln!("error: missing model path\n\n{HELP}");
            return ExitCode::from(1);
        }
    };

    // Locate inventory.
    let inventory_path = args
        .inventory
        .clone()
        .unwrap_or_else(default_inventory_path);
    let inventory = match Inventory::from_roadmap_file(&inventory_path) {
        Ok(inv) => inv,
        Err(e) => {
            eprintln!(
                "warning: could not read inventory at {}: {e}. \
                 Falling back to registry-only mode.",
                inventory_path.display()
            );
            Inventory::from_roadmap_str("")
        }
    };

    // Read model bytes.
    let bytes = match std::fs::read(&model_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("error: could not read model {}: {e}", model_path.display());
            return ExitCode::from(1);
        }
    };
    let label = model_path.display().to_string();
    let report = match walk_bytes(&bytes, &label, &inventory) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(1);
        }
    };

    if args.json {
        #[cfg(feature = "json")]
        {
            println!("{}", report.to_json());
        }
        #[cfg(not(feature = "json"))]
        {
            eprintln!("error: --json requires building with the `json` feature");
            return ExitCode::from(1);
        }
    } else if args.quiet || !args.detailed {
        print!("{}", report.to_text_summary(args.quiet));
        if args.detailed {
            // Detailed concerns: emit after summary.
            let concerns: Vec<&String> = report
                .per_op
                .iter()
                .flat_map(|op| op.attribute_concerns.iter())
                .collect();
            if !concerns.is_empty() {
                println!("\nAttribute concerns:");
                for c in concerns {
                    println!("  - {c}");
                }
            }
        }
    } else {
        // Full Markdown output.
        print!("{}", report.to_markdown(args.detailed));
    }

    if args.strict {
        match report.verdict {
            Verdict::LoadableToday => ExitCode::SUCCESS,
            Verdict::HasUnrecognizedOps | Verdict::NotLoadable => ExitCode::from(2),
            _ => ExitCode::from(2),
        }
    } else {
        ExitCode::SUCCESS
    }
}

fn default_inventory_path() -> PathBuf {
    // Walk upwards from CWD looking for docs/onnx-coverage-roadmap.md.
    if let Ok(cwd) = std::env::current_dir() {
        let mut here = cwd.as_path();
        loop {
            let candidate = here.join("docs").join("onnx-coverage-roadmap.md");
            if candidate.exists() {
                return candidate;
            }
            match here.parent() {
                Some(p) => here = p,
                None => break,
            }
        }
    }
    PathBuf::from("docs/onnx-coverage-roadmap.md")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_args() {
        let a = parse_args(vec!["probe".into(), "model.onnx".into()]).unwrap();
        assert_eq!(a.model.as_deref(), Some(std::path::Path::new("model.onnx")));
        assert!(!a.json);
        assert!(!a.strict);
    }

    #[test]
    fn parses_flags() {
        let a = parse_args(vec![
            "probe".into(),
            "--json".into(),
            "--detailed".into(),
            "--strict".into(),
            "m.onnx".into(),
        ])
        .unwrap();
        assert!(a.json);
        assert!(a.detailed);
        assert!(a.strict);
    }

    #[test]
    fn rejects_unknown_flag() {
        let e = parse_args(vec!["probe".into(), "--frobnicate".into()]).unwrap_err();
        assert!(e.contains("unknown flag"));
    }

    #[test]
    fn inventory_override_requires_value() {
        let e = parse_args(vec!["probe".into(), "--inventory".into()]).unwrap_err();
        assert!(e.contains("--inventory"));
    }
}
