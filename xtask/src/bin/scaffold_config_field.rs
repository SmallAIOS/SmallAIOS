// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! `cargo xtask scaffold-config-field <path> --type <ft> --reload <rk>`
//! — print the boilerplate diff a developer copy-pastes to add a new
//! [`mgmt::Config`] field.
//!
//! Spec context:
//! `openspec/changes/management-login-v1/specs/mgmt-config-surface-trait/spec.md`
//! Q3 — "developer-ergonomics scaffolder".
//!
//! v1 prints the diff; auto-application is a future polish — once the
//! universal-exposure walker is hooked up to `cargo build`, the
//! scaffolder will become responsible for keeping the registry in
//! sync, but for now this is a copy-paste helper.

use std::process::ExitCode;

use xtask::scaffold::{render_diff, FieldType, ReloadKind, ScaffoldArgs};

fn print_usage() {
    eprintln!(
        "usage: cargo xtask scaffold-config-field <path> --type <bool|u32|u64|string|password_classes> --reload <live|boot>"
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "--help" || args[0] == "-h" {
        print_usage();
        return ExitCode::SUCCESS;
    }

    let path = args[0].clone();
    let mut kind: Option<FieldType> = None;
    let mut reload: Option<ReloadKind> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--type" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --type requires a value");
                    print_usage();
                    return ExitCode::FAILURE;
                }
                match FieldType::parse(&args[i]) {
                    Ok(k) => kind = Some(k),
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            "--reload" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --reload requires a value");
                    print_usage();
                    return ExitCode::FAILURE;
                }
                match ReloadKind::parse(&args[i]) {
                    Ok(r) => reload = Some(r),
                    Err(e) => {
                        eprintln!("error: {e}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            other => {
                eprintln!("error: unknown argument '{other}'");
                print_usage();
                return ExitCode::FAILURE;
            }
        }
        i += 1;
    }

    let kind = match kind {
        Some(k) => k,
        None => {
            eprintln!("error: --type is required");
            print_usage();
            return ExitCode::FAILURE;
        }
    };
    let reload = match reload {
        Some(r) => r,
        None => {
            eprintln!("error: --reload is required");
            print_usage();
            return ExitCode::FAILURE;
        }
    };

    let s = render_diff(&ScaffoldArgs { path, kind, reload });
    print!("{s}");
    ExitCode::SUCCESS
}
