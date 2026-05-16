// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Build script for the SmallAIOS x86-64 HAL crate.
//!
//! ## Speculative-execution compiler hardening (spec-exec-mitigations-v1 Ph.2)
//!
//! When the `spec-exec-x86` feature is active AND we are compiling the
//! freestanding kernel target (`x86_64-unknown-none`), the kernel image must
//! be built with two LLVM-level Spectre mitigations.
//!
//! **Retpoline** (`-Z retpoline`) mitigates Spectre-v2 / Branch Target
//! Injection: every indirect `call`/`jmp` becomes a return-trampoline so the
//! indirect-branch predictor cannot be steered. nightly-2026-02-01 rejects
//! the older `-C target-feature=+retpoline-*` form and directs callers to the
//! dedicated `-Zretpoline` flag; that flag emits the thunk *inline* so the
//! freestanding kernel needs no externally-provided `__x86_indirect_thunk_*`
//! symbols (see `.cargo/config.toml`).
//!
//! **Speculative Load Hardening / SLH**
//! (`-C llvm-args=-x86-speculative-load-hardening`) mitigates Spectre-v1 /
//! bounds-check bypass: it masks loaded addresses with a
//! misprediction-derived poison so a mispredicted branch cannot become an
//! attacker-controlled transient load, complementing the explicit `lfence`
//! capability barrier in `smallaios_kernel::spec_exec`.
//!
//! ### Why the flags live in target-scoped `.cargo/config.toml`, not here
//!
//! Cargo build scripts may only emit `-l`/`-L` via `cargo::rustc-flags`;
//! arbitrary `-C` codegen flags are rejected by Cargo (by design). The flags
//! are therefore set in `.cargo/config.toml` under
//! `[target.x86_64-unknown-none]`. Crucially this is **target-scoped, not a
//! blanket global `RUSTFLAGS`**: it applies *only* to the bare-metal kernel
//! triple and never to host build-script / proc-macro compilation (which
//! builds for the host triple, e.g. `aarch64-apple-darwin`). That satisfies
//! the design constraint "do not set blanket global RUSTFLAGS (would hit
//! build-script host code)".
//!
//! This build script's job is the **feature/target gate, the build-time
//! diagnostic, and a `cfg` marker** so the rest of the crate (and humans
//! reading `cargo:warning` output) can see, at build time, whether the
//! hardening is in force and whether it is consistent with the feature set.
//!
//! ### Verification (Phase 2, task 1.3 / 1.13)
//!
//! ```text
//! just build-kernel-x86            # release; default features ⇒ spec-exec-x86
//! BIN=target/x86_64-unknown-none/release/smallaios-x86_64
//! # Retpoline: indirect branches must route through __x86_indirect_thunk_*;
//! # there must be NO bare `call *%rNN` / `jmp *%rNN` in hardened code:
//! objdump -d "$BIN" | grep -E '(call|jmp) +\*%r' | grep -v x86_indirect_thunk
//!   # expected: empty
//! objdump -d "$BIN" | grep -c 'x86_indirect_thunk'   # expected: > 0
//! # SLH: branchless poison dataflow (sbb/cmov against an all-ones mask reg)
//! # around capability-gated load sites:
//! objdump -d "$BIN" | grep -nE 'sbb|cmov' | head
//! ```

use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_SPEC_EXEC_X86");
    println!("cargo:rerun-if-env-changed=TARGET");
    // Declared so downstream `cfg(spec_exec_x86_hardened)` is not an
    // "unexpected cfg" lint under the workspace's check-cfg policy.
    println!("cargo:rustc-check-cfg=cfg(spec_exec_x86_hardened)");

    let spec_exec = env::var_os("CARGO_FEATURE_SPEC_EXEC_X86").is_some();
    let target = env::var("TARGET").unwrap_or_default();
    let is_kernel_target = target == "x86_64-unknown-none";

    if !spec_exec {
        println!(
            "cargo:warning=[spec-exec-x86] feature OFF — kernel built WITHOUT \
             Retpoline/SLH (Spectre v1/v2 compiler mitigations disabled). \
             Build with default features (or --features spec-exec-x86) for \
             the hardened image."
        );
        return;
    }

    if !is_kernel_target {
        // Active feature on a host/library build (`cargo test`/`clippy` on the
        // workspace). The codegen flags do not apply to the host triple; stay
        // quiet so the host harness is unaffected.
        return;
    }

    // Feature + kernel target: the hardened path. The actual `-C` flags are
    // injected by `.cargo/config.toml` `[target.x86_64-unknown-none]`. Set a
    // cfg marker and emit a visible confirmation.
    println!("cargo:rustc-cfg=spec_exec_x86_hardened");
    println!(
        "cargo:warning=[spec-exec-x86] Retpoline + SLH compiler hardening \
         ACTIVE for {target} (flags via target-scoped .cargo/config.toml; \
         verify with `objdump -d` per build.rs docs / PR Verification)."
    );
}
