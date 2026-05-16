// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AArch64 security hardening.
//!
//! Part of OpenSpec change `spec-exec-mitigations-v1` (Phase 3 — aarch64).
//! `tasks.md` is tracked on the spine branch.
//!
//! # Speculative-execution mitigations on this architecture
//!
//! | Attack class           | Mitigation on AArch64 (Cortex-A78AE / ARMv8.5-A) |
//! |------------------------|--------------------------------------------------|
//! | Spectre v1 (BCB)       | `csdb` after the capability check ([`spec_exec`]) |
//! | Spectre v2 (BTI)       | **BTI — see cross-reference below**              |
//! | Spectre v4 (SSB)       | Silicon `SSBS` / `ID_AA64PFR1_EL1.SSBS` (future) |
//! | Meltdown / Retbleed    | Not applicable to Cortex-A78AE (CSV3 reported)   |
//! | DMA-path speculation   | `dsb sy` before DMA setup ([`spec_exec`])        |
//!
//! ## BTI (Branch Target Identification) — cross-reference
//!
//! **Branch Target Identification provides the Spectre-v2 (indirect-branch
//! target injection) mitigation on AArch64, and it is _not_ implemented by
//! this change.** It is owned by the `aarch64-mte-pac-hardening-v1` OpenSpec
//! change:
//!
//! * BTI is emitted by the default `aarch64-unknown-none` /
//!   `aarch64-unknown-uefi` codegen via Rust's `-Z branch-protection=bti`
//!   (the compiler places a `bti c` landing pad at every indirect-branch
//!   target). See `aarch64-mte-pac-hardening-v1/proposal.md` (line 50) and
//!   that change's `tasks.md` 1.9–1.10, which add
//!   `-C codegen-options=branch-protection=pac-ret,bti` to the build flags
//!   and verify `bti c` / `pacibsp` / `autibsp` emission via `objdump`.
//! * Because every legal indirect-branch target is a BTI landing pad and
//!   speculative branches to non-landing-pad addresses are architecturally
//!   constrained, an attacker cannot steer an indirect branch to an arbitrary
//!   speculation gadget. That is the Spectre-v2 indirect-call mitigation on
//!   this arch.
//!
//! `spec-exec-mitigations-v1` therefore does **not** re-implement BTI; it
//! only documents the dependency here so the DO-178C safety case can cite a
//! single canonical Spectre-v2 mitigation on aarch64. The `csdb` / `dsb sy`
//! barriers in [`spec_exec`] cover the Spectre-v1-class load *past the
//! capability gate*, which BTI does not address.

pub mod spec_exec;
