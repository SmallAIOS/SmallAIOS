// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RISC-V security HAL.
//!
//! Houses speculative-execution / control-flow-integrity scaffolding for the
//! RISC-V port. RISC-V is currently a boot-only target (no syscall workloads
//! exercised yet), and the relevant hardening extensions are mostly
//! unratified, so this module is deliberately a stub whose *shape* — not yet
//! its silicon programming — is correct-by-construction.
//!
//! See `docs/spec-exec-audit-riscv.md` for the planned mitigation matrix and
//! the OpenSpec change `spec-exec-mitigations-v1` (Phase 4) for tracking.

pub mod spec_exec;
