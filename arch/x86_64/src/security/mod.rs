// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! x86-64 security hardening.
//!
//! Currently houses the speculative-execution mitigations
//! (`spec-exec-mitigations-v1`, Phase 2). See `spec_exec` and
//! `docs/spec-exec-audit.md`.

pub mod spec_exec;
