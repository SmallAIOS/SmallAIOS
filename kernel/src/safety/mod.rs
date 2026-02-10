// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Safety-critical process infrastructure.
//!
//! Provides runtime traceability, coding standard enforcement,
//! and safety metric collection for DO-178C DAL A compliance.

pub mod coding_standard;
pub mod coverage;
pub mod traceability;
pub mod verification;
