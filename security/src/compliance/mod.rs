// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Security compliance and cross-reference data.
//!
//! Provides:
//! - NIST SP 800-53 control mapping to SmallAIOS security mechanisms
//! - Data classification enforcement (Public/Internal/Restricted)
//! - Vulnerability disclosure tracking with triage SLA
//! - FIPS 140-3 Level 1 crypto module boundary definition
//! - Safety-critical cross-reference tables (DO-178C, IEC 61508, ISO 26262)
//! - Security test category definitions

pub mod classification;
pub mod crypto_boundary;
pub mod nist_controls;
pub mod safety_crossref;
pub mod test_categories;
pub mod vulnerability_disclosure;
