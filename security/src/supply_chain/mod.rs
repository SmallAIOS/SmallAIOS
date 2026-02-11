// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Supply chain security subsystem.
//!
//! Provides data structures and validation for:
//! - CycloneDX SBOM generation (software bill of materials)
//! - Vulnerability assessment (cargo-audit integration point)
//! - Build attestation (ML-DSA-65 signed build provenance)
//! - Vendor assessment (hardware component tracking)
//! - Reproducible build verification

pub mod attestation;
pub mod ci_tests;
pub mod clean_room;
pub mod oci_sbom;
pub mod sbom;
pub mod vendor;
pub mod vulnerability;
