// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Incident response subsystem.
//!
//! Provides automated incident handling per NIST SP 800-61:
//! - Severity classification (Critical/High/Medium/Low)
//! - Automated containment actions (capability revocation, task termination, network isolation)
//! - Evidence preservation (audit snapshots, task state, memory stats, capability registry)
//! - Incident event publishing (structured events for Zenoh export)

pub mod communication;
pub mod containment;
pub mod event;
pub mod evidence;
pub mod post_incident;
pub mod severity_classifier;

pub use containment::{ContainmentAction, ContainmentResult};
pub use event::{IncidentEvent, IncidentSeverity};
pub use evidence::EvidenceSnapshot;
