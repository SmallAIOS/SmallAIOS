// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARINC 664 Part 7 (AFDX) — Avionics Full-Duplex Switched Ethernet.
//!
//! Implements Virtual Link (VL) configuration, BAG traffic shaping,
//! sequence number management, dual-network redundancy, and integrity checking.

pub mod adapter;
pub mod ethernet;
pub mod filter;
pub mod frame;
pub mod integrity;
pub mod redundancy;
pub mod shaper;
pub mod vl;

pub use adapter::AfdxZenohAdapter;
pub use ethernet::{AfdxEthernetBridge, AfdxPort};
pub use filter::{AfdxFilter, FilterResult};
pub use frame::{AfdxFrame, AfdxTrailer};
pub use integrity::IntegrityChecker;
pub use redundancy::{Network, RedundancyManager};
pub use shaper::BagShaper;
pub use vl::VirtualLink;
