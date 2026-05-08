// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS Kernel Core
//!
//! Provides the fundamental kernel services for AI inference workloads:
//! - Physical and virtual memory management (buddy allocator, slab allocator, tensor pool)
//! - Cooperative task scheduler with async executor
//! - Syscall dispatch interface
//! - Interrupt handling

#![no_std]

extern crate alloc;

pub mod auth;
pub mod hal;
pub mod mem;
pub mod safety;
pub mod sched;
pub mod state;
pub mod syscall;

#[cfg(feature = "verified-boot")]
pub mod boot_integrity;

/// Kernel version. Sourced from the workspace `Cargo.toml` package
/// version at compile time so the constant tracks `cargo-release`
/// bumps automatically rather than drifting against a hardcoded
/// string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Kernel name
pub const NAME: &str = "SmallAIOS";
