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

pub mod mem;
pub mod sched;
pub mod syscall;

/// Kernel version
pub const VERSION: &str = "0.1.0";

/// Kernel name
pub const NAME: &str = "SmallAIOS";
