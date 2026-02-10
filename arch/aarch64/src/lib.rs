// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARM64 (AArch64) Hardware Abstraction Layer
//!
//! Provides platform-specific initialization and hardware access:
//! - UEFI/DTB boot entry
//! - Exception vector table
//! - GICv3/v4 interrupt controller
//! - AArch64 page table management (4K/16K/64K granule)
//! - CPU feature detection (NEON, SVE/SVE2, SME, PAC, BTI, MTE)
//! - PSCI for SMP boot
//! - PL011 UART console

#![no_std]
