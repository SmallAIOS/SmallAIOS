// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! NVIDIA GPU Hardware Abstraction Layer
//!
//! Minimal GPU driver for compute workloads:
//! - PCIe enumeration and BAR mapping
//! - GPU identification and initialization
//! - VRAM memory management (allocator, GPU page tables)
//! - Command FIFO (push buffer submission)
//! - Compute engine kernel launch
//! - DMA/Copy Engine for host↔device transfers
//! - MSI-X interrupt handling for completion notification
//!
//! Supports: Maxwell (CC 5.3), Volta (7.0), Turing (7.5),
//!           Ampere (8.0/8.7), Hopper (9.0), Blackwell (10.0)

#![no_std]
