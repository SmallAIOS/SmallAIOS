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
#![allow(dead_code)]

extern crate alloc;

pub mod compute;
pub mod cuda_provider;
pub mod dma;
pub mod gpu_id;
pub mod gpu_init;
pub mod memory;
pub mod pcie;
pub mod ptx;

/// GPU error type used throughout the NVIDIA crate.
#[derive(Clone, Debug, PartialEq)]
pub enum GpuError {
    /// Device not supported or not recognized.
    UnsupportedDevice,
    /// Operation invalid for the current state.
    InvalidState,
    /// PCI BAR not found or not mapped.
    BarNotFound,
    /// GPU initialization failure.
    InitFailed,
    /// VRAM allocation or addressing error.
    MemoryError,
    /// DMA/copy-engine error.
    DmaError,
    /// Kernel launch error.
    LaunchError,
    /// Resource not found by ID.
    NotFound,
    /// Queue is full.
    QueueFull,
    /// Invalid configuration parameter.
    InvalidConfig,
    /// Transfer exceeds maximum allowed size.
    TransferTooLarge,
}
