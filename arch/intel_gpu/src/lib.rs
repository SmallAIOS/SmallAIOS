// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Intel GPU Hardware Abstraction Layer
//!
//! Minimal GPU driver for compute workloads on Intel Xe architecture:
//! - PCIe enumeration and BAR mapping (vendor 0x8086)
//! - GPU identification and initialization (Xe-LP, Xe-HPG, Xe-HPC)
//! - Local memory management (allocator, GPU page tables)
//! - DMA/Copy Engine for host<->device transfers
//! - EU-based compute dispatch (workgroup submission)
//! - SPIR-V kernel registry
//! - Level Zero execution provider for ONNX inference
//!
//! Supports: Xe-LP (Gen12 iGPU), Xe-HPG (Arc A-series),
//!           Xe-HPC (Data Center GPU Max / Ponte Vecchio)

#![no_std]
#![allow(dead_code)]

extern crate alloc;

pub mod compute;
pub mod dma;
pub mod gpu_id;
pub mod gpu_init;
pub mod level_zero_provider;
pub mod memory;
pub mod pcie;
pub mod spirv;

/// GPU error type used throughout the Intel GPU crate.
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
    /// Memory allocation or addressing error.
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
