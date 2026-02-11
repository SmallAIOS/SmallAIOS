// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! AMD GPU Hardware Abstraction Layer
//!
//! Minimal GPU driver for compute workloads:
//! - PCIe enumeration and BAR mapping
//! - GPU identification and initialization
//! - VRAM memory management (allocator, GPU page tables)
//! - SDMA engine for host<->device transfers
//! - Wavefront-based compute engine kernel launch
//! - ROCm/HIP execution provider for ONNX inference
//!
//! Supports: RDNA 1/2/3 (RX 5000/6000/7000),
//!           CDNA 1/2/3 (MI100/MI200/MI300)

#![no_std]
#![allow(dead_code)]

extern crate alloc;

pub mod compute;
pub mod dma;
pub mod gpu_id;
pub mod gpu_init;
pub mod hip_kernels;
pub mod memory;
pub mod pcie;
pub mod rocm_provider;

/// GPU error type used throughout the AMD crate.
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
    /// DMA/SDMA engine error.
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
