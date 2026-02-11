// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Intel GPU Hardware Abstraction Layer
//!
//! Minimal GPU driver for compute workloads on Intel discrete and integrated
//! GPUs:
//! - PCIe enumeration and BAR mapping
//! - GPU identification and initialization
//! - VRAM/GTT memory management (allocator, GPU page tables)
//! - Blitter command streamer (DMA/copy engine)
//! - EU-based compute engine with SIMD8/16/32 dispatch
//! - SPIR-V kernel registry
//! - Level Zero execution provider for ONNX inference
//!
//! Supports: Xe-LP (integrated), Xe-HPG (Arc A-series),
//!           Xe-HPC (Data Center GPU Max), Xe-LPG (Meteor Lake),
//!           Xe2 (Battlemage)

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
pub mod spirv_kernels;

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
    /// VRAM allocation or addressing error.
    MemoryError,
    /// DMA/blitter-engine error.
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
