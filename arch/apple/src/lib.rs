// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Apple Metal GPU Backend for SmallAIOS
//!
//! Container-mode only (macOS / Apple Silicon). Provides GPU compute via the
//! Metal framework for ONNX inference workloads.
//!
//! # Architecture
//!
//! - [`MetalProvider`] implements [`smallaios_compute::ComputeProvider`]
//! - Buffers use `StorageModeShared` for unified memory on Apple Silicon
//! - Kernels are compiled from MSL source at runtime
//! - Pre-built MSL shaders in [`shaders`] cover common ONNX operators

#![no_std]
#![allow(dead_code)]

extern crate alloc;

#[cfg(target_os = "macos")]
mod metal_provider;
#[cfg(target_os = "macos")]
pub mod shaders;

#[cfg(target_os = "macos")]
pub use metal_provider::{MetalBuffer, MetalError, MetalKernel, MetalProvider, MetalTensorCache};

// Re-export stubs for non-macOS so the crate still compiles in the workspace.
#[cfg(not(target_os = "macos"))]
mod stub;
#[cfg(not(target_os = "macos"))]
pub use stub::{MetalBuffer, MetalError, MetalKernel, MetalProvider, MetalTensorCache};
