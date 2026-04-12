// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Stub types for non-macOS platforms.
//!
//! Allows the crate to be referenced in workspace-wide operations (fmt, clippy)
//! without compilation errors on Linux/Windows. All operations return errors.

use alloc::string::String;
use smallaios_compute::{BackendType, ComputeProvider, DeviceInfo};

/// Stub buffer (non-macOS).
pub struct MetalBuffer;

/// Stub kernel (non-macOS).
pub struct MetalKernel;

/// Error type for Metal operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalError(pub String);

impl core::fmt::Display for MetalError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Metal error: {}", self.0)
    }
}

/// Stub tensor cache for non-macOS platforms.
pub struct MetalTensorCache;

impl MetalTensorCache {
    /// Always returns an empty cache on non-macOS.
    pub fn new() -> Self {
        MetalTensorCache
    }
}

/// Stub Metal provider for non-macOS platforms.
pub struct MetalProvider;

impl MetalProvider {
    /// Always returns an error on non-macOS platforms.
    pub fn new() -> Result<Self, MetalError> {
        Err(MetalError(String::from("Metal is only available on macOS")))
    }
}

impl ComputeProvider for MetalProvider {
    type Buffer = MetalBuffer;
    type Kernel = MetalKernel;
    type Error = MetalError;

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: String::from("Metal (unavailable)"),
            memory_bytes: 0,
            compute_units: 0,
            backend_type: BackendType::Metal,
        }
    }

    fn init(&mut self) -> Result<(), Self::Error> {
        Err(MetalError(String::from("Metal unavailable")))
    }

    fn alloc(&mut self, _size: usize) -> Result<Self::Buffer, Self::Error> {
        Err(MetalError(String::from("Metal unavailable")))
    }

    fn free(&mut self, _buf: Self::Buffer) -> Result<(), Self::Error> {
        Err(MetalError(String::from("Metal unavailable")))
    }

    fn copy_host_to_device(
        &mut self,
        _src: &[u8],
        _dst: &mut Self::Buffer,
    ) -> Result<(), Self::Error> {
        Err(MetalError(String::from("Metal unavailable")))
    }

    fn copy_device_to_host(&self, _src: &Self::Buffer, _dst: &mut [u8]) -> Result<(), Self::Error> {
        Err(MetalError(String::from("Metal unavailable")))
    }

    fn load_kernel(&mut self, _name: &str, _source: &[u8]) -> Result<Self::Kernel, Self::Error> {
        Err(MetalError(String::from("Metal unavailable")))
    }

    fn launch(
        &mut self,
        _kernel: &Self::Kernel,
        _grid: [u32; 3],
        _block: [u32; 3],
        _args: &[&Self::Buffer],
    ) -> Result<(), Self::Error> {
        Err(MetalError(String::from("Metal unavailable")))
    }

    fn synchronize(&mut self) -> Result<(), Self::Error> {
        Err(MetalError(String::from("Metal unavailable")))
    }

    fn supports_op(&self, _op: &str) -> bool {
        false
    }
}
