// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CUDA device memory manager.
//!
//! Provides `DeviceBuffer` for RAII-managed GPU allocations and
//! `DeviceWeightStore` for bulk-transferring model initializers to VRAM.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::string::String;

use super::ffi;
use super::CudaError;

/// RAII wrapper around a CUDA device memory allocation.
///
/// Automatically calls `cudaFree` on drop.
pub struct DeviceBuffer {
    ptr: *mut core::ffi::c_void,
    size: usize,
}

// DeviceBuffer holds a raw GPU pointer — not Send/Sync by default.
// CUDA operations are serialized per-stream, and we use a single default stream,
// so cross-thread sharing is safe as long as the CUDA context is current.
unsafe impl Send for DeviceBuffer {}
unsafe impl Sync for DeviceBuffer {}

impl DeviceBuffer {
    /// Allocate `size` bytes of device memory.
    pub fn alloc(size: usize) -> Result<Self, CudaError> {
        if size == 0 {
            return Ok(Self {
                ptr: core::ptr::null_mut(),
                size: 0,
            });
        }
        let mut ptr: *mut core::ffi::c_void = core::ptr::null_mut();
        let err = unsafe { ffi::cudaMalloc(&mut ptr, size) };
        if err != ffi::CUDA_SUCCESS {
            return Err(CudaError::AllocFailed { size, code: err });
        }
        Ok(Self { ptr, size })
    }

    /// Copy data from host memory to this device buffer.
    pub fn copy_from_host(&self, data: &[u8]) -> Result<(), CudaError> {
        if data.len() > self.size {
            return Err(CudaError::CopyFailed {
                msg: "host data larger than device buffer",
                code: -1,
            });
        }
        if data.is_empty() {
            return Ok(());
        }
        let err = unsafe {
            ffi::cudaMemcpy(
                self.ptr,
                data.as_ptr() as *const core::ffi::c_void,
                data.len(),
                ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
            )
        };
        if err != ffi::CUDA_SUCCESS {
            return Err(CudaError::CopyFailed {
                msg: "host-to-device",
                code: err,
            });
        }
        Ok(())
    }

    /// Copy data from this device buffer to host memory.
    pub fn copy_to_host(&self, dst: &mut [u8]) -> Result<(), CudaError> {
        let copy_len = dst.len().min(self.size);
        if copy_len == 0 {
            return Ok(());
        }
        let err = unsafe {
            ffi::cudaMemcpy(
                dst.as_mut_ptr() as *mut core::ffi::c_void,
                self.ptr as *const core::ffi::c_void,
                copy_len,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
            )
        };
        if err != ffi::CUDA_SUCCESS {
            return Err(CudaError::CopyFailed {
                msg: "device-to-host",
                code: err,
            });
        }
        Ok(())
    }

    /// Raw device pointer (for passing to cuBLAS/cuDNN).
    pub fn as_ptr(&self) -> *const core::ffi::c_void {
        self.ptr as *const _
    }

    /// Raw mutable device pointer.
    pub fn as_mut_ptr(&self) -> *mut core::ffi::c_void {
        self.ptr
    }

    /// Size in bytes.
    pub fn size(&self) -> usize {
        self.size
    }
}

impl Drop for DeviceBuffer {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                ffi::cudaFree(self.ptr);
            }
        }
    }
}

/// Stores model weight tensors (initializers) in GPU VRAM.
///
/// Weights are transferred once at session load time and reused across
/// all inference calls. Keyed by initializer name.
pub struct DeviceWeightStore {
    weights: BTreeMap<String, DeviceBuffer>,
    total_bytes: usize,
}

impl Default for DeviceWeightStore {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceWeightStore {
    /// Create an empty weight store.
    pub fn new() -> Self {
        Self {
            weights: BTreeMap::new(),
            total_bytes: 0,
        }
    }

    /// Transfer a weight tensor to GPU VRAM.
    pub fn load_weight(&mut self, name: String, data: &[u8]) -> Result<(), CudaError> {
        let buf = DeviceBuffer::alloc(data.len())?;
        buf.copy_from_host(data)?;
        self.total_bytes += data.len();
        self.weights.insert(name, buf);
        Ok(())
    }

    /// Get a device pointer for a previously loaded weight.
    pub fn get(&self, name: &str) -> Option<&DeviceBuffer> {
        self.weights.get(name)
    }

    /// Total VRAM consumed by loaded weights.
    pub fn total_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Number of loaded weight tensors.
    pub fn count(&self) -> usize {
        self.weights.len()
    }
}
