// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Memory syscalls (0x00-0x0F).
//!
//! - `mem_alloc(size, align, flags) -> *mut u8`
//! - `mem_free(ptr, size)`
//! - `mem_map(phys, virt, size, flags)` — MMIO mapping
//! - `mem_protect(ptr, size, flags)` — change permissions
//! - `tensor_alloc(shape_ptr, ndim, dtype) -> TensorHandle`
//! - `tensor_free(handle)`
//! - `tensor_map_gpu(handle, device) -> GpuPtr`
//! - `tensor_unmap_gpu(handle, device)`

use super::{SyscallArgs, SyscallError, SyscallResult};

/// Memory allocation flags.
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum MemFlags {
    /// Default allocation from kernel heap.
    Normal = 0,
    /// DMA-capable allocation (physically contiguous).
    Dma = 1,
    /// Use 2 MiB huge pages.
    HugePage2M = 2,
    /// Use 1 GiB huge pages.
    HugePage1G = 3,
    /// Zero-initialized memory.
    Zeroed = 4,
}

/// Memory protection flags.
#[derive(Debug, Clone, Copy)]
#[repr(u32)]
pub enum ProtFlags {
    /// No access.
    None = 0,
    /// Read-only.
    Read = 1,
    /// Read-write.
    ReadWrite = 3,
    /// Read-execute.
    ReadExecute = 5,
}

/// Tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum TensorDtype {
    Float32 = 0,
    Float16 = 1,
    Int8 = 2,
    Uint8 = 3,
    Int32 = 4,
    Int64 = 5,
    Float64 = 6,
    BFloat16 = 7,
}

impl TensorDtype {
    /// Size of one element in bytes.
    pub const fn element_size(self) -> usize {
        match self {
            Self::Float32 | Self::Int32 => 4,
            Self::Float16 | Self::BFloat16 => 2,
            Self::Int8 | Self::Uint8 => 1,
            Self::Int64 | Self::Float64 => 8,
        }
    }

    /// Try to convert from raw u32.
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Float32),
            1 => Some(Self::Float16),
            2 => Some(Self::Int8),
            3 => Some(Self::Uint8),
            4 => Some(Self::Int32),
            5 => Some(Self::Int64),
            6 => Some(Self::Float64),
            7 => Some(Self::BFloat16),
            _ => None,
        }
    }
}

/// Allocate kernel memory.
///
/// Args: [size, align, flags, 0, 0, 0]
/// Returns: pointer as usize on success, negative error code on failure.
pub fn sys_mem_alloc(args: &SyscallArgs) -> SyscallResult {
    let size = args.args[0];
    let align = args.args[1];

    // Validate arguments
    if size == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if align != 0 && !align.is_power_of_two() {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with actual buddy/slab allocator
    // For now, return not-supported to indicate stub
    let _ = (size, align);
    SyscallError::NotSupported.as_i64()
}

/// Free kernel memory.
///
/// Args: [ptr, size, 0, 0, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_mem_free(args: &SyscallArgs) -> SyscallResult {
    let ptr = args.args[0];
    let size = args.args[1];

    if ptr == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if size == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with actual allocator
    let _ = (ptr, size);
    SyscallError::NotSupported.as_i64()
}

/// Map physical memory (MMIO).
///
/// Args: [phys_addr, virt_addr, size, flags, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_mem_map(args: &SyscallArgs) -> SyscallResult {
    let phys = args.args[0];
    let virt = args.args[1];
    let size = args.args[2];

    if size == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with page table management
    let _ = (phys, virt, size);
    SyscallError::NotSupported.as_i64()
}

/// Change memory protection flags.
///
/// Args: [ptr, size, prot_flags, 0, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_mem_protect(args: &SyscallArgs) -> SyscallResult {
    let ptr = args.args[0];
    let size = args.args[1];

    if ptr == 0 || size == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with page table management
    let _ = (ptr, size);
    SyscallError::NotSupported.as_i64()
}

/// Allocate a tensor buffer.
///
/// Args: [shape_ptr, ndim, dtype, 0, 0, 0]
/// Returns: tensor handle on success, negative error code on failure.
pub fn sys_tensor_alloc(args: &SyscallArgs) -> SyscallResult {
    let shape_ptr = args.args[0];
    let ndim = args.args[1];
    let dtype_raw = args.args[2] as u32;

    if shape_ptr == 0 || ndim == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if ndim > 8 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if TensorDtype::from_u32(dtype_raw).is_none() {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with tensor pool
    let _ = (shape_ptr, ndim, dtype_raw);
    SyscallError::NotSupported.as_i64()
}

/// Free a tensor buffer.
///
/// Args: [handle, 0, 0, 0, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_tensor_free(args: &SyscallArgs) -> SyscallResult {
    let handle = args.args[0];

    if handle == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }

    // TODO: Integrate with tensor pool
    let _ = handle;
    SyscallError::NotSupported.as_i64()
}

/// Map a tensor buffer for GPU access.
///
/// Args: [handle, device_id, 0, 0, 0, 0]
/// Returns: GPU pointer on success, negative error code on failure.
pub fn sys_tensor_map_gpu(args: &SyscallArgs) -> SyscallResult {
    let handle = args.args[0];
    let device_id = args.args[1];

    if handle == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }

    // TODO: Integrate with GPU memory management
    let _ = (handle, device_id);
    SyscallError::NotSupported.as_i64()
}

/// Unmap a tensor buffer from GPU.
///
/// Args: [handle, device_id, 0, 0, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_tensor_unmap_gpu(args: &SyscallArgs) -> SyscallResult {
    let handle = args.args[0];
    let device_id = args.args[1];

    if handle == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }

    // TODO: Integrate with GPU memory management
    let _ = (handle, device_id);
    SyscallError::NotSupported.as_i64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mem_alloc_zero_size() {
        let args = SyscallArgs::new(0x00, [0, 8, 0, 0, 0, 0]);
        assert_eq!(sys_mem_alloc(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_mem_alloc_bad_alignment() {
        let args = SyscallArgs::new(0x00, [4096, 3, 0, 0, 0, 0]);
        assert_eq!(sys_mem_alloc(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_mem_alloc_zero_alignment_ok() {
        // align=0 means default alignment, should not be rejected as invalid
        let args = SyscallArgs::new(0x00, [4096, 0, 0, 0, 0, 0]);
        // Should return NotSupported (stub) not InvalidArgument
        assert_eq!(sys_mem_alloc(&args), SyscallError::NotSupported.as_i64());
    }

    #[test]
    fn test_mem_alloc_valid_args() {
        let args = SyscallArgs::new(0x00, [4096, 16, 0, 0, 0, 0]);
        assert_eq!(sys_mem_alloc(&args), SyscallError::NotSupported.as_i64());
    }

    #[test]
    fn test_mem_free_null_ptr() {
        let args = SyscallArgs::new(0x01, [0, 4096, 0, 0, 0, 0]);
        assert_eq!(sys_mem_free(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_mem_free_zero_size() {
        let args = SyscallArgs::new(0x01, [0x1000, 0, 0, 0, 0, 0]);
        assert_eq!(sys_mem_free(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_mem_map_zero_size() {
        let args = SyscallArgs::new(0x02, [0x1000, 0x2000, 0, 0, 0, 0]);
        assert_eq!(sys_mem_map(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_mem_protect_null_ptr() {
        let args = SyscallArgs::new(0x03, [0, 4096, 1, 0, 0, 0]);
        assert_eq!(
            sys_mem_protect(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_tensor_alloc_zero_ndim() {
        let args = SyscallArgs::new(0x04, [0x1000, 0, 0, 0, 0, 0]);
        assert_eq!(
            sys_tensor_alloc(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_tensor_alloc_too_many_dims() {
        let args = SyscallArgs::new(0x04, [0x1000, 9, 0, 0, 0, 0]);
        assert_eq!(
            sys_tensor_alloc(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_tensor_alloc_invalid_dtype() {
        let args = SyscallArgs::new(0x04, [0x1000, 4, 99, 0, 0, 0]);
        assert_eq!(
            sys_tensor_alloc(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_tensor_alloc_null_shape() {
        let args = SyscallArgs::new(0x04, [0, 4, 0, 0, 0, 0]);
        assert_eq!(
            sys_tensor_alloc(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_tensor_free_zero_handle() {
        let args = SyscallArgs::new(0x05, [0, 0, 0, 0, 0, 0]);
        assert_eq!(
            sys_tensor_free(&args),
            SyscallError::InvalidHandle.as_i64()
        );
    }

    #[test]
    fn test_tensor_map_gpu_zero_handle() {
        let args = SyscallArgs::new(0x06, [0, 0, 0, 0, 0, 0]);
        assert_eq!(
            sys_tensor_map_gpu(&args),
            SyscallError::InvalidHandle.as_i64()
        );
    }

    #[test]
    fn test_tensor_unmap_gpu_zero_handle() {
        let args = SyscallArgs::new(0x07, [0, 0, 0, 0, 0, 0]);
        assert_eq!(
            sys_tensor_unmap_gpu(&args),
            SyscallError::InvalidHandle.as_i64()
        );
    }

    #[test]
    fn test_tensor_dtype_element_size() {
        assert_eq!(TensorDtype::Float32.element_size(), 4);
        assert_eq!(TensorDtype::Float16.element_size(), 2);
        assert_eq!(TensorDtype::Int8.element_size(), 1);
        assert_eq!(TensorDtype::Uint8.element_size(), 1);
        assert_eq!(TensorDtype::Int32.element_size(), 4);
        assert_eq!(TensorDtype::Int64.element_size(), 8);
        assert_eq!(TensorDtype::Float64.element_size(), 8);
        assert_eq!(TensorDtype::BFloat16.element_size(), 2);
    }

    #[test]
    fn test_tensor_dtype_from_u32() {
        for i in 0..8u32 {
            assert!(TensorDtype::from_u32(i).is_some());
        }
        assert!(TensorDtype::from_u32(8).is_none());
        assert!(TensorDtype::from_u32(255).is_none());
    }
}
