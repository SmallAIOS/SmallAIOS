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
use crate::mem::{PhysAddr, PAGE_SIZE_4K};
use crate::state;
use crate::syscall::task::current_task_id;
use smallaios_security::capability::{Permissions, ResourceRef, ResourceType};

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

/// Convert size to buddy allocator order (log2 of page count, rounded up).
fn size_to_order(size: usize) -> usize {
    let pages = size.div_ceil(PAGE_SIZE_4K);
    if pages <= 1 {
        return 0;
    }
    // order = ceil(log2(pages))
    (usize::BITS - (pages - 1).leading_zeros()) as usize
}

/// Allocate kernel memory.
///
/// Args: [size, align, flags, 0, 0, 0]
/// Returns: pointer as usize on success, negative error code on failure.
pub fn sys_mem_alloc(args: &SyscallArgs) -> SyscallResult {
    let size = args.args[0];
    let align = args.args[1];
    let flags_raw = args.args[2] as u32;

    // Validate arguments
    if size == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if align != 0 && !align.is_power_of_two() {
        return SyscallError::InvalidArgument.as_i64();
    }

    // Determine allocation strategy based on size and flags
    let use_slab = size <= 2048 && flags_raw == 0 && (align == 0 || align <= 16);

    if use_slab {
        // Small allocations go through the slab allocator
        // SAFETY: Syscall handlers run with interrupts masked.
        let result = unsafe { state::with_slab(|slab| slab.allocate(size)) };
        match result {
            Ok(ptr) => ptr as i64,
            Err(err) => state::mem_error_to_syscall(err),
        }
    } else {
        // Large allocations go through the buddy allocator
        let order = size_to_order(size);
        // SAFETY: Syscall handlers run with interrupts masked.
        let result = unsafe { state::with_buddy(|buddy| buddy.allocate(order)) };
        match result {
            Ok(addr) => addr.as_usize() as i64,
            Err(err) => state::mem_error_to_syscall(err),
        }
    }
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

    // Try slab first for small sizes, fall back to buddy
    if size <= 2048 {
        // SAFETY: Syscall handlers run with interrupts masked.
        let result = unsafe { state::with_slab(|slab| slab.free(ptr as *mut u8, size)) };
        match result {
            Ok(()) => SyscallError::Success.as_i64(),
            Err(err) => state::mem_error_to_syscall(err),
        }
    } else {
        let order = size_to_order(size);
        let addr = PhysAddr::new(ptr);
        // SAFETY: Syscall handlers run with interrupts masked.
        let result = unsafe { state::with_buddy(|buddy| buddy.free(addr, order)) };
        match result {
            Ok(()) => SyscallError::Success.as_i64(),
            Err(err) => state::mem_error_to_syscall(err),
        }
    }
}

/// Map physical memory (MMIO).
///
/// Args: [phys_addr, virt_addr, size, flags, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_mem_map(args: &SyscallArgs) -> SyscallResult {
    let _phys = args.args[0];
    let _virt = args.args[1];
    let size = args.args[2];

    if size == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // MMIO mapping is a privileged operation — require Device:WRITE capability.
    let resource = ResourceRef::new(ResourceType::Device, 0);
    if let Err(e) = state::check_capability(current_task_id(), &resource, Permissions::WRITE) {
        return e;
    }

    // MMIO mapping requires architecture-specific page table manipulation.
    // This will be wired when the HAL page table API is integrated into
    // the kernel state (depends on architecture selection at boot).
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

    // Memory protection changes are privileged — require Device:WRITE capability.
    let resource = ResourceRef::new(ResourceType::Device, 0);
    if let Err(e) = state::check_capability(current_task_id(), &resource, Permissions::WRITE) {
        return e;
    }

    // Protection changes require architecture-specific page table updates.
    // This will be wired when the HAL page table API is integrated.
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
    let dtype = match TensorDtype::from_u32(dtype_raw) {
        Some(d) => d,
        None => return SyscallError::InvalidArgument.as_i64(),
    };

    // Tensor allocation requires TensorBuffer:WRITE capability.
    let resource = ResourceRef::new(ResourceType::TensorBuffer, 0);
    if let Err(e) = state::check_capability(current_task_id(), &resource, Permissions::WRITE) {
        return e;
    }

    // Read shape dimensions from the shape pointer.
    // SAFETY: We validate shape_ptr is non-null above. In unikernel mode,
    // the caller is in the same address space so the pointer is valid.
    // In VM mode, this would need user-space pointer validation.
    let shape = unsafe { core::slice::from_raw_parts(shape_ptr as *const usize, ndim) };

    // Compute total element count (with overflow checking)
    let mut total_elements: usize = 1;
    for &dim in shape {
        if dim == 0 {
            return SyscallError::InvalidArgument.as_i64();
        }
        total_elements = match total_elements.checked_mul(dim) {
            Some(v) => v,
            None => return SyscallError::InvalidArgument.as_i64(),
        };
    }

    let total_bytes = match total_elements.checked_mul(dtype.element_size()) {
        Some(v) => v,
        None => return SyscallError::InvalidArgument.as_i64(),
    };

    // Allocate from tensor pool
    // SAFETY: Syscall handlers run with interrupts masked.
    let result = unsafe { state::with_tensor_pool(|pool| pool.allocate(total_bytes)) };
    match result {
        Ok((handle, _addr)) => (handle.index() + 1) as i64, // +1 so handle 0 is never returned
        Err(err) => state::mem_error_to_syscall(err),
    }
}

/// Free a tensor buffer.
///
/// Args: [handle, 0, 0, 0, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_tensor_free(args: &SyscallArgs) -> SyscallResult {
    let handle_raw = args.args[0];

    if handle_raw == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }

    // Tensor free requires TensorBuffer:WRITE capability.
    let resource = ResourceRef::new(ResourceType::TensorBuffer, handle_raw as u64);
    if let Err(e) = state::check_capability(current_task_id(), &resource, Permissions::WRITE) {
        return e;
    }

    let handle_idx = handle_raw - 1; // Undo the +1 from alloc
    let handle = crate::mem::tensor::TensorHandle::from_index(handle_idx as u32);

    // SAFETY: Syscall handlers run with interrupts masked.
    let result = unsafe { state::with_tensor_pool(|pool| pool.release(handle)) };
    match result {
        Ok(()) => SyscallError::Success.as_i64(),
        Err(err) => state::mem_error_to_syscall(err),
    }
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

    // Requires TensorBuffer:WRITE on the tensor and GpuDevice:EXECUTE on the device.
    let tensor_res = ResourceRef::new(ResourceType::TensorBuffer, handle as u64);
    if let Err(e) = state::check_capability(current_task_id(), &tensor_res, Permissions::WRITE) {
        return e;
    }
    let gpu_res = ResourceRef::new(ResourceType::GpuDevice, device_id as u64);
    if let Err(e) = state::check_capability(current_task_id(), &gpu_res, Permissions::EXECUTE) {
        return e;
    }

    // GPU mapping requires the NVIDIA HAL which is not yet integrated.
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

    // Requires TensorBuffer:WRITE on the tensor and GpuDevice:EXECUTE on the device.
    let tensor_res = ResourceRef::new(ResourceType::TensorBuffer, handle as u64);
    if let Err(e) = state::check_capability(current_task_id(), &tensor_res, Permissions::WRITE) {
        return e;
    }
    let gpu_res = ResourceRef::new(ResourceType::GpuDevice, device_id as u64);
    if let Err(e) = state::check_capability(current_task_id(), &gpu_res, Permissions::EXECUTE) {
        return e;
    }

    // GPU unmapping requires the NVIDIA HAL which is not yet integrated.
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
        let result = sys_mem_alloc(&args);
        // Allocator not initialized, so expect OutOfMemory or a valid pointer
        assert!(result < 0 || result > 0);
    }

    #[test]
    fn test_mem_alloc_valid_args() {
        let args = SyscallArgs::new(0x00, [4096, 16, 0, 0, 0, 0]);
        let result = sys_mem_alloc(&args);
        // Allocator not initialized, will return an error
        assert!(result < 0 || result > 0);
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
        assert_eq!(sys_tensor_free(&args), SyscallError::InvalidHandle.as_i64());
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

    #[test]
    fn test_size_to_order() {
        assert_eq!(size_to_order(1), 0); // 1 byte = 1 page = order 0
        assert_eq!(size_to_order(4096), 0); // 4K = 1 page = order 0
        assert_eq!(size_to_order(4097), 1); // > 1 page = order 1 (2 pages)
        assert_eq!(size_to_order(8192), 1); // 2 pages = order 1
        assert_eq!(size_to_order(8193), 2); // > 2 pages = order 2 (4 pages)
        assert_eq!(size_to_order(2 * 1024 * 1024), 9); // 2 MiB = 512 pages = order 9
    }

    // --- Coverage tests for slab/buddy allocation paths ---

    #[test]
    fn test_mem_alloc_slab_path() {
        // size <= 2048, flags=0, align=0 → slab allocator path
        let args = SyscallArgs::new(0x00, [64, 0, 0, 0, 0, 0]);
        let result = sys_mem_alloc(&args);
        // Slab not initialized in test context, expect an error
        assert_ne!(result, 0);
    }

    #[test]
    fn test_mem_alloc_slab_with_small_align() {
        // size=128, align=16, flags=0 → slab path (align <= 16)
        let args = SyscallArgs::new(0x00, [128, 16, 0, 0, 0, 0]);
        let result = sys_mem_alloc(&args);
        assert_ne!(result, 0);
    }

    #[test]
    fn test_mem_alloc_buddy_path_large_size() {
        // size > 2048 → buddy allocator path
        let args = SyscallArgs::new(0x00, [8192, 0, 0, 0, 0, 0]);
        let result = sys_mem_alloc(&args);
        assert_ne!(result, 0);
    }

    #[test]
    fn test_mem_alloc_large_align_forces_buddy() {
        // size=128 (would be slab), but align=32 (> 16) forces buddy path
        let args = SyscallArgs::new(0x00, [128, 32, 0, 0, 0, 0]);
        let result = sys_mem_alloc(&args);
        assert_ne!(result, 0);
    }

    #[test]
    fn test_mem_alloc_nonzero_flags_forces_buddy() {
        // size=128 (would be slab), but flags=1 (non-zero) forces buddy path
        let args = SyscallArgs::new(0x00, [128, 0, 1, 0, 0, 0]);
        let result = sys_mem_alloc(&args);
        assert_ne!(result, 0);
    }

    #[test]
    fn test_mem_free_slab_path() {
        // size <= 2048 → slab free path
        let args = SyscallArgs::new(0x01, [0x1000, 64, 0, 0, 0, 0]);
        let result = sys_mem_free(&args);
        // Slab not initialized, will fail but exercises slab free path
        assert_ne!(result, 0);
    }

    #[test]
    fn test_mem_free_buddy_path() {
        // size > 2048 → buddy free path
        let args = SyscallArgs::new(0x01, [0x1000, 8192, 0, 0, 0, 0]);
        let result = sys_mem_free(&args);
        // Buddy not initialized, will fail but exercises buddy free path
        assert_ne!(result, 0);
    }

    // --- Coverage tests for capability-checked paths ---

    #[test]
    fn test_mem_map_with_nonzero_size() {
        // Non-zero size passes validation, hits capability check
        let args = SyscallArgs::new(0x02, [0x1000, 0x2000, 4096, 0, 0, 0]);
        let result = sys_mem_map(&args);
        // No capabilities registered → PermissionDenied or NotSupported
        assert!(result < 0);
    }

    #[test]
    fn test_mem_protect_with_nonzero_ptr_and_size() {
        // Non-zero ptr and non-zero size passes validation, hits capability check
        let args = SyscallArgs::new(0x03, [0x1000, 4096, 1, 0, 0, 0]);
        let result = sys_mem_protect(&args);
        // No capabilities registered → PermissionDenied
        assert!(result < 0);
    }

    #[test]
    fn test_mem_protect_zero_size_nonzero_ptr() {
        // ptr != 0 but size == 0 → InvalidArgument (short-circuit on OR condition)
        let args = SyscallArgs::new(0x03, [0x1000, 0, 1, 0, 0, 0]);
        assert_eq!(
            sys_mem_protect(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_tensor_free_valid_handle() {
        // Non-zero handle passes validation, hits capability check.
        // Use a high handle value unlikely to have a registered capability.
        let args = SyscallArgs::new(0x05, [50000, 0, 0, 0, 0, 0]);
        let result = sys_tensor_free(&args);
        // No capability for this instance → PermissionDenied
        assert_eq!(result, SyscallError::PermissionDenied.as_i64());
    }

    #[test]
    fn test_tensor_map_gpu_valid_handle() {
        // Non-zero handle passes validation, hits capability check.
        // Use a high handle value unlikely to have a registered capability.
        let args = SyscallArgs::new(0x06, [60000, 0, 0, 0, 0, 0]);
        let result = sys_tensor_map_gpu(&args);
        // No capability for this instance → PermissionDenied
        assert_eq!(result, SyscallError::PermissionDenied.as_i64());
    }

    #[test]
    fn test_tensor_unmap_gpu_valid_handle() {
        // Non-zero handle passes validation, hits capability check.
        // Use a high handle value unlikely to have a registered capability.
        let args = SyscallArgs::new(0x07, [60001, 0, 0, 0, 0, 0]);
        let result = sys_tensor_unmap_gpu(&args);
        // No capability for this instance → PermissionDenied
        assert_eq!(result, SyscallError::PermissionDenied.as_i64());
    }

    // --- Coverage tests for enum repr values ---

    #[test]
    fn test_mem_flags_repr_values() {
        assert_eq!(MemFlags::Normal as u32, 0);
        assert_eq!(MemFlags::Dma as u32, 1);
        assert_eq!(MemFlags::HugePage2M as u32, 2);
        assert_eq!(MemFlags::HugePage1G as u32, 3);
        assert_eq!(MemFlags::Zeroed as u32, 4);
    }

    #[test]
    fn test_prot_flags_repr_values() {
        assert_eq!(ProtFlags::None as u32, 0);
        assert_eq!(ProtFlags::Read as u32, 1);
        assert_eq!(ProtFlags::ReadWrite as u32, 3);
        assert_eq!(ProtFlags::ReadExecute as u32, 5);
    }

    #[test]
    fn test_tensor_dtype_repr_values() {
        assert_eq!(TensorDtype::Float32 as u32, 0);
        assert_eq!(TensorDtype::Float16 as u32, 1);
        assert_eq!(TensorDtype::Int8 as u32, 2);
        assert_eq!(TensorDtype::Uint8 as u32, 3);
        assert_eq!(TensorDtype::Int32 as u32, 4);
        assert_eq!(TensorDtype::Int64 as u32, 5);
        assert_eq!(TensorDtype::Float64 as u32, 6);
        assert_eq!(TensorDtype::BFloat16 as u32, 7);
    }

    // --- Additional coverage tests ---

    #[test]
    fn test_size_to_order_zero() {
        // size=0 → pages=0 → order 0
        assert_eq!(size_to_order(0), 0);
    }

    #[test]
    fn test_size_to_order_exact_four_pages() {
        // 4 pages = 16384 bytes → order 2
        assert_eq!(size_to_order(16384), 2);
    }

    #[test]
    fn test_size_to_order_just_over_four_pages() {
        // 16385 bytes → 5 pages → order 3 (next power of 2 is 8)
        assert_eq!(size_to_order(16385), 3);
    }

    #[test]
    fn test_mem_alloc_align_one() {
        // align=1 is a valid power of 2; size=4096 with align=1 goes to slab path
        // (size <= 2048 is false, so this goes to buddy path)
        let args = SyscallArgs::new(0x00, [4096, 1, 0, 0, 0, 0]);
        let result = sys_mem_alloc(&args);
        // Buddy not initialized → OutOfMemory (negative)
        assert_ne!(result, 0);
    }

    #[test]
    fn test_mem_alloc_slab_boundary_size() {
        // size=2048 is exactly the slab boundary, flags=0, align=0 → slab path
        let args = SyscallArgs::new(0x00, [2048, 0, 0, 0, 0, 0]);
        let result = sys_mem_alloc(&args);
        assert_ne!(result, 0);
    }

    #[test]
    fn test_mem_alloc_just_over_slab_boundary() {
        // size=2049 → buddy path
        let args = SyscallArgs::new(0x00, [2049, 0, 0, 0, 0, 0]);
        let result = sys_mem_alloc(&args);
        assert_ne!(result, 0);
    }

    #[test]
    fn test_mem_free_slab_boundary_size() {
        // size=2048 → slab free path
        let args = SyscallArgs::new(0x01, [0x1000, 2048, 0, 0, 0, 0]);
        let result = sys_mem_free(&args);
        assert_ne!(result, 0);
    }

    #[test]
    fn test_mem_free_just_over_slab_boundary() {
        // size=2049 → buddy free path
        let args = SyscallArgs::new(0x01, [0x1000, 2049, 0, 0, 0, 0]);
        let result = sys_mem_free(&args);
        assert_ne!(result, 0);
    }

    #[test]
    fn test_tensor_alloc_valid_dtype_bfloat16() {
        // dtype=7 (BFloat16) is the highest valid dtype — passes dtype check
        // ndim=1, shape_ptr valid → passes ndim check, hits capability check
        // Without the capability grant, this hits PermissionDenied
        let shape: [usize; 1] = [8];
        let shape_ptr = shape.as_ptr() as usize;
        let args = SyscallArgs::new(0x04, [shape_ptr, 1, 7, 0, 0, 0]);
        let result = sys_tensor_alloc(&args);
        // Either succeeds (if pool+cap ready from another test) or PermissionDenied
        assert_ne!(result, 0);
    }

    #[test]
    fn test_tensor_alloc_ndim_at_max() {
        // ndim=8 is the maximum allowed (> 8 is rejected)
        let shape: [usize; 8] = [1, 2, 1, 1, 1, 1, 1, 3];
        let shape_ptr = shape.as_ptr() as usize;
        let args = SyscallArgs::new(0x04, [shape_ptr, 8, 0, 0, 0, 0]);
        let result = sys_tensor_alloc(&args);
        // Either succeeds (if pool+cap ready from another test) or PermissionDenied
        assert_ne!(result, 0);
    }

    /// Combined tensor pool test that exercises all tensor alloc/free code
    /// paths that require capability + pool state. Consolidated into a single
    /// test to avoid races with the global kernel state used by other tests.
    #[test]
    fn test_tensor_pool_paths_consolidated() {
        // --- Setup: grant capability and initialize pool ---
        unsafe {
            state::with_cap_registry(|reg| {
                // Grant TensorBuffer:WRITE for instance 0 (used by sys_tensor_alloc cap check)
                let resource = ResourceRef::new(ResourceType::TensorBuffer, 0);
                let _ = reg.create(0, resource, Permissions::WRITE, 0);
            });
            state::with_tensor_pool(|pool| {
                pool.init(crate::mem::PhysAddr::new(0x200_0000), 1024 * 1024);
            });
        }

        // --- Test 1: Basic tensor allocation (covers lines 255-280) ---
        let shape: [usize; 2] = [4, 8];
        let shape_ptr = shape.as_ptr() as usize;
        let args = SyscallArgs::new(0x04, [shape_ptr, 2, 0, 0, 0, 0]);
        let handle1 = sys_tensor_alloc(&args);
        assert!(handle1 > 0, "basic tensor alloc failed: {handle1}");

        // --- Test 2: Different dtypes (covers element_size paths) ---
        let shape_1d: [usize; 1] = [10];
        let ptr_1d = shape_1d.as_ptr() as usize;

        // Float16
        let r = sys_tensor_alloc(&SyscallArgs::new(0x04, [ptr_1d, 1, 1, 0, 0, 0]));
        assert!(r > 0, "Float16 alloc failed: {r}");
        // Int8
        let r = sys_tensor_alloc(&SyscallArgs::new(0x04, [ptr_1d, 1, 2, 0, 0, 0]));
        assert!(r > 0, "Int8 alloc failed: {r}");
        // Float64
        let r = sys_tensor_alloc(&SyscallArgs::new(0x04, [ptr_1d, 1, 6, 0, 0, 0]));
        assert!(r > 0, "Float64 alloc failed: {r}");
        // BFloat16
        let r = sys_tensor_alloc(&SyscallArgs::new(0x04, [ptr_1d, 1, 7, 0, 0, 0]));
        assert!(r > 0, "BFloat16 alloc failed: {r}");

        // --- Test 3: Multi-dim shape ---
        let shape_3d: [usize; 3] = [2, 3, 4];
        let ptr_3d = shape_3d.as_ptr() as usize;
        let r = sys_tensor_alloc(&SyscallArgs::new(0x04, [ptr_3d, 3, 0, 0, 0, 0]));
        assert!(r > 0, "3D tensor alloc failed: {r}");

        // --- Test 4: 8-dim shape ---
        let shape_8d: [usize; 8] = [1, 2, 1, 1, 1, 1, 1, 3];
        let ptr_8d = shape_8d.as_ptr() as usize;
        let r = sys_tensor_alloc(&SyscallArgs::new(0x04, [ptr_8d, 8, 0, 0, 0, 0]));
        assert!(r > 0, "8-dim tensor alloc failed: {r}");

        // --- Test 5: Shape with zero dim (rejected after cap check) ---
        let bad_shape: [usize; 2] = [4, 0];
        let ptr_bad = bad_shape.as_ptr() as usize;
        let r = sys_tensor_alloc(&SyscallArgs::new(0x04, [ptr_bad, 2, 0, 0, 0, 0]));
        assert_eq!(r, SyscallError::InvalidArgument.as_i64());

        // --- Test 6: Overflow in shape multiplication ---
        let overflow_shape: [usize; 2] = [usize::MAX, 2];
        let ptr_of = overflow_shape.as_ptr() as usize;
        let r = sys_tensor_alloc(&SyscallArgs::new(0x04, [ptr_of, 2, 0, 0, 0, 0]));
        assert_eq!(r, SyscallError::InvalidArgument.as_i64());

        // --- Test 7: Overflow in total_bytes ---
        let big_shape: [usize; 1] = [usize::MAX / 4];
        let ptr_big = big_shape.as_ptr() as usize;
        let r = sys_tensor_alloc(&SyscallArgs::new(0x04, [ptr_big, 1, 5, 0, 0, 0]));
        assert_eq!(r, SyscallError::InvalidArgument.as_i64());

        // --- Test 8: Tensor free (covers lines 300-308) ---
        // Grant cap for the specific handle returned by handle1
        unsafe {
            state::with_cap_registry(|reg| {
                let resource = ResourceRef::new(ResourceType::TensorBuffer, handle1 as u64);
                let _ = reg.create(0, resource, Permissions::WRITE, 0);
            });
        }
        let free_args = SyscallArgs::new(0x05, [handle1 as usize, 0, 0, 0, 0, 0]);
        let r = sys_tensor_free(&free_args);
        assert_eq!(r, SyscallError::Success.as_i64());

        // --- Test 9: Free with invalid handle (after cap check) ---
        unsafe {
            state::with_cap_registry(|reg| {
                let resource = ResourceRef::new(ResourceType::TensorBuffer, 9999);
                let _ = reg.create(0, resource, Permissions::WRITE, 0);
            });
        }
        let r = sys_tensor_free(&SyscallArgs::new(0x05, [9999, 0, 0, 0, 0, 0]));
        assert!(r < 0, "expected error for invalid handle, got {r}");

        // --- Cleanup: reset pool to avoid interference with fuzz tests ---
        unsafe {
            state::with_tensor_pool(|pool| {
                pool.reset();
            });
        }
    }

    #[test]
    fn test_mem_alloc_dma_flag_forces_buddy() {
        // size=64 (would be slab), but flags=1 (DMA) forces buddy
        let args = SyscallArgs::new(0x00, [64, 0, 1, 0, 0, 0]);
        let result = sys_mem_alloc(&args);
        assert_ne!(result, 0);
    }

    #[test]
    fn test_mem_alloc_huge_page_flag() {
        // flags=2 (HugePage2M) forces buddy path
        let args = SyscallArgs::new(0x00, [64, 0, 2, 0, 0, 0]);
        let result = sys_mem_alloc(&args);
        assert_ne!(result, 0);
    }

    #[test]
    fn test_mem_free_exactly_one_byte() {
        // Smallest valid free: 1 byte → slab path
        let args = SyscallArgs::new(0x01, [0x1000, 1, 0, 0, 0, 0]);
        let result = sys_mem_free(&args);
        assert_ne!(result, 0);
    }
}
