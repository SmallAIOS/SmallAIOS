// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Device syscalls (0x40-0x4F).
//!
//! - `dev_enumerate() -> DeviceList`
//! - `dev_open(id) -> DevHandle`
//! - `dev_close(handle)`
//! - `dev_ioctl(handle, cmd, arg) -> isize`
//! - `dev_dma_alloc(size, align) -> DmaBuffer`

use super::{SyscallArgs, SyscallError, SyscallResult};

/// Maximum DMA buffer size: 256 MiB.
pub const MAX_DMA_SIZE: usize = 256 * 1024 * 1024;

/// Enumerate available devices.
///
/// Args: [buf_ptr, buf_len, 0, 0, 0, 0]
/// Returns: bytes written on success, negative error code on failure.
pub fn sys_dev_enumerate(args: &SyscallArgs) -> SyscallResult {
    let buf_ptr = args.args[0];
    let buf_len = args.args[1];

    // Allow zero buf_ptr to query required size
    if buf_ptr == 0 && buf_len == 0 {
        // TODO: Return required buffer size
        return SyscallError::NotSupported.as_i64();
    }

    if buf_ptr == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with device HAL
    let _ = (buf_ptr, buf_len);
    SyscallError::NotSupported.as_i64()
}

/// Open a device by ID.
///
/// Args: [device_id, 0, 0, 0, 0, 0]
/// Returns: device handle on success, negative error code on failure.
pub fn sys_dev_open(args: &SyscallArgs) -> SyscallResult {
    let device_id = args.args[0];

    // device_id 0 is valid (first device), so no validation needed here.
    // The device subsystem will return NotFound for invalid IDs.
    let _ = device_id;
    SyscallError::NotSupported.as_i64()
}

/// Close a device handle.
///
/// Args: [handle, 0, 0, 0, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_dev_close(args: &SyscallArgs) -> SyscallResult {
    let handle = args.args[0];

    if handle == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }

    // TODO: Integrate with device HAL
    let _ = handle;
    SyscallError::NotSupported.as_i64()
}

/// Device I/O control.
///
/// Args: [handle, cmd, arg, 0, 0, 0]
/// Returns: command-specific result, negative error code on failure.
pub fn sys_dev_ioctl(args: &SyscallArgs) -> SyscallResult {
    let handle = args.args[0];
    let cmd = args.args[1];
    let arg = args.args[2];

    if handle == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }

    // TODO: Integrate with device HAL — dispatch to device driver
    let _ = (handle, cmd, arg);
    SyscallError::NotSupported.as_i64()
}

/// Allocate a DMA-capable buffer.
///
/// Args: [size, align, 0, 0, 0, 0]
/// Returns: DMA buffer descriptor on success, negative error code on failure.
pub fn sys_dev_dma_alloc(args: &SyscallArgs) -> SyscallResult {
    let size = args.args[0];
    let align = args.args[1];

    if size == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if size > MAX_DMA_SIZE {
        return SyscallError::InvalidArgument.as_i64();
    }
    if align != 0 && !align.is_power_of_two() {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with DMA-capable allocator
    let _ = (size, align);
    SyscallError::NotSupported.as_i64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dev_enumerate_query_mode() {
        let args = SyscallArgs::new(0x40, [0, 0, 0, 0, 0, 0]);
        assert_eq!(
            sys_dev_enumerate(&args),
            SyscallError::NotSupported.as_i64()
        );
    }

    #[test]
    fn test_dev_enumerate_null_nonzero() {
        let args = SyscallArgs::new(0x40, [0, 256, 0, 0, 0, 0]);
        assert_eq!(
            sys_dev_enumerate(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_dev_open_valid() {
        let args = SyscallArgs::new(0x41, [0, 0, 0, 0, 0, 0]);
        assert_eq!(sys_dev_open(&args), SyscallError::NotSupported.as_i64());
    }

    #[test]
    fn test_dev_close_zero_handle() {
        let args = SyscallArgs::new(0x42, [0, 0, 0, 0, 0, 0]);
        assert_eq!(sys_dev_close(&args), SyscallError::InvalidHandle.as_i64());
    }

    #[test]
    fn test_dev_ioctl_zero_handle() {
        let args = SyscallArgs::new(0x43, [0, 1, 0, 0, 0, 0]);
        assert_eq!(sys_dev_ioctl(&args), SyscallError::InvalidHandle.as_i64());
    }

    #[test]
    fn test_dev_dma_alloc_zero_size() {
        let args = SyscallArgs::new(0x44, [0, 4096, 0, 0, 0, 0]);
        assert_eq!(
            sys_dev_dma_alloc(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_dev_dma_alloc_too_large() {
        let args = SyscallArgs::new(0x44, [MAX_DMA_SIZE + 1, 4096, 0, 0, 0, 0]);
        assert_eq!(
            sys_dev_dma_alloc(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_dev_dma_alloc_bad_alignment() {
        let args = SyscallArgs::new(0x44, [4096, 3, 0, 0, 0, 0]);
        assert_eq!(
            sys_dev_dma_alloc(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_dev_dma_alloc_valid() {
        let args = SyscallArgs::new(0x44, [4096, 4096, 0, 0, 0, 0]);
        assert_eq!(
            sys_dev_dma_alloc(&args),
            SyscallError::NotSupported.as_i64()
        );
    }
}
