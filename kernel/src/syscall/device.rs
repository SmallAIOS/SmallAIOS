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

// ---------------------------------------------------------------------------
// Peripheral ioctl command constants
// ---------------------------------------------------------------------------

// I2C ioctl commands (0x100-0x10F)
/// Configure I2C bus speed, addressing mode, etc.
pub const I2C_SET_CONFIG: usize = 0x100;
/// Write bytes to an I2C target device.
pub const I2C_WRITE: usize = 0x101;
/// Read bytes from an I2C target device.
pub const I2C_READ: usize = 0x102;
/// Write then read in a single I2C transaction (repeated START).
pub const I2C_WRITE_READ: usize = 0x103;
/// Attempt I2C bus recovery.
pub const I2C_BUS_RECOVERY: usize = 0x104;

// SPI ioctl commands (0x110-0x11F)
/// Configure SPI mode, clock, word size.
pub const SPI_SET_CONFIG: usize = 0x110;
/// Full-duplex SPI transfer.
pub const SPI_TRANSFER: usize = 0x111;
/// SPI write-only transfer.
pub const SPI_WRITE: usize = 0x112;
/// SPI read-only transfer.
pub const SPI_READ: usize = 0x113;
/// SPI DMA transfer.
pub const SPI_TRANSFER_DMA: usize = 0x114;

// GPIO ioctl commands (0x120-0x12F)
/// Configure a GPIO pin.
pub const GPIO_CONFIGURE: usize = 0x120;
/// Read a GPIO pin level.
pub const GPIO_READ: usize = 0x121;
/// Write a GPIO pin level.
pub const GPIO_WRITE: usize = 0x122;
/// Atomic multi-pin set/clear via bitmask.
pub const GPIO_SET_MASK: usize = 0x123;
/// Read all GPIO pins as bitmask.
pub const GPIO_READ_ALL: usize = 0x124;
/// Enable GPIO pin interrupt.
pub const GPIO_ENABLE_IRQ: usize = 0x125;
/// Disable GPIO pin interrupt.
pub const GPIO_DISABLE_IRQ: usize = 0x126;

// UART ioctl commands (0x130-0x13F)
/// Configure UART baud rate, framing, flow control.
pub const UART_SET_CONFIG: usize = 0x130;
/// Write bytes to UART.
pub const UART_WRITE: usize = 0x131;
/// Read bytes from UART.
pub const UART_READ: usize = 0x132;
/// Read until newline (NMEA sentence).
pub const UART_READ_LINE: usize = 0x133;
/// Flush UART TX FIFO.
pub const UART_FLUSH: usize = 0x134;
/// Query available RX bytes.
pub const UART_RX_AVAILABLE: usize = 0x135;

// CSI Camera ioctl commands (0x140-0x14F)
/// Configure CSI resolution, format, frame rate.
pub const CSI_SET_CONFIG: usize = 0x140;
/// Start continuous capture.
pub const CSI_START_CAPTURE: usize = 0x141;
/// Stop capture.
pub const CSI_STOP_CAPTURE: usize = 0x142;
/// Dequeue a completed frame.
pub const CSI_DEQUEUE_FRAME: usize = 0x143;
/// Return a frame buffer to the capture queue.
pub const CSI_ENQUEUE_FRAME: usize = 0x144;
/// Get physical address of a frame buffer (zero-copy ONNX).
pub const CSI_GET_BUFFER_ADDR: usize = 0x145;

// I2S Audio ioctl commands (0x150-0x15F)
/// Configure I2S sample rate, format, channels.
pub const I2S_SET_CONFIG: usize = 0x150;
/// Start audio capture.
pub const I2S_START_CAPTURE: usize = 0x151;
/// Stop audio capture.
pub const I2S_STOP_CAPTURE: usize = 0x152;
/// Dequeue a completed audio buffer.
pub const I2S_DEQUEUE_BUFFER: usize = 0x153;
/// Return an audio buffer to the capture ring.
pub const I2S_ENQUEUE_BUFFER: usize = 0x154;
/// Get physical address of an audio buffer (zero-copy ONNX).
pub const I2S_GET_BUFFER_ADDR: usize = 0x155;

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

    // -- Peripheral ioctl constants -----------------------------------------

    #[test]
    fn test_ioctl_constants_no_overlap() {
        // Verify no two peripheral ioctl command ranges overlap
        let ranges = [
            (I2C_SET_CONFIG, I2C_BUS_RECOVERY, "I2C"),
            (SPI_SET_CONFIG, SPI_TRANSFER_DMA, "SPI"),
            (GPIO_CONFIGURE, GPIO_DISABLE_IRQ, "GPIO"),
            (UART_SET_CONFIG, UART_RX_AVAILABLE, "UART"),
            (CSI_SET_CONFIG, CSI_GET_BUFFER_ADDR, "CSI"),
            (I2S_SET_CONFIG, I2S_GET_BUFFER_ADDR, "I2S"),
        ];
        for i in 0..ranges.len() {
            for j in (i + 1)..ranges.len() {
                // Ranges must not overlap
                assert!(
                    ranges[i].1 < ranges[j].0 || ranges[j].1 < ranges[i].0,
                    "{} and {} ioctl ranges overlap",
                    ranges[i].2,
                    ranges[j].2
                );
            }
        }
    }

    #[test]
    fn test_ioctl_constant_values() {
        assert_eq!(I2C_SET_CONFIG, 0x100);
        assert_eq!(SPI_SET_CONFIG, 0x110);
        assert_eq!(GPIO_CONFIGURE, 0x120);
        assert_eq!(UART_SET_CONFIG, 0x130);
        assert_eq!(CSI_SET_CONFIG, 0x140);
        assert_eq!(I2S_SET_CONFIG, 0x150);
    }
}
