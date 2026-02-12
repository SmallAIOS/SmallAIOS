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
use core::sync::atomic::{AtomicU32, Ordering};

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
/// Maximum number of registered devices.
pub const MAX_DEVICES: usize = 32;

/// Maximum number of open device handles.
pub const MAX_HANDLES: usize = 64;

/// USB ioctl command codes for device syscall dispatch.
pub mod ioctl {
    /// Get device type (returns DeviceType as u64).
    pub const DEV_GET_TYPE: usize = 0x0001;
    /// Get device vendor ID.
    pub const DEV_GET_VID: usize = 0x0002;
    /// Get device product ID.
    pub const DEV_GET_PID: usize = 0x0003;
    /// USB: perform control transfer. arg = pointer to ControlTransferArgs.
    pub const USB_CONTROL_TRANSFER: usize = 0x0100;
    /// USB: perform bulk IN transfer. arg = pointer to BulkTransferArgs.
    pub const USB_BULK_IN: usize = 0x0101;
    /// USB: perform bulk OUT transfer. arg = pointer to BulkTransferArgs.
    pub const USB_BULK_OUT: usize = 0x0102;
    /// USB: configure endpoint. arg = endpoint address.
    pub const USB_CONFIGURE_EP: usize = 0x0103;
    /// USB: reset device.
    pub const USB_RESET: usize = 0x0104;
    /// SDR: set frequency. arg = frequency in Hz.
    pub const SDR_SET_FREQ: usize = 0x0200;
    /// SDR: set sample rate. arg = sample rate in Hz.
    pub const SDR_SET_SAMPLE_RATE: usize = 0x0201;
    /// SDR: set gain. arg = gain in dB.
    pub const SDR_SET_GAIN: usize = 0x0202;
    /// SDR: start RX streaming.
    pub const SDR_START_RX: usize = 0x0203;
    /// SDR: stop streaming.
    pub const SDR_STOP: usize = 0x0204;
    /// SDR: start TX streaming.
    pub const SDR_START_TX: usize = 0x0205;
}

/// Device type identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DeviceType {
    /// Unknown device type.
    Unknown = 0,
    /// USB host controller.
    UsbHost = 1,
    /// USB device (connected peripheral).
    UsbDevice = 2,
    /// USB gadget (local device controller).
    UsbGadget = 3,
    /// SDR receiver.
    SdrDevice = 4,
}

/// Device entry in the global device table.
#[derive(Debug, Clone, Copy)]
pub struct DeviceEntry {
    /// Device type.
    pub device_type: DeviceType,
    /// Vendor ID (USB VID or equivalent).
    pub vendor_id: u16,
    /// Product ID (USB PID or equivalent).
    pub product_id: u16,
    /// Device-specific identifier (e.g., USB address, slot number).
    pub device_id: u16,
    /// Whether this entry is occupied.
    pub active: bool,
}

impl DeviceEntry {
    const fn empty() -> Self {
        Self {
            device_type: DeviceType::Unknown,
            vendor_id: 0,
            product_id: 0,
            device_id: 0,
            active: false,
        }
    }
}

/// Global device table.
static mut DEVICE_TABLE: [DeviceEntry; MAX_DEVICES] = [DeviceEntry::empty(); MAX_DEVICES];

/// Device handle entry.
#[derive(Debug, Clone, Copy)]
struct HandleEntry {
    /// Index into DEVICE_TABLE.
    device_index: u16,
    /// Whether this handle is active.
    active: bool,
}

impl HandleEntry {
    const fn empty() -> Self {
        Self {
            device_index: 0,
            active: false,
        }
    }
}

/// Global handle table. Handle values are 1-indexed (0 = invalid).
static mut HANDLE_TABLE: [HandleEntry; MAX_HANDLES] = [HandleEntry::empty(); MAX_HANDLES];

/// Handle generation counter for uniqueness.
static HANDLE_GENERATION: AtomicU32 = AtomicU32::new(1);

/// Registers a device in the global device table.
/// Returns the device index on success, or None if the table is full.
pub fn register_device(entry: DeviceEntry) -> Option<usize> {
    // Safety: single-threaded kernel initialization
    unsafe {
        let table = &raw mut DEVICE_TABLE;
        for (i, slot) in (*table).iter_mut().enumerate() {
            if !slot.active {
                *slot = DeviceEntry {
                    active: true,
                    ..entry
                };
                return Some(i);
            }
        }
    }
    None
}

/// Unregisters a device from the global device table.
pub fn unregister_device(index: usize) -> bool {
    unsafe {
        let dtable = &raw mut DEVICE_TABLE;
        if index < MAX_DEVICES && (*dtable)[index].active {
            (*dtable)[index].active = false;
            // Close all handles pointing to this device
            let htable = &raw mut HANDLE_TABLE;
            for handle in (*htable).iter_mut() {
                if handle.active && handle.device_index as usize == index {
                    handle.active = false;
                }
            }
            return true;
        }
    }
    false
}

/// Returns the number of active devices.
pub fn device_count() -> usize {
    unsafe {
        let table = &raw const DEVICE_TABLE;
        (*table).iter().filter(|d| d.active).count()
    }
}

/// Returns a device entry by index.
pub fn get_device(index: usize) -> Option<DeviceEntry> {
    unsafe {
        let dtable = &raw const DEVICE_TABLE;
        if index < MAX_DEVICES && (*dtable)[index].active {
            Some((*dtable)[index])
        } else {
            None
        }
    }
}

/// Enumerate available devices.
///
/// Args: [buf_ptr, buf_len, 0, 0, 0, 0]
/// Returns: number of active devices when buf_ptr=0 (query mode),
///          or bytes written on success.
pub fn sys_dev_enumerate(args: &SyscallArgs) -> SyscallResult {
    let buf_ptr = args.args[0];
    let buf_len = args.args[1];

    // Query mode: return number of active devices
    if buf_ptr == 0 && buf_len == 0 {
        return device_count() as i64;
    }

    if buf_ptr == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // Validate buffer pointer
    if buf_len > isize::MAX as usize {
        return SyscallError::InvalidArgument.as_i64();
    }

    // Each device entry serializes as 8 bytes:
    // [type:1, vid_lo:1, vid_hi:1, pid_lo:1, pid_hi:1, dev_id_lo:1, dev_id_hi:1, reserved:1]
    let entry_size = 8usize;
    let mut written = 0usize;

    unsafe {
        let buf = core::slice::from_raw_parts_mut(buf_ptr as *mut u8, buf_len);
        let dtable = &raw const DEVICE_TABLE;
        for dev in (*dtable).iter() {
            if !dev.active {
                continue;
            }
            if written + entry_size > buf_len {
                break;
            }
            buf[written] = dev.device_type as u8;
            buf[written + 1] = dev.vendor_id as u8;
            buf[written + 2] = (dev.vendor_id >> 8) as u8;
            buf[written + 3] = dev.product_id as u8;
            buf[written + 4] = (dev.product_id >> 8) as u8;
            buf[written + 5] = dev.device_id as u8;
            buf[written + 6] = (dev.device_id >> 8) as u8;
            buf[written + 7] = 0; // reserved
            written += entry_size;
        }
    }

    written as i64
}

/// Open a device by index.
///
/// Args: [device_index, 0, 0, 0, 0, 0]
/// Returns: device handle (1-indexed) on success, negative error code on failure.
pub fn sys_dev_open(args: &SyscallArgs) -> SyscallResult {
    let device_index = args.args[0];

    // Validate device exists
    if device_index >= MAX_DEVICES {
        return SyscallError::NotFound.as_i64();
    }

    unsafe {
        let dtable = &raw const DEVICE_TABLE;
        if !(*dtable)[device_index].active {
            return SyscallError::NotFound.as_i64();
        }

        // Allocate a handle
        let htable = &raw mut HANDLE_TABLE;
        for (i, handle) in (*htable).iter_mut().enumerate() {
            if !handle.active {
                handle.device_index = device_index as u16;
                handle.active = true;
                let _ = HANDLE_GENERATION.fetch_add(1, Ordering::Relaxed);
                return (i + 1) as i64; // 1-indexed handle
            }
        }
    }

    SyscallError::Busy.as_i64() // No free handles
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

    let idx = handle - 1;
    if idx >= MAX_HANDLES {
        return SyscallError::InvalidHandle.as_i64();
    }

    unsafe {
        let htable = &raw mut HANDLE_TABLE;
        if !(*htable)[idx].active {
            return SyscallError::InvalidHandle.as_i64();
        }
        (*htable)[idx].active = false;
    }

    SyscallError::Success.as_i64()
}

/// Device I/O control.
///
/// Args: [handle, cmd, arg, 0, 0, 0]
/// Returns: command-specific result, negative error code on failure.
pub fn sys_dev_ioctl(args: &SyscallArgs) -> SyscallResult {
    let handle = args.args[0];
    let cmd = args.args[1];
    let _arg = args.args[2];

    if handle == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }

    let idx = handle - 1;
    if idx >= MAX_HANDLES {
        return SyscallError::InvalidHandle.as_i64();
    }

    let device_entry = unsafe {
        let htable = &raw const HANDLE_TABLE;
        let dtable = &raw const DEVICE_TABLE;
        if !(*htable)[idx].active {
            return SyscallError::InvalidHandle.as_i64();
        }
        let dev_idx = (*htable)[idx].device_index as usize;
        if dev_idx >= MAX_DEVICES || !(*dtable)[dev_idx].active {
            return SyscallError::DeviceError.as_i64();
        }
        (*dtable)[dev_idx]
    };

    // Dispatch based on command
    match cmd {
        ioctl::DEV_GET_TYPE => device_entry.device_type as u8 as i64,
        ioctl::DEV_GET_VID => device_entry.vendor_id as i64,
        ioctl::DEV_GET_PID => device_entry.product_id as i64,

        // USB commands — require USB device type
        ioctl::USB_CONTROL_TRANSFER
        | ioctl::USB_BULK_IN
        | ioctl::USB_BULK_OUT
        | ioctl::USB_CONFIGURE_EP
        | ioctl::USB_RESET => {
            match device_entry.device_type {
                DeviceType::UsbHost | DeviceType::UsbDevice | DeviceType::UsbGadget => {
                    // USB operations dispatched via UsbHostController/UsbDeviceController HAL trait
                    // Actual transfer implementation requires a registered HAL driver
                    SyscallError::NotSupported.as_i64()
                }
                _ => SyscallError::InvalidArgument.as_i64(),
            }
        }

        // SDR commands — require SDR device type
        ioctl::SDR_SET_FREQ
        | ioctl::SDR_SET_SAMPLE_RATE
        | ioctl::SDR_SET_GAIN
        | ioctl::SDR_START_RX
        | ioctl::SDR_STOP
        | ioctl::SDR_START_TX => {
            if device_entry.device_type != DeviceType::SdrDevice {
                return SyscallError::InvalidArgument.as_i64();
            }
            // SDR operations dispatched to registered SDR driver
            SyscallError::NotSupported.as_i64()
        }

        _ => SyscallError::InvalidArgument.as_i64(),
    }
}

/// Allocate a DMA-capable buffer.
///
/// Args: [size, align, 0, 0, 0, 0]
/// Returns: DMA buffer physical address on success, negative error code on failure.
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

    // Default alignment to page size (4096) if not specified
    let _effective_align = if align == 0 { 4096 } else { align };

    // DMA allocation requires platform-specific implementation
    // via the DMA allocator in the arch crate
    SyscallError::NotSupported.as_i64()
}

/// Resets device and handle tables (for testing).
#[cfg(test)]
pub fn reset_tables() {
    unsafe {
        let dtable = &raw mut DEVICE_TABLE;
        for d in (*dtable).iter_mut() {
            *d = DeviceEntry::empty();
        }
        let htable = &raw mut HANDLE_TABLE;
        for h in (*htable).iter_mut() {
            *h = HandleEntry::empty();
        }
    }
    HANDLE_GENERATION.store(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    extern crate std;
    use std::sync::Mutex;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn setup() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK
            .lock()
            .unwrap_or_else(|e: std::sync::PoisonError<_>| e.into_inner());
        reset_tables();
        guard
    }

    // ── Existing tests ──

    #[test]
    fn test_dev_enumerate_query_mode() {
        let _guard = setup();
        let args = SyscallArgs::new(0x40, [0, 0, 0, 0, 0, 0]);
        // No devices registered → returns 0
        assert_eq!(sys_dev_enumerate(&args), 0);
    }

    #[test]
    fn test_dev_enumerate_null_nonzero() {
        let _guard = setup();
        let args = SyscallArgs::new(0x40, [0, 256, 0, 0, 0, 0]);
        assert_eq!(
            sys_dev_enumerate(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_dev_open_not_found() {
        let _guard = setup();
        let args = SyscallArgs::new(0x41, [0, 0, 0, 0, 0, 0]);
        assert_eq!(sys_dev_open(&args), SyscallError::NotFound.as_i64());
    }

    #[test]
    fn test_dev_close_zero_handle() {
        let _guard = setup();
        let args = SyscallArgs::new(0x42, [0, 0, 0, 0, 0, 0]);
        assert_eq!(sys_dev_close(&args), SyscallError::InvalidHandle.as_i64());
    }

    #[test]
    fn test_dev_ioctl_zero_handle() {
        let _guard = setup();
        let args = SyscallArgs::new(0x43, [0, 1, 0, 0, 0, 0]);
        assert_eq!(sys_dev_ioctl(&args), SyscallError::InvalidHandle.as_i64());
    }

    #[test]
    fn test_dev_dma_alloc_zero_size() {
        let _guard = setup();
        let args = SyscallArgs::new(0x44, [0, 4096, 0, 0, 0, 0]);
        assert_eq!(
            sys_dev_dma_alloc(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_dev_dma_alloc_too_large() {
        let _guard = setup();
        let args = SyscallArgs::new(0x44, [MAX_DMA_SIZE + 1, 4096, 0, 0, 0, 0]);
        assert_eq!(
            sys_dev_dma_alloc(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_dev_dma_alloc_bad_alignment() {
        let _guard = setup();
        let args = SyscallArgs::new(0x44, [4096, 3, 0, 0, 0, 0]);
        assert_eq!(
            sys_dev_dma_alloc(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_dev_dma_alloc_valid() {
        let _guard = setup();
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

    // ── USB device management tests ──

    #[test]
    fn test_register_device() {
        let _guard = setup();
        let entry = DeviceEntry {
            device_type: DeviceType::UsbDevice,
            vendor_id: 0x1D50,
            product_id: 0x6089,
            device_id: 1,
            active: true,
        };
        let idx = register_device(entry);
        assert!(idx.is_some());
        assert_eq!(device_count(), 1);

        let dev = get_device(idx.unwrap()).unwrap();
        assert_eq!(dev.vendor_id, 0x1D50);
        assert_eq!(dev.product_id, 0x6089);
        assert_eq!(dev.device_type, DeviceType::UsbDevice);
    }

    #[test]
    fn test_unregister_device() {
        let _guard = setup();
        let entry = DeviceEntry {
            device_type: DeviceType::SdrDevice,
            vendor_id: 0x0456,
            product_id: 0xB673,
            device_id: 2,
            active: true,
        };
        let idx = register_device(entry).unwrap();
        assert_eq!(device_count(), 1);

        assert!(unregister_device(idx));
        assert_eq!(device_count(), 0);
        assert!(get_device(idx).is_none());
    }

    #[test]
    fn test_enumerate_with_usb_devices() {
        let _guard = setup();
        // Register two USB devices
        register_device(DeviceEntry {
            device_type: DeviceType::UsbDevice,
            vendor_id: 0x1D50,
            product_id: 0x6089,
            device_id: 1,
            active: true,
        });
        register_device(DeviceEntry {
            device_type: DeviceType::SdrDevice,
            vendor_id: 0x0456,
            product_id: 0xB673,
            device_id: 2,
            active: true,
        });

        // Query count
        let args = SyscallArgs::new(0x40, [0, 0, 0, 0, 0, 0]);
        assert_eq!(sys_dev_enumerate(&args), 2);

        // Enumerate into buffer
        let mut buf = [0u8; 64];
        let args = SyscallArgs::new(0x40, [buf.as_mut_ptr() as usize, buf.len(), 0, 0, 0, 0]);
        let written = sys_dev_enumerate(&args);
        assert_eq!(written, 16); // 2 devices * 8 bytes each

        // Check first device
        assert_eq!(buf[0], DeviceType::UsbDevice as u8);
        let vid = u16::from_le_bytes([buf[1], buf[2]]);
        let pid = u16::from_le_bytes([buf[3], buf[4]]);
        assert_eq!(vid, 0x1D50);
        assert_eq!(pid, 0x6089);

        // Check second device
        assert_eq!(buf[8], DeviceType::SdrDevice as u8);
        let vid = u16::from_le_bytes([buf[9], buf[10]]);
        let pid = u16::from_le_bytes([buf[11], buf[12]]);
        assert_eq!(vid, 0x0456);
        assert_eq!(pid, 0xB673);
    }

    #[test]
    fn test_open_close_device() {
        let _guard = setup();
        register_device(DeviceEntry {
            device_type: DeviceType::UsbDevice,
            vendor_id: 0x1D50,
            product_id: 0x6089,
            device_id: 1,
            active: true,
        });

        // Open device at index 0
        let args = SyscallArgs::new(0x41, [0, 0, 0, 0, 0, 0]);
        let handle = sys_dev_open(&args);
        assert!(handle > 0, "Expected positive handle, got {}", handle);

        // Close the handle
        let args = SyscallArgs::new(0x42, [handle as usize, 0, 0, 0, 0, 0]);
        assert_eq!(sys_dev_close(&args), SyscallError::Success.as_i64());

        // Double-close should fail
        let args = SyscallArgs::new(0x42, [handle as usize, 0, 0, 0, 0, 0]);
        assert_eq!(sys_dev_close(&args), SyscallError::InvalidHandle.as_i64());
    }

    #[test]
    fn test_ioctl_get_device_info() {
        let _guard = setup();
        register_device(DeviceEntry {
            device_type: DeviceType::UsbDevice,
            vendor_id: 0x1D50,
            product_id: 0x6089,
            device_id: 1,
            active: true,
        });

        let open_args = SyscallArgs::new(0x41, [0, 0, 0, 0, 0, 0]);
        let handle = sys_dev_open(&open_args) as usize;

        // Get device type
        let args = SyscallArgs::new(0x43, [handle, ioctl::DEV_GET_TYPE, 0, 0, 0, 0]);
        assert_eq!(sys_dev_ioctl(&args), DeviceType::UsbDevice as u8 as i64);

        // Get VID
        let args = SyscallArgs::new(0x43, [handle, ioctl::DEV_GET_VID, 0, 0, 0, 0]);
        assert_eq!(sys_dev_ioctl(&args), 0x1D50);

        // Get PID
        let args = SyscallArgs::new(0x43, [handle, ioctl::DEV_GET_PID, 0, 0, 0, 0]);
        assert_eq!(sys_dev_ioctl(&args), 0x6089);
    }

    #[test]
    fn test_ioctl_usb_commands() {
        let _guard = setup();
        register_device(DeviceEntry {
            device_type: DeviceType::UsbDevice,
            vendor_id: 0x1D50,
            product_id: 0x6089,
            device_id: 1,
            active: true,
        });

        let open_args = SyscallArgs::new(0x41, [0, 0, 0, 0, 0, 0]);
        let handle = sys_dev_open(&open_args) as usize;

        // USB control transfer (returns NotSupported since no HAL driver registered)
        let args = SyscallArgs::new(0x43, [handle, ioctl::USB_CONTROL_TRANSFER, 0, 0, 0, 0]);
        assert_eq!(sys_dev_ioctl(&args), SyscallError::NotSupported.as_i64());

        // USB bulk IN
        let args = SyscallArgs::new(0x43, [handle, ioctl::USB_BULK_IN, 0, 0, 0, 0]);
        assert_eq!(sys_dev_ioctl(&args), SyscallError::NotSupported.as_i64());

        // USB reset
        let args = SyscallArgs::new(0x43, [handle, ioctl::USB_RESET, 0, 0, 0, 0]);
        assert_eq!(sys_dev_ioctl(&args), SyscallError::NotSupported.as_i64());
    }

    #[test]
    fn test_ioctl_sdr_on_usb_device_rejected() {
        let _guard = setup();
        register_device(DeviceEntry {
            device_type: DeviceType::UsbDevice,
            vendor_id: 0x1D50,
            product_id: 0x6089,
            device_id: 1,
            active: true,
        });

        let open_args = SyscallArgs::new(0x41, [0, 0, 0, 0, 0, 0]);
        let handle = sys_dev_open(&open_args) as usize;

        // SDR commands on a non-SDR device should be rejected
        let args = SyscallArgs::new(0x43, [handle, ioctl::SDR_SET_FREQ, 0, 0, 0, 0]);
        assert_eq!(sys_dev_ioctl(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_ioctl_sdr_commands() {
        let _guard = setup();
        register_device(DeviceEntry {
            device_type: DeviceType::SdrDevice,
            vendor_id: 0x0456,
            product_id: 0xB673,
            device_id: 2,
            active: true,
        });

        let open_args = SyscallArgs::new(0x41, [0, 0, 0, 0, 0, 0]);
        let handle = sys_dev_open(&open_args) as usize;

        // SDR commands on SDR device (returns NotSupported since no driver registered)
        let args = SyscallArgs::new(0x43, [handle, ioctl::SDR_SET_FREQ, 2_400_000_000, 0, 0, 0]);
        assert_eq!(sys_dev_ioctl(&args), SyscallError::NotSupported.as_i64());

        let args = SyscallArgs::new(0x43, [handle, ioctl::SDR_START_RX, 0, 0, 0, 0]);
        assert_eq!(sys_dev_ioctl(&args), SyscallError::NotSupported.as_i64());
    }

    #[test]
    fn test_ioctl_unknown_command() {
        let _guard = setup();
        register_device(DeviceEntry {
            device_type: DeviceType::UsbDevice,
            vendor_id: 0x1D50,
            product_id: 0x6089,
            device_id: 1,
            active: true,
        });

        let open_args = SyscallArgs::new(0x41, [0, 0, 0, 0, 0, 0]);
        let handle = sys_dev_open(&open_args) as usize;

        let args = SyscallArgs::new(0x43, [handle, 0xFFFF, 0, 0, 0, 0]);
        assert_eq!(sys_dev_ioctl(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_unregister_closes_handles() {
        let _guard = setup();
        let idx = register_device(DeviceEntry {
            device_type: DeviceType::UsbDevice,
            vendor_id: 0x1D50,
            product_id: 0x6089,
            device_id: 1,
            active: true,
        })
        .unwrap();

        let open_args = SyscallArgs::new(0x41, [0, 0, 0, 0, 0, 0]);
        let handle = sys_dev_open(&open_args) as usize;

        // Unregister device should close all associated handles
        unregister_device(idx);

        // Handle should now be invalid
        let args = SyscallArgs::new(0x43, [handle, ioctl::DEV_GET_TYPE, 0, 0, 0, 0]);
        assert_eq!(sys_dev_ioctl(&args), SyscallError::InvalidHandle.as_i64());
    }

    #[test]
    fn test_dev_dma_alloc_default_alignment() {
        let _guard = setup();
        // align=0 should work (defaults to page size)
        let args = SyscallArgs::new(0x44, [4096, 0, 0, 0, 0, 0]);
        assert_eq!(
            sys_dev_dma_alloc(&args),
            SyscallError::NotSupported.as_i64()
        );
    }

    #[test]
    fn test_open_nonexistent_device_index() {
        let _guard = setup();
        let args = SyscallArgs::new(0x41, [MAX_DEVICES + 1, 0, 0, 0, 0, 0]);
        assert_eq!(sys_dev_open(&args), SyscallError::NotFound.as_i64());
    }
}
