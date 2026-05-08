// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! System syscalls (0x50-0x5F).
//!
//! - `sys_info() -> SystemInfo`
//! - `sys_time() -> u64` (nanoseconds since boot)
//! - `sys_shutdown(code)`
//! - `sys_log(level, msg, len)`
//! - `sys_random(buf, len)` — CSPRNG
//! - `sys_watchdog_pet()` — service hardware watchdog
//! - `sys_watchdog_remaining() -> u32` — query remaining watchdog time

use super::{SyscallArgs, SyscallError, SyscallResult};
use crate::state;
use crate::syscall::task::current_task_id;
use smallaios_security::capability::{Permissions, ResourceRef, ResourceType};

/// Log levels for sys_log.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum LogLevel {
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
    Trace = 4,
}

impl LogLevel {
    /// Try to convert from raw u32.
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Error),
            1 => Some(Self::Warn),
            2 => Some(Self::Info),
            3 => Some(Self::Debug),
            4 => Some(Self::Trace),
            _ => None,
        }
    }
}

/// Maximum log message length: 4096 bytes.
pub const MAX_LOG_MSG_LEN: usize = 4096;

/// Maximum random buffer size per call: 256 bytes.
pub const MAX_RANDOM_LEN: usize = 256;

/// System information structure returned by sys_info.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SystemInfo {
    /// Kernel version major.
    pub version_major: u16,
    /// Kernel version minor.
    pub version_minor: u16,
    /// Kernel version patch.
    pub version_patch: u16,
    /// Number of CPU cores available.
    pub cpu_count: u16,
    /// Total physical memory in bytes.
    pub total_memory: u64,
    /// Free physical memory in bytes.
    pub free_memory: u64,
    /// Architecture identifier (0=x86_64, 1=aarch64).
    pub arch: u32,
    /// Kernel state initialized flag.
    pub initialized: u32,
}

/// Get system information.
///
/// Args: [buf_ptr, buf_len, 0, 0, 0, 0]
/// Returns: 0 on success (info written to buffer), negative error code on failure.
///
/// When buf_ptr is 0, returns the required buffer size as a positive value.
pub fn sys_info(args: &SyscallArgs) -> SyscallResult {
    let buf_ptr = args.args[0];
    let buf_len = args.args[1];

    let info_size = core::mem::size_of::<SystemInfo>();

    if buf_ptr == 0 {
        // Return required buffer size
        return info_size as i64;
    }

    if buf_len < info_size {
        return SyscallError::BufferTooSmall.as_i64();
    }

    // Validate pointer alignment for SystemInfo (requires 8-byte alignment)
    if !buf_ptr.is_multiple_of(core::mem::align_of::<SystemInfo>()) {
        return SyscallError::BadAddress.as_i64();
    }

    let info = SystemInfo {
        version_major: 0,
        version_minor: 1,
        version_patch: 0,
        cpu_count: 1,    // Unikernel — single core
        total_memory: 0, // Will be filled when buddy allocator reports stats
        free_memory: 0,
        #[cfg(target_arch = "x86_64")]
        arch: 0,
        #[cfg(target_arch = "aarch64")]
        arch: 1,
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        arch: 0xFF,
        initialized: if state::is_initialized() { 1 } else { 0 },
    };

    // SAFETY: In unikernel mode, the caller is in the same address space.
    unsafe {
        let dst = buf_ptr as *mut SystemInfo;
        core::ptr::write(dst, info);
    }

    SyscallError::Success.as_i64()
}

/// Get nanoseconds since boot.
///
/// Args: [0, 0, 0, 0, 0, 0]
/// Returns: nanoseconds since boot (positive value).
pub fn sys_time(_args: &SyscallArgs) -> SyscallResult {
    // Read the architecture high-resolution timer. On x86-64 this returns
    // TSC cycles, on ARM64 CNTPCT_EL0 ticks, on RISC-V CLINT mtime. On host
    // test builds this is a monotonic tick counter. The unit (cycles vs ns)
    // depends on whether TIMER_FREQ_HZ has been calibrated — callers that
    // need nanoseconds should multiply by the calibrated frequency.
    let now = crate::sched::timer::Timestamp::now();
    // Clamp to the positive i64 range so it encodes as a success value in
    // the syscall ABI (negative values are reserved for errors).
    (now.as_u64() & (i64::MAX as u64)) as SyscallResult
}

/// Shut down the system.
///
/// Args: [exit_code, 0, 0, 0, 0, 0]
/// Returns: does not return on success, negative error code on failure.
pub fn sys_shutdown(args: &SyscallArgs) -> SyscallResult {
    let _exit_code = args.args[0];

    // Shutdown is a privileged operation — require SystemControl:EXECUTE capability.
    let resource = ResourceRef::new(ResourceType::SystemControl, 0);
    if let Err(e) = state::check_capability(current_task_id(), &resource, Permissions::EXECUTE) {
        return e;
    }

    // TODO: Trigger graceful shutdown:
    // 1. Stop accepting new inference requests
    // 2. Wait for in-progress inferences (with timeout)
    // 3. Flush audit logs
    // 4. Zeroize crypto keys
    // 5. ACPI power off or QEMU exit

    SyscallError::Success.as_i64()
}

/// Write a log message.
///
/// Args: [level, msg_ptr, msg_len, 0, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_log(args: &SyscallArgs) -> SyscallResult {
    let level_raw = args.args[0] as u32;
    let msg_ptr = args.args[1];
    let msg_len = args.args[2];

    if LogLevel::from_u32(level_raw).is_none() {
        return SyscallError::InvalidArgument.as_i64();
    }
    if msg_ptr == 0 || msg_len == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if msg_len > MAX_LOG_MSG_LEN {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Write to kernel ring buffer, publish via IPC
    let _ = (level_raw, msg_ptr, msg_len);
    SyscallError::Success.as_i64()
}

/// Fill buffer with cryptographically secure random bytes.
///
/// Args: [buf_ptr, buf_len, 0, 0, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_random(args: &SyscallArgs) -> SyscallResult {
    let buf_ptr = args.args[0];
    let buf_len = args.args[1];

    if buf_ptr == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if buf_len == 0 || buf_len > MAX_RANDOM_LEN {
        return SyscallError::InvalidArgument.as_i64();
    }

    // SAFETY: In unikernel mode, the caller is in the same address space.
    // Syscall handlers run with interrupts masked, so CSPRNG access is exclusive.
    let buf = unsafe { core::slice::from_raw_parts_mut(buf_ptr as *mut u8, buf_len) };
    let result = unsafe { state::csprng_generate(buf) };
    match result {
        Ok(()) => SyscallError::Success.as_i64(),
        Err(_) => SyscallError::NotSupported.as_i64(),
    }
}

/// Service the hardware watchdog timer.
///
/// Args: [0, 0, 0, 0, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_watchdog_pet(_args: &SyscallArgs) -> SyscallResult {
    // TODO: Write to hardware watchdog service register
    SyscallError::Success.as_i64()
}

/// Query remaining watchdog time (in milliseconds).
///
/// Args: [0, 0, 0, 0, 0, 0]
/// Returns: remaining time in ms on success, negative error code on failure.
pub fn sys_watchdog_remaining(_args: &SyscallArgs) -> SyscallResult {
    // TODO: Read from hardware watchdog count register
    // For now, return max (30000ms = 30s default timeout)
    30000
}

/// `boot_success() -> 0 | -errno`
///
/// Wave-0 stub. Root-only commit of an A/B boot slot. The real handler
/// lands in `embedded-filesystem-v1` Phase 10 — until then this returns
/// `-ENOSYS`. Routed here so the dispatch table has a stable entry
/// point at 0x57.
pub fn sys_boot_success(_args: &SyscallArgs) -> SyscallResult {
    -38 // -ENOSYS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sys_info_null_ptr_returns_size() {
        let args = SyscallArgs::new(0x50, [0, 0, 0, 0, 0, 0]);
        let result = sys_info(&args);
        let expected_size = core::mem::size_of::<SystemInfo>() as i64;
        assert_eq!(result, expected_size);
        assert!(result > 0);
    }

    #[test]
    fn test_sys_info_with_buffer() {
        let mut info = core::mem::MaybeUninit::<SystemInfo>::uninit();
        let args = SyscallArgs::new(
            0x50,
            [
                info.as_mut_ptr() as usize,
                core::mem::size_of::<SystemInfo>(),
                0,
                0,
                0,
                0,
            ],
        );
        assert_eq!(sys_info(&args), SyscallError::Success.as_i64());
        let info = unsafe { info.assume_init() };
        assert_eq!(info.version_major, 0);
        assert_eq!(info.version_minor, 1);
        assert_eq!(info.cpu_count, 1);
    }

    #[test]
    fn test_sys_info_buffer_too_small() {
        let args = SyscallArgs::new(0x50, [0x1000, 1, 0, 0, 0, 0]);
        assert_eq!(sys_info(&args), SyscallError::BufferTooSmall.as_i64());
    }

    #[test]
    fn test_sys_time_returns_non_negative() {
        let args = SyscallArgs::zero(0x51);
        let t = sys_time(&args);
        assert!(t >= 0, "sys_time returned negative value: {}", t);
    }

    #[test]
    fn test_sys_time_monotonic() {
        let args = SyscallArgs::zero(0x51);
        let a = sys_time(&args);
        let b = sys_time(&args);
        assert!(b >= a, "sys_time not monotonic: {} -> {}", a, b);
    }

    #[test]
    fn test_sys_shutdown_requires_capability() {
        // Without SystemControl capability, shutdown is denied.
        let args = SyscallArgs::new(0x52, [0, 0, 0, 0, 0, 0]);
        assert_eq!(sys_shutdown(&args), SyscallError::PermissionDenied.as_i64());
    }

    #[test]
    fn test_sys_log_invalid_level() {
        let args = SyscallArgs::new(0x53, [5, 0x1000, 10, 0, 0, 0]);
        assert_eq!(sys_log(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_sys_log_null_msg() {
        let args = SyscallArgs::new(0x53, [0, 0, 10, 0, 0, 0]);
        assert_eq!(sys_log(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_sys_log_zero_len() {
        let args = SyscallArgs::new(0x53, [0, 0x1000, 0, 0, 0, 0]);
        assert_eq!(sys_log(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_sys_log_too_long() {
        let args = SyscallArgs::new(0x53, [0, 0x1000, MAX_LOG_MSG_LEN + 1, 0, 0, 0]);
        assert_eq!(sys_log(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_sys_log_valid() {
        for level in 0..=4u32 {
            let args = SyscallArgs::new(0x53, [level as usize, 0x1000, 10, 0, 0, 0]);
            assert_eq!(sys_log(&args), SyscallError::Success.as_i64());
        }
    }

    #[test]
    fn test_sys_random_null_buf() {
        let args = SyscallArgs::new(0x54, [0, 32, 0, 0, 0, 0]);
        assert_eq!(sys_random(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_sys_random_zero_len() {
        let args = SyscallArgs::new(0x54, [0x1000, 0, 0, 0, 0, 0]);
        assert_eq!(sys_random(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_sys_random_too_long() {
        let args = SyscallArgs::new(0x54, [0x1000, MAX_RANDOM_LEN + 1, 0, 0, 0, 0]);
        assert_eq!(sys_random(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_sys_random_valid() {
        let args = SyscallArgs::new(0x54, [0x1000, 32, 0, 0, 0, 0]);
        assert_eq!(sys_random(&args), SyscallError::NotSupported.as_i64());
    }

    #[test]
    fn test_sys_watchdog_pet_returns_success() {
        let args = SyscallArgs::zero(0x55);
        assert_eq!(sys_watchdog_pet(&args), SyscallError::Success.as_i64());
    }

    #[test]
    fn test_sys_watchdog_remaining_returns_value() {
        let args = SyscallArgs::zero(0x56);
        let result = sys_watchdog_remaining(&args);
        assert!(result > 0, "watchdog remaining should be positive");
        assert_eq!(result, 30000);
    }

    #[test]
    fn test_log_level_from_u32() {
        assert_eq!(LogLevel::from_u32(0), Some(LogLevel::Error));
        assert_eq!(LogLevel::from_u32(1), Some(LogLevel::Warn));
        assert_eq!(LogLevel::from_u32(2), Some(LogLevel::Info));
        assert_eq!(LogLevel::from_u32(3), Some(LogLevel::Debug));
        assert_eq!(LogLevel::from_u32(4), Some(LogLevel::Trace));
        assert_eq!(LogLevel::from_u32(5), None);
        assert_eq!(LogLevel::from_u32(255), None);
    }
}
