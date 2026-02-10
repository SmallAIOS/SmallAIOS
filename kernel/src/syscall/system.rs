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

/// Get system information.
///
/// Args: [buf_ptr, buf_len, 0, 0, 0, 0]
/// Returns: 0 on success (info written to buffer), negative error code on failure.
///
/// When buf_ptr is 0, returns the required buffer size.
pub fn sys_info(args: &SyscallArgs) -> SyscallResult {
    let buf_ptr = args.args[0];
    let _buf_len = args.args[1];

    if buf_ptr == 0 {
        // TODO: Return required size for SystemInfo struct
        return SyscallError::Success.as_i64();
    }

    // TODO: Fill SystemInfo struct with kernel version, arch, CPU count, memory stats
    SyscallError::Success.as_i64()
}

/// Get nanoseconds since boot.
///
/// Args: [0, 0, 0, 0, 0, 0]
/// Returns: nanoseconds since boot (positive value).
pub fn sys_time(_args: &SyscallArgs) -> SyscallResult {
    // TODO: Read from architecture timer (TSC on x86-64, CNTPCT_EL0 on ARM64)
    // For now, return 0 (boot time)
    SyscallError::Success.as_i64()
}

/// Shut down the system.
///
/// Args: [exit_code, 0, 0, 0, 0, 0]
/// Returns: does not return on success, negative error code on failure.
pub fn sys_shutdown(args: &SyscallArgs) -> SyscallResult {
    let _exit_code = args.args[0];

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

    // TODO: Integrate with CSPRNG (SHAKE256-based, seeded from RDRAND/RNDR)
    let _ = (buf_ptr, buf_len);
    SyscallError::NotSupported.as_i64()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sys_info_null_ptr_returns_success() {
        let args = SyscallArgs::new(0x50, [0, 0, 0, 0, 0, 0]);
        assert_eq!(sys_info(&args), SyscallError::Success.as_i64());
    }

    #[test]
    fn test_sys_info_with_buffer() {
        let args = SyscallArgs::new(0x50, [0x1000, 256, 0, 0, 0, 0]);
        assert_eq!(sys_info(&args), SyscallError::Success.as_i64());
    }

    #[test]
    fn test_sys_time_returns_success() {
        let args = SyscallArgs::zero(0x51);
        assert_eq!(sys_time(&args), SyscallError::Success.as_i64());
    }

    #[test]
    fn test_sys_shutdown_returns_success() {
        let args = SyscallArgs::new(0x52, [0, 0, 0, 0, 0, 0]);
        assert_eq!(sys_shutdown(&args), SyscallError::Success.as_i64());
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
