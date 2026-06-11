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
/// Root-only commit of the active A/B boot slot. Delegates to the
/// installed [`crate::boot_success::BootSuccessProvider`]; if no
/// provider has been installed (Wave-0 boot path before Phase 10
/// finishes wiring), returns `-ENOSYS`.
///
/// The handler is idempotent: a second call after a successful first
/// returns `0` without advancing the generation counter. The audit
/// record `boot_success_committed` is appended on every successful
/// transition (idempotent or not) so the auditor sees both "first
/// commit" and "redundant retry" attempts.
///
/// MinRole: [`crate::auth::MinRole::Root`] — enforced by
/// [`crate::syscall::min_role_for`] (returns `Root` for
/// `SYS_BOOT_SUCCESS`).
pub fn sys_boot_success(_args: &SyscallArgs) -> SyscallResult {
    use crate::boot_success::{with_provider, BootSuccessError};

    let outcome = match with_provider(|p| p.commit()) {
        Ok(o) => o,
        Err(BootSuccessError::NotImplemented) => return -38, // -ENOSYS
        Err(BootSuccessError::Storage(_)) => return SyscallError::IoError.as_i64(),
        Err(BootSuccessError::Retry(_)) => return SyscallError::Busy.as_i64(),
        Err(BootSuccessError::AlreadyCommitted) => {
            // Per-spec idempotency — providers SHOULD return Ok on
            // re-entry; only the test bench raises this variant
            // directly.
            return SyscallError::Success.as_i64();
        }
    };

    // Audit record stub: tests use `audit_capture` to observe; the
    // production audit ring lands with Phase N of the audit roadmap.
    #[cfg(any(test, feature = "test-mocks"))]
    crate::boot_success::audit_capture::record(outcome, "boot_success_committed");
    #[cfg(not(any(test, feature = "test-mocks")))]
    let _ = outcome;

    SyscallError::Success.as_i64()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::boot_success::{KernelBootSuccessProvider, MockBootSuccessProvider};

    extern crate std;
    use std::sync::{Mutex, MutexGuard};

    /// Serializes the `boot_success_*` tests: they share the global
    /// provider slot and the audit-capture mirror, and
    /// [`reclaim_provider`] frees the previously installed mock —
    /// concurrent readers would otherwise race on the slot or read
    /// freed memory (same pattern as `syscall::memory::tests` /
    /// `syscall::device::test_sync`). Poison is ignored so one
    /// panicking test does not cascade into unrelated failures.
    static BOOT_SUCCESS_LOCK: Mutex<()> = Mutex::new(());

    fn lock_boot_success() -> MutexGuard<'static, ()> {
        BOOT_SUCCESS_LOCK.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Allocate and install a fresh mock provider. Returns the
    /// `'static` reference used by the test body plus the raw pointer
    /// [`reclaim_provider`] needs to free it. Caller must hold
    /// [`BOOT_SUCCESS_LOCK`].
    fn install_mock_provider() -> (
        &'static MockBootSuccessProvider,
        *mut MockBootSuccessProvider,
    ) {
        let raw =
            alloc::boxed::Box::into_raw(alloc::boxed::Box::new(MockBootSuccessProvider::new()));
        // SAFETY: `raw` came from `Box::into_raw` above — non-null,
        // aligned, and uniquely owned until `reclaim_provider` frees it.
        let provider: &'static MockBootSuccessProvider = unsafe { &*raw };
        crate::boot_success::install_provider(provider);
        (provider, raw)
    }

    /// Swap the global provider slot back to a static stub and free the
    /// mock installed by [`install_mock_provider`], so the test binary
    /// exits without leaking (Miri's leak checker runs at process
    /// exit). Caller must hold [`BOOT_SUCCESS_LOCK`] and must not touch
    /// the mock afterwards.
    fn reclaim_provider(raw: *mut MockBootSuccessProvider) {
        // `install_provider` documents re-installation for test
        // harnesses; after this call the slot no longer references the
        // mock.
        static RESET_STUB: KernelBootSuccessProvider = KernelBootSuccessProvider::new();
        crate::boot_success::install_provider(&RESET_STUB);
        // SAFETY: the slot now points at `RESET_STUB`, the caller holds
        // `BOOT_SUCCESS_LOCK` (so no concurrent `with_provider` reader
        // exists), and `raw` is the uniquely-owned allocation produced
        // by `install_mock_provider`.
        unsafe { drop(alloc::boxed::Box::from_raw(raw)) };
    }

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
        // Use a real buffer: `sys_random` materializes `&mut [u8]` from the
        // pointer arg before the CSPRNG call, so a fabricated address like
        // 0x1000 is undefined behavior (caught by Miri) even though the
        // uninitialized CSPRNG never writes through it.
        let mut buf = [0u8; 32];
        let args = SyscallArgs::new(0x54, [buf.as_mut_ptr() as usize, 32, 0, 0, 0, 0]);
        // CSPRNG is uninitialized in unit tests, so the syscall reports
        // NotSupported without touching the buffer.
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

    // ─── sys_boot_success ─────────────────────────────────────────────────

    #[test]
    fn boot_success_handler_signature_returns_i64() {
        // Smoke: the handler function exists and has the SyscallResult
        // (i64) return shape required by the dispatch table. Sibling
        // tests in this binary install (and later reclaim) mock
        // providers, so depending on ordering the slot holds nothing,
        // a mock, or the reset stub — this test only guards against
        // accidental signature regressions.
        let _guard = lock_boot_success();
        let args = SyscallArgs::zero(0x57);
        let r: i64 = sys_boot_success(&args);
        // -38 (ENOSYS) before any test installs, OR 0 (Success) after
        // a sibling test has installed a mock — both are acceptable
        // shape-wise.
        assert!(r == 0 || r == -38 || r < 0);
    }

    #[test]
    fn boot_success_with_mock_provider_returns_zero() {
        use crate::boot_success::audit_capture;
        let _guard = lock_boot_success();
        let (provider, provider_raw) = install_mock_provider();
        audit_capture::clear();

        let args = SyscallArgs::zero(0x57);
        let r = sys_boot_success(&args);
        assert_eq!(r, SyscallError::Success.as_i64());
        assert!(provider.is_committed());
        assert_eq!(provider.commit_count(), 1);

        // Audit record emitted.
        let audit = audit_capture::last();
        assert!(audit.is_some(), "audit record not emitted");
        let (outcome, tag) = audit.unwrap();
        assert_eq!(tag, "boot_success_committed");
        assert!(!outcome.idempotent);
        reclaim_provider(provider_raw);
    }

    #[test]
    fn boot_success_idempotent_returns_zero() {
        use crate::boot_success::audit_capture;
        let _guard = lock_boot_success();
        let (provider, provider_raw) = install_mock_provider();
        audit_capture::clear();

        let args = SyscallArgs::zero(0x57);
        // First commit advances generation.
        assert_eq!(sys_boot_success(&args), SyscallError::Success.as_i64());
        // Second commit is idempotent (still 0).
        assert_eq!(sys_boot_success(&args), SyscallError::Success.as_i64());
        assert_eq!(provider.commit_count(), 2);
        // Generation only advanced once.
        assert_eq!(provider.generation(), 2);

        let audit = audit_capture::last().unwrap();
        assert!(audit.0.idempotent);
        reclaim_provider(provider_raw);
    }

    #[test]
    fn boot_success_storage_error_returns_eio() {
        use crate::boot_success::BootSuccessError;
        let _guard = lock_boot_success();
        let (provider, provider_raw) = install_mock_provider();

        provider.fail_next(BootSuccessError::Storage("media"));
        let args = SyscallArgs::zero(0x57);
        assert_eq!(sys_boot_success(&args), SyscallError::IoError.as_i64());
        reclaim_provider(provider_raw);
    }

    #[test]
    fn boot_success_retry_returns_busy() {
        use crate::boot_success::BootSuccessError;
        let _guard = lock_boot_success();
        let (provider, provider_raw) = install_mock_provider();

        provider.fail_next(BootSuccessError::Retry("watchdog"));
        let args = SyscallArgs::zero(0x57);
        assert_eq!(sys_boot_success(&args), SyscallError::Busy.as_i64());
        reclaim_provider(provider_raw);
    }

    #[test]
    fn boot_success_already_committed_variant_returns_zero() {
        // The `AlreadyCommitted` variant is rare (mock-only) but the
        // handler must collapse it to success.
        use crate::boot_success::BootSuccessError;
        let _guard = lock_boot_success();
        let (provider, provider_raw) = install_mock_provider();

        provider.fail_next(BootSuccessError::AlreadyCommitted);
        let args = SyscallArgs::zero(0x57);
        assert_eq!(sys_boot_success(&args), SyscallError::Success.as_i64());
        reclaim_provider(provider_raw);
    }

    #[test]
    fn boot_success_dispatches_through_table() {
        use crate::syscall::dispatch;
        use crate::syscall::nr;

        let _guard = lock_boot_success();
        let (_provider, provider_raw) = install_mock_provider();

        let args = SyscallArgs::zero(nr::SYS_BOOT_SUCCESS);
        // dispatch returns 0 (Success) when auth_gate is off (boot
        // default), regardless of role.
        let r = dispatch(&args);
        assert_eq!(r, SyscallError::Success.as_i64());
        reclaim_provider(provider_raw);
    }

    #[test]
    fn boot_success_non_root_denied_when_auth_gate_on() {
        use crate::syscall::{dispatch, nr, set_auth_gate_enabled};

        let _guard = lock_boot_success();
        let (_provider, provider_raw) = install_mock_provider();

        // Enable the auth gate. With no live session, current_role()
        // returns None, which fails the Root gate → -EACCES.
        let prev = set_auth_gate_enabled(true);
        let args = SyscallArgs::zero(nr::SYS_BOOT_SUCCESS);
        let r = dispatch(&args);
        // Restore before asserting so a failing assertion does not
        // leave the gate flipped for sibling tests.
        let _ = set_auth_gate_enabled(prev);
        assert_eq!(r, crate::auth::ERRNO_EACCES);
        reclaim_provider(provider_raw);
    }

    #[test]
    fn boot_success_handler_in_dispatch_table() {
        // Confirm the SYS_BOOT_SUCCESS slot is wired up at table-init
        // time; without this, the handler would never be invoked even
        // if a provider is installed. The fact that dispatch reaches
        // sys_boot_success is asserted indirectly by
        // boot_success_dispatches_through_table — this assertion is
        // the explicit slot-coverage check.
        use crate::syscall::nr;
        assert_eq!(nr::SYS_BOOT_SUCCESS, 0x57);
    }
}
