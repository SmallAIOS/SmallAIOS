// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SBI (Supervisor Binary Interface) wrapper functions (23.6, 23.7).
//!
//! SBI provides firmware services from M-mode (OpenSBI) to S-mode (SmallAIOS).
//! Calls are made via the `ecall` instruction with:
//! - a7 = extension ID (EID)
//! - a6 = function ID (FID)
//! - a0-a5 = arguments
//! - Returns: a0 = error code, a1 = value

/// SBI call return value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SbiResult {
    pub error: i64,
    pub value: i64,
}

impl SbiResult {
    pub fn is_success(&self) -> bool {
        self.error == 0
    }
}

// SBI error codes
pub const SBI_SUCCESS: i64 = 0;
pub const SBI_ERR_FAILED: i64 = -1;
pub const SBI_ERR_NOT_SUPPORTED: i64 = -2;
pub const SBI_ERR_INVALID_PARAM: i64 = -3;
pub const SBI_ERR_DENIED: i64 = -4;
pub const SBI_ERR_INVALID_ADDRESS: i64 = -5;
pub const SBI_ERR_ALREADY_AVAILABLE: i64 = -6;
pub const SBI_ERR_ALREADY_STARTED: i64 = -7;
pub const SBI_ERR_ALREADY_STOPPED: i64 = -8;

// SBI extension IDs
pub const EXT_BASE: u64 = 0x10;
pub const EXT_TIMER: u64 = 0x54494D45;    // "TIME"
pub const EXT_IPI: u64 = 0x735049;        // "sPI"
pub const EXT_RFENCE: u64 = 0x52464E43;   // "RFNC"
pub const EXT_HSM: u64 = 0x48534D;        // "HSM"
pub const EXT_SRST: u64 = 0x53525354;     // "SRST"
pub const EXT_DBCN: u64 = 0x4442434E;     // "DBCN"

// Legacy extension IDs
pub const LEGACY_CONSOLE_PUTCHAR: u64 = 1;
pub const LEGACY_CONSOLE_GETCHAR: u64 = 2;
pub const LEGACY_SHUTDOWN: u64 = 8;

/// Perform an SBI call with the given extension/function IDs and arguments.
///
/// # Safety
/// The caller must ensure the arguments are valid for the specified extension.
#[inline(always)]
unsafe fn sbi_call(eid: u64, fid: u64, a0: u64, a1: u64, a2: u64) -> SbiResult {
    let error: i64;
    let value: i64;
    core::arch::asm!(
        "ecall",
        in("a0") a0,
        in("a1") a1,
        in("a2") a2,
        in("a6") fid,
        in("a7") eid,
        lateout("a0") error,
        lateout("a1") value,
    );
    SbiResult { error, value }
}

// ---------------------------------------------------------------------------
// Base Extension (EID 0x10)
// ---------------------------------------------------------------------------

/// Get the SBI specification version.
pub fn sbi_get_spec_version() -> SbiResult {
    unsafe { sbi_call(EXT_BASE, 0, 0, 0, 0) }
}

/// Get the SBI implementation ID.
pub fn sbi_get_impl_id() -> SbiResult {
    unsafe { sbi_call(EXT_BASE, 1, 0, 0, 0) }
}

/// Get the SBI implementation version.
pub fn sbi_get_impl_version() -> SbiResult {
    unsafe { sbi_call(EXT_BASE, 2, 0, 0, 0) }
}

/// Probe whether an SBI extension is available.
pub fn sbi_probe_extension(eid: u64) -> SbiResult {
    unsafe { sbi_call(EXT_BASE, 3, eid, 0, 0) }
}

// ---------------------------------------------------------------------------
// Timer Extension (EID 0x54494D45)
// ---------------------------------------------------------------------------

/// Schedule the next timer interrupt at the given absolute time.
pub fn sbi_set_timer(stime_value: u64) {
    unsafe {
        sbi_call(EXT_TIMER, 0, stime_value, 0, 0);
    }
}

// ---------------------------------------------------------------------------
// IPI Extension (EID 0x735049)
// ---------------------------------------------------------------------------

/// Send an IPI to a set of harts specified by a bitmask.
///
/// `hart_mask` is a bitmask of target harts.
/// `hart_mask_base` is the starting hart ID for the mask.
pub fn sbi_send_ipi(hart_mask: u64, hart_mask_base: u64) -> SbiResult {
    unsafe { sbi_call(EXT_IPI, 0, hart_mask, hart_mask_base, 0) }
}

// ---------------------------------------------------------------------------
// Remote Fence Extension (EID 0x52464E43)
// ---------------------------------------------------------------------------

/// Execute FENCE.I on remote harts.
pub fn sbi_remote_fence_i(hart_mask: u64, hart_mask_base: u64) -> SbiResult {
    unsafe { sbi_call(EXT_RFENCE, 0, hart_mask, hart_mask_base, 0) }
}

/// Execute SFENCE.VMA on remote harts for all addresses and ASIDs.
pub fn sbi_remote_sfence_vma(
    hart_mask: u64,
    hart_mask_base: u64,
) -> SbiResult {
    unsafe { sbi_call(EXT_RFENCE, 1, hart_mask, hart_mask_base, 0) }
}

// ---------------------------------------------------------------------------
// HSM Extension (EID 0x48534D) — Hart State Management (23.7)
// ---------------------------------------------------------------------------

/// Hart states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HartState {
    Started = 0,
    Stopped = 1,
    StartPending = 2,
    StopPending = 3,
    Suspended = 4,
    SuspendPending = 5,
    ResumePending = 6,
}

impl HartState {
    pub fn from_value(v: i64) -> Option<Self> {
        match v {
            0 => Some(HartState::Started),
            1 => Some(HartState::Stopped),
            2 => Some(HartState::StartPending),
            3 => Some(HartState::StopPending),
            4 => Some(HartState::Suspended),
            5 => Some(HartState::SuspendPending),
            6 => Some(HartState::ResumePending),
            _ => None,
        }
    }
}

/// Start a secondary hart.
///
/// The hart begins executing at `start_addr` in S-mode with `opaque` in a0.
pub fn sbi_hart_start(hart_id: u64, start_addr: u64, opaque: u64) -> SbiResult {
    unsafe { sbi_call(EXT_HSM, 0, hart_id, start_addr, opaque) }
}

/// Stop the current hart. This call does not return on success.
pub fn sbi_hart_stop() -> SbiResult {
    unsafe { sbi_call(EXT_HSM, 1, 0, 0, 0) }
}

/// Query the status of a hart.
pub fn sbi_hart_get_status(hart_id: u64) -> SbiResult {
    unsafe { sbi_call(EXT_HSM, 2, hart_id, 0, 0) }
}

// ---------------------------------------------------------------------------
// System Reset Extension (EID 0x53525354)
// ---------------------------------------------------------------------------

/// Perform a system reset.
///
/// `reset_type`: 0 = shutdown, 1 = cold reboot, 2 = warm reboot
/// `reset_reason`: 0 = no reason, 1 = system failure
pub fn sbi_system_reset(reset_type: u32, reset_reason: u32) -> SbiResult {
    unsafe { sbi_call(EXT_SRST, 0, reset_type as u64, reset_reason as u64, 0) }
}

/// Shutdown the system.
pub fn sbi_shutdown() -> ! {
    sbi_system_reset(0, 0);
    // Fallback: try legacy shutdown.
    unsafe {
        sbi_call(LEGACY_SHUTDOWN, 0, 0, 0, 0);
    }
    loop {
        unsafe { core::arch::asm!("wfi") };
    }
}

// ---------------------------------------------------------------------------
// Debug Console Extension (EID 0x4442434E)
// ---------------------------------------------------------------------------

/// Write bytes to the debug console (DBCN extension).
///
/// Falls back to legacy console_putchar if DBCN is not available.
pub fn sbi_console_write(bytes: &[u8]) {
    // Try DBCN first: FID=0, a0=num_bytes, a1=base_addr_lo, a2=base_addr_hi
    let result = unsafe {
        sbi_call(
            EXT_DBCN,
            0,
            bytes.len() as u64,
            bytes.as_ptr() as u64,
            0,
        )
    };

    if result.error == SBI_ERR_NOT_SUPPORTED {
        // Fallback to legacy putchar.
        for &b in bytes {
            unsafe {
                sbi_call(LEGACY_CONSOLE_PUTCHAR, 0, b as u64, 0, 0);
            }
        }
    }
}

/// Write a single character via the legacy console putchar interface.
pub fn sbi_console_putchar(ch: u8) {
    unsafe {
        sbi_call(LEGACY_CONSOLE_PUTCHAR, 0, ch as u64, 0, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sbi_error_codes() {
        assert_eq!(SBI_SUCCESS, 0);
        assert_eq!(SBI_ERR_FAILED, -1);
        assert_eq!(SBI_ERR_NOT_SUPPORTED, -2);
    }

    #[test]
    fn test_sbi_result_success() {
        let r = SbiResult { error: 0, value: 42 };
        assert!(r.is_success());
        assert_eq!(r.value, 42);
    }

    #[test]
    fn test_sbi_result_error() {
        let r = SbiResult { error: -1, value: 0 };
        assert!(!r.is_success());
    }

    #[test]
    fn test_hart_state_from_value() {
        assert_eq!(HartState::from_value(0), Some(HartState::Started));
        assert_eq!(HartState::from_value(1), Some(HartState::Stopped));
        assert_eq!(HartState::from_value(2), Some(HartState::StartPending));
        assert_eq!(HartState::from_value(3), Some(HartState::StopPending));
        assert_eq!(HartState::from_value(4), Some(HartState::Suspended));
        assert_eq!(HartState::from_value(5), Some(HartState::SuspendPending));
        assert_eq!(HartState::from_value(6), Some(HartState::ResumePending));
        assert_eq!(HartState::from_value(7), None);
        assert_eq!(HartState::from_value(-1), None);
    }

    #[test]
    fn test_extension_ids() {
        assert_eq!(EXT_BASE, 0x10);
        assert_eq!(EXT_TIMER, 0x54494D45);
        assert_eq!(EXT_HSM, 0x48534D);
        assert_eq!(EXT_DBCN, 0x4442434E);
    }
}
