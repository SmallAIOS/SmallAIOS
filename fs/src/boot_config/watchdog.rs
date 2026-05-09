// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Boot-success watchdog interface.
//!
//! Per `fs-ab-boot/spec.md`, after an A/B update the new record is
//! written with `tentative=1, boot_success=0`. A 60 s watchdog is
//! armed at boot. When the kernel calls [`boot_success`] (typically
//! via the `SYS_BOOT_SUCCESS = 0x57` syscall, wired in Phase 10), the
//! tentative flag is cleared and `boot_success=1` is committed via
//! the same atomic-write-to-inactive-slot path.
//!
//! If the watchdog fires before [`boot_success`] is called, the
//! bootloader on the next boot observes `tentative=1, boot_success=0`
//! and rolls back to the previous slot (handled by the bootloader,
//! not this module).
//!
//! Hardware integration is platform-specific; this module exposes a
//! [`Watchdog`] trait, a [`MockWatchdog`] for tests, and a
//! [`KernelWatchdog`] stub that returns
//! [`WatchdogError::NotImplemented`] until the per-arch driver is
//! wired in.

extern crate alloc;

use core::cell::Cell;

use crate::block::BlockDevice;

use super::{
    encode_record, read_active, update_atomic, BootConfigError, BootConfigRecord, SlotPosition,
};

/// Default watchdog window in seconds, matching the spec's
/// `fs.boot.watchdog_seconds` default.
pub const DEFAULT_WATCHDOG_SECONDS: u32 = 60;

/// Errors surfaced by a [`Watchdog`] implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WatchdogError {
    /// Per-arch driver not yet wired in.
    NotImplemented,
    /// Caller passed `0` to [`Watchdog::arm`], which would never fire.
    ZeroSeconds,
}

impl core::fmt::Display for WatchdogError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => f.write_str("watchdog not implemented on this arch"),
            Self::ZeroSeconds => f.write_str("watchdog arm window cannot be 0 seconds"),
        }
    }
}

/// Boot-success watchdog.
///
/// Implementations integrate with the per-arch hardware watchdog
/// timer (Tegra234 WDT on Jetson Orin, vTimer + KVM hypercall on
/// virtio guest, etc.). All operations are best-effort: the spec
/// allows the kernel to proceed even if `arm()` fails on a board
/// without watchdog hardware (the rollback path is informational
/// rather than a hard requirement on those boards).
pub trait Watchdog {
    /// Arm the watchdog with `seconds` of slack. After this many
    /// seconds without a [`disarm`](Self::disarm) call (or a kernel
    /// `boot_success` invocation), the hardware SHALL reset the box.
    fn arm(&mut self, seconds: u32) -> Result<(), WatchdogError>;

    /// Disarm the watchdog. Idempotent — calling on an already-
    /// disarmed watchdog returns `Ok(())`.
    fn disarm(&mut self) -> Result<(), WatchdogError>;
}

/// In-memory mock for tests. Tracks arm/disarm state and the last
/// armed `seconds` value.
#[derive(Debug, Default)]
pub struct MockWatchdog {
    armed: Cell<bool>,
    last_seconds: Cell<u32>,
    arm_count: Cell<u32>,
    disarm_count: Cell<u32>,
}

impl MockWatchdog {
    /// Construct a fresh, disarmed mock.
    pub fn new() -> Self {
        Self::default()
    }

    /// `true` if currently armed.
    pub fn is_armed(&self) -> bool {
        self.armed.get()
    }

    /// Last `seconds` value passed to [`arm`](Watchdog::arm).
    pub fn last_seconds(&self) -> u32 {
        self.last_seconds.get()
    }

    /// Total successful arm calls.
    pub fn arm_count(&self) -> u32 {
        self.arm_count.get()
    }

    /// Total successful disarm calls.
    pub fn disarm_count(&self) -> u32 {
        self.disarm_count.get()
    }
}

impl Watchdog for MockWatchdog {
    fn arm(&mut self, seconds: u32) -> Result<(), WatchdogError> {
        if seconds == 0 {
            return Err(WatchdogError::ZeroSeconds);
        }
        self.armed.set(true);
        self.last_seconds.set(seconds);
        self.arm_count.set(self.arm_count.get() + 1);
        Ok(())
    }

    fn disarm(&mut self) -> Result<(), WatchdogError> {
        self.armed.set(false);
        self.disarm_count.set(self.disarm_count.get() + 1);
        Ok(())
    }
}

/// Production [`Watchdog`] facade.
///
/// Per-arch wiring (Phase 10 deferral plan):
///
/// - **x86-64**: Intel TCO (Total Cost of Ownership) watchdog timer or
///   HPET-backed software watchdog. Spec defers real bringup until the
///   x86-64 ACPI parser lands; the stub returns
///   [`WatchdogError::NotImplemented`] so the kernel's `boot_success`
///   path proceeds without rolling back. **Acceptable** because
///   x86-64 production targets currently boot under QEMU, where the
///   spec explicitly accepts a no-op watchdog (see virtio note below).
///
/// - **AArch64 (Tegra234, Jetson Orin)**: Tegra234 WDT (`TKE0_WDT0`)
///   register block at `0x0c2e0000`. The phase-2 BSP work in
///   `unikernel-orin-bringup-v1` will hand us a stable mapping; until
///   then the stub returns `NotImplemented`. Real bringup tracked in
///   `change/orin-watchdog-v1` (out of scope for this PR).
///
/// - **AArch64 (generic UEFI)**: ARM Generic Timer + EL1
///   physical-timer-interrupt fallback. Same deferral.
///
/// - **virtio-blk (QEMU)**: no-op. QEMU's default machine has no
///   watchdog; a software-only impl would defeat the purpose. The
///   stub's `NotImplemented` return is the correct behaviour for this
///   environment — the kernel logs the fact and continues.
///
/// In all cases, [`super::boot_success`] is robust to a watchdog that
/// fails to disarm: the on-disk transition is committed first, and the
/// caller is given a [`super::BootSuccessError::Watchdog`] for
/// non-fatal logging.
#[derive(Debug, Default)]
pub struct KernelWatchdog;

impl KernelWatchdog {
    /// Construct a stub watchdog.
    pub fn new() -> Self {
        Self
    }
}

impl Watchdog for KernelWatchdog {
    fn arm(&mut self, _seconds: u32) -> Result<(), WatchdogError> {
        Err(WatchdogError::NotImplemented)
    }

    fn disarm(&mut self) -> Result<(), WatchdogError> {
        Err(WatchdogError::NotImplemented)
    }
}

/// Watchdog policy helper used by the kernel boot path.
///
/// On boot the kernel reads the active boot-config record. If
/// `tentative=1`, it arms the watchdog with the configured
/// `fs.boot.watchdog_seconds` (default
/// [`DEFAULT_WATCHDOG_SECONDS`]). Until [`boot_success`] runs, a
/// watchdog timeout reboots the box; the bootloader sees the
/// tentative record on the next boot and rolls back to the previous
/// slot.
///
/// `arm_if_tentative` consolidates the boot-time decision so the
/// integration tests can exercise it without a real boot path.
pub fn arm_if_tentative<W: Watchdog + ?Sized>(
    record: &BootConfigRecord,
    watchdog: &mut W,
    seconds: u32,
) -> Result<bool, WatchdogError> {
    if !record.tentative {
        return Ok(false);
    }
    if record.boot_success {
        // Already-committed records do not need the rollback gate.
        return Ok(false);
    }
    watchdog.arm(seconds)?;
    Ok(true)
}

/// Confirm successful boot — the moral equivalent of the
/// `SYS_BOOT_SUCCESS` syscall body.
///
/// Reads the active record, asserts `tentative=1, boot_success=0` is
/// the kernel's expected state (or returns `Ok(())` if the active
/// record already has `boot_success=1`, which is the idempotent
/// re-entry case), then atomically writes a new record with
/// `tentative=0, boot_success=1, generation = max+1` and disarms the
/// watchdog. Errors from `disarm` are surfaced as
/// [`BootSuccessError::Watchdog`] but the on-disk transition is
/// committed first so a watchdog driver bug never blocks the rollback
/// gate from closing.
pub fn boot_success<D: BlockDevice + ?Sized, W: Watchdog + ?Sized>(
    device: &mut D,
    partition_lba: u64,
    watchdog: &mut W,
) -> Result<(), BootSuccessError> {
    let (active, _slot) = read_active(device, partition_lba)?;
    if active.boot_success {
        // Already confirmed; idempotent. Try to disarm anyway.
        let _ = watchdog.disarm();
        return Ok(());
    }

    let new_gen = active
        .generation
        .checked_add(1)
        .ok_or(BootSuccessError::GenerationOverflow)?;
    let new_record = BootConfigRecord {
        valid: true,
        active_slot: active.active_slot,
        tentative: false,
        generation: new_gen,
        boot_success: true,
        record_hash: [0u8; 32],
        ..active
    };
    update_atomic(device, partition_lba, new_record)?;
    watchdog.disarm().map_err(BootSuccessError::Watchdog)?;
    Ok(())
}

/// Errors specific to [`boot_success`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BootSuccessError {
    /// Underlying boot-config layer failed.
    Config(BootConfigError),
    /// Generation counter would overflow `u64`. Spec mandates
    /// monotonic + never-reuse; on the (effectively impossible)
    /// 2^64 update mark, callers should rotate the disk.
    GenerationOverflow,
    /// Watchdog disarm failed after the on-disk transition succeeded.
    /// The boot is still successful; the caller may log + retry.
    Watchdog(WatchdogError),
}

impl From<BootConfigError> for BootSuccessError {
    fn from(e: BootConfigError) -> Self {
        Self::Config(e)
    }
}

impl core::fmt::Display for BootSuccessError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Config(e) => write!(f, "boot_success: {e}"),
            Self::GenerationOverflow => f.write_str("boot_success generation overflow"),
            Self::Watchdog(e) => write!(f, "boot_success watchdog disarm: {e}"),
        }
    }
}

/// Helper used by the delta-update flow: write the boot-config record
/// the spec requires after a successful staging — `active_slot`
/// flipped, `tentative=1`, `boot_success=0`, `generation = max+1`.
///
/// Returns the [`SlotPosition`] (X or Y) the new record landed in.
/// The caller (delta apply path) feeds this into its `update_staged`
/// audit event.
pub fn stage_new_active_slot<D: BlockDevice + ?Sized>(
    device: &mut D,
    partition_lba: u64,
    new_active: super::ActiveSquashfsSlot,
) -> Result<(BootConfigRecord, SlotPosition), BootConfigError> {
    let (current, _) = match read_active(device, partition_lba) {
        Ok(v) => v,
        Err(BootConfigError::BothSlotsInvalid) => {
            // First-ever staging — there's nothing to base a generation on.
            // Use generation=1.
            let rec = BootConfigRecord {
                tentative: true,
                ..BootConfigRecord::new(new_active, 1)
            };
            let buf = encode_record(&rec);
            // Write to SlotX explicitly.
            device.write_block(partition_lba, &buf)?;
            device.flush()?;
            return Ok((rec, SlotPosition::SlotX));
        }
        Err(e) => return Err(e),
    };
    let new_gen = current
        .generation
        .checked_add(1)
        .ok_or(BootConfigError::GenerationNotMonotonic)?;
    let new_record = BootConfigRecord {
        valid: true,
        active_slot: new_active,
        tentative: true,
        generation: new_gen,
        boot_success: false,
        record_hash: [0u8; 32],
        ..current
    };
    let pos = update_atomic(device, partition_lba, new_record)?;
    Ok((new_record, pos))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::mock::MockBlockDevice;
    use crate::boot_config::{ActiveSquashfsSlot, BOOT_CONFIG_RECORD_SIZE};

    fn fresh_partition() -> MockBlockDevice {
        MockBlockDevice::new(BOOT_CONFIG_RECORD_SIZE as u32, 4)
    }

    fn seed_record(dev: &mut MockBlockDevice, gen: u64, tentative: bool, success: bool) {
        let rec = BootConfigRecord {
            valid: true,
            active_slot: ActiveSquashfsSlot::A,
            tentative,
            generation: gen,
            boot_success: success,
            ..BootConfigRecord::new(ActiveSquashfsSlot::A, gen)
        };
        let buf = encode_record(&rec);
        dev.write_block(0, &buf).unwrap();
    }

    #[test]
    fn mock_watchdog_arm_disarm() {
        let mut wd = MockWatchdog::new();
        assert!(!wd.is_armed());
        wd.arm(60).unwrap();
        assert!(wd.is_armed());
        assert_eq!(wd.last_seconds(), 60);
        assert_eq!(wd.arm_count(), 1);
        wd.disarm().unwrap();
        assert!(!wd.is_armed());
        assert_eq!(wd.disarm_count(), 1);
    }

    #[test]
    fn mock_watchdog_zero_seconds_rejected() {
        let mut wd = MockWatchdog::new();
        assert!(matches!(wd.arm(0), Err(WatchdogError::ZeroSeconds)));
    }

    #[test]
    fn kernel_watchdog_returns_not_implemented() {
        let mut wd = KernelWatchdog::new();
        assert!(matches!(wd.arm(60), Err(WatchdogError::NotImplemented)));
        assert!(matches!(wd.disarm(), Err(WatchdogError::NotImplemented)));
    }

    #[test]
    fn boot_success_clears_tentative() {
        let mut dev = fresh_partition();
        seed_record(
            &mut dev, 5, /*tentative=*/ true, /*success=*/ false,
        );
        let mut wd = MockWatchdog::new();
        wd.arm(60).unwrap();
        boot_success(&mut dev, 0, &mut wd).unwrap();
        // New active record should have tentative=0, success=1, gen=6.
        let (rec, _) = read_active(&dev, 0).unwrap();
        assert!(!rec.tentative);
        assert!(rec.boot_success);
        assert_eq!(rec.generation, 6);
        assert!(!wd.is_armed());
    }

    #[test]
    fn boot_success_idempotent_when_already_confirmed() {
        let mut dev = fresh_partition();
        seed_record(
            &mut dev, 5, /*tentative=*/ false, /*success=*/ true,
        );
        let mut wd = MockWatchdog::new();
        wd.arm(60).unwrap();
        boot_success(&mut dev, 0, &mut wd).unwrap();
        // Generation must not have advanced.
        let (rec, _) = read_active(&dev, 0).unwrap();
        assert_eq!(rec.generation, 5);
        // Watchdog disarm still attempted.
        assert!(!wd.is_armed());
    }

    #[test]
    fn boot_success_propagates_block_error() {
        let mut dev = fresh_partition();
        seed_record(
            &mut dev, 5, /*tentative=*/ true, /*success=*/ false,
        );
        // Force the next write to fail.
        dev.set_permanent_failure(crate::block::BlockError::MediaError);
        let mut wd = MockWatchdog::new();
        let r = boot_success(&mut dev, 0, &mut wd);
        assert!(matches!(r, Err(BootSuccessError::Config(_))));
    }

    #[test]
    fn stage_new_active_slot_first_run() {
        let mut dev = fresh_partition();
        let (rec, pos) = stage_new_active_slot(&mut dev, 0, ActiveSquashfsSlot::B).unwrap();
        assert_eq!(pos, SlotPosition::SlotX);
        assert_eq!(rec.generation, 1);
        assert_eq!(rec.active_slot, ActiveSquashfsSlot::B);
        assert!(rec.tentative);
        assert!(!rec.boot_success);
    }

    #[test]
    fn stage_new_active_slot_flips_active() {
        let mut dev = fresh_partition();
        seed_record(&mut dev, 4, false, true); // active=A, gen=4
        let (rec, _) = stage_new_active_slot(&mut dev, 0, ActiveSquashfsSlot::B).unwrap();
        assert_eq!(rec.active_slot, ActiveSquashfsSlot::B);
        assert_eq!(rec.generation, 5);
        assert!(rec.tentative);
    }

    #[test]
    fn boot_success_error_display() {
        use core::fmt::Write;
        let mut s = alloc::string::String::new();
        write!(s, "{}", BootSuccessError::GenerationOverflow).unwrap();
        assert!(s.contains("overflow"));
        s.clear();
        write!(
            s,
            "{}",
            BootSuccessError::Watchdog(WatchdogError::NotImplemented)
        )
        .unwrap();
        assert!(s.contains("not implemented"));
    }

    #[test]
    fn watchdog_error_display() {
        use core::fmt::Write;
        let mut s = alloc::string::String::new();
        write!(s, "{}", WatchdogError::NotImplemented).unwrap();
        assert!(s.contains("not implemented"));
        s.clear();
        write!(s, "{}", WatchdogError::ZeroSeconds).unwrap();
        assert!(s.contains("0 seconds"));
    }

    // ─── arm_if_tentative ──────────────────────────────────────────────────

    fn record(tentative: bool, success: bool) -> BootConfigRecord {
        BootConfigRecord {
            valid: true,
            active_slot: ActiveSquashfsSlot::A,
            tentative,
            generation: 1,
            boot_success: success,
            ..BootConfigRecord::new(ActiveSquashfsSlot::A, 1)
        }
    }

    #[test]
    fn arm_if_tentative_arms_when_tentative_unconfirmed() {
        let mut wd = MockWatchdog::new();
        let r = record(true, false);
        let armed = arm_if_tentative(&r, &mut wd, 60).unwrap();
        assert!(armed);
        assert!(wd.is_armed());
        assert_eq!(wd.last_seconds(), 60);
    }

    #[test]
    fn arm_if_tentative_no_op_when_not_tentative() {
        let mut wd = MockWatchdog::new();
        let r = record(false, false);
        let armed = arm_if_tentative(&r, &mut wd, 60).unwrap();
        assert!(!armed);
        assert!(!wd.is_armed());
    }

    #[test]
    fn arm_if_tentative_no_op_when_already_committed() {
        // tentative=1 but boot_success=1 means a stale record left
        // over from a successful previous boot — the rollback gate
        // is already closed, so we do not re-arm.
        let mut wd = MockWatchdog::new();
        let r = record(true, true);
        let armed = arm_if_tentative(&r, &mut wd, 60).unwrap();
        assert!(!armed);
        assert!(!wd.is_armed());
    }

    #[test]
    fn arm_if_tentative_propagates_zero_seconds() {
        let mut wd = MockWatchdog::new();
        let r = record(true, false);
        let result = arm_if_tentative(&r, &mut wd, 0);
        assert_eq!(result, Err(WatchdogError::ZeroSeconds));
        assert!(!wd.is_armed());
    }

    #[test]
    fn arm_if_tentative_propagates_kernel_not_implemented() {
        let mut wd = KernelWatchdog::new();
        let r = record(true, false);
        let result = arm_if_tentative(&r, &mut wd, 60);
        assert_eq!(result, Err(WatchdogError::NotImplemented));
    }

    // ─── Round-trip arm + boot_success disarms ────────────────────────────

    #[test]
    fn arm_then_boot_success_disarms() {
        let mut dev = fresh_partition();
        seed_record(
            &mut dev, 5, /*tentative=*/ true, /*success=*/ false,
        );
        let mut wd = MockWatchdog::new();
        let r = record(true, false);
        assert!(arm_if_tentative(&r, &mut wd, 30).unwrap());
        assert!(wd.is_armed());

        boot_success(&mut dev, 0, &mut wd).unwrap();
        assert!(!wd.is_armed());
        assert_eq!(wd.disarm_count(), 1);
    }

    #[test]
    fn boot_success_disarms_even_when_provider_already_committed() {
        // Spec scenario: the kernel boots, watchdog is armed, but the
        // record is already `boot_success=1` (e.g., recovery boot).
        // boot_success() short-circuits but still disarms.
        let mut dev = fresh_partition();
        seed_record(
            &mut dev, 5, /*tentative=*/ false, /*success=*/ true,
        );
        let mut wd = MockWatchdog::new();
        wd.arm(60).unwrap();
        boot_success(&mut dev, 0, &mut wd).unwrap();
        assert!(!wd.is_armed());
    }

    #[test]
    fn arm_uses_default_window_when_seconds_default() {
        let mut wd = MockWatchdog::new();
        let r = record(true, false);
        arm_if_tentative(&r, &mut wd, DEFAULT_WATCHDOG_SECONDS).unwrap();
        assert_eq!(wd.last_seconds(), DEFAULT_WATCHDOG_SECONDS);
    }

    #[test]
    fn watchdog_arm_count_increments_per_call() {
        let mut wd = MockWatchdog::new();
        wd.arm(60).unwrap();
        wd.arm(60).unwrap();
        wd.arm(60).unwrap();
        assert_eq!(wd.arm_count(), 3);
    }

    #[test]
    fn watchdog_disarm_idempotent_on_unarmed_mock() {
        let mut wd = MockWatchdog::new();
        // Disarming an already-disarmed mock is OK.
        wd.disarm().unwrap();
        wd.disarm().unwrap();
        assert_eq!(wd.disarm_count(), 2);
    }
}
