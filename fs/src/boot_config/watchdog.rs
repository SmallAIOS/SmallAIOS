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

/// Production stub. Real per-arch integration lands later; until
/// then, every operation returns [`WatchdogError::NotImplemented`].
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
}
