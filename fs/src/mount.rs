// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Boot-time mount sequence for SmallAIOS on-disk filesystems.
//!
//! [`mount_all`] orchestrates the steps documented in the
//! `embedded-filesystem-v1` change Phase 6:
//!
//! 1. Parse the GPT and validate the v1 SmallAIOS layout
//!    (5 partitions, fixed type GUIDs, fixed order). Layout
//!    mismatches are surfaced as [`MountError::GptLayoutMismatch`].
//! 2. Read the A/B boot config record from partition #5; pick the
//!    active squashfs slot (per the highest-valid-generation rule).
//! 3. Mount the squashfs partition for the active slot at `/models/`.
//!    On signature/manifest failure fall through to the OTHER slot.
//!    If both slots fail, raise the [`InferenceLockout`] guard, log
//!    `both_slots_invalid` to the audit ring, and continue boot — per
//!    `fs-integrity` the system MUST stay reachable so login, audit,
//!    and `/data/` can serve a recovery image.
//! 4. Mount the F2FS `/data/` partition. If unformatted AND physical
//!    presence is asserted → format (deferred — see note below). If
//!    unformatted AND no presence → return
//!    [`MountError::DataUnformatted`].
//! 5. Hand back the populated mount-table descriptor to the kernel.
//!
//! The kernel boot sequencer plugs the concrete `Squashfs<D>` and
//! `F2fs<'_, D>` runtime handles into the VFS slots after this
//! function returns; the mount table itself records only the markers
//! that path resolution needs (which mount the kernel SHOULD route
//! a given path to). Mirrors how the existing `/flash/` mount works.
//!
//! ## F2FS auto-format
//!
//! F2FS write-path support lands in Phase 7; the v1 read-only F2FS
//! reader cannot format an unformatted partition by itself. This
//! module therefore signals the caller via
//! [`MountAllOutcome::data_needs_format`] when the F2FS partition
//! does not have a valid superblock. The kernel boot path is
//! responsible for shelling out to the future
//! `smallaios-fs::f2fs::format` helper once it lands; until then the
//! signal MUST be accompanied by a recovery image that ships a
//! pre-formatted `/data/` partition.
//!
//! Owned by `embedded-filesystem-v1` Phase 6.

extern crate alloc;

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::block::BlockDevice;
use crate::boot_config::{
    read_active, ActiveSquashfsSlot, BootConfigError, BootConfigRecord, SlotPosition,
};
use crate::f2fs::{F2fs, F2fsError};
use crate::gpt::layout::{parse_layout, SmallaiosLayout};
use crate::gpt::{parse_gpt, GptError};
use crate::integrity::{InferenceLockout, LockoutReason};
use crate::squashfs::{Squashfs, SquashfsError};

/// Trait the mount sequencer uses to ask "is the operator physically
/// present for high-trust operations like auto-formatting `/data/`?".
///
/// Mirrors [`smallaios_auth::presence::PhysicalPresenceProvider`] but
/// stays inside the `fs` crate so we don't have to add `auth` to the
/// `fs` dep graph. The kernel boot path constructs an adapter that
/// forwards to the auth registry. An empty / never-asserting provider
/// causes [`mount_all`] to refuse the auto-format; this is the
/// fail-closed default.
pub trait PhysicalPresenceProvider {
    /// `true` iff the operator is physically present.
    fn asserted(&self) -> bool;
}

/// Adapter that always reports "not present". Used in tests where
/// presence does not matter and as the fail-closed default.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoPresence;

impl PhysicalPresenceProvider for NoPresence {
    fn asserted(&self) -> bool {
        false
    }
}

/// Adapter that always reports "present". Tests that exercise the
/// auto-format branch flip a [`StaticPresence`] in.
#[derive(Debug, Default, Clone, Copy)]
pub struct StaticPresence(pub bool);

impl PhysicalPresenceProvider for StaticPresence {
    fn asserted(&self) -> bool {
        self.0
    }
}

/// Errors surfaced by [`mount_all`].
///
/// Note that "both squashfs slots invalid" is NOT in this enum — per
/// `fs-integrity` the kernel MUST keep booting in that case; the
/// state is captured in [`MountAllOutcome::lockout`] instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MountError {
    /// GPT parse / validation failed at the disk level (bad header
    /// CRCs, layout mismatch, etc.).
    GptError(String),
    /// The disk's GPT does NOT match the v1 SmallAIOS layout.
    GptLayoutMismatch,
    /// Both A/B boot-config records failed validation; nothing to
    /// boot from.
    BothBootRecordsInvalid,
    /// Boot-config decode produced an error other than "both invalid"
    /// (corrupt magic, etc.). The kernel SHALL halt with a recovery
    /// hint.
    BootConfigError(String),
    /// `/data/` partition exists but does not contain a valid F2FS
    /// superblock AND physical presence was NOT asserted, so the
    /// kernel CANNOT auto-format. Operator must boot with presence
    /// asserted (e.g. via `auth.skip-firstboot`) to recover.
    DataUnformatted,
    /// `/data/` partition contains a parseable F2FS superblock but
    /// some other non-recoverable error (corrupt checkpoint, etc.).
    DataMountFailed(String),
    /// Both squashfs slots are invalid. Kept around in case the
    /// caller wants the strict-mode behavior; `mount_all` itself
    /// reports lockout in [`MountAllOutcome`] rather than returning
    /// this error so boot proceeds. Test code uses this variant.
    BothSlotsInvalid,
}

impl fmt::Display for MountError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GptError(msg) => write!(f, "GPT error: {msg}"),
            Self::GptLayoutMismatch => f.write_str("GPT layout does not match v1 SmallAIOS spec"),
            Self::BothBootRecordsInvalid => f.write_str("both A/B boot-config records invalid"),
            Self::BootConfigError(msg) => write!(f, "boot-config error: {msg}"),
            Self::DataUnformatted => {
                f.write_str("/data/ partition unformatted; physical presence required")
            }
            Self::DataMountFailed(msg) => write!(f, "/data/ mount failed: {msg}"),
            Self::BothSlotsInvalid => f.write_str("both squashfs slots failed verification"),
        }
    }
}

impl From<GptError> for MountError {
    fn from(e: GptError) -> Self {
        match e {
            GptError::LayoutMismatch => Self::GptLayoutMismatch,
            other => {
                use alloc::format;
                Self::GptError(format!("{other:?}"))
            }
        }
    }
}

impl From<BootConfigError> for MountError {
    fn from(e: BootConfigError) -> Self {
        match e {
            BootConfigError::BothSlotsInvalid => Self::BothBootRecordsInvalid,
            other => {
                use alloc::format;
                Self::BootConfigError(format!("{other:?}"))
            }
        }
    }
}

/// Status of one squashfs slot after [`mount_all`] tried to mount it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SlotStatus {
    /// Slot mounted and verified successfully.
    Ok,
    /// Slot mount failed with the given error (signature mismatch,
    /// bad magic, corrupted manifest, etc.).
    Failed(SquashfsError),
    /// Slot was not attempted (e.g. fallback chain reached `Ok` on
    /// the active slot first).
    NotAttempted,
}

impl SlotStatus {
    /// `true` iff this slot mounted successfully.
    pub fn is_ok(&self) -> bool {
        matches!(self, Self::Ok)
    }
}

/// What [`mount_all`] tells the kernel boot sequencer.
///
/// Carries the parsed GPT layout, the active boot record, the
/// per-slot mount outcomes, and the [`InferenceLockout`] guard. The
/// kernel uses this to plug the concrete runtime handles into the
/// VFS, then calls [`smallaios_posix::vfs::MountTable::mount_squashfs`]
/// + [`mount_f2fs`] to flip the path-resolution markers.
#[derive(Debug, Clone)]
pub struct MountAllOutcome {
    /// Parsed v1 SmallAIOS layout. The kernel reads partition LBAs /
    /// sizes from here for runtime-handle wiring.
    pub layout: SmallaiosLayout,
    /// The active boot-config record selected by the highest-valid-
    /// generation rule.
    pub active_boot_record: BootConfigRecord,
    /// Which physical slot of the boot-config partition the active
    /// record was read from.
    pub active_slot_position: SlotPosition,
    /// Slot A mount status.
    pub slot_a: SlotStatus,
    /// Slot B mount status.
    pub slot_b: SlotStatus,
    /// Which squashfs slot ended up serving `/models/`. `None` if both
    /// failed verification.
    pub active_squashfs_slot: Option<ActiveSquashfsSlot>,
    /// Inference-lockout guard. If `is_locked()`, both squashfs slots
    /// failed verification and `model_load` SHALL return `-EROFS`.
    pub lockout: InferenceLockout,
    /// `/data/` mount status — `true` if F2FS mounted cleanly.
    pub data_mounted: bool,
    /// `true` iff `/data/` lacked a valid F2FS superblock and the
    /// kernel SHALL run the (future) `f2fs::format` helper before
    /// proceeding. Always `false` if presence was not asserted (in
    /// that case `mount_all` returns `DataUnformatted` instead).
    pub data_needs_format: bool,
}

impl MountAllOutcome {
    /// `true` iff `/models/` ended up mounted.
    pub fn models_mounted(&self) -> bool {
        self.active_squashfs_slot.is_some()
    }

    /// Whether boot SHOULD proceed. Always true unless this struct
    /// was hand-crafted by tests with an inconsistent state.
    pub fn boot_proceeds(&self) -> bool {
        // Per spec, EVERY case where this struct is returned the
        // boot should proceed — failures land in `MountError`. The
        // method exists for clarity at the kernel call site.
        true
    }
}

/// Run the boot-time mount sequence.
///
/// See module docs for the step-by-step description. On success
/// returns a populated [`MountAllOutcome`] the kernel uses to wire
/// VFS handles + the inference-lockout guard. On hard failure
/// (unparseable GPT, wrong layout, both boot records invalid,
/// unformatted `/data/` without presence) returns [`MountError`].
pub fn mount_all<D: BlockDevice, P: PhysicalPresenceProvider + ?Sized>(
    device: &mut D,
    presence: &P,
    public_key: &[u8],
) -> Result<MountAllOutcome, MountError> {
    // ─── Step 1: GPT parse + layout validation ──────────────────────────────
    let entries = parse_gpt(device as &dyn BlockDevice).map_err(MountError::from)?;
    let layout = parse_layout(&entries).map_err(MountError::from)?;

    // ─── Step 2: A/B boot config ───────────────────────────────────────────
    let bs = device.block_size_bytes() as u64;
    if bs == 0 {
        return Err(MountError::BootConfigError(String::from("zero block size")));
    }
    let boot_partition_lba = layout.boot_config.starting_lba;
    let (active_boot_record, active_slot_position) =
        read_active(&*device, boot_partition_lba).map_err(MountError::from)?;

    // ─── Step 3: squashfs A/B slot mount ───────────────────────────────────
    let (slot_a, slot_b, active_squashfs_slot) =
        mount_squashfs_with_fallback(device, &layout, active_boot_record.active_slot, public_key);

    let lockout = if active_squashfs_slot.is_none() {
        crate::squashfs::audit_stub(LockoutReason::BothSlotsInvalid.audit_tag(), "");
        InferenceLockout::locked(LockoutReason::BothSlotsInvalid)
    } else {
        InferenceLockout::permissive()
    };

    // ─── Step 4: F2FS /data/ mount ─────────────────────────────────────────
    let data_partition_lba = layout.data_f2fs.starting_lba;
    let data_partition_size_lbas = layout
        .data_f2fs
        .ending_lba
        .saturating_sub(layout.data_f2fs.starting_lba)
        .saturating_add(1);

    let (data_mounted, data_needs_format) = mount_f2fs_or_signal_format(
        device,
        presence,
        data_partition_lba,
        data_partition_size_lbas,
    )?;

    Ok(MountAllOutcome {
        layout,
        active_boot_record,
        active_slot_position,
        slot_a,
        slot_b,
        active_squashfs_slot,
        lockout,
        data_mounted,
        data_needs_format,
    })
}

/// Try to mount the squashfs slot indicated by `preferred`. On any
/// verification error fall through to the other slot. Returns
/// `(slot_a_status, slot_b_status, active_slot_logical)`.
fn mount_squashfs_with_fallback<D: BlockDevice>(
    device: &D,
    layout: &SmallaiosLayout,
    preferred: ActiveSquashfsSlot,
    public_key: &[u8],
) -> (SlotStatus, SlotStatus, Option<ActiveSquashfsSlot>) {
    // Probe `preferred` first, then the other slot.
    let other = preferred.flip();
    let preferred_result = try_mount_one_slot(device, layout, preferred, public_key);
    if preferred_result.is_ok() {
        // Preferred mounted; we don't even probe the other slot
        // (matches the spec's "If signature/manifest fails, try
        // the OTHER slot").
        let (a, b) = match preferred {
            ActiveSquashfsSlot::A => (SlotStatus::Ok, SlotStatus::NotAttempted),
            ActiveSquashfsSlot::B => (SlotStatus::NotAttempted, SlotStatus::Ok),
        };
        return (a, b, Some(preferred));
    }
    let other_result = try_mount_one_slot(device, layout, other, public_key);

    let preferred_status = match preferred_result {
        Ok(()) => SlotStatus::Ok,
        Err(e) => SlotStatus::Failed(e),
    };
    let other_status = match other_result {
        Ok(()) => SlotStatus::Ok,
        Err(e) => SlotStatus::Failed(e),
    };

    let (slot_a, slot_b) = match preferred {
        ActiveSquashfsSlot::A => (preferred_status.clone(), other_status.clone()),
        ActiveSquashfsSlot::B => (other_status.clone(), preferred_status.clone()),
    };

    let active = if other_status.is_ok() {
        Some(other)
    } else if preferred_status.is_ok() {
        Some(preferred)
    } else {
        None
    };

    (slot_a, slot_b, active)
}

fn try_mount_one_slot<D: BlockDevice>(
    device: &D,
    layout: &SmallaiosLayout,
    slot: ActiveSquashfsSlot,
    public_key: &[u8],
) -> Result<(), SquashfsError> {
    let part = match slot {
        ActiveSquashfsSlot::A => &layout.squashfs_a,
        ActiveSquashfsSlot::B => &layout.squashfs_b,
    };
    let lba = part.starting_lba;
    let size_lbas = part.ending_lba.saturating_sub(lba).saturating_add(1);
    // `Squashfs::mount` runs the manifest/signature check internally
    // and verifies every block against the manifest hash array. We
    // discard the in-memory handle here — the kernel constructs a
    // fresh one to retain after `mount_all` succeeds.
    Squashfs::<D>::mount(device, lba, size_lbas, public_key).map(|_| ())
}

/// Mount F2FS at `/data/`. Returns `(mounted, needs_format)`:
/// - `(true, false)` — clean mount, runtime handle ready.
/// - `(false, true)` — unformatted; presence was asserted, kernel
///   SHOULD format and retry.
/// - `Err(DataUnformatted)` — unformatted AND no presence.
/// - `Err(DataMountFailed)` — superblock parses but other error.
fn mount_f2fs_or_signal_format<D: BlockDevice, P: PhysicalPresenceProvider + ?Sized>(
    device: &mut D,
    presence: &P,
    partition_lba: u64,
    size_lbas: u64,
) -> Result<(bool, bool), MountError> {
    match F2fs::<D>::mount(device, partition_lba, size_lbas) {
        Ok(_) => Ok((true, false)),
        Err(F2fsError::BadMagic) | Err(F2fsError::SuperblockCrcMismatch) => {
            // Treat both "no recognizable superblock magic" and
            // "both superblock CRCs failed" as "needs format" if
            // presence is asserted — operator chose to wipe.
            if presence.asserted() {
                Ok((false, true))
            } else {
                Err(MountError::DataUnformatted)
            }
        }
        Err(other) => {
            use alloc::format;
            Err(MountError::DataMountFailed(format!("{other}")))
        }
    }
}

/// Snapshot of all the mount-table-relevant facts the kernel needs
/// from a [`MountAllOutcome`]. Used by tests + the future
/// `posix::vfs::MountTable::apply_outcome` adapter to flip the path
/// resolution markers in one shot.
#[derive(Debug, Clone)]
pub struct MountTableMarkers {
    /// Active squashfs slot for `/models/`. `None` means lockout.
    pub models_slot: Option<ActiveSquashfsSlot>,
    /// `true` iff the F2FS `/data/` mount is live.
    pub data_active: bool,
    /// `true` iff the kernel must run the format helper before the
    /// `/data/` mount can proceed.
    pub data_needs_format: bool,
    /// Inference-lockout guard.
    pub lockout: InferenceLockout,
    /// LBA + size of the squashfs partition that ended up active
    /// (so the kernel can construct the runtime handle).
    pub models_partition: Option<(u64, u64)>,
    /// LBA + size of the F2FS `/data/` partition.
    pub data_partition: Option<(u64, u64)>,
}

impl MountAllOutcome {
    /// Project the outcome into the markers the VFS mount-table
    /// needs.
    pub fn markers(&self) -> MountTableMarkers {
        let models_partition = self.active_squashfs_slot.map(|slot| {
            let part = match slot {
                ActiveSquashfsSlot::A => &self.layout.squashfs_a,
                ActiveSquashfsSlot::B => &self.layout.squashfs_b,
            };
            let lba = part.starting_lba;
            let size_lbas = part.ending_lba.saturating_sub(lba).saturating_add(1);
            (lba, size_lbas)
        });
        let data_partition = if self.data_mounted || self.data_needs_format {
            let part = &self.layout.data_f2fs;
            let lba = part.starting_lba;
            let size_lbas = part.ending_lba.saturating_sub(lba).saturating_add(1);
            Some((lba, size_lbas))
        } else {
            None
        };
        MountTableMarkers {
            models_slot: self.active_squashfs_slot,
            data_active: self.data_mounted,
            data_needs_format: self.data_needs_format,
            lockout: self.lockout,
            models_partition,
            data_partition,
        }
    }

    /// List of audit events the kernel SHOULD append after `mount_all`
    /// returns. Each event is paired with a short detail string.
    pub fn pending_audit_events(&self) -> Vec<(&'static str, String)> {
        use alloc::format;
        let mut events = Vec::new();
        if let SlotStatus::Failed(e) = &self.slot_a {
            events.push(("squashfs_slot_a_failed", format!("{e}")));
        }
        if let SlotStatus::Failed(e) = &self.slot_b {
            events.push(("squashfs_slot_b_failed", format!("{e}")));
        }
        if self.lockout.is_locked() {
            events.push((
                self.lockout
                    .reason()
                    .map(|r| r.audit_tag())
                    .unwrap_or("inference_locked"),
                String::from("model_load disabled"),
            ));
        }
        if self.data_needs_format {
            events.push((
                "data_format_required",
                String::from("F2FS superblock missing; presence asserted"),
            ));
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Tiny stub helpers ─────────────────────────────────────────────────

    /// In-memory disk that can be pre-wired with arbitrary 4 KiB blocks.
    /// Used here for unit tests that target the lifting of mount errors
    /// without paying the cost of building a full GPT layout. The
    /// integration tests in `fs/tests/mount_conformance.rs` build a real
    /// disk with valid GPT + boot_config + squashfs slots.
    use crate::block::mock::MockBlockDevice;

    #[test]
    fn no_presence_returns_false() {
        let p = NoPresence;
        assert!(!p.asserted());
    }

    #[test]
    fn static_presence_round_trips() {
        let p = StaticPresence(true);
        assert!(p.asserted());
        let p2 = StaticPresence(false);
        assert!(!p2.asserted());
    }

    #[test]
    fn mount_error_display_renders() {
        // Sanity check that every Display branch produces non-empty
        // output. The kernel surfaces these to the console.
        let cases: &[MountError] = &[
            MountError::GptError(String::from("ouch")),
            MountError::GptLayoutMismatch,
            MountError::BothBootRecordsInvalid,
            MountError::BootConfigError(String::from("ouch")),
            MountError::DataUnformatted,
            MountError::DataMountFailed(String::from("ouch")),
            MountError::BothSlotsInvalid,
        ];
        for e in cases {
            let s = alloc::format!("{e}");
            assert!(!s.is_empty(), "empty Display for {:?}", e);
        }
    }

    #[test]
    fn mount_error_from_gpt_layout_mismatch() {
        let e: MountError = GptError::LayoutMismatch.into();
        assert_eq!(e, MountError::GptLayoutMismatch);
    }

    #[test]
    fn mount_error_from_boot_both_invalid() {
        let e: MountError = BootConfigError::BothSlotsInvalid.into();
        assert_eq!(e, MountError::BothBootRecordsInvalid);
    }

    #[test]
    fn mount_error_from_boot_other() {
        let e: MountError = BootConfigError::BadMagic.into();
        match e {
            MountError::BootConfigError(_) => {}
            other => panic!("expected BootConfigError, got {:?}", other),
        }
    }

    #[test]
    fn slot_status_helpers() {
        assert!(SlotStatus::Ok.is_ok());
        assert!(!SlotStatus::Failed(SquashfsError::BadMagic).is_ok());
        assert!(!SlotStatus::NotAttempted.is_ok());
    }

    #[test]
    fn mount_all_blank_disk_returns_gpt_error() {
        // A blank 4 GiB device has no GPT — `parse_gpt` returns an
        // error that `mount_all` lifts to `MountError::GptError(_)`.
        let mut device = MockBlockDevice::new(4096, 1024);
        let err = mount_all(&mut device, &NoPresence, &[0u8; 32]).unwrap_err();
        // Could be GptError or GptLayoutMismatch depending on which
        // sub-step trips first; just ensure some error came back.
        assert!(matches!(
            err,
            MountError::GptError(_) | MountError::GptLayoutMismatch
        ));
    }

    #[test]
    fn outcome_markers_for_locked_inference() {
        // Synthesize an outcome by hand to test marker extraction on
        // the both-slots-bad path.
        let layout = SmallaiosLayout {
            esp: dummy_entry(0, 100),
            squashfs_a: dummy_entry(101, 200),
            squashfs_b: dummy_entry(201, 300),
            data_f2fs: dummy_entry(301, 400),
            boot_config: dummy_entry(401, 402),
        };
        let outcome = MountAllOutcome {
            layout,
            active_boot_record: BootConfigRecord::new(ActiveSquashfsSlot::A, 1),
            active_slot_position: SlotPosition::SlotX,
            slot_a: SlotStatus::Failed(SquashfsError::ManifestSignatureInvalid),
            slot_b: SlotStatus::Failed(SquashfsError::ManifestSignatureInvalid),
            active_squashfs_slot: None,
            lockout: InferenceLockout::locked(LockoutReason::BothSlotsInvalid),
            data_mounted: true,
            data_needs_format: false,
        };
        let m = outcome.markers();
        assert!(m.lockout.is_locked());
        assert!(m.models_slot.is_none());
        assert!(m.models_partition.is_none());
        assert!(m.data_active);
        assert!(m.data_partition.is_some());
    }

    #[test]
    fn outcome_markers_for_happy_path() {
        let layout = SmallaiosLayout {
            esp: dummy_entry(0, 100),
            squashfs_a: dummy_entry(101, 200),
            squashfs_b: dummy_entry(201, 300),
            data_f2fs: dummy_entry(301, 400),
            boot_config: dummy_entry(401, 402),
        };
        let outcome = MountAllOutcome {
            layout,
            active_boot_record: BootConfigRecord::new(ActiveSquashfsSlot::B, 1),
            active_slot_position: SlotPosition::SlotY,
            slot_a: SlotStatus::NotAttempted,
            slot_b: SlotStatus::Ok,
            active_squashfs_slot: Some(ActiveSquashfsSlot::B),
            lockout: InferenceLockout::permissive(),
            data_mounted: true,
            data_needs_format: false,
        };
        let m = outcome.markers();
        assert!(!m.lockout.is_locked());
        assert_eq!(m.models_slot, Some(ActiveSquashfsSlot::B));
        assert_eq!(m.models_partition, Some((201, 100)));
        assert!(m.data_active);
        assert_eq!(m.data_partition, Some((301, 100)));
    }

    #[test]
    fn pending_audit_events_lists_failures() {
        let layout = SmallaiosLayout {
            esp: dummy_entry(0, 100),
            squashfs_a: dummy_entry(101, 200),
            squashfs_b: dummy_entry(201, 300),
            data_f2fs: dummy_entry(301, 400),
            boot_config: dummy_entry(401, 402),
        };
        let outcome = MountAllOutcome {
            layout,
            active_boot_record: BootConfigRecord::new(ActiveSquashfsSlot::A, 1),
            active_slot_position: SlotPosition::SlotX,
            slot_a: SlotStatus::Failed(SquashfsError::ManifestSignatureInvalid),
            slot_b: SlotStatus::Failed(SquashfsError::ManifestSignatureInvalid),
            active_squashfs_slot: None,
            lockout: InferenceLockout::locked(LockoutReason::BothSlotsInvalid),
            data_mounted: true,
            data_needs_format: false,
        };
        let events = outcome.pending_audit_events();
        assert!(events.iter().any(|(t, _)| *t == "squashfs_slot_a_failed"));
        assert!(events.iter().any(|(t, _)| *t == "squashfs_slot_b_failed"));
        assert!(events.iter().any(|(t, _)| *t == "both_slots_invalid"));
    }

    #[test]
    fn pending_audit_events_for_format_needed() {
        let layout = SmallaiosLayout {
            esp: dummy_entry(0, 100),
            squashfs_a: dummy_entry(101, 200),
            squashfs_b: dummy_entry(201, 300),
            data_f2fs: dummy_entry(301, 400),
            boot_config: dummy_entry(401, 402),
        };
        let outcome = MountAllOutcome {
            layout,
            active_boot_record: BootConfigRecord::new(ActiveSquashfsSlot::A, 1),
            active_slot_position: SlotPosition::SlotX,
            slot_a: SlotStatus::Ok,
            slot_b: SlotStatus::NotAttempted,
            active_squashfs_slot: Some(ActiveSquashfsSlot::A),
            lockout: InferenceLockout::permissive(),
            data_mounted: false,
            data_needs_format: true,
        };
        let events = outcome.pending_audit_events();
        assert!(events.iter().any(|(t, _)| *t == "data_format_required"));
    }

    #[test]
    fn happy_outcome_pending_audit_events_empty() {
        let layout = SmallaiosLayout {
            esp: dummy_entry(0, 100),
            squashfs_a: dummy_entry(101, 200),
            squashfs_b: dummy_entry(201, 300),
            data_f2fs: dummy_entry(301, 400),
            boot_config: dummy_entry(401, 402),
        };
        let outcome = MountAllOutcome {
            layout,
            active_boot_record: BootConfigRecord::new(ActiveSquashfsSlot::A, 1),
            active_slot_position: SlotPosition::SlotX,
            slot_a: SlotStatus::Ok,
            slot_b: SlotStatus::NotAttempted,
            active_squashfs_slot: Some(ActiveSquashfsSlot::A),
            lockout: InferenceLockout::permissive(),
            data_mounted: true,
            data_needs_format: false,
        };
        assert!(outcome.pending_audit_events().is_empty());
    }

    fn dummy_entry(start: u64, end: u64) -> crate::gpt::PartitionEntry {
        crate::gpt::PartitionEntry {
            type_guid: crate::gpt::Guid::nil(),
            unique_guid: crate::gpt::Guid::nil(),
            starting_lba: start,
            ending_lba: end,
            attributes: 0,
            name: crate::gpt::PartitionName::empty(),
        }
    }
}
