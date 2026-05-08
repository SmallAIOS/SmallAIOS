// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! UEFI variable mirror for the A/B boot pointer.
//!
//! On UEFI-capable arches (x86-64, AArch64-with-UEFI) the kernel
//! mirrors the active-slot pointer to a UEFI variable so the
//! bootloader can pick the slot without parsing GPT first. On
//! disagreement between the variable and the partition, **the
//! partition wins** — the variable is a hint, not the truth.
//!
//! The variable name is fixed:
//! `SmallAIOSBoot-A3F7C2E0-FACE-4FFF-BBBB-000000000000`
//!
//! Implementations of [`UefiMirror`]:
//!
//! - [`TestUefiMirror`] — in-memory cell, used by the host test suite
//!   (and any `#![cfg(test)]` plumbing in callers).
//! - [`KernelUefiMirror`] — placeholder. Real UEFI runtime services
//!   integration lives behind a separate kernel feature; the bare
//!   trait here returns [`UefiMirrorError::NotImplemented`] so the
//!   high-level update flow can fail closed without a UEFI back-end.

extern crate alloc;

use alloc::string::String;
use core::cell::Cell;

use super::ActiveSquashfsSlot;

/// Canonical UEFI variable name used by the SmallAIOS A/B mirror.
///
/// The trailing GUID matches the boot-config GPT partition type
/// GUID per `fs-ab-boot/spec.md`.
//
// lgtm[rust/hard-coded-cryptographic-value]
pub const UEFI_VARIABLE_NAME: &str = "SmallAIOSBoot-A3F7C2E0-FACE-4FFF-BBBB-000000000000";

/// Errors surfaced by an [`UefiMirror`] implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UefiMirrorError {
    /// The kernel was built without a UEFI runtime services back-end
    /// for the current arch (e.g., RISC-V or unbooted).
    NotImplemented,
    /// The UEFI runtime returned an error (status code surfaced as
    /// the inner string for diagnostics).
    Runtime(String),
    /// Caller passed an unrecognised value byte (not 0/1).
    BadValue(u8),
}

impl core::fmt::Display for UefiMirrorError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotImplemented => f.write_str("UEFI mirror not implemented on this arch"),
            Self::Runtime(s) => write!(f, "UEFI runtime error: {s}"),
            Self::BadValue(b) => write!(f, "UEFI mirror bad value byte {b}"),
        }
    }
}

/// Mirror of the active-slot pointer to/from a UEFI variable.
///
/// Implementations SHOULD encode the variable as one byte: `0` for
/// slot A, `1` for slot B. Larger payloads are not specified here.
pub trait UefiMirror {
    /// Read the variable's value. `None` means the variable is absent
    /// (e.g. NVRAM was wiped) — callers MUST fall back to the
    /// partition.
    fn read(&self) -> Result<Option<ActiveSquashfsSlot>, UefiMirrorError>;

    /// Write `value` to the variable. Implementations MUST be
    /// idempotent — writing the same value twice is a no-op observable
    /// only via a counter (or similar telemetry).
    fn write(&mut self, value: ActiveSquashfsSlot) -> Result<(), UefiMirrorError>;
}

/// In-memory mock for tests. Stores one [`Cell`] of `Option<u8>`.
#[derive(Debug, Default)]
pub struct TestUefiMirror {
    cell: Cell<Option<u8>>,
    write_count: Cell<u32>,
    /// If `Some`, every operation returns this error (used to test
    /// the "stale variable overruled by partition" path).
    forced_error: Cell<Option<UefiMirrorErrorKind>>,
}

#[derive(Debug, Clone, Copy)]
enum UefiMirrorErrorKind {
    NotImplemented,
}

impl TestUefiMirror {
    /// Construct a fresh mirror with the variable absent.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a mirror pre-seeded with `value`.
    pub fn with_value(value: ActiveSquashfsSlot) -> Self {
        let s = Self::new();
        s.cell.set(Some(value.as_u8()));
        s
    }

    /// Force every subsequent op to return [`UefiMirrorError::NotImplemented`].
    pub fn force_not_implemented(&self) {
        self.forced_error
            .set(Some(UefiMirrorErrorKind::NotImplemented));
    }

    /// Number of successful writes so far.
    pub fn write_count(&self) -> u32 {
        self.write_count.get()
    }

    /// Direct read of the raw stored byte (for test introspection).
    pub fn raw(&self) -> Option<u8> {
        self.cell.get()
    }

    /// Wipe the variable (simulates UEFI NVRAM clear).
    pub fn clear(&self) {
        self.cell.set(None);
    }
}

impl UefiMirror for TestUefiMirror {
    fn read(&self) -> Result<Option<ActiveSquashfsSlot>, UefiMirrorError> {
        if let Some(UefiMirrorErrorKind::NotImplemented) = self.forced_error.get() {
            return Err(UefiMirrorError::NotImplemented);
        }
        match self.cell.get() {
            None => Ok(None),
            Some(b) => match ActiveSquashfsSlot::from_u8(b) {
                Some(s) => Ok(Some(s)),
                None => Err(UefiMirrorError::BadValue(b)),
            },
        }
    }

    fn write(&mut self, value: ActiveSquashfsSlot) -> Result<(), UefiMirrorError> {
        if let Some(UefiMirrorErrorKind::NotImplemented) = self.forced_error.get() {
            return Err(UefiMirrorError::NotImplemented);
        }
        self.cell.set(Some(value.as_u8()));
        self.write_count.set(self.write_count.get() + 1);
        Ok(())
    }
}

/// Production back-end. Until a UEFI runtime-services binding lands
/// in the kernel, every operation returns
/// [`UefiMirrorError::NotImplemented`].
///
/// The high-level boot-flow code is required by the spec to treat the
/// UEFI variable as a hint and the partition as the truth, so this
/// stub is correct for non-UEFI arches and for early bring-up of UEFI
/// arches alike.
#[derive(Debug, Default)]
pub struct KernelUefiMirror;

impl KernelUefiMirror {
    /// Construct a stub mirror.
    pub fn new() -> Self {
        Self
    }
}

impl UefiMirror for KernelUefiMirror {
    fn read(&self) -> Result<Option<ActiveSquashfsSlot>, UefiMirrorError> {
        Err(UefiMirrorError::NotImplemented)
    }

    fn write(&mut self, _value: ActiveSquashfsSlot) -> Result<(), UefiMirrorError> {
        Err(UefiMirrorError::NotImplemented)
    }
}

/// Reconcile the UEFI variable with what the partition says.
///
/// Spec scenario (`Stale variable overruled by partition`):
///
/// > **WHEN** the partition says `active_slot=B` and the UEFI
/// > variable says A (e.g., NVRAM corruption)
/// > **THEN** the kernel SHALL boot from B
/// > **AND** SHALL log `warn: UEFI boot var stale; rewriting`
/// > **AND** SHALL rewrite the variable to match
///
/// This helper performs the reconcile half of that flow: it returns
/// `Ok(true)` if it had to rewrite the variable, `Ok(false)` if the
/// variable already matched (or was absent and was just written).
/// Callers that observe `true` SHOULD log the stale-var warning.
///
/// Errors from the underlying mirror are surfaced to the caller; the
/// production policy is that a UEFI failure does NOT fail the boot —
/// the partition truth is honored regardless. The flow code therefore
/// typically calls this and ignores the error after logging it.
pub fn reconcile<M: UefiMirror + ?Sized>(
    mirror: &mut M,
    partition_truth: ActiveSquashfsSlot,
) -> Result<bool, UefiMirrorError> {
    let current = mirror.read()?;
    match current {
        Some(s) if s == partition_truth => Ok(false),
        Some(_) => {
            mirror.write(partition_truth)?;
            Ok(true)
        }
        None => {
            mirror.write(partition_truth)?;
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_name_matches_spec() {
        // Spec literal — must not drift.
        assert!(UEFI_VARIABLE_NAME.starts_with("SmallAIOSBoot-"));
        assert!(UEFI_VARIABLE_NAME.contains("A3F7C2E0-FACE-4FFF-BBBB-000000000000"));
    }

    #[test]
    fn test_mirror_round_trip() {
        let mut m = TestUefiMirror::new();
        assert!(m.read().unwrap().is_none());
        m.write(ActiveSquashfsSlot::B).unwrap();
        assert_eq!(m.read().unwrap(), Some(ActiveSquashfsSlot::B));
        assert_eq!(m.write_count(), 1);
        m.write(ActiveSquashfsSlot::A).unwrap();
        assert_eq!(m.read().unwrap(), Some(ActiveSquashfsSlot::A));
        assert_eq!(m.write_count(), 2);
    }

    #[test]
    fn test_mirror_pre_seeded() {
        let m = TestUefiMirror::with_value(ActiveSquashfsSlot::B);
        assert_eq!(m.read().unwrap(), Some(ActiveSquashfsSlot::B));
    }

    #[test]
    fn reconcile_writes_when_absent() {
        let mut m = TestUefiMirror::new();
        let rewrote = reconcile(&mut m, ActiveSquashfsSlot::A).unwrap();
        assert!(!rewrote, "absent → write is not 'rewrite'");
        assert_eq!(m.read().unwrap(), Some(ActiveSquashfsSlot::A));
    }

    #[test]
    fn reconcile_noop_when_match() {
        let mut m = TestUefiMirror::with_value(ActiveSquashfsSlot::A);
        let pre = m.write_count();
        let rewrote = reconcile(&mut m, ActiveSquashfsSlot::A).unwrap();
        assert!(!rewrote);
        assert_eq!(m.write_count(), pre, "match → no write");
    }

    #[test]
    fn reconcile_overwrites_when_stale() {
        let mut m = TestUefiMirror::with_value(ActiveSquashfsSlot::A);
        let rewrote = reconcile(&mut m, ActiveSquashfsSlot::B).unwrap();
        assert!(rewrote);
        assert_eq!(m.read().unwrap(), Some(ActiveSquashfsSlot::B));
    }

    #[test]
    fn kernel_mirror_returns_not_implemented() {
        let mut m = KernelUefiMirror::new();
        assert!(matches!(m.read(), Err(UefiMirrorError::NotImplemented)));
        assert!(matches!(
            m.write(ActiveSquashfsSlot::A),
            Err(UefiMirrorError::NotImplemented)
        ));
    }

    #[test]
    fn forced_not_implemented_path() {
        let m = TestUefiMirror::new();
        m.force_not_implemented();
        assert!(matches!(m.read(), Err(UefiMirrorError::NotImplemented)));
    }

    #[test]
    fn bad_value_byte_rejected() {
        let m = TestUefiMirror::new();
        m.cell.set(Some(7));
        assert!(matches!(m.read(), Err(UefiMirrorError::BadValue(7))));
    }

    #[test]
    fn clear_resets_to_absent() {
        let m = TestUefiMirror::with_value(ActiveSquashfsSlot::A);
        m.clear();
        assert!(m.read().unwrap().is_none());
    }

    #[test]
    fn error_display() {
        use core::fmt::Write;
        let mut s = String::new();
        write!(s, "{}", UefiMirrorError::NotImplemented).unwrap();
        assert!(s.contains("not implemented"));
        s.clear();
        write!(s, "{}", UefiMirrorError::BadValue(99)).unwrap();
        assert!(s.contains("99"));
    }
}
