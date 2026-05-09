// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end conformance test for the overlay model-add /
//! model-remove syscalls (0x36 / 0x37).
//!
//! Spec: `openspec/changes/embedded-overlay-v1/specs/fs-overlay-syscalls/spec.md`.
//!
//! This test drives [`handle_onnx_model_add`] / [`handle_onnx_model_remove`]
//! through a `MockOverlayBackend` + a `CapturingOverlayAuditSink` to
//! validate the spec's scenarios end-to-end. The kernel can't depend
//! on `smallaios-fs` so the production wiring of these handlers
//! against `OverlayWriter` is exercised in `fs::overlay::kernel_adapter`'s
//! own unit tests.

use smallaios_kernel::auth::{OverlayLockTable, Role, SessionId};
use smallaios_kernel::syscall::model::{
    handle_onnx_model_add, handle_onnx_model_remove, AddOutcome, BackendError,
    CapturingOverlayAuditSink, ContentSource, ModelCtx, OverlayAuditEvent, OverlayBackend,
    SliceSource,
};

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use smallaios_security::crypto::sha3::{Sha3_256, Sha3_256Digest};

const ERRNO_EPERM: i64 = -1;
const ERRNO_EBUSY: i64 = -16;
const ERRNO_EINVAL: i64 = -22;
const ERRNO_ENOSPC: i64 = -28;

#[derive(Debug, Default)]
struct MockOverlayBackend {
    files: BTreeMap<String, Vec<u8>>,
    sha3: BTreeMap<String, Sha3_256Digest>,
    sig: BTreeMap<String, Vec<u8>>,
    whiteouts: BTreeMap<String, ()>,
    cleanup_calls: Vec<String>,
    cap_remaining: u64,
}

impl MockOverlayBackend {
    fn new(cap: u64) -> Self {
        Self {
            cap_remaining: cap,
            ..Self::default()
        }
    }
}

impl OverlayBackend for MockOverlayBackend {
    fn add(
        &mut self,
        name: &str,
        source: &mut dyn ContentSource,
        expected_size: u64,
        signature: Option<&[u8]>,
    ) -> Result<AddOutcome, BackendError> {
        if expected_size > self.cap_remaining {
            return Err(BackendError::NoSpace);
        }
        let mut hasher = Sha3_256::new();
        let mut buf = [0u8; 8192];
        let mut bytes = Vec::new();
        loop {
            match source.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if (bytes.len() as u64) + (n as u64) > self.cap_remaining {
                        return Err(BackendError::NoSpace);
                    }
                    hasher.update(&buf[..n]).unwrap();
                    bytes.extend_from_slice(&buf[..n]);
                }
                Err(()) => return Err(BackendError::Io),
            }
        }
        let digest = hasher.finalize().unwrap();
        let size = bytes.len() as u64;
        self.files.insert(name.to_string(), bytes);
        self.sha3.insert(name.to_string(), digest);
        let signature_written = if let Some(sig) = signature {
            self.sig.insert(name.to_string(), sig.to_vec());
            true
        } else {
            self.sig.remove(name);
            false
        };
        Ok(AddOutcome {
            name: name.to_string(),
            sha3: digest,
            size,
            signature_written,
        })
    }

    fn remove_upper(&mut self, name: &str) -> Result<(), BackendError> {
        self.files.remove(name);
        self.sha3.remove(name);
        self.sig.remove(name);
        Ok(())
    }

    fn write_whiteout(&mut self, name: &str) -> Result<(), BackendError> {
        self.whiteouts.insert(name.to_string(), ());
        Ok(())
    }

    fn remove_whiteout(&mut self, name: &str) -> Result<(), BackendError> {
        self.whiteouts.remove(name);
        Ok(())
    }

    fn cleanup_orphan_tmp_for(&mut self, name: &str) {
        self.cleanup_calls.push(name.to_string());
    }
}

fn ctx<'a>(
    locks: &'a mut OverlayLockTable,
    backend: &'a mut MockOverlayBackend,
    audit: &'a mut CapturingOverlayAuditSink,
    role: Option<Role>,
    user_id: u32,
    allow_operator_unhide: bool,
) -> ModelCtx<'a, MockOverlayBackend, CapturingOverlayAuditSink> {
    ModelCtx {
        locks,
        backend,
        audit,
        holder: SessionId::from_raw((u64::from(user_id) | 0x0100_0000) as u32),
        role,
        user_id,
        allow_operator_unhide,
        require_signed: false,
    }
}

// ─── Spec scenarios — `model_add syscall` ────────────────────────────────────

#[test]
fn spec_operator_successfully_adds_a_model() {
    let mut locks = OverlayLockTable::new();
    let mut backend = MockOverlayBackend::new(100_000_000);
    let mut audit = CapturingOverlayAuditSink::new();
    let model_bytes = [0xAAu8; 50_000];
    let mut c = ctx(
        &mut locks,
        &mut backend,
        &mut audit,
        Some(Role::Operator),
        1234,
        true,
    );
    let mut src = SliceSource::new(&model_bytes);
    let r = handle_onnx_model_add(&mut c, b"custom.onnx", &mut src, 50_000, None);
    assert_eq!(r, 0);
    assert_eq!(backend.files.get("custom.onnx").unwrap().len(), 50_000);
    assert_eq!(audit.len(), 1);
    match &audit.events[0] {
        OverlayAuditEvent::ModelAdded {
            who, name, size, ..
        } => {
            assert_eq!(*who, 1234);
            assert_eq!(name, "custom.onnx");
            assert_eq!(*size, 50_000);
        }
        e => panic!("unexpected: {:?}", e),
    }
}

#[test]
fn spec_viewer_denied() {
    let mut locks = OverlayLockTable::new();
    let mut backend = MockOverlayBackend::new(100_000);
    let mut audit = CapturingOverlayAuditSink::new();
    let mut c = ctx(
        &mut locks,
        &mut backend,
        &mut audit,
        Some(Role::Viewer),
        7,
        true,
    );
    let mut src = SliceSource::new(b"x");
    let r = handle_onnx_model_add(&mut c, b"foo.onnx", &mut src, 1, None);
    assert_eq!(r, ERRNO_EPERM);
    assert_eq!(audit.len(), 1);
    assert!(matches!(
        audit.events[0],
        OverlayAuditEvent::DenyModelAdd { .. }
    ));
}

#[test]
fn spec_reserved_suffix_rejected_no_lock() {
    let mut locks = OverlayLockTable::new();
    let mut backend = MockOverlayBackend::new(100_000);
    let mut audit = CapturingOverlayAuditSink::new();
    let mut c = ctx(
        &mut locks,
        &mut backend,
        &mut audit,
        Some(Role::Operator),
        7,
        true,
    );
    let mut src = SliceSource::new(b"x");
    let r = handle_onnx_model_add(&mut c, b"foo.whiteout", &mut src, 1, None);
    assert_eq!(r, ERRNO_EINVAL);
    assert!(!locks.is_held(b"foo.whiteout"));
    assert!(backend.files.is_empty());
}

// ─── Spec scenarios — `model_remove syscall` ─────────────────────────────────

#[test]
fn spec_root_removes_upper_file() {
    let mut locks = OverlayLockTable::new();
    let mut backend = MockOverlayBackend::new(100_000);
    backend
        .files
        .insert("custom.onnx".to_string(), b"hello".to_vec());
    let mut audit = CapturingOverlayAuditSink::new();
    let mut c = ctx(
        &mut locks,
        &mut backend,
        &mut audit,
        Some(Role::Root),
        2,
        true,
    );
    let r = handle_onnx_model_remove(&mut c, b"custom.onnx", 0);
    assert_eq!(r, 0);
    assert!(!backend.files.contains_key("custom.onnx"));
    assert!(matches!(
        audit.events[0],
        OverlayAuditEvent::ModelRemoved { ref name, .. } if name == "custom.onnx"
    ));
}

#[test]
fn spec_root_hides_lower_file() {
    let mut locks = OverlayLockTable::new();
    let mut backend = MockOverlayBackend::new(100_000);
    let mut audit = CapturingOverlayAuditSink::new();
    let mut c = ctx(
        &mut locks,
        &mut backend,
        &mut audit,
        Some(Role::Root),
        2,
        true,
    );
    let r = handle_onnx_model_remove(&mut c, b"llama-3.onnx", 1);
    assert_eq!(r, 0);
    assert!(backend.whiteouts.contains_key("llama-3.onnx"));
    assert!(matches!(
        audit.events[0],
        OverlayAuditEvent::ModelHidden { ref name, .. } if name == "llama-3.onnx"
    ));
}

#[test]
fn spec_operator_denied_remove_audited() {
    let mut locks = OverlayLockTable::new();
    let mut backend = MockOverlayBackend::new(100_000);
    let mut audit = CapturingOverlayAuditSink::new();
    let mut c = ctx(
        &mut locks,
        &mut backend,
        &mut audit,
        Some(Role::Operator),
        9,
        false,
    );
    for mode in 0..=2u8 {
        let r = handle_onnx_model_remove(&mut c, b"foo.onnx", mode);
        if mode == 2 {
            // operator + allow_operator_unhide=false → denied
            assert_eq!(r, ERRNO_EPERM, "mode 2 with policy off");
        } else {
            assert_eq!(r, ERRNO_EPERM, "mode {mode}");
        }
    }
    // Three deny records appended.
    assert_eq!(audit.events.len(), 3);
    for e in &audit.events {
        assert!(matches!(e, OverlayAuditEvent::DenyModelRemove { .. }));
    }
}

#[test]
fn spec_operator_may_unhide_when_policy_permits() {
    let mut locks = OverlayLockTable::new();
    let mut backend = MockOverlayBackend::new(100_000);
    backend.whiteouts.insert("foo.onnx".to_string(), ());
    let mut audit = CapturingOverlayAuditSink::new();
    let mut c = ctx(
        &mut locks,
        &mut backend,
        &mut audit,
        Some(Role::Operator),
        9,
        true, // allow_operator_unhide=true
    );
    let r = handle_onnx_model_remove(&mut c, b"foo.onnx", 2);
    assert_eq!(r, 0);
    assert!(!backend.whiteouts.contains_key("foo.onnx"));
    match &audit.events[0] {
        OverlayAuditEvent::ModelUnhidden { who, name, role } => {
            assert_eq!(*who, 9);
            assert_eq!(name, "foo.onnx");
            assert_eq!(*role, Role::Operator.as_u8());
        }
        e => panic!("unexpected: {:?}", e),
    }
}

// ─── Per-name advisory lock contention ───────────────────────────────────────

#[test]
fn concurrent_adds_for_same_name_reject_second_with_ebusy() {
    let mut locks = OverlayLockTable::new();
    let other = SessionId::from_raw(0x0200_0002);
    assert!(locks.try_acquire(b"foo.onnx", other).is_ok());

    let mut backend = MockOverlayBackend::new(100_000);
    let mut audit = CapturingOverlayAuditSink::new();
    let mut c = ctx(
        &mut locks,
        &mut backend,
        &mut audit,
        Some(Role::Operator),
        7,
        true,
    );
    let mut src = SliceSource::new(b"x");
    let r = handle_onnx_model_add(&mut c, b"foo.onnx", &mut src, 1, None);
    assert_eq!(r, ERRNO_EBUSY);
}

// ─── Capacity-cap audit on overflow ──────────────────────────────────────────

#[test]
fn capacity_cap_audits_capacity_exceeded_and_returns_enospc() {
    let mut locks = OverlayLockTable::new();
    let mut backend = MockOverlayBackend::new(10);
    let mut audit = CapturingOverlayAuditSink::new();
    let mut c = ctx(
        &mut locks,
        &mut backend,
        &mut audit,
        Some(Role::Operator),
        7,
        true,
    );
    let big = [0xAAu8; 500];
    let mut src = SliceSource::new(&big);
    let r = handle_onnx_model_add(&mut c, b"big.onnx", &mut src, 500, None);
    assert_eq!(r, ERRNO_ENOSPC);
    assert!(matches!(
        audit.events[0],
        OverlayAuditEvent::ModelAddCapacityExceeded { ref name, declared: 500, .. }
            if name == "big.onnx"
    ));
}

// ─── Full RBAC matrix sweep (roles × verbs) ──────────────────────────────────

#[test]
fn full_rbac_matrix_sweep() {
    // 4 callers (Root, Operator, Viewer, none) × 4 actions
    // (add, remove mode 0/1/2). Every cell asserts the spec's
    // permitted/denied outcome.
    let cases = [
        // (role, allow_unhide, verb_kind, should_succeed)
        (Some(Role::Root), false, "add", true),
        (Some(Role::Root), false, "rm0", true),
        (Some(Role::Root), false, "rm1", true),
        (Some(Role::Root), false, "rm2", true),
        (Some(Role::Operator), false, "add", true),
        (Some(Role::Operator), false, "rm0", false),
        (Some(Role::Operator), false, "rm1", false),
        (Some(Role::Operator), false, "rm2", false),
        (Some(Role::Operator), true, "rm2", true),
        (Some(Role::Viewer), true, "add", false),
        (Some(Role::Viewer), true, "rm0", false),
        (Some(Role::Viewer), true, "rm1", false),
        (Some(Role::Viewer), true, "rm2", false),
        (None, true, "add", false),
        (None, true, "rm0", false),
        (None, true, "rm1", false),
        (None, true, "rm2", false),
    ];
    for (role, allow_unhide, verb, should_succeed) in cases {
        let mut locks = OverlayLockTable::new();
        let mut backend = MockOverlayBackend::new(100_000);
        backend.whiteouts.insert("foo.onnx".to_string(), ());
        let mut audit = CapturingOverlayAuditSink::new();
        let mut c = ctx(&mut locks, &mut backend, &mut audit, role, 7, allow_unhide);
        let r = match verb {
            "add" => {
                let mut src = SliceSource::new(b"x");
                handle_onnx_model_add(&mut c, b"foo.onnx", &mut src, 1, None)
            }
            "rm0" => handle_onnx_model_remove(&mut c, b"foo.onnx", 0),
            "rm1" => handle_onnx_model_remove(&mut c, b"foo.onnx", 1),
            "rm2" => handle_onnx_model_remove(&mut c, b"foo.onnx", 2),
            _ => unreachable!(),
        };
        if should_succeed {
            assert_eq!(
                r, 0,
                "role={role:?}, verb={verb}, allow_unhide={allow_unhide}"
            );
        } else {
            assert!(
                r < 0,
                "role={role:?}, verb={verb}, allow_unhide={allow_unhide}"
            );
        }
    }
}
