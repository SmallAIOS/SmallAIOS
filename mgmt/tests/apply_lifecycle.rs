// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Apply-lifecycle integration tests for the TOML loader.
//!
//! Spec:
//! - `openspec/changes/management-login-v1/specs/mgmt-config-model/spec.md`
//!   ("Apply lifecycle" — parse, validate, stage, fsync, rename, notify).
//! - `openspec/changes/management-login-v1/specs/mgmt-config-layout/spec.md`
//!   (per-file permissions, file-mode-stricter rule).
//!
//! Uses a SharedFsFixture that pairs a single in-memory store with
//! both the [`ReadBackend`] and the [`AtomicWriter`] sides — mirrors
//! the kernel's runtime where `f2fs::F2fs` provides both halves.

#![allow(clippy::result_large_err)]

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use smallaios_auth::atomic_write::{AtomicWriteError, AtomicWriter};
use smallaios_mgmt::config::Config;
use smallaios_mgmt::loader_toml::{
    audit_identity, render_namespace_table, FirstBootPolicy, Namespace, ReadBackend, TomlLoader,
    V1_LAYOUT,
};
use smallaios_mgmt::surface::{ConfigPath, ConfigSurface, SurfaceKind, Value};
use smallaios_mgmt::toml::{parse as toml_parse, serialize as toml_serialize};

/// Shared in-memory FS that satisfies both [`ReadBackend`] and
/// [`AtomicWriter`].
#[derive(Clone)]
struct SharedFs {
    files: Rc<RefCell<BTreeMap<String, Vec<u8>>>>,
    modes: Rc<RefCell<BTreeMap<String, u16>>>,
    write_mode: u16,
}

impl SharedFs {
    fn new(write_mode: u16) -> Self {
        Self {
            files: Rc::new(RefCell::new(BTreeMap::new())),
            modes: Rc::new(RefCell::new(BTreeMap::new())),
            write_mode,
        }
    }

    fn seed_with_mode(&self, path: &str, bytes: &[u8], mode: u16) {
        self.files
            .borrow_mut()
            .insert(path.to_string(), bytes.to_vec());
        self.modes.borrow_mut().insert(path.to_string(), mode);
    }

    fn read(&self, path: &str) -> Option<Vec<u8>> {
        self.files.borrow().get(path).cloned()
    }
}

impl ReadBackend for SharedFs {
    fn read_file(&self, path: &str) -> Result<Option<Vec<u8>>, &'static str> {
        Ok(self.files.borrow().get(path).cloned())
    }
    fn mode_of(&self, path: &str) -> Result<Option<u16>, &'static str> {
        Ok(self.modes.borrow().get(path).copied())
    }
}

impl AtomicWriter for SharedFs {
    fn write_atomic(&self, path: &str, contents: &[u8]) -> Result<(), AtomicWriteError> {
        if path.is_empty() {
            return Err(AtomicWriteError::InvalidPath);
        }
        let mode = self.write_mode;
        self.files
            .borrow_mut()
            .insert(path.to_string(), contents.to_vec());
        self.modes.borrow_mut().insert(path.to_string(), mode);
        Ok(())
    }
    fn cleanup_orphan_tmp(&self, dir: &str) -> Result<(), AtomicWriteError> {
        let prefix = if dir.ends_with('/') {
            dir.to_string()
        } else {
            let mut s = String::from(dir);
            s.push('/');
            s
        };
        let to_remove: Vec<String> = self
            .files
            .borrow()
            .keys()
            .filter(|k| k.starts_with(&prefix) && k.ends_with(".tmp"))
            .cloned()
            .collect();
        for k in to_remove {
            self.files.borrow_mut().remove(&k);
        }
        Ok(())
    }
}

fn make_loader_with_skeleton(write_mode: u16) -> (TomlLoader<SharedFs, SharedFs>, SharedFs) {
    let fs = SharedFs::new(write_mode);
    let mut loader = TomlLoader::new(fs.clone(), fs.clone(), "/data");
    loader.load(FirstBootPolicy::Generate).unwrap();
    (loader, fs)
}

#[test]
fn first_boot_generation_creates_all_layout_entries() {
    let (_loader, fs) = make_loader_with_skeleton(0o600);
    for entry in V1_LAYOUT {
        let abs = format!("/data/{}", entry.relative);
        assert!(
            fs.read(&abs).is_some(),
            "first-boot generation missed {}",
            entry.relative
        );
    }
}

#[test]
fn first_boot_generation_writes_idle_defaults_to_disk() {
    let (_loader, fs) = make_loader_with_skeleton(0o600);
    let bytes = fs.read("/data/system.toml").unwrap();
    let s = std::str::from_utf8(&bytes).unwrap();
    let parsed = toml_parse(s).unwrap();
    let idle = parsed.get("idle").unwrap().as_table().unwrap();
    assert_eq!(
        idle.get("root_minutes"),
        Some(&smallaios_mgmt::toml::TomlValue::Integer(15))
    );
}

#[test]
fn live_write_persists_to_layout_file() {
    let (mut loader, fs) = make_loader_with_skeleton(0o600);
    let p = ConfigPath::new("idle.root_minutes").unwrap();
    loader.write(&p, Value::U32(45), "root@toml").unwrap();
    let bytes = fs.read("/data/system.toml").unwrap();
    let s = std::str::from_utf8(&bytes).unwrap();
    let parsed = toml_parse(s).unwrap();
    let idle = parsed.get("idle").unwrap().as_table().unwrap();
    assert_eq!(
        idle.get("root_minutes"),
        Some(&smallaios_mgmt::toml::TomlValue::Integer(45))
    );
}

#[test]
fn boot_only_write_persists_to_disk_but_not_in_memory() {
    let (mut loader, fs) = make_loader_with_skeleton(0o600);
    let p = ConfigPath::new("fs.boot_watchdog_seconds").unwrap();
    loader.write(&p, Value::U32(30), "root@toml").unwrap();
    // On disk: new value (operator can grep the file).
    let bytes = fs.read("/data/system.toml").unwrap();
    let s = std::str::from_utf8(&bytes).unwrap();
    let parsed = toml_parse(s).unwrap();
    let fs_t = parsed.get("fs").unwrap().as_table().unwrap();
    assert_eq!(
        fs_t.get("boot_watchdog_seconds"),
        Some(&smallaios_mgmt::toml::TomlValue::Integer(30))
    );
    // In memory: unchanged.
    assert_eq!(loader.read(&p).unwrap(), Value::U32(60));
    // Pending list records the deferred write.
    assert_eq!(loader.pending_reboot().len(), 1);
}

#[test]
fn validation_rejected_write_leaves_disk_and_memory_unchanged() {
    let (mut loader, fs) = make_loader_with_skeleton(0o600);
    let p = ConfigPath::new("idle.root_minutes").unwrap();
    let before_disk = fs.read("/data/system.toml").unwrap();
    let before_mem = loader.read(&p).unwrap();
    let err = loader.write(&p, Value::U32(0), "root@toml").unwrap_err();
    assert!(matches!(err, smallaios_mgmt::surface::Error::Validation(_)));
    assert_eq!(loader.read(&p).unwrap(), before_mem);
    assert_eq!(fs.read("/data/system.toml").unwrap(), before_disk);
}

#[test]
fn cross_field_violation_leaves_disk_and_memory_unchanged() {
    let (mut loader, fs) = make_loader_with_skeleton(0o600);
    let p = ConfigPath::new("idle.root_minutes").unwrap();
    let before_disk = fs.read("/data/system.toml").unwrap();
    // root > operator violates cross-field invariant.
    let err = loader.write(&p, Value::U32(120), "root@toml").unwrap_err();
    assert!(matches!(err, smallaios_mgmt::surface::Error::Validation(_)));
    assert_eq!(fs.read("/data/system.toml").unwrap(), before_disk);
}

#[test]
fn shadow_at_0644_rejected_when_declared_0640() {
    let fs = SharedFs::new(0o640);
    // Spec: mgmt/policy.toml declared 0o640 — seed at 0o644 to violate.
    fs.seed_with_mode(
        "/data/system.toml",
        toml_serialize(&render_namespace_table(
            &Config::v1_defaults(),
            Namespace::System,
        ))
        .unwrap()
        .as_bytes(),
        0o644,
    );
    fs.seed_with_mode(
        "/data/mgmt/policy.toml",
        toml_serialize(&render_namespace_table(
            &Config::v1_defaults(),
            Namespace::MgmtPolicy,
        ))
        .unwrap()
        .as_bytes(),
        0o644, // declared 0o640 — laxer.
    );
    fs.seed_with_mode("/data/mgmt/zenoh.toml", b"", 0o640);
    fs.seed_with_mode("/data/update/policy.toml", b"", 0o640);
    fs.seed_with_mode("/data/automotive/uds.toml", b"", 0o640);

    let mut loader = TomlLoader::new(fs.clone(), fs.clone(), "/data");
    let err = loader.load(FirstBootPolicy::Strict).unwrap_err();
    assert!(matches!(
        err,
        smallaios_mgmt::loader_toml::LoaderError::ModeViolation { .. }
    ));
}

#[test]
fn stricter_mode_accepted() {
    let fs = SharedFs::new(0o600);
    fs.seed_with_mode(
        "/data/system.toml",
        toml_serialize(&render_namespace_table(
            &Config::v1_defaults(),
            Namespace::System,
        ))
        .unwrap()
        .as_bytes(),
        0o600, // stricter than 0o644
    );
    fs.seed_with_mode(
        "/data/mgmt/policy.toml",
        toml_serialize(&render_namespace_table(
            &Config::v1_defaults(),
            Namespace::MgmtPolicy,
        ))
        .unwrap()
        .as_bytes(),
        0o600, // stricter than 0o640
    );
    fs.seed_with_mode("/data/mgmt/zenoh.toml", b"", 0o600);
    fs.seed_with_mode("/data/update/policy.toml", b"", 0o600);
    fs.seed_with_mode("/data/automotive/uds.toml", b"", 0o600);
    let mut loader = TomlLoader::new(fs.clone(), fs.clone(), "/data");
    loader.load(FirstBootPolicy::Strict).unwrap();
}

#[test]
fn missing_on_strict_reload_returns_error() {
    let fs = SharedFs::new(0o600);
    let mut loader = TomlLoader::new(fs.clone(), fs.clone(), "/data");
    let err = loader.load(FirstBootPolicy::Strict).unwrap_err();
    assert!(matches!(
        err,
        smallaios_mgmt::loader_toml::LoaderError::Missing { .. }
    ));
}

#[test]
fn notify_event_carries_who_and_before_after() {
    let (mut loader, _fs) = make_loader_with_skeleton(0o600);
    let mut sub = loader.raw_subscribe();
    let p = ConfigPath::new("idle.root_minutes").unwrap();
    let id = audit_identity("root", None);
    loader.write(&p, Value::U32(20), &id.render()).unwrap();
    let ev = sub.poll().unwrap().unwrap();
    assert_eq!(ev.path, p);
    assert_eq!(ev.before, Value::U32(15));
    assert_eq!(ev.after, Value::U32(20));
    assert_eq!(ev.who, "root@toml");
}

#[test]
fn audit_identity_renders_role_and_surface() {
    let id = audit_identity("operator", Some("alice".to_string()));
    assert_eq!(id.surface, SurfaceKind::Toml);
    assert_eq!(id.render(), "operator:alice@toml");
}

#[test]
fn live_field_change_visible_via_subsequent_read() {
    let (mut loader, _fs) = make_loader_with_skeleton(0o600);
    let p = ConfigPath::new("password.dictionary_check").unwrap();
    assert_eq!(loader.read(&p).unwrap(), Value::Bool(true));
    loader.write(&p, Value::Bool(false), "root@toml").unwrap();
    assert_eq!(loader.read(&p).unwrap(), Value::Bool(false));
}

#[test]
fn five_consecutive_writes_to_same_field_publish_five_events() {
    let (mut loader, _fs) = make_loader_with_skeleton(0o600);
    let mut sub = loader.raw_subscribe();
    let p = ConfigPath::new("idle.root_minutes").unwrap();
    for n in [10u32, 12, 14, 16, 18] {
        loader.write(&p, Value::U32(n), "root@toml").unwrap();
    }
    let (events, dropped) = sub.drain();
    assert_eq!(dropped, 0);
    assert_eq!(events.len(), 5);
    let lasts: Vec<u32> = events
        .iter()
        .map(|e| match e.after {
            Value::U32(n) => n,
            _ => 0,
        })
        .collect();
    assert_eq!(lasts, vec![10, 12, 14, 16, 18]);
}

#[test]
fn boot_only_writes_accumulate_in_pending_reboot_and_record_who() {
    let (mut loader, _fs) = make_loader_with_skeleton(0o600);
    let id_a = audit_identity("root", None);
    let id_b = audit_identity("operator", Some("bob".into()));
    loader
        .write(
            &ConfigPath::new("fs.boot_watchdog_seconds").unwrap(),
            Value::U32(30),
            &id_a.render(),
        )
        .unwrap();
    loader
        .write(
            &ConfigPath::new("flash.prefer_flash_for_models").unwrap(),
            Value::Bool(false),
            &id_b.render(),
        )
        .unwrap();
    let pending = loader.pending_reboot();
    assert_eq!(pending.len(), 2);
    assert_eq!(pending[0].who, "root@toml");
    assert_eq!(pending[1].who, "operator:bob@toml");
}

#[test]
fn round_trip_through_disk_produces_byte_identical_files() {
    // Spec: serialise(parse(serialise(c))) == serialise(c).
    let cfg = Config::v1_defaults();
    let sys = toml_serialize(&render_namespace_table(&cfg, Namespace::System)).unwrap();
    let parsed = toml_parse(&sys).unwrap();
    let re_serialised = toml_serialize(&parsed).unwrap();
    assert_eq!(sys, re_serialised);
    let pol = toml_serialize(&render_namespace_table(&cfg, Namespace::MgmtPolicy)).unwrap();
    let parsed = toml_parse(&pol).unwrap();
    let re_serialised = toml_serialize(&parsed).unwrap();
    assert_eq!(pol, re_serialised);
}

#[test]
fn validation_failure_does_not_publish_notify_event() {
    let (mut loader, _fs) = make_loader_with_skeleton(0o600);
    let mut sub = loader.raw_subscribe();
    let p = ConfigPath::new("idle.root_minutes").unwrap();
    let _err = loader.write(&p, Value::U32(0), "root@toml").unwrap_err();
    // No event published.
    assert_eq!(sub.poll(), Ok(None));
}

#[test]
fn boot_only_write_does_publish_event_so_audit_records_it() {
    // The event channel sees boot-deferred writes too — the audit
    // ring is the canonical record. The reload semantics only affect
    // when the *in-memory cache* sees the new value.
    let (mut loader, _fs) = make_loader_with_skeleton(0o600);
    let mut sub = loader.raw_subscribe();
    let p = ConfigPath::new("fs.boot_watchdog_seconds").unwrap();
    loader.write(&p, Value::U32(30), "root@toml").unwrap();
    let ev = sub.poll().unwrap().unwrap();
    assert_eq!(ev.after, Value::U32(30));
    // before is the still-current in-memory value (60).
    assert_eq!(ev.before, Value::U32(60));
}

#[test]
fn reloading_after_live_writes_picks_up_new_disk_state() {
    let (mut loader, _fs) = make_loader_with_skeleton(0o600);
    loader
        .write(
            &ConfigPath::new("idle.root_minutes").unwrap(),
            Value::U32(45),
            "root@toml",
        )
        .unwrap();
    // Reload using strict policy — disk is now populated by the
    // generation step + live write.
    loader.load(FirstBootPolicy::Strict).unwrap();
    assert_eq!(
        loader
            .read(&ConfigPath::new("idle.root_minutes").unwrap())
            .unwrap(),
        Value::U32(45)
    );
}

#[test]
fn reloading_does_not_lose_boot_pending_writes() {
    let (mut loader, _fs) = make_loader_with_skeleton(0o600);
    loader
        .write(
            &ConfigPath::new("fs.boot_watchdog_seconds").unwrap(),
            Value::U32(30),
            "root@toml",
        )
        .unwrap();
    // Boot-deferred fields appear on disk even though in-memory is
    // unchanged. After reload, the on-disk new value takes effect
    // (the reboot has happened).
    loader.load(FirstBootPolicy::Strict).unwrap();
    assert_eq!(
        loader
            .read(&ConfigPath::new("fs.boot_watchdog_seconds").unwrap())
            .unwrap(),
        Value::U32(30)
    );
}

#[test]
fn corrupt_file_rejects_load_strict() {
    let fs = SharedFs::new(0o600);
    fs.seed_with_mode("/data/system.toml", b"this is = not = valid", 0o644);
    fs.seed_with_mode(
        "/data/mgmt/policy.toml",
        toml_serialize(&render_namespace_table(
            &Config::v1_defaults(),
            Namespace::MgmtPolicy,
        ))
        .unwrap()
        .as_bytes(),
        0o640,
    );
    fs.seed_with_mode("/data/mgmt/zenoh.toml", b"", 0o640);
    fs.seed_with_mode("/data/update/policy.toml", b"", 0o640);
    fs.seed_with_mode("/data/automotive/uds.toml", b"", 0o640);
    let mut loader = TomlLoader::new(fs.clone(), fs.clone(), "/data");
    let err = loader.load(FirstBootPolicy::Strict).unwrap_err();
    assert!(matches!(
        err,
        smallaios_mgmt::loader_toml::LoaderError::Toml(_)
    ));
}

#[test]
fn full_apply_lifecycle_six_steps_end_to_end() {
    // Spec checklist:
    //   1. parse  -> via Value variant constructed by caller
    //   2. validate -> validate_field + validate_cross_field
    //   3. stage  -> AtomicWriter::write_atomic to <file>.tmp
    //   4. fsync + rename -> AtomicWriter contract
    //   5. notify -> NotifyEvent published
    //   6. (boot vs live) -> live: in-memory updates
    //
    // We can't peek into individual stage steps from outside, but we
    // assert observable post-conditions: notify event arrives, in-
    // memory and disk values agree, validation gate is in front.
    let (mut loader, fs) = make_loader_with_skeleton(0o600);
    let mut sub = loader.raw_subscribe();
    let p = ConfigPath::new("metrics.cpu_interval_ms").unwrap();
    loader.write(&p, Value::U32(2000), "root@toml").unwrap();
    // (1)+(2) implicit: write_atomic was called only because validate passed.
    // (3)+(4) post-condition: file on disk reflects new value.
    let bytes = fs.read("/data/mgmt/policy.toml").unwrap();
    let parsed = toml_parse(std::str::from_utf8(&bytes).unwrap()).unwrap();
    let metrics = parsed.get("metrics").unwrap().as_table().unwrap();
    assert_eq!(
        metrics.get("cpu_interval_ms"),
        Some(&smallaios_mgmt::toml::TomlValue::Integer(2000))
    );
    // (5) notify event delivered.
    let ev = sub.poll().unwrap().unwrap();
    assert_eq!(ev.after, Value::U32(2000));
    // (6) live: in-memory mirror updated.
    assert_eq!(loader.read(&p).unwrap(), Value::U32(2000));
}

#[test]
fn writes_to_separate_namespaces_do_not_collide() {
    // Writing to `idle` mutates system.toml; writing to `metrics`
    // mutates mgmt/policy.toml. Each layout file is rewritten
    // independently.
    let (mut loader, fs) = make_loader_with_skeleton(0o600);
    loader
        .write(
            &ConfigPath::new("idle.root_minutes").unwrap(),
            Value::U32(20),
            "root@toml",
        )
        .unwrap();
    loader
        .write(
            &ConfigPath::new("metrics.cpu_interval_ms").unwrap(),
            Value::U32(2000),
            "root@toml",
        )
        .unwrap();

    let sys =
        toml_parse(std::str::from_utf8(&fs.read("/data/system.toml").unwrap()).unwrap()).unwrap();
    let pol = toml_parse(std::str::from_utf8(&fs.read("/data/mgmt/policy.toml").unwrap()).unwrap())
        .unwrap();
    assert_eq!(
        sys.get("idle")
            .unwrap()
            .as_table()
            .unwrap()
            .get("root_minutes"),
        Some(&smallaios_mgmt::toml::TomlValue::Integer(20))
    );
    assert_eq!(
        pol.get("metrics")
            .unwrap()
            .as_table()
            .unwrap()
            .get("cpu_interval_ms"),
        Some(&smallaios_mgmt::toml::TomlValue::Integer(2000))
    );
    // sys file should NOT contain metrics.
    assert!(!sys.contains_key("metrics"));
    // pol file should NOT contain idle.
    assert!(!pol.contains_key("idle"));
}

#[test]
fn type_mismatch_returns_type_mismatch_surface_error() {
    let (mut loader, _fs) = make_loader_with_skeleton(0o600);
    let p = ConfigPath::new("idle.root_minutes").unwrap();
    // Wrong variant: Bool instead of U32.
    let err = loader
        .write(&p, Value::Bool(true), "root@toml")
        .unwrap_err();
    match err {
        smallaios_mgmt::surface::Error::TypeMismatch { expected, got } => {
            assert_eq!(expected, "u32");
            assert_eq!(got, "bool");
        }
        other => panic!("expected TypeMismatch, got {:?}", other),
    }
}

#[test]
fn unknown_path_returns_not_found_surface_error() {
    let (loader, _fs) = make_loader_with_skeleton(0o600);
    let p = ConfigPath::new("nope.nada").unwrap();
    let err = loader.read(&p).unwrap_err();
    assert!(matches!(err, smallaios_mgmt::surface::Error::NotFound));
}
