// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Universal-exposure CI walker.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-config-surface-trait/spec.md`
//! Q3 — "Build fails on missing handler".
//!
//! This integration test enumerates every active [`mgmt::ConfigSurface`]
//! implementation and asserts that every field in
//! [`mgmt::config::registry::CONFIG_FIELDS`] is reachable via the
//! surface's `read` method. Adding a new `Config` field without
//! adding a handler on every active surface SHALL fail this test.
//!
//! v1's surface count is 1 (TOML loader). Phase 7 introduces the
//! Zenoh admin surface; when that lands, the test will multiply by
//! the new surface and continue to enforce the invariant.

#![allow(clippy::result_large_err)]

use smallaios_auth::atomic_write::TestAtomicWriter;
use smallaios_mgmt::config::registry;
use smallaios_mgmt::config::Config;
use smallaios_mgmt::loader_toml::{
    FirstBootPolicy, InMemoryReadBackend, Namespace, TomlLoader, V1_LAYOUT,
};
use smallaios_mgmt::surface::{ConfigPath, ConfigSurface, SurfaceKind};

/// Build a TOML loader pre-loaded with [`Config::v1_defaults`] so
/// reads succeed without depending on a runtime FS backend.
fn make_toml_loader() -> TomlLoader<InMemoryReadBackend, TestAtomicWriter> {
    let reader = InMemoryReadBackend::new();
    // Seed every layout entry with valid defaults so strict load
    // succeeds. We seed at the declared mode for each entry.
    use smallaios_mgmt::loader_toml::render_namespace_table;
    use smallaios_mgmt::toml::serialize as toml_serialize;
    let cfg = Config::v1_defaults();
    for entry in V1_LAYOUT {
        let abs = format!("/data/{}", entry.relative);
        let bytes: Vec<u8> =
            if entry.namespace == Namespace::System || entry.namespace == Namespace::MgmtPolicy {
                let table = render_namespace_table(&cfg, entry.namespace);
                toml_serialize(&table).unwrap().into_bytes()
            } else {
                Vec::new()
            };
        reader.seed_with_mode(&abs, &bytes, entry.declared_mode);
    }
    let writer = TestAtomicWriter::new();
    let mut loader = TomlLoader::new(reader, writer, "/data");
    loader.load(FirstBootPolicy::Strict).unwrap();
    loader
}

#[test]
fn every_config_field_reachable_via_toml_surface() {
    let loader = make_toml_loader();
    let mut missing: Vec<&'static str> = Vec::new();
    for field in registry::CONFIG_FIELDS {
        let path = ConfigPath::new(field.path)
            .unwrap_or_else(|e| panic!("registry field {} has invalid path: {:?}", field.path, e));
        if let Err(e) = loader.read(&path) {
            eprintln!(
                "TOML surface cannot read {}: {:?} (kind={:?}, scope={:?})",
                field.path, e, field.kind, field.scope
            );
            missing.push(field.path);
        }
    }
    assert!(
        missing.is_empty(),
        "{} field(s) unreachable via TOML surface: {:?}",
        missing.len(),
        missing
    );
}

#[test]
fn every_config_field_writeable_via_toml_surface() {
    // Stronger check: every field must accept *some* legal value
    // through the trait surface. We use the field's existing default
    // value (round-trip), which always satisfies validators.
    let mut loader = make_toml_loader();
    let mut failures: Vec<(&'static str, String)> = Vec::new();
    for field in registry::CONFIG_FIELDS {
        let path = ConfigPath::new(field.path).unwrap();
        let current = match loader.read(&path) {
            Ok(v) => v,
            Err(e) => {
                failures.push((field.path, format!("read failed: {:?}", e)));
                continue;
            }
        };
        if let Err(e) = loader.write(&path, current.clone(), "ci@toml") {
            failures.push((field.path, format!("write failed: {:?}", e)));
        }
    }
    assert!(
        failures.is_empty(),
        "{} field(s) reject round-trip writes via TOML surface: {:?}",
        failures.len(),
        failures
    );
}

#[test]
fn every_config_field_subscribable_via_toml_surface() {
    let loader = make_toml_loader();
    for field in registry::CONFIG_FIELDS {
        let path = ConfigPath::new(field.path).unwrap();
        loader
            .subscribe(&path)
            .unwrap_or_else(|e| panic!("subscribe failed for {}: {:?}", field.path, e));
    }
}

#[test]
fn surface_kind_universe_matches_active_surfaces() {
    // v1: only the TOML surface is active. Phase 7 adds Zenoh admin.
    // This test asserts the universe matches v1 exactly so a future
    // surface addition forces a deliberate update of the walker
    // alongside the loader.
    let active: Vec<SurfaceKind> = active_surfaces();
    assert_eq!(active.len(), 1, "v1 expects exactly one active surface");
    assert!(active.contains(&SurfaceKind::Toml));
    // Tty + Zenoh come in later phases.
    assert!(!active.contains(&SurfaceKind::Tty));
    assert!(!active.contains(&SurfaceKind::Zenoh));
}

/// List of surfaces the universal-exposure walker iterates. v1 has
/// only the TOML loader; Phase 7 (Zenoh admin) extends this list.
///
/// Phase 7 wiring placeholder: when `bus-zenoh` lands, append
/// `SurfaceKind::Zenoh` here behind a `#[cfg(feature = "bus-zenoh")]`
/// gate.
fn active_surfaces() -> Vec<SurfaceKind> {
    vec![SurfaceKind::Toml]
}

#[test]
fn no_field_is_universally_orphaned() {
    // Symmetric check: walk active_surfaces() x CONFIG_FIELDS and
    // ensure each (field, surface) pair has a reachable handler.
    // For v1 this collapses to "every field is reachable on TOML"
    // but the loop structure is here so adding Zenoh in Phase 7 is
    // a one-line change.
    let loader = make_toml_loader();
    for surface in active_surfaces() {
        match surface {
            SurfaceKind::Toml => {
                for field in registry::CONFIG_FIELDS {
                    let path = ConfigPath::new(field.path).unwrap();
                    loader.read(&path).unwrap_or_else(|e| {
                        panic!(
                            "TOML surface cannot read {} (universal-exposure): {:?}",
                            field.path, e
                        )
                    });
                }
            }
            SurfaceKind::Tty | SurfaceKind::Zenoh => {
                // Inactive in v1 — nothing to check yet.
            }
        }
    }
}

#[test]
fn registry_field_count_matches_walker_iterations() {
    let loader = make_toml_loader();
    let count = registry::CONFIG_FIELDS.len();
    let mut iterated = 0;
    for field in registry::CONFIG_FIELDS {
        let path = ConfigPath::new(field.path).unwrap();
        let _ = loader.read(&path).unwrap();
        iterated += 1;
    }
    assert_eq!(iterated, count);
    assert!(count >= 30, "v1 registry SHOULD have at least 30 fields");
}
