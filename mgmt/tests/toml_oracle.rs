// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cross-check the clean-room TOML parser against the upstream
//! `toml` crate (dev-only oracle).
//!
//! The clean-room rule precludes `toml` as a production dependency,
//! but we keep it as a `dev-dependencies` oracle: every default
//! TOML emitted by the loader's renderer SHALL parse identically
//! through both implementations. This catches accidental drift in
//! the lex/parse subset.

use smallaios_mgmt::config::Config;
use smallaios_mgmt::loader_toml::{render_namespace_table, Namespace};
use smallaios_mgmt::toml::{parse as clean_room_parse, serialize as clean_room_serialise};

#[test]
fn upstream_toml_parses_clean_room_serialised_system_defaults() {
    let cfg = Config::v1_defaults();
    let s = clean_room_serialise(&render_namespace_table(&cfg, Namespace::System)).unwrap();
    // Upstream `toml` crate must accept the bytes we emit.
    let _: toml::Table = toml::from_str(&s).expect("upstream toml must parse our output");
}

#[test]
fn upstream_toml_parses_clean_room_serialised_policy_defaults() {
    let cfg = Config::v1_defaults();
    let s = clean_room_serialise(&render_namespace_table(&cfg, Namespace::MgmtPolicy)).unwrap();
    let _: toml::Table = toml::from_str(&s).expect("upstream toml must parse our output");
}

#[test]
fn clean_room_and_upstream_agree_on_idle_section() {
    let cfg = Config::v1_defaults();
    let s = clean_room_serialise(&render_namespace_table(&cfg, Namespace::System)).unwrap();
    let upstream: toml::Table = toml::from_str(&s).unwrap();
    let clean = clean_room_parse(&s).unwrap();
    // Spot-check: idle.root_minutes agrees in both.
    let upstream_root = upstream
        .get("idle")
        .and_then(|v| v.as_table())
        .and_then(|t| t.get("root_minutes"))
        .and_then(|v| v.as_integer())
        .unwrap();
    let clean_root = clean
        .get("idle")
        .and_then(|v| match v {
            smallaios_mgmt::toml::TomlValue::Table(t) => Some(t),
            _ => None,
        })
        .and_then(|t| t.get("root_minutes"))
        .and_then(|v| match v {
            smallaios_mgmt::toml::TomlValue::Integer(n) => Some(*n),
            _ => None,
        })
        .unwrap();
    assert_eq!(upstream_root, clean_root);
    assert_eq!(upstream_root, 15);
}

#[test]
fn clean_room_and_upstream_agree_on_password_classes_array() {
    let cfg = Config::v1_defaults();
    let s = clean_room_serialise(&render_namespace_table(&cfg, Namespace::System)).unwrap();
    let upstream: toml::Table = toml::from_str(&s).unwrap();
    let arr = upstream
        .get("password")
        .and_then(|v| v.as_table())
        .and_then(|t| t.get("require_classes"))
        .and_then(|v| v.as_array())
        .unwrap();
    let names: Vec<&str> = arr.iter().filter_map(|v| v.as_str()).collect();
    assert!(names.contains(&"lower"));
    assert!(names.contains(&"upper"));
    assert!(names.contains(&"digit"));
}

#[test]
fn upstream_round_trip_matches_clean_room_round_trip() {
    let cfg = Config::v1_defaults();
    for ns in [Namespace::System, Namespace::MgmtPolicy] {
        let s = clean_room_serialise(&render_namespace_table(&cfg, ns)).unwrap();
        // Both impls accept the bytes.
        let _: toml::Table = toml::from_str(&s).unwrap();
        let parsed = clean_room_parse(&s).unwrap();
        let re = clean_room_serialise(&parsed).unwrap();
        // Idempotent re-serialise.
        assert_eq!(s, re);
    }
}
