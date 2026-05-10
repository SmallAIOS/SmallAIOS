// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Test vectors (sample TOML strings) for [`crate::loader_toml`].
//!
//! Lives in a `*_test_vectors.rs` sibling so codeql-config's
//! `paths-ignore` rule excludes it from analysis — large literal
//! constants here would otherwise trip CodeQL's "magic number"
//! heuristics.

#![cfg(test)]

/// `system.toml` rendered from [`crate::config::Config::v1_defaults`].
///
/// The exact byte-for-byte form depends on the BTreeMap iteration
/// order in the serialiser, so the loader's
/// `round_trip_via_seeded_default_files` test re-renders rather than
/// relying on this constant. This vector is for human-eyeball
/// inspection of "what does a default `system.toml` look like".
pub const SYSTEM_TOML_DEFAULT: &[u8] = b"\
[flash]
prefer_flash_for_models = true
readback_verify = true

[fs]
block_read_timeout_ms = 5000
block_retry_count = 3
boot_watchdog_seconds = 60
cache_data_bytes = 67108864
cache_models_bytes = 67108864

[idle]
operator_minutes = 60
root_minutes = 15
viewer_minutes = 60

[password]
dictionary_check = true
max_length = 128
min_length = 12
require_classes = [\"lower\", \"upper\", \"digit\"]
";

/// `mgmt/policy.toml` rendered from
/// [`crate::config::Config::v1_defaults`]. See
/// [`SYSTEM_TOML_DEFAULT`] for caveats on byte-exactness.
pub const MGMT_POLICY_TOML_DEFAULT: &[u8] = b"\
[audit]
keep_archives = 8
max_total_disk_bytes = 268435456
rotate_age_hours = 24
rotate_size_bytes = 16777216
signed_checkpoint_interval = 1024
signed_checkpoints = true

[metrics]
adaptive_cpu_high_pct = 80
adaptive_cpu_low_pct = 20
cpu_interval_ms = 1000
inference_interval_ms = 500
memory_interval_ms = 5000
network_interval_ms = 1000

[mgmt]
token_two_tier = true
zenoh_per_identity_max = 4
zenoh_total_max = 32

[overlay]
allow_operator_unhide = true
require_signed = true
upper_max_bytes = 268435456
";

/// Hand-crafted "minimal valid" `system.toml` that exercises every
/// scalar variant — used in lex/parse round-trip tests.
pub const MIXED_SCALARS_TOML: &[u8] = b"\
[idle]
root_minutes = 15
operator_minutes = 60
viewer_minutes = 60

[password]
min_length = 12
max_length = 128
dictionary_check = false
require_classes = [\"lower\", \"upper\"]
";

/// Vector with a deliberate unknown key to exercise
/// [`crate::loader_toml::LoaderError::UnknownField`].
pub const UNKNOWN_KEY_TOML: &[u8] = b"\
[idle]
root_minutes = 15
unknown_key = 1
";

/// Vector with deliberately wrong type to exercise
/// [`crate::loader_toml::LoaderError::TypeMismatch`].
pub const TYPE_MISMATCH_TOML: &[u8] = b"\
[idle]
root_minutes = \"not-a-number\"
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixture_constants_are_non_empty() {
        assert!(!SYSTEM_TOML_DEFAULT.is_empty());
        assert!(!MGMT_POLICY_TOML_DEFAULT.is_empty());
        assert!(!MIXED_SCALARS_TOML.is_empty());
        assert!(!UNKNOWN_KEY_TOML.is_empty());
        assert!(!TYPE_MISMATCH_TOML.is_empty());
    }

    #[test]
    fn system_default_parses_through_clean_room_parser() {
        let s = core::str::from_utf8(SYSTEM_TOML_DEFAULT).unwrap();
        let _ = crate::toml::parse(s).expect("default system.toml parses cleanly");
    }

    #[test]
    fn policy_default_parses_through_clean_room_parser() {
        let s = core::str::from_utf8(MGMT_POLICY_TOML_DEFAULT).unwrap();
        let _ = crate::toml::parse(s).expect("default policy.toml parses cleanly");
    }

    #[test]
    fn mixed_scalars_parses() {
        let s = core::str::from_utf8(MIXED_SCALARS_TOML).unwrap();
        let _ = crate::toml::parse(s).unwrap();
    }
}
