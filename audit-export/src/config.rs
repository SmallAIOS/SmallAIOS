// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Typed configuration model for `/data/audit_export/immudb.toml`.
//!
//! `audit-export` is `#![no_std]` — actual filesystem reads,
//! mode-checks, and `mgmt::surface::ConfigSurface` wiring all
//! live in `container/` (Phase 7's container-side glue lands in
//! a follow-on alongside `audit-export status` / `audit-export
//! config` shell command surfaces). This module provides:
//!
//! - [`Config`] — the typed struct that mirrors `immudb.toml`.
//! - [`Config::validate`] — pure-function check enforcing every
//!   spec rule (`enabled=true` requires `endpoint`; bounded
//!   ranges on `buffer_bytes`, `batch_size`, `batch_interval_ms`,
//!   backoff; non-empty `database`).
//! - [`Config::to_toml_string`] — canonical first-boot
//!   serializer; emits a deterministic byte sequence so the
//!   `config_write` audit records have stable `before` / `after`
//!   diffs.
//! - [`redact_token_value`] — secret-redaction helper applied to
//!   audit records that mention the token-keyfile path. Matches
//!   the `mgmt-config-layout`-added rule.
//! - [`DEFAULT_TOML_PATH`], [`DEFAULT_TOKEN_PATH`],
//!   [`DEFAULT_STATE_PATH`] — well-known paths.

use crate::pipeline::PipelineConfig;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

/// Canonical TOML config file path.
pub const DEFAULT_TOML_PATH: &str = "/data/audit_export/immudb.toml";

/// Canonical bearer-token keyfile path (0600 enforced).
pub const DEFAULT_TOKEN_PATH: &str = "/data/audit_export/immudb.token";

/// Canonical persisted-state path.
pub const DEFAULT_STATE_PATH: &str = "/data/audit_export/last_state.bin";

/// Bounds for buffer_bytes (range 1 MiB – 64 MiB per design.md).
pub const BUFFER_BYTES_MIN: usize = 1024 * 1024;
pub const BUFFER_BYTES_MAX: usize = 64 * 1024 * 1024;

/// Bounds for batch_size (records, range 1–10,000 per spec).
pub const BATCH_SIZE_MIN: u32 = 1;
pub const BATCH_SIZE_MAX: u32 = 10_000;

/// Bounds for batch_interval_ms (range 100–60,000 per spec).
pub const BATCH_INTERVAL_MS_MIN: u64 = 100;
pub const BATCH_INTERVAL_MS_MAX: u64 = 60_000;

/// Bounds for backoff_initial_ms.
pub const BACKOFF_INITIAL_MIN_MS: u64 = 100;
pub const BACKOFF_INITIAL_MAX_MS: u64 = 60_000;

/// Bounds for backoff_cap_ms.
pub const BACKOFF_CAP_MIN_MS: u64 = 1_000;
pub const BACKOFF_CAP_MAX_MS: u64 = 3_600_000;

/// `[exporter] auth_mode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthMode {
    /// v1 default: static bearer token from `token_path`.
    Bearer,
    /// v2 (deferred per D3): mutual TLS using a client cert
    /// referenced elsewhere in `mgmt/policy.toml`.
    Mtls,
}

/// Typed `immudb.toml` schema.
#[derive(Debug, Clone)]
pub struct Config {
    pub enabled: bool,
    pub endpoint: String,
    pub fallback_endpoints: Vec<String>,
    pub auth_mode: AuthMode,
    pub token_path: String,
    pub state_path: String,
    pub database: String,
    pub batch_size: u32,
    pub batch_interval_ms: u64,
    pub buffer_bytes: usize,
    pub backoff_initial_ms: u64,
    pub backoff_cap_ms: u64,

    pub require_pqc: bool,
    pub server_pubkey_fingerprint: String,
    /// Path to a PEM bundle of trust anchors for the immudb server
    /// chain (`[tls] trust_store_path`). Mandatory when `enabled`:
    /// an empty trust store refuses every chain (design.md D5), so
    /// starting the exporter without one could only ever fail at
    /// connect time. `Config::validate` rejects that up front.
    pub trust_store_path: String,

    pub include_actions: Vec<String>,
    pub exclude_actions: Vec<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            enabled: false,
            endpoint: String::new(),
            fallback_endpoints: Vec::new(),
            auth_mode: AuthMode::Bearer,
            token_path: DEFAULT_TOKEN_PATH.into(),
            state_path: DEFAULT_STATE_PATH.into(),
            database: "smallaios_audit".into(),
            batch_size: 100,
            batch_interval_ms: 1_000,
            buffer_bytes: 4 * 1024 * 1024,
            backoff_initial_ms: 10_000,
            backoff_cap_ms: 300_000,
            require_pqc: false,
            server_pubkey_fingerprint: String::new(),
            trust_store_path: String::new(),
            include_actions: Vec::new(),
            exclude_actions: alloc::vec![
                String::from("audit_export_attempt"),
                String::from("audit_export_overflow"),
                String::from("audit_export_proof_failure"),
                String::from("audit_export_rollback_suspected"),
                String::from("audit_export_decode_failure"),
                String::from("audit_export_state_init"),
                String::from("immudb_state"),
            ],
        }
    }
}

/// Reasons a `Config` failed validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// `enabled = true` but `endpoint` is empty.
    EnabledRequiresEndpoint,
    /// `enabled = true` but `server_pubkey_fingerprint` is empty
    /// (D11 — fingerprint pinning is mandatory).
    EnabledRequiresPubkeyFingerprint,
    /// `server_pubkey_fingerprint` is not 64 hex chars.
    PubkeyFingerprintMalformed,
    BatchSizeOutOfRange {
        value: u32,
    },
    BatchIntervalOutOfRange {
        value: u64,
    },
    BufferBytesOutOfRange {
        value: usize,
    },
    BackoffInitialOutOfRange {
        value: u64,
    },
    BackoffCapOutOfRange {
        value: u64,
    },
    BackoffCapBelowInitial,
    DatabaseEmpty,
    /// Endpoint did not start with `https://`.
    EndpointNotHttps,
    /// `auth_mode = "mtls"` requested but feature is off / v2.
    MtlsNotYetSupported,
    /// `enabled = true` but `trust_store_path` is empty. An empty
    /// trust store refuses every chain (design.md D5), so this
    /// would fail at every connect attempt; refuse at config time.
    TrustStoreRequired,
}

impl Config {
    /// Enforce every documented validation rule.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.enabled && self.endpoint.is_empty() {
            return Err(ConfigError::EnabledRequiresEndpoint);
        }
        if self.enabled && !self.endpoint.is_empty() && !self.endpoint.starts_with("https://") {
            return Err(ConfigError::EndpointNotHttps);
        }
        if self.enabled && self.server_pubkey_fingerprint.is_empty() {
            return Err(ConfigError::EnabledRequiresPubkeyFingerprint);
        }
        if !self.server_pubkey_fingerprint.is_empty() && !is_hex_64(&self.server_pubkey_fingerprint)
        {
            return Err(ConfigError::PubkeyFingerprintMalformed);
        }
        if self.enabled && self.trust_store_path.is_empty() {
            return Err(ConfigError::TrustStoreRequired);
        }
        if self.batch_size < BATCH_SIZE_MIN || self.batch_size > BATCH_SIZE_MAX {
            return Err(ConfigError::BatchSizeOutOfRange {
                value: self.batch_size,
            });
        }
        if self.batch_interval_ms < BATCH_INTERVAL_MS_MIN
            || self.batch_interval_ms > BATCH_INTERVAL_MS_MAX
        {
            return Err(ConfigError::BatchIntervalOutOfRange {
                value: self.batch_interval_ms,
            });
        }
        if self.buffer_bytes < BUFFER_BYTES_MIN || self.buffer_bytes > BUFFER_BYTES_MAX {
            return Err(ConfigError::BufferBytesOutOfRange {
                value: self.buffer_bytes,
            });
        }
        if self.backoff_initial_ms < BACKOFF_INITIAL_MIN_MS
            || self.backoff_initial_ms > BACKOFF_INITIAL_MAX_MS
        {
            return Err(ConfigError::BackoffInitialOutOfRange {
                value: self.backoff_initial_ms,
            });
        }
        if self.backoff_cap_ms < BACKOFF_CAP_MIN_MS || self.backoff_cap_ms > BACKOFF_CAP_MAX_MS {
            return Err(ConfigError::BackoffCapOutOfRange {
                value: self.backoff_cap_ms,
            });
        }
        if self.backoff_cap_ms < self.backoff_initial_ms {
            return Err(ConfigError::BackoffCapBelowInitial);
        }
        if self.database.is_empty() {
            return Err(ConfigError::DatabaseEmpty);
        }
        if self.auth_mode == AuthMode::Mtls {
            return Err(ConfigError::MtlsNotYetSupported);
        }
        Ok(())
    }

    /// Build a [`PipelineConfig`] from this config.
    pub fn to_pipeline_config(&self) -> PipelineConfig {
        PipelineConfig {
            batch_size: self.batch_size,
            batch_interval_ms: self.batch_interval_ms,
            buffer_bytes: self.buffer_bytes,
            backoff_initial_ms: self.backoff_initial_ms,
            backoff_cap_ms: self.backoff_cap_ms,
            overflow_log_interval_ms: 60_000,
            include_actions: self.include_actions.clone(),
            exclude_actions: self.exclude_actions.clone(),
        }
    }

    /// Render the canonical first-boot TOML.
    pub fn to_toml_string(&self) -> String {
        let mut s = String::new();
        s.push_str("# /data/audit_export/immudb.toml — generated by audit-export\n");
        s.push_str("# See openspec/changes/verifiable-audit-log-v1/ for schema docs.\n\n");
        s.push_str("[exporter]\n");
        s.push_str(&format!("enabled        = {}\n", self.enabled));
        s.push_str(&format!(
            "endpoint       = \"{}\"\n",
            escape_toml(&self.endpoint)
        ));
        s.push_str("fallback_endpoints = [");
        for (i, e) in self.fallback_endpoints.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("\"{}\"", escape_toml(e)));
        }
        s.push_str("]\n");
        s.push_str(&format!(
            "auth_mode      = \"{}\"\n",
            match self.auth_mode {
                AuthMode::Bearer => "bearer",
                AuthMode::Mtls => "mtls",
            }
        ));
        s.push_str(&format!(
            "token_path     = \"{}\"\n",
            escape_toml(&self.token_path)
        ));
        s.push_str(&format!(
            "state_path     = \"{}\"\n",
            escape_toml(&self.state_path)
        ));
        s.push_str(&format!(
            "database       = \"{}\"\n",
            escape_toml(&self.database)
        ));
        s.push_str(&format!("batch_size     = {}\n", self.batch_size));
        s.push_str(&format!("batch_interval_ms = {}\n", self.batch_interval_ms));
        s.push_str(&format!("buffer_bytes   = {}\n", self.buffer_bytes));
        s.push_str(&format!(
            "backoff_initial_ms = {}\n",
            self.backoff_initial_ms
        ));
        s.push_str(&format!("backoff_cap_ms = {}\n", self.backoff_cap_ms));
        s.push_str("\n[tls]\n");
        s.push_str(&format!("require_pqc    = {}\n", self.require_pqc));
        s.push_str(&format!(
            "server_pubkey_fingerprint = \"{}\"\n",
            escape_toml(&self.server_pubkey_fingerprint)
        ));
        s.push_str(&format!(
            "trust_store_path = \"{}\"\n",
            escape_toml(&self.trust_store_path)
        ));
        s.push_str("\n[record_filter]\n");
        s.push_str("include_actions = [");
        for (i, a) in self.include_actions.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("\"{}\"", escape_toml(a)));
        }
        s.push_str("]\n");
        s.push_str("exclude_actions = [");
        for (i, a) in self.exclude_actions.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&format!("\"{}\"", escape_toml(a)));
        }
        s.push_str("]\n");
        s
    }
}

fn is_hex_64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Minimal TOML string-escape: only `"` and `\` plus control bytes
/// need escaping for our config (which never contains arbitrary
/// user free text).
fn escape_toml(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

/// Redact bytes appearing inside an audit record's `before` /
/// `after` JSON when the record refers to the token keyfile.
///
/// Replaces the value with a fixed-length placeholder string
/// `<redacted:N>` where `N` is the byte length of the original
/// (so an auditor can tell whether the token actually rotated
/// vs. a no-op write that left it unchanged, without ever
/// learning the bytes themselves).
pub fn redact_token_value(raw: &[u8]) -> String {
    format!("<redacted:{}>", raw.len())
}

/// True iff `path` should trigger the secret-redaction rule
/// (currently: anything ending in `/immudb.token`).
pub fn is_secret_path(path: &str) -> bool {
    path.ends_with("/immudb.token") || path == "immudb.token"
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok() -> Config {
        Config {
            enabled: true,
            endpoint: "https://immudb.example.com:3322".into(),
            server_pubkey_fingerprint: "a".repeat(64),
            trust_store_path: "/data/audit_export/anchors.pem".into(),
            ..Config::default()
        }
    }

    #[test]
    fn default_disabled_validates() {
        Config::default().validate().unwrap();
    }

    #[test]
    fn enabled_without_endpoint_rejected() {
        let c = Config {
            enabled: true,
            server_pubkey_fingerprint: "a".repeat(64),
            ..Config::default()
        };
        assert_eq!(
            c.validate().unwrap_err(),
            ConfigError::EnabledRequiresEndpoint
        );
    }

    #[test]
    fn enabled_without_https_rejected() {
        let mut c = ok();
        c.endpoint = "http://immudb.example.com:3322".into();
        assert_eq!(c.validate().unwrap_err(), ConfigError::EndpointNotHttps);
    }

    #[test]
    fn enabled_without_pubkey_fingerprint_rejected() {
        let mut c = ok();
        c.server_pubkey_fingerprint = String::new();
        assert_eq!(
            c.validate().unwrap_err(),
            ConfigError::EnabledRequiresPubkeyFingerprint
        );
    }

    #[test]
    fn enabled_without_trust_store_rejected() {
        let mut c = ok();
        c.trust_store_path = String::new();
        assert_eq!(c.validate().unwrap_err(), ConfigError::TrustStoreRequired);
    }

    #[test]
    fn disabled_without_trust_store_is_fine() {
        // The exporter is off, so there is nothing to connect to and
        // no anchors to require.
        let mut c = ok();
        c.enabled = false;
        c.trust_store_path = String::new();
        c.validate().unwrap();
    }

    #[test]
    fn malformed_pubkey_fingerprint_rejected() {
        let mut c = ok();
        c.server_pubkey_fingerprint = "not-hex".into();
        assert_eq!(
            c.validate().unwrap_err(),
            ConfigError::PubkeyFingerprintMalformed
        );
    }

    #[test]
    fn pubkey_fingerprint_wrong_length_rejected() {
        let mut c = ok();
        c.server_pubkey_fingerprint = "ab".into();
        assert_eq!(
            c.validate().unwrap_err(),
            ConfigError::PubkeyFingerprintMalformed
        );
    }

    #[test]
    fn batch_size_zero_rejected() {
        let mut c = ok();
        c.batch_size = 0;
        assert!(matches!(
            c.validate().unwrap_err(),
            ConfigError::BatchSizeOutOfRange { .. }
        ));
    }

    #[test]
    fn batch_size_too_large_rejected() {
        let mut c = ok();
        c.batch_size = 20_000;
        assert!(matches!(
            c.validate().unwrap_err(),
            ConfigError::BatchSizeOutOfRange { .. }
        ));
    }

    #[test]
    fn buffer_below_floor_rejected() {
        let mut c = ok();
        c.buffer_bytes = 100;
        assert!(matches!(
            c.validate().unwrap_err(),
            ConfigError::BufferBytesOutOfRange { .. }
        ));
    }

    #[test]
    fn buffer_above_ceiling_rejected() {
        let mut c = ok();
        c.buffer_bytes = 100 * 1024 * 1024;
        assert!(matches!(
            c.validate().unwrap_err(),
            ConfigError::BufferBytesOutOfRange { .. }
        ));
    }

    #[test]
    fn backoff_cap_below_initial_rejected() {
        let mut c = ok();
        c.backoff_initial_ms = 30_000;
        c.backoff_cap_ms = 10_000;
        assert_eq!(
            c.validate().unwrap_err(),
            ConfigError::BackoffCapBelowInitial
        );
    }

    #[test]
    fn empty_database_rejected() {
        let mut c = ok();
        c.database = String::new();
        assert_eq!(c.validate().unwrap_err(), ConfigError::DatabaseEmpty);
    }

    #[test]
    fn mtls_in_v1_rejected() {
        let mut c = ok();
        c.auth_mode = AuthMode::Mtls;
        assert_eq!(c.validate().unwrap_err(), ConfigError::MtlsNotYetSupported);
    }

    #[test]
    fn batch_interval_floor_enforced() {
        let mut c = ok();
        c.batch_interval_ms = 50; // below 100 ms floor
        assert!(matches!(
            c.validate().unwrap_err(),
            ConfigError::BatchIntervalOutOfRange { .. }
        ));
    }

    #[test]
    fn to_pipeline_config_carries_filters() {
        let mut c = ok();
        c.include_actions = alloc::vec!["a".into()];
        c.exclude_actions = alloc::vec!["b".into()];
        let pc = c.to_pipeline_config();
        assert_eq!(pc.batch_size, c.batch_size);
        assert_eq!(pc.buffer_bytes, c.buffer_bytes);
        assert_eq!(pc.include_actions, c.include_actions);
        assert_eq!(pc.exclude_actions, c.exclude_actions);
    }

    #[test]
    fn toml_render_is_deterministic() {
        let c = Config::default();
        let a = c.to_toml_string();
        let b = c.to_toml_string();
        assert_eq!(a, b);
    }

    #[test]
    fn toml_render_contains_key_fields() {
        let c = Config::default();
        let s = c.to_toml_string();
        assert!(s.contains("[exporter]"));
        assert!(s.contains("[tls]"));
        assert!(s.contains("[record_filter]"));
        assert!(s.contains("enabled        = false"));
        assert!(s.contains("database       = \"smallaios_audit\""));
        assert!(s.contains("audit_export_attempt"));
        assert!(s.contains("immudb_state"));
    }

    #[test]
    fn toml_escapes_quote_and_backslash() {
        let c = Config {
            endpoint: String::from("https://w\"e\\ird"),
            ..Config::default()
        };
        let s = c.to_toml_string();
        assert!(s.contains("https://w\\\"e\\\\ird"));
    }

    #[test]
    fn redact_token_value_uses_length() {
        let r = redact_token_value(b"hello-world");
        assert_eq!(r, "<redacted:11>");
    }

    #[test]
    fn redact_token_value_never_leaks_bytes() {
        let token = b"super-secret-bearer-token-9001";
        let r = redact_token_value(token);
        // Render MUST NOT contain any byte from the token.
        for slice_start in 0..token.len() {
            for slice_end in (slice_start + 4)..=token.len() {
                let frag = core::str::from_utf8(&token[slice_start..slice_end]).unwrap();
                assert!(
                    !r.contains(frag),
                    "redacted value leaked fragment '{}'",
                    frag
                );
            }
        }
    }

    #[test]
    fn is_secret_path_recognizes_token() {
        assert!(is_secret_path("/data/audit_export/immudb.token"));
        assert!(is_secret_path("immudb.token"));
        assert!(!is_secret_path("/data/audit_export/immudb.toml"));
        assert!(!is_secret_path("/data/audit_export/last_state.bin"));
    }
}
