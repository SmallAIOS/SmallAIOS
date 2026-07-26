// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Container-side runtime glue for the `audit-export` crate.
//!
//! `audit-export` itself is `#![no_std]` and has no I/O. This
//! module — built only when the `audit-export` cargo feature is
//! enabled — implements:
//!
//! - [`FilesystemStateStore`]: real `stage → fsync → rename`
//!   persistence of `last_state.bin` (task 4.5 / 7.7).
//! - [`load_keyfile`]: reads `/data/audit_export/immudb.token`,
//!   refuses any mode laxer than 0600 (task 7.3).
//! - [`load_config`]: parses the TOML at the standard path into
//!   a typed `audit_export::config::Config` and validates it
//!   (task 7.1).
//! - [`ensure_directory`]: idempotent first-boot creation of
//!   `/data/audit_export/` with mode 0700 (task 7.8).
//! - [`transport`] — `TlsGrpcTransport` adapter implementing
//!   `audit_export::immudb::transport::GrpcTransport` over
//!   `smallaios-net::http2` + an abstract
//!   [`transport::TlsStreamLike`] stream. The actual TLS 1.3
//!   client is a separate follow-on
//!   (`tls-tcp-client-v1`) that drops in via the trait.

pub mod transport;

use smallaios_audit_export::config::{AuthMode, Config};
use smallaios_audit_export::immudb::state::{
    PersistedState, StateCodecError, StateStore, StateStoreError,
};
use std::fs;
use std::io::{self, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};

/// Errors surfaced by the container-side audit-export plumbing.
#[derive(Debug)]
pub enum RuntimeError {
    Io(io::Error),
    /// Mode of a sensitive file was laxer than required.
    /// `actual` is the observed mode bitmask; `required` is the
    /// strictest mode we accept (`0o600`).
    InsecurePermissions {
        path: PathBuf,
        actual: u32,
        required: u32,
    },
    /// State-file decode failed.
    StateCorrupt(StateCodecError),
    /// Token file empty or contained illegal bytes.
    InvalidToken,
    /// TOML parse / validation failed.
    BadConfig(String),
}

impl From<io::Error> for RuntimeError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Real filesystem-backed [`StateStore`].
///
/// Honors the spec's stage-and-rename rule: `save()` writes to
/// `<path>.tmp`, calls `fsync` on the body and the parent
/// directory, then atomically renames over `<path>`. A crash
/// mid-write leaves either the previous state intact or no file
/// at all — never a partial write.
pub struct FilesystemStateStore {
    path: PathBuf,
}

impl FilesystemStateStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    fn tmp_path(&self) -> PathBuf {
        let mut s = self.path.clone();
        s.set_extension(match s.extension() {
            Some(e) => {
                let mut owned = e.to_owned();
                owned.push(".tmp");
                owned
            }
            None => std::ffi::OsString::from("tmp"),
        });
        s
    }
}

impl StateStore for FilesystemStateStore {
    fn load(&self) -> Result<Option<PersistedState>, StateStoreError> {
        match fs::read(&self.path) {
            Ok(bytes) => PersistedState::decode(&bytes)
                .map(Some)
                .map_err(|_| StateStoreError::Corrupt),
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
            Err(_) => Err(StateStoreError::Io),
        }
    }

    fn save(&mut self, state: &PersistedState) -> Result<(), StateStoreError> {
        let bytes = state.encode();
        let tmp = self.tmp_path();
        // Stage with 0600 from the start so a crash mid-write
        // never exposes the state to a non-root reader.
        let mut opts = fs::OpenOptions::new();
        opts.write(true).create(true).truncate(true).mode(0o600);
        {
            let mut f = opts.open(&tmp).map_err(|_| StateStoreError::Io)?;
            f.write_all(&bytes).map_err(|_| StateStoreError::Io)?;
            f.sync_all().map_err(|_| StateStoreError::Io)?;
        }
        fs::rename(&tmp, &self.path).map_err(|_| StateStoreError::Io)?;
        // Best-effort directory fsync — ignored if the parent
        // cannot be opened. POSIX guarantees the rename is
        // durable after this; on platforms without dir-fsync
        // the rename alone is the atomicity primitive.
        if let Some(dir) = self.path.parent() {
            if let Ok(d) = fs::File::open(dir) {
                let _ = d.sync_all();
            }
        }
        Ok(())
    }
}

/// Strictest mode we accept for `immudb.token`.
const TOKEN_REQUIRED_MODE: u32 = 0o600;

/// Bytes mask we apply when comparing modes (low 9 bits = rwxrwxrwx).
const PERMISSION_BITS: u32 = 0o777;

/// Load the bearer token from disk. Returns the raw bytes (no
/// trailing newline). Refuses any mode laxer than 0600.
pub fn load_keyfile(path: impl AsRef<Path>) -> Result<Vec<u8>, RuntimeError> {
    let path = path.as_ref();
    let meta = fs::metadata(path)?;
    let mode = meta.mode() & PERMISSION_BITS;
    if mode & !TOKEN_REQUIRED_MODE != 0 {
        return Err(RuntimeError::InsecurePermissions {
            path: path.to_path_buf(),
            actual: mode,
            required: TOKEN_REQUIRED_MODE,
        });
    }
    let mut buf = Vec::new();
    let mut f = fs::File::open(path)?;
    f.read_to_end(&mut buf)?;
    // Strip trailing newlines (`echo "tok" > file` writes one).
    while matches!(buf.last(), Some(b'\n' | b'\r')) {
        buf.pop();
    }
    if buf.is_empty() {
        return Err(RuntimeError::InvalidToken);
    }
    if buf.iter().any(|&b| b < 0x20 || b == 0x7f) {
        return Err(RuntimeError::InvalidToken);
    }
    Ok(buf)
}

/// Idempotent first-boot creation of the audit-export data
/// directory with mode 0700. Returns `Ok(())` whether the
/// directory existed or was just created.
pub fn ensure_directory(path: impl AsRef<Path>) -> Result<(), RuntimeError> {
    let path = path.as_ref();
    if let Err(e) = fs::create_dir(path) {
        if e.kind() != io::ErrorKind::AlreadyExists {
            return Err(RuntimeError::Io(e));
        }
    }
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))?;
    Ok(())
}

/// Parse + validate a TOML payload into a `Config`. Uses the
/// `audit-export::config` validation rules.
///
/// This is a deliberately tiny parser covering only the keys
/// the schema declares; for the production loader path,
/// `mgmt::toml` is the workspace-wide parser that the
/// `ConfigSurface` wiring uses. Here we provide a self-contained
/// loader so the binary can boot the exporter even before that
/// integration lands.
pub fn load_config(toml_bytes: &[u8]) -> Result<Config, RuntimeError> {
    let text =
        std::str::from_utf8(toml_bytes).map_err(|_| RuntimeError::BadConfig("non-UTF-8".into()))?;
    let mut config = Config::default();
    let mut section: Option<String> = None;
    for (lineno, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
            section = Some(rest.trim().to_string());
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| RuntimeError::BadConfig(format!("line {}: no '='", lineno + 1)))?;
        let key = key.trim();
        let value = value.trim();
        let qualified = match &section {
            Some(s) => format!("{s}.{key}"),
            None => key.to_string(),
        };
        apply_kv(&mut config, &qualified, value)
            .map_err(|e| RuntimeError::BadConfig(format!("line {}: {}", lineno + 1, e)))?;
    }
    config
        .validate()
        .map_err(|e| RuntimeError::BadConfig(format!("{e:?}")))?;
    Ok(config)
}

fn apply_kv(cfg: &mut Config, key: &str, value: &str) -> Result<(), String> {
    match key {
        "exporter.enabled" => cfg.enabled = parse_bool(value)?,
        "exporter.endpoint" => cfg.endpoint = parse_string(value)?,
        "exporter.fallback_endpoints" => cfg.fallback_endpoints = parse_string_array(value)?,
        "exporter.auth_mode" => {
            cfg.auth_mode = match parse_string(value)?.as_str() {
                "bearer" => AuthMode::Bearer,
                "mtls" => AuthMode::Mtls,
                other => return Err(format!("invalid auth_mode '{other}'")),
            }
        }
        "exporter.token_path" => cfg.token_path = parse_string(value)?,
        "exporter.state_path" => cfg.state_path = parse_string(value)?,
        "exporter.database" => cfg.database = parse_string(value)?,
        "exporter.batch_size" => cfg.batch_size = parse_u64(value)? as u32,
        "exporter.batch_interval_ms" => cfg.batch_interval_ms = parse_u64(value)?,
        "exporter.buffer_bytes" => cfg.buffer_bytes = parse_u64(value)? as usize,
        "exporter.backoff_initial_ms" => cfg.backoff_initial_ms = parse_u64(value)?,
        "exporter.backoff_cap_ms" => cfg.backoff_cap_ms = parse_u64(value)?,
        "tls.require_pqc" => cfg.require_pqc = parse_bool(value)?,
        "tls.server_pubkey_fingerprint" => cfg.server_pubkey_fingerprint = parse_string(value)?,
        "tls.trust_store_path" => cfg.trust_store_path = parse_string(value)?,
        "record_filter.include_actions" => cfg.include_actions = parse_string_array(value)?,
        "record_filter.exclude_actions" => cfg.exclude_actions = parse_string_array(value)?,
        other => return Err(format!("unknown key '{other}'")),
    }
    Ok(())
}

fn parse_bool(v: &str) -> Result<bool, String> {
    match v {
        "true" => Ok(true),
        "false" => Ok(false),
        other => Err(format!("invalid bool '{other}'")),
    }
}

fn parse_u64(v: &str) -> Result<u64, String> {
    v.parse().map_err(|_| format!("invalid integer '{v}'"))
}

fn parse_string(v: &str) -> Result<String, String> {
    let v = v
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix('"'))
        .ok_or_else(|| format!("expected quoted string, got '{v}'"))?;
    // Minimal escape handling: \\, \", \n, \r, \t.
    let mut out = String::with_capacity(v.len());
    let mut chars = v.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('\\') => out.push('\\'),
            Some('"') => out.push('"'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some(other) => return Err(format!("bad escape '\\{other}'")),
            None => return Err("trailing backslash".into()),
        }
    }
    Ok(out)
}

fn parse_string_array(v: &str) -> Result<Vec<String>, String> {
    let v = v
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| format!("expected array, got '{v}'"))?;
    if v.trim().is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for part in split_top_level(v) {
        out.push(parse_string(part.trim())?);
    }
    Ok(out)
}

fn split_top_level(input: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut in_string = false;
    let mut escape = false;
    for (i, c) in input.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        match c {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            ',' if !in_string => {
                parts.push(&input[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&input[start..]);
    parts
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_audit_export::config::{DEFAULT_STATE_PATH, DEFAULT_TOKEN_PATH};
    use tempfile::TempDir;

    fn write_with_mode(dir: &Path, name: &str, contents: &[u8], mode: u32) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, contents).unwrap();
        fs::set_permissions(&p, fs::Permissions::from_mode(mode)).unwrap();
        p
    }

    #[test]
    fn ensure_directory_creates_with_0700() {
        let dir = TempDir::new().unwrap();
        let target = dir.path().join("audit_export");
        ensure_directory(&target).unwrap();
        let mode = fs::metadata(&target).unwrap().mode() & PERMISSION_BITS;
        assert_eq!(mode, 0o700);
    }

    #[test]
    fn ensure_directory_idempotent() {
        let dir = TempDir::new().unwrap();
        let target = dir.path().join("audit_export");
        ensure_directory(&target).unwrap();
        ensure_directory(&target).unwrap();
        assert!(target.is_dir());
    }

    #[test]
    fn keyfile_loader_strips_trailing_newline() {
        let dir = TempDir::new().unwrap();
        let p = write_with_mode(dir.path(), "tok", b"abc123\n", 0o600);
        let bytes = load_keyfile(&p).unwrap();
        assert_eq!(bytes, b"abc123");
    }

    #[test]
    fn keyfile_loader_rejects_0644() {
        let dir = TempDir::new().unwrap();
        let p = write_with_mode(dir.path(), "tok", b"abc", 0o644);
        match load_keyfile(&p).unwrap_err() {
            RuntimeError::InsecurePermissions {
                actual, required, ..
            } => {
                assert_eq!(actual, 0o644);
                assert_eq!(required, 0o600);
            }
            other => panic!("expected InsecurePermissions, got {other:?}"),
        }
    }

    #[test]
    fn keyfile_loader_accepts_stricter_0400() {
        let dir = TempDir::new().unwrap();
        let p = write_with_mode(dir.path(), "tok", b"abc", 0o400);
        load_keyfile(&p).unwrap();
    }

    #[test]
    fn keyfile_loader_rejects_empty() {
        let dir = TempDir::new().unwrap();
        let p = write_with_mode(dir.path(), "tok", b"", 0o600);
        assert!(matches!(
            load_keyfile(&p).unwrap_err(),
            RuntimeError::InvalidToken
        ));
    }

    #[test]
    fn keyfile_loader_rejects_control_bytes() {
        let dir = TempDir::new().unwrap();
        let p = write_with_mode(dir.path(), "tok", b"abc\x01def", 0o600);
        assert!(matches!(
            load_keyfile(&p).unwrap_err(),
            RuntimeError::InvalidToken
        ));
    }

    #[test]
    fn state_store_round_trip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("last_state.bin");
        let mut store = FilesystemStateStore::new(&path);
        let s = PersistedState {
            db: "smallaios_audit".into(),
            tx_id: 99,
            tx_hash: vec![0xaa; 32],
            public_key: vec![0xbb; 32],
            signature: vec![0xcc; 64],
        };
        store.save(&s).unwrap();
        let loaded = store.load().unwrap().unwrap();
        assert_eq!(loaded, s);
        // tmp file must not linger.
        assert!(!store.tmp_path().exists());
    }

    #[test]
    fn state_store_cold_start_returns_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("missing.bin");
        let store = FilesystemStateStore::new(&path);
        assert!(store.load().unwrap().is_none());
    }

    #[test]
    fn state_store_written_file_is_0600() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("last_state.bin");
        let mut store = FilesystemStateStore::new(&path);
        let s = PersistedState {
            db: "x".into(),
            tx_id: 1,
            tx_hash: vec![],
            public_key: vec![],
            signature: vec![],
        };
        store.save(&s).unwrap();
        let mode = fs::metadata(&path).unwrap().mode() & PERMISSION_BITS;
        assert_eq!(mode, 0o600);
    }

    #[test]
    fn state_store_atomic_replaces_previous() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("last_state.bin");
        let mut store = FilesystemStateStore::new(&path);
        let s1 = PersistedState {
            db: "v1".into(),
            tx_id: 1,
            tx_hash: vec![1; 32],
            public_key: vec![],
            signature: vec![],
        };
        store.save(&s1).unwrap();
        let s2 = PersistedState {
            db: "v2".into(),
            tx_id: 2,
            tx_hash: vec![2; 32],
            public_key: vec![],
            signature: vec![],
        };
        store.save(&s2).unwrap();
        let loaded = store.load().unwrap().unwrap();
        assert_eq!(loaded, s2);
    }

    #[test]
    fn state_store_corrupt_file_returns_err() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("last_state.bin");
        fs::write(&path, b"not a state file at all").unwrap();
        let store = FilesystemStateStore::new(&path);
        assert!(matches!(
            store.load().unwrap_err(),
            StateStoreError::Corrupt
        ));
    }

    #[test]
    fn load_config_minimal_disabled() {
        let toml = b"[exporter]\nenabled = false\n";
        let c = load_config(toml).unwrap();
        assert!(!c.enabled);
    }

    #[test]
    fn load_config_enabled_without_trust_store_rejected() {
        // tls-tcp-client-v1 task 6.4: an enabled exporter with no
        // anchor bundle could only ever fail at connect time, so
        // the loader refuses it up front.
        let toml = br#"
[exporter]
enabled  = true
endpoint = "https://immudb.example.com:3322"

[tls]
server_pubkey_fingerprint = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
"#;
        match load_config(toml) {
            Err(RuntimeError::BadConfig(msg)) => assert!(
                msg.contains("TrustStoreRequired"),
                "unexpected config error: {msg}"
            ),
            other => panic!("expected TrustStoreRequired, got {other:?}"),
        }
    }

    #[test]
    fn load_config_full_enabled() {
        let toml = br#"
[exporter]
enabled        = true
endpoint       = "https://immudb.example.com:3322"
fallback_endpoints = []
auth_mode      = "bearer"
token_path     = "/data/audit_export/immudb.token"
state_path     = "/data/audit_export/last_state.bin"
database       = "smallaios_audit"
batch_size     = 100
batch_interval_ms = 1000
buffer_bytes   = 4194304
backoff_initial_ms = 10000
backoff_cap_ms = 300000

[tls]
require_pqc    = false
server_pubkey_fingerprint = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
trust_store_path = "/data/audit_export/anchors.pem"

[record_filter]
include_actions = []
exclude_actions = ["audit_export_attempt", "immudb_state"]
"#;
        let c = load_config(toml).unwrap();
        assert!(c.enabled);
        assert_eq!(c.endpoint, "https://immudb.example.com:3322");
        assert_eq!(c.trust_store_path, "/data/audit_export/anchors.pem");
        assert_eq!(c.database, "smallaios_audit");
        assert_eq!(c.token_path, DEFAULT_TOKEN_PATH);
        assert_eq!(c.state_path, DEFAULT_STATE_PATH);
        assert_eq!(
            c.exclude_actions,
            vec!["audit_export_attempt", "immudb_state"]
        );
    }

    #[test]
    fn load_config_rejects_invalid_per_validate() {
        // enabled=true but endpoint empty — validate() rejects.
        let toml = b"[exporter]\nenabled = true\n";
        match load_config(toml).unwrap_err() {
            RuntimeError::BadConfig(msg) => assert!(msg.contains("EnabledRequires")),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn load_config_rejects_unknown_key() {
        let toml = b"[exporter]\nmystery = 42\n";
        let err = load_config(toml).unwrap_err();
        assert!(matches!(err, RuntimeError::BadConfig(_)));
    }

    #[test]
    fn load_config_strips_inline_comments() {
        let toml = br#"
[exporter]
enabled = false # trailing comment
database = "foo" # another
"#;
        let c = load_config(toml).unwrap();
        assert!(!c.enabled);
        assert_eq!(c.database, "foo");
    }

    #[test]
    fn load_config_handles_escaped_string() {
        let toml = br#"
[exporter]
enabled = false
endpoint = "x\\y"
"#;
        let c = load_config(toml).unwrap();
        assert_eq!(c.endpoint, "x\\y");
    }
}
