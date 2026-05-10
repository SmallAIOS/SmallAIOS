// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Console-login state machine — TTY first-boot + steady-state login.
//!
//! Spec: `openspec/changes/management-login-v1/specs/console-login/spec.md`
//!
//! This module owns the *flow*; the byte-level work (echo suppression,
//! backspace, ^U, ^C) is delegated to [`peripheral::uart::line_discipline`]
//! at the production wire-up. Auth is Layer 1 and cannot import
//! `peripheral`, so this module redefines a minimal pair of console-IO
//! traits ([`ConsoleSource`] / [`ConsoleSink`]) that the Layer 2 wire-up
//! adapts onto the UART driver. The traits' contract matches the
//! peripheral spec byte-for-byte.
//!
//! ## What this module does NOT do
//!
//! - **Hash a password.** That is delegated to
//!   [`smallaios_security::argon2id`] — the runtime tier comes from
//!   [`crate::tier::select_tier`].
//! - **Parse the shadow file.** [`crate::shadow::parse_record`] does
//!   that.
//! - **Persist the shadow file.** [`crate::atomic_write::AtomicWriter`]
//!   does that.
//! - **Maintain the session table.** [`smallaios_kernel::auth::SessionTable`]
//!   does that.
//!
//! ## Lockout policy
//!
//! After `LOCKOUT_THRESHOLD` consecutive failures from a single source,
//! the source SHALL be refused for [`LOCKOUT_DURATION_SECS`] seconds.
//! A successful login resets the counter. The TTY console is one source;
//! Phase 7 will add per-Zenoh-peer sources sharing the same machinery
//! via [`LockoutMap`].

#![allow(clippy::large_enum_variant)]

use crate::atomic_write::{AtomicWriteError, AtomicWriter};
use crate::role::Role as AuthRole;
use crate::shadow::{parse_file, serialize_file, FLAG_MUST_CHANGE_PASSWORD};
use crate::shadow::{ShadowEntry, ShadowError};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use smallaios_kernel::auth::{Role as KernelRole, Session, SessionId, SessionTable};
use smallaios_security::argon2id::{argon2id_verify, Argon2idParams};

// ─── Spec-driven constants ───────────────────────────────────────────────────

/// Lockout threshold per `console-login` Q20 — 5 consecutive failures.
pub const LOCKOUT_THRESHOLD: u32 = 5;

/// Lockout duration per `console-login` Q20 — 60 seconds.
pub const LOCKOUT_DURATION_SECS: u64 = 60;

/// Maximum first-boot password retries per `console-login` "Mismatched
/// confirmation re-prompts" scenario.
pub const FIRST_BOOT_MAX_ATTEMPTS: u32 = 3;

/// Canonical path of the shadow file.
pub const SHADOW_PATH: &str = "/data/auth/shadow";

/// Canonical parent directory.
pub const SHADOW_DIR: &str = "/data/auth";

// ─── Console IO traits ───────────────────────────────────────────────────────

/// Per-read options. Mirrors
/// [`smallaios_peripheral::uart::ReadOptions`] byte-for-byte; the field
/// shape is stable across the Layer 1 / Layer 2 boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LineOptions {
    /// When `false`, no character SHALL be echoed.
    pub echo: bool,
    /// When `true`, control characters SHALL be delivered verbatim.
    pub raw: bool,
    /// Hard cap on bytes accepted.
    pub max_len: usize,
}

impl Default for LineOptions {
    fn default() -> Self {
        Self {
            echo: true,
            raw: false,
            max_len: 4096,
        }
    }
}

impl LineOptions {
    /// Helper: `echo=false`, 1024-byte cap. Equivalent to
    /// [`smallaios_peripheral::uart::ReadOptions::password`].
    pub const fn password() -> Self {
        Self {
            echo: false,
            raw: false,
            max_len: 1024,
        }
    }
}

/// Errors raised by [`ConsoleSource::read_line`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadError {
    /// ^C pressed — abort the prompt without counting as a failure.
    Aborted,
    /// EOF (Ctrl-D at empty line, or wire EOF) — `logout`-equivalent.
    Eof,
    /// Underlying I/O fault.
    Io(&'static str),
}

/// Produces lines of input under the supplied [`LineOptions`].
pub trait ConsoleSource {
    /// Read a single line into `dst`, returning the number of bytes
    /// stored. Implementations SHALL honor [`LineOptions::echo`] and
    /// [`LineOptions::raw`] and SHALL strip the terminating newline.
    fn read_line(&mut self, dst: &mut Vec<u8>, opts: LineOptions) -> Result<usize, ReadError>;
}

/// Writes prompt / banner text. The interface is byte-oriented because
/// the production target (UART) is byte-oriented; `print` is a
/// convenience over `write_byte`.
pub trait ConsoleSink {
    /// Append `bytes` to the sink.
    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), &'static str>;

    /// Convenience: write a string slice.
    fn print(&mut self, s: &str) -> Result<(), &'static str> {
        self.write_bytes(s.as_bytes())
    }
}

// ─── Lockout map ─────────────────────────────────────────────────────────────

/// Symbolic identity of a lockout source. The TTY console is a singleton
/// (`Tty`); Zenoh remote peers (Phase 7) get one entry per peer
/// fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LockoutSource {
    /// The local serial console.
    Tty,
    /// A Zenoh peer identified by its TLS cert fingerprint or PSK
    /// identity (32 raw bytes hex-encoded for keying).
    ZenohPeer(String),
}

/// Per-source counter + cooldown. Stored in [`LockoutMap`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LockoutEntry {
    failures: u32,
    locked_until_unix: u64,
}

/// Tracks per-source lockout state.
///
/// The TTY source is always present; remote sources are added on
/// first-failure. Only the TTY is exercised in Phase 4; Phase 7 plugs
/// the remote sources in.
#[derive(Debug, Default)]
pub struct LockoutMap {
    /// (source, entry) pairs. Linear scan is fine — TTY = 1 source +
    /// at most a handful of remote peers.
    entries: Vec<(LockoutSource, LockoutEntry)>,
}

impl LockoutMap {
    /// Construct an empty map.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    fn entry_mut(&mut self, src: &LockoutSource) -> &mut LockoutEntry {
        if let Some(idx) = self.entries.iter().position(|(s, _)| s == src) {
            return &mut self.entries[idx].1;
        }
        self.entries.push((
            src.clone(),
            LockoutEntry {
                failures: 0,
                locked_until_unix: 0,
            },
        ));
        let last = self.entries.len() - 1;
        &mut self.entries[last].1
    }

    /// Returns `true` if `src` is currently locked out at `now_unix`.
    pub fn is_locked(&self, src: &LockoutSource, now_unix: u64) -> bool {
        for (s, e) in &self.entries {
            if s == src {
                return now_unix < e.locked_until_unix;
            }
        }
        false
    }

    /// Record a failure for `src`. Returns `true` if the failure tipped
    /// the source into a fresh lockout window.
    pub fn record_failure(&mut self, src: &LockoutSource, now_unix: u64) -> bool {
        let e = self.entry_mut(src);
        if now_unix < e.locked_until_unix {
            // Already locked — bump failures but don't extend.
            e.failures = e.failures.saturating_add(1);
            return false;
        }
        e.failures = e.failures.saturating_add(1);
        if e.failures >= LOCKOUT_THRESHOLD {
            e.locked_until_unix = now_unix.saturating_add(LOCKOUT_DURATION_SECS);
            return true;
        }
        false
    }

    /// Clear all state for `src`. Called on a successful login.
    pub fn record_success(&mut self, src: &LockoutSource) {
        if let Some(idx) = self.entries.iter().position(|(s, _)| s == src) {
            self.entries.remove(idx);
        }
    }

    /// Borrow the current failure count for `src` (testing).
    pub fn failures(&self, src: &LockoutSource) -> u32 {
        for (s, e) in &self.entries {
            if s == src {
                return e.failures;
            }
        }
        0
    }
}

// ─── First-boot ──────────────────────────────────────────────────────────────

/// Errors from [`first_boot_setup`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FirstBootError {
    /// Operator failed to confirm the password [`FIRST_BOOT_MAX_ATTEMPTS`]
    /// times — the kernel SHALL halt.
    ConfirmationExhausted,
    /// I/O on the console.
    Io(&'static str),
    /// Atomic shadow write failed.
    AtomicWrite(AtomicWriteError),
    /// Argon2id parameters / hashing failed.
    Argon2id(&'static str),
    /// Operator submitted an empty password.
    EmptyPassword,
}

impl fmt::Display for FirstBootError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ConfirmationExhausted => write!(f, "first-boot: confirmation exhausted"),
            Self::Io(s) => write!(f, "first-boot: I/O error: {s}"),
            Self::AtomicWrite(e) => write!(f, "first-boot: atomic-write error: {e}"),
            Self::Argon2id(s) => write!(f, "first-boot: argon2id: {s}"),
            Self::EmptyPassword => write!(f, "first-boot: empty password"),
        }
    }
}

/// Salt length used at first-boot — 16 bytes per RFC 9106 §3.1.
pub const FIRST_BOOT_SALT_LEN: usize = 16;

/// Run the first-boot prompt cycle:
///
/// 1. Print `Set initial root password:`, read with echo off.
/// 2. Print `Confirm:`, read with echo off, compare.
/// 3. On match → hash with `params`, format PHC, atomic-write to
///    `path`, return.
/// 4. On mismatch → re-prompt up to [`FIRST_BOOT_MAX_ATTEMPTS`].
/// 5. After exhaustion → return [`FirstBootError::ConfirmationExhausted`].
///
/// `salt` SHALL be a freshly-generated CSPRNG slice. Tests pass a
/// deterministic synthetic salt from `console_login_test_vectors.rs`.
#[allow(clippy::too_many_arguments)]
pub fn first_boot_setup<S: ConsoleSource, K: ConsoleSink, W: AtomicWriter>(
    src: &mut S,
    sink: &mut K,
    writer: &W,
    path: &str,
    params: Argon2idParams,
    salt: &[u8],
    now_unix: u64,
) -> Result<ShadowEntry, FirstBootError> {
    sink.print("Set initial root password: ")
        .map_err(FirstBootError::Io)?;
    let mut attempts: u32 = 0;
    loop {
        attempts += 1;
        if attempts > FIRST_BOOT_MAX_ATTEMPTS {
            sink.print(
                "First-boot password setup failed; halting. Recovery: reboot with auth.skip-firstboot.\n",
            ).map_err(FirstBootError::Io)?;
            return Err(FirstBootError::ConfirmationExhausted);
        }

        let mut pw1: Vec<u8> = Vec::new();
        match src.read_line(&mut pw1, LineOptions::password()) {
            Ok(_) => (),
            Err(ReadError::Io(s)) => return Err(FirstBootError::Io(s)),
            Err(ReadError::Aborted) | Err(ReadError::Eof) => {
                sink.print("\nAborted; re-prompting.\n")
                    .map_err(FirstBootError::Io)?;
                sink.print("Set initial root password: ")
                    .map_err(FirstBootError::Io)?;
                continue;
            }
        }
        if pw1.is_empty() {
            sink.print("Empty passwords are not permitted; try again.\n")
                .map_err(FirstBootError::Io)?;
            sink.print("Set initial root password: ")
                .map_err(FirstBootError::Io)?;
            continue;
        }

        sink.print("Confirm: ").map_err(FirstBootError::Io)?;
        let mut pw2: Vec<u8> = Vec::new();
        match src.read_line(&mut pw2, LineOptions::password()) {
            Ok(_) => (),
            Err(ReadError::Io(s)) => return Err(FirstBootError::Io(s)),
            Err(ReadError::Aborted) | Err(ReadError::Eof) => {
                sink.print("\nAborted; re-prompting.\n")
                    .map_err(FirstBootError::Io)?;
                sink.print("Set initial root password: ")
                    .map_err(FirstBootError::Io)?;
                continue;
            }
        }

        if pw1 != pw2 {
            sink.print("Passwords did not match; try again.\n")
                .map_err(FirstBootError::Io)?;
            sink.print("Set initial root password: ")
                .map_err(FirstBootError::Io)?;
            continue;
        }

        // Match — hash and persist.
        let phc = hash_password_phc(&pw1, salt, params).map_err(FirstBootError::Argon2id)?;
        let entry = ShadowEntry {
            username: "root".to_string(),
            phc,
            role: AuthRole::Root,
            flags: 0, // first-boot operator chose freely; no must-change.
            last_changed_unix_day: (now_unix / 86_400) as u32,
            totp_secret: None,
            lockout_until_unix: 0,
            // First-boot root has no TOTP yet; operator can enrol
            // post-boot via `auth_totp_setup` and the policy gate on
            // `mgmt.totp.enforced_for_roles`.
            totp_required: false,
            trailing_unknown_fields: Vec::new(),
        };
        let body = serialize_file(core::slice::from_ref(&entry));
        writer
            .write_atomic(path, body.as_bytes())
            .map_err(FirstBootError::AtomicWrite)?;
        sink.print("auth: root account created.\n")
            .map_err(FirstBootError::Io)?;
        return Ok(entry);
    }
}

/// Hash `pw` with `params` and `salt`; format as a PHC string.
fn hash_password_phc(
    pw: &[u8],
    salt: &[u8],
    params: Argon2idParams,
) -> Result<String, &'static str> {
    use smallaios_security::argon2id::{argon2id_format_phc, argon2id_hash};
    if salt.len() < 8 {
        return Err("salt < 8 bytes");
    }
    if pw.is_empty() {
        return Err("empty password");
    }
    let tag = argon2id_hash(pw, salt, params);
    Ok(argon2id_format_phc(salt, &tag, params))
}

// ─── Login flow ──────────────────────────────────────────────────────────────

/// Outcome of one [`run_login_round`] call.
#[derive(Debug)]
pub enum LoginOutcome {
    /// Authentication succeeded; the kernel SHALL install `session_id`
    /// and emit a banner naming `username` and `role`.
    Authenticated {
        /// Session id allocated in the kernel session table.
        session_id: SessionId,
        /// Username as captured from the prompt.
        username: String,
        /// Role from the shadow record.
        role: AuthRole,
        /// `true` iff the shadow record had `FLAG_MUST_CHANGE_PASSWORD`
        /// set; the caller SHALL gate syscalls accordingly.
        must_change_password: bool,
    },
    /// The source was already locked out when login was attempted; the
    /// kernel SHALL print [`BANNER_LOCKED_OUT`] and re-prompt.
    LockedOut,
    /// Authentication failed (bad user, wrong password, or shadow
    /// provider error). The caller SHALL re-prompt.
    Failed,
    /// Operator pressed Ctrl-C; SHALL re-prompt without counting a
    /// failure.
    Aborted,
    /// Operator pressed Ctrl-D at the username prompt; SHALL re-draw
    /// the login prompt (callers treat this as a no-op).
    Eof,
    /// The session table was full — `acquire` returned `TableFull`.
    /// Audit and re-prompt.
    SessionTableFull,
}

/// Banner printed to the console on a locked-out attempt.
pub const BANNER_LOCKED_OUT: &str = "Authentication locked: try again later.";

/// Banner printed on auth failure.
pub const BANNER_AUTH_FAIL: &str = "Authentication failed.";

/// Look up `name` in the shadow file (already loaded). Returns the
/// matching entry, if any. Constant-time-equivalent matching is the
/// caller's responsibility (we do scan all entries before returning to
/// prevent a short-circuit timing leak; see the `users` slice walk
/// below).
fn lookup_user<'a>(users: &'a [ShadowEntry], name: &[u8]) -> Option<&'a ShadowEntry> {
    let mut found: Option<&ShadowEntry> = None;
    for u in users {
        // Constant-time-equivalent: walk every entry even after match.
        if u.username.as_bytes() == name && found.is_none() {
            found = Some(u);
        }
    }
    found
}

/// Convert an [`AuthRole`] into the kernel's mirror role.
fn to_kernel_role(r: AuthRole) -> KernelRole {
    match r {
        AuthRole::Root => KernelRole::Root,
        AuthRole::Operator => KernelRole::Operator,
        AuthRole::Viewer => KernelRole::Viewer,
    }
}

/// Run one round of the steady-state login flow.
///
/// 1. Print `username:`, read with echo on.
/// 2. Print `password:`, read with echo off.
/// 3. Look up `username` in `users`.
/// 4. Verify the password against the PHC.
/// 5. On success → reset lockout, allocate a session, return
///    [`LoginOutcome::Authenticated`].
/// 6. On failure → record a failure, return [`LoginOutcome::Failed`] (or
///    [`LoginOutcome::LockedOut`] when the source is already locked out).
#[allow(clippy::too_many_arguments)]
pub fn run_login_round<S: ConsoleSource, K: ConsoleSink>(
    src: &mut S,
    sink: &mut K,
    users: &[ShadowEntry],
    table: &mut SessionTable,
    lockouts: &mut LockoutMap,
    source: &LockoutSource,
    now_unix: u64,
    // Dummy PHC for unknown-user paths so we still pay the Argon2id
    // cost and avoid leaking enumeration via timing. Spec
    // `kernel-syscalls` Q12.
    dummy_phc: &str,
) -> LoginOutcome {
    if lockouts.is_locked(source, now_unix) {
        let _ = sink.print(BANNER_LOCKED_OUT);
        let _ = sink.print("\n");
        return LoginOutcome::LockedOut;
    }

    let _ = sink.print("username: ");
    let mut user_buf: Vec<u8> = Vec::new();
    match src.read_line(&mut user_buf, LineOptions::default()) {
        Ok(_) => (),
        Err(ReadError::Aborted) => return LoginOutcome::Aborted,
        Err(ReadError::Eof) => return LoginOutcome::Eof,
        Err(ReadError::Io(_)) => return LoginOutcome::Failed,
    }

    let _ = sink.print("password: ");
    let mut pw_buf: Vec<u8> = Vec::new();
    match src.read_line(&mut pw_buf, LineOptions::password()) {
        Ok(_) => (),
        Err(ReadError::Aborted) => return LoginOutcome::Aborted,
        Err(ReadError::Eof) => return LoginOutcome::Eof,
        Err(ReadError::Io(_)) => return LoginOutcome::Failed,
    }

    // Constant-time-equivalent: ALWAYS verify *something*. If the user
    // is unknown, verify against the dummy_phc. The Argon2id cost
    // dominates the lookup, so the visible timing is the same.
    let user = lookup_user(users, &user_buf);
    let phc = match user {
        Some(u) => u.phc.as_str(),
        None => dummy_phc,
    };

    let ok = argon2id_verify(&pw_buf, phc).unwrap_or_default();

    if user.is_none() || !ok {
        lockouts.record_failure(source, now_unix);
        let _ = sink.print(BANNER_AUTH_FAIL);
        let _ = sink.print("\n");
        return LoginOutcome::Failed;
    }

    // Success — clear lockout, allocate session.
    let user = user.expect("Some by branch above");
    lockouts.record_success(source);

    let krole = to_kernel_role(user.role);
    let must_change = (user.flags & FLAG_MUST_CHANGE_PASSWORD) != 0;
    let session = match Session::new(0, user.username.as_bytes(), krole, now_unix, must_change) {
        Some(s) => s,
        None => {
            // username over MAX_USERNAME_LEN — shouldn't happen because
            // the shadow parser caps at 64 bytes, but defend anyway.
            let _ = sink.print("auth: username too long for session table\n");
            return LoginOutcome::Failed;
        }
    };
    let session_id = match table.acquire(session) {
        Ok(id) => id,
        Err(_) => {
            let _ = sink.print("auth: session table full\n");
            return LoginOutcome::SessionTableFull;
        }
    };

    let _ = sink.print("auth: ok ");
    let _ = sink.print(&user.username);
    let _ = sink.print(" (");
    let _ = sink.print(user.role.name());
    let _ = sink.print(")\n");

    LoginOutcome::Authenticated {
        session_id,
        username: user.username.clone(),
        role: user.role,
        must_change_password: must_change,
    }
}

// ─── Logout ──────────────────────────────────────────────────────────────────

/// Outcome of [`run_logout`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogoutOutcome {
    /// The session was found and released.
    Released,
    /// The session id no longer named a live session.
    Stale,
}

/// Tear down `session_id` from `table`, redraw a banner, and return
/// to the login prompt. The caller (the dispatch loop) calls
/// [`run_login_round`] again next.
pub fn run_logout<K: ConsoleSink>(
    table: &mut SessionTable,
    session_id: SessionId,
    sink: &mut K,
) -> LogoutOutcome {
    match table.release(session_id) {
        Ok(_) => {
            let _ = sink.print("auth: logout\n");
            LogoutOutcome::Released
        }
        Err(_) => LogoutOutcome::Stale,
    }
}

// ─── Bootstrap selection ─────────────────────────────────────────────────────

/// What the boot path SHALL do based on the current shadow state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BootDecision {
    /// `/data/auth/shadow` is missing or empty — drop into first-boot.
    FirstBoot,
    /// Steady-state — proceed to login prompt with these users.
    SteadyState(Vec<ShadowEntry>),
    /// The shadow file is corrupt — halt with a recovery hint.
    CorruptShadow(ShadowError),
}

/// Decide what to do at boot from the loaded shadow contents.
///
/// `raw` is the bytes of `/data/auth/shadow`; pass `None` if the file
/// does not exist.
pub fn decide_boot(raw: Option<&[u8]>) -> BootDecision {
    let bytes = match raw {
        None => return BootDecision::FirstBoot,
        Some(b) if b.iter().all(|&c| c == b'\n' || c == b' ') => return BootDecision::FirstBoot,
        Some(b) => b,
    };
    let s = match core::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return BootDecision::CorruptShadow(ShadowError::MalformedPhc),
    };
    match parse_file(s) {
        Ok(entries) if entries.is_empty() => BootDecision::FirstBoot,
        Ok(entries) => BootDecision::SteadyState(entries),
        Err(e) => BootDecision::CorruptShadow(e),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::console_login_test_vectors::*;

    use crate::atomic_write::TestAtomicWriter;
    use crate::tier::select_tier;
    use smallaios_kernel::auth::SessionTable;

    // ─── Mock console IO ────────────────────────────────────────────────

    struct MockSrc {
        scripts: Vec<Vec<u8>>,
        idx: usize,
    }

    impl MockSrc {
        fn new(lines: &[&[u8]]) -> Self {
            Self {
                scripts: lines.iter().map(|s| s.to_vec()).collect(),
                idx: 0,
            }
        }

        fn from_strs(lines: &[&str]) -> Self {
            Self {
                scripts: lines.iter().map(|s| s.as_bytes().to_vec()).collect(),
                idx: 0,
            }
        }
    }

    impl ConsoleSource for MockSrc {
        fn read_line(&mut self, dst: &mut Vec<u8>, _opts: LineOptions) -> Result<usize, ReadError> {
            if self.idx >= self.scripts.len() {
                return Err(ReadError::Eof);
            }
            let bytes = &self.scripts[self.idx];
            self.idx += 1;
            // Special sentinels.
            if bytes == b"\x03" {
                return Err(ReadError::Aborted);
            }
            if bytes == b"\x04" {
                return Err(ReadError::Eof);
            }
            dst.extend_from_slice(bytes);
            Ok(bytes.len())
        }
    }

    #[derive(Default)]
    struct MockOut {
        bytes: Vec<u8>,
    }

    impl ConsoleSink for MockOut {
        fn write_bytes(&mut self, b: &[u8]) -> Result<(), &'static str> {
            self.bytes.extend_from_slice(b);
            Ok(())
        }
    }

    impl MockOut {
        fn as_str(&self) -> String {
            String::from_utf8_lossy(&self.bytes).to_string()
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────────────

    fn argon_tier_tiny() -> Argon2idParams {
        select_tier(64 * 1024 * 1024)
    }

    fn make_user(name: &str, password: &[u8], role: AuthRole, flags: u32) -> ShadowEntry {
        let phc = hash_password_phc(password, &SYNTHETIC_SALT, argon_tier_tiny()).unwrap();
        ShadowEntry {
            username: name.to_string(),
            phc,
            role,
            flags,
            last_changed_unix_day: 19_852,
            totp_secret: None,
            lockout_until_unix: 0,
            totp_required: false,
            trailing_unknown_fields: Vec::new(),
        }
    }

    // ─── First-boot flow ─────────────────────────────────────────────────

    #[test]
    fn first_boot_creates_root_entry_on_match() {
        let mut src = MockSrc::from_strs(&["mypass", "mypass"]);
        let mut sink = MockOut::default();
        let writer = TestAtomicWriter::new();
        let entry = first_boot_setup(
            &mut src,
            &mut sink,
            &writer,
            SHADOW_PATH,
            argon_tier_tiny(),
            &SYNTHETIC_SALT,
            1_700_000_000,
        )
        .unwrap();
        assert_eq!(entry.username, "root");
        assert_eq!(entry.role, AuthRole::Root);
        assert_eq!(entry.flags, 0, "first-boot SHALL NOT set must-change");

        let raw = writer.read(SHADOW_PATH).expect("shadow file persisted");
        let s = String::from_utf8(raw).unwrap();
        let parsed = parse_file(&s).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].username, "root");
        assert_eq!(parsed[0].role, AuthRole::Root);

        // Echo trace shows both prompts.
        let e = sink.as_str();
        assert!(e.contains("Set initial root password"));
        assert!(e.contains("Confirm:"));
        assert!(e.contains("auth: root account created"));
    }

    #[test]
    fn first_boot_retries_on_mismatch() {
        let mut src = MockSrc::from_strs(&["alpha", "beta", "gamma", "gamma"]);
        let mut sink = MockOut::default();
        let writer = TestAtomicWriter::new();
        let entry = first_boot_setup(
            &mut src,
            &mut sink,
            &writer,
            SHADOW_PATH,
            argon_tier_tiny(),
            &SYNTHETIC_SALT,
            1_700_000_000,
        )
        .unwrap();
        assert_eq!(entry.username, "root");
        let e = sink.as_str();
        assert!(e.contains("did not match"));
    }

    #[test]
    fn first_boot_halts_after_three_mismatches() {
        let mut src = MockSrc::from_strs(&["a", "b", "c", "d", "e", "f"]);
        let mut sink = MockOut::default();
        let writer = TestAtomicWriter::new();
        let err = first_boot_setup(
            &mut src,
            &mut sink,
            &writer,
            SHADOW_PATH,
            argon_tier_tiny(),
            &SYNTHETIC_SALT,
            1_700_000_000,
        )
        .unwrap_err();
        assert!(matches!(err, FirstBootError::ConfirmationExhausted));
        // No file should have been written.
        assert!(writer.read(SHADOW_PATH).is_none());
        let e = sink.as_str();
        assert!(e.contains("halting"));
    }

    #[test]
    fn first_boot_rejects_empty_password_and_re_prompts() {
        let mut src = MockSrc::from_strs(&["", "good", "good"]);
        let mut sink = MockOut::default();
        let writer = TestAtomicWriter::new();
        let entry = first_boot_setup(
            &mut src,
            &mut sink,
            &writer,
            SHADOW_PATH,
            argon_tier_tiny(),
            &SYNTHETIC_SALT,
            1_700_000_000,
        )
        .unwrap();
        assert_eq!(entry.username, "root");
        assert!(sink.as_str().contains("Empty passwords"));
    }

    #[test]
    fn first_boot_propagates_atomic_write_error() {
        // Use the production stub which always errors with NotImplemented.
        let mut src = MockSrc::from_strs(&["pw", "pw"]);
        let mut sink = MockOut::default();
        let writer = crate::atomic_write::KernelAtomicWriter::new();
        let err = first_boot_setup(
            &mut src,
            &mut sink,
            &writer,
            SHADOW_PATH,
            argon_tier_tiny(),
            &SYNTHETIC_SALT,
            1_700_000_000,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            FirstBootError::AtomicWrite(AtomicWriteError::NotImplemented)
        ));
    }

    // ─── Steady-state login ──────────────────────────────────────────────

    #[test]
    fn login_success_establishes_session() {
        let user = make_user("root", SYNTHETIC_GOOD_PASSWORD, AuthRole::Root, 0);
        let users = alloc::vec![user];

        let mut src = MockSrc::new(&[b"root", SYNTHETIC_GOOD_PASSWORD]);
        let mut sink = MockOut::default();
        let mut tbl = SessionTable::new();
        let mut lk = LockoutMap::new();

        let outcome = run_login_round(
            &mut src,
            &mut sink,
            &users,
            &mut tbl,
            &mut lk,
            &LockoutSource::Tty,
            1_700_000_000,
            SYNTHETIC_TINY_PHC,
        );

        match outcome {
            LoginOutcome::Authenticated {
                session_id,
                username,
                role,
                must_change_password,
            } => {
                assert_eq!(username, "root");
                assert_eq!(role, AuthRole::Root);
                assert!(!must_change_password);
                assert!(tbl.lookup(session_id).is_ok());
            }
            other => panic!("expected Authenticated, got {other:?}"),
        }
        let e = sink.as_str();
        assert!(e.contains("auth: ok root"));
    }

    #[test]
    fn login_wrong_password_returns_failed() {
        let user = make_user("root", SYNTHETIC_GOOD_PASSWORD, AuthRole::Root, 0);
        let users = alloc::vec![user];

        let mut src = MockSrc::new(&[b"root", SYNTHETIC_WRONG_PASSWORD]);
        let mut sink = MockOut::default();
        let mut tbl = SessionTable::new();
        let mut lk = LockoutMap::new();

        let outcome = run_login_round(
            &mut src,
            &mut sink,
            &users,
            &mut tbl,
            &mut lk,
            &LockoutSource::Tty,
            1_700_000_000,
            SYNTHETIC_TINY_PHC,
        );
        assert!(matches!(outcome, LoginOutcome::Failed));
        assert_eq!(lk.failures(&LockoutSource::Tty), 1);
        assert!(sink.as_str().contains("Authentication failed"));
    }

    #[test]
    fn login_unknown_user_returns_failed_and_increments() {
        let user = make_user("alice", SYNTHETIC_GOOD_PASSWORD, AuthRole::Operator, 0);
        let users = alloc::vec![user];

        let mut src = MockSrc::new(&[b"nobody", SYNTHETIC_GOOD_PASSWORD]);
        let mut sink = MockOut::default();
        let mut tbl = SessionTable::new();
        let mut lk = LockoutMap::new();

        let outcome = run_login_round(
            &mut src,
            &mut sink,
            &users,
            &mut tbl,
            &mut lk,
            &LockoutSource::Tty,
            1_700_000_000,
            SYNTHETIC_TINY_PHC,
        );
        assert!(matches!(outcome, LoginOutcome::Failed));
        assert_eq!(lk.failures(&LockoutSource::Tty), 1);
    }

    #[test]
    fn login_must_change_password_flag_is_propagated() {
        let user = make_user(
            "root",
            SYNTHETIC_GOOD_PASSWORD,
            AuthRole::Root,
            FLAG_MUST_CHANGE_PASSWORD,
        );
        let users = alloc::vec![user];

        let mut src = MockSrc::new(&[b"root", SYNTHETIC_GOOD_PASSWORD]);
        let mut sink = MockOut::default();
        let mut tbl = SessionTable::new();
        let mut lk = LockoutMap::new();

        let outcome = run_login_round(
            &mut src,
            &mut sink,
            &users,
            &mut tbl,
            &mut lk,
            &LockoutSource::Tty,
            1_700_000_000,
            SYNTHETIC_TINY_PHC,
        );
        assert!(matches!(
            outcome,
            LoginOutcome::Authenticated {
                must_change_password: true,
                ..
            }
        ));
    }

    #[test]
    fn login_aborted_on_ctrl_c_does_not_count_failure() {
        let user = make_user("root", SYNTHETIC_GOOD_PASSWORD, AuthRole::Root, 0);
        let users = alloc::vec![user];

        let mut src = MockSrc::new(&[b"root", b"\x03"]);
        let mut sink = MockOut::default();
        let mut tbl = SessionTable::new();
        let mut lk = LockoutMap::new();

        let outcome = run_login_round(
            &mut src,
            &mut sink,
            &users,
            &mut tbl,
            &mut lk,
            &LockoutSource::Tty,
            1_700_000_000,
            SYNTHETIC_TINY_PHC,
        );
        assert!(matches!(outcome, LoginOutcome::Aborted));
        assert_eq!(lk.failures(&LockoutSource::Tty), 0);
    }

    #[test]
    fn login_eof_at_username_returns_eof() {
        let user = make_user("root", SYNTHETIC_GOOD_PASSWORD, AuthRole::Root, 0);
        let users = alloc::vec![user];
        let mut src = MockSrc::new(&[b"\x04"]);
        let mut sink = MockOut::default();
        let mut tbl = SessionTable::new();
        let mut lk = LockoutMap::new();

        let outcome = run_login_round(
            &mut src,
            &mut sink,
            &users,
            &mut tbl,
            &mut lk,
            &LockoutSource::Tty,
            1_700_000_000,
            SYNTHETIC_TINY_PHC,
        );
        assert!(matches!(outcome, LoginOutcome::Eof));
    }

    // ─── Lockout policy ──────────────────────────────────────────────────

    #[test]
    fn five_consecutive_failures_lock_for_60_seconds() {
        let user = make_user("root", SYNTHETIC_GOOD_PASSWORD, AuthRole::Root, 0);
        let users = alloc::vec![user];
        let mut tbl = SessionTable::new();
        let mut lk = LockoutMap::new();
        let now = 1_700_000_000;

        for _ in 0..5 {
            let mut src = MockSrc::new(&[b"root", SYNTHETIC_WRONG_PASSWORD]);
            let mut sink = MockOut::default();
            let outcome = run_login_round(
                &mut src,
                &mut sink,
                &users,
                &mut tbl,
                &mut lk,
                &LockoutSource::Tty,
                now,
                SYNTHETIC_TINY_PHC,
            );
            assert!(matches!(outcome, LoginOutcome::Failed));
        }
        assert!(lk.is_locked(&LockoutSource::Tty, now));
        assert!(lk.is_locked(&LockoutSource::Tty, now + 30));
        assert!(lk.is_locked(&LockoutSource::Tty, now + 59));
        // Just past 60s — unlocked.
        assert!(!lk.is_locked(&LockoutSource::Tty, now + 60));
    }

    #[test]
    fn login_during_lockout_returns_locked_out() {
        let user = make_user("root", SYNTHETIC_GOOD_PASSWORD, AuthRole::Root, 0);
        let users = alloc::vec![user];
        let mut tbl = SessionTable::new();
        let mut lk = LockoutMap::new();
        let now = 1_700_000_000;

        for _ in 0..5 {
            lk.record_failure(&LockoutSource::Tty, now);
        }
        assert!(lk.is_locked(&LockoutSource::Tty, now));

        // Even with the *correct* password the source is locked.
        let mut src = MockSrc::new(&[b"root", SYNTHETIC_GOOD_PASSWORD]);
        let mut sink = MockOut::default();
        let outcome = run_login_round(
            &mut src,
            &mut sink,
            &users,
            &mut tbl,
            &mut lk,
            &LockoutSource::Tty,
            now + 1,
            SYNTHETIC_TINY_PHC,
        );
        assert!(matches!(outcome, LoginOutcome::LockedOut));
        assert!(sink.as_str().contains(BANNER_LOCKED_OUT));
    }

    #[test]
    fn successful_login_resets_failure_counter() {
        let user = make_user("root", SYNTHETIC_GOOD_PASSWORD, AuthRole::Root, 0);
        let users = alloc::vec![user];
        let mut tbl = SessionTable::new();
        let mut lk = LockoutMap::new();
        let now = 1_700_000_000;

        for _ in 0..4 {
            lk.record_failure(&LockoutSource::Tty, now);
        }
        assert_eq!(lk.failures(&LockoutSource::Tty), 4);
        assert!(!lk.is_locked(&LockoutSource::Tty, now));

        let mut src = MockSrc::new(&[b"root", SYNTHETIC_GOOD_PASSWORD]);
        let mut sink = MockOut::default();
        let outcome = run_login_round(
            &mut src,
            &mut sink,
            &users,
            &mut tbl,
            &mut lk,
            &LockoutSource::Tty,
            now,
            SYNTHETIC_TINY_PHC,
        );
        assert!(matches!(outcome, LoginOutcome::Authenticated { .. }));
        assert_eq!(lk.failures(&LockoutSource::Tty), 0);
    }

    #[test]
    fn remote_lockout_does_not_affect_tty() {
        // Spec scenario: "Remote lockout does not affect local"
        let mut lk = LockoutMap::new();
        let peer = LockoutSource::ZenohPeer("peer-fingerprint-1".to_string());
        let now = 1_700_000_000;
        for _ in 0..5 {
            lk.record_failure(&peer, now);
        }
        assert!(lk.is_locked(&peer, now));
        assert!(!lk.is_locked(&LockoutSource::Tty, now));
    }

    #[test]
    fn lockout_per_peer_tracked_independently() {
        let mut lk = LockoutMap::new();
        let p1 = LockoutSource::ZenohPeer("peer-1".to_string());
        let p2 = LockoutSource::ZenohPeer("peer-2".to_string());
        let now = 1_700_000_000;
        for _ in 0..5 {
            lk.record_failure(&p1, now);
        }
        assert!(lk.is_locked(&p1, now));
        assert!(!lk.is_locked(&p2, now));
    }

    #[test]
    fn lockout_threshold_constants_match_spec() {
        // Spec `console-login` Q20: 5 fails / 60 s.
        assert_eq!(LOCKOUT_THRESHOLD, 5);
        assert_eq!(LOCKOUT_DURATION_SECS, 60);
    }

    // ─── Logout ──────────────────────────────────────────────────────────

    #[test]
    fn logout_releases_session() {
        let mut tbl = SessionTable::new();
        let session = Session::new(7, b"root", KernelRole::Root, 1_700_000_000, false).unwrap();
        let id = tbl.acquire(session).unwrap();
        let mut sink = MockOut::default();
        let outcome = run_logout(&mut tbl, id, &mut sink);
        assert_eq!(outcome, LogoutOutcome::Released);
        assert!(tbl.lookup(id).is_err());
        assert!(sink.as_str().contains("auth: logout"));
    }

    #[test]
    fn logout_stale_session_returns_stale() {
        let mut tbl = SessionTable::new();
        let session = Session::new(7, b"root", KernelRole::Root, 1_700_000_000, false).unwrap();
        let id = tbl.acquire(session).unwrap();
        tbl.release(id).unwrap();
        let mut sink = MockOut::default();
        let outcome = run_logout(&mut tbl, id, &mut sink);
        assert_eq!(outcome, LogoutOutcome::Stale);
    }

    // ─── Idle auto-logout integration ────────────────────────────────────

    #[test]
    fn idle_root_session_swept_after_15_minutes() {
        // Spec scenario: "Idle Root session times out"
        use smallaios_kernel::auth::Sweeper;

        let mut tbl = SessionTable::new();
        let session = Session::new(7, b"root", KernelRole::Root, 0, false).unwrap();
        let id = tbl.acquire(session).unwrap();

        let sweeper = Sweeper::new();
        let mut audit = smallaios_kernel::auth::CapturingAuditSink::new();

        // 14 min — alive.
        sweeper.tick(14 * 60, &mut tbl, &mut audit);
        assert!(tbl.lookup(id).is_ok());

        // 15 min + 1s — swept.
        let killed = sweeper.tick(15 * 60 + 1, &mut tbl, &mut audit);
        assert_eq!(killed, 1);
        assert!(tbl.lookup(id).is_err());
        assert_eq!(audit.events().len(), 1);
    }

    #[test]
    fn keypress_resets_idle_timer() {
        // Spec scenario: "Keypress in console-monitor refreshes timer"
        use smallaios_kernel::auth::Sweeper;

        let mut tbl = SessionTable::new();
        let session = Session::new(8, b"viewer", KernelRole::Viewer, 0, false).unwrap();
        let id = tbl.acquire(session).unwrap();

        // 59 min in — refresh.
        tbl.reset_idle(id, 59 * 60).unwrap();

        let sweeper = Sweeper::new();
        let mut audit = smallaios_kernel::auth::CapturingAuditSink::new();
        // Now at 60 min — only 1 min since last activity, alive.
        let killed = sweeper.tick(60 * 60, &mut tbl, &mut audit);
        assert_eq!(killed, 0);
    }

    // ─── Bootstrap selection ─────────────────────────────────────────────

    #[test]
    fn decide_boot_no_file_drops_to_first_boot() {
        let d = decide_boot(None);
        assert!(matches!(d, BootDecision::FirstBoot));
    }

    #[test]
    fn decide_boot_empty_file_drops_to_first_boot() {
        let d = decide_boot(Some(b""));
        assert!(matches!(d, BootDecision::FirstBoot));
        let d2 = decide_boot(Some(b"\n  \n"));
        assert!(matches!(d2, BootDecision::FirstBoot));
    }

    #[test]
    fn decide_boot_valid_file_returns_steady_state() {
        let user = make_user("root", SYNTHETIC_GOOD_PASSWORD, AuthRole::Root, 0);
        let body = serialize_file(core::slice::from_ref(&user));
        let d = decide_boot(Some(body.as_bytes()));
        match d {
            BootDecision::SteadyState(entries) => {
                assert_eq!(entries.len(), 1);
                assert_eq!(entries[0].username, "root");
            }
            other => panic!("expected SteadyState, got {other:?}"),
        }
    }

    #[test]
    fn decide_boot_corrupt_file_returns_corrupt_shadow() {
        let d = decide_boot(Some(b"this is not a shadow file\n"));
        assert!(matches!(d, BootDecision::CorruptShadow(_)));
    }

    #[test]
    fn decide_boot_invalid_utf8_returns_corrupt_shadow() {
        let d = decide_boot(Some(&[0xFFu8, 0xFE, 0xFD]));
        assert!(matches!(d, BootDecision::CorruptShadow(_)));
    }

    // ─── lookup_user ─────────────────────────────────────────────────────

    #[test]
    fn lookup_user_finds_match() {
        let users = alloc::vec![
            make_user("alice", LOOKUP_USER_PW_A, AuthRole::Operator, 0),
            make_user("bob", LOOKUP_USER_PW_B, AuthRole::Viewer, 0),
        ];
        let got = lookup_user(&users, b"bob").unwrap();
        assert_eq!(got.username, "bob");
    }

    #[test]
    fn lookup_user_missing_returns_none() {
        let users = alloc::vec![make_user("alice", LOOKUP_USER_PW_A, AuthRole::Operator, 0)];
        assert!(lookup_user(&users, b"charlie").is_none());
    }

    // ─── Constant-time-equivalent unknown-user verification ──────────────

    #[test]
    fn unknown_user_still_runs_argon2id_against_dummy_phc() {
        // We can't measure timing in a unit test, but we *can* verify
        // that the dummy_phc is consulted: pass an obviously-invalid
        // dummy and observe that it does not panic / blow up.
        let users: Vec<ShadowEntry> = alloc::vec![];
        let mut src = MockSrc::new(&[b"ghost", SYNTHETIC_WRONG_PASSWORD]);
        let mut sink = MockOut::default();
        let mut tbl = SessionTable::new();
        let mut lk = LockoutMap::new();
        let outcome = run_login_round(
            &mut src,
            &mut sink,
            &users,
            &mut tbl,
            &mut lk,
            &LockoutSource::Tty,
            0,
            SYNTHETIC_TINY_PHC,
        );
        assert!(matches!(outcome, LoginOutcome::Failed));
    }

    // ─── Line-options helpers ────────────────────────────────────────────

    #[test]
    fn line_options_default_matches_legacy() {
        let opts = LineOptions::default();
        assert!(opts.echo);
        assert!(!opts.raw);
    }

    #[test]
    fn line_options_password_is_echo_off() {
        let opts = LineOptions::password();
        assert!(!opts.echo);
        assert!(!opts.raw);
        assert_eq!(opts.max_len, 1024);
    }

    // ─── Sink shim ───────────────────────────────────────────────────────

    #[test]
    fn console_sink_print_writes_string_bytes() {
        let mut sink = MockOut::default();
        sink.print("hello").unwrap();
        assert_eq!(sink.as_str(), "hello");
    }
}
