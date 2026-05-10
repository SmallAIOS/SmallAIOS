// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Recovery boot-arg path — `auth.skip-firstboot`.
//!
//! Spec: `openspec/changes/management-login-v1/specs/console-login/spec.md`
//! Requirement "Recovery skip-firstboot boot argument".
//!
//! The kernel SHALL parse `auth.skip-firstboot` from the boot arg list
//! and SHALL only honor it when at least one
//! [`crate::presence::PhysicalPresenceProvider`] asserts. When honored,
//! the recovery path:
//!
//! 1. Generates a cryptographically random 16-character ASCII password
//!    (or accepts an inline pre-baked PHC via
//!    `auth.skip-firstboot=<phc>`).
//! 2. Hashes with the runtime-tier Argon2id parameters (or uses the
//!    inline PHC verbatim).
//! 3. Writes a fresh shadow record with `must_change_password_on_login`
//!    set.
//! 4. Prints the cleartext password ONCE on the serial console.
//! 5. Appends an audit record naming the recovery event.
//!
//! Without presence, the boot arg SHALL be silently ignored.
//!
//! ## RNG abstraction
//!
//! Auth (Layer 1) does not own a kernel CSPRNG; the boot path supplies
//! one through the [`Rng`] trait. Tests use [`DeterministicRng`] for
//! reproducibility.

use crate::atomic_write::{AtomicWriteError, AtomicWriter};
use crate::console_login::{ConsoleSink, SHADOW_PATH};
use crate::role::Role as AuthRole;
use crate::shadow::{serialize_file, ShadowEntry, FLAG_MUST_CHANGE_PASSWORD};
use crate::tier::select_tier;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;
use smallaios_security::argon2id::{argon2id_format_phc, argon2id_hash, Argon2idParams};

/// Length of the auto-generated recovery password — 16 ASCII characters
/// per `console-login` spec.
pub const RECOVERY_PASSWORD_LEN: usize = 16;

/// Recovery password alphabet — 64 printable ASCII characters chosen so
/// that 6 bits of entropy fit one character. Excludes `:`, whitespace,
/// and shell metacharacters (`'$&|<>`'`) that would complicate display
/// or copy-paste off the serial console.
///
/// lgtm[rust/hard-coded-cryptographic-value] — public alphabet, not a credential
const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

// ─── Boot arg ────────────────────────────────────────────────────────────────

/// Parsed shape of the `auth.skip-firstboot[=<value>]` boot arg.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryBootArg {
    /// Bare `auth.skip-firstboot` — generate a random password.
    Generate,
    /// `auth.skip-firstboot=<phc>` — install the given PHC verbatim.
    PreBakedPhc(String),
}

/// Parse the boot-arg list (a sequence of `key[=value]` whitespace-
/// separated tokens) and return the `RecoveryBootArg` if present.
///
/// The parser is permissive about leading/trailing whitespace and
/// double-quoted values:
/// `auth.skip-firstboot="$argon2id$..."` is accepted.
pub fn parse_recovery_boot_arg(cmdline: &str) -> Option<RecoveryBootArg> {
    for token in cmdline.split_whitespace() {
        if let Some(rest) = token.strip_prefix("auth.skip-firstboot") {
            if rest.is_empty() {
                return Some(RecoveryBootArg::Generate);
            }
            if let Some(value) = rest.strip_prefix('=') {
                let trimmed = value
                    .trim_start_matches('"')
                    .trim_end_matches('"')
                    .trim_start_matches('\'')
                    .trim_end_matches('\'');
                if trimmed.is_empty() {
                    return Some(RecoveryBootArg::Generate);
                }
                return Some(RecoveryBootArg::PreBakedPhc(trimmed.to_string()));
            }
        }
    }
    None
}

// ─── RNG ─────────────────────────────────────────────────────────────────────

/// Minimal CSPRNG abstraction. The kernel boot path supplies a
/// platform-specific implementation; tests use [`DeterministicRng`].
pub trait Rng {
    /// Fill `dst` with random bytes.
    fn fill(&mut self, dst: &mut [u8]);
}

/// Deterministic RNG used by recovery tests. Drives an xorshift64*
/// stream — not cryptographically secure, only suitable for fixtures.
#[derive(Debug, Clone)]
pub struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    /// Construct with a seed.
    pub const fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_BABE
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

impl Rng for DeterministicRng {
    fn fill(&mut self, dst: &mut [u8]) {
        let mut i = 0;
        while i < dst.len() {
            let v = self.next_u64();
            for b in v.to_le_bytes() {
                if i >= dst.len() {
                    return;
                }
                dst[i] = b;
                i += 1;
            }
        }
    }
}

/// Generate a [`RECOVERY_PASSWORD_LEN`]-character ASCII password by
/// drawing 6-bit indices from `rng` and mapping through [`ALPHABET`].
pub fn generate_recovery_password<R: Rng>(rng: &mut R) -> String {
    let mut raw = [0u8; RECOVERY_PASSWORD_LEN];
    rng.fill(&mut raw);
    let mut out = String::with_capacity(RECOVERY_PASSWORD_LEN);
    for &b in &raw {
        let idx = (b & 0b0011_1111) as usize;
        out.push(ALPHABET[idx] as char);
    }
    out
}

// ─── Errors / outcomes ───────────────────────────────────────────────────────

/// Errors raised by [`run_recovery`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecoveryError {
    /// Atomic shadow write failed.
    AtomicWrite(AtomicWriteError),
    /// Argon2id hashing failed (invalid parameters / salt).
    Argon2id(&'static str),
    /// Pre-baked PHC string was malformed (missing `$argon2id$` prefix
    /// or otherwise unparseable).
    MalformedPhc,
    /// Console I/O error while emitting the cleartext password.
    Io(&'static str),
}

impl fmt::Display for RecoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AtomicWrite(e) => write!(f, "recovery: atomic-write error: {e}"),
            Self::Argon2id(s) => write!(f, "recovery: argon2id: {s}"),
            Self::MalformedPhc => write!(f, "recovery: pre-baked PHC malformed"),
            Self::Io(s) => write!(f, "recovery: I/O: {s}"),
        }
    }
}

/// What [`run_recovery`] decided to do.
#[derive(Debug, PartialEq, Eq)]
pub enum RecoveryOutcome {
    /// Boot arg ignored — presence not asserted. The system SHALL
    /// proceed with the normal first-boot prompt.
    PresenceMissing,
    /// Recovery succeeded. `cleartext_password` is `None` for the
    /// pre-baked-hash form (the operator already knows the password)
    /// and `Some(...)` for the random-generation form.
    Recovered {
        /// Cleartext password to print to the console (random form
        /// only). `None` for the pre-baked-hash form.
        cleartext_password: Option<String>,
        /// PHC string installed in the shadow record.
        installed_phc: String,
    },
}

// ─── Main entry point ────────────────────────────────────────────────────────

/// Execute the recovery path against `arg`.
///
/// Returns [`RecoveryOutcome::PresenceMissing`] if `presence_asserted`
/// is `false`; otherwise creates a fresh shadow record with
/// `must_change_password_on_login` set and persists it via `writer`.
#[allow(clippy::too_many_arguments)]
pub fn run_recovery<R: Rng, W: AtomicWriter, K: ConsoleSink>(
    arg: &RecoveryBootArg,
    presence_asserted: bool,
    presence_provider_name: Option<&'static str>,
    rng: &mut R,
    writer: &W,
    sink: &mut K,
    available_ram_bytes: u64,
    salt: &[u8],
    now_unix: u64,
) -> Result<RecoveryOutcome, RecoveryError> {
    if !presence_asserted {
        // Spec scenario: "Skip-firstboot ignored without presence"
        return Ok(RecoveryOutcome::PresenceMissing);
    }

    let (phc, cleartext) = match arg {
        RecoveryBootArg::Generate => {
            let password = generate_recovery_password(rng);
            let params = select_tier(available_ram_bytes);
            let phc = hash_password(password.as_bytes(), salt, params)
                .map_err(RecoveryError::Argon2id)?;
            (phc, Some(password))
        }
        RecoveryBootArg::PreBakedPhc(phc) => {
            if !phc.starts_with("$argon2id$") || phc.matches('$').count() != 5 {
                return Err(RecoveryError::MalformedPhc);
            }
            (phc.clone(), None)
        }
    };

    let entry = ShadowEntry {
        username: "root".to_string(),
        phc: phc.clone(),
        role: AuthRole::Root,
        flags: FLAG_MUST_CHANGE_PASSWORD,
        last_changed_unix_day: (now_unix / 86_400) as u32,
        totp_secret: None,
        lockout_until_unix: 0,
        // Recovery accounts opt out of TOTP — the operator MUST be
        // able to re-enrol their authenticator after recovery, so we
        // don't gate the immediate post-recovery login.
        totp_required: false,
        trailing_unknown_fields: Vec::new(),
    };
    let body = serialize_file(core::slice::from_ref(&entry));
    writer
        .write_atomic(SHADOW_PATH, body.as_bytes())
        .map_err(RecoveryError::AtomicWrite)?;

    // Audit / banner — the kernel boot path also appends an `auth_recovery`
    // audit record naming `presence_provider_name`.
    sink.print("auth: recovery — temporary root password (rotate immediately):\n")
        .map_err(RecoveryError::Io)?;
    if let Some(pw) = cleartext.as_deref() {
        sink.print("    password: ").map_err(RecoveryError::Io)?;
        sink.print(pw).map_err(RecoveryError::Io)?;
        sink.print("\n").map_err(RecoveryError::Io)?;
    } else {
        sink.print("    (pre-baked PHC accepted; use the password you already have)\n")
            .map_err(RecoveryError::Io)?;
    }
    if let Some(name) = presence_provider_name {
        sink.print("    presence: ").map_err(RecoveryError::Io)?;
        sink.print(name).map_err(RecoveryError::Io)?;
        sink.print("\n").map_err(RecoveryError::Io)?;
    }

    Ok(RecoveryOutcome::Recovered {
        cleartext_password: cleartext,
        installed_phc: phc,
    })
}

fn hash_password(pw: &[u8], salt: &[u8], params: Argon2idParams) -> Result<String, &'static str> {
    if salt.len() < 8 {
        return Err("salt < 8 bytes");
    }
    let tag = argon2id_hash(pw, salt, params);
    Ok(argon2id_format_phc(salt, &tag, params))
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atomic_write::TestAtomicWriter;
    use crate::console_login::ConsoleSink;
    use crate::console_login_test_vectors::*;
    use crate::shadow::parse_file;

    #[derive(Default)]
    struct CapSink {
        bytes: Vec<u8>,
    }

    impl ConsoleSink for CapSink {
        fn write_bytes(&mut self, b: &[u8]) -> Result<(), &'static str> {
            self.bytes.extend_from_slice(b);
            Ok(())
        }
    }

    impl CapSink {
        fn as_str(&self) -> String {
            String::from_utf8_lossy(&self.bytes).to_string()
        }
    }

    // ─── Boot arg parser ─────────────────────────────────────────────────

    #[test]
    fn parse_bare_recovery_arg() {
        assert_eq!(
            parse_recovery_boot_arg("console=ttyS0 auth.skip-firstboot quiet"),
            Some(RecoveryBootArg::Generate)
        );
    }

    #[test]
    fn parse_pre_baked_phc_arg() {
        let cmd = alloc::format!(
            "console=ttyS0 auth.skip-firstboot={} quiet",
            SYNTHETIC_RECOVERY_PHC
        );
        assert_eq!(
            parse_recovery_boot_arg(&cmd),
            Some(RecoveryBootArg::PreBakedPhc(
                SYNTHETIC_RECOVERY_PHC.to_string()
            ))
        );
    }

    #[test]
    fn parse_pre_baked_phc_with_quotes() {
        let cmd = alloc::format!("auth.skip-firstboot=\"{}\"", SYNTHETIC_RECOVERY_PHC);
        assert_eq!(
            parse_recovery_boot_arg(&cmd),
            Some(RecoveryBootArg::PreBakedPhc(
                SYNTHETIC_RECOVERY_PHC.to_string()
            ))
        );
    }

    #[test]
    fn parse_returns_none_when_absent() {
        assert!(parse_recovery_boot_arg("console=ttyS0 quiet").is_none());
    }

    #[test]
    fn parse_empty_value_treated_as_generate() {
        assert_eq!(
            parse_recovery_boot_arg("auth.skip-firstboot="),
            Some(RecoveryBootArg::Generate)
        );
    }

    #[test]
    fn parse_does_not_match_partial_token() {
        // `auth.skip-firstboot-extra=1` is a different token; the
        // parser SHALL NOT honor it. The boundary is enforced because
        // the suffix after `auth.skip-firstboot` must be `=` or empty.
        assert!(parse_recovery_boot_arg("auth.skip-firstboot-extra=1").is_none());
        // Same for `auth.skip-firstbooted`.
        assert!(parse_recovery_boot_arg("auth.skip-firstbooted").is_none());
    }

    // ─── Presence gate ───────────────────────────────────────────────────

    #[test]
    fn recovery_without_presence_returns_presence_missing() {
        // Spec scenario: "Skip-firstboot ignored without presence"
        let mut rng = DeterministicRng::new(1);
        let writer = TestAtomicWriter::new();
        let mut sink = CapSink::default();
        let outcome = run_recovery(
            &RecoveryBootArg::Generate,
            false, // no presence
            None,
            &mut rng,
            &writer,
            &mut sink,
            64 * 1024 * 1024,
            &SYNTHETIC_SALT,
            1_700_000_000,
        )
        .unwrap();
        assert_eq!(outcome, RecoveryOutcome::PresenceMissing);
        assert!(writer.read(SHADOW_PATH).is_none());
    }

    #[test]
    fn recovery_with_presence_creates_must_change_record() {
        // Spec scenario: "Skip-firstboot honored with presence asserted"
        let mut rng = DeterministicRng::new(0xCAFEBABE);
        let writer = TestAtomicWriter::new();
        let mut sink = CapSink::default();
        let outcome = run_recovery(
            &RecoveryBootArg::Generate,
            true,
            Some("x86-gpio"),
            &mut rng,
            &writer,
            &mut sink,
            64 * 1024 * 1024,
            &SYNTHETIC_SALT,
            1_700_000_000,
        )
        .unwrap();
        let cleartext = match outcome {
            RecoveryOutcome::Recovered {
                cleartext_password,
                installed_phc,
            } => {
                assert!(installed_phc.starts_with("$argon2id$"));
                cleartext_password.expect("random form yields cleartext")
            }
            other => panic!("expected Recovered, got {other:?}"),
        };
        assert_eq!(cleartext.len(), RECOVERY_PASSWORD_LEN);

        // Persisted record has must-change set.
        let raw = writer.read(SHADOW_PATH).unwrap();
        let entries = parse_file(core::str::from_utf8(&raw).unwrap()).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].username, "root");
        assert_eq!(entries[0].role, AuthRole::Root);
        assert_eq!(
            entries[0].flags & FLAG_MUST_CHANGE_PASSWORD,
            FLAG_MUST_CHANGE_PASSWORD
        );

        // The cleartext is printed exactly once.
        let banner = sink.as_str();
        assert!(banner.contains("recovery"));
        assert_eq!(banner.matches(&cleartext).count(), 1);
        assert!(banner.contains("x86-gpio"));
    }

    #[test]
    fn recovery_pre_baked_phc_form() {
        // Spec scenario: "Pre-baked hash form"
        let mut rng = DeterministicRng::new(7);
        let writer = TestAtomicWriter::new();
        let mut sink = CapSink::default();
        let outcome = run_recovery(
            &RecoveryBootArg::PreBakedPhc(SYNTHETIC_RECOVERY_PHC.to_string()),
            true,
            Some("verified-boot-arg"),
            &mut rng,
            &writer,
            &mut sink,
            64 * 1024 * 1024,
            &SYNTHETIC_SALT,
            1_700_000_000,
        )
        .unwrap();
        match outcome {
            RecoveryOutcome::Recovered {
                cleartext_password,
                installed_phc,
            } => {
                assert!(cleartext_password.is_none());
                assert_eq!(installed_phc, SYNTHETIC_RECOVERY_PHC);
            }
            other => panic!("expected Recovered, got {other:?}"),
        }

        // Shadow record uses the verbatim PHC + must-change.
        let raw = writer.read(SHADOW_PATH).unwrap();
        let entries = parse_file(core::str::from_utf8(&raw).unwrap()).unwrap();
        assert_eq!(entries[0].phc, SYNTHETIC_RECOVERY_PHC);
        assert_eq!(
            entries[0].flags & FLAG_MUST_CHANGE_PASSWORD,
            FLAG_MUST_CHANGE_PASSWORD
        );
    }

    #[test]
    fn recovery_malformed_pre_baked_phc_rejected() {
        let mut rng = DeterministicRng::new(7);
        let writer = TestAtomicWriter::new();
        let mut sink = CapSink::default();
        let err = run_recovery(
            &RecoveryBootArg::PreBakedPhc("not-a-phc".to_string()),
            true,
            None,
            &mut rng,
            &writer,
            &mut sink,
            64 * 1024 * 1024,
            &SYNTHETIC_SALT,
            1_700_000_000,
        )
        .unwrap_err();
        assert_eq!(err, RecoveryError::MalformedPhc);
        // Nothing persisted on error.
        assert!(writer.read(SHADOW_PATH).is_none());
    }

    #[test]
    fn recovery_atomic_write_error_propagates() {
        // Use the production stub which always returns NotImplemented.
        let mut rng = DeterministicRng::new(7);
        let writer = crate::atomic_write::KernelAtomicWriter::new();
        let mut sink = CapSink::default();
        let err = run_recovery(
            &RecoveryBootArg::Generate,
            true,
            Some("x86-gpio"),
            &mut rng,
            &writer,
            &mut sink,
            64 * 1024 * 1024,
            &SYNTHETIC_SALT,
            1_700_000_000,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            RecoveryError::AtomicWrite(AtomicWriteError::NotImplemented)
        ));
    }

    // ─── Password generation ─────────────────────────────────────────────

    #[test]
    fn generate_password_is_16_chars() {
        let mut rng = DeterministicRng::new(42);
        let pw = generate_recovery_password(&mut rng);
        assert_eq!(pw.len(), RECOVERY_PASSWORD_LEN);
    }

    #[test]
    fn generate_password_uses_alphabet() {
        let mut rng = DeterministicRng::new(43);
        let pw = generate_recovery_password(&mut rng);
        for ch in pw.bytes() {
            assert!(ALPHABET.contains(&ch), "char 0x{ch:02x} not in alphabet");
        }
    }

    #[test]
    fn generate_password_changes_with_seed() {
        let pw1 = generate_recovery_password(&mut DeterministicRng::new(1));
        let pw2 = generate_recovery_password(&mut DeterministicRng::new(2));
        assert_ne!(pw1, pw2);
    }

    // ─── RNG helper ──────────────────────────────────────────────────────

    #[test]
    fn deterministic_rng_zero_seed_is_remapped() {
        let mut rng = DeterministicRng::new(0);
        let mut buf = [0u8; 8];
        rng.fill(&mut buf);
        assert!(buf.iter().any(|&b| b != 0), "all-zero RNG output");
    }

    #[test]
    fn deterministic_rng_fill_partial_buffer() {
        let mut rng = DeterministicRng::new(99);
        let mut buf = [0u8; 3];
        rng.fill(&mut buf);
        // No assertion on values — just confirm we don't OOB.
        assert_eq!(buf.len(), 3);
    }

    // ─── Banner contents ─────────────────────────────────────────────────

    #[test]
    fn banner_includes_provider_name() {
        let mut rng = DeterministicRng::new(101);
        let writer = TestAtomicWriter::new();
        let mut sink = CapSink::default();
        run_recovery(
            &RecoveryBootArg::Generate,
            true,
            Some("qemu-fw-cfg"),
            &mut rng,
            &writer,
            &mut sink,
            64 * 1024 * 1024,
            &SYNTHETIC_SALT,
            1_700_000_000,
        )
        .unwrap();
        assert!(sink.as_str().contains("qemu-fw-cfg"));
    }

    #[test]
    fn banner_for_pre_baked_form_does_not_print_cleartext() {
        let mut rng = DeterministicRng::new(101);
        let writer = TestAtomicWriter::new();
        let mut sink = CapSink::default();
        run_recovery(
            &RecoveryBootArg::PreBakedPhc(SYNTHETIC_RECOVERY_PHC.to_string()),
            true,
            None,
            &mut rng,
            &writer,
            &mut sink,
            64 * 1024 * 1024,
            &SYNTHETIC_SALT,
            1_700_000_000,
        )
        .unwrap();
        assert!(sink.as_str().contains("pre-baked"));
        assert!(
            !sink.as_str().contains("password: "),
            "must NOT print a cleartext password line"
        );
    }
}
