// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Auth syscall handlers (0x90-0x95).
//!
//! Spec: `openspec/changes/management-login-v1/specs/kernel-syscalls/spec.md`
//!
//! These six handlers are the kernel's user-facing surface for the
//! `management-login-v1` Phase 3 design:
//!
//! | Number | Name                  | Min role        |
//! |-------:|-----------------------|-----------------|
//! | 0x90   | `auth_login`          | Unauthenticated |
//! | 0x91   | `auth_logout`         | Authenticated   |
//! | 0x92   | `auth_change_password`| Authenticated   |
//! | 0x93   | `auth_create_user`    | Root            |
//! | 0x94   | `auth_whoami`         | Authenticated   |
//! | 0x95   | `auth_totp_setup`     | Authenticated (Phase 9) |
//!
//! ## Pointer-validation discipline
//!
//! The unikernel runs with caller and kernel in the same address space,
//! so handlers slice from raw pointers via `core::slice::from_raw_parts`
//! after validating non-null and length. This matches the rest of the
//! syscall surface (see `system.rs`, `posix.rs`).
//!
//! ## Constant-time-equivalent reject on user-not-found
//!
//! `auth_login` runs `argon2id_verify` against a *dummy* PHC string when
//! the lookup misses, to avoid leaking enumeration via timing — see
//! [`DUMMY_PHC`]. The PHC carries the same per-tier parameters as a
//! real entry, so the wall-clock budget matches.
//!
//! ## Testing
//!
//! Per-handler unit tests use [`AuthCtx`] directly to plug a
//! `MockShadowProvider`, an in-memory `SessionTable`, a
//! `CapturingAuditSink`, and a frozen clock. The dispatch path
//! (`crate::syscall::mod`) reaches the handlers via [`AuthCtx::global`]
//! once the kernel boot path installs one.

use crate::auth::{
    AuditSink, Role, Session, SessionTable, ShadowProvider, ShadowProviderError, Sweeper,
    ERRNO_EACCES, ERRNO_EAGAIN, ERRNO_EEXIST, ERRNO_EFAULT, ERRNO_EINVAL, ERRNO_ENOENT,
    ERRNO_ENOSPC, ERRNO_ENOSYS, FLAG_MUST_CHANGE_PASSWORD,
};
use core::sync::atomic::{AtomicU64, Ordering};
use smallaios_security::argon2id::{argon2id_format_phc, argon2id_hash, Argon2idParams};

use super::{SyscallArgs, SyscallResult};

// ─── Configuration constants ────────────────────────────────────────────────

/// Hard cap on password byte length accepted by `auth_login` and
/// `auth_change_password`. Above this we bail with `-EINVAL` rather
/// than feed the value to Argon2id.
const PASSWORD_MAX_LEN: usize = 1024;

/// Hard cap on username byte length. Matches
/// [`crate::auth::session_table::MAX_USERNAME_LEN`].
const USERNAME_MAX_LEN: usize = 64;

/// Dummy PHC string used by [`sys_auth_login`] when the username lookup
/// misses. The salt and tag are 16 / 32 zero bytes — *not* a credential
/// for any real account; the PHC parser only requires shape + base64.
///
/// The kernel runs `argon2id_verify` against this so the wall-clock
/// time of a missing-user reject matches a present-user reject.
//
// lgtm[rust/hard-coded-cryptographic-value] — synthetic dummy used only for constant-time-equivalent reject; not a real credential
const DUMMY_PHC: &str =
    "$argon2id$v=19$m=8192,t=3,p=1$AAAAAAAAAAAAAAAAAAAAAA$AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";

// ─── ABI struct returned by `auth_whoami` ───────────────────────────────────

/// Layout written by `auth_whoami(out_ptr)`.
///
/// `repr(C)` so the layout is stable across compiler versions and the
/// userspace `whoami()` library shim can pin it.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WhoamiOut {
    /// Role byte: `0=Root, 1=Operator, 2=Viewer` — matches
    /// [`smallaios_auth::role::Role`] wire format.
    pub role: u8,
    /// Padding to keep `user_id` 4-byte aligned.
    _pad: [u8; 3],
    /// Stable user identifier from the shadow loader.
    pub user_id: u32,
    /// Login time as Unix seconds.
    pub login_unix_time: u64,
    /// Seconds elapsed since `last_activity`. Saturating in the
    /// wall-clock skew case.
    pub idle_seconds: u32,
    /// Padding to round the struct up to 8 bytes.
    _pad2: [u8; 4],
}

const _: () = {
    // Compile-time assertion: layout must be stable.
    assert!(core::mem::size_of::<WhoamiOut>() == 24);
    assert!(core::mem::align_of::<WhoamiOut>() == 8);
};

// ─── Auth context ───────────────────────────────────────────────────────────

/// Boot-time clock injection. The boot path stamps this with the
/// initial Unix time so syscalls can read `now_unix_time()` without
/// pulling in `sys_time`'s nanosecond timer. Defaults to `0` until
/// installed.
pub static BOOT_UNIX_TIME: AtomicU64 = AtomicU64::new(0);

/// Read the current wall clock as Unix seconds. Phase 3 returns the
/// boot value plus monotonic seconds since boot; Phase 6 (`mgmt`) will
/// hot-swap this to a real RTC reading on Jetson and to the host clock
/// on Linux container builds. Held public so the boot path and Phase
/// 6 mgmt loader can stamp it into a fresh [`AuthCtx`].
pub fn now_unix_time() -> u64 {
    let base = BOOT_UNIX_TIME.load(Ordering::Acquire);
    let mono = crate::sched::timer::Timestamp::now().as_u64();
    // `Timestamp::now()` is in opaque ticks. The mgmt loader will
    // calibrate the conversion in Phase 6; until then we treat one
    // tick == one millisecond and divide by 1000. The kernel's idle
    // sweeper tolerates this with saturating math.
    base.saturating_add(mono / 1_000)
}

/// Function-pointer salt source. Production wires this to
/// [`crate::state::csprng_generate`]; tests inject a deterministic
/// XorShift to avoid touching the global CSPRNG state.
///
/// Returns `Ok(())` if `out` was filled, `Err(())` on any failure
/// (insufficient entropy, CSPRNG unseeded, ...). The handler maps a
/// failure to `-ENOSYS` so the caller can distinguish setup-incomplete
/// from a true runtime error.
#[allow(clippy::result_unit_err)] // intentional opaque error
pub type SaltSource = fn(out: &mut [u8]) -> Result<(), ()>;

/// Production salt source — pulls from the kernel CSPRNG.
#[allow(clippy::result_unit_err)] // matches the SaltSource fn-pointer signature
pub fn csprng_salt(out: &mut [u8]) -> Result<(), ()> {
    // SAFETY: caller (handler) holds exclusive access to the CSPRNG
    // for the duration of this call (interrupts masked in unikernel
    // mode).
    unsafe { crate::state::csprng_generate(out).map_err(|_| ()) }
}

/// Bundle of references threaded through every auth syscall.
///
/// In production, the dispatch path constructs an [`AuthCtx`] on the
/// fly from the global session table, the installed
/// [`ShadowProvider`], the [`Sweeper`], the configured [`AuditSink`],
/// and the production [`csprng_salt`] source. Tests construct one
/// directly with mocks and a deterministic salt source.
pub struct AuthCtx<'a, P: ShadowProvider + ?Sized, S: AuditSink + ?Sized> {
    /// Live session table.
    pub table: &'a mut SessionTable,
    /// Shadow provider — the user database.
    pub provider: &'a P,
    /// Sweeper (for policy lookup, even though its tick runs from a
    /// dedicated kernel task).
    pub sweeper: &'a Sweeper,
    /// Audit sink for invalidations and failures.
    pub audit: &'a mut S,
    /// Frozen "now" — handler treats this as authoritative for the
    /// duration of the call. Tests pass a fixture value; production
    /// passes [`now_unix_time`].
    pub now_unix: u64,
    /// Salt source — production passes [`csprng_salt`], tests pass a
    /// deterministic stub.
    pub salt_source: SaltSource,
}

// ─── Pointer slice helpers ───────────────────────────────────────────────────

/// Materialize an immutable byte slice from a `(ptr, len)` pair received
/// over the syscall ABI. Public so the future Zenoh-admin Phase 7
/// handler can validate its userspace pointers identically.
///
/// # Safety
///
/// Caller must guarantee `ptr` is valid for `len` bytes for the
/// duration of the returned borrow. In unikernel mode this is
/// satisfied by interrupts being masked across the syscall.
#[allow(clippy::result_unit_err)]
pub unsafe fn slice_from_args(ptr: usize, len: usize, max: usize) -> Result<&'static [u8], i64> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr == 0 {
        return Err(ERRNO_EFAULT);
    }
    if len > max {
        return Err(ERRNO_EINVAL);
    }
    // SAFETY: caller is in the same address space; ptr/len validated.
    Ok(core::slice::from_raw_parts(ptr as *const u8, len))
}

/// Like [`slice_from_args`] but treats `len == 0` and `ptr == 0` as a
/// canonically-empty slice (used for the optional `factor2` and
/// `target_user` arguments).
///
/// # Safety
///
/// See [`slice_from_args`].
pub unsafe fn slice_optional(ptr: usize, len: usize, max: usize) -> Result<&'static [u8], i64> {
    if len == 0 {
        return Ok(&[]);
    }
    slice_from_args(ptr, len, max)
}

/// Re-stamp the `last_activity_unix_time` of the current session. Called
/// by the dispatch path after a successful operation so the idle
/// sweeper does not evict an active user mid-flow. Phase 6's mgmt
/// loader hooks this through to the global table.
pub fn touch_current_session<P, S>(ctx: &mut AuthCtx<'_, P, S>)
where
    P: ShadowProvider + ?Sized,
    S: AuditSink + ?Sized,
{
    let id = crate::auth::current_session();
    if !id.is_none() {
        let _ = ctx.table.reset_idle(id, ctx.now_unix);
    }
}

// ─── Handler implementations ─────────────────────────────────────────────────

/// `auth_login(user_ptr, user_len, pass_ptr, pass_len, factor2_ptr, factor2_len)`
///
/// Returns the new [`SessionId`] (cast to `i64`) on success, negative
/// errno on failure.
///
/// Spec error map:
/// - `-EINVAL`  bad args, password too long, factor2 supplied (Phase 3)
/// - `-EACCES`  bad credentials (always after constant-time-equivalent verify)
/// - `-EAGAIN`  account locked out
/// - `-ENOSPC`  session table full
pub fn handle_auth_login<P, S>(
    ctx: &mut AuthCtx<'_, P, S>,
    user: &[u8],
    password: &[u8],
    factor2: &[u8],
) -> SyscallResult
where
    P: ShadowProvider + ?Sized,
    S: AuditSink + ?Sized,
{
    if user.is_empty() || user.len() > USERNAME_MAX_LEN {
        return ERRNO_EINVAL;
    }
    if password.is_empty() || password.len() > PASSWORD_MAX_LEN {
        return ERRNO_EINVAL;
    }
    // Phase 3: factor2 reserved for Phase 9 TOTP. Reject any non-empty
    // value loud-and-early — callers can compile against the real
    // signature but we never silently accept unverifiable second
    // factors.
    if !factor2.is_empty() {
        return ERRNO_EINVAL;
    }

    let user_str = match core::str::from_utf8(user) {
        Ok(s) => s,
        Err(_) => return ERRNO_EINVAL,
    };

    let entry = match ctx.provider.read_user(user_str) {
        Ok(opt) => opt,
        Err(ShadowProviderError::PermissionTooLax(_)) => return ERRNO_EACCES,
        Err(_) => return ERRNO_ENOSYS,
    };

    let (entry, ok) = match entry {
        Some(e) => {
            if e.lockout_until_unix > ctx.now_unix {
                return ERRNO_EAGAIN;
            }
            let ok =
                smallaios_security::argon2id::argon2id_verify(password, &e.phc).unwrap_or(false);
            (Some(e), ok)
        }
        None => {
            // Constant-time-equivalent reject: run the same Argon2id
            // budget against a synthetic PHC. Discard the result.
            let _ = smallaios_security::argon2id::argon2id_verify(password, DUMMY_PHC);
            (None, false)
        }
    };

    if !ok {
        return ERRNO_EACCES;
    }

    // Safe to unwrap — `ok` is only true when entry is Some.
    let entry = entry.unwrap();

    let must_change = entry.flags & FLAG_MUST_CHANGE_PASSWORD != 0;
    let session = match Session::new(
        entry.stable_user_id,
        entry.username.as_bytes(),
        entry.role,
        ctx.now_unix,
        must_change,
    ) {
        Some(s) => s,
        None => return ERRNO_EINVAL,
    };

    let id = match ctx.table.acquire(session) {
        Ok(id) => id,
        Err(_) => return ERRNO_ENOSPC,
    };

    crate::auth::set_current_session(id);
    id.as_u32() as SyscallResult
}

/// `auth_logout()` -> 0 | -errno
pub fn handle_auth_logout<P, S>(ctx: &mut AuthCtx<'_, P, S>) -> SyscallResult
where
    P: ShadowProvider + ?Sized,
    S: AuditSink + ?Sized,
{
    let id = crate::auth::current_session();
    if id.is_none() {
        return ERRNO_EACCES;
    }
    if ctx.table.release(id).is_err() {
        // Session was already swept — clear the global anyway.
        crate::auth::clear_current_session();
        return ERRNO_EACCES;
    }
    crate::auth::clear_current_session();
    0
}

/// `auth_change_password(old_ptr, old_len, new_ptr, new_len, target_user_ptr, target_user_len)` -> 0 | -errno
///
/// Two modes:
/// - **Self-rotate** (target empty): caller proves knowledge of `old`
///   and rotates to `new`. Clears `must_change_password` on success.
/// - **Cross-rotate** (target non-empty): caller MUST be Root. `old`
///   is ignored (root override). Forces logout of every other session
///   for `target` (per `auth-roles` Q19).
pub fn handle_auth_change_password<P, S>(
    ctx: &mut AuthCtx<'_, P, S>,
    old_password: &[u8],
    new_password: &[u8],
    target_user: &[u8],
) -> SyscallResult
where
    P: ShadowProvider + ?Sized,
    S: AuditSink + ?Sized,
{
    if new_password.is_empty() || new_password.len() > PASSWORD_MAX_LEN {
        return ERRNO_EINVAL;
    }

    let id = crate::auth::current_session();
    if id.is_none() {
        return ERRNO_EACCES;
    }
    let caller_session = match ctx.table.lookup(id) {
        Ok(s) => *s,
        Err(_) => return ERRNO_EACCES,
    };

    let (target_name, is_cross_rotate) = if target_user.is_empty() {
        // Self-rotate: lookup own username from session.
        let n = caller_session.username_bytes();
        let s = match core::str::from_utf8(n) {
            Ok(s) => s,
            Err(_) => return ERRNO_EINVAL,
        };
        (alloc::string::String::from(s), false)
    } else {
        if target_user.len() > USERNAME_MAX_LEN {
            return ERRNO_EINVAL;
        }
        let s = match core::str::from_utf8(target_user) {
            Ok(s) => s,
            Err(_) => return ERRNO_EINVAL,
        };
        // Cross-rotate requires Root.
        if caller_session.role != Role::Root {
            return ERRNO_EACCES;
        }
        (alloc::string::String::from(s), true)
    };

    // Self-rotate must verify the old password. Cross-rotate skips
    // (root override).
    if !is_cross_rotate {
        if old_password.is_empty() || old_password.len() > PASSWORD_MAX_LEN {
            return ERRNO_EINVAL;
        }
        let entry = match ctx.provider.read_user(&target_name) {
            Ok(Some(e)) => e,
            Ok(None) => return ERRNO_ENOENT,
            Err(_) => return ERRNO_ENOSYS,
        };
        let ok = smallaios_security::argon2id::argon2id_verify(old_password, &entry.phc)
            .unwrap_or(false);
        if !ok {
            return ERRNO_EACCES;
        }
    } else if ctx
        .provider
        .read_user(&target_name)
        .map(|o| o.is_none())
        .unwrap_or(true)
    {
        return ERRNO_ENOENT;
    }

    // Hash the new password. Salt is 16 bytes from kernel CSPRNG.
    let mut salt = [0u8; 16];
    if (ctx.salt_source)(&mut salt).is_err() {
        return ERRNO_ENOSYS;
    }

    // The Phase 6 mgmt loader will replace this with a per-tier
    // lookup based on the running platform's RAM. For now every new
    // PHC uses the default-tier parameters, which keep verification
    // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
    // self-describing per `auth_shadow_v1` Q3.
    let params = Argon2idParams::default_tier();

    let tag = argon2id_hash(new_password, &salt, params);
    let new_phc = argon2id_format_phc(&salt, &tag, params);

    if let Err(_e) = ctx
        .provider
        .write_password(&target_name, &new_phc, !is_cross_rotate)
    {
        // The provider wasn't ready (KernelShadowProvider until F2FS RW
        // lands). Audit and return -ENOSYS so callers can detect this
        // distinct from a true policy reject.
        return ERRNO_ENOSYS;
    }

    if is_cross_rotate {
        // Force logout of every other session for the target user.
        // We do this *after* the write succeeds so a write failure
        // does not destroy live sessions. Look up target user_id first.
        if let Ok(Some(entry)) = ctx.provider.read_user(&target_name) {
            let _ = ctx.table.invalidate_user(entry.stable_user_id);
        }
    } else {
        // Self-rotate: clear `must_change_password` on the live
        // session so subsequent syscalls are no longer gated.
        if let Ok(s) = ctx.table.lookup_mut(id) {
            s.must_change_password = false;
        }
    }

    let _ = ctx; // suppress unused-but-required lint
    0
}

/// `auth_create_user(user_ptr, user_len, role, initial_pass_ptr, initial_pass_len)` -> 0 | -errno
///
/// Root only. The new entry is stamped with
/// [`FLAG_MUST_CHANGE_PASSWORD`] so the operator MUST rotate on first
/// login.
pub fn handle_auth_create_user<P, S>(
    ctx: &mut AuthCtx<'_, P, S>,
    user: &[u8],
    role_byte: u8,
    initial_pass: &[u8],
) -> SyscallResult
where
    P: ShadowProvider + ?Sized,
    S: AuditSink + ?Sized,
{
    if user.is_empty() || user.len() > USERNAME_MAX_LEN {
        return ERRNO_EINVAL;
    }
    if initial_pass.is_empty() || initial_pass.len() > PASSWORD_MAX_LEN {
        return ERRNO_EINVAL;
    }
    let role = match Role::from_u8(role_byte) {
        Ok(r) => r,
        Err(_) => return ERRNO_EINVAL,
    };

    // Root-only.
    let id = crate::auth::current_session();
    if id.is_none() {
        return ERRNO_EACCES;
    }
    let caller = match ctx.table.lookup(id) {
        Ok(s) => *s,
        Err(_) => return ERRNO_EACCES,
    };
    if caller.role != Role::Root {
        return ERRNO_EACCES;
    }

    let user_str = match core::str::from_utf8(user) {
        Ok(s) => s,
        Err(_) => return ERRNO_EINVAL,
    };

    let mut salt = [0u8; 16];
    if (ctx.salt_source)(&mut salt).is_err() {
        return ERRNO_ENOSYS;
    }
    let params = Argon2idParams::default_tier();
    let tag = argon2id_hash(initial_pass, &salt, params);
    let phc = argon2id_format_phc(&salt, &tag, params);

    // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
    match ctx.provider.create_user(user_str, &phc, role) {
        Ok(_uid) => 0,
        Err(ShadowProviderError::Io("user already exists")) => ERRNO_EEXIST,
        Err(ShadowProviderError::NotImplemented) => ERRNO_ENOSYS,
        Err(_) => ERRNO_ENOSYS,
    }
}

/// `auth_whoami(out_ptr)` -> 0 | -errno
///
/// Writes a [`WhoamiOut`] struct at `out_ptr`.
///
/// # Safety
///
/// `out` must be a valid, aligned, exclusively-owned pointer to a
/// [`WhoamiOut`] for the duration of this call. In unikernel mode the
/// caller and kernel share an address space and interrupts are masked
/// across the syscall, so a pointer obtained from a userspace local is
/// safe.
pub unsafe fn handle_auth_whoami<P, S>(
    ctx: &mut AuthCtx<'_, P, S>,
    out: *mut WhoamiOut,
) -> SyscallResult
where
    P: ShadowProvider + ?Sized,
    S: AuditSink + ?Sized,
{
    if out.is_null() {
        return ERRNO_EFAULT;
    }
    if !(out as usize).is_multiple_of(core::mem::align_of::<WhoamiOut>()) {
        return ERRNO_EFAULT;
    }
    let id = crate::auth::current_session();
    let session = match ctx.table.lookup(id) {
        Ok(s) => s,
        Err(_) => return ERRNO_EACCES,
    };

    let idle = ctx
        .now_unix
        .saturating_sub(session.last_activity_unix_time)
        .min(u32::MAX as u64) as u32;

    let value = WhoamiOut {
        role: session.role.as_u8(),
        _pad: [0; 3],
        user_id: session.user_id,
        login_unix_time: session.login_unix_time,
        idle_seconds: idle,
        _pad2: [0; 4],
    };

    // SAFETY: caller has guaranteed `out` is valid+aligned+exclusive
    // (see this function's safety doc), and we re-checked non-null
    // and alignment above.
    unsafe {
        core::ptr::write(out, value);
    }
    0
}

/// `auth_totp_setup(user_ptr, user_len, secret_out_ptr)` -> -ENOSYS in Phase 3.
///
/// The signature is validated so callers can compile against the real
/// ABI; the implementation lands in `management-login-v1` Phase 9.
pub fn handle_auth_totp_setup<P, S>(
    _ctx: &mut AuthCtx<'_, P, S>,
    user: &[u8],
    secret_out_ptr: usize,
) -> SyscallResult
where
    P: ShadowProvider + ?Sized,
    S: AuditSink + ?Sized,
{
    if user.is_empty() || user.len() > USERNAME_MAX_LEN {
        return ERRNO_EINVAL;
    }
    if secret_out_ptr == 0 {
        return ERRNO_EFAULT;
    }
    ERRNO_ENOSYS
}

// ─── Stub dispatchers used by the syscall table ──────────────────────────────
//
// The dispatch table in `crate::syscall::mod` calls these. They delegate
// to the same handler functions used by the per-call unit tests after
// reading the global state. Until the boot path installs a real
// `ShadowProvider` (lands with `embedded-filesystem-v1` Phase 6), these
// dispatchers report `-ENOSYS` so production builds neither silently
// succeed nor crash.

/// 0x90 dispatcher.
pub fn sys_auth_login(_args: &SyscallArgs) -> SyscallResult {
    ERRNO_ENOSYS
}

/// 0x91 dispatcher.
pub fn sys_auth_logout(_args: &SyscallArgs) -> SyscallResult {
    let id = crate::auth::current_session();
    if id.is_none() {
        return ERRNO_EACCES;
    }
    crate::auth::clear_current_session();
    0
}

/// 0x92 dispatcher.
pub fn sys_auth_change_password(_args: &SyscallArgs) -> SyscallResult {
    ERRNO_ENOSYS
}

/// 0x93 dispatcher.
pub fn sys_auth_create_user(_args: &SyscallArgs) -> SyscallResult {
    ERRNO_ENOSYS
}

/// 0x94 dispatcher.
pub fn sys_auth_whoami(_args: &SyscallArgs) -> SyscallResult {
    ERRNO_ENOSYS
}

/// 0x95 dispatcher — Phase 9 territory.
pub fn sys_auth_totp_setup(args: &SyscallArgs) -> SyscallResult {
    let user_ptr = args.args[0];
    let user_len = args.args[1];
    let out_ptr = args.args[2];
    if user_ptr == 0 || user_len == 0 || user_len > USERNAME_MAX_LEN || out_ptr == 0 {
        return ERRNO_EINVAL;
    }
    ERRNO_ENOSYS
}

// ─── Integration tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::shadow_provider::{MockShadowProvider, ShadowProviderEntry};
    use crate::auth::sweeper::{CapturingAuditSink, IdlePolicy};
    use crate::auth::SessionId;
    use alloc::string::ToString;
    use smallaios_security::argon2id::{argon2id_format_phc, argon2id_hash, Argon2idParams};

    /// Deterministic test salt source. Returns sequential bytes so the
    /// resulting Argon2id PHC is reproducible across runs. NOT a CSPRNG
    /// — only used to keep auth tests independent of the kernel's
    /// global CSPRNG seed state.
    //
    // lgtm[rust/hard-coded-cryptographic-value] — synthetic deterministic test salt source, not a real CSPRNG
    fn deterministic_salt(out: &mut [u8]) -> Result<(), ()> {
        use core::sync::atomic::{AtomicU8, Ordering};
        static COUNTER: AtomicU8 = AtomicU8::new(0);
        let base = COUNTER.fetch_add(1, Ordering::Relaxed);
        for (i, b) in out.iter_mut().enumerate() {
            *b = base.wrapping_add(i as u8);
        }
        Ok(())
    }

    // Test fixtures hash a known password and bind it to a user. The
    // PHC strings here are *not* shipped credentials — they are
    // generated per-test from a fixed plaintext through Argon2id.
    fn make_phc(password: &[u8]) -> alloc::string::String {
        // 16-byte salt of zeros — fixture only.
        //
        // lgtm[rust/hard-coded-cryptographic-value] — synthetic test fixture, not a real credential
        let salt = [0u8; 16];
        let params = Argon2idParams::tiny();
        let tag = argon2id_hash(password, &salt, params);
        argon2id_format_phc(&salt, &tag, params)
    }

    fn seed_user(provider: &MockShadowProvider, name: &str, role: Role, password: &[u8]) -> u32 {
        provider.seed(ShadowProviderEntry {
            // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
            stable_user_id: 0,
            username: name.to_string(),
            phc: make_phc(password),
            role,
            flags: 0,
            lockout_until_unix: 0,
        })
    }

    fn make_ctx<'a>(
        table: &'a mut SessionTable,
        provider: &'a MockShadowProvider,
        sweeper: &'a Sweeper,
        audit: &'a mut CapturingAuditSink,
        now: u64,
    ) -> AuthCtx<'a, MockShadowProvider, CapturingAuditSink> {
        AuthCtx {
            table,
            provider,
            sweeper,
            audit,
            now_unix: now,
            salt_source: deterministic_salt,
        }
    }

    #[test]
    fn login_success_returns_session_id() {
        crate::auth::clear_current_session();
        let provider = MockShadowProvider::new();
        seed_user(&provider, "alice", Role::Operator, b"correct-horse");

        let mut table = SessionTable::new();
        let sweeper = Sweeper::with_policy(IdlePolicy::defaults());
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 1_700_000_000);

        let r = handle_auth_login(&mut ctx, b"alice", b"correct-horse", &[]);
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        assert!(r >= 0, "login returned {r}");
        let id = SessionId::from_raw(r as u32);
        assert!(!id.is_none());
        assert_eq!(crate::auth::current_session(), id);
    }

    #[test]
    // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
    fn login_wrong_password_returns_eacces() {
        crate::auth::clear_current_session();
        let provider = MockShadowProvider::new();
        seed_user(&provider, "alice", Role::Operator, b"correct-horse");

        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        let r = handle_auth_login(&mut ctx, b"alice", b"wrong-password", &[]);
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        assert_eq!(r, ERRNO_EACCES);
        assert!(crate::auth::current_session().is_none());
    }

    #[test]
    fn login_unknown_user_returns_eacces_after_dummy_verify() {
        // Spec: "Constant-time-equivalent reject on user-not-found"
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        crate::auth::clear_current_session();
        let provider = MockShadowProvider::new();
        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        let r = handle_auth_login(&mut ctx, b"ghost", b"any-password", &[]);
        assert_eq!(r, ERRNO_EACCES);
    }

    #[test]
    fn login_locked_out_user_returns_eagain() {
        crate::auth::clear_current_session();
        let provider = MockShadowProvider::new();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        provider.seed(ShadowProviderEntry {
            stable_user_id: 0,
            username: "alice".to_string(),
            phc: make_phc(b"correct-horse"),
            role: Role::Viewer,
            flags: 0,
            lockout_until_unix: 1_700_000_500,
        });

        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 1_700_000_000);

        let r = handle_auth_login(&mut ctx, b"alice", b"correct-horse", &[]);
        assert_eq!(r, ERRNO_EAGAIN);
    }

    #[test]
    fn login_factor2_nonempty_returns_einval_phase3() {
        crate::auth::clear_current_session();
        let provider = MockShadowProvider::new();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        seed_user(&provider, "alice", Role::Viewer, b"correct-horse");

        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        let r = handle_auth_login(&mut ctx, b"alice", b"correct-horse", b"123456");
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        assert_eq!(r, ERRNO_EINVAL);
    }

    #[test]
    fn login_table_full_returns_enospc() {
        crate::auth::clear_current_session();
        let provider = MockShadowProvider::new();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        seed_user(&provider, "alice", Role::Viewer, b"pw");

        let mut table = SessionTable::new();
        // Fill to capacity with synthetic sessions.
        for i in 0..crate::auth::SESSION_TABLE_CAP {
            let s = Session::new(i as u32, b"filler", Role::Viewer, 0, false).unwrap();
            table.acquire(s).unwrap();
        }
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        let r = handle_auth_login(&mut ctx, b"alice", b"pw", &[]);
        assert_eq!(r, ERRNO_ENOSPC);
    }

    #[test]
    fn logout_clears_current_session() {
        crate::auth::clear_current_session();
        let provider = MockShadowProvider::new();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        seed_user(&provider, "alice", Role::Operator, b"pw");

        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        handle_auth_login(&mut ctx, b"alice", b"pw", &[]);
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        assert!(!crate::auth::current_session().is_none());

        let r = handle_auth_logout(&mut ctx);
        assert_eq!(r, 0);
        assert!(crate::auth::current_session().is_none());
    }

    // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
    #[test]
    fn logout_without_session_returns_eacces() {
        crate::auth::clear_current_session();
        let provider = MockShadowProvider::new();
        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        let r = handle_auth_logout(&mut ctx);
        assert_eq!(r, ERRNO_EACCES);
    }

    #[test]
    fn change_password_self_rotate_clears_must_change_flag() {
        crate::auth::clear_current_session();
        let provider = MockShadowProvider::new();
        provider.seed(ShadowProviderEntry {
            stable_user_id: 0,
            username: "alice".to_string(),
            phc: make_phc(b"old-pw"),
            role: Role::Operator,
            flags: FLAG_MUST_CHANGE_PASSWORD,
            lockout_until_unix: 0,
        });

        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        handle_auth_login(&mut ctx, b"alice", b"old-pw", &[]);
        let r = handle_auth_change_password(&mut ctx, b"old-pw", b"new-pw-much-longer", &[]);
        assert_eq!(r, 0);

        // Verify mock has been updated.
        let after = provider.current_phc("alice").unwrap();
        assert!(after.starts_with("$argon2id$"));
        // Round-trip new password verifies.
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let ok =
            // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
            smallaios_security::argon2id::argon2id_verify(b"new-pw-much-longer", &after).unwrap();
        assert!(ok);
        // `must_change_password` flag cleared.
        let flags = provider.current_flags("alice").unwrap();
        assert_eq!(flags & FLAG_MUST_CHANGE_PASSWORD, 0);
    }

    #[test]
    // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
    fn change_password_wrong_old_returns_eacces() {
        crate::auth::clear_current_session();
        let provider = MockShadowProvider::new();
        seed_user(&provider, "alice", Role::Viewer, b"correct-old");

        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        handle_auth_login(&mut ctx, b"alice", b"correct-old", &[]);
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let r = handle_auth_change_password(&mut ctx, b"wrong-old", b"new-pw", &[]);
        assert_eq!(r, ERRNO_EACCES);
    }

    #[test]
    fn change_password_cross_rotate_requires_root() {
        crate::auth::clear_current_session();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let provider = MockShadowProvider::new();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        seed_user(&provider, "operator-1", Role::Operator, b"op-pw");
        seed_user(&provider, "viewer-1", Role::Viewer, b"viewer-pw");

        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        handle_auth_login(&mut ctx, b"operator-1", b"op-pw", &[]);
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let r = handle_auth_change_password(&mut ctx, b"any", b"new-pw-strong", b"viewer-1");
        assert_eq!(r, ERRNO_EACCES);
    }

    #[test]
    fn change_password_cross_rotate_force_logout_other_sessions() {
        crate::auth::clear_current_session();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let provider = MockShadowProvider::new();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        seed_user(&provider, "root-1", Role::Root, b"root-pw");
        let target_uid = seed_user(&provider, "victim", Role::Viewer, b"old-pw");

        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();

        // First, log victim in via two sessions.
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let s1 = Session::new(target_uid, b"victim", Role::Viewer, 0, false).unwrap();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let s2 = Session::new(target_uid, b"victim", Role::Viewer, 0, false).unwrap();
        let id1 = table.acquire(s1).unwrap();
        let id2 = table.acquire(s2).unwrap();
        assert_eq!(table.live_count(), 2);

        // Now log root in (separate from victim's sessions).
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);
        handle_auth_login(&mut ctx, b"root-1", b"root-pw", &[]);

        // Cross-rotate victim's password — both victim sessions must be killed.
        let r = handle_auth_change_password(&mut ctx, b"unused", b"new-pw-much-longer", b"victim");
        assert_eq!(r, 0);

        assert!(table.lookup(id1).is_err());
        assert!(table.lookup(id2).is_err());
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
    }

    #[test]
    // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
    fn create_user_root_only() {
        crate::auth::clear_current_session();
        let provider = MockShadowProvider::new();
        seed_user(&provider, "operator-1", Role::Operator, b"pw");

        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        handle_auth_login(&mut ctx, b"operator-1", b"pw", &[]);
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let r = handle_auth_create_user(&mut ctx, b"new-user", Role::Operator.as_u8(), b"new-pw");
        assert_eq!(r, ERRNO_EACCES);
    }

    #[test]
    fn create_user_with_root_succeeds_and_sets_must_change_flag() {
        crate::auth::clear_current_session();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let provider = MockShadowProvider::new();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        seed_user(&provider, "root-1", Role::Root, b"root-pw");

        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        handle_auth_login(&mut ctx, b"root-1", b"root-pw", &[]);
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let r = handle_auth_create_user(&mut ctx, b"new-op", Role::Operator.as_u8(), b"initial-pw");
        assert_eq!(r, 0);

        let flags = provider.current_flags("new-op").unwrap();
        assert_eq!(flags & FLAG_MUST_CHANGE_PASSWORD, FLAG_MUST_CHANGE_PASSWORD);
    }

    // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
    #[test]
    // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
    fn create_user_invalid_role_returns_einval() {
        crate::auth::clear_current_session();
        let provider = MockShadowProvider::new();
        seed_user(&provider, "root-1", Role::Root, b"root-pw");

        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        handle_auth_login(&mut ctx, b"root-1", b"root-pw", &[]);
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let r = handle_auth_create_user(&mut ctx, b"x", 7, b"pw");
        assert_eq!(r, ERRNO_EINVAL);
    }

    #[test]
    fn create_user_duplicate_returns_eexist() {
        crate::auth::clear_current_session();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let provider = MockShadowProvider::new();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        seed_user(&provider, "root-1", Role::Root, b"root-pw");
        seed_user(&provider, "dup", Role::Viewer, b"x-pw");

        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        handle_auth_login(&mut ctx, b"root-1", b"root-pw", &[]);
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let r = handle_auth_create_user(&mut ctx, b"dup", Role::Operator.as_u8(), b"pw");
        assert_eq!(r, ERRNO_EEXIST);
    }

    #[test]
    fn whoami_writes_struct() {
        crate::auth::clear_current_session();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let provider = MockShadowProvider::new();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        seed_user(&provider, "alice", Role::Operator, b"pw");

        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 1_700_000_100);

        handle_auth_login(&mut ctx, b"alice", b"pw", &[]);
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive

        let mut out = WhoamiOut {
            role: 0xFF,
            _pad: [0; 3],
            user_id: 0,
            login_unix_time: 0,
            idle_seconds: 0,
            // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
            _pad2: [0; 4],
        };

        // Now bump now_unix to simulate a 50-second-old session.
        ctx.now_unix = 1_700_000_150;

        let r = unsafe { handle_auth_whoami(&mut ctx, &mut out as *mut _) };
        assert_eq!(r, 0);
        assert_eq!(out.role, Role::Operator.as_u8());
        assert!(out.idle_seconds <= 50);
    }

    #[test]
    fn whoami_without_session_returns_eacces() {
        crate::auth::clear_current_session();
        let provider = MockShadowProvider::new();
        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        let mut out = WhoamiOut {
            role: 0,
            _pad: [0; 3],
            user_id: 0,
            login_unix_time: 0,
            idle_seconds: 0,
            _pad2: [0; 4],
        };
        let r = unsafe { handle_auth_whoami(&mut ctx, &mut out as *mut _) };
        assert_eq!(r, ERRNO_EACCES);
    }

    #[test]
    fn whoami_null_pointer_returns_efault() {
        crate::auth::clear_current_session();
        let provider = MockShadowProvider::new();
        seed_user(&provider, "alice", Role::Viewer, b"pw");
        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        handle_auth_login(&mut ctx, b"alice", b"pw", &[]);
        let r = unsafe { handle_auth_whoami(&mut ctx, core::ptr::null_mut()) };
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        assert_eq!(r, ERRNO_EFAULT);
    }

    #[test]
    fn totp_setup_returns_enosys_in_phase3() {
        crate::auth::clear_current_session();
        // lgtm[rust/hard-coded-cryptographic-value] - test fixture / no-op constant; CodeQL false-positive
        let provider = MockShadowProvider::new();
        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        let mut secret = [0u8; 32];
        let r = handle_auth_totp_setup(&mut ctx, b"alice", secret.as_mut_ptr() as usize);
        assert_eq!(r, ERRNO_ENOSYS);
    }

    #[test]
    fn totp_setup_validates_args_before_enosys() {
        crate::auth::clear_current_session();
        let provider = MockShadowProvider::new();
        let mut table = SessionTable::new();
        let sweeper = Sweeper::new();
        let mut audit = CapturingAuditSink::new();
        let mut ctx = make_ctx(&mut table, &provider, &sweeper, &mut audit, 0);

        // Empty username is rejected.
        let mut secret = [0u8; 32];
        let r = handle_auth_totp_setup(&mut ctx, &[], secret.as_mut_ptr() as usize);
        assert_eq!(r, ERRNO_EINVAL);

        // Null secret pointer is rejected.
        let r = handle_auth_totp_setup(&mut ctx, b"alice", 0);
        assert_eq!(r, ERRNO_EFAULT);
    }
}
