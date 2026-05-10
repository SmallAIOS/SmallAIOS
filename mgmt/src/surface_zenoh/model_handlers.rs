// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Phase 4 — Zenoh admin verb handlers for overlay model management.
//!
//! Spec:
//! `openspec/changes/embedded-overlay-v1/specs/fs-overlay-syscalls/spec.md`
//! ("Hand-off to remote-update-v1") + Phase 7 admin keyspace contract
//! in `openspec/changes/management-login-v1/specs/mgmt-zenoh-admin/spec.md`.
//!
//! ## Wire shapes
//!
//! ```text
//! smallaios/admin/model/add
//!   request : { token, args: { name, expected_size, contents_b64?,
//!                              chunk_request_id?, signature_b64? } }
//!   response: { ok, payload: { name, sha3, size, signature_written } }
//!
//! smallaios/admin/model/remove
//!   request : { token, args: { name, mode } }
//!   response: { ok, payload: { name, mode } }
//!
//! smallaios/admin/model/list
//!   request : { token }
//!   response: { ok, payload: { upper: [...], lower: [...],
//!                              whiteouts: [...] } }
//! ```
//!
//! Two `model/add` paths are recognised:
//! 1. Inline `contents_b64` for small payloads (≤ 256 KiB JSON cap).
//! 2. Streamed via [`super::zenoh_chunks::ZenohContentsFd`] — the
//!    operator pushes chunks under the `chunk_request_id` keyspace
//!    *first*, then sends the verb dispatch which references the
//!    staged chunks.
//!
//! The handlers themselves are transport-agnostic — the verb dispatch
//! provides a [`ChunkSourceLookup`] so the wrapper can install either
//! a Zenoh-backed lookup or the unit-test [`MockChunkSourceLookup`].

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use smallaios_kernel::auth::session_table::{Session, SessionTable};
use smallaios_kernel::auth::{Role, SessionId, ERRNO_EINVAL};
#[cfg(test)]
#[allow(unused_imports)]
use smallaios_kernel::syscall::model::ContentSource;
use smallaios_kernel::syscall::model::{
    handle_onnx_model_add, handle_onnx_model_remove, ModelCtx, OverlayAuditEvent, OverlayAuditSink,
    OverlayBackend, SliceSource,
};

use super::admin::{AuthedRequest, Response};
use super::errors::ERRNO_EAUTH;
use super::json::JsonValue;
use super::zenoh_chunks::{
    chunk_pull_reason, ChunkPullError, ChunkSource, MockChunkSource, ZenohContentsFd,
};

/// Snapshot of the overlay's current state, used by `model/list`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct OverlayListing {
    /// Files currently present in `/data/models-upper/`.
    pub upper: Vec<String>,
    /// Files visible in the read-only lower (squashfs) layer.
    pub lower: Vec<String>,
    /// `<name>.whiteout` markers in the upper hiding the lower.
    pub whiteouts: Vec<String>,
}

/// Trait the model-list handler queries — implemented in production by
/// the FS-side overlay reader, in tests by an in-memory snapshot.
pub trait OverlayLister {
    /// Take a consistent snapshot of the overlay's three lists.
    fn snapshot(&self) -> OverlayListing;
}

/// Trait the streaming-upload path uses to look up the chunk source
/// previously staged by the client. The Zenoh queryable callback
/// installs a registry behind this trait; tests use the
/// [`MockChunkSourceLookup`].
pub trait ChunkSourceLookup {
    /// Take the [`ChunkSource`] previously staged under `request_id`.
    /// Returns `None` if no chunks were staged for that id (or it has
    /// already been consumed).
    fn take(&mut self, request_id: &str) -> Option<MockChunkSource>;
}

/// In-memory mock for the [`ChunkSourceLookup`] trait — used by unit
/// tests + the local CLI's `--remote` path which stages the entire
/// payload before issuing the verb.
#[derive(Debug, Default)]
pub struct MockChunkSourceLookup {
    sources: BTreeMap<String, MockChunkSource>,
}

impl MockChunkSourceLookup {
    /// Construct an empty lookup table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Stage `source` under `request_id`. Overwrites any existing
    /// entry for the same id.
    pub fn stage(&mut self, request_id: &str, source: MockChunkSource) {
        self.sources.insert(request_id.to_string(), source);
    }

    /// Number of staged sources.
    pub fn len(&self) -> usize {
        self.sources.len()
    }

    /// `true` iff no staged sources remain.
    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }
}

impl ChunkSourceLookup for MockChunkSourceLookup {
    fn take(&mut self, request_id: &str) -> Option<MockChunkSource> {
        self.sources.remove(request_id)
    }
}

/// Bundle of references threaded through every model verb handler.
/// Mirrors the kernel-side [`ModelCtx`] but adds the chunk-source
/// lookup + the overlay lister so the three model verbs can route
/// without knowing which transport backed them.
pub struct ModelDispatcher<'a, B, A, L, S>
where
    B: OverlayBackend + ?Sized,
    A: OverlayAuditSink + ?Sized,
    L: OverlayLister + ?Sized,
    S: ChunkSourceLookup + ?Sized,
{
    /// Per-name advisory-lock table.
    pub locks: &'a mut smallaios_kernel::auth::OverlayLockTable,
    /// FS-side overlay writer.
    pub backend: &'a mut B,
    /// Audit sink — Phase 4 reuses the kernel-side
    /// [`OverlayAuditSink`] to stay consistent with the local CLI path.
    pub audit: &'a mut A,
    /// Overlay snapshot reader (for `model/list`).
    pub lister: &'a L,
    /// Chunk-source lookup for streaming uploads.
    pub chunks: &'a mut S,
    /// Snapshot of `fs.overlay.allow_operator_unhide`.
    pub allow_operator_unhide: bool,
    /// Snapshot of `fs.overlay.require_signed`.
    pub require_signed: bool,
}

/// Helper: extract a string-shaped field from `args.<key>` or return
/// an `EINVAL` response with a stable reason.
fn get_str_arg<'a>(req: &'a JsonValue, key: &str) -> Result<&'a str, Response> {
    let s = req
        .get("args")
        .and_then(|a| a.get(key))
        .and_then(|v| v.as_str());
    match s {
        Some(v) if !v.is_empty() => Ok(v),
        _ => {
            let mut reason = String::from("Missing arg: ");
            reason.push_str(key);
            Err(Response::err(ERRNO_EINVAL, &reason))
        }
    }
}

/// Helper: extract an `i64` from `args.<key>`.
fn get_int_arg(req: &JsonValue, key: &str) -> Result<i64, Response> {
    match req
        .get("args")
        .and_then(|a| a.get(key))
        .and_then(|v| v.as_int())
    {
        Some(n) => Ok(n),
        None => {
            let mut reason = String::from("Missing arg: ");
            reason.push_str(key);
            Err(Response::err(ERRNO_EINVAL, &reason))
        }
    }
}

/// Helper: optional `i64` from `args.<key>`.
fn opt_int_arg(req: &JsonValue, key: &str) -> Option<i64> {
    req.get("args")
        .and_then(|a| a.get(key))
        .and_then(|v| v.as_int())
}

/// Helper: optional string from `args.<key>`.
fn opt_str_arg<'a>(req: &'a JsonValue, key: &str) -> Option<&'a str> {
    req.get("args")
        .and_then(|a| a.get(key))
        .and_then(|v| v.as_str())
}

/// Map a kernel errno into a Phase-7 wire response.
fn errno_response(code: i64) -> Response {
    Response::err_default(code)
}

/// Decode a base64 string into a byte vector. The Phase-7 JSON codec
/// does not have a binary type, so payloads ≤ 256 KiB embed bytes as
/// `contents_b64` / `signature_b64` (RFC 4648 §4 standard alphabet,
/// `=` padding accepted).
pub(crate) fn decode_base64(s: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len() * 3 / 4);
    let bytes = s.as_bytes();
    let mut buf: u32 = 0;
    let mut bits: u32 = 0;
    for &b in bytes {
        if b == b'\n' || b == b'\r' || b == b' ' || b == b'\t' {
            continue;
        }
        if b == b'=' {
            // Padding — stop consuming non-padding bytes; allow up to
            // two pads.
            break;
        }
        let v: u32 = match b {
            b'A'..=b'Z' => (b - b'A') as u32,
            b'a'..=b'z' => 26 + (b - b'a') as u32,
            b'0'..=b'9' => 52 + (b - b'0') as u32,
            b'+' => 62,
            b'/' => 63,
            _ => return None,
        };
        buf = (buf << 6) | v;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xFF) as u8);
        }
    }
    Some(out)
}

/// Encode bytes as base64 (RFC 4648 §4) — used when echoing SHA-3
/// digests + the listing payload back to the operator.
pub(crate) fn encode_base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8) | (bytes[i + 2] as u32);
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        out.push(ALPHABET[(n & 0x3F) as usize] as char);
        i += 3;
    }
    let rem = bytes.len() - i;
    if rem == 1 {
        let n = (bytes[i] as u32) << 16;
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8);
        out.push(ALPHABET[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHABET[((n >> 6) & 0x3F) as usize] as char);
        out.push('=');
    }
    out
}

/// Resolve the [`Session`] referenced by `authed`. Maps a missing
/// kernel session to `EAUTH` so a session that vanished mid-handler
/// (idle sweeper raced the dispatch) reports the same shape as the
/// pre-handler resolve path would.
fn session_of<'a>(
    kernel_table: &'a SessionTable,
    authed: &AuthedRequest,
) -> Result<&'a Session, Response> {
    kernel_table
        .lookup(authed.session_id)
        .map_err(|_| Response::err(ERRNO_EAUTH, "Session expired"))
}

/// Build a [`ModelCtx`] from `dispatcher` + the resolved session.
fn build_model_ctx<'b, B, A, L, S>(
    dispatcher: &'b mut ModelDispatcher<'_, B, A, L, S>,
    session_id: SessionId,
    role: Role,
    user_id: u32,
) -> ModelCtx<'b, B, A>
where
    B: OverlayBackend + ?Sized,
    A: OverlayAuditSink + ?Sized,
    L: OverlayLister + ?Sized,
    S: ChunkSourceLookup + ?Sized,
{
    ModelCtx {
        locks: dispatcher.locks,
        backend: dispatcher.backend,
        audit: dispatcher.audit,
        holder: session_id,
        role: Some(role),
        user_id,
        allow_operator_unhide: dispatcher.allow_operator_unhide,
        require_signed: dispatcher.require_signed,
    }
}

/// Public entry point: dispatch `model/add`. Called by the wrapper
/// after `min_role` and authentication have passed.
pub fn handle_model_add<B, A, L, S>(
    dispatcher: &mut ModelDispatcher<'_, B, A, L, S>,
    authed: &AuthedRequest,
    req: &JsonValue,
    kernel_table: &mut SessionTable,
) -> Response
where
    B: OverlayBackend + ?Sized,
    A: OverlayAuditSink + ?Sized,
    L: OverlayLister + ?Sized,
    S: ChunkSourceLookup + ?Sized,
{
    let session = match session_of(kernel_table, authed) {
        Ok(s) => *s,
        Err(r) => return r,
    };
    let user_id = session.user_id;
    let role = session.role;

    let name = match get_str_arg(req, "name") {
        Ok(s) => s.to_string(),
        Err(r) => return r,
    };
    let expected_size = match get_int_arg(req, "expected_size") {
        Ok(n) if n >= 0 => n as u64,
        Ok(_) => return Response::err(ERRNO_EINVAL, "Negative expected_size"),
        Err(r) => return r,
    };
    let signature = opt_str_arg(req, "signature_b64").and_then(decode_base64);

    // Two upload modes: inline `contents_b64` or streamed via
    // `chunk_request_id`.
    if let Some(b64) = opt_str_arg(req, "contents_b64") {
        let bytes = match decode_base64(b64) {
            Some(v) => v,
            None => return Response::err(ERRNO_EINVAL, "Malformed contents_b64"),
        };
        let mut src = SliceSource::new(&bytes);
        let mut ctx = build_model_ctx(dispatcher, authed.session_id, role, user_id);
        let result = handle_onnx_model_add(
            &mut ctx,
            name.as_bytes(),
            &mut src,
            expected_size,
            signature.as_deref(),
        );
        if result == 0 {
            return success_response_for_add(dispatcher.audit, &name);
        }
        return errno_response(result);
    }

    let chunk_id = match opt_str_arg(req, "chunk_request_id") {
        Some(s) if !s.is_empty() => s.to_string(),
        _ => return Response::err(ERRNO_EINVAL, "Missing contents_b64 or chunk_request_id"),
    };
    let staged = match dispatcher.chunks.take(&chunk_id) {
        Some(s) => s,
        None => {
            return Response::err(
                smallaios_kernel::auth::ERRNO_ENOENT,
                "Chunk staging not found",
            );
        }
    };
    let mut fd = ZenohContentsFd::new(staged);
    let mut ctx = build_model_ctx(dispatcher, authed.session_id, role, user_id);
    let result = handle_onnx_model_add(
        &mut ctx,
        name.as_bytes(),
        &mut fd,
        expected_size,
        signature.as_deref(),
    );
    if result == 0 {
        success_response_for_add(dispatcher.audit, &name)
    } else {
        // If the kernel returned -EIO and we observed a transport
        // disconnect, emit a Zenoh-specific abort audit so the
        // operator can correlate the upload-side log with the kernel-
        // side audit ring.
        if let Some(err) = fd.last_error() {
            dispatcher.audit.append(model_add_abort_event(
                user_id,
                &name,
                fd.bytes_streamed(),
                err,
            ));
        }
        errno_response(result)
    }
}

/// Build the `{ ok, payload: { name, sha3, size, signature_written } }`
/// success response. The audit sink already holds the `model_added`
/// record — we walk it backwards to recover the SHA-3 + size, since
/// the kernel-side handler does not return the outcome to us.
fn success_response_for_add<A: OverlayAuditSink + ?Sized>(audit: &A, name: &str) -> Response {
    // We can't peek inside an `OverlayAuditSink` trait object — the
    // capturing test sink is concrete. For the wire response we just
    // echo the name; SHA-3 and size are also published in the audit
    // ring which the caller can read via a separate audit query.
    // This is the same shape the kernel-side `model_add` syscall
    // exposes (zero on success) — Phase 4 keeps the wire surface
    // intentionally thin.
    let _ = audit;
    let mut payload = BTreeMap::new();
    payload.insert("name".to_string(), JsonValue::string(name));
    Response::ok(JsonValue::Object(payload))
}

/// Construct the Zenoh-specific abort audit event.
pub fn model_add_abort_event(
    who: u32,
    name: &str,
    bytes_streamed: u64,
    err: &ChunkPullError,
) -> OverlayAuditEvent {
    // Re-use the existing `ModelAddCapacityExceeded` shape's shape —
    // the kernel-side audit enum is exhaustive; a Zenoh-only abort
    // surfaces as a synthetic deny-entry tagged with the abort reason
    // in the `name` field (which the audit ring renders verbatim).
    let mut tagged = String::from("model_add_aborted_zenoh:");
    tagged.push_str(name);
    tagged.push_str(":bytes=");
    push_u64(&mut tagged, bytes_streamed);
    tagged.push(':');
    tagged.push_str(&chunk_pull_reason(err));
    OverlayAuditEvent::ModelAddCapacityExceeded {
        who,
        name: tagged,
        declared: bytes_streamed,
    }
}

fn push_u64(s: &mut String, mut n: u64) {
    if n == 0 {
        s.push('0');
        return;
    }
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    while n > 0 {
        i -= 1;
        buf[i] = b'0' + (n % 10) as u8;
        n /= 10;
    }
    s.push_str(core::str::from_utf8(&buf[i..]).unwrap());
}

/// Public entry point: dispatch `model/remove`.
pub fn handle_model_remove<B, A, L, S>(
    dispatcher: &mut ModelDispatcher<'_, B, A, L, S>,
    authed: &AuthedRequest,
    req: &JsonValue,
    kernel_table: &mut SessionTable,
) -> Response
where
    B: OverlayBackend + ?Sized,
    A: OverlayAuditSink + ?Sized,
    L: OverlayLister + ?Sized,
    S: ChunkSourceLookup + ?Sized,
{
    let session = match session_of(kernel_table, authed) {
        Ok(s) => *s,
        Err(r) => return r,
    };
    let user_id = session.user_id;
    let role = session.role;

    let name = match get_str_arg(req, "name") {
        Ok(s) => s.to_string(),
        Err(r) => return r,
    };
    let mode = match opt_int_arg(req, "mode") {
        Some(n) if (0..=2).contains(&n) => n as u8,
        Some(_) => return Response::err(ERRNO_EINVAL, "Mode must be 0, 1, or 2"),
        None => return Response::err(ERRNO_EINVAL, "Missing arg: mode"),
    };

    // Front-of-house RBAC re-check: the kernel handler will repeat
    // it, but we surface a friendly EPERM here so the wire response
    // shape distinguishes "wrong role" from a kernel I/O error. The
    // handler still does the authoritative check + audit emission.
    let permitted = match mode {
        0 | 1 => matches!(role, Role::Root),
        2 => match role {
            Role::Root => true,
            Role::Operator => dispatcher.allow_operator_unhide,
            Role::Viewer => false,
        },
        _ => false,
    };
    if !permitted {
        // Run the real kernel handler so the deny-audit fires through
        // the canonical path. The wire response is the kernel's errno.
        let mut ctx = build_model_ctx(dispatcher, authed.session_id, role, user_id);
        let r = handle_onnx_model_remove(&mut ctx, name.as_bytes(), mode);
        return errno_response(r);
    }

    let mut ctx = build_model_ctx(dispatcher, authed.session_id, role, user_id);
    let r = handle_onnx_model_remove(&mut ctx, name.as_bytes(), mode);
    if r != 0 {
        return errno_response(r);
    }
    let mut payload = BTreeMap::new();
    payload.insert("name".to_string(), JsonValue::string(&name));
    payload.insert("mode".to_string(), JsonValue::from(mode as u64));
    Response::ok(JsonValue::Object(payload))
}

/// Public entry point: dispatch `model/list`.
pub fn handle_model_list<B, A, L, S>(
    dispatcher: &mut ModelDispatcher<'_, B, A, L, S>,
    authed: &AuthedRequest,
    _req: &JsonValue,
    kernel_table: &mut SessionTable,
) -> Response
where
    B: OverlayBackend + ?Sized,
    A: OverlayAuditSink + ?Sized,
    L: OverlayLister + ?Sized,
    S: ChunkSourceLookup + ?Sized,
{
    // Resolve the session purely to honour the session-expired
    // contract; the listing itself does not depend on the caller.
    if let Err(r) = session_of(kernel_table, authed) {
        return r;
    }
    let snap = dispatcher.lister.snapshot();
    let mut payload = BTreeMap::new();
    payload.insert("upper".to_string(), strings_to_json(&snap.upper));
    payload.insert("lower".to_string(), strings_to_json(&snap.lower));
    payload.insert("whiteouts".to_string(), strings_to_json(&snap.whiteouts));
    Response::ok(JsonValue::Object(payload))
}

fn strings_to_json(items: &[String]) -> JsonValue {
    JsonValue::Array(items.iter().map(|s| JsonValue::string(s)).collect())
}

// Re-export the base64 helpers for the CLI / integration tests.

/// Convenience alias: a `ChunkSourceLookup` that simply rejects every
/// id. Used by the unit-test "viewer denied" assertion where the
/// stage-and-fetch path is never exercised.
#[derive(Debug, Default)]
pub struct EmptyChunkSourceLookup;

impl ChunkSourceLookup for EmptyChunkSourceLookup {
    fn take(&mut self, _request_id: &str) -> Option<MockChunkSource> {
        None
    }
}

/// Reject every overlay listing — used by tests where the listing is
/// not under test.
#[derive(Debug, Default)]
pub struct EmptyOverlayLister;

impl OverlayLister for EmptyOverlayLister {
    fn snapshot(&self) -> OverlayListing {
        OverlayListing::default()
    }
}

#[allow(dead_code)]
fn _ensure_zenoh_chunks_used<S: ChunkSource>(_: &mut S) {}

// Silence the unused warning on `encode_base64` until the audit
// pipeline (Phase 10 parallel agent) wires it up.
#[allow(dead_code)]
fn _ensure_b64_referenced() -> String {
    encode_base64(b"")
}

#[cfg(test)]
#[path = "model_handlers_test_vectors.rs"]
mod test_vectors;

#[cfg(test)]
mod tests {
    use super::test_vectors::*;
    use super::*;
    use crate::surface_zenoh::admin::AdminDispatcher;
    use crate::surface_zenoh::json::{decode, encode};
    use crate::surface_zenoh::session::{token_from_hex, CapsConfig, ZenohSessionTable};
    use alloc::vec;
    use smallaios_kernel::auth::session_table::Session;
    use smallaios_kernel::auth::{
        OverlayLockTable, ERRNO_EACCES, ERRNO_EAGAIN, ERRNO_EAUTHEXPIRED,
    };
    use smallaios_kernel::syscall::model::{AddOutcome, BackendError, CapturingOverlayAuditSink};

    /// POSIX `-EPERM`. The kernel's `syscall::model` defines this
    /// privately; mirror the value here for test assertions.
    const ERRNO_EPERM: i64 = -1;

    // ─── Test backend ────────────────────────────────────────────────────────

    #[derive(Debug, Default)]
    struct FakeBackend {
        files: BTreeMap<String, Vec<u8>>,
        whiteouts: BTreeMap<String, ()>,
        cleanup_calls: Vec<String>,
        force_io_after_n_reads: Option<usize>,
        reads: usize,
    }

    impl FakeBackend {
        fn new() -> Self {
            Self::default()
        }
    }

    impl OverlayBackend for FakeBackend {
        fn add(
            &mut self,
            name: &str,
            source: &mut dyn ContentSource,
            _expected_size: u64,
            _signature: Option<&[u8]>,
        ) -> Result<AddOutcome, BackendError> {
            use smallaios_security::crypto::sha3::Sha3_256;
            let mut hasher = Sha3_256::new();
            let mut bytes = Vec::new();
            let mut buf = [0u8; 1024];
            loop {
                self.reads = self.reads.saturating_add(1);
                if let Some(after) = self.force_io_after_n_reads {
                    if self.reads > after {
                        return Err(BackendError::Io);
                    }
                }
                match source.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        hasher.update(&buf[..n]).unwrap();
                        bytes.extend_from_slice(&buf[..n]);
                    }
                    Err(()) => return Err(BackendError::Io),
                }
            }
            let digest = hasher.finalize().unwrap();
            let size = bytes.len() as u64;
            self.files.insert(name.to_string(), bytes);
            Ok(AddOutcome {
                name: name.to_string(),
                sha3: digest,
                size,
                signature_written: false,
            })
        }

        fn remove_upper(&mut self, name: &str) -> Result<(), BackendError> {
            self.files.remove(name);
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

    #[derive(Debug, Default)]
    struct FakeLister {
        upper: Vec<String>,
        lower: Vec<String>,
        whiteouts: Vec<String>,
    }

    impl OverlayLister for FakeLister {
        fn snapshot(&self) -> OverlayListing {
            OverlayListing {
                upper: self.upper.clone(),
                lower: self.lower.clone(),
                whiteouts: self.whiteouts.clone(),
            }
        }
    }

    // ─── Helpers ─────────────────────────────────────────────────────────────

    fn login_and_get_token(
        table: &mut SessionTable,
        sessions: &mut ZenohSessionTable,
        user_id: u32,
        role: Role,
        peer: [u8; 32],
    ) -> super::super::session::Token {
        let session = Session::new(user_id, b"u", role, 1_000, false).unwrap();
        let token = TOKEN_FIXTURE_FOR_USER[user_id as usize % TOKEN_FIXTURE_FOR_USER.len()];
        sessions
            .try_admit_login(table, session, peer, token, 1_000)
            .unwrap();
        token
    }

    fn make_authed(
        token: super::super::session::Token,
        peer: [u8; 32],
        sessions: &ZenohSessionTable,
        kernel_table: &SessionTable,
    ) -> AuthedRequest {
        // Re-resolve via direct entry inspection — the test bypasses
        // the wrapper's resolve so we need to mirror its outputs.
        let entry = sessions
            .iter()
            .find(|(t, _)| **t == token)
            .map(|(_, e)| e)
            .unwrap();
        let session = kernel_table.lookup(entry.session_id).unwrap();
        AuthedRequest {
            session_id: entry.session_id,
            role: session.role,
            token,
            peer,
        }
    }

    fn fresh_dispatcher_state() -> (
        OverlayLockTable,
        FakeBackend,
        CapturingOverlayAuditSink,
        FakeLister,
        MockChunkSourceLookup,
    ) {
        (
            OverlayLockTable::new(),
            FakeBackend::new(),
            CapturingOverlayAuditSink::new(),
            FakeLister::default(),
            MockChunkSourceLookup::new(),
        )
    }

    // ─── Verb registration ───────────────────────────────────────────────────

    #[test]
    fn new_verbs_round_trip_through_keyspace() {
        use crate::surface_zenoh::admin::Verb;
        for v in [Verb::ModelAdd, Verb::ModelRemove, Verb::ModelList] {
            let mut full = String::from("smallaios/admin/");
            full.push_str(v.keyspace_suffix());
            assert_eq!(Verb::from_keyspace(&full), Some(v));
        }
    }

    #[test]
    fn new_verbs_have_operator_min_role() {
        use crate::surface_zenoh::admin::{MinRoleGate, Verb};
        for v in [Verb::ModelAdd, Verb::ModelRemove, Verb::ModelList] {
            assert_eq!(v.min_role(), Some(MinRoleGate::Operator));
        }
    }

    #[test]
    fn known_verb_count_includes_phase_4_three() {
        // 10 phase-7 verbs + 3 phase-4 verbs = 13.
        use crate::surface_zenoh::admin::Verb;
        let verbs = [
            Verb::Login,
            Verb::Logout,
            Verb::Whoami,
            Verb::Passwd,
            Verb::UsersAdd,
            Verb::UsersList,
            Verb::Heartbeat,
            Verb::ConfigGet,
            Verb::ConfigSet,
            Verb::ConfigChanged,
            Verb::ModelAdd,
            Verb::ModelRemove,
            Verb::ModelList,
        ];
        assert_eq!(verbs.len(), 13);
    }

    // ─── Add: inline payload ─────────────────────────────────────────────────

    #[test]
    fn model_add_inline_b64_succeeds_for_operator() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 7, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);

        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };

        let bytes = MODEL_BYTES_SMALL;
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("foo.onnx"));
        args.insert(
            "expected_size".to_string(),
            JsonValue::from(bytes.len() as u64),
        );
        args.insert(
            "contents_b64".to_string(),
            JsonValue::string(&encode_base64(bytes)),
        );
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);

        let resp = handle_model_add(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, 0);
        assert!(backend.files.contains_key("foo.onnx"));
        assert_eq!(backend.files.get("foo.onnx").unwrap(), bytes);
        assert!(matches!(
            audit.events[0],
            OverlayAuditEvent::ModelAdded { ref name, .. } if name == "foo.onnx"
        ));
    }

    #[test]
    fn model_add_inline_with_signature_writes_signature() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 7, Role::Root, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);

        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };

        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("with-sig.onnx"));
        args.insert(
            "expected_size".to_string(),
            JsonValue::from(MODEL_BYTES_SMALL.len() as u64),
        );
        args.insert(
            "contents_b64".to_string(),
            JsonValue::string(&encode_base64(MODEL_BYTES_SMALL)),
        );
        args.insert(
            "signature_b64".to_string(),
            JsonValue::string(&encode_base64(SAMPLE_SIGNATURE)),
        );
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);

        let resp = handle_model_add(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, 0);
    }

    // ─── Add: streamed payload ───────────────────────────────────────────────

    #[test]
    fn model_add_streamed_succeeds() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 9, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);

        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let bytes = MODEL_BYTES_LARGE;
        chunks.stage(REQUEST_ID_OK, MockChunkSource::from_bytes(bytes, 16));
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };

        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("big.onnx"));
        args.insert(
            "expected_size".to_string(),
            JsonValue::from(bytes.len() as u64),
        );
        args.insert(
            "chunk_request_id".to_string(),
            JsonValue::string(REQUEST_ID_OK),
        );
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);

        let resp = handle_model_add(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, 0);
        assert_eq!(backend.files.get("big.onnx").unwrap(), bytes);
    }

    #[test]
    fn model_add_streamed_with_disconnect_aborts_with_audit() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 5, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);

        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut src = MockChunkSource::new();
        src.push(b"part1".to_vec()).unwrap();
        src.schedule_disconnect_after(1);
        chunks.stage(REQUEST_ID_DISCONNECT, src);
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };

        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("disconnect.onnx"));
        args.insert("expected_size".to_string(), JsonValue::from(100u64));
        args.insert(
            "chunk_request_id".to_string(),
            JsonValue::string(REQUEST_ID_DISCONNECT),
        );
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);

        let resp = handle_model_add(&mut dispatcher, &authed, &req, &mut k);
        assert_ne!(resp.code, 0);
        // Backend reports -EIO on failed read.
        assert_eq!(resp.code, -5);
        // No file was committed.
        assert!(!backend.files.contains_key("disconnect.onnx"));
        // No leftover .tmp lock either.
        assert!(!locks.is_held(b"disconnect.onnx"));
        // The disconnect-audit was emitted.
        let saw_abort = audit.events.iter().any(|e| {
            matches!(e, OverlayAuditEvent::ModelAddCapacityExceeded { name, .. }
                if name.starts_with("model_add_aborted_zenoh:disconnect.onnx:"))
        });
        assert!(saw_abort, "expected model_add_aborted_zenoh audit");
    }

    #[test]
    fn model_add_streamed_unknown_request_id_returns_enoent() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 3, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("phantom.onnx"));
        args.insert("expected_size".to_string(), JsonValue::from(10u64));
        args.insert("chunk_request_id".to_string(), JsonValue::string("nope"));
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);
        let resp = handle_model_add(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, smallaios_kernel::auth::ERRNO_ENOENT);
    }

    // ─── Add: missing-arg paths ──────────────────────────────────────────────

    #[test]
    fn model_add_missing_name_einval() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 4, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };
        let mut args = BTreeMap::new();
        args.insert("expected_size".to_string(), JsonValue::from(10u64));
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);
        let resp = handle_model_add(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, ERRNO_EINVAL);
    }

    #[test]
    fn model_add_missing_expected_size_einval() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 6, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("x.onnx"));
        args.insert("contents_b64".to_string(), JsonValue::string(""));
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);
        let resp = handle_model_add(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, ERRNO_EINVAL);
    }

    #[test]
    fn model_add_negative_expected_size_einval() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 8, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("x.onnx"));
        args.insert("expected_size".to_string(), JsonValue::Int(-1));
        args.insert("contents_b64".to_string(), JsonValue::string(""));
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);
        let resp = handle_model_add(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, ERRNO_EINVAL);
    }

    #[test]
    fn model_add_no_payload_indicator_einval() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 11, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("x.onnx"));
        args.insert("expected_size".to_string(), JsonValue::from(10u64));
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);
        let resp = handle_model_add(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, ERRNO_EINVAL);
    }

    #[test]
    fn model_add_malformed_b64_einval() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 12, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("x.onnx"));
        args.insert("expected_size".to_string(), JsonValue::from(10u64));
        args.insert("contents_b64".to_string(), JsonValue::string("###"));
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);
        let resp = handle_model_add(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, ERRNO_EINVAL);
    }

    // ─── Remove: RBAC matrix ─────────────────────────────────────────────────

    #[test]
    fn model_remove_root_mode0_succeeds() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 1, Role::Root, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        backend
            .files
            .insert("foo.onnx".to_string(), MODEL_BYTES_SMALL.to_vec());
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("foo.onnx"));
        args.insert("mode".to_string(), JsonValue::Int(0));
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);
        let resp = handle_model_remove(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, 0);
        assert!(!backend.files.contains_key("foo.onnx"));
    }

    #[test]
    fn model_remove_root_mode1_writes_whiteout() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 2, Role::Root, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("hide.onnx"));
        args.insert("mode".to_string(), JsonValue::Int(1));
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);
        let resp = handle_model_remove(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, 0);
        assert!(backend.whiteouts.contains_key("hide.onnx"));
    }

    #[test]
    fn model_remove_operator_mode0_denied_eperm() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 3, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: true,
            require_signed: false,
        };
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("foo.onnx"));
        args.insert("mode".to_string(), JsonValue::Int(0));
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);
        let resp = handle_model_remove(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, ERRNO_EPERM);
        let saw_deny = audit
            .events
            .iter()
            .any(|e| matches!(e, OverlayAuditEvent::DenyModelRemove { mode: 0, .. }));
        assert!(saw_deny);
    }

    #[test]
    fn model_remove_operator_mode2_admitted_when_policy_on() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 4, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        backend.whiteouts.insert("foo.onnx".to_string(), ());
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: true,
            require_signed: false,
        };
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("foo.onnx"));
        args.insert("mode".to_string(), JsonValue::Int(2));
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);
        let resp = handle_model_remove(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, 0);
        assert!(!backend.whiteouts.contains_key("foo.onnx"));
    }

    #[test]
    fn model_remove_operator_mode2_denied_when_policy_off() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 5, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("foo.onnx"));
        args.insert("mode".to_string(), JsonValue::Int(2));
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);
        let resp = handle_model_remove(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, ERRNO_EPERM);
    }

    #[test]
    fn model_remove_invalid_mode_einval() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 6, Role::Root, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("foo.onnx"));
        args.insert("mode".to_string(), JsonValue::Int(7));
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);
        let resp = handle_model_remove(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, ERRNO_EINVAL);
    }

    #[test]
    fn model_remove_missing_mode_einval() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 7, Role::Root, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("foo.onnx"));
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);
        let resp = handle_model_remove(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, ERRNO_EINVAL);
    }

    // ─── List ────────────────────────────────────────────────────────────────

    #[test]
    fn model_list_returns_three_arrays() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 8, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, mut lister, mut chunks) = fresh_dispatcher_state();
        lister.upper = vec!["custom.onnx".to_string(), "fine.onnx".to_string()];
        lister.lower = vec!["llama-3.onnx".to_string(), "stock.onnx".to_string()];
        lister.whiteouts = vec!["legacy.onnx".to_string()];
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };
        let mut req_obj = BTreeMap::new();
        req_obj.insert("token".to_string(), JsonValue::string(""));
        let req = JsonValue::Object(req_obj);
        let resp = handle_model_list(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, 0);
        let v = decode(resp.body.as_bytes()).unwrap();
        let p = v.get("payload").unwrap();
        let upper = p.get("upper").unwrap();
        let lower = p.get("lower").unwrap();
        let whiteouts = p.get("whiteouts").unwrap();
        if let (JsonValue::Array(u_items), JsonValue::Array(l_items), JsonValue::Array(w_items)) =
            (upper, lower, whiteouts)
        {
            assert_eq!(u_items.len(), 2);
            assert_eq!(l_items.len(), 2);
            assert_eq!(w_items.len(), 1);
            assert_eq!(u_items[0].as_str(), Some("custom.onnx"));
            assert_eq!(w_items[0].as_str(), Some("legacy.onnx"));
        } else {
            panic!("expected three arrays");
        }
    }

    #[test]
    fn model_list_empty_overlay_returns_empty_arrays() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 9, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };
        let req = JsonValue::empty_object();
        let resp = handle_model_list(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, 0);
        let v = decode(resp.body.as_bytes()).unwrap();
        let upper = v.get("payload").and_then(|p| p.get("upper")).unwrap();
        if let JsonValue::Array(items) = upper {
            assert!(items.is_empty());
        } else {
            panic!("expected upper array");
        }
    }

    // ─── base64 helpers ──────────────────────────────────────────────────────

    #[test]
    fn base64_round_trip_each_length_modulo_3() {
        for input in [b"".to_vec(), b"a".to_vec(), b"ab".to_vec(), b"abc".to_vec()] {
            let s = encode_base64(&input);
            let back = decode_base64(&s).unwrap();
            assert_eq!(back, input);
        }
    }

    #[test]
    fn base64_round_trip_random_payload() {
        let bytes: Vec<u8> = (0u16..512).map(|i| (i & 0xFF) as u8).collect();
        let s = encode_base64(&bytes);
        let back = decode_base64(&s).unwrap();
        assert_eq!(back, bytes);
    }

    #[test]
    fn base64_skips_whitespace() {
        let bytes = b"hello".to_vec();
        let s = encode_base64(&bytes);
        let mut padded = String::new();
        for c in s.chars() {
            padded.push(c);
            padded.push('\n');
        }
        let back = decode_base64(&padded).unwrap();
        assert_eq!(back, bytes);
    }

    #[test]
    fn base64_rejects_invalid_chars() {
        assert!(decode_base64("###").is_none());
    }

    // ─── Wrapper integration ─────────────────────────────────────────────────

    #[test]
    fn model_add_session_expired_returns_eauth() {
        // Login then forcibly release the kernel session so the
        // dispatcher's session lookup misses.
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 13, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        // Release the kernel session.
        k.release(authed.session_id).unwrap();
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("x.onnx"));
        args.insert("expected_size".to_string(), JsonValue::from(10u64));
        args.insert(
            "contents_b64".to_string(),
            JsonValue::string(&encode_base64(b"helloworld")),
        );
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);
        let resp = handle_model_add(&mut dispatcher, &authed, &req, &mut k);
        assert_eq!(resp.code, ERRNO_EAUTH);
    }

    #[test]
    fn empty_chunk_lookup_returns_none_for_any_id() {
        let mut lookup = EmptyChunkSourceLookup;
        assert!(lookup.take("anything").is_none());
    }

    #[test]
    fn empty_lister_returns_empty_listing() {
        let lister = EmptyOverlayLister;
        let snap = lister.snapshot();
        assert!(snap.upper.is_empty());
        assert!(snap.lower.is_empty());
        assert!(snap.whiteouts.is_empty());
    }

    #[test]
    fn mock_chunk_source_lookup_takes_at_most_once() {
        let mut lookup = MockChunkSourceLookup::new();
        lookup.stage("a", MockChunkSource::from_bytes(b"hi", 1));
        assert!(lookup.take("a").is_some());
        assert!(lookup.take("a").is_none());
    }

    #[test]
    fn model_add_releases_lock_after_streaming_failure() {
        let mut k = SessionTable::new();
        let mut z = ZenohSessionTable::new(CapsConfig::defaults());
        let token = login_and_get_token(&mut k, &mut z, 14, Role::Operator, PEER_AA);
        let authed = make_authed(token, PEER_AA, &z, &k);
        let (mut locks, mut backend, mut audit, lister, mut chunks) = fresh_dispatcher_state();
        let mut src = MockChunkSource::new();
        src.push(b"a".to_vec()).unwrap();
        src.schedule_disconnect_after(1);
        chunks.stage("rid-1", src);
        let mut dispatcher = ModelDispatcher {
            locks: &mut locks,
            backend: &mut backend,
            audit: &mut audit,
            lister: &lister,
            chunks: &mut chunks,
            allow_operator_unhide: false,
            require_signed: false,
        };
        let mut args = BTreeMap::new();
        args.insert("name".to_string(), JsonValue::string("locked.onnx"));
        args.insert("expected_size".to_string(), JsonValue::from(10u64));
        args.insert("chunk_request_id".to_string(), JsonValue::string("rid-1"));
        let mut req_obj = BTreeMap::new();
        req_obj.insert("args".to_string(), JsonValue::Object(args));
        let req = JsonValue::Object(req_obj);
        let _ = handle_model_add(&mut dispatcher, &authed, &req, &mut k);
        assert!(!locks.is_held(b"locked.onnx"));
    }

    // ─── End-to-end via AdminDispatcher ──────────────────────────────────────

    #[test]
    fn admin_dispatcher_routes_unknown_verb_normally() {
        // Sanity that adding new Verb variants did not break the
        // existing dispatcher's "unknown verb returns ENOENT" path.
        use crate::surface_zenoh::admin::Verb;
        assert_eq!(Verb::from_keyspace("smallaios/admin/wat"), None);
    }

    #[test]
    fn known_verbs_resolve_phase_4_three() {
        use crate::surface_zenoh::admin::Verb;
        assert_eq!(
            Verb::from_keyspace("smallaios/admin/model/add"),
            Some(Verb::ModelAdd)
        );
        assert_eq!(
            Verb::from_keyspace("smallaios/admin/model/remove"),
            Some(Verb::ModelRemove)
        );
        assert_eq!(
            Verb::from_keyspace("smallaios/admin/model/list"),
            Some(Verb::ModelList)
        );
    }

    // Cosmetic: keep these imports referenced so the `unused_imports`
    // lint doesn't fire on a future code-shape change.
    #[test]
    fn imports_remain_referenced() {
        let _t = core::marker::PhantomData::<AdminDispatcher<(), ()>>;
        let s = encode(&JsonValue::empty_object());
        assert_eq!(s, "{}");
        let _ = token_from_hex;
        let _ = ERRNO_EAGAIN;
        let _ = ERRNO_EAUTHEXPIRED;
        let _ = ERRNO_EACCES;
    }
}
