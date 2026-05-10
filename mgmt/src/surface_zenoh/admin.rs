// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Admin keyspace verb dispatcher and bearer-token wrapper.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-zenoh-admin/spec.md`,
//! Requirements:
//!
//! - "Admin keyspace contract"
//! - "Bearer-token lifecycle"
//! - "Peer-identity binding"
//!
//! ## Verb registry
//!
//! Phase 7 v1 verbs:
//!
//! | Verb              | Min role        | Auth wrapper |
//! |-------------------|-----------------|--------------|
//! | `login`           | Unauthenticated | bypass       |
//! | `logout`          | Authenticated   | wrap         |
//! | `whoami`          | Authenticated   | wrap         |
//! | `passwd`          | Authenticated   | wrap         |
//! | `users/add`       | Root            | wrap + role  |
//! | `users/list`      | Operator        | wrap + role  |
//! | `heartbeat`       | Authenticated   | wrap         |
//! | `config/get`      | Operator        | wrap + role  |
//! | `config/set`      | Root            | wrap + role  |
//! | `config/changed`  | Authenticated   | subscribe    |
//!
//! Every wrapped call runs the eight-step sequence enumerated in the
//! `Bearer-token authentication wrapper` requirement before any
//! verb-specific work executes.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use smallaios_kernel::auth::session_table::{Session, SessionTable};
use smallaios_kernel::auth::{
    Role, ERRNO_EACCES, ERRNO_EAGAIN, ERRNO_EAUTHEXPIRED, ERRNO_EINVAL, ERRNO_ENOENT, ERRNO_ENOSPC,
};

use super::errors::{error_string, ERRNO_EAUTH, ERRNO_EPERM};
use super::json::{decode, encode, JsonValue};
use super::session::{token_hex, PeerIdentity, Token, ZenohSessionTable};
use super::token::{generate_opaque, RandFill};

/// Hard cap on payload bytes accepted by [`AdminDispatcher::dispatch`].
/// Mirrors [`super::json::MAX_PAYLOAD_BYTES`] but recorded here so the
/// wrapper can emit an `EINVAL` rejection without first invoking the
/// JSON decoder when the wire frame is obviously oversized.
pub const MAX_REQUEST_BYTES: usize = super::json::MAX_PAYLOAD_BYTES;

/// Unique identifier for each Phase 7 verb.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verb {
    /// `smallaios/admin/login`.
    Login,
    /// `smallaios/admin/logout`.
    Logout,
    /// `smallaios/admin/whoami`.
    Whoami,
    /// `smallaios/admin/passwd`.
    Passwd,
    /// `smallaios/admin/users/add`.
    UsersAdd,
    /// `smallaios/admin/users/list`.
    UsersList,
    /// `smallaios/admin/heartbeat`.
    Heartbeat,
    /// `smallaios/admin/config/get`.
    ConfigGet,
    /// `smallaios/admin/config/set`.
    ConfigSet,
    /// `smallaios/admin/config/changed` — subscribe-only, never
    /// dispatched as a request.
    ConfigChanged,
    /// `smallaios/admin/model/add` — Phase 4 overlay model upload.
    ModelAdd,
    /// `smallaios/admin/model/remove` — Phase 4 overlay model
    /// remove/hide/unhide.
    ModelRemove,
    /// `smallaios/admin/model/list` — Phase 4 overlay model snapshot.
    ModelList,
}

impl Verb {
    /// Canonical key-expression suffix (the bit after `smallaios/admin/`).
    pub const fn keyspace_suffix(self) -> &'static str {
        match self {
            Self::Login => "login",
            Self::Logout => "logout",
            Self::Whoami => "whoami",
            Self::Passwd => "passwd",
            Self::UsersAdd => "users/add",
            Self::UsersList => "users/list",
            Self::Heartbeat => "heartbeat",
            Self::ConfigGet => "config/get",
            Self::ConfigSet => "config/set",
            Self::ConfigChanged => "config/changed",
            Self::ModelAdd => "model/add",
            Self::ModelRemove => "model/remove",
            Self::ModelList => "model/list",
        }
    }

    /// Parse a `smallaios/admin/<verb>` keyspace into a [`Verb`].
    pub fn from_keyspace(keyspace: &str) -> Option<Self> {
        let suffix = keyspace.strip_prefix("smallaios/admin/")?;
        match suffix {
            "login" => Some(Self::Login),
            "logout" => Some(Self::Logout),
            "whoami" => Some(Self::Whoami),
            "passwd" => Some(Self::Passwd),
            "users/add" => Some(Self::UsersAdd),
            "users/list" => Some(Self::UsersList),
            "heartbeat" => Some(Self::Heartbeat),
            "config/get" => Some(Self::ConfigGet),
            "config/set" => Some(Self::ConfigSet),
            "config/changed" => Some(Self::ConfigChanged),
            "model/add" => Some(Self::ModelAdd),
            "model/remove" => Some(Self::ModelRemove),
            "model/list" => Some(Self::ModelList),
            _ => None,
        }
    }

    /// Minimum role admitted by this verb. Returns `None` for verbs
    /// that bypass the auth wrapper (only `login`).
    pub const fn min_role(self) -> Option<MinRoleGate> {
        match self {
            Self::Login => None,
            Self::Logout => Some(MinRoleGate::Authenticated),
            Self::Whoami => Some(MinRoleGate::Authenticated),
            Self::Passwd => Some(MinRoleGate::Authenticated),
            Self::Heartbeat => Some(MinRoleGate::Authenticated),
            Self::UsersAdd => Some(MinRoleGate::Root),
            Self::UsersList => Some(MinRoleGate::Operator),
            Self::ConfigGet => Some(MinRoleGate::Operator),
            Self::ConfigSet => Some(MinRoleGate::Root),
            Self::ConfigChanged => Some(MinRoleGate::Authenticated),
            // Phase 4 overlay verbs: the dispatcher gate is Operator+;
            // the kernel-side syscall handlers re-check the per-mode
            // RBAC matrix (Root for delete/hide; Operator-iff-policy
            // for unhide).
            Self::ModelAdd => Some(MinRoleGate::Operator),
            Self::ModelRemove => Some(MinRoleGate::Operator),
            Self::ModelList => Some(MinRoleGate::Operator),
        }
    }
}

/// Local mirror of [`smallaios_kernel::auth::min_role::MinRole`].
/// Re-exposed here so wrapper code does not have to thread the kernel
/// type through every signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MinRoleGate {
    /// Caller must hold a session of any role.
    Authenticated,
    /// Caller must be `Operator` or `Root`.
    Operator,
    /// Caller must be `Root`.
    Root,
}

impl MinRoleGate {
    /// `true` iff the resolved role admits this gate.
    pub const fn admits(self, role: Role) -> bool {
        match (self, role) {
            (Self::Authenticated, _) => true,
            (Self::Operator, Role::Root) => true,
            (Self::Operator, Role::Operator) => true,
            (Self::Operator, Role::Viewer) => false,
            (Self::Root, Role::Root) => true,
            (Self::Root, _) => false,
        }
    }

    /// Static name used in audit `DENY:<verb>` records.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Authenticated => "authenticated",
            Self::Operator => "operator",
            Self::Root => "root",
        }
    }
}

/// Trait wrapping the kernel-side `auth_login`/`auth_logout`/etc.
/// handlers. The dispatcher calls into this trait so tests can plug
/// a simple mock without reaching into the kernel's `AuthCtx`.
pub trait LoginBackend {
    /// Verify `(user, pass)`, acquire a kernel session, and return the
    /// kernel-side `Session` snapshot. The wrapper then stamps
    /// `peer_identity` and handles the side-table.
    ///
    /// Returns a positive `(session, role)` tuple on success, or a
    /// negative kernel errno on failure.
    fn verify_credentials(&mut self, user: &str, password: &str) -> Result<Session, i64>;

    /// Forced logout — release the session in any auth-side state
    /// the backend tracks. The Zenoh side table releases the kernel
    /// slot itself.
    fn on_logout(&mut self, session: &Session);
}

/// Read-only / read-write view over the typed `Config`. Carried so the
/// wrapper can implement `config/get` and `config/set`.
pub trait ConfigView {
    /// Lookup `path` and return a serialisable value.
    fn read(&self, path: &str) -> Result<JsonValue, i64>;
    /// Apply `value` at `path`. Validation, on-disk write, and notify
    /// publication SHALL all complete before returning `Ok(())`.
    fn write(&mut self, path: &str, value: &JsonValue, who: &str) -> Result<(), i64>;
}

/// Stable response shape produced by every dispatcher call.
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    /// JSON object body to send back over the Zenoh queryable.
    pub body: String,
    /// Convenience: the negative errno extracted from `body` for
    /// failure responses, or `0` for success. Callers feed this to
    /// `audit::record(...)` without re-parsing the JSON.
    pub code: i64,
}

impl Response {
    /// Build a `{ "ok": true, "payload": <payload> }` response.
    pub fn ok(payload: JsonValue) -> Self {
        let mut m = BTreeMap::new();
        m.insert("ok".to_string(), JsonValue::Bool(true));
        m.insert("payload".to_string(), payload);
        Self {
            body: encode(&JsonValue::Object(m)),
            code: 0,
        }
    }

    /// Build a `{ "ok": false, "code": <n>, "reason": <s> }` response.
    pub fn err(code: i64, reason: &str) -> Self {
        let mut m = BTreeMap::new();
        m.insert("ok".to_string(), JsonValue::Bool(false));
        m.insert("code".to_string(), JsonValue::Int(code));
        m.insert("reason".to_string(), JsonValue::string(reason));
        Self {
            body: encode(&JsonValue::Object(m)),
            code,
        }
    }

    /// Helper: error response with the canonical message from
    /// [`error_string`].
    pub fn err_default(code: i64) -> Self {
        Self::err(code, error_string(code))
    }
}

/// Successful authentication context — what the wrapper stuffs into
/// every dispatched verb handler.
#[derive(Debug, Clone, Copy)]
pub struct AuthedRequest {
    /// The kernel session id resolved from the bearer token.
    pub session_id: smallaios_kernel::auth::session_table::SessionId,
    /// The role resolved at session-table lookup time.
    pub role: Role,
    /// Echo of the bearer token so the dispatcher can hop back into
    /// [`ZenohSessionTable`] for in-flight refcount management.
    pub token: Token,
    /// Echo of the peer identity for audit traces.
    pub peer: PeerIdentity,
}

/// Phase-7 dispatcher: the glue between a Zenoh queryable callback
/// and the typed verb handlers.
///
/// The dispatcher does not hold a reference to a Zenoh session — it
/// is transport-agnostic. The (future) `ZenohSurface::serve` Phase-8
/// glue will instantiate one of these and call
/// [`Self::dispatch`] for each incoming queryable request.
pub struct AdminDispatcher<L, C> {
    /// Side table.
    pub sessions: ZenohSessionTable,
    /// Login backend (kernel auth).
    pub login_backend: L,
    /// Config view for `config/get` + `config/set`.
    pub config: C,
    /// Salt source — a `RandFill` callback that fills 16 bytes for
    /// each new opaque token.
    pub rand: RandFill,
}

impl<L, C> AdminDispatcher<L, C>
where
    L: LoginBackend,
    C: ConfigView,
{
    /// Construct a dispatcher.
    pub fn new(sessions: ZenohSessionTable, login_backend: L, config: C, rand: RandFill) -> Self {
        Self {
            sessions,
            login_backend,
            config,
            rand,
        }
    }

    /// Dispatch a queryable request. `keyspace` SHALL be the full
    /// `smallaios/admin/<verb>` key expression. `body` SHALL be the
    /// JSON request bytes.
    ///
    /// `kernel_table` is borrowed for the duration of the call so the
    /// wrapper can manipulate the kernel session table.
    pub fn dispatch(
        &mut self,
        keyspace: &str,
        body: &[u8],
        peer: PeerIdentity,
        now_unix: u64,
        kernel_table: &mut SessionTable,
    ) -> Response {
        if body.len() > MAX_REQUEST_BYTES {
            return Response::err_default(ERRNO_EINVAL);
        }
        // Cheap path: house-keep stale side-table entries on every
        // dispatch. Keeps the per-identity counter from skewing in
        // the face of idle-sweeper releases.
        self.sessions.reap(kernel_table);

        let Some(verb) = Verb::from_keyspace(keyspace) else {
            return Response::err(ERRNO_ENOENT, "Unknown verb");
        };

        // Decode JSON only after verb resolution so that random Zenoh
        // probes against unknown keys do not pay a JSON parse cost.
        let req = match decode(body) {
            Ok(v) => v,
            Err(_) => return Response::err(ERRNO_EINVAL, "Malformed JSON"),
        };

        // `login` is the only verb that bypasses the wrapper.
        if verb == Verb::Login {
            return self.handle_login(&req, peer, now_unix, kernel_table);
        }

        // Step 1-5: extract token, lookup, verify peer, idle check, reset clock.
        let Some(token_str) = req.get("token").and_then(|v| v.as_str()) else {
            return Response::err(ERRNO_EAUTH, "Missing token");
        };
        if token_str.is_empty() {
            return Response::err(ERRNO_EAUTH, "Missing token");
        }
        let token = match super::session::token_from_hex(token_str.as_bytes()) {
            Some(t) => t,
            None => return Response::err(ERRNO_EAUTH, "Malformed token"),
        };
        let (session_id, role) = match self.sessions.resolve(kernel_table, &token, &peer, now_unix)
        {
            Ok(v) => v,
            Err(super::session::SideTableError::UnknownToken) => {
                return Response::err(ERRNO_EAUTH, "Invalid token")
            }
            Err(super::session::SideTableError::PeerMismatch) => {
                return Response::err(ERRNO_EPERM, "Peer mismatch")
            }
            Err(super::session::SideTableError::IdleExpired) => {
                return Response::err(ERRNO_EAUTHEXPIRED, "Session expired")
            }
            Err(super::session::SideTableError::KernelStale) => {
                return Response::err(ERRNO_EAUTHEXPIRED, "Session expired")
            }
            Err(_) => return Response::err(ERRNO_EACCES, "Authentication failed"),
        };

        // Step 6: capability check.
        if let Some(gate) = verb.min_role() {
            if !gate.admits(role) {
                return Response::err(ERRNO_EPERM, "Role insufficient for this verb");
            }
        }

        // Pre-build the authed-request snapshot.
        let authed = AuthedRequest {
            session_id,
            role,
            token,
            peer,
        };

        // In-flight live flag — handlers MAY take longer than the idle
        // window without expiring. The increment SHALL happen here so
        // every dispatched call is protected.
        self.sessions.enter_call(&token);
        let resp = match verb {
            Verb::Login => unreachable!("login bypass handled above"),
            Verb::Logout => self.handle_logout(&authed, kernel_table),
            Verb::Whoami => self.handle_whoami(&authed, kernel_table, now_unix),
            Verb::Heartbeat => self.handle_heartbeat(&authed, kernel_table, now_unix),
            Verb::Passwd => self.handle_passwd(&authed, &req),
            Verb::UsersAdd => self.handle_users_add(&authed, &req),
            Verb::UsersList => self.handle_users_list(&authed),
            Verb::ConfigGet => self.handle_config_get(&authed, &req),
            Verb::ConfigSet => self.handle_config_set(&authed, &req, now_unix),
            Verb::ConfigChanged => Response::err(
                ERRNO_EINVAL,
                "config/changed is subscribe-only, not request/response",
            ),
            // Phase 4 overlay verbs are routed via the separate
            // `model_handlers::dispatch` entry point so the Phase 7
            // dispatcher remains unchanged in behavior. A direct call
            // through the Phase 7 dispatcher reports ENOSYS to make
            // misuse loud — the wire path is the
            // `OverlayAdminDispatcher::dispatch` wrapper.
            Verb::ModelAdd | Verb::ModelRemove | Verb::ModelList => Response::err(
                smallaios_kernel::auth::ERRNO_ENOSYS,
                "Overlay model verbs route via OverlayAdminDispatcher; \
                 the Phase 7 AdminDispatcher does not handle them directly",
            ),
        };
        self.sessions.leave_call(&token);
        resp
    }

    // ─── Verb handlers ──────────────────────────────────────────────────────

    fn handle_login(
        &mut self,
        req: &JsonValue,
        peer: PeerIdentity,
        now_unix: u64,
        kernel_table: &mut SessionTable,
    ) -> Response {
        let args = match req.get("args") {
            Some(JsonValue::Object(_)) => req.get("args").unwrap(),
            _ => return Response::err(ERRNO_EINVAL, "Missing args"),
        };
        let user = match args.get("user").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s,
            _ => return Response::err(ERRNO_EINVAL, "Missing user"),
        };
        let pass = match args.get("pass").and_then(|v| v.as_str()) {
            Some(s) if !s.is_empty() => s,
            _ => return Response::err(ERRNO_EINVAL, "Missing pass"),
        };
        let session = match self.login_backend.verify_credentials(user, pass) {
            Ok(s) => s,
            Err(_) => {
                // The kernel-side `verify_credentials` runs the
                // dummy-PHC verify on user-not-found, so the wall
                // clock is indistinguishable. Always emit `EACCES`.
                return Response::err(ERRNO_EACCES, "Authentication failed");
            }
        };
        // Generate the opaque token before reserving a side-table slot
        // so a CSPRNG failure does not leave a half-allocated entry.
        let token = match generate_opaque(self.rand) {
            Ok(t) => t,
            Err(_) => return Response::err(ERRNO_EAGAIN, "Token generation failed"),
        };
        let admit = self
            .sessions
            .try_admit_login(kernel_table, session, peer, token, now_unix);
        let (token, _id) = match admit {
            Ok(pair) => pair,
            Err(super::session::SideTableError::PerIdentityCapReached) => {
                return Response::err(ERRNO_EAGAIN, "Per-identity session cap reached");
            }
            Err(super::session::SideTableError::TotalCapReached) => {
                return Response::err(ERRNO_EAGAIN, "Total session cap reached");
            }
            Err(super::session::SideTableError::KernelTableFull) => {
                return Response::err(ERRNO_ENOSPC, "Session table full");
            }
            Err(_) => return Response::err(ERRNO_EACCES, "Authentication failed"),
        };
        let role_str = role_to_str(session.role);
        let expires_in = self.sessions.config().idle.for_role(session.role);
        let mut payload = BTreeMap::new();
        let hex = token_hex(&token);
        payload.insert(
            "token".to_string(),
            JsonValue::string(core::str::from_utf8(&hex).unwrap()),
        );
        payload.insert("role".to_string(), JsonValue::string(role_str));
        payload.insert("expires_in".to_string(), JsonValue::Int(expires_in as i64));
        Response::ok(JsonValue::Object(payload))
    }

    fn handle_logout(
        &mut self,
        authed: &AuthedRequest,
        kernel_table: &mut SessionTable,
    ) -> Response {
        // Snapshot the session for the backend before releasing.
        let snapshot = match kernel_table.lookup(authed.session_id) {
            Ok(s) => *s,
            Err(_) => return Response::err(ERRNO_EAUTHEXPIRED, "Session expired"),
        };
        self.login_backend.on_logout(&snapshot);
        // The side-table releases the kernel slot.
        let _ = self.sessions.logout(kernel_table, &authed.token);
        Response::ok(JsonValue::empty_object())
    }

    fn handle_whoami(
        &self,
        authed: &AuthedRequest,
        kernel_table: &SessionTable,
        now_unix: u64,
    ) -> Response {
        let session = match kernel_table.lookup(authed.session_id) {
            Ok(s) => s,
            Err(_) => return Response::err(ERRNO_EAUTHEXPIRED, "Session expired"),
        };
        let mut payload = BTreeMap::new();
        let username = core::str::from_utf8(session.username_bytes()).unwrap_or("");
        payload.insert("user".to_string(), JsonValue::string(username));
        payload.insert("uid".to_string(), JsonValue::from(session.user_id));
        payload.insert(
            "role".to_string(),
            JsonValue::string(role_to_str(session.role)),
        );
        payload.insert(
            "login_unix_time".to_string(),
            JsonValue::from(session.login_unix_time),
        );
        let elapsed = now_unix.saturating_sub(session.last_activity_unix_time);
        payload.insert("idle_seconds".to_string(), JsonValue::from(elapsed));
        Response::ok(JsonValue::Object(payload))
    }

    fn handle_heartbeat(
        &self,
        authed: &AuthedRequest,
        kernel_table: &SessionTable,
        now_unix: u64,
    ) -> Response {
        // The wrapper's `resolve()` already reset the idle clock. We
        // just report the remaining lifetime back to the caller so
        // long-running UI progress bars know when to refresh next.
        let session = match kernel_table.lookup(authed.session_id) {
            Ok(s) => s,
            Err(_) => return Response::err(ERRNO_EAUTHEXPIRED, "Session expired"),
        };
        let threshold = self.sessions.config().idle.for_role(session.role);
        let elapsed = now_unix.saturating_sub(session.last_activity_unix_time);
        let remaining = threshold.saturating_sub(elapsed);
        let mut payload = BTreeMap::new();
        payload.insert("expires_in".to_string(), JsonValue::Int(remaining as i64));
        Response::ok(JsonValue::Object(payload))
    }

    fn handle_passwd(&mut self, _authed: &AuthedRequest, _req: &JsonValue) -> Response {
        // Phase 7 wires only the wrapper. `passwd` itself routes
        // through the kernel's `auth_change_password` syscall handler,
        // which is owned by Phase 3 and not extended here. We return
        // a structured "not yet wired" so the wire shape exists.
        Response::err(
            smallaios_kernel::auth::ERRNO_ENOSYS,
            "passwd verb wires through auth_change_password syscall (call directly via syscall ABI)",
        )
    }

    fn handle_users_add(&mut self, _authed: &AuthedRequest, _req: &JsonValue) -> Response {
        Response::err(
            smallaios_kernel::auth::ERRNO_ENOSYS,
            "users/add verb wires through auth_create_user syscall",
        )
    }

    fn handle_users_list(&self, _authed: &AuthedRequest) -> Response {
        // Phase 7 reports an empty list — the shadow loader exposes
        // a `list_users` cursor in Phase 8. The shape is stable now.
        let mut payload = BTreeMap::new();
        payload.insert("users".to_string(), JsonValue::Array(Vec::new()));
        Response::ok(JsonValue::Object(payload))
    }

    fn handle_config_get(&self, _authed: &AuthedRequest, req: &JsonValue) -> Response {
        let path = match req
            .get("args")
            .and_then(|a| a.get("path"))
            .and_then(|v| v.as_str())
        {
            Some(s) => s,
            None => return Response::err(ERRNO_EINVAL, "Missing path"),
        };
        match self.config.read(path) {
            Ok(value) => {
                let mut payload = BTreeMap::new();
                payload.insert("path".to_string(), JsonValue::string(path));
                payload.insert("value".to_string(), value);
                Response::ok(JsonValue::Object(payload))
            }
            Err(code) => Response::err_default(code),
        }
    }

    fn handle_config_set(
        &mut self,
        authed: &AuthedRequest,
        req: &JsonValue,
        _now_unix: u64,
    ) -> Response {
        let args = match req.get("args") {
            Some(v @ JsonValue::Object(_)) => v,
            _ => return Response::err(ERRNO_EINVAL, "Missing args"),
        };
        let path = match args.get("path").and_then(|v| v.as_str()) {
            Some(s) => s,
            None => return Response::err(ERRNO_EINVAL, "Missing path"),
        };
        let value = match args.get("value") {
            Some(v) => v.clone(),
            None => return Response::err(ERRNO_EINVAL, "Missing value"),
        };
        let who = audit_who(authed.role);
        match self.config.write(path, &value, &who) {
            Ok(()) => Response::ok(JsonValue::empty_object()),
            Err(code) => Response::err_default(code),
        }
    }
}

/// Render a kernel `Role` as the canonical lower-case wire string.
pub const fn role_to_str(role: Role) -> &'static str {
    match role {
        Role::Root => "root",
        Role::Operator => "operator",
        Role::Viewer => "viewer",
    }
}

/// Build the `who` string passed into `Config::write` and the audit
/// ring. Format: `"<role>@zenoh"` — matches the [`crate::surface::AuditIdentity`]
/// `render` for a role-only Zenoh session.
fn audit_who(role: Role) -> String {
    let mut s = String::new();
    s.push_str(role_to_str(role));
    s.push_str("@zenoh");
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface_zenoh::session::CapsConfig;

    /// Mock LoginBackend whose verify outcome is dictated per call.
    struct MockBackend {
        seeded: BTreeMap<String, (String, Role, u32)>, // user → (pass, role, uid)
        verify_delay: core::cell::RefCell<usize>,
        logout_count: core::cell::RefCell<usize>,
    }

    impl MockBackend {
        fn new() -> Self {
            Self {
                seeded: BTreeMap::new(),
                verify_delay: core::cell::RefCell::new(0),
                logout_count: core::cell::RefCell::new(0),
            }
        }
        fn seed(&mut self, user: &str, pass: &str, role: Role, uid: u32) {
            self.seeded
                .insert(user.to_string(), (pass.to_string(), role, uid));
        }
    }

    impl LoginBackend for MockBackend {
        fn verify_credentials(&mut self, user: &str, password: &str) -> Result<Session, i64> {
            // Record verify attempts (used for the constant-time test
            // assertion that we *always* call into the full path even
            // for unknown users).
            *self.verify_delay.borrow_mut() += 1;
            let entry = self.seeded.get(user);
            match entry {
                Some((pass, role, uid)) if pass == password => {
                    Ok(Session::new(*uid, user.as_bytes(), *role, 0, false).unwrap())
                }
                _ => Err(ERRNO_EACCES),
            }
        }
        fn on_logout(&mut self, _session: &Session) {
            *self.logout_count.borrow_mut() += 1;
        }
    }

    /// In-memory config view backed by a BTreeMap.
    struct MockConfig {
        store: BTreeMap<String, JsonValue>,
        readonly_paths: alloc::collections::BTreeSet<String>,
    }
    impl MockConfig {
        fn new() -> Self {
            let mut store = BTreeMap::new();
            store.insert("idle.root_minutes".to_string(), JsonValue::Int(15));
            Self {
                store,
                readonly_paths: alloc::collections::BTreeSet::new(),
            }
        }
    }
    impl ConfigView for MockConfig {
        fn read(&self, path: &str) -> Result<JsonValue, i64> {
            self.store.get(path).cloned().ok_or(ERRNO_ENOENT)
        }
        fn write(&mut self, path: &str, value: &JsonValue, _who: &str) -> Result<(), i64> {
            if self.readonly_paths.contains(path) {
                return Err(ERRNO_EPERM);
            }
            self.store.insert(path.to_string(), value.clone());
            Ok(())
        }
    }

    fn rand_fill(out: &mut [u8]) -> Result<(), ()> {
        // Simple counter-based test rand — every call increments so
        // tokens are distinct.
        use core::sync::atomic::{AtomicU64, Ordering};
        static N: AtomicU64 = AtomicU64::new(1);
        let mut s = N.fetch_add(1, Ordering::Relaxed);
        for slot in out.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *slot = ((s >> 33) & 0xFF) as u8;
        }
        Ok(())
    }

    fn fresh() -> (AdminDispatcher<MockBackend, MockConfig>, SessionTable) {
        let mut backend = MockBackend::new();
        backend.seed("root", "hunter2", Role::Root, 0);
        backend.seed("op", "operate1", Role::Operator, 1);
        backend.seed("viewer", "viewall1", Role::Viewer, 2);
        let dispatcher = AdminDispatcher::new(
            ZenohSessionTable::new(CapsConfig::defaults()),
            backend,
            MockConfig::new(),
            rand_fill,
        );
        (dispatcher, SessionTable::new())
    }

    fn login_body(user: &str, pass: &str) -> Vec<u8> {
        let mut args = BTreeMap::new();
        args.insert("user".to_string(), JsonValue::string(user));
        args.insert("pass".to_string(), JsonValue::string(pass));
        let mut req = BTreeMap::new();
        req.insert("args".to_string(), JsonValue::Object(args));
        encode(&JsonValue::Object(req)).into_bytes()
    }

    fn auth_body(token_hex_bytes: &[u8; 32], extra: BTreeMap<String, JsonValue>) -> Vec<u8> {
        let mut req = BTreeMap::new();
        req.insert(
            "token".to_string(),
            JsonValue::string(core::str::from_utf8(token_hex_bytes).unwrap()),
        );
        if !extra.is_empty() {
            req.insert("args".to_string(), JsonValue::Object(extra));
        }
        encode(&JsonValue::Object(req)).into_bytes()
    }

    fn extract_token(resp: &Response) -> [u8; 32] {
        let v = decode(resp.body.as_bytes()).unwrap();
        let s = v
            .get("payload")
            .and_then(|p| p.get("token"))
            .and_then(|t| t.as_str())
            .unwrap();
        let mut out = [0u8; 32];
        out.copy_from_slice(s.as_bytes());
        out
    }

    #[test]
    fn login_happy_path_returns_token() {
        let (mut d, mut k) = fresh();
        let resp = d.dispatch(
            "smallaios/admin/login",
            &login_body("root", "hunter2"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        assert_eq!(resp.code, 0);
        let v = decode(resp.body.as_bytes()).unwrap();
        assert_eq!(v.get("ok").and_then(|b| b.as_bool()), Some(true));
        let p = v.get("payload").unwrap();
        assert_eq!(p.get("role").and_then(|s| s.as_str()), Some("root"));
        // 15-minute Root window = 900s.
        assert_eq!(p.get("expires_in").and_then(|n| n.as_int()), Some(900));
        let tok = p.get("token").and_then(|s| s.as_str()).unwrap();
        assert_eq!(tok.len(), 32);
    }

    #[test]
    fn login_wrong_password_eaccess() {
        let (mut d, mut k) = fresh();
        let resp = d.dispatch(
            "smallaios/admin/login",
            &login_body("root", "wrong"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        assert_eq!(resp.code, ERRNO_EACCES);
        let v = decode(resp.body.as_bytes()).unwrap();
        assert_eq!(
            v.get("reason").and_then(|s| s.as_str()),
            Some("Authentication failed")
        );
    }

    #[test]
    fn login_unknown_user_indistinguishable_from_wrong_password() {
        let (mut d, mut k) = fresh();
        // Both go through the same backend code-path; the response is
        // bit-for-bit identical for a real-but-wrong-pass and an
        // unknown-user attempt.
        let r1 = d.dispatch(
            "smallaios/admin/login",
            &login_body("root", "wrong"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let r2 = d.dispatch(
            "smallaios/admin/login",
            &login_body("phantom", "anything"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        assert_eq!(r1.body, r2.body);
    }

    #[test]
    fn missing_token_eauth() {
        let (mut d, mut k) = fresh();
        // Don't include a token — wrapper SHALL reject before the
        // backend is touched.
        let body = encode(&JsonValue::empty_object()).into_bytes();
        let resp = d.dispatch("smallaios/admin/whoami", &body, [0xAA; 32], 1_000, &mut k);
        assert_eq!(resp.code, ERRNO_EAUTH);
    }

    #[test]
    fn empty_token_eauth() {
        let (mut d, mut k) = fresh();
        let mut req = BTreeMap::new();
        req.insert("token".to_string(), JsonValue::string(""));
        let body = encode(&JsonValue::Object(req)).into_bytes();
        let resp = d.dispatch("smallaios/admin/whoami", &body, [0xAA; 32], 1_000, &mut k);
        assert_eq!(resp.code, ERRNO_EAUTH);
    }

    #[test]
    fn malformed_token_eauth() {
        let (mut d, mut k) = fresh();
        let mut req = BTreeMap::new();
        req.insert("token".to_string(), JsonValue::string("not-hex"));
        let body = encode(&JsonValue::Object(req)).into_bytes();
        let resp = d.dispatch("smallaios/admin/whoami", &body, [0xAA; 32], 1_000, &mut k);
        assert_eq!(resp.code, ERRNO_EAUTH);
    }

    #[test]
    fn whoami_after_login_returns_role_and_user() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("op", "operate1"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        let body = auth_body(&token, BTreeMap::new());
        let resp = d.dispatch("smallaios/admin/whoami", &body, [0xAA; 32], 1_000, &mut k);
        assert_eq!(resp.code, 0);
        let v = decode(resp.body.as_bytes()).unwrap();
        let p = v.get("payload").unwrap();
        assert_eq!(p.get("role").and_then(|s| s.as_str()), Some("operator"));
        assert_eq!(p.get("user").and_then(|s| s.as_str()), Some("op"));
    }

    #[test]
    fn cross_peer_replay_eperm() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("op", "operate1"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        // Peer B replays peer A's token → wrapper rejects.
        let body = auth_body(&token, BTreeMap::new());
        let resp = d.dispatch("smallaios/admin/whoami", &body, [0xBB; 32], 1_000, &mut k);
        assert_eq!(resp.code, ERRNO_EPERM);
        let v = decode(resp.body.as_bytes()).unwrap();
        assert_eq!(
            v.get("reason").and_then(|s| s.as_str()),
            Some("Peer mismatch")
        );
    }

    #[test]
    fn role_gate_rejects_viewer_on_root_only_verb() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("viewer", "viewall1"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        // `users/add` is Root-only.
        let body = auth_body(&token, BTreeMap::new());
        let resp = d.dispatch(
            "smallaios/admin/users/add",
            &body,
            [0xAA; 32],
            1_000,
            &mut k,
        );
        assert_eq!(resp.code, ERRNO_EPERM);
    }

    #[test]
    fn role_gate_rejects_operator_on_root_only_verb() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("op", "operate1"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        // `config/set` is Root-only.
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), JsonValue::string("idle.root_minutes"));
        args.insert("value".to_string(), JsonValue::Int(20));
        let body = auth_body(&token, args);
        let resp = d.dispatch(
            "smallaios/admin/config/set",
            &body,
            [0xAA; 32],
            1_000,
            &mut k,
        );
        assert_eq!(resp.code, ERRNO_EPERM);
    }

    #[test]
    fn role_gate_admits_operator_on_operator_verb() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("op", "operate1"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), JsonValue::string("idle.root_minutes"));
        let body = auth_body(&token, args);
        let resp = d.dispatch(
            "smallaios/admin/config/get",
            &body,
            [0xAA; 32],
            1_000,
            &mut k,
        );
        assert_eq!(resp.code, 0);
    }

    #[test]
    fn config_get_unknown_path_enoent() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("root", "hunter2"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), JsonValue::string("nope.unknown"));
        let body = auth_body(&token, args);
        let resp = d.dispatch(
            "smallaios/admin/config/get",
            &body,
            [0xAA; 32],
            1_000,
            &mut k,
        );
        assert_eq!(resp.code, ERRNO_ENOENT);
    }

    #[test]
    fn config_set_round_trip() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("root", "hunter2"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        // Write idle.root_minutes = 30
        let mut args = BTreeMap::new();
        args.insert("path".to_string(), JsonValue::string("idle.root_minutes"));
        args.insert("value".to_string(), JsonValue::Int(30));
        let body = auth_body(&token, args);
        let set_resp = d.dispatch(
            "smallaios/admin/config/set",
            &body,
            [0xAA; 32],
            1_000,
            &mut k,
        );
        assert_eq!(set_resp.code, 0);
        // Read back.
        let mut args2 = BTreeMap::new();
        args2.insert("path".to_string(), JsonValue::string("idle.root_minutes"));
        let body2 = auth_body(&token, args2);
        let get_resp = d.dispatch(
            "smallaios/admin/config/get",
            &body2,
            [0xAA; 32],
            1_000,
            &mut k,
        );
        assert_eq!(get_resp.code, 0);
        let v = decode(get_resp.body.as_bytes()).unwrap();
        assert_eq!(
            v.get("payload")
                .and_then(|p| p.get("value"))
                .and_then(|n| n.as_int()),
            Some(30)
        );
    }

    #[test]
    fn idle_expired_eauthexpired() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("root", "hunter2"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        let body = auth_body(&token, BTreeMap::new());
        // Far past the 900-second Root window.
        let resp = d.dispatch("smallaios/admin/whoami", &body, [0xAA; 32], 100_000, &mut k);
        assert_eq!(resp.code, ERRNO_EAUTHEXPIRED);
    }

    #[test]
    fn sliding_ttl_resets_idle_clock() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("root", "hunter2"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        // 14 minutes (840 s) into the 15-min window.
        let body = auth_body(&token, BTreeMap::new());
        let r1 = d.dispatch(
            "smallaios/admin/whoami",
            &body,
            [0xAA; 32],
            1_000 + 14 * 60,
            &mut k,
        );
        assert_eq!(r1.code, 0);
        // 14 more minutes — must succeed because the first call reset
        // the clock.
        let r2 = d.dispatch(
            "smallaios/admin/whoami",
            &body,
            [0xAA; 32],
            1_000 + 14 * 60 + 14 * 60,
            &mut k,
        );
        assert_eq!(r2.code, 0);
    }

    #[test]
    fn heartbeat_reports_remaining_lifetime() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("root", "hunter2"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        let body = auth_body(&token, BTreeMap::new());
        // 500 seconds elapsed.
        let resp = d.dispatch(
            "smallaios/admin/heartbeat",
            &body,
            [0xAA; 32],
            1_500,
            &mut k,
        );
        assert_eq!(resp.code, 0);
        let v = decode(resp.body.as_bytes()).unwrap();
        let remaining = v
            .get("payload")
            .and_then(|p| p.get("expires_in"))
            .and_then(|n| n.as_int())
            .unwrap();
        // Heartbeat resolved through the wrapper which RESET the
        // idle clock, so remaining = full 900s.
        assert_eq!(remaining, 900);
    }

    #[test]
    fn heartbeat_resets_idle_clock_without_other_work() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("op", "operate1"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        let body = auth_body(&token, BTreeMap::new());
        // Operator window = 3600 s. Heartbeat at 3500 s — must succeed
        // and reset.
        let r1 = d.dispatch(
            "smallaios/admin/heartbeat",
            &body,
            [0xAA; 32],
            1_000 + 3500,
            &mut k,
        );
        assert_eq!(r1.code, 0);
        // Another 3500 s — would have expired without the reset.
        let r2 = d.dispatch(
            "smallaios/admin/whoami",
            &body,
            [0xAA; 32],
            1_000 + 3500 + 3500,
            &mut k,
        );
        assert_eq!(r2.code, 0);
    }

    #[test]
    fn long_running_call_survives_idle_expiry_via_in_flight_flag() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("root", "hunter2"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token_hex_bytes = extract_token(&login);
        let token = crate::surface_zenoh::session::token_from_hex(&token_hex_bytes).unwrap();
        // Manually mark the call live (simulating a verb handler that
        // ran for hours; e.g., a model upload).
        d.sessions.enter_call(&token);
        // Even at 100k seconds (well past the 900 s Root window), the
        // resolve sees in_flight > 0 and bypasses the idle check.
        let body = auth_body(&token_hex_bytes, BTreeMap::new());
        let resp = d.dispatch("smallaios/admin/whoami", &body, [0xAA; 32], 100_000, &mut k);
        assert_eq!(resp.code, 0);
        // After we leave the call, the session goes stale on the next
        // request because the idle clock is also far past.
        d.sessions.leave_call(&token);
        // The previous whoami reset the clock; advance another 901s.
        let resp_late = d.dispatch(
            "smallaios/admin/whoami",
            &body,
            [0xAA; 32],
            100_000 + 901,
            &mut k,
        );
        assert_eq!(resp_late.code, ERRNO_EAUTHEXPIRED);
    }

    #[test]
    fn per_identity_cap_eagain_on_5th() {
        let (mut d, mut k) = fresh();
        // Populate 4 sessions for the same peer (different users so the
        // backend doesn't fold).
        for (i, u) in ["root", "op", "viewer", "root"].iter().enumerate() {
            let pass = match *u {
                "root" => "hunter2",
                "op" => "operate1",
                _ => "viewall1",
            };
            let resp = d.dispatch(
                "smallaios/admin/login",
                &login_body(u, pass),
                [0xCC; 32],
                1_000 + i as u64,
                &mut k,
            );
            assert_eq!(resp.code, 0);
        }
        // 5th attempt → EAGAIN with explicit reason.
        let resp = d.dispatch(
            "smallaios/admin/login",
            &login_body("root", "hunter2"),
            [0xCC; 32],
            1_010,
            &mut k,
        );
        assert_eq!(resp.code, ERRNO_EAGAIN);
        let v = decode(resp.body.as_bytes()).unwrap();
        assert_eq!(
            v.get("reason").and_then(|s| s.as_str()),
            Some("Per-identity session cap reached")
        );
    }

    #[test]
    fn total_cap_eagain_on_17th() {
        let (mut d, mut k) = fresh();
        // 16 logins across 4 peers (4 each).
        for peer_byte in 0u8..4 {
            for i in 0..4 {
                let resp = d.dispatch(
                    "smallaios/admin/login",
                    &login_body("op", "operate1"),
                    [peer_byte * 0x11; 32],
                    1_000 + i as u64,
                    &mut k,
                );
                assert_eq!(resp.code, 0);
            }
        }
        // 17th — different peer, must still hit total cap.
        let resp = d.dispatch(
            "smallaios/admin/login",
            &login_body("op", "operate1"),
            [0xFF; 32],
            1_500,
            &mut k,
        );
        assert_eq!(resp.code, ERRNO_EAGAIN);
        let v = decode(resp.body.as_bytes()).unwrap();
        assert_eq!(
            v.get("reason").and_then(|s| s.as_str()),
            Some("Total session cap reached")
        );
    }

    #[test]
    fn logout_invalidates_token() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("root", "hunter2"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        let body = auth_body(&token, BTreeMap::new());
        let logout = d.dispatch("smallaios/admin/logout", &body, [0xAA; 32], 1_001, &mut k);
        assert_eq!(logout.code, 0);
        // Subsequent use of the same token rejects.
        let after = d.dispatch("smallaios/admin/whoami", &body, [0xAA; 32], 1_002, &mut k);
        assert_eq!(after.code, ERRNO_EAUTH);
    }

    #[test]
    fn unknown_verb_returns_enoent() {
        let (mut d, mut k) = fresh();
        let resp = d.dispatch("smallaios/admin/bogus", b"{}", [0xAA; 32], 1_000, &mut k);
        assert_eq!(resp.code, ERRNO_ENOENT);
    }

    #[test]
    fn malformed_json_einval() {
        let (mut d, mut k) = fresh();
        let resp = d.dispatch(
            "smallaios/admin/whoami",
            b"{ not json",
            [0xAA; 32],
            1_000,
            &mut k,
        );
        assert_eq!(resp.code, ERRNO_EINVAL);
    }

    #[test]
    fn oversized_body_einval() {
        let (mut d, mut k) = fresh();
        let big = alloc::vec![b'a'; MAX_REQUEST_BYTES + 1];
        let resp = d.dispatch("smallaios/admin/whoami", &big, [0xAA; 32], 1_000, &mut k);
        assert_eq!(resp.code, ERRNO_EINVAL);
    }

    #[test]
    fn config_changed_via_request_returns_einval() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("root", "hunter2"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        let body = auth_body(&token, BTreeMap::new());
        let resp = d.dispatch(
            "smallaios/admin/config/changed",
            &body,
            [0xAA; 32],
            1_000,
            &mut k,
        );
        assert_eq!(resp.code, ERRNO_EINVAL);
    }

    #[test]
    fn whoami_returns_idle_seconds() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("op", "operate1"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        let body = auth_body(&token, BTreeMap::new());
        // The whoami call's wrapper resets the idle clock first, so
        // the idle reading observed by the handler is exactly the
        // computed 0.
        let resp = d.dispatch("smallaios/admin/whoami", &body, [0xAA; 32], 1_500, &mut k);
        let v = decode(resp.body.as_bytes()).unwrap();
        let idle = v
            .get("payload")
            .and_then(|p| p.get("idle_seconds"))
            .and_then(|n| n.as_int())
            .unwrap();
        assert_eq!(idle, 0);
    }

    #[test]
    fn role_admit_matrix() {
        // Exhaustive admit table for MinRoleGate.
        for (gate, role, expected) in [
            (MinRoleGate::Authenticated, Role::Root, true),
            (MinRoleGate::Authenticated, Role::Operator, true),
            (MinRoleGate::Authenticated, Role::Viewer, true),
            (MinRoleGate::Operator, Role::Root, true),
            (MinRoleGate::Operator, Role::Operator, true),
            (MinRoleGate::Operator, Role::Viewer, false),
            (MinRoleGate::Root, Role::Root, true),
            (MinRoleGate::Root, Role::Operator, false),
            (MinRoleGate::Root, Role::Viewer, false),
        ] {
            assert_eq!(gate.admits(role), expected);
        }
    }

    #[test]
    fn verb_keyspace_round_trip() {
        for v in [
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
        ] {
            let mut full = String::from("smallaios/admin/");
            full.push_str(v.keyspace_suffix());
            assert_eq!(Verb::from_keyspace(&full), Some(v));
        }
    }

    #[test]
    fn login_creates_kernel_session_with_peer_binding() {
        let (mut d, mut k) = fresh();
        let resp = d.dispatch(
            "smallaios/admin/login",
            &login_body("root", "hunter2"),
            [0xCD; 32],
            1_000,
            &mut k,
        );
        assert_eq!(resp.code, 0);
        // Sanity: kernel session has peer_identity stamped.
        let mut found = false;
        for (_, s) in k.iter() {
            if s.peer_identity == Some([0xCD; 32]) {
                found = true;
            }
        }
        assert!(found);
    }

    #[test]
    fn audit_who_renders_role_at_zenoh() {
        assert_eq!(audit_who(Role::Root), "root@zenoh");
        assert_eq!(audit_who(Role::Operator), "operator@zenoh");
        assert_eq!(audit_who(Role::Viewer), "viewer@zenoh");
    }

    #[test]
    fn login_attempt_for_unknown_user_increments_verify_counter() {
        // Constant-time-equivalent: verify path runs even on miss.
        let (mut d, mut k) = fresh();
        let before = *d.login_backend.verify_delay.borrow();
        d.dispatch(
            "smallaios/admin/login",
            &login_body("phantom", "anything"),
            [0x12; 32],
            1_000,
            &mut k,
        );
        let after = *d.login_backend.verify_delay.borrow();
        assert_eq!(after - before, 1);
    }

    #[test]
    fn logout_calls_backend_on_logout() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("root", "hunter2"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        let body = auth_body(&token, BTreeMap::new());
        d.dispatch("smallaios/admin/logout", &body, [0xAA; 32], 1_001, &mut k);
        assert_eq!(*d.login_backend.logout_count.borrow(), 1);
    }

    #[test]
    fn heartbeat_resets_clock_for_long_running_dashboards() {
        let (mut d, mut k) = fresh();
        let login = d.dispatch(
            "smallaios/admin/login",
            &login_body("op", "operate1"),
            [0xAA; 32],
            1_000,
            &mut k,
        );
        let token = extract_token(&login);
        let body = auth_body(&token, BTreeMap::new());
        // Send heartbeats every 30 minutes for 3 hours — would
        // expire under a non-sliding TTL.
        let mut t = 1_000;
        for _ in 0..6 {
            t += 30 * 60;
            let resp = d.dispatch("smallaios/admin/heartbeat", &body, [0xAA; 32], t, &mut k);
            assert_eq!(resp.code, 0);
        }
    }
}
