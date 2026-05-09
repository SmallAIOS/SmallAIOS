// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Zenoh admin/telemetry surface.
//!
//! Spec:
//! - `openspec/changes/management-login-v1/specs/mgmt-zenoh-admin/spec.md`
//! - `openspec/changes/management-login-v1/specs/ipc-zenoh/spec.md`
//!
//! Phase 7 lands the admin keyspace, the bearer-token wrapper, the
//! per-identity / total session caps, and the in-flight live flag.
//! Phase 8 will land telemetry publications under
//! `smallaios/metrics/**`.
//!
//! ## Module map
//!
//! - [`json`] — clean-room `#![no_std]` JSON encoder/decoder for
//!   admin payloads.
//! - [`session`] — Zenoh-side session table (peer-identity binding,
//!   per-identity / total caps, in-flight live flag, opaque-token
//!   index).
//! - [`token`] — opaque + ML-DSA-65 / Ed25519-legacy token modes.
//! - [`admin`] — verb dispatcher and bearer-token wrapper.
//! - [`errors`] — POSIX errno table shared with the syscall surface.
//!
//! ## ZenohSurface
//!
//! [`ZenohSurface`] is the [`crate::ConfigSurface`] implementation
//! over Zenoh. It is constructed with a [`crate::Config`] view and a
//! [`notify::BroadcastTx`] (so writes can be re-published on
//! `smallaios/admin/config/changed`). Its `read`/`write`/`subscribe`
//! methods are the same shape as [`crate::loader_toml::TomlLoader`]'s
//! and run through the same apply lifecycle. The
//! [`AdminDispatcher`](admin::AdminDispatcher) wraps `ZenohSurface`
//! when serving the wire — direct callers bypass the wire and reach
//! the trait methods.

pub mod admin;
pub mod errors;
pub mod json;
pub mod session;
pub mod token;

#[cfg(test)]
mod zenoh_test_vectors;

use alloc::string::{String, ToString};
use core::cell::RefCell;

use crate::config::Config;
use crate::notify::{BroadcastTx, NotifyEvent, Subscriber};
use crate::surface::{
    AuditIdentity, ConfigPath, ConfigSurface, Error as SurfaceError, SurfaceKind, Value,
};

pub use admin::{
    AdminDispatcher, AuthedRequest, ConfigView, LoginBackend, MinRoleGate, Response, Verb,
    MAX_REQUEST_BYTES,
};
pub use errors::{error_string, ERRNO_EAUTH, ERRNO_EPERM};
pub use json::{decode as json_decode, encode as json_encode, JsonError, JsonValue};
pub use session::{
    token_from_hex, token_hex, CapsConfig, IdleThresholds, PeerIdentity, SideTableError, Token,
    ZenohSessionEntry, ZenohSessionTable,
};
pub use token::{classify, generate_opaque, SignedClaims, TokenError, TokenShape};

/// `ConfigSurface` impl backed by an in-memory [`Config`] plus a
/// [`BroadcastTx`] for the apply lifecycle's notify step.
///
/// `ZenohSurface` is **transport-agnostic** — it does not own a Zenoh
/// session. The (Phase 8) `serve()` glue creates a `ZenohSurface` and
/// then hooks it into Zenoh queryables / subscribers; tests construct
/// a `ZenohSurface` directly.
pub struct ZenohSurface {
    cached: Config,
    notify: RefCell<BroadcastTx>,
}

impl ZenohSurface {
    /// Construct a new surface with `initial` as the starting
    /// [`Config`]. The internal broadcast channel uses the default
    /// capacity. Callers who need a wider ring should call
    /// [`Self::with_notify`].
    pub fn new(initial: Config) -> Self {
        let (tx, _rx_initial) = crate::notify::channel();
        Self {
            cached: initial,
            notify: RefCell::new(tx),
        }
    }

    /// Construct with an explicit [`BroadcastTx`] — useful when the
    /// caller wants subscribers from Phase 7's `config/changed`
    /// queryable to share a ring with the [`crate::loader_toml::TomlLoader`].
    pub fn with_notify(initial: Config, notify: BroadcastTx) -> Self {
        Self {
            cached: initial,
            notify: RefCell::new(notify),
        }
    }

    /// Borrow the cached typed [`Config`].
    pub fn config(&self) -> &Config {
        &self.cached
    }

    /// Replace the cached config wholesale. Used by the boot path to
    /// install the [`crate::loader_toml::TomlLoader`]'s post-load
    /// snapshot. Does NOT publish a notify event — every individual
    /// path change SHALL go through [`Self::write`] for traceability.
    pub fn replace_config(&mut self, new_config: Config) {
        self.cached = new_config;
    }

    /// Subscribe to a path filter. Convenience over the trait's
    /// [`ConfigSurface::subscribe`].
    pub fn subscribe_path(&self, path: ConfigPath) -> Subscriber {
        self.notify.borrow().subscribe_path(path)
    }

    /// Helper used by tests + the in-flight `config/changed` queryable
    /// to publish synthetic events without going through `write`.
    pub fn publish_event(&self, event: NotifyEvent) {
        self.notify.borrow().publish(event);
    }

    /// Surface kind tag — always `SurfaceKind::Zenoh`.
    pub const fn kind(&self) -> SurfaceKind {
        SurfaceKind::Zenoh
    }
}

impl ConfigSurface for ZenohSurface {
    fn read(&self, path: &ConfigPath) -> Result<Value, SurfaceError> {
        self.cached.read_path(path)
    }

    fn write(&mut self, path: &ConfigPath, value: Value, who: &str) -> Result<(), SurfaceError> {
        let before = self.cached.read_path(path)?;
        self.cached.write_path(path, value.clone())?;
        // Validate AFTER the in-memory mutation so cross-field
        // invariants see the post-write state. On failure the cached
        // copy is rolled back to `before`.
        if let Err(e) = crate::validate::validate_all(&self.cached) {
            self.cached.write_path(path, before).ok();
            return Err(e);
        }
        let event = NotifyEvent {
            path: path.clone(),
            before,
            after: value,
            who: who.to_string(),
            when_ns: 0,
        };
        self.notify.borrow().publish(event);
        Ok(())
    }

    fn subscribe(&self, path: &ConfigPath) -> Result<Subscriber, SurfaceError> {
        Ok(self.notify.borrow().subscribe_path(path.clone()))
    }
}

/// Adapter that turns a `&mut ZenohSurface` into the
/// [`ConfigView`] the dispatcher consumes. Acts as a path translator
/// between the JSON wire form (`{"path": "...", "value": <json>}`)
/// and the typed [`ConfigPath`] / [`Value`] surface API.
pub struct ZenohConfigView<'a> {
    surface: &'a mut ZenohSurface,
}

impl<'a> ZenohConfigView<'a> {
    /// Construct an adapter borrowing `surface`.
    pub fn new(surface: &'a mut ZenohSurface) -> Self {
        Self { surface }
    }
}

impl<'a> ConfigView for ZenohConfigView<'a> {
    fn read(&self, path: &str) -> Result<JsonValue, i64> {
        let cp = ConfigPath::new(path).map_err(surface_error_to_errno)?;
        let v = self.surface.read(&cp).map_err(surface_error_to_errno)?;
        Ok(value_to_json(v))
    }

    fn write(&mut self, path: &str, value: &JsonValue, who: &str) -> Result<(), i64> {
        let cp = ConfigPath::new(path).map_err(surface_error_to_errno)?;
        // The surface's typed `write_path` accepts only the existing
        // `Value` variants. We coerce by reading the current value's
        // type tag and parsing the JSON value into the matching
        // variant.
        let current = self.surface.read(&cp).map_err(surface_error_to_errno)?;
        let new_value = json_to_typed(value, &current).ok_or(errors::ERRNO_EPERM)?;
        ConfigSurface::write(self.surface, &cp, new_value, who).map_err(surface_error_to_errno)
    }
}

/// Map a [`SurfaceError`] to a POSIX errno consistent with the rest
/// of the wrapper.
pub fn surface_error_to_errno(e: SurfaceError) -> i64 {
    match e {
        SurfaceError::NotFound => smallaios_kernel::auth::ERRNO_ENOENT,
        SurfaceError::TypeMismatch { .. } => smallaios_kernel::auth::ERRNO_EINVAL,
        SurfaceError::Validation(_) => smallaios_kernel::auth::ERRNO_EINVAL,
        SurfaceError::InvalidPath(_) => smallaios_kernel::auth::ERRNO_EINVAL,
        SurfaceError::Io(_) => smallaios_kernel::auth::ERRNO_ENOSPC,
        SurfaceError::Permission(_) => errors::ERRNO_EPERM,
        SurfaceError::Unsupported(_) => smallaios_kernel::auth::ERRNO_ENOSYS,
    }
}

/// Convert a typed [`Value`] into a wire [`JsonValue`].
pub fn value_to_json(v: Value) -> JsonValue {
    match v {
        Value::Bool(b) => JsonValue::Bool(b),
        Value::U32(n) => JsonValue::from(n),
        Value::U64(n) => JsonValue::from(n),
        Value::String(s) => JsonValue::String(s),
        Value::PasswordClasses(p) => JsonValue::from(u64::from(p.0)),
    }
}

/// Convert a wire [`JsonValue`] into a typed [`Value`] matching the
/// shape of `current`. Returns `None` when the JSON shape can't be
/// coerced — the caller maps that to `EPERM` so misbehaving clients
/// don't get pinpointed type information they can probe with.
pub fn json_to_typed(value: &JsonValue, current: &Value) -> Option<Value> {
    match current {
        Value::Bool(_) => value.as_bool().map(Value::Bool),
        Value::U32(_) => {
            let n = value.as_int()?;
            if !(0..=u32::MAX as i64).contains(&n) {
                return None;
            }
            Some(Value::U32(n as u32))
        }
        Value::U64(_) => {
            let n = value.as_int()?;
            if n < 0 {
                return None;
            }
            Some(Value::U64(n as u64))
        }
        Value::String(_) => value.as_str().map(|s| Value::String(s.to_string())),
        Value::PasswordClasses(_) => {
            let n = value.as_int()?;
            if !(0..=255).contains(&n) {
                return None;
            }
            Some(Value::PasswordClasses(
                crate::surface::PasswordClassSet::from_mask(n as u8),
            ))
        }
    }
}

/// Render the canonical Zenoh `who` string for an audit identity.
/// Mirrors [`AuditIdentity::render`] but with the surface fixed to
/// `Zenoh`, saving the caller the construction.
pub fn render_zenoh_who(role: &'static str, user: Option<&str>) -> String {
    let id = AuditIdentity {
        role,
        user: user.map(|s| s.to_string()),
        surface: SurfaceKind::Zenoh,
    };
    id.render()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn read_returns_default_value() {
        let surface = ZenohSurface::new(Config::v1_defaults());
        let path = ConfigPath::new("idle.root_minutes").unwrap();
        let v = surface.read(&path).unwrap();
        assert_eq!(v.as_u32().unwrap(), 15);
    }

    #[test]
    fn write_persists_and_re_reads() {
        let mut surface = ZenohSurface::new(Config::v1_defaults());
        let path = ConfigPath::new("idle.root_minutes").unwrap();
        ConfigSurface::write(&mut surface, &path, Value::U32(20), "root@zenoh").unwrap();
        assert_eq!(surface.read(&path).unwrap().as_u32().unwrap(), 20);
    }

    #[test]
    fn write_rejects_validation_failure_and_rolls_back() {
        let mut surface = ZenohSurface::new(Config::v1_defaults());
        let path = ConfigPath::new("idle.root_minutes").unwrap();
        // 0 is below the validator's minimum.
        let err = ConfigSurface::write(&mut surface, &path, Value::U32(0), "root@zenoh");
        assert!(err.is_err());
        // Rollback: still 15.
        assert_eq!(surface.read(&path).unwrap().as_u32().unwrap(), 15);
    }

    #[test]
    fn subscribe_observes_writes() {
        let mut surface = ZenohSurface::new(Config::v1_defaults());
        let path = ConfigPath::new("idle.root_minutes").unwrap();
        let mut sub = ConfigSurface::subscribe(&surface, &path).unwrap();
        ConfigSurface::write(&mut surface, &path, Value::U32(45), "root@zenoh").unwrap();
        let event = sub.poll().unwrap().unwrap();
        assert_eq!(event.path, path);
        assert_eq!(event.before.as_u32().unwrap(), 15);
        assert_eq!(event.after.as_u32().unwrap(), 45);
        assert_eq!(event.who, "root@zenoh");
    }

    #[test]
    fn config_view_round_trip() {
        let mut surface = ZenohSurface::new(Config::v1_defaults());
        let mut view = ZenohConfigView::new(&mut surface);
        let v = view.read("idle.root_minutes").unwrap();
        assert_eq!(v.as_int(), Some(15));
        view.write("idle.root_minutes", &JsonValue::Int(45), "root@zenoh")
            .unwrap();
        let v2 = view.read("idle.root_minutes").unwrap();
        assert_eq!(v2.as_int(), Some(45));
    }

    #[test]
    fn config_view_unknown_path_enoent() {
        let mut surface = ZenohSurface::new(Config::v1_defaults());
        let view = ZenohConfigView::new(&mut surface);
        let err = view.read("nope.gone").unwrap_err();
        assert_eq!(err, smallaios_kernel::auth::ERRNO_ENOENT);
    }

    #[test]
    fn config_view_writes_validate_and_roll_back() {
        let mut surface = ZenohSurface::new(Config::v1_defaults());
        {
            let mut view = ZenohConfigView::new(&mut surface);
            // 0 fails the validator.
            assert!(view
                .write("idle.root_minutes", &JsonValue::Int(0), "root@zenoh")
                .is_err());
        }
        // Sanity: cached value is unchanged.
        let path = ConfigPath::new("idle.root_minutes").unwrap();
        assert_eq!(surface.read(&path).unwrap().as_u32().unwrap(), 15);
    }

    #[test]
    fn json_to_typed_handles_each_variant() {
        // bool
        let cur = Value::Bool(true);
        assert_eq!(
            json_to_typed(&JsonValue::Bool(false), &cur),
            Some(Value::Bool(false))
        );
        // u32
        let cur = Value::U32(0);
        assert_eq!(
            json_to_typed(&JsonValue::Int(42), &cur),
            Some(Value::U32(42))
        );
        // u32 rejects negative
        assert_eq!(json_to_typed(&JsonValue::Int(-1), &cur), None);
        // u32 rejects overflow
        assert_eq!(
            json_to_typed(&JsonValue::Int(i64::from(u32::MAX) + 1), &cur),
            None
        );
        // string
        let cur = Value::String("x".to_string());
        assert_eq!(
            json_to_typed(&JsonValue::string("y"), &cur),
            Some(Value::String("y".to_string()))
        );
    }

    #[test]
    fn surface_kind_tag_is_zenoh() {
        let s = ZenohSurface::new(Config::v1_defaults());
        assert_eq!(s.kind(), SurfaceKind::Zenoh);
    }

    #[test]
    fn render_zenoh_who_renders_correctly() {
        assert_eq!(render_zenoh_who("root", None), "root@zenoh");
        assert_eq!(
            render_zenoh_who("operator", Some("alice")),
            "operator:alice@zenoh"
        );
    }

    #[test]
    fn replace_config_swaps_full_config() {
        let mut s = ZenohSurface::new(Config::v1_defaults());
        let mut alt = Config::v1_defaults();
        alt.idle.root_minutes = 99;
        s.replace_config(alt);
        let path = ConfigPath::new("idle.root_minutes").unwrap();
        assert_eq!(s.read(&path).unwrap().as_u32().unwrap(), 99);
    }

    #[test]
    fn surface_error_mapping_covers_every_variant() {
        assert_eq!(
            surface_error_to_errno(SurfaceError::NotFound),
            smallaios_kernel::auth::ERRNO_ENOENT
        );
        assert_eq!(
            surface_error_to_errno(SurfaceError::TypeMismatch {
                expected: "u32",
                got: "bool"
            }),
            smallaios_kernel::auth::ERRNO_EINVAL
        );
        assert_eq!(
            surface_error_to_errno(SurfaceError::Validation("x")),
            smallaios_kernel::auth::ERRNO_EINVAL
        );
        assert_eq!(
            surface_error_to_errno(SurfaceError::InvalidPath("x")),
            smallaios_kernel::auth::ERRNO_EINVAL
        );
        assert_eq!(
            surface_error_to_errno(SurfaceError::Io("x")),
            smallaios_kernel::auth::ERRNO_ENOSPC
        );
        assert_eq!(
            surface_error_to_errno(SurfaceError::Permission("x")),
            ERRNO_EPERM
        );
        assert_eq!(
            surface_error_to_errno(SurfaceError::Unsupported("x")),
            smallaios_kernel::auth::ERRNO_ENOSYS
        );
    }
}
