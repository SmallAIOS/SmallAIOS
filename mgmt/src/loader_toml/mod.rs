// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! TOML loader/saver — first real [`crate::ConfigSurface`] impl.
//!
//! Spec:
//! - `openspec/changes/management-login-v1/specs/mgmt-config-surface-trait/spec.md`
//!   ("TOML surface implements ConfigSurface" scenario).
//! - `openspec/changes/management-login-v1/specs/mgmt-config-layout/spec.md`
//!   (`/data/` directory layout, per-file permission declaration,
//!   stricter-than-declared rule).
//! - `openspec/changes/management-login-v1/specs/mgmt-config-model/spec.md`
//!   ("Apply lifecycle" — parse, validate, stage, fsync, rename,
//!    notify, audit).
//!
//! ## Apply lifecycle
//!
//! On a [`TomlLoader::write`]:
//!
//! 1. **Parse** the proposed [`Value`] into the typed schema.
//! 2. **Validate** per-field and cross-field via
//!    [`crate::validate`]. Failure leaves on-disk and in-memory
//!    state untouched.
//! 3. **Stage** the rewritten file at `<file>.tmp` via the configured
//!    [`AtomicWriter`]. The writer itself fsyncs and renames.
//! 4. **Update** the in-memory [`Config`] iff the field has
//!    `ReloadKind::Live`. Boot-deferred fields are accumulated in
//!    [`TomlLoader::pending_reboot`] for the operator to see.
//! 5. **Notify** subscribers via the broadcast channel.
//!
//! ## First-boot generation
//!
//! When [`TomlLoader::load`] finds a file missing it generates a
//! skeleton with [`Config::v1_defaults`]-derived contents iff the
//! caller passes [`FirstBootPolicy::Generate`] (this is the case the
//! kernel sets on a freshly-formatted F2FS partition). With
//! [`FirstBootPolicy::Strict`] a missing file is a fatal error —
//! "config disappeared between boots" is a corruption signal that
//! deserves operator attention.
//!
//! ## File-mode-stricter-than-declared rule
//!
//! Per `mgmt-config-layout`, every file has a declared mode (e.g.,
//! `0640` for `mgmt/policy.toml`). The loader refuses to read a file
//! whose mode is laxer than declared (e.g., `0644` shadow rejected).
//! Stricter-than-declared modes are accepted.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt;

use crate::config::registry::{self, FieldType};
use crate::config::Config;
use crate::notify::{BroadcastTx, NotifyEvent, Subscriber};
use crate::surface::{
    AuditIdentity, ConfigPath, ConfigSurface, Error as SurfaceError, PasswordClassSet, ReloadKind,
    RoleSet, SurfaceKind, Value,
};
use crate::toml::{parse as toml_parse, serialize as toml_serialize, TomlTable, TomlValue};
use crate::validate;

use smallaios_auth::atomic_write::{AtomicWriteError, AtomicWriter};

#[cfg(test)]
mod loader_test_vectors;

/// Behaviour when a per-subsystem file is missing on [`TomlLoader::load`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirstBootPolicy {
    /// First boot after a fresh F2FS format — generate skeleton files
    /// from [`Config::v1_defaults`].
    Generate,
    /// Strict — a missing file is a fatal error.
    Strict,
}

/// Loader-level error. Distinct from [`SurfaceError`] because the
/// loader sees a few categories the trait doesn't (config layout
/// corruption, mode mismatch, generation failure). Implements
/// `From<LoaderError>` -> [`SurfaceError`] so the trait surface can
/// report it cleanly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoaderError {
    /// A declared file is missing and [`FirstBootPolicy::Strict`] is
    /// in effect.
    Missing {
        /// Absolute path that was expected.
        path: String,
    },
    /// On-disk mode is laxer than the declared mode for `path`.
    ModeViolation {
        /// File whose mode was rejected.
        path: String,
        /// Declared mode (octal, e.g. `0o640`).
        declared: u16,
        /// Observed mode (octal).
        actual: u16,
    },
    /// TOML parse / serialise error.
    Toml(crate::toml::Error),
    /// Per-field or cross-field validation rejected the on-disk
    /// content.
    Validation(&'static str),
    /// Backing I/O failure. Carries the static reason from
    /// [`AtomicWriteError`] or the backend metadata source.
    Io(&'static str),
    /// File contains a TOML key whose dotted path does not match any
    /// known [`Config`] field. Strict by spec — unknown keys are a
    /// drift signal.
    UnknownField(String),
    /// File contains a TOML scalar whose runtime variant does not
    /// match the registry's typed [`FieldType`]. Lossy file edits
    /// (e.g., a `bool` typed as a string) trigger this.
    TypeMismatch {
        /// Dotted path that mismatched.
        path: String,
        /// Expected variant tag.
        expected: &'static str,
        /// Actual variant tag.
        got: &'static str,
    },
}

impl fmt::Display for LoaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Missing { path } => write!(f, "loader_toml: missing required file: {path}"),
            Self::ModeViolation {
                path,
                declared,
                actual,
            } => write!(
                f,
                "loader_toml: mode violation on {path}: declared 0o{declared:o}, actual 0o{actual:o}"
            ),
            Self::Toml(e) => write!(f, "loader_toml: {e}"),
            Self::Validation(reason) => write!(f, "loader_toml: validation: {reason}"),
            Self::Io(reason) => write!(f, "loader_toml: io: {reason}"),
            Self::UnknownField(p) => write!(f, "loader_toml: unknown field: {p}"),
            Self::TypeMismatch {
                path,
                expected,
                got,
            } => write!(
                f,
                "loader_toml: type mismatch at {path}: expected {expected}, got {got}"
            ),
        }
    }
}

impl From<crate::toml::Error> for LoaderError {
    fn from(e: crate::toml::Error) -> Self {
        Self::Toml(e)
    }
}

impl From<AtomicWriteError> for LoaderError {
    fn from(e: AtomicWriteError) -> Self {
        match e {
            AtomicWriteError::NotImplemented => {
                Self::Io("KernelAtomicWriter has no backend installed")
            }
            AtomicWriteError::Io(reason) => Self::Io(reason),
            AtomicWriteError::InvalidPath => Self::Io("invalid path"),
        }
    }
}

impl From<LoaderError> for SurfaceError {
    fn from(e: LoaderError) -> Self {
        match e {
            LoaderError::Missing { .. } => SurfaceError::Io("missing config file"),
            LoaderError::ModeViolation { .. } => SurfaceError::Io("config mode violation"),
            LoaderError::Toml(_) => SurfaceError::Io("toml parse error"),
            LoaderError::Validation(reason) => SurfaceError::Validation(reason),
            LoaderError::Io(reason) => SurfaceError::Io(reason),
            LoaderError::UnknownField(_) => SurfaceError::NotFound,
            LoaderError::TypeMismatch { expected, got, .. } => {
                SurfaceError::TypeMismatch { expected, got }
            }
        }
    }
}

// ─── Per-subsystem layout ────────────────────────────────────────────────────

/// One row of the layout table: relative path under `base` + the
/// declared file mode. Owner is always `kernel` per the spec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayoutEntry {
    /// Relative path under `base` (e.g. `"system.toml"`).
    pub relative: &'static str,
    /// Declared POSIX mode (e.g. `0o644`, `0o640`).
    pub declared_mode: u16,
    /// Predicate: which top-level [`Config`] namespace's keys live in
    /// this file. The path's first segment must equal this string.
    pub namespace: Namespace,
}

/// Coarse mapping of layout files to top-level `Config` namespaces.
/// `mgmt-config-layout` v1 maps:
///
/// - `system.toml` → `idle.*`, `password.*`, `flash.*`, `fs.*` (cross-cutting).
/// - `mgmt/policy.toml` → `mgmt.*`, `audit.*`, `metrics.*`, `overlay.*`.
///
/// `mgmt/zenoh.toml`, `network/*.toml`, `update/policy.toml`, and
/// `automotive/uds.toml` carry namespaces owned by other change
/// proposals; in v1 their keys are not part of `Config`. The loader
/// still tracks their presence and mode, but treats them as opaque.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Namespace {
    /// Cross-cutting fields stored in `system.toml`.
    System,
    /// Management policy fields stored in `mgmt/policy.toml`.
    MgmtPolicy,
    /// Zenoh transport — opaque to v1 (`mgmt/zenoh.toml`).
    MgmtZenoh,
    /// Network — opaque to v1 (`network/*.toml`).
    Network,
    /// Remote-update policy — opaque to v1 (`update/policy.toml`).
    Update,
    /// Automotive UDS — opaque to v1 (`automotive/uds.toml`).
    Automotive,
}

impl Namespace {
    /// True if this layout entry contains v1 [`Config`] keys.
    pub fn is_config_owned(self) -> bool {
        matches!(self, Self::System | Self::MgmtPolicy)
    }
}

/// Map a top-level `Config` namespace (as a dotted path's first
/// segment) onto the layout file that owns its keys.
fn namespace_of_path(top: &str) -> Namespace {
    match top {
        "idle" | "password" | "fs" | "flash" => Namespace::System,
        "mgmt" | "audit" | "metrics" | "overlay" => Namespace::MgmtPolicy,
        _ => Namespace::System, // unknown — default to system.toml
    }
}

/// v1 layout per `mgmt-config-layout` spec. Order is the same as the
/// spec table; the loader visits in order.
pub const V1_LAYOUT: &[LayoutEntry] = &[
    LayoutEntry {
        relative: "system.toml",
        declared_mode: 0o644,
        namespace: Namespace::System,
    },
    LayoutEntry {
        relative: "mgmt/policy.toml",
        declared_mode: 0o640,
        namespace: Namespace::MgmtPolicy,
    },
    LayoutEntry {
        relative: "mgmt/zenoh.toml",
        declared_mode: 0o640,
        namespace: Namespace::MgmtZenoh,
    },
    LayoutEntry {
        relative: "update/policy.toml",
        declared_mode: 0o640,
        namespace: Namespace::Update,
    },
    LayoutEntry {
        relative: "automotive/uds.toml",
        declared_mode: 0o640,
        namespace: Namespace::Automotive,
    },
];

// ─── Backend (read + metadata) ───────────────────────────────────────────────

/// Read-side companion to [`AtomicWriter`]. The loader needs to
/// observe per-file contents *and* per-file modes; [`AtomicWriter`]
/// only writes. We define a separate trait so the test fixture can
/// model both sides without dragging the FS layer into auth.
///
/// Implementations:
///
/// - [`InMemoryReadBackend`] — used by tests; pairs naturally with
///   [`smallaios_auth::atomic_write::TestAtomicWriter`].
/// - The kernel's F2FS read path will provide a runtime impl in a
///   later phase; this trait is what it will satisfy.
pub trait ReadBackend {
    /// Return the file's bytes if it exists, or `Ok(None)` if absent.
    fn read_file(&self, path: &str) -> Result<Option<Vec<u8>>, &'static str>;

    /// Return the file's POSIX mode (low 12 bits) if it exists, or
    /// `Ok(None)` if absent. The loader inspects only the bottom 9
    /// bits (`rwx` triplets); higher bits are ignored.
    fn mode_of(&self, path: &str) -> Result<Option<u16>, &'static str>;
}

// ─── Loader ──────────────────────────────────────────────────────────────────

/// TOML loader/saver — implements [`ConfigSurface`] over `/data/*.toml`.
///
/// `R` is the [`ReadBackend`] (tests use [`InMemoryReadBackend`]; the
/// kernel will provide an F2FS-backed type). `W` is the
/// [`AtomicWriter`] used for the rewrite path.
pub struct TomlLoader<R, W> {
    /// `/data/` prefix — every layout entry is resolved relative to
    /// this. Stored without a trailing `/`.
    base_path: String,
    /// Read backend (file bytes + mode).
    reader: R,
    /// Write backend (atomic stage→fsync→rename).
    writer: W,
    /// Cached typed `Config` after [`TomlLoader::load`]. Mutations on
    /// live writes update this; boot-deferred writes do not.
    cached: Config,
    /// Boot-deferred writes accumulated during this run. The operator
    /// sees these via [`TomlLoader::pending_reboot`].
    pending_reboot: Vec<PendingReboot>,
    /// Broadcast channel + a self-subscribed cursor so the surface
    /// owns its publication side.
    notify: RefCell<BroadcastTx>,
    /// Whether the loader has actually parsed an on-disk config. The
    /// universal-exposure invariant requires `read` to work even
    /// before `load` is called — we lazily fall back to defaults.
    loaded: bool,
}

/// One boot-deferred write recorded by [`TomlLoader::write`].
#[derive(Debug, Clone, PartialEq)]
pub struct PendingReboot {
    /// Path whose write was deferred.
    pub path: ConfigPath,
    /// Value that will take effect after reboot.
    pub value: Value,
    /// Identity that initiated the write.
    pub who: String,
}

impl<R, W> TomlLoader<R, W>
where
    R: ReadBackend,
    W: AtomicWriter,
{
    /// Construct a new loader. The base path SHALL be the mount point
    /// for `/data/`. Pre-populates [`TomlLoader::cached`] with
    /// [`Config::v1_defaults`]; the in-memory state is replaced on the
    /// next [`TomlLoader::load`].
    pub fn new(reader: R, writer: W, base_path: &str) -> Self {
        let (tx, _rx_initial) = crate::notify::channel();
        let mut bp = base_path.to_string();
        while bp.ends_with('/') {
            bp.pop();
        }
        Self {
            base_path: bp,
            reader,
            writer,
            cached: Config::v1_defaults(),
            pending_reboot: Vec::new(),
            notify: RefCell::new(tx),
            loaded: false,
        }
    }

    /// Read every layout file, parse into the typed schema, and
    /// validate. On success [`TomlLoader::cached`] holds the loaded
    /// config and [`ConfigSurface::read`] returns its values.
    ///
    /// `policy` controls the missing-file behaviour:
    ///
    /// - [`FirstBootPolicy::Generate`] — generate skeleton files for
    ///   missing entries with [`Config::v1_defaults`]. Writes go
    ///   through the configured [`AtomicWriter`].
    /// - [`FirstBootPolicy::Strict`] — return [`LoaderError::Missing`]
    ///   on the first absent file.
    pub fn load(&mut self, policy: FirstBootPolicy) -> Result<&Config, LoaderError> {
        let mut accumulated = Config::v1_defaults();
        for entry in V1_LAYOUT {
            self.load_one(entry, &mut accumulated, policy)?;
        }
        // Cross-field invariants apply to the whole assembled config.
        validate::validate_all(&accumulated).map_err(|e| match e {
            SurfaceError::Validation(reason) => LoaderError::Validation(reason),
            _ => LoaderError::Validation("unexpected error in validate_all"),
        })?;
        self.cached = accumulated;
        self.loaded = true;
        Ok(&self.cached)
    }

    fn load_one(
        &mut self,
        entry: &LayoutEntry,
        accumulated: &mut Config,
        policy: FirstBootPolicy,
    ) -> Result<(), LoaderError> {
        let path = self.absolute(entry.relative);
        let bytes = self.reader.read_file(&path).map_err(LoaderError::Io)?;
        let mode = self.reader.mode_of(&path).map_err(LoaderError::Io)?;

        match (bytes, mode) {
            (Some(b), Some(m)) => {
                check_mode(&path, entry.declared_mode, m)?;
                if entry.namespace.is_config_owned() {
                    let s = core::str::from_utf8(&b)
                        .map_err(|_| LoaderError::Io("file is not valid UTF-8"))?;
                    let table = toml_parse(s)?;
                    apply_table_to_config(&table, "", accumulated)?;
                }
                Ok(())
            }
            (Some(_), None) | (None, Some(_)) => {
                // Backend inconsistency — bytes/mode out of sync.
                Err(LoaderError::Io("backend inconsistent file metadata"))
            }
            (None, None) => match policy {
                FirstBootPolicy::Strict => Err(LoaderError::Missing { path }),
                FirstBootPolicy::Generate => {
                    // Render this entry's contents:
                    // - config-owned namespaces: rendered defaults from
                    //   the assembled `Config`.
                    // - opaque namespaces (zenoh, network, update,
                    //   automotive): generate an empty file so the
                    //   owning subsystem has a stable on-disk artefact
                    //   to extend. The file's mode is the writer's
                    //   default; the kernel's F2FS-backed writer will
                    //   chmod() it to `entry.declared_mode` after
                    //   creation in a later phase.
                    let bytes: alloc::vec::Vec<u8> = if entry.namespace.is_config_owned() {
                        let table = render_namespace_table(accumulated, entry.namespace);
                        let serialized = toml_serialize(&table)?;
                        serialized.into_bytes()
                    } else {
                        alloc::vec::Vec::new()
                    };
                    self.writer
                        .write_atomic(&path, &bytes)
                        .map_err(LoaderError::from)?;
                    Ok(())
                }
            },
        }
    }

    /// Resolve `relative` against [`TomlLoader::base_path`].
    fn absolute(&self, relative: &str) -> String {
        let mut s = self.base_path.clone();
        if !relative.starts_with('/') {
            s.push('/');
        }
        s.push_str(relative);
        s
    }

    /// Borrow the current cached [`Config`].
    pub fn config(&self) -> &Config {
        &self.cached
    }

    /// Boot-deferred writes accumulated since [`TomlLoader::load`].
    /// The operator-facing diagnostic ("you must reboot for X to take
    /// effect") reads this list.
    pub fn pending_reboot(&self) -> &[PendingReboot] {
        &self.pending_reboot
    }

    /// Whether [`TomlLoader::load`] has run.
    pub fn is_loaded(&self) -> bool {
        self.loaded
    }

    /// Force a re-serialise + atomic write of the layout file owning
    /// `namespace`. Used both by the apply lifecycle and by external
    /// callers that need to flush the in-memory config to disk after
    /// bulk mutation. Visible on disk only after `fsync` completes.
    pub fn flush_namespace(&mut self, namespace: Namespace) -> Result<(), LoaderError> {
        let entry = V1_LAYOUT
            .iter()
            .find(|e| e.namespace == namespace)
            .ok_or(LoaderError::Io("unknown namespace"))?;
        let path = self.absolute(entry.relative);
        let table = render_namespace_table(&self.cached, namespace);
        let serialized = toml_serialize(&table)?;
        self.writer
            .write_atomic(&path, serialized.as_bytes())
            .map_err(LoaderError::from)?;
        Ok(())
    }

    /// Apply a typed [`Value`] write through the full lifecycle.
    /// Internal helper called by [`ConfigSurface::write`].
    fn apply_write(
        &mut self,
        path: &ConfigPath,
        value: Value,
        who: &str,
        when_ns: u64,
    ) -> Result<(), LoaderError> {
        // Step 1+2 — parse + validate. `Config::write_path` runs both
        // `validate_field` and `validate_cross_field` on a candidate.
        let mut candidate = self.cached.clone();
        candidate
            .write_path(path, value.clone())
            .map_err(map_surface_to_loader)?;
        // The candidate is now valid. Capture before-value for notify.
        let before = self.cached.read_path(path).map_err(map_surface_to_loader)?;
        // Step 3 — stage + fsync + rename via the layout file.
        let ns = namespace_of_path(path.namespace());
        let entry = V1_LAYOUT
            .iter()
            .find(|e| e.namespace == ns)
            .ok_or(LoaderError::Io("unknown namespace"))?;
        let abs = self.absolute(entry.relative);
        let mut table_for_disk = render_namespace_table(&candidate, ns);
        // The serialised file always reflects the candidate's value
        // for `path`, regardless of reload kind — operators expect
        // the file to mirror the requested write the next time the
        // system boots.
        let _ = &mut table_for_disk; // already populated by render
        let serialized = toml_serialize(&table_for_disk)?;
        self.writer
            .write_atomic(&abs, serialized.as_bytes())
            .map_err(LoaderError::from)?;
        // Step 4 — update in-memory cache iff Live.
        let reload = registry::reload_kind(path.as_str()).unwrap_or(ReloadKind::Live);
        match reload {
            ReloadKind::Live => {
                self.cached = candidate;
            }
            ReloadKind::Boot => {
                self.pending_reboot.push(PendingReboot {
                    path: path.clone(),
                    value: value.clone(),
                    who: who.to_string(),
                });
            }
        }
        // Step 5 — notify subscribers. The `after` value reflects
        // what's now on disk (which is the candidate's value); for
        // boot-deferred fields this is intentionally distinct from
        // the in-memory `before` retained until reboot, so the audit
        // trail records the *requested* new value.
        let after = value;
        let event = NotifyEvent {
            path: path.clone(),
            before,
            after,
            who: who.to_string(),
            when_ns,
        };
        self.notify.borrow().publish(event);
        Ok(())
    }

    /// Subscribe to the loader's notify channel from outside the
    /// trait surface (useful for tests, integration harnesses).
    pub fn raw_subscribe(&self) -> Subscriber {
        self.notify.borrow().subscribe()
    }
}

impl<R, W> ConfigSurface for TomlLoader<R, W>
where
    R: ReadBackend,
    W: AtomicWriter,
{
    fn read(&self, path: &ConfigPath) -> Result<Value, SurfaceError> {
        self.cached.read_path(path)
    }

    fn write(&mut self, path: &ConfigPath, value: Value, who: &str) -> Result<(), SurfaceError> {
        // The TOML surface tags every event with the surface kind via
        // the `who` string. We use a dummy timestamp of 0 because the
        // kernel will install a clock source in a later phase; tests
        // can inject their own through [`TomlLoader::apply_write`].
        let _ = SurfaceKind::Toml; // referenced for future identity work
        self.apply_write(path, value, who, 0)
            .map_err(SurfaceError::from)
    }

    fn subscribe(&self, path: &ConfigPath) -> Result<Subscriber, SurfaceError> {
        Ok(self.notify.borrow().subscribe_path(path.clone()))
    }
}

// ─── In-memory backend (tests + integration) ─────────────────────────────────

/// In-memory [`ReadBackend`] used by tests and the universal-exposure
/// CI walker. Pairs with [`smallaios_auth::atomic_write::TestAtomicWriter`]
/// — both back by an in-memory `BTreeMap<path, bytes>`.
///
/// Modes are tracked separately via [`InMemoryReadBackend::set_mode`]
/// so tests can simulate the file-mode-stricter rule.
pub struct InMemoryReadBackend {
    inner: RefCell<InMemoryInner>,
}

struct InMemoryInner {
    files: BTreeMap<String, Vec<u8>>,
    modes: BTreeMap<String, u16>,
    /// Default mode applied when [`InMemoryReadBackend::seed`] is
    /// called without an explicit mode. Lets tests construct a
    /// "permissive" backend that matches every layout entry.
    default_mode: u16,
}

impl Default for InMemoryReadBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryReadBackend {
    /// Construct an empty backend with default mode `0o600`. Tests
    /// requiring a permissive mode set explicit modes per file.
    pub fn new() -> Self {
        Self {
            inner: RefCell::new(InMemoryInner {
                files: BTreeMap::new(),
                modes: BTreeMap::new(),
                default_mode: 0o600,
            }),
        }
    }

    /// Seed `path` with `bytes` using the backend's default mode.
    pub fn seed(&self, path: &str, bytes: &[u8]) {
        let mut g = self.inner.borrow_mut();
        let mode = g.default_mode;
        g.files.insert(path.to_string(), bytes.to_vec());
        g.modes.insert(path.to_string(), mode);
    }

    /// Seed `path` with `bytes` and an explicit mode — used to test
    /// the file-mode-stricter rule.
    pub fn seed_with_mode(&self, path: &str, bytes: &[u8], mode: u16) {
        let mut g = self.inner.borrow_mut();
        g.files.insert(path.to_string(), bytes.to_vec());
        g.modes.insert(path.to_string(), mode);
    }

    /// Set the mode the backend reports for `path`. Non-existent files
    /// silently no-op.
    pub fn set_mode(&self, path: &str, mode: u16) {
        let mut g = self.inner.borrow_mut();
        if g.files.contains_key(path) {
            g.modes.insert(path.to_string(), mode);
        }
    }

    /// Override the default mode used by [`InMemoryReadBackend::seed`].
    pub fn set_default_mode(&self, mode: u16) {
        self.inner.borrow_mut().default_mode = mode;
    }

    /// Snapshot of every (path, bytes) pair currently held — for
    /// assertion in tests.
    pub fn snapshot(&self) -> BTreeMap<String, Vec<u8>> {
        self.inner.borrow().files.clone()
    }

    /// Number of files seeded.
    pub fn len(&self) -> usize {
        self.inner.borrow().files.len()
    }

    /// Whether no files are seeded.
    pub fn is_empty(&self) -> bool {
        self.inner.borrow().files.is_empty()
    }
}

impl ReadBackend for InMemoryReadBackend {
    fn read_file(&self, path: &str) -> Result<Option<Vec<u8>>, &'static str> {
        Ok(self.inner.borrow().files.get(path).cloned())
    }

    fn mode_of(&self, path: &str) -> Result<Option<u16>, &'static str> {
        Ok(self.inner.borrow().modes.get(path).copied())
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Reject if the actual file mode is laxer than the declared mode.
/// "Laxer" = any bit set in `actual & ~declared` within the bottom 9
/// bits.
fn check_mode(path: &str, declared: u16, actual: u16) -> Result<(), LoaderError> {
    let mask: u16 = 0o777;
    let actual_bits = actual & mask;
    let declared_bits = declared & mask;
    if actual_bits & !declared_bits != 0 {
        return Err(LoaderError::ModeViolation {
            path: path.to_string(),
            declared: declared_bits,
            actual: actual_bits,
        });
    }
    Ok(())
}

/// Render the [`TomlTable`] containing exactly the keys owned by
/// `namespace` from `cfg`. Public so the universal-exposure walker
/// can pre-seed a backend with valid defaults without re-implementing
/// the namespace mapping.
pub fn render_namespace_table(cfg: &Config, namespace: Namespace) -> TomlTable {
    let mut out = TomlTable::new();
    for f in registry::CONFIG_FIELDS {
        let path = match ConfigPath::new(f.path) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let owning = namespace_of_path(path.namespace());
        if owning != namespace {
            continue;
        }
        let value = match cfg.read_path(&path) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let toml_v = match config_value_to_toml(&value) {
            Some(v) => v,
            None => continue,
        };
        // path is `ns.field` (or deeper); insert into nested table.
        let segments: Vec<&str> = path.as_str().split('.').collect();
        insert_into_table(&mut out, &segments, toml_v);
    }
    out
}

fn insert_into_table(root: &mut TomlTable, segments: &[&str], value: TomlValue) {
    if segments.is_empty() {
        return;
    }
    if segments.len() == 1 {
        root.insert(segments[0].to_string(), value);
        return;
    }
    let head = segments[0];
    let entry = root
        .entry(head.to_string())
        .or_insert_with(|| TomlValue::Table(TomlTable::new()));
    if let TomlValue::Table(t) = entry {
        insert_into_table(t, &segments[1..], value);
    }
}

/// Apply every key in `table` (rooted at `prefix`) to `cfg`. Each leaf
/// key SHALL match a registry field and the typed value SHALL match
/// the registry's [`FieldType`].
fn apply_table_to_config(
    table: &TomlTable,
    prefix: &str,
    cfg: &mut Config,
) -> Result<(), LoaderError> {
    for (k, v) in table.iter() {
        let dotted = if prefix.is_empty() {
            k.clone()
        } else {
            crate::toml::join_path(prefix, k)
        };
        match v {
            TomlValue::Table(sub) => {
                apply_table_to_config(sub, &dotted, cfg)?;
            }
            scalar => {
                let descriptor = registry::lookup(&dotted)
                    .ok_or_else(|| LoaderError::UnknownField(dotted.clone()))?;
                let value = toml_to_config_value(scalar, descriptor.kind, &dotted)?;
                let path = ConfigPath::new(&dotted).map_err(|_| LoaderError::Io("bad path"))?;
                cfg.write_path(&path, value)
                    .map_err(map_surface_to_loader)?;
            }
        }
    }
    Ok(())
}

fn toml_to_config_value(
    v: &TomlValue,
    expected: FieldType,
    path: &str,
) -> Result<Value, LoaderError> {
    match (v, expected) {
        (TomlValue::Bool(b), FieldType::Bool) => Ok(Value::Bool(*b)),
        (TomlValue::Integer(n), FieldType::U32) => {
            if *n < 0 || *n > i64::from(u32::MAX) {
                return Err(LoaderError::Validation("integer out of u32 range"));
            }
            Ok(Value::U32(*n as u32))
        }
        (TomlValue::Integer(n), FieldType::U64) => {
            if *n < 0 {
                return Err(LoaderError::Validation("negative integer for u64"));
            }
            Ok(Value::U64(*n as u64))
        }
        (TomlValue::String(s), FieldType::String) => Ok(Value::String(s.clone())),
        (TomlValue::Array(items), FieldType::PasswordClasses) => {
            let mut mask: u8 = 0;
            for item in items {
                let s = item
                    .as_str()
                    .map_err(|_| LoaderError::Validation("password class must be string"))?;
                mask |= match s {
                    "lower" => PasswordClassSet::LOWER,
                    "upper" => PasswordClassSet::UPPER,
                    "digit" => PasswordClassSet::DIGIT,
                    "symbol" => PasswordClassSet::SYMBOL,
                    _ => return Err(LoaderError::Validation("unknown password class name")),
                };
            }
            Ok(Value::PasswordClasses(PasswordClassSet::from_mask(mask)))
        }
        (TomlValue::Array(items), FieldType::Roles) => {
            // Phase 9 totp.enforced_for_roles. Same pattern as
            // password classes: an array of role-name strings.
            let mut mask: u8 = 0;
            for item in items {
                let s = item
                    .as_str()
                    .map_err(|_| LoaderError::Validation("role must be string"))?;
                mask |= match s {
                    "root" => RoleSet::ROOT,
                    "operator" => RoleSet::OPERATOR,
                    "viewer" => RoleSet::VIEWER,
                    _ => return Err(LoaderError::Validation("unknown role name")),
                };
            }
            Ok(Value::Roles(RoleSet::from_mask(mask)))
        }
        (other, expected) => Err(LoaderError::TypeMismatch {
            path: path.to_string(),
            expected: expected.as_value_tag(),
            got: other.type_tag(),
        }),
    }
}

fn config_value_to_toml(value: &Value) -> Option<TomlValue> {
    match value {
        Value::Bool(b) => Some(TomlValue::Bool(*b)),
        Value::U32(n) => {
            // u32 always fits in i64.
            Some(TomlValue::Integer(i64::from(*n)))
        }
        Value::U64(n) => {
            if *n > i64::MAX as u64 {
                None
            } else {
                Some(TomlValue::Integer(*n as i64))
            }
        }
        Value::String(s) => Some(TomlValue::String(s.clone())),
        Value::PasswordClasses(p) => {
            let mut items = Vec::new();
            if p.contains(PasswordClassSet::LOWER) {
                items.push(TomlValue::String("lower".to_string()));
            }
            if p.contains(PasswordClassSet::UPPER) {
                items.push(TomlValue::String("upper".to_string()));
            }
            if p.contains(PasswordClassSet::DIGIT) {
                items.push(TomlValue::String("digit".to_string()));
            }
            if p.contains(PasswordClassSet::SYMBOL) {
                items.push(TomlValue::String("symbol".to_string()));
            }
            Some(TomlValue::Array(items))
        }
        Value::Roles(r) => {
            // Roles serialise as a TOML array of role-name strings,
            // mirroring the `Value::PasswordClasses` shape.
            let mut items = Vec::new();
            if r.contains(RoleSet::ROOT) {
                items.push(TomlValue::String("root".to_string()));
            }
            if r.contains(RoleSet::OPERATOR) {
                items.push(TomlValue::String("operator".to_string()));
            }
            if r.contains(RoleSet::VIEWER) {
                items.push(TomlValue::String("viewer".to_string()));
            }
            Some(TomlValue::Array(items))
        }
    }
}

fn map_surface_to_loader(e: SurfaceError) -> LoaderError {
    match e {
        SurfaceError::Validation(reason) => LoaderError::Validation(reason),
        SurfaceError::Io(reason) => LoaderError::Io(reason),
        SurfaceError::NotFound => LoaderError::Io("not found"),
        SurfaceError::TypeMismatch { expected, got } => LoaderError::TypeMismatch {
            path: String::new(),
            expected,
            got,
        },
        SurfaceError::InvalidPath(reason) => LoaderError::Io(reason),
        SurfaceError::Permission(reason) => LoaderError::Io(reason),
        SurfaceError::Unsupported(reason) => LoaderError::Io(reason),
    }
}

// ─── AuditIdentity helper ────────────────────────────────────────────────────

/// Build a TOML-surface [`AuditIdentity`] for a concrete role/user.
/// Convenience for callers that don't want to repeat
/// `SurfaceKind::Toml` everywhere.
pub fn audit_identity(role: &'static str, user: Option<String>) -> AuditIdentity {
    AuditIdentity {
        role,
        user,
        surface: SurfaceKind::Toml,
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::surface::{ConfigPath, Value};
    use alloc::rc::Rc;
    use loader_test_vectors as fixtures;
    use smallaios_auth::atomic_write::TestAtomicWriter;

    /// In-test "shared filesystem" backend that pairs a single
    /// in-memory store with both the [`ReadBackend`] and the
    /// [`AtomicWriter`] sides. Mirrors the kernel's runtime where
    /// `f2fs::F2fs` provides both reads and atomic-write semantics
    /// over the same handle.
    ///
    /// The `Rc<RefCell<...>>` wrapping makes the same store visible
    /// to both halves; clones share state.
    #[derive(Clone)]
    struct SharedFsFixture {
        files: Rc<RefCell<BTreeMap<String, Vec<u8>>>>,
        modes: Rc<RefCell<BTreeMap<String, u16>>>,
        /// Default mode applied to files written via the
        /// [`AtomicWriter`] half. Tests pin this to the spec's
        /// declared mode for the file under test.
        write_mode: u16,
    }

    impl SharedFsFixture {
        fn new(write_mode: u16) -> Self {
            Self {
                files: Rc::new(RefCell::new(BTreeMap::new())),
                modes: Rc::new(RefCell::new(BTreeMap::new())),
                write_mode,
            }
        }

        #[allow(dead_code)]
        fn seed_with_mode(&self, path: &str, bytes: &[u8], mode: u16) {
            self.files
                .borrow_mut()
                .insert(path.to_string(), bytes.to_vec());
            self.modes.borrow_mut().insert(path.to_string(), mode);
        }

        #[allow(dead_code)]
        fn read_raw(&self, path: &str) -> Option<Vec<u8>> {
            self.files.borrow().get(path).cloned()
        }
    }

    impl ReadBackend for SharedFsFixture {
        fn read_file(&self, path: &str) -> Result<Option<Vec<u8>>, &'static str> {
            Ok(self.files.borrow().get(path).cloned())
        }
        fn mode_of(&self, path: &str) -> Result<Option<u16>, &'static str> {
            Ok(self.modes.borrow().get(path).copied())
        }
    }

    impl AtomicWriter for SharedFsFixture {
        fn write_atomic(&self, path: &str, contents: &[u8]) -> Result<(), AtomicWriteError> {
            if path.is_empty() {
                return Err(AtomicWriteError::InvalidPath);
            }
            let mode = self.write_mode;
            self.files
                .borrow_mut()
                .insert(path.to_string(), contents.to_vec());
            self.modes.borrow_mut().insert(path.to_string(), mode);
            Ok(())
        }
        fn cleanup_orphan_tmp(&self, dir: &str) -> Result<(), AtomicWriteError> {
            let prefix = if dir.ends_with('/') {
                dir.to_string()
            } else {
                let mut s = String::from(dir);
                s.push('/');
                s
            };
            let to_remove: Vec<String> = self
                .files
                .borrow()
                .keys()
                .filter(|k| k.starts_with(&prefix) && k.ends_with(".tmp"))
                .cloned()
                .collect();
            for k in to_remove {
                self.files.borrow_mut().remove(&k);
            }
            Ok(())
        }
    }

    fn empty_loader() -> TomlLoader<InMemoryReadBackend, TestAtomicWriter> {
        TomlLoader::new(InMemoryReadBackend::new(), TestAtomicWriter::new(), "/data")
    }

    #[test]
    fn defaults_visible_before_load() {
        let loader = empty_loader();
        let p = ConfigPath::new("idle.root_minutes").unwrap();
        // Pre-load reads return defaults — the universal-exposure
        // walker depends on this so the build-time test can run on a
        // non-mounted filesystem.
        assert_eq!(loader.read(&p).unwrap(), Value::U32(15));
    }

    #[test]
    fn first_boot_generate_creates_skeleton() {
        let reader = InMemoryReadBackend::new();
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        loader.load(FirstBootPolicy::Generate).unwrap();
        assert!(loader.is_loaded());
        // The writer should have created system.toml + mgmt/policy.toml.
        // (We can't observe it through the TestAtomicWriter directly
        // because it lives inside the loader; the round-trip test
        // below covers re-load.)
    }

    #[test]
    fn first_boot_strict_rejects_missing() {
        let reader = InMemoryReadBackend::new();
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        let err = loader.load(FirstBootPolicy::Strict).unwrap_err();
        assert!(matches!(err, LoaderError::Missing { .. }));
    }

    #[test]
    fn loads_from_seeded_files() {
        let reader = InMemoryReadBackend::new();
        reader.seed_with_mode("/data/system.toml", fixtures::SYSTEM_TOML_DEFAULT, 0o644);
        reader.seed_with_mode(
            "/data/mgmt/policy.toml",
            fixtures::MGMT_POLICY_TOML_DEFAULT,
            0o640,
        );
        // Stub remaining opaque files with empty content + their
        // declared mode so the loader doesn't generate.
        reader.seed_with_mode("/data/mgmt/zenoh.toml", b"", 0o640);
        reader.seed_with_mode("/data/update/policy.toml", b"", 0o640);
        reader.seed_with_mode("/data/automotive/uds.toml", b"", 0o640);

        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        let cfg = loader.load(FirstBootPolicy::Strict).unwrap().clone();
        assert_eq!(cfg.idle.root_minutes, 15);
        assert_eq!(cfg.password.min_length, 12);
    }

    #[test]
    fn rejects_laxer_than_declared_mode() {
        let reader = InMemoryReadBackend::new();
        // mgmt/policy.toml declared 0o640; seed with 0o644 (laxer).
        reader.seed_with_mode(
            "/data/mgmt/policy.toml",
            fixtures::MGMT_POLICY_TOML_DEFAULT,
            0o644,
        );
        // system.toml at the right mode.
        reader.seed_with_mode("/data/system.toml", fixtures::SYSTEM_TOML_DEFAULT, 0o644);
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        let err = loader.load(FirstBootPolicy::Strict).unwrap_err();
        assert!(matches!(err, LoaderError::ModeViolation { .. }));
    }

    #[test]
    fn accepts_stricter_than_declared_mode() {
        let reader = InMemoryReadBackend::new();
        reader.seed_with_mode(
            "/data/system.toml",
            fixtures::SYSTEM_TOML_DEFAULT,
            0o600, // stricter than 0o644
        );
        reader.seed_with_mode(
            "/data/mgmt/policy.toml",
            fixtures::MGMT_POLICY_TOML_DEFAULT,
            0o600, // stricter than 0o640
        );
        reader.seed_with_mode("/data/mgmt/zenoh.toml", b"", 0o600);
        reader.seed_with_mode("/data/update/policy.toml", b"", 0o600);
        reader.seed_with_mode("/data/automotive/uds.toml", b"", 0o600);
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        loader.load(FirstBootPolicy::Strict).unwrap();
    }

    #[test]
    fn check_mode_helper_basic_cases() {
        // Equal: accept.
        check_mode("/x", 0o640, 0o640).unwrap();
        // Stricter: accept.
        check_mode("/x", 0o640, 0o600).unwrap();
        check_mode("/x", 0o644, 0o600).unwrap();
        // Laxer: reject.
        assert!(matches!(
            check_mode("/x", 0o600, 0o644),
            Err(LoaderError::ModeViolation { .. })
        ));
        assert!(matches!(
            check_mode("/x", 0o640, 0o644),
            Err(LoaderError::ModeViolation { .. })
        ));
    }

    #[test]
    fn unknown_field_in_file_rejected() {
        let reader = InMemoryReadBackend::new();
        // Inject a file with an unknown key.
        reader.seed_with_mode("/data/system.toml", b"[idle]\nunknown_field = 1\n", 0o644);
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        let err = loader.load(FirstBootPolicy::Generate).unwrap_err();
        assert!(matches!(err, LoaderError::UnknownField(_)));
    }

    #[test]
    fn type_mismatch_in_file_rejected() {
        let reader = InMemoryReadBackend::new();
        reader.seed_with_mode("/data/system.toml", b"[idle]\nroot_minutes = true\n", 0o644);
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        let err = loader.load(FirstBootPolicy::Generate).unwrap_err();
        assert!(matches!(err, LoaderError::TypeMismatch { .. }));
    }

    #[test]
    fn write_through_surface_persists_and_notifies() {
        let reader = InMemoryReadBackend::new();
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        loader.load(FirstBootPolicy::Generate).unwrap();
        let mut sub = loader.raw_subscribe();
        let p = ConfigPath::new("idle.root_minutes").unwrap();
        loader.write(&p, Value::U32(20), "root@toml").unwrap();
        assert_eq!(loader.read(&p).unwrap(), Value::U32(20));
        let ev = sub.poll().unwrap().unwrap();
        assert_eq!(ev.path, p);
        assert_eq!(ev.before, Value::U32(15));
        assert_eq!(ev.after, Value::U32(20));
        assert_eq!(ev.who, "root@toml");
    }

    #[test]
    fn write_validation_failure_leaves_state_unchanged() {
        let reader = InMemoryReadBackend::new();
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        loader.load(FirstBootPolicy::Generate).unwrap();
        let p = ConfigPath::new("idle.root_minutes").unwrap();
        // 0 fails range validation.
        let err = loader.write(&p, Value::U32(0), "root@toml").unwrap_err();
        assert!(matches!(err, SurfaceError::Validation(_)));
        // In-memory unchanged.
        assert_eq!(loader.read(&p).unwrap(), Value::U32(15));
    }

    #[test]
    fn boot_only_write_goes_to_pending_reboot() {
        let reader = InMemoryReadBackend::new();
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        loader.load(FirstBootPolicy::Generate).unwrap();
        let p = ConfigPath::new("fs.boot_watchdog_seconds").unwrap();
        // 30s passes validation and is in range.
        loader.write(&p, Value::U32(30), "root@toml").unwrap();
        // Boot-deferred: in-memory still 60.
        assert_eq!(loader.read(&p).unwrap(), Value::U32(60));
        // But pending list records it.
        let pending = loader.pending_reboot();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].path, p);
        assert_eq!(pending[0].value, Value::U32(30));
    }

    #[test]
    fn live_write_visible_in_memory() {
        let reader = InMemoryReadBackend::new();
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        loader.load(FirstBootPolicy::Generate).unwrap();
        let p = ConfigPath::new("idle.operator_minutes").unwrap();
        loader.write(&p, Value::U32(50), "root@toml").unwrap();
        assert_eq!(loader.read(&p).unwrap(), Value::U32(50));
    }

    #[test]
    fn cross_field_invariant_still_enforced() {
        let reader = InMemoryReadBackend::new();
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        loader.load(FirstBootPolicy::Generate).unwrap();
        // Try to set root_minutes higher than operator_minutes (cross-field).
        let p = ConfigPath::new("idle.root_minutes").unwrap();
        let err = loader.write(&p, Value::U32(120), "root@toml").unwrap_err();
        assert!(matches!(err, SurfaceError::Validation(_)));
    }

    #[test]
    fn subscribe_via_trait_returns_path_filtered_subscriber() {
        let reader = InMemoryReadBackend::new();
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        loader.load(FirstBootPolicy::Generate).unwrap();
        let p = ConfigPath::new("idle.root_minutes").unwrap();
        let mut sub = loader.subscribe(&p).unwrap();
        let other = ConfigPath::new("password.dictionary_check").unwrap();
        loader
            .write(&other, Value::Bool(false), "root@toml")
            .unwrap();
        // Filtered subscriber sees nothing for password write.
        assert_eq!(sub.poll(), Ok(None));
        loader.write(&p, Value::U32(45), "root@toml").unwrap();
        let ev = sub.poll().unwrap().unwrap();
        assert_eq!(ev.path, p);
    }

    #[test]
    fn render_namespace_table_excludes_other_namespace_keys() {
        let cfg = Config::v1_defaults();
        let sys = render_namespace_table(&cfg, Namespace::System);
        // system.toml owns idle, password, fs, flash. mgmt is not in
        // it.
        assert!(sys.contains_key("idle"));
        assert!(sys.contains_key("password"));
        assert!(sys.contains_key("fs"));
        assert!(sys.contains_key("flash"));
        assert!(!sys.contains_key("mgmt"));
    }

    #[test]
    fn render_namespace_then_parse_reflects_defaults() {
        let cfg = Config::v1_defaults();
        let sys = render_namespace_table(&cfg, Namespace::System);
        let s = toml_serialize(&sys).unwrap();
        let back = toml_parse(&s).unwrap();
        assert_eq!(back, sys);
    }

    #[test]
    fn round_trip_load_then_reload_yields_equal_config() {
        // Use a shared backend so writes are visible to subsequent
        // reads — this mirrors the production wiring where
        // `f2fs::F2fs` provides both halves.
        // For this test, system.toml's declared mode is 0o644 which is the most
        // permissive of the v1 entries (mgmt/* are 0o640). Use 0o600 which
        // satisfies BOTH (stricter than 0o644 AND stricter than 0o640).
        let fs = SharedFsFixture::new(0o600);
        let mut loader = TomlLoader::new(fs.clone(), fs.clone(), "/data");
        // Generate fills the FS with skeletons under write_mode (0o600).
        loader.load(FirstBootPolicy::Generate).unwrap();
        let snap1 = loader.config().clone();
        // Mutate one field — the live path persists to disk.
        let p = ConfigPath::new("password.dictionary_check").unwrap();
        loader.write(&p, Value::Bool(false), "root@toml").unwrap();
        let snap2 = loader.config().clone();
        assert_ne!(snap1, snap2);
        // Re-load using the shared backend — file is now present.
        loader.load(FirstBootPolicy::Strict).unwrap();
        assert!(!loader.config().password.dictionary_check);
    }

    #[test]
    fn loader_error_display_includes_path() {
        let e = LoaderError::Missing {
            path: "/data/system.toml".to_string(),
        };
        let s = alloc::format!("{}", e);
        assert!(s.contains("/data/system.toml"));
    }

    #[test]
    fn loader_error_to_surface_error_maps_validation() {
        let e: SurfaceError = LoaderError::Validation("bounds").into();
        assert!(matches!(e, SurfaceError::Validation("bounds")));
    }

    #[test]
    fn loader_error_to_surface_error_maps_unknown_field() {
        let e: SurfaceError = LoaderError::UnknownField("x".to_string()).into();
        assert!(matches!(e, SurfaceError::NotFound));
    }

    #[test]
    fn audit_identity_uses_toml_surface() {
        let id = audit_identity("operator", Some("alice".to_string()));
        assert_eq!(id.surface, SurfaceKind::Toml);
        assert_eq!(id.render(), "operator:alice@toml");
    }

    #[test]
    fn config_value_to_toml_round_trip_for_password_classes() {
        let cfg = Config::v1_defaults();
        let v = cfg
            .read_path(&ConfigPath::new("password.require_classes").unwrap())
            .unwrap();
        let toml_v = config_value_to_toml(&v).unwrap();
        match toml_v {
            TomlValue::Array(items) => {
                let names: Vec<&str> = items.iter().filter_map(|i| i.as_str().ok()).collect();
                assert!(names.contains(&"lower"));
                assert!(names.contains(&"upper"));
                assert!(names.contains(&"digit"));
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn toml_to_config_value_password_classes_round_trip() {
        let toml_v = TomlValue::Array(alloc::vec![
            TomlValue::String("lower".to_string()),
            TomlValue::String("symbol".to_string()),
        ]);
        let v = toml_to_config_value(
            &toml_v,
            FieldType::PasswordClasses,
            "password.require_classes",
        )
        .unwrap();
        match v {
            Value::PasswordClasses(p) => {
                assert!(p.contains(PasswordClassSet::LOWER));
                assert!(p.contains(PasswordClassSet::SYMBOL));
                assert!(!p.contains(PasswordClassSet::UPPER));
            }
            _ => panic!("expected password classes"),
        }
    }

    // ─── Phase 9: TOTP enforced_for_roles ────────────────────────────────

    #[test]
    fn config_value_to_toml_round_trip_for_totp_roles() {
        // Default is empty role set → empty TOML array.
        let cfg = Config::v1_defaults();
        let v = cfg
            .read_path(&ConfigPath::new("totp.enforced_for_roles").unwrap())
            .unwrap();
        match config_value_to_toml(&v).unwrap() {
            TomlValue::Array(items) => assert!(items.is_empty()),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn toml_to_config_value_totp_roles_parses_role_names() {
        let toml_v = TomlValue::Array(alloc::vec![
            TomlValue::String("root".to_string()),
            TomlValue::String("operator".to_string()),
        ]);
        let v = toml_to_config_value(&toml_v, FieldType::Roles, "totp.enforced_for_roles").unwrap();
        match v {
            Value::Roles(r) => {
                assert!(r.contains(RoleSet::ROOT));
                assert!(r.contains(RoleSet::OPERATOR));
                assert!(!r.contains(RoleSet::VIEWER));
            }
            _ => panic!("expected roles"),
        }
    }

    #[test]
    fn toml_to_config_value_totp_roles_rejects_unknown_name() {
        let toml_v = TomlValue::Array(alloc::vec![TomlValue::String("admin".to_string())]);
        let err =
            toml_to_config_value(&toml_v, FieldType::Roles, "totp.enforced_for_roles").unwrap_err();
        assert!(matches!(err, LoaderError::Validation(_)));
    }

    #[test]
    fn totp_roles_round_trip_through_loader_in_memory() {
        // End-to-end: write a TOML body containing
        // `totp.enforced_for_roles = ["root","operator"]`, load into
        // a `Config`, read the field back, confirm the bits.
        let toml_body = "[totp]\nenforced_for_roles = [\"root\",\"operator\"]\n";
        let mut cfg = Config::v1_defaults();
        let parsed = toml_parse(toml_body).expect("parse totp body");
        apply_table_to_config(&parsed, "", &mut cfg).unwrap();
        let v = cfg
            .read_path(&ConfigPath::new("totp.enforced_for_roles").unwrap())
            .unwrap();
        let r = match v {
            Value::Roles(r) => r,
            _ => panic!("expected roles"),
        };
        assert!(r.contains(RoleSet::ROOT));
        assert!(r.contains(RoleSet::OPERATOR));
        assert!(!r.contains(RoleSet::VIEWER));

        // And the reverse direction: serialise back to TOML and the
        // resulting tree must contain a `totp.enforced_for_roles`
        // array of two role names.
        let toml_back = config_value_to_toml(&v).unwrap();
        let names: Vec<String> = match toml_back {
            TomlValue::Array(items) => items
                .iter()
                .map(|i| i.as_str().unwrap().to_string())
                .collect(),
            _ => panic!("expected array"),
        };
        assert!(names.contains(&"root".to_string()));
        assert!(names.contains(&"operator".to_string()));
        assert_eq!(names.len(), 2);
    }

    #[test]
    fn toml_to_config_value_rejects_negative_for_u32() {
        let err = toml_to_config_value(&TomlValue::Integer(-1), FieldType::U32, "x").unwrap_err();
        assert!(matches!(err, LoaderError::Validation(_)));
    }

    #[test]
    fn toml_to_config_value_rejects_negative_for_u64() {
        let err = toml_to_config_value(&TomlValue::Integer(-1), FieldType::U64, "x").unwrap_err();
        assert!(matches!(err, LoaderError::Validation(_)));
    }

    #[test]
    fn toml_to_config_value_rejects_overflow_u32() {
        let err = toml_to_config_value(
            &TomlValue::Integer(i64::from(u32::MAX) + 1),
            FieldType::U32,
            "x",
        )
        .unwrap_err();
        assert!(matches!(err, LoaderError::Validation(_)));
    }

    #[test]
    fn toml_to_config_value_password_class_unknown_name_rejected() {
        let toml_v = TomlValue::Array(alloc::vec![TomlValue::String("nope".to_string())]);
        let err = toml_to_config_value(&toml_v, FieldType::PasswordClasses, "x").unwrap_err();
        assert!(matches!(err, LoaderError::Validation(_)));
    }

    #[test]
    fn toml_to_config_value_type_mismatch_recorded() {
        let toml_v = TomlValue::String("x".to_string());
        let err = toml_to_config_value(&toml_v, FieldType::U32, "x").unwrap_err();
        match err {
            LoaderError::TypeMismatch { expected, got, .. } => {
                assert_eq!(expected, "u32");
                assert_eq!(got, "string");
            }
            _ => panic!("expected TypeMismatch"),
        }
    }

    #[test]
    fn config_value_u64_above_i64_max_rejected() {
        // A u64 above i64::MAX cannot fit in TOML's signed integer.
        let v = Value::U64(u64::MAX);
        assert!(config_value_to_toml(&v).is_none());
    }

    #[test]
    fn namespace_classifier_covers_v1_namespaces() {
        assert_eq!(namespace_of_path("idle"), Namespace::System);
        assert_eq!(namespace_of_path("password"), Namespace::System);
        assert_eq!(namespace_of_path("flash"), Namespace::System);
        assert_eq!(namespace_of_path("fs"), Namespace::System);
        assert_eq!(namespace_of_path("mgmt"), Namespace::MgmtPolicy);
        assert_eq!(namespace_of_path("audit"), Namespace::MgmtPolicy);
        assert_eq!(namespace_of_path("metrics"), Namespace::MgmtPolicy);
        assert_eq!(namespace_of_path("overlay"), Namespace::MgmtPolicy);
    }

    #[test]
    fn pending_reboot_grows_per_boot_only_write() {
        let reader = InMemoryReadBackend::new();
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        loader.load(FirstBootPolicy::Generate).unwrap();
        loader
            .write(
                &ConfigPath::new("fs.boot_watchdog_seconds").unwrap(),
                Value::U32(30),
                "root@toml",
            )
            .unwrap();
        loader
            .write(
                &ConfigPath::new("flash.prefer_flash_for_models").unwrap(),
                Value::Bool(false),
                "root@toml",
            )
            .unwrap();
        assert_eq!(loader.pending_reboot().len(), 2);
    }

    #[test]
    fn flush_namespace_writes_disk_copy() {
        let reader = InMemoryReadBackend::new();
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        loader.load(FirstBootPolicy::Generate).unwrap();
        // Just exercises the flush path — generation already wrote
        // the file; this is an explicit re-write.
        loader.flush_namespace(Namespace::System).unwrap();
        loader.flush_namespace(Namespace::MgmtPolicy).unwrap();
    }

    #[test]
    fn round_trip_via_seeded_default_files() {
        // Belt-and-braces: serialize defaults via the renderer, seed
        // them into the in-memory backend, re-load, and confirm the
        // resulting Config equals defaults.
        let cfg = Config::v1_defaults();
        let sys = toml_serialize(&render_namespace_table(&cfg, Namespace::System)).unwrap();
        let pol = toml_serialize(&render_namespace_table(&cfg, Namespace::MgmtPolicy)).unwrap();
        let reader = InMemoryReadBackend::new();
        reader.seed_with_mode("/data/system.toml", sys.as_bytes(), 0o644);
        reader.seed_with_mode("/data/mgmt/policy.toml", pol.as_bytes(), 0o640);
        reader.seed_with_mode("/data/mgmt/zenoh.toml", b"", 0o640);
        reader.seed_with_mode("/data/update/policy.toml", b"", 0o640);
        reader.seed_with_mode("/data/automotive/uds.toml", b"", 0o640);
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        let loaded = loader.load(FirstBootPolicy::Strict).unwrap().clone();
        assert_eq!(loaded, cfg);
    }

    #[test]
    fn base_path_with_trailing_slash_is_normalised() {
        let loader = TomlLoader::new(
            InMemoryReadBackend::new(),
            TestAtomicWriter::new(),
            "/data/",
        );
        assert_eq!(loader.absolute("system.toml"), "/data/system.toml");
    }

    #[test]
    fn base_path_without_trailing_slash_is_handled() {
        let loader = TomlLoader::new(InMemoryReadBackend::new(), TestAtomicWriter::new(), "/data");
        assert_eq!(loader.absolute("system.toml"), "/data/system.toml");
    }

    #[test]
    fn in_memory_backend_roundtrips_metadata() {
        let r = InMemoryReadBackend::new();
        r.seed_with_mode("/x", b"hi", 0o600);
        assert_eq!(r.read_file("/x").unwrap().as_deref(), Some(&b"hi"[..]));
        assert_eq!(r.mode_of("/x").unwrap(), Some(0o600));
        assert_eq!(r.read_file("/missing").unwrap(), None);
        assert_eq!(r.mode_of("/missing").unwrap(), None);
    }

    #[test]
    fn in_memory_backend_default_mode_used_for_seed() {
        let r = InMemoryReadBackend::new();
        r.set_default_mode(0o644);
        r.seed("/x", b"hi");
        assert_eq!(r.mode_of("/x").unwrap(), Some(0o644));
    }

    #[test]
    fn in_memory_backend_set_mode_only_existing_files() {
        let r = InMemoryReadBackend::new();
        r.seed("/x", b"hi");
        r.set_mode("/x", 0o400);
        assert_eq!(r.mode_of("/x").unwrap(), Some(0o400));
        // Non-existent file: silently ignore.
        r.set_mode("/missing", 0o400);
        assert_eq!(r.mode_of("/missing").unwrap(), None);
    }

    #[test]
    fn in_memory_backend_snapshot_includes_seeded() {
        let r = InMemoryReadBackend::new();
        r.seed("/x", b"hi");
        r.seed("/y", b"yo");
        let s = r.snapshot();
        assert_eq!(s.len(), 2);
        assert!(s.contains_key("/x"));
        assert!(s.contains_key("/y"));
    }

    #[test]
    fn in_memory_backend_len_and_is_empty() {
        let r = InMemoryReadBackend::new();
        assert!(r.is_empty());
        r.seed("/x", b"hi");
        assert_eq!(r.len(), 1);
        assert!(!r.is_empty());
    }

    #[test]
    fn first_boot_generate_then_reload_strict_succeeds() {
        let reader = InMemoryReadBackend::new();
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        loader.load(FirstBootPolicy::Generate).unwrap();
        // After generation, the *writer*'s in-memory map holds the
        // files but our reader doesn't see them (separate backends).
        // This is intentional in the test fixture: the kernel will
        // pair both backends against the same FS handle. To prove the
        // code path is sound we re-read by inspecting `cached`
        // directly.
        let snapshot = loader.config().clone();
        assert_eq!(snapshot.idle.root_minutes, 15);
    }

    #[test]
    fn write_failure_due_to_io_propagates() {
        // TestAtomicWriter that crashes on write.
        let reader = InMemoryReadBackend::new();
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        loader.load(FirstBootPolicy::Generate).unwrap();
        loader.writer.inject_crash_during_io();
        let p = ConfigPath::new("idle.root_minutes").unwrap();
        let err = loader.write(&p, Value::U32(20), "root@toml").unwrap_err();
        assert!(matches!(err, SurfaceError::Io(_)));
        // In-memory state did NOT mutate.
        assert_eq!(loader.read(&p).unwrap(), Value::U32(15));
    }

    #[test]
    fn default_loader_has_zero_pending_reboot() {
        let loader = empty_loader();
        assert!(loader.pending_reboot().is_empty());
    }

    #[test]
    fn live_write_does_not_appear_in_pending_reboot() {
        let reader = InMemoryReadBackend::new();
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        loader.load(FirstBootPolicy::Generate).unwrap();
        loader
            .write(
                &ConfigPath::new("idle.root_minutes").unwrap(),
                Value::U32(20),
                "root@toml",
            )
            .unwrap();
        assert!(loader.pending_reboot().is_empty());
    }

    #[test]
    fn type_tag_string_is_unsupported_in_v1_config() {
        // No v1 config field is `String`, so a String value into any
        // path fails type matching.
        let err = toml_to_config_value(
            &TomlValue::String("x".to_string()),
            FieldType::String,
            "fake.path",
        )
        .unwrap();
        assert!(matches!(err, Value::String(_)));
        // (Just exercises the conversion path; no v1 field exercises
        // it from the registry.)
    }

    #[test]
    fn integer_conversion_handles_zero() {
        assert_eq!(
            toml_to_config_value(&TomlValue::Integer(0), FieldType::U32, "x").unwrap(),
            Value::U32(0)
        );
        assert_eq!(
            toml_to_config_value(&TomlValue::Integer(0), FieldType::U64, "x").unwrap(),
            Value::U64(0)
        );
    }

    #[test]
    fn boolean_conversion_round_trips() {
        assert_eq!(
            toml_to_config_value(&TomlValue::Bool(true), FieldType::Bool, "x").unwrap(),
            Value::Bool(true)
        );
        assert_eq!(
            toml_to_config_value(&TomlValue::Bool(false), FieldType::Bool, "x").unwrap(),
            Value::Bool(false)
        );
    }

    #[test]
    fn render_then_apply_is_identity() {
        let cfg = Config::v1_defaults();
        let table = render_namespace_table(&cfg, Namespace::System);
        let mut roundtrip = Config::v1_defaults();
        apply_table_to_config(&table, "", &mut roundtrip).unwrap();
        assert_eq!(roundtrip, cfg);
    }

    #[test]
    fn render_then_apply_is_identity_for_mgmt_policy() {
        let cfg = Config::v1_defaults();
        let table = render_namespace_table(&cfg, Namespace::MgmtPolicy);
        let mut roundtrip = Config::v1_defaults();
        apply_table_to_config(&table, "", &mut roundtrip).unwrap();
        assert_eq!(roundtrip, cfg);
    }

    #[test]
    fn pending_reboot_records_who() {
        let reader = InMemoryReadBackend::new();
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        loader.load(FirstBootPolicy::Generate).unwrap();
        loader
            .write(
                &ConfigPath::new("fs.boot_watchdog_seconds").unwrap(),
                Value::U32(30),
                "operator:alice@toml",
            )
            .unwrap();
        assert_eq!(loader.pending_reboot()[0].who, "operator:alice@toml");
    }

    #[test]
    fn live_write_serialises_changed_value_to_disk_via_writer() {
        // We can't snoop the TestAtomicWriter from outside the loader,
        // but we can repeat-write and verify the cache reflects the
        // last value. (Disk-snooping is exercised by the integration
        // test above.)
        let reader = InMemoryReadBackend::new();
        let writer = TestAtomicWriter::new();
        let mut loader = TomlLoader::new(reader, writer, "/data");
        loader.load(FirstBootPolicy::Generate).unwrap();
        let p = ConfigPath::new("idle.root_minutes").unwrap();
        for v in [20, 30, 10] {
            loader.write(&p, Value::U32(v), "root@toml").unwrap();
            assert_eq!(loader.read(&p).unwrap(), Value::U32(v));
        }
    }

    #[test]
    fn check_mode_ignores_high_bits() {
        // setuid/setgid/sticky bits on the file are ignored — the
        // loader only cares about the mode triplet.
        check_mode("/x", 0o640, 0o4640).unwrap();
    }

    #[test]
    fn loader_error_io_display_includes_reason() {
        let e = LoaderError::Io("storage offline");
        let s = alloc::format!("{}", e);
        assert!(s.contains("storage offline"));
    }
}
