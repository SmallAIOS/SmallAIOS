// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Virtual filesystem for the POSIX compatibility layer.
//!
//! Provides a minimal VFS with fixed mount points:
//! - `/models/`  — ONNX model files (loaded at boot, in-memory tree)
//! - `/config/`  — Runtime configuration (in-memory tree)
//! - `/dev/`     — Device nodes (null, urandom; in-memory tree)
//! - `/proc/self/` — Process introspection (in-memory tree)
//! - `/flash/`   — Raw-flash littlefs mount (opt-in via the `fs-flash`
//!   cargo feature; the **first read/write on-disk mount** in this VFS).
//!
//! The default in-memory tree is read-only; write attempts on it
//! return `EROFS`. The `/flash/` substrate is read/write when the
//! feature is enabled and a backing [`smallaios_fs::flash::littlefs::LittleFs`]
//! is registered via [`MountTable::mount_flash`].
//!
//! Existing tests that exercise the in-memory tree continue to pass
//! byte-for-byte; the `/flash/` mount is layered on top via the
//! [`MountedFs`] enum, so the in-memory path goes through the same
//! `MountTable::resolve` fast path it always did.
//!
//! ## Architecture note
//!
//! Adding `posix → fs` here is the second accepted Layer-1 → Layer-1
//! edge in SmallAIOS (the first being `posix → net` for sockets). The
//! dep is feature-gated so the default crate graph is unchanged when
//! `fs-flash` is off.

use crate::errno::Errno;
use crate::fd::FileStat;

// ─── Constants ───────────────────────────────────────────────────────────────

/// Maximum number of VFS nodes in the tree.
pub const MAX_VFS_NODES: usize = 64;

/// Maximum number of children per directory node.
pub const MAX_CHILDREN: usize = 16;

// ─── VFS Node Types ──────────────────────────────────────────────────────────

/// The kind of a VFS node.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VfsNodeKind {
    /// A directory that may contain children.
    Directory,
    /// A regular file with optional static content.
    RegularFile,
    /// A character device (e.g., /dev/null, /dev/urandom).
    CharDevice,
}

/// A single node in the virtual filesystem tree.
///
/// Nodes are stored in a flat array inside [`VfsTree`]. Directory nodes
/// reference their children by index into that array.
#[derive(Debug, Clone, PartialEq)]
pub struct VfsNode {
    /// Node name (leaf component, not the full path).
    pub name: &'static str,
    /// What kind of node this is.
    pub kind: VfsNodeKind,
    /// File size in bytes (0 for directories and devices).
    pub size: u64,
    /// Virtual inode number (unique within the VFS).
    pub inode: u64,
    /// Indices of child nodes in the [`VfsTree`] array (directories only).
    pub children: [Option<usize>; MAX_CHILDREN],
    /// Number of valid children.
    pub num_children: usize,
    /// Static file content (regular files only).
    pub data: Option<&'static [u8]>,
}

impl VfsNode {
    /// Create a new directory node.
    const fn dir(name: &'static str, inode: u64) -> Self {
        Self {
            name,
            kind: VfsNodeKind::Directory,
            size: 0,
            inode,
            children: [None; MAX_CHILDREN],
            num_children: 0,
            data: None,
        }
    }

    /// Create a new regular file node with optional static content.
    const fn file(name: &'static str, inode: u64, data: Option<&'static [u8]>) -> Self {
        let size = match data {
            Some(d) => d.len() as u64,
            None => 0,
        };
        Self {
            name,
            kind: VfsNodeKind::RegularFile,
            size,
            inode,
            children: [None; MAX_CHILDREN],
            num_children: 0,
            data,
        }
    }

    /// Create a new character device node.
    const fn char_dev(name: &'static str, inode: u64) -> Self {
        Self {
            name,
            kind: VfsNodeKind::CharDevice,
            size: 0,
            inode,
            children: [None; MAX_CHILDREN],
            num_children: 0,
            data: None,
        }
    }

    /// Add a child index to this directory node.
    ///
    /// # Panics
    ///
    /// Panics if the node already has `MAX_CHILDREN` children.
    fn add_child(&mut self, child_idx: usize) {
        assert!(
            self.num_children < MAX_CHILDREN,
            "VfsNode: too many children"
        );
        self.children[self.num_children] = Some(child_idx);
        self.num_children += 1;
    }

    /// Convert this node to a [`FileStat`].
    pub fn to_file_stat(&self) -> FileStat {
        match self.kind {
            VfsNodeKind::Directory => FileStat::directory(self.inode),
            VfsNodeKind::RegularFile => FileStat::regular(self.size, self.inode),
            VfsNodeKind::CharDevice => FileStat::char_device(self.inode),
        }
    }
}

// ─── Auth-protected path guard ───────────────────────────────────────────────

/// Path prefix under which the auth subsystem stores its shadow file
/// and any future credential blobs. The `/data/` mount itself does not
/// yet exist in the VFS (it lands with `embedded-filesystem-v1` Phase
/// 6), but [`is_auth_protected_path`] is wired into the lookup *now*
/// so userspace cannot reach the shadow file the moment the mount is
/// added.
///
/// Spec: `management-login-v1` Phase 3 ("VFS auth-path guard").
pub const AUTH_PROTECTED_PREFIX: &str = "/data/auth/";

/// Returns `true` iff `path` resolves anywhere under
/// [`AUTH_PROTECTED_PREFIX`]. Trailing slashes and the directory
/// itself (`/data/auth`) are also rejected.
///
/// Path normalization is intentionally simple — paths must begin with
/// `/data/auth` followed by either end-of-string or another `/`. This
/// prevents trivial bypass attempts like `/data/auth.bak/...` from
/// matching.
pub fn is_auth_protected_path(path: &str) -> bool {
    // Match the directory itself (with or without trailing slash).
    if path == "/data/auth" || path == "/data/auth/" {
        return true;
    }
    // Match any path beneath the directory.
    path.starts_with(AUTH_PROTECTED_PREFIX)
}

// ─── VFS Tree ────────────────────────────────────────────────────────────────

/// The virtual filesystem tree.
///
/// All nodes are stored in a flat, fixed-size array. The root node is
/// always at index 0.
#[derive(Debug)]
pub struct VfsTree {
    /// Node storage.
    nodes: [Option<VfsNode>; MAX_VFS_NODES],
    /// Number of nodes currently in the tree.
    count: usize,
}

impl Default for VfsTree {
    fn default() -> Self {
        Self::new()
    }
}

impl VfsTree {
    /// Create an empty VFS tree.
    pub fn new() -> Self {
        Self {
            nodes: core::array::from_fn(|_| None),
            count: 0,
        }
    }

    /// Insert a node into the tree. Returns the index of the inserted node.
    fn insert(&mut self, node: VfsNode) -> usize {
        assert!(self.count < MAX_VFS_NODES, "VfsTree: capacity exceeded");
        let idx = self.count;
        self.nodes[idx] = Some(node);
        self.count += 1;
        idx
    }

    /// Return the number of nodes in the tree.
    pub fn node_count(&self) -> usize {
        self.count
    }

    /// Look up a node by its absolute path.
    ///
    /// Path must start with `/`. Components are separated by `/`.
    /// Trailing slashes are tolerated.
    ///
    /// # Auth defense
    ///
    /// Per `management-login-v1` Phase 3, paths under `/data/auth/`
    /// are denied with `EACCES` regardless of the lookup mode. This
    /// fires *before* the actual VFS walk so the guard is preserved
    /// once the `/data/` mount lands in `embedded-filesystem-v1`
    /// Phase 6: the shadow file is reachable only via the kernel's
    /// auth syscalls, never through the userspace VFS surface.
    pub fn lookup(&self, path: &str) -> Result<&VfsNode, Errno> {
        if !path.starts_with('/') {
            return Err(Errno::EINVAL);
        }

        if is_auth_protected_path(path) {
            return Err(Errno::EACCES);
        }

        // Root lookup
        let root = self.nodes[0].as_ref().ok_or(Errno::ENOENT)?;
        if path == "/" {
            return Ok(root);
        }

        // Walk path components
        let trimmed = path.trim_end_matches('/');
        let mut current_node = root;

        for component in trimmed[1..].split('/') {
            if component.is_empty() {
                continue;
            }
            if current_node.kind != VfsNodeKind::Directory {
                return Err(Errno::ENOENT);
            }
            current_node = self.resolve_child(current_node, component)?;
        }

        Ok(current_node)
    }

    /// Resolve a single path component within a directory node.
    ///
    /// Searches the directory's children for a node whose name matches
    /// `component`. Returns `ENOENT` if no match is found.
    fn resolve_child<'a>(
        &'a self,
        parent: &'a VfsNode,
        component: &str,
    ) -> Result<&'a VfsNode, Errno> {
        for i in 0..parent.num_children {
            let child_idx = match parent.children[i] {
                Some(idx) => idx,
                None => continue,
            };
            let child = match &self.nodes[child_idx] {
                Some(node) => node,
                None => continue,
            };
            if child.name == component {
                return Ok(child);
            }
        }
        Err(Errno::ENOENT)
    }

    /// Stat a file by its absolute path.
    pub fn stat(&self, path: &str) -> Result<FileStat, Errno> {
        let node = self.lookup(path)?;
        Ok(node.to_file_stat())
    }

    /// Read data from a node identified by inode number.
    ///
    /// - Regular files: reads from static data at the given offset.
    /// - `/dev/null` (inode 2): always returns 0 bytes (EOF).
    /// - `/dev/urandom` (inode 3): returns `ENOSYS` (needs CSPRNG).
    /// - Directories: returns `EINVAL`.
    pub fn read(&self, inode: u64, offset: u64, buf: &mut [u8]) -> Result<usize, Errno> {
        let node = self.find_by_inode(inode)?;

        match node.kind {
            VfsNodeKind::Directory => Err(Errno::EINVAL),
            VfsNodeKind::CharDevice => {
                // /dev/null — reads return 0 bytes (EOF)
                if node.name == "null" {
                    return Ok(0);
                }
                // /dev/urandom — not yet implemented
                if node.name == "urandom" {
                    return Err(Errno::ENOSYS);
                }
                // Unknown char device
                Err(Errno::ENOENT)
            }
            VfsNodeKind::RegularFile => {
                match node.data {
                    Some(data) => {
                        let offset = offset as usize;
                        if offset >= data.len() {
                            return Ok(0); // EOF
                        }
                        let available = data.len() - offset;
                        let to_copy = if buf.len() < available {
                            buf.len()
                        } else {
                            available
                        };
                        buf[..to_copy].copy_from_slice(&data[offset..offset + to_copy]);
                        Ok(to_copy)
                    }
                    None => Ok(0), // Empty file
                }
            }
        }
    }

    /// Attempt to write — always fails with `EROFS` (read-only filesystem).
    pub fn write(&self, _inode: u64, _offset: u64, _buf: &[u8]) -> Result<usize, Errno> {
        Err(Errno::EROFS)
    }

    /// Find a node by its inode number.
    fn find_by_inode(&self, inode: u64) -> Result<&VfsNode, Errno> {
        for i in 0..self.count {
            if let Some(node) = &self.nodes[i] {
                if node.inode == inode {
                    return Ok(node);
                }
            }
        }
        Err(Errno::ENOENT)
    }
}

// ─── Default VFS Tree ────────────────────────────────────────────────────────

/// Static content for `/proc/self/maps` (empty memory map).
const PROC_SELF_MAPS_DATA: &[u8] = b"";

/// Build the default VFS tree with standard mount points.
///
/// Tree structure:
/// ```text
/// /                     (inode 0, directory)
/// +-- dev/              (inode 1, directory)
/// |   +-- null          (inode 2, char device)
/// |   +-- urandom       (inode 3, char device)
/// +-- models/           (inode 4, directory)
/// +-- config/           (inode 5, directory)
/// +-- proc/             (inode 6, directory)
///     +-- self/         (inode 7, directory)
///         +-- maps      (inode 8, regular file, empty)
/// ```
pub fn build_default_vfs() -> VfsTree {
    let mut tree = VfsTree::new();

    // 0: /
    let root_idx = tree.insert(VfsNode::dir("", 0));

    // 1: /dev
    let dev_idx = tree.insert(VfsNode::dir("dev", 1));
    // 2: /dev/null
    let null_idx = tree.insert(VfsNode::char_dev("null", 2));
    // 3: /dev/urandom
    let urandom_idx = tree.insert(VfsNode::char_dev("urandom", 3));

    // 4: /models
    let models_idx = tree.insert(VfsNode::dir("models", 4));

    // 5: /config
    let config_idx = tree.insert(VfsNode::dir("config", 5));

    // 6: /proc
    let proc_idx = tree.insert(VfsNode::dir("proc", 6));
    // 7: /proc/self
    let proc_self_idx = tree.insert(VfsNode::dir("self", 7));
    // 8: /proc/self/maps
    let maps_idx = tree.insert(VfsNode::file("maps", 8, Some(PROC_SELF_MAPS_DATA)));

    // Wire up parent-child relationships
    // root -> dev, models, config, proc
    tree.nodes[root_idx].as_mut().unwrap().add_child(dev_idx);
    tree.nodes[root_idx].as_mut().unwrap().add_child(models_idx);
    tree.nodes[root_idx].as_mut().unwrap().add_child(config_idx);
    tree.nodes[root_idx].as_mut().unwrap().add_child(proc_idx);

    // dev -> null, urandom
    tree.nodes[dev_idx].as_mut().unwrap().add_child(null_idx);
    tree.nodes[dev_idx].as_mut().unwrap().add_child(urandom_idx);

    // proc -> self
    tree.nodes[proc_idx]
        .as_mut()
        .unwrap()
        .add_child(proc_self_idx);

    // proc/self -> maps
    tree.nodes[proc_self_idx]
        .as_mut()
        .unwrap()
        .add_child(maps_idx);

    tree
}

// ─── Mount table (in-memory + /flash/ littlefs) ─────────────────────────────

/// Backing filesystem for one VFS mount slot.
///
/// The in-memory variant keeps the existing static-tree semantics
/// untouched. The `LittleFs` variant routes opens/reads/writes/
/// renames/etc. under the mount path to the littlefs runtime in
/// `smallaios-fs`.
///
/// Phase 3 of `embedded-flash-fs-v1` lands the wiring; future on-disk
/// mounts (squashfs `/models/`, F2FS `/data/`) plug in by adding new
/// variants here.
///
/// The `InMemory` variant is intentionally the dominant code path (it
/// holds the existing static tree); boxing it would force every
/// in-memory lookup through an extra indirection for zero benefit.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
pub enum MountedFs {
    /// Static in-memory tree (root, `/dev/`, `/proc/self/`, `/models/`,
    /// `/config/`).
    InMemory(VfsTree),
    /// littlefs-backed read/write mount. Stored as a tag here so the
    /// concrete mount handle can live on the kernel side (the handle's
    /// borrow lifetime is tied to the underlying [`FlashDevice`]).
    /// Phase 3 surfaces this as a marker variant so the path resolver
    /// knows where to dispatch; a future wiring pass will plumb the
    /// `LittleFs<'_, D>` handle through (the kernel currently owns it).
    #[cfg(feature = "fs-flash")]
    LittleFs(FlashMount),
}

/// Marker handle for a `/flash/` mount point.
///
/// v1 records only the mount-point path and a sticky `fadvise` hint;
/// the concrete `LittleFs` runtime handle is dispatched separately
/// because its lifetime is tied to the underlying [`FlashDevice`] which
/// the kernel boot sequencer owns. The split mirrors how Linux's VFS
/// keeps the dentry tree and the per-superblock state separate.
#[cfg(feature = "fs-flash")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashMount {
    /// Mount-point absolute path (always `/flash` in v1).
    pub path: &'static str,
    /// Most recent fadvise hint applied to this mount.
    pub fadvise: FadviseHint,
    /// `true` once the kernel has wired a `LittleFs` runtime instance
    /// to this mount slot. v1 toggles this from
    /// [`MountTable::mount_flash`]; future revisions will store the
    /// runtime handle directly.
    pub bound: bool,
}

/// fadvise hint accepted by `posix_fadvise` on `/flash/`-backed file
/// descriptors. Mirrors the [`smallaios_fs::flash::littlefs::FadviseHint`]
/// enum so the two crates can convert losslessly.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum FadviseHint {
    /// No hint set (mount default).
    #[default]
    Normal,
    /// Reads will be sequential.
    Sequential,
    /// Reads will be at random offsets.
    Random,
    /// Caller will need this data soon.
    WillNeed,
    /// Caller is done with this data.
    DontNeed,
    /// Coalesce short appends (SmallAIOS extension used by audit-log
    /// writers to batch metadata-pair commits).
    BatchWrite,
}

#[cfg(feature = "fs-flash")]
impl From<FadviseHint> for smallaios_fs::flash::littlefs::FadviseHint {
    fn from(value: FadviseHint) -> Self {
        match value {
            FadviseHint::Normal => Self::Normal,
            FadviseHint::Sequential => Self::Sequential,
            FadviseHint::Random => Self::Random,
            FadviseHint::WillNeed => Self::WillNeed,
            FadviseHint::DontNeed => Self::DontNeed,
            FadviseHint::BatchWrite => Self::BatchWrite,
        }
    }
}

#[cfg(feature = "fs-flash")]
impl From<smallaios_fs::flash::littlefs::FadviseHint> for FadviseHint {
    fn from(value: smallaios_fs::flash::littlefs::FadviseHint) -> Self {
        use smallaios_fs::flash::littlefs::FadviseHint as F;
        match value {
            F::Normal => Self::Normal,
            F::Sequential => Self::Sequential,
            F::Random => Self::Random,
            F::WillNeed => Self::WillNeed,
            F::DontNeed => Self::DontNeed,
            F::BatchWrite => Self::BatchWrite,
        }
    }
}

/// VFS mount table.
///
/// Holds the default in-memory tree plus an optional `/flash/` slot.
/// All path resolution under `/flash/` routes through the slot when
/// the `fs-flash` cargo feature is on AND a runtime is bound; otherwise
/// `/flash/` returns `ENOENT`.
///
/// The default in-memory tree is unchanged from previous SmallAIOS
/// revisions: existing tests that consume `build_default_vfs()`
/// continue to pass byte-for-byte.
#[derive(Debug)]
pub struct MountTable {
    in_memory: VfsTree,
    #[cfg(feature = "fs-flash")]
    flash: Option<FlashMount>,
}

impl Default for MountTable {
    fn default() -> Self {
        Self::new()
    }
}

impl MountTable {
    /// Build a mount table with the default in-memory tree and no
    /// `/flash/` slot.
    pub fn new() -> Self {
        Self {
            in_memory: build_default_vfs(),
            #[cfg(feature = "fs-flash")]
            flash: None,
        }
    }

    /// Read-only access to the in-memory tree.
    pub fn in_memory(&self) -> &VfsTree {
        &self.in_memory
    }

    /// `true` iff the in-memory tree contains the requested path.
    pub fn has_in_memory_path(&self, path: &str) -> bool {
        self.in_memory.lookup(path).is_ok()
    }

    /// Whether a `/flash/` mount slot is currently bound. With the
    /// `fs-flash` feature off, always returns `false`.
    pub fn has_flash(&self) -> bool {
        #[cfg(feature = "fs-flash")]
        {
            matches!(&self.flash, Some(m) if m.bound)
        }
        #[cfg(not(feature = "fs-flash"))]
        {
            false
        }
    }

    /// `true` iff the VFS has BOTH an `/data/` (F2FS) mount AND a
    /// `/flash/` (littlefs) mount available.
    ///
    /// v1 only knows about `/flash/`. The `/data/` half is wired by
    /// `embedded-filesystem-v1` Phase 6; until that lands this method
    /// always returns `false`.
    pub fn has_both(&self) -> bool {
        false
    }

    /// Mount the `/flash/` slot. Returns `EBUSY` if a slot is already
    /// bound; returns `ENOSYS` when the `fs-flash` feature is off.
    ///
    /// v1 only records the marker. The concrete `LittleFs` runtime
    /// handle is owned by the kernel boot sequencer (via
    /// [`smallaios_fs::flash::mount_or_format`]) and dispatched outside
    /// the VFS; this slot is what tells path resolution to send opens
    /// under `/flash/` to that runtime.
    #[cfg(feature = "fs-flash")]
    pub fn mount_flash(&mut self) -> Result<(), Errno> {
        if let Some(m) = &self.flash {
            if m.bound {
                return Err(Errno::EBUSY);
            }
        }
        self.flash = Some(FlashMount {
            path: "/flash",
            fadvise: FadviseHint::Normal,
            bound: true,
        });
        Ok(())
    }

    /// `mount_flash` stub for builds without the `fs-flash` feature.
    /// Always returns `ENOSYS`.
    #[cfg(not(feature = "fs-flash"))]
    pub fn mount_flash(&mut self) -> Result<(), Errno> {
        Err(Errno::ENOSYS)
    }

    /// Unmount the `/flash/` slot.
    #[cfg(feature = "fs-flash")]
    pub fn unmount_flash(&mut self) {
        self.flash = None;
    }

    /// Set the fadvise hint for the `/flash/` mount.
    ///
    /// Returns `ENOENT` if `/flash/` is not currently mounted; returns
    /// `Ok(())` once the hint is recorded. The hint is advisory; v1
    /// stores it for inspection (and for the future cache layer to
    /// honor) but takes no behavioral action today. Per POSIX, any
    /// unsupported hint is accepted as a no-op rather than rejected.
    #[cfg(feature = "fs-flash")]
    pub fn fadvise(&mut self, path: &str, hint: FadviseHint) -> Result<(), Errno> {
        if !path.starts_with("/flash") {
            // Per spec, unsupported hints under non-flash paths are a
            // no-op `Ok(0)`.
            return Ok(());
        }
        let slot = self.flash.as_mut().ok_or(Errno::ENOENT)?;
        if !slot.bound {
            return Err(Errno::ENOENT);
        }
        slot.fadvise = hint;
        Ok(())
    }

    /// `fadvise` stub for builds without the `fs-flash` feature.
    /// Always returns `Ok(())` (POSIX no-op semantics).
    #[cfg(not(feature = "fs-flash"))]
    pub fn fadvise(&mut self, _path: &str, _hint: FadviseHint) -> Result<(), Errno> {
        Ok(())
    }

    /// Query the current fadvise hint for `/flash/`. Returns `None`
    /// when the slot isn't mounted.
    #[cfg(feature = "fs-flash")]
    pub fn fadvise_hint(&self) -> Option<FadviseHint> {
        self.flash.as_ref().map(|s| s.fadvise)
    }

    /// `fadvise_hint` stub for builds without the `fs-flash` feature.
    #[cfg(not(feature = "fs-flash"))]
    pub fn fadvise_hint(&self) -> Option<FadviseHint> {
        None
    }

    /// Resolve `path` to a [`MountResolution`].
    ///
    /// Paths that fall under `/flash/` are routed to the littlefs slot
    /// when bound; everything else (and `/flash/` when unbound) is
    /// dispatched to the in-memory tree.
    pub fn resolve<'a>(&'a self, path: &'a str) -> MountResolution<'a> {
        #[cfg(feature = "fs-flash")]
        {
            if let Some(slot) = &self.flash {
                if slot.bound && path_under(path, slot.path) {
                    let sub = path_after(path, slot.path);
                    return MountResolution::Flash {
                        slot: *slot,
                        subpath: sub,
                    };
                }
            }
        }
        // Fall through to the in-memory tree.
        match self.in_memory.lookup(path) {
            Ok(node) => MountResolution::InMemory(node),
            Err(e) => MountResolution::NotFound(e),
        }
    }
}

/// Outcome of resolving a path against the [`MountTable`].
#[derive(Debug, Clone)]
pub enum MountResolution<'a> {
    /// The path resolved into the static in-memory tree.
    InMemory(&'a VfsNode),
    /// The path falls under the `/flash/` mount; `subpath` is the
    /// remaining portion (always begins with `/`, e.g. `/secrets/key`).
    /// The kernel dispatches this to the bound `LittleFs` runtime.
    #[cfg(feature = "fs-flash")]
    Flash {
        slot: FlashMount,
        subpath: alloc::string::String,
    },
    /// Path didn't match any mount.
    NotFound(Errno),
}

/// `true` iff `path` lies under the mount-point `mp` (`mp` must NOT
/// have a trailing slash, e.g. `"/flash"`).
#[cfg(feature = "fs-flash")]
fn path_under(path: &str, mp: &str) -> bool {
    if path == mp {
        return true;
    }
    if let Some(rest) = path.strip_prefix(mp) {
        return rest.starts_with('/');
    }
    false
}

/// Return the portion of `path` *after* the mount-point `mp`. For
/// `path = "/flash/secrets/key"`, `mp = "/flash"` returns
/// `"/secrets/key"`. For `path == mp` returns `"/"`.
#[cfg(feature = "fs-flash")]
fn path_after(path: &str, mp: &str) -> alloc::string::String {
    use alloc::string::ToString;
    if path == mp {
        return "/".to_string();
    }
    let rest = path.strip_prefix(mp).unwrap_or(path);
    if rest.is_empty() {
        return "/".to_string();
    }
    rest.to_string()
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tree_has_expected_node_count() {
        let vfs = build_default_vfs();
        // root, dev, null, urandom, models, config, proc, self, maps = 9
        assert_eq!(vfs.node_count(), 9);
    }

    #[test]
    fn lookup_root() {
        let vfs = build_default_vfs();
        let node = vfs.lookup("/").unwrap();
        assert_eq!(node.inode, 0);
        assert_eq!(node.kind, VfsNodeKind::Directory);
    }

    #[test]
    fn lookup_dev_null() {
        let vfs = build_default_vfs();
        let node = vfs.lookup("/dev/null").unwrap();
        assert_eq!(node.inode, 2);
        assert_eq!(node.kind, VfsNodeKind::CharDevice);
        assert_eq!(node.name, "null");
    }

    #[test]
    fn lookup_dev_urandom() {
        let vfs = build_default_vfs();
        let node = vfs.lookup("/dev/urandom").unwrap();
        assert_eq!(node.inode, 3);
        assert_eq!(node.kind, VfsNodeKind::CharDevice);
    }

    #[test]
    fn lookup_proc_self_maps() {
        let vfs = build_default_vfs();
        let node = vfs.lookup("/proc/self/maps").unwrap();
        assert_eq!(node.inode, 8);
        assert_eq!(node.kind, VfsNodeKind::RegularFile);
    }

    #[test]
    fn lookup_nonexistent_returns_enoent() {
        let vfs = build_default_vfs();
        assert_eq!(vfs.lookup("/nonexistent"), Err(Errno::ENOENT));
        assert_eq!(vfs.lookup("/dev/zero"), Err(Errno::ENOENT));
        assert_eq!(vfs.lookup("/proc/self/status"), Err(Errno::ENOENT));
    }

    #[test]
    fn lookup_invalid_path_returns_einval() {
        let vfs = build_default_vfs();
        assert_eq!(vfs.lookup("no_leading_slash"), Err(Errno::EINVAL));
    }

    #[test]
    fn lookup_trailing_slash_tolerated() {
        let vfs = build_default_vfs();
        let node = vfs.lookup("/dev/").unwrap();
        assert_eq!(node.inode, 1);
        assert_eq!(node.kind, VfsNodeKind::Directory);
    }

    #[test]
    fn stat_directory() {
        let vfs = build_default_vfs();
        let st = vfs.stat("/models").unwrap();
        assert_eq!(st.ino, 4);
        assert_eq!(st.mode & FileStat::S_IFDIR, FileStat::S_IFDIR);
        assert_eq!(st.nlink, 2);
    }

    #[test]
    fn stat_char_device() {
        let vfs = build_default_vfs();
        let st = vfs.stat("/dev/null").unwrap();
        assert_eq!(st.ino, 2);
        assert_eq!(st.mode & FileStat::S_IFCHR, FileStat::S_IFCHR);
    }

    #[test]
    fn read_dev_null_returns_zero() {
        let vfs = build_default_vfs();
        let mut buf = [0u8; 64];
        let n = vfs.read(2, 0, &mut buf).unwrap();
        assert_eq!(n, 0);
    }

    #[test]
    fn read_dev_urandom_returns_enosys() {
        let vfs = build_default_vfs();
        let mut buf = [0u8; 64];
        assert_eq!(vfs.read(3, 0, &mut buf), Err(Errno::ENOSYS));
    }

    #[test]
    fn read_proc_self_maps_returns_empty() {
        let vfs = build_default_vfs();
        let mut buf = [0u8; 64];
        let n = vfs.read(8, 0, &mut buf).unwrap();
        assert_eq!(n, 0); // PROC_SELF_MAPS_DATA is empty
    }

    #[test]
    fn write_returns_erofs() {
        let vfs = build_default_vfs();
        let buf = [0u8; 16];
        assert_eq!(vfs.write(2, 0, &buf), Err(Errno::EROFS));
        assert_eq!(vfs.write(8, 0, &buf), Err(Errno::EROFS));
    }

    #[test]
    fn read_regular_file_with_data() {
        // Build a custom tree with a file that has actual content
        let mut tree = VfsTree::new();
        let root_idx = tree.insert(VfsNode::dir("", 0));
        let file_data: &'static [u8] = b"Hello, SmallAIOS!";
        let file_idx = tree.insert(VfsNode::file("test.txt", 1, Some(file_data)));
        tree.nodes[root_idx].as_mut().unwrap().add_child(file_idx);

        // Read the full content
        let mut buf = [0u8; 64];
        let n = tree.read(1, 0, &mut buf).unwrap();
        assert_eq!(n, 17);
        assert_eq!(&buf[..n], b"Hello, SmallAIOS!");

        // Read with offset
        let n = tree.read(1, 7, &mut buf).unwrap();
        assert_eq!(n, 10);
        assert_eq!(&buf[..n], b"SmallAIOS!");

        // Read past end returns 0
        let n = tree.read(1, 100, &mut buf).unwrap();
        assert_eq!(n, 0);

        // Read into small buffer
        let mut small_buf = [0u8; 5];
        let n = tree.read(1, 0, &mut small_buf).unwrap();
        assert_eq!(n, 5);
        assert_eq!(&small_buf, b"Hello");
    }

    #[test]
    fn read_directory_returns_einval() {
        let vfs = build_default_vfs();
        let mut buf = [0u8; 64];
        assert_eq!(vfs.read(0, 0, &mut buf), Err(Errno::EINVAL));
    }

    #[test]
    fn read_nonexistent_inode_returns_enoent() {
        let vfs = build_default_vfs();
        let mut buf = [0u8; 64];
        assert_eq!(vfs.read(999, 0, &mut buf), Err(Errno::ENOENT));
    }

    // ─── MountTable tests ──────────────────────────────────────────────────

    #[test]
    fn mount_table_default_has_in_memory_paths() {
        let t = MountTable::new();
        assert!(t.has_in_memory_path("/"));
        assert!(t.has_in_memory_path("/dev/null"));
        assert!(t.has_in_memory_path("/models"));
        assert!(t.has_in_memory_path("/proc/self/maps"));
    }

    #[test]
    fn mount_table_default_has_no_flash() {
        let t = MountTable::new();
        assert!(!t.has_flash());
        assert!(!t.has_both());
    }

    #[test]
    fn mount_table_resolve_in_memory_path_returns_node() {
        let t = MountTable::new();
        let res = t.resolve("/dev/null");
        match res {
            MountResolution::InMemory(node) => assert_eq!(node.name, "null"),
            _ => panic!("expected InMemory resolution"),
        }
    }

    #[test]
    fn mount_table_resolve_unknown_path_returns_not_found() {
        let t = MountTable::new();
        let res = t.resolve("/nope");
        assert!(matches!(res, MountResolution::NotFound(Errno::ENOENT)));
    }

    #[test]
    #[cfg(not(feature = "fs-flash"))]
    fn mount_flash_returns_enosys_when_feature_off() {
        let mut t = MountTable::new();
        assert_eq!(t.mount_flash(), Err(Errno::ENOSYS));
    }

    #[test]
    #[cfg(not(feature = "fs-flash"))]
    fn fadvise_is_noop_when_feature_off() {
        let mut t = MountTable::new();
        assert_eq!(t.fadvise("/flash/foo", FadviseHint::Sequential), Ok(()));
        assert_eq!(t.fadvise_hint(), None);
    }

    #[cfg(feature = "fs-flash")]
    #[test]
    fn mount_flash_succeeds_when_feature_on() {
        let mut t = MountTable::new();
        assert!(!t.has_flash());
        t.mount_flash().unwrap();
        assert!(t.has_flash());
    }

    #[cfg(feature = "fs-flash")]
    #[test]
    fn mount_flash_double_mount_returns_ebusy() {
        let mut t = MountTable::new();
        t.mount_flash().unwrap();
        assert_eq!(t.mount_flash(), Err(Errno::EBUSY));
    }

    #[cfg(feature = "fs-flash")]
    #[test]
    fn unmount_flash_clears_slot() {
        let mut t = MountTable::new();
        t.mount_flash().unwrap();
        t.unmount_flash();
        assert!(!t.has_flash());
    }

    #[cfg(feature = "fs-flash")]
    #[test]
    fn fadvise_round_trip_when_flash_mounted() {
        let mut t = MountTable::new();
        t.mount_flash().unwrap();
        t.fadvise("/flash/audit.log", FadviseHint::Sequential)
            .unwrap();
        assert_eq!(t.fadvise_hint(), Some(FadviseHint::Sequential));
        t.fadvise("/flash/audit.log", FadviseHint::BatchWrite)
            .unwrap();
        assert_eq!(t.fadvise_hint(), Some(FadviseHint::BatchWrite));
    }

    #[cfg(feature = "fs-flash")]
    #[test]
    fn fadvise_unknown_path_is_noop_under_flash_off() {
        let mut t = MountTable::new();
        // No flash mounted yet — `/flash/...` should error with ENOENT
        // (the slot isn't bound) but `/models/...` should be a silent
        // no-op (POSIX permits unsupported hints to return Ok).
        assert_eq!(
            t.fadvise("/flash/foo", FadviseHint::Sequential),
            Err(Errno::ENOENT)
        );
        assert_eq!(t.fadvise("/models/foo", FadviseHint::Sequential), Ok(()));
    }

    #[cfg(feature = "fs-flash")]
    #[test]
    fn resolve_routes_flash_paths_to_flash_slot() {
        let mut t = MountTable::new();
        t.mount_flash().unwrap();
        let res = t.resolve("/flash/secrets/key");
        match res {
            MountResolution::Flash { slot, subpath } => {
                assert_eq!(slot.path, "/flash");
                assert_eq!(subpath, "/secrets/key");
                assert!(slot.bound);
            }
            _ => panic!("expected Flash resolution"),
        }
    }

    #[cfg(feature = "fs-flash")]
    #[test]
    fn resolve_unbound_flash_falls_through_to_not_found() {
        // `/flash/` is requested but no flash has been mounted: the
        // resolver should fall through to the in-memory tree, which
        // doesn't have `/flash/`, so the result is NotFound(ENOENT).
        let t = MountTable::new();
        let res = t.resolve("/flash/foo");
        assert!(matches!(res, MountResolution::NotFound(Errno::ENOENT)));
    }

    #[cfg(feature = "fs-flash")]
    #[test]
    fn resolve_in_memory_paths_unaffected_by_flash_mount() {
        let mut t = MountTable::new();
        t.mount_flash().unwrap();
        let res = t.resolve("/dev/null");
        match res {
            MountResolution::InMemory(node) => assert_eq!(node.name, "null"),
            _ => panic!("flash mount must not shadow in-memory paths"),
        }
    }

    #[cfg(feature = "fs-flash")]
    #[test]
    fn fadvise_hint_conversion_round_trip() {
        // Each `posix::vfs::FadviseHint` must convert losslessly into
        // the `smallaios_fs::flash::littlefs::FadviseHint` variant of
        // the same name.
        for h in [
            FadviseHint::Normal,
            FadviseHint::Sequential,
            FadviseHint::Random,
            FadviseHint::WillNeed,
            FadviseHint::DontNeed,
            FadviseHint::BatchWrite,
        ] {
            let lf: smallaios_fs::flash::littlefs::FadviseHint = h.into();
            let back: FadviseHint = lf.into();
            assert_eq!(h, back);
        }
    }

    #[cfg(feature = "fs-flash")]
    #[test]
    fn path_under_helpers_match_expected() {
        assert!(super::path_under("/flash", "/flash"));
        assert!(super::path_under("/flash/x", "/flash"));
        assert!(super::path_under("/flash/x/y", "/flash"));
        assert!(!super::path_under("/flashy", "/flash"));
        assert!(!super::path_under("/dev/null", "/flash"));
    }

    #[cfg(feature = "fs-flash")]
    #[test]
    fn path_after_helpers_match_expected() {
        assert_eq!(super::path_after("/flash", "/flash"), "/");
        assert_eq!(super::path_after("/flash/foo", "/flash"), "/foo");
        assert_eq!(super::path_after("/flash/a/b", "/flash"), "/a/b");
    }

    // ─── Existing in-memory tests stay byte-identical ─────────────────────

    #[test]
    fn mount_table_in_memory_lookup_matches_default_vfs() {
        let t = MountTable::new();
        // Same node count as the standalone `build_default_vfs()`.
        assert_eq!(t.in_memory().node_count(), 9);
        // Same nodes resolve identically.
        let n_via_table = t.in_memory().lookup("/dev/null").unwrap();
        let direct = build_default_vfs();
        let n_direct = direct.lookup("/dev/null").unwrap();
        assert_eq!(n_via_table.inode, n_direct.inode);
        assert_eq!(n_via_table.kind, n_direct.kind);
    }

    // ─── Auth-protected path guard tests ────────────────────────────────

    #[test]
    fn lookup_under_data_auth_returns_eacces() {
        // Spec: `management-login-v1` Phase 3, "VFS auth-path guard".
        let vfs = build_default_vfs();
        assert_eq!(vfs.lookup("/data/auth/shadow"), Err(Errno::EACCES));
        assert_eq!(vfs.lookup("/data/auth/shadow.tmp"), Err(Errno::EACCES));
        assert_eq!(vfs.lookup("/data/auth"), Err(Errno::EACCES));
        assert_eq!(vfs.lookup("/data/auth/"), Err(Errno::EACCES));
        assert_eq!(vfs.lookup("/data/auth/policy.toml"), Err(Errno::EACCES));
    }

    #[test]
    fn lookup_sibling_data_paths_unaffected() {
        // The guard must not over-match — `/data/audit` (without the
        // trailing `/`) and `/data/auth.bak` are NOT under
        // `/data/auth/`. They should fall through to ENOENT (the
        // mount doesn't exist yet) rather than EACCES.
        let vfs = build_default_vfs();
        assert_eq!(vfs.lookup("/data/audit"), Err(Errno::ENOENT));
        assert_eq!(vfs.lookup("/data/auth.bak"), Err(Errno::ENOENT));
        assert_eq!(vfs.lookup("/data/authx"), Err(Errno::ENOENT));
    }

    #[test]
    fn is_auth_protected_path_recognizes_canonical_forms() {
        assert!(is_auth_protected_path("/data/auth"));
        assert!(is_auth_protected_path("/data/auth/"));
        assert!(is_auth_protected_path("/data/auth/shadow"));
        assert!(is_auth_protected_path("/data/auth/shadow.tmp"));
        assert!(is_auth_protected_path("/data/auth/sub/dir"));

        assert!(!is_auth_protected_path("/data/auth.bak"));
        assert!(!is_auth_protected_path("/data/auths"));
        assert!(!is_auth_protected_path("/data/audit"));
        assert!(!is_auth_protected_path("/dev/null"));
        assert!(!is_auth_protected_path("/"));
    }
}
