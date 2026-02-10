// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Read-only virtual filesystem for the POSIX compatibility layer.
//!
//! Provides a minimal VFS with fixed mount points:
//! - `/models/`  — ONNX model files (loaded at boot)
//! - `/config/`  — Runtime configuration
//! - `/dev/`     — Device nodes (null, urandom)
//! - `/proc/self/` — Process introspection
//!
//! The filesystem is entirely read-only; all write attempts return `EROFS`.

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

// ─── VFS Tree ────────────────────────────────────────────────────────────────

/// The virtual filesystem tree.
///
/// All nodes are stored in a flat, fixed-size array. The root node is
/// always at index 0.
pub struct VfsTree {
    /// Node storage.
    nodes: [Option<VfsNode>; MAX_VFS_NODES],
    /// Number of nodes currently in the tree.
    count: usize,
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
    pub fn lookup(&self, path: &str) -> Result<&VfsNode, Errno> {
        if !path.starts_with('/') {
            return Err(Errno::EINVAL);
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
            // Current node must be a directory
            if current_node.kind != VfsNodeKind::Directory {
                return Err(Errno::ENOENT);
            }
            // Search children
            let mut found = false;
            for i in 0..current_node.num_children {
                if let Some(child_idx) = current_node.children[i] {
                    if let Some(child) = &self.nodes[child_idx] {
                        if child.name == component {
                            current_node = child;
                            found = true;
                            break;
                        }
                    }
                }
            }
            if !found {
                return Err(Errno::ENOENT);
            }
        }

        Ok(current_node)
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
    tree.nodes[proc_idx].as_mut().unwrap().add_child(proc_self_idx);

    // proc/self -> maps
    tree.nodes[proc_self_idx].as_mut().unwrap().add_child(maps_idx);

    tree
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
}
