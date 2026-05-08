// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! F2FS inode parser.
//!
//! Each F2FS inode lives in one 4 KiB node block. The on-disk layout
//! is `struct f2fs_inode` followed by `nid[5]` indirect-block pointers
//! (single, double, triple etc.) and a node footer. Layout pinned to
//! Linux 6.6 `fs/f2fs/f2fs.h:struct f2fs_inode` and
//! `struct node_footer` at the end of every node block.
//!
//! ## Layout (offsets within the 4 KiB inode block)
//!
//! | Off  | Size | Field                          |
//! |-----:|-----:|--------------------------------|
//! |   0  |   2  | `i_mode`                       |
//! |   2  |   1  | `i_advise`                     |
//! |   3  |   1  | `i_inline`                     |
//! |   4  |   4  | `i_uid`                        |
//! |   8  |   4  | `i_gid`                        |
//! |  12  |   4  | `i_links`                      |
//! |  16  |   8  | `i_size`                       |
//! |  24  |   8  | `i_blocks`                     |
//! |  32  |   8  | `i_atime`                      |
//! |  40  |   8  | `i_ctime`                      |
//! |  48  |   8  | `i_mtime`                      |
//! |  56  |   4  | `i_atime_nsec`                 |
//! |  60  |   4  | `i_ctime_nsec`                 |
//! |  64  |   4  | `i_mtime_nsec`                 |
//! |  68  |   4  | `i_generation`                 |
//! |  72  |   4  | `i_xattr_nid`                  |
//! |  76  |   4  | `i_flags`                      |
//! |  80  |   4  | `i_pino` (parent inode)        |
//! |  84  |   4  | `i_namelen`                    |
//! |  88  | 255  | `i_name[F2FS_NAME_LEN]`        |
//! | 343  |   1  | `i_dir_level`                  |
//! | 344  |  ...  | inline xattr / extra attr area |
//! | ...  |  ...  | `i_addr[923]` (block addrs)    |
//! | 4060 |  4×5 | `i_nid[5]` (indirect node ids) |
//! | 4080 |  16  | `node_footer`                  |
//!
//! For the v1 read path we parse: mode, uid/gid, size, mtime/ctime/atime,
//! block-address array (`i_addr[]`), and the first single-indirect
//! node-id (`i_nid[0]`). Double/triple indirection raises
//! [`F2fsError::DeepIndirectionUnsupported`].

extern crate alloc;

use alloc::string::String;

use super::error::F2fsError;
use super::{u16_le, u32_le, u64_le, F2FS_BLOCK_SIZE};

/// Number of direct block-address slots in the inode (`i_addr[923]` in
/// Linux 6.6, but the lower-numbered ones may be consumed by inline
/// data / xattrs depending on `i_inline` flags).
pub const ADDRS_PER_INODE: usize = 923;

/// Number of indirect node-id slots in the inode (`i_nid[5]`).
pub const NIDS_PER_INODE: usize = 5;

/// Node footer offset within the 4 KiB block.
pub const NODE_FOOTER_OFFSET: usize = F2FS_BLOCK_SIZE as usize - 16;

/// `i_inline` bit: data is inlined in the inode block (small files).
pub const INLINE_DATA: u8 = 0x02;
/// `i_inline` bit: dentry is inlined in the inode block (small dirs).
pub const INLINE_DENTRY: u8 = 0x04;
/// `i_inline` bit: extra attributes are present (`i_extra_isize` field).
pub const INLINE_EXTRA_ATTR: u8 = 0x10;
/// `i_inline` bit: inline xattr is enabled.
pub const INLINE_XATTR: u8 = 0x01;

/// POSIX file-mode bits (low 16 bits of `i_mode`).
pub const S_IFMT: u16 = 0xF000;
/// `i_mode` value: regular file.
pub const S_IFREG: u16 = 0x8000;
/// `i_mode` value: directory.
pub const S_IFDIR: u16 = 0x4000;
/// `i_mode` value: symlink.
pub const S_IFLNK: u16 = 0xA000;
/// `i_mode` value: character device.
pub const S_IFCHR: u16 = 0x2000;
/// `i_mode` value: block device.
pub const S_IFBLK: u16 = 0x6000;
/// `i_mode` value: FIFO.
pub const S_IFIFO: u16 = 0x1000;
/// `i_mode` value: UNIX domain socket.
pub const S_IFSOCK: u16 = 0xC000;

/// Decoded F2FS inode kind (mapped from `i_mode & S_IFMT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InodeKind {
    Regular,
    Directory,
    Symlink,
    CharDevice,
    BlockDevice,
    Fifo,
    Socket,
    Other,
}

impl InodeKind {
    pub fn from_mode(mode: u16) -> Self {
        match mode & S_IFMT {
            S_IFREG => Self::Regular,
            S_IFDIR => Self::Directory,
            S_IFLNK => Self::Symlink,
            S_IFCHR => Self::CharDevice,
            S_IFBLK => Self::BlockDevice,
            S_IFIFO => Self::Fifo,
            S_IFSOCK => Self::Socket,
            _ => Self::Other,
        }
    }
}

/// Parsed F2FS inode (load-bearing fields for the read path).
#[derive(Debug, Clone)]
pub struct Inode {
    pub mode: u16,
    pub kind: InodeKind,
    pub uid: u32,
    pub gid: u32,
    pub links: u32,
    pub size: u64,
    pub blocks: u64,
    pub atime: u64,
    pub ctime: u64,
    pub mtime: u64,
    pub atime_nsec: u32,
    pub ctime_nsec: u32,
    pub mtime_nsec: u32,
    pub generation: u32,
    pub xattr_nid: u32,
    pub flags: u32,
    pub pino: u32,
    pub name: String,
    pub dir_level: u8,
    pub inline_flags: u8,
    /// Inline data / inode-resident block addresses. The slot count
    /// is fixed at [`ADDRS_PER_INODE`]; unused slots hold 0 (= NULL_ADDR).
    pub i_addr: alloc::vec::Vec<u32>,
    /// Indirect node-ids `[direct1, direct2, indirect1, indirect2, double_indirect]`.
    pub i_nid: [u32; NIDS_PER_INODE],
    /// True if inline data is present (small file fits in inode).
    pub has_inline_data: bool,
    /// True if inline dentry is present (small dir).
    pub has_inline_dentry: bool,
}

impl Inode {
    /// Parse an inode from a 4 KiB inode block.
    pub fn parse(block: &[u8]) -> Result<Self, F2fsError> {
        if block.len() < F2FS_BLOCK_SIZE as usize {
            return Err(F2fsError::InodeCorrupt);
        }
        let mode = u16_le(&block[0..2]);
        let _advise = block[2];
        let inline_flags = block[3];
        let uid = u32_le(&block[4..8]);
        let gid = u32_le(&block[8..12]);
        let links = u32_le(&block[12..16]);
        let size = u64_le(&block[16..24]);
        let blocks = u64_le(&block[24..32]);
        let atime = u64_le(&block[32..40]);
        let ctime = u64_le(&block[40..48]);
        let mtime = u64_le(&block[48..56]);
        let atime_nsec = u32_le(&block[56..60]);
        let ctime_nsec = u32_le(&block[60..64]);
        let mtime_nsec = u32_le(&block[64..68]);
        let generation = u32_le(&block[68..72]);
        let xattr_nid = u32_le(&block[72..76]);
        let flags = u32_le(&block[76..80]);
        let pino = u32_le(&block[80..84]);
        let namelen = u32_le(&block[84..88]);
        // `i_name` field is 255 bytes from offset 88.
        let name_end = core::cmp::min(namelen as usize, 255);
        let name_bytes = &block[88..88 + name_end];
        let name = String::from_utf8_lossy(name_bytes).into_owned();
        let dir_level = block[343];

        // Block-address array starts at offset 360 in Linux 6.6's
        // canonical layout, after `i_dir_level` and reserved fields.
        // The exact start depends on whether `EXTRA_ATTR` is enabled
        // and what `i_extra_isize` declares; for the v1 reader we use
        // the default-mkfs layout where `i_addr[]` begins at byte 360.
        const I_ADDR_OFFSET: usize = 360;
        // i_nid[5] sits at the canonical position 4060..4080 (just
        // before the 16-byte node footer).
        const I_NID_OFFSET: usize = NODE_FOOTER_OFFSET - NIDS_PER_INODE * 4;

        let i_addr_bytes = ADDRS_PER_INODE * 4;
        if I_ADDR_OFFSET + i_addr_bytes > I_NID_OFFSET {
            return Err(F2fsError::InodeCorrupt);
        }

        let mut i_addr = alloc::vec::Vec::with_capacity(ADDRS_PER_INODE);
        for i in 0..ADDRS_PER_INODE {
            let o = I_ADDR_OFFSET + i * 4;
            i_addr.push(u32_le(&block[o..o + 4]));
        }

        let mut i_nid = [0u32; NIDS_PER_INODE];
        for (i, slot) in i_nid.iter_mut().enumerate() {
            let o = I_NID_OFFSET + i * 4;
            *slot = u32_le(&block[o..o + 4]);
        }

        let kind = InodeKind::from_mode(mode);
        let has_inline_data = (inline_flags & INLINE_DATA) != 0;
        let has_inline_dentry = (inline_flags & INLINE_DENTRY) != 0;

        Ok(Self {
            mode,
            kind,
            uid,
            gid,
            links,
            size,
            blocks,
            atime,
            ctime,
            mtime,
            atime_nsec,
            ctime_nsec,
            mtime_nsec,
            generation,
            xattr_nid,
            flags,
            pino,
            name,
            dir_level,
            inline_flags,
            i_addr,
            i_nid,
            has_inline_data,
            has_inline_dentry,
        })
    }

    /// True if this inode is a regular file.
    pub fn is_regular(&self) -> bool {
        self.kind == InodeKind::Regular
    }

    /// True if this inode is a directory.
    pub fn is_directory(&self) -> bool {
        self.kind == InodeKind::Directory
    }

    /// Get the file_offset → physical block address mapping for the
    /// `block_idx`-th 4 KiB block of the file.
    ///
    /// Walks the direct-pointer array first; if `block_idx` exceeds
    /// [`ADDRS_PER_INODE`], the caller must follow the appropriate
    /// indirect node-id from `i_nid`. v1 only handles direct pointers
    /// and one level of single-indirect (handled by [`crate::f2fs::data`]).
    pub fn direct_block_addr(&self, block_idx: usize) -> Option<u32> {
        self.i_addr.get(block_idx).copied()
    }
}

/// F2FS node-block footer (last 16 bytes of every node block).
///
/// Layout (Linux 6.6 `struct node_footer`):
///
/// | Off  | Size | Field        |
/// |-----:|-----:|--------------|
/// |   0  |   4  | `nid`        |
/// |   4  |   4  | `ino`        |
/// |   8  |   4  | `flag`       |
/// |  12  |   8  | `cp_ver`     |
/// |  20  |   4  | `next_blkaddr` |
///
/// (The footer is 16 bytes by Linux 6.6 layout; the `cp_ver`/`next_blkaddr`
/// extension lives in the trailing 8 bytes of the parent block when
/// the older 24-byte footer layout is used. v1 reads the canonical
/// 16-byte version.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NodeFooter {
    pub nid: u32,
    pub ino: u32,
    pub flag: u32,
}

impl NodeFooter {
    /// Parse the 16-byte node footer at the end of a node block.
    pub fn parse(block: &[u8]) -> Result<Self, F2fsError> {
        if block.len() < F2FS_BLOCK_SIZE as usize {
            return Err(F2fsError::InodeCorrupt);
        }
        let off = NODE_FOOTER_OFFSET;
        let nid = u32_le(&block[off..off + 4]);
        let ino = u32_le(&block[off + 4..off + 8]);
        let flag = u32_le(&block[off + 8..off + 12]);
        Ok(Self { nid, ino, flag })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_inode_block(mode: u16, size: u64, inline: u8) -> alloc::vec::Vec<u8> {
        let mut buf = alloc::vec![0u8; F2FS_BLOCK_SIZE as usize];
        buf[0..2].copy_from_slice(&mode.to_le_bytes());
        buf[3] = inline;
        buf[16..24].copy_from_slice(&size.to_le_bytes());
        buf
    }

    #[test]
    fn parse_regular_file() {
        let buf = make_inode_block(S_IFREG | 0o644, 1024, 0);
        let inode = Inode::parse(&buf).unwrap();
        assert_eq!(inode.mode, S_IFREG | 0o644);
        assert_eq!(inode.kind, InodeKind::Regular);
        assert!(inode.is_regular());
        assert_eq!(inode.size, 1024);
        assert_eq!(inode.i_addr.len(), ADDRS_PER_INODE);
    }

    #[test]
    fn parse_directory() {
        let buf = make_inode_block(S_IFDIR | 0o755, 4096, 0);
        let inode = Inode::parse(&buf).unwrap();
        assert!(inode.is_directory());
        assert_eq!(inode.kind, InodeKind::Directory);
    }

    #[test]
    fn parse_symlink() {
        let buf = make_inode_block(S_IFLNK | 0o777, 0, 0);
        let inode = Inode::parse(&buf).unwrap();
        assert_eq!(inode.kind, InodeKind::Symlink);
    }

    #[test]
    fn parse_inline_data_flag() {
        let buf = make_inode_block(S_IFREG | 0o644, 100, INLINE_DATA);
        let inode = Inode::parse(&buf).unwrap();
        assert!(inode.has_inline_data);
        assert!(!inode.has_inline_dentry);
    }

    #[test]
    fn parse_inline_dentry_flag() {
        let buf = make_inode_block(S_IFDIR | 0o755, 100, INLINE_DENTRY);
        let inode = Inode::parse(&buf).unwrap();
        assert!(inode.has_inline_dentry);
    }

    #[test]
    fn parse_truncated_rejected() {
        let buf = alloc::vec![0u8; 100];
        assert!(matches!(Inode::parse(&buf), Err(F2fsError::InodeCorrupt)));
    }

    #[test]
    fn direct_block_addr_round_trip() {
        let mut buf = make_inode_block(S_IFREG, 0, 0);
        // Place a known address at i_addr[5].
        let off = 360 + 5 * 4;
        buf[off..off + 4].copy_from_slice(&0xCAFEu32.to_le_bytes());
        let inode = Inode::parse(&buf).unwrap();
        assert_eq!(inode.direct_block_addr(5), Some(0xCAFE));
        assert_eq!(inode.direct_block_addr(0), Some(0));
        assert_eq!(inode.direct_block_addr(ADDRS_PER_INODE), None);
    }

    #[test]
    fn mode_bit_decoding() {
        assert_eq!(InodeKind::from_mode(S_IFREG), InodeKind::Regular);
        assert_eq!(InodeKind::from_mode(S_IFDIR), InodeKind::Directory);
        assert_eq!(InodeKind::from_mode(S_IFLNK), InodeKind::Symlink);
        assert_eq!(InodeKind::from_mode(S_IFBLK), InodeKind::BlockDevice);
        assert_eq!(InodeKind::from_mode(S_IFCHR), InodeKind::CharDevice);
        assert_eq!(InodeKind::from_mode(S_IFIFO), InodeKind::Fifo);
        assert_eq!(InodeKind::from_mode(S_IFSOCK), InodeKind::Socket);
        assert_eq!(InodeKind::from_mode(0), InodeKind::Other);
    }

    #[test]
    fn timestamps_decoded() {
        let mut buf = make_inode_block(S_IFREG, 0, 0);
        buf[32..40].copy_from_slice(&100u64.to_le_bytes());
        buf[40..48].copy_from_slice(&200u64.to_le_bytes());
        buf[48..56].copy_from_slice(&300u64.to_le_bytes());
        let inode = Inode::parse(&buf).unwrap();
        assert_eq!(inode.atime, 100);
        assert_eq!(inode.ctime, 200);
        assert_eq!(inode.mtime, 300);
    }

    #[test]
    fn node_footer_round_trip() {
        let mut buf = alloc::vec![0u8; F2FS_BLOCK_SIZE as usize];
        let off = NODE_FOOTER_OFFSET;
        buf[off..off + 4].copy_from_slice(&7u32.to_le_bytes());
        buf[off + 4..off + 8].copy_from_slice(&7u32.to_le_bytes());
        buf[off + 8..off + 12].copy_from_slice(&3u32.to_le_bytes());
        let footer = NodeFooter::parse(&buf).unwrap();
        assert_eq!(footer.nid, 7);
        assert_eq!(footer.ino, 7);
        assert_eq!(footer.flag, 3);
    }

    #[test]
    fn name_lossy_decode() {
        let mut buf = make_inode_block(S_IFREG, 0, 0);
        let name = b"hello.txt";
        buf[84..88].copy_from_slice(&(name.len() as u32).to_le_bytes());
        buf[88..88 + name.len()].copy_from_slice(name);
        let inode = Inode::parse(&buf).unwrap();
        assert_eq!(inode.name, "hello.txt");
    }

    #[test]
    fn i_nid_round_trip() {
        let mut buf = make_inode_block(S_IFREG, 0, 0);
        let off = NODE_FOOTER_OFFSET - NIDS_PER_INODE * 4;
        for i in 0..NIDS_PER_INODE {
            buf[off + i * 4..off + i * 4 + 4].copy_from_slice(&((i + 100) as u32).to_le_bytes());
        }
        let inode = Inode::parse(&buf).unwrap();
        assert_eq!(inode.i_nid, [100, 101, 102, 103, 104]);
    }
}
