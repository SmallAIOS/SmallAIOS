// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! File descriptor table and lifecycle management.
//!
//! Each task has a file descriptor table mapping integer descriptors to
//! open file objects. Standard descriptors:
//! - 0: stdin (connected to /dev/null)
//! - 1: stdout (connected to serial console)
//! - 2: stderr (connected to serial console)
//!
//! File descriptors support reference counting for dup/dup2 operations.

use crate::errno::Errno;

// ─── Constants ───────────────────────────────────────────────────────────────

/// Maximum number of open file descriptors per task.
pub const MAX_FDS: usize = 256;

/// Standard input file descriptor.
pub const STDIN_FD: i32 = 0;

/// Standard output file descriptor.
pub const STDOUT_FD: i32 = 1;

/// Standard error file descriptor.
pub const STDERR_FD: i32 = 2;

// ─── File Descriptor Types ──────────────────────────────────────────────────

/// The kind of resource backing a file descriptor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FdKind {
    /// Regular file in the virtual filesystem.
    File,
    /// /dev/null — reads return EOF, writes are discarded.
    DevNull,
    /// /dev/urandom — reads return random bytes.
    DevUrandom,
    /// Serial console (stdout/stderr).
    Console,
    /// TCP or UDP socket.
    Socket,
    /// Epoll instance.
    Epoll,
    /// Timer file descriptor (timerfd).
    Timer,
    /// Signal file descriptor (signalfd).
    Signal,
}

/// Open file flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpenFlags(pub u32);

impl OpenFlags {
    pub const O_RDONLY: u32 = 0;
    pub const O_WRONLY: u32 = 1;
    pub const O_RDWR: u32 = 2;
    pub const O_CREAT: u32 = 0o100;
    pub const O_TRUNC: u32 = 0o1000;
    pub const O_APPEND: u32 = 0o2000;
    pub const O_NONBLOCK: u32 = 0o4000;
    pub const O_CLOEXEC: u32 = 0o2000000;

    pub const fn new(flags: u32) -> Self {
        Self(flags)
    }

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn is_readable(self) -> bool {
        let access = self.0 & 3;
        access == Self::O_RDONLY || access == Self::O_RDWR
    }

    pub const fn is_writable(self) -> bool {
        let access = self.0 & 3;
        access == Self::O_WRONLY || access == Self::O_RDWR
    }

    pub const fn is_nonblocking(self) -> bool {
        self.0 & Self::O_NONBLOCK != 0
    }
}

/// File status information (similar to struct stat).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FileStat {
    /// File size in bytes.
    pub size: u64,
    /// File mode (permissions + type bits).
    pub mode: u32,
    /// Number of hard links.
    pub nlink: u32,
    /// Inode number (virtual).
    pub ino: u64,
}

impl FileStat {
    /// File type bits.
    pub const S_IFREG: u32 = 0o100000;
    pub const S_IFDIR: u32 = 0o040000;
    pub const S_IFCHR: u32 = 0o020000;
    pub const S_IFSOCK: u32 = 0o140000;

    pub const fn regular(size: u64, ino: u64) -> Self {
        Self {
            size,
            mode: Self::S_IFREG | 0o444,
            nlink: 1,
            ino,
        }
    }

    pub const fn directory(ino: u64) -> Self {
        Self {
            size: 0,
            mode: Self::S_IFDIR | 0o555,
            nlink: 2,
            ino,
        }
    }

    pub const fn char_device(ino: u64) -> Self {
        Self {
            size: 0,
            mode: Self::S_IFCHR | 0o666,
            nlink: 1,
            ino,
        }
    }
}

/// An open file descriptor entry in the table.
#[derive(Debug, Clone, PartialEq)]
pub struct FdEntry {
    /// What kind of resource this fd points to.
    pub kind: FdKind,
    /// Open flags.
    pub flags: OpenFlags,
    /// Current seek offset (for regular files).
    pub offset: u64,
    /// Virtual inode number for the backing VFS node.
    pub vnode_id: u64,
    /// Reference count (for dup/dup2).
    pub ref_count: u32,
}

impl FdEntry {
    pub fn new(kind: FdKind, flags: OpenFlags, vnode_id: u64) -> Self {
        Self {
            kind,
            flags,
            offset: 0,
            vnode_id,
            ref_count: 1,
        }
    }
}

// ─── File Descriptor Table ──────────────────────────────────────────────────

/// Per-task file descriptor table.
pub struct FdTable {
    /// File descriptor entries. None = slot is free.
    entries: [Option<FdEntry>; MAX_FDS],
    /// Number of open descriptors.
    count: usize,
}

impl Default for FdTable {
    fn default() -> Self {
        Self::new()
    }
}

impl FdTable {
    /// Create a new file descriptor table with stdin/stdout/stderr.
    pub fn new() -> Self {
        let mut table = Self {
            entries: core::array::from_fn(|_| None),
            count: 0,
        };

        // fd 0: stdin → /dev/null
        table.entries[0] = Some(FdEntry::new(
            FdKind::DevNull,
            OpenFlags::new(OpenFlags::O_RDONLY),
            0,
        ));
        // fd 1: stdout → console
        table.entries[1] = Some(FdEntry::new(
            FdKind::Console,
            OpenFlags::new(OpenFlags::O_WRONLY),
            1,
        ));
        // fd 2: stderr → console
        table.entries[2] = Some(FdEntry::new(
            FdKind::Console,
            OpenFlags::new(OpenFlags::O_WRONLY),
            2,
        ));
        table.count = 3;
        table
    }

    /// Find the lowest available file descriptor number.
    fn find_free_slot(&self) -> Result<usize, Errno> {
        for (i, entry) in self.entries.iter().enumerate() {
            if entry.is_none() {
                return Ok(i);
            }
        }
        Err(Errno::EMFILE)
    }

    /// Allocate a new file descriptor with the given entry.
    pub fn alloc(&mut self, entry: FdEntry) -> Result<i32, Errno> {
        let slot = self.find_free_slot()?;
        self.entries[slot] = Some(entry);
        self.count += 1;
        Ok(slot as i32)
    }

    /// Get a reference to an fd entry.
    pub fn get(&self, fd: i32) -> Result<&FdEntry, Errno> {
        if fd < 0 || fd as usize >= MAX_FDS {
            return Err(Errno::EBADF);
        }
        self.entries[fd as usize].as_ref().ok_or(Errno::EBADF)
    }

    /// Get a mutable reference to an fd entry.
    pub fn get_mut(&mut self, fd: i32) -> Result<&mut FdEntry, Errno> {
        if fd < 0 || fd as usize >= MAX_FDS {
            return Err(Errno::EBADF);
        }
        self.entries[fd as usize].as_mut().ok_or(Errno::EBADF)
    }

    /// Close a file descriptor.
    pub fn close(&mut self, fd: i32) -> Result<(), Errno> {
        if fd < 0 || fd as usize >= MAX_FDS {
            return Err(Errno::EBADF);
        }
        let slot = fd as usize;
        if self.entries[slot].is_none() {
            return Err(Errno::EBADF);
        }
        // Decrement ref_count; only free if it reaches zero
        let should_free = {
            let entry = self.entries[slot].as_mut().unwrap();
            entry.ref_count = entry.ref_count.saturating_sub(1);
            entry.ref_count == 0
        };
        if should_free {
            self.entries[slot] = None;
            self.count -= 1;
        }
        Ok(())
    }

    /// Duplicate a file descriptor (dup).
    pub fn dup(&mut self, old_fd: i32) -> Result<i32, Errno> {
        let entry = self.get(old_fd)?.clone();
        let new_fd = self.alloc(entry)?;
        // Share the same underlying resource: bump ref_count on both
        if let Some(e) = &mut self.entries[old_fd as usize] {
            e.ref_count += 1;
        }
        if let Some(e) = &mut self.entries[new_fd as usize] {
            e.ref_count += 1;
        }
        Ok(new_fd)
    }

    /// Duplicate a file descriptor to a specific number (dup2).
    pub fn dup2(&mut self, old_fd: i32, new_fd: i32) -> Result<i32, Errno> {
        if new_fd < 0 || new_fd as usize >= MAX_FDS {
            return Err(Errno::EBADF);
        }
        if old_fd == new_fd {
            // Verify old_fd is valid
            self.get(old_fd)?;
            return Ok(new_fd);
        }
        // Close new_fd if it's open (ignore EBADF)
        let _ = self.close(new_fd);
        let entry = self.get(old_fd)?.clone();
        self.entries[new_fd as usize] = Some(entry);
        self.count += 1;
        // Bump ref_count on old
        if let Some(e) = &mut self.entries[old_fd as usize] {
            e.ref_count += 1;
        }
        if let Some(e) = &mut self.entries[new_fd as usize] {
            e.ref_count += 1;
        }
        Ok(new_fd)
    }

    /// Return the number of open file descriptors.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Check if a file descriptor is valid/open.
    pub fn is_open(&self, fd: i32) -> bool {
        if fd < 0 || fd as usize >= MAX_FDS {
            return false;
        }
        self.entries[fd as usize].is_some()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fd_table_new_has_stdio() {
        let table = FdTable::new();
        assert_eq!(table.count(), 3);
        assert!(table.is_open(STDIN_FD));
        assert!(table.is_open(STDOUT_FD));
        assert!(table.is_open(STDERR_FD));
    }

    #[test]
    fn fd_table_stdio_types() {
        let table = FdTable::new();
        assert_eq!(table.get(STDIN_FD).unwrap().kind, FdKind::DevNull);
        assert_eq!(table.get(STDOUT_FD).unwrap().kind, FdKind::Console);
        assert_eq!(table.get(STDERR_FD).unwrap().kind, FdKind::Console);
    }

    #[test]
    fn fd_table_alloc_returns_lowest_free() {
        let mut table = FdTable::new();
        let entry = FdEntry::new(FdKind::File, OpenFlags::new(0), 100);
        let fd = table.alloc(entry).unwrap();
        assert_eq!(fd, 3); // 0,1,2 are taken
        assert_eq!(table.count(), 4);
    }

    #[test]
    fn fd_table_close_and_reuse() {
        let mut table = FdTable::new();
        let entry = FdEntry::new(FdKind::File, OpenFlags::new(0), 100);
        let fd = table.alloc(entry).unwrap();
        assert_eq!(fd, 3);
        table.close(fd).unwrap();
        assert!(!table.is_open(fd));
        // New alloc should reuse slot 3
        let entry2 = FdEntry::new(FdKind::File, OpenFlags::new(0), 200);
        let fd2 = table.alloc(entry2).unwrap();
        assert_eq!(fd2, 3);
    }

    #[test]
    fn fd_table_close_invalid() {
        let mut table = FdTable::new();
        assert_eq!(table.close(-1), Err(Errno::EBADF));
        assert_eq!(table.close(256), Err(Errno::EBADF));
        assert_eq!(table.close(100), Err(Errno::EBADF));
    }

    #[test]
    fn fd_table_get_invalid() {
        let table = FdTable::new();
        assert_eq!(table.get(-1), Err(Errno::EBADF));
        assert_eq!(table.get(MAX_FDS as i32), Err(Errno::EBADF));
        assert_eq!(table.get(100), Err(Errno::EBADF));
    }

    #[test]
    fn fd_table_dup() {
        let mut table = FdTable::new();
        let new_fd = table.dup(STDOUT_FD).unwrap();
        assert_eq!(new_fd, 3);
        assert_eq!(table.get(new_fd).unwrap().kind, FdKind::Console);
    }

    #[test]
    fn fd_table_dup_invalid() {
        let mut table = FdTable::new();
        assert_eq!(table.dup(100), Err(Errno::EBADF));
    }

    #[test]
    fn fd_table_dup2_same_fd() {
        let mut table = FdTable::new();
        let result = table.dup2(STDOUT_FD, STDOUT_FD).unwrap();
        assert_eq!(result, STDOUT_FD);
    }

    #[test]
    fn fd_table_dup2_to_new_slot() {
        let mut table = FdTable::new();
        let result = table.dup2(STDOUT_FD, 10).unwrap();
        assert_eq!(result, 10);
        assert_eq!(table.get(10).unwrap().kind, FdKind::Console);
    }

    #[test]
    fn fd_table_dup2_replaces_existing() {
        let mut table = FdTable::new();
        let entry = FdEntry::new(FdKind::File, OpenFlags::new(0), 100);
        let fd = table.alloc(entry).unwrap();
        // Dup stdout to fd 3, replacing the file
        table.dup2(STDOUT_FD, fd).unwrap();
        assert_eq!(table.get(fd).unwrap().kind, FdKind::Console);
    }

    #[test]
    fn open_flags_readable_writable() {
        let rdonly = OpenFlags::new(OpenFlags::O_RDONLY);
        assert!(rdonly.is_readable());
        assert!(!rdonly.is_writable());

        let wronly = OpenFlags::new(OpenFlags::O_WRONLY);
        assert!(!wronly.is_readable());
        assert!(wronly.is_writable());

        let rdwr = OpenFlags::new(OpenFlags::O_RDWR);
        assert!(rdwr.is_readable());
        assert!(rdwr.is_writable());
    }

    #[test]
    fn open_flags_nonblocking() {
        let flags = OpenFlags::new(OpenFlags::O_RDONLY | OpenFlags::O_NONBLOCK);
        assert!(flags.is_nonblocking());
        let flags2 = OpenFlags::new(OpenFlags::O_RDONLY);
        assert!(!flags2.is_nonblocking());
    }

    #[test]
    fn file_stat_regular() {
        let stat = FileStat::regular(1024, 42);
        assert_eq!(stat.size, 1024);
        assert_eq!(stat.ino, 42);
        assert_eq!(stat.mode & FileStat::S_IFREG, FileStat::S_IFREG);
    }

    #[test]
    fn file_stat_directory() {
        let stat = FileStat::directory(1);
        assert_eq!(stat.size, 0);
        assert_eq!(stat.mode & FileStat::S_IFDIR, FileStat::S_IFDIR);
    }

    #[test]
    fn file_stat_char_device() {
        let stat = FileStat::char_device(5);
        assert_eq!(stat.mode & FileStat::S_IFCHR, FileStat::S_IFCHR);
    }
}
