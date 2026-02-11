// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS POSIX Compatibility Layer
//!
//! Implements the ~5% of POSIX that AI inference workloads actually use:
//! - File descriptors (open, read, write, close, fstat)
//! - Memory mapping (mmap, munmap, mprotect)
//! - Threading (pthread create/join, mutex, condvar, rwlock)
//! - Sockets (TCP client/server via native stack)
//! - Epoll (async I/O multiplexing)
//! - Time (clock_gettime, nanosleep)
//! - Random (getrandom backed by PQC-grade CSPRNG)
//! - Virtual filesystem (read-only: /models/, /config/, /dev/, /proc/self/)
//!
//! Unsupported calls return ENOSYS (-38).

#![no_std]

extern crate alloc;

pub mod epoll;
pub mod errno;
pub mod fd;
pub mod mmap;
pub mod pthread;
pub mod signal;
pub mod socket;
pub mod time;
pub mod vfs;

// ---------------------------------------------------------------------------
// Integration tests: Rust std-style operations against the POSIX layer (6.10)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Tests exercising Rust std library-like operations against the POSIX
    //! compatibility layer. These verify that the abstractions behave like
    //! a Rust program would expect from a POSIX platform.

    use super::*;
    use alloc::format;
    use alloc::vec;

    // -- File descriptor operations (like std::fs::File) --

    #[test]
    fn test_fd_table_simulates_file_open_read_close() {
        let mut table = fd::FdTable::new();
        // Simulate: open("/models/model.onnx", O_RDONLY)
        let entry = fd::FdEntry::new(
            fd::FdKind::File,
            fd::OpenFlags::new(fd::OpenFlags::O_RDONLY),
            100,
        );
        let fd = table.alloc(entry).unwrap();
        assert!(fd >= 3); // After stdin/stdout/stderr

        // Verify we can get the entry
        let e = table.get(fd).unwrap();
        assert_eq!(e.kind, fd::FdKind::File);
        assert!(e.flags.is_readable());
        assert!(!e.flags.is_writable());

        // Close
        table.close(fd).unwrap();
        assert!(!table.is_open(fd));
    }

    #[test]
    fn test_fd_dup_simulates_stdout_redirect() {
        let mut table = fd::FdTable::new();
        // Simulate: dup2(stdout, new_fd) — redirecting stdout
        let new_fd = table.dup(fd::STDOUT_FD).unwrap();
        let e = table.get(new_fd).unwrap();
        assert_eq!(e.kind, fd::FdKind::Console);
    }

    #[test]
    fn test_fd_table_sequential_alloc() {
        let mut table = fd::FdTable::new();
        // Simulate opening multiple files like a Rust program would
        let mut fds = vec![];
        for i in 0..10 {
            let entry = fd::FdEntry::new(
                fd::FdKind::File,
                fd::OpenFlags::new(fd::OpenFlags::O_RDONLY),
                100 + i as u64,
            );
            let fd = table.alloc(entry).unwrap();
            fds.push(fd);
        }
        assert_eq!(fds.len(), 10);
        // All fds should be unique
        for i in 0..fds.len() {
            for j in i + 1..fds.len() {
                assert_ne!(fds[i], fds[j]);
            }
        }
        // Close all
        for fd in &fds {
            table.close(*fd).unwrap();
        }
        assert_eq!(table.count(), 3); // Only stdin/stdout/stderr remain
    }

    // -- Memory mapping (like std's allocator using mmap) --

    #[test]
    fn test_mmap_anonymous_private_validated() {
        let mut table = mmap::MmapTable::new();
        // This is what Rust's allocator would call for large allocations:
        // mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE, -1, 0)
        let result = table.mmap(
            0,
            64 * 1024,
            mmap::PROT_READ | mmap::PROT_WRITE,
            mmap::MAP_ANONYMOUS | mmap::MAP_PRIVATE,
            -1,
            0,
        );
        // Currently a stub — ENOSYS after validation passes
        assert_eq!(result, Err(errno::Errno::ENOSYS));
    }

    #[test]
    fn test_mmap_page_align_up_for_allocator() {
        // Rust allocator rounds up to page size
        assert_eq!(mmap::page_align_up(1), 4096);
        assert_eq!(mmap::page_align_up(4096), 4096);
        assert_eq!(mmap::page_align_up(4097), 8192);
        assert_eq!(mmap::page_align_up(100_000), 102400);
    }

    // -- Time operations (like std::time) --

    #[test]
    fn test_timespec_duration_conversion() {
        // Simulate std::time::Duration-like operations
        let ts = time::Timespec::new(1, 500_000_000);
        assert_eq!(ts.to_nanos(), 1_500_000_000);
        assert!(ts.is_valid());
    }

    #[test]
    fn test_clock_getres_monotonic_is_nanosecond() {
        // Rust's Instant uses CLOCK_MONOTONIC
        let res = time::clock_getres(time::ClockId::Monotonic).unwrap();
        assert_eq!(res.tv_sec, 0);
        assert_eq!(res.tv_nsec, 1);
    }

    // -- Threading (like std::thread) --

    #[test]
    fn test_thread_attr_default_stack_size() {
        // std::thread uses a default stack size; ours should be reasonable
        let attr = pthread::ThreadAttr::new();
        assert!(attr.stack_size >= 16 * 1024); // At least 16 KiB
        assert!(!attr.detached);
    }

    #[test]
    fn test_thread_attr_builder_pattern() {
        let attr = pthread::ThreadAttr::new()
            .with_stack_size(128 * 1024)
            .with_detached(true);
        assert_eq!(attr.stack_size, 128 * 1024);
        assert!(attr.detached);
    }

    // -- Error handling (like std::io::Error) --

    #[test]
    fn test_errno_maps_to_std_io_error_kinds() {
        // Verify errno values match what std::io::Error would expect
        assert_eq!(errno::Errno::ENOENT as i32, 2);
        assert_eq!(errno::Errno::EACCES as i32, 13);
        assert_eq!(errno::Errno::EEXIST as i32, 17);
        assert_eq!(errno::Errno::EINVAL as i32, 22);
        assert_eq!(errno::Errno::ENOSYS as i32, 38);
        assert_eq!(errno::Errno::ECONNREFUSED as i32, 111);
        assert_eq!(errno::Errno::ETIMEDOUT as i32, 110);
    }

    #[test]
    fn test_errno_display_messages() {
        assert_eq!(
            format!("{}", errno::Errno::ENOENT),
            "No such file or directory"
        );
        assert_eq!(format!("{}", errno::Errno::EACCES), "Permission denied");
        assert_eq!(
            format!("{}", errno::Errno::ENOSYS),
            "Function not implemented"
        );
    }

    #[test]
    fn test_errno_round_trip() {
        // All errno values should round-trip through from_raw
        let errnos = [
            errno::Errno::EPERM,
            errno::Errno::ENOENT,
            errno::Errno::EBADF,
            errno::Errno::EAGAIN,
            errno::Errno::ENOMEM,
            errno::Errno::EACCES,
            errno::Errno::EINVAL,
            errno::Errno::ENOSYS,
            errno::Errno::ECONNREFUSED,
            errno::Errno::ECONNRESET,
            errno::Errno::ETIMEDOUT,
        ];
        for e in &errnos {
            let raw = *e as i32;
            assert_eq!(
                errno::Errno::from_raw(raw),
                Some(*e),
                "round-trip failed for {:?}",
                e
            );
        }
    }

    // -- Getrandom (like std uses for HashMap seed) --

    #[test]
    fn test_getrandom_zero_length_succeeds() {
        // std sometimes calls getrandom with 0 length
        let mut buf = [0u8; 0];
        assert_eq!(time::getrandom(&mut buf, 0, 0), Ok(0));
    }

    // -- VFS path operations --

    #[test]
    fn test_file_stat_types() {
        let regular = fd::FileStat::regular(1024, 1);
        assert!(regular.mode & fd::FileStat::S_IFREG != 0);
        assert_eq!(regular.size, 1024);

        let dir = fd::FileStat::directory(2);
        assert!(dir.mode & fd::FileStat::S_IFDIR != 0);

        let chr = fd::FileStat::char_device(3);
        assert!(chr.mode & fd::FileStat::S_IFCHR != 0);
    }

    // -- Socket operations (like std::net) --

    #[test]
    fn test_open_flags_rdwr_like_socket() {
        // TCP sockets are opened read-write
        let flags = fd::OpenFlags::new(fd::OpenFlags::O_RDWR);
        assert!(flags.is_readable());
        assert!(flags.is_writable());
    }

    // -- Combined lifecycle test --

    #[test]
    fn test_full_posix_lifecycle() {
        // Simulate a typical AI inference program's POSIX usage:
        // 1. Open model file
        // 2. Dup stdout for logging
        // 3. Set up mmap for tensor memory
        // 4. Clean up

        let mut fd_table = fd::FdTable::new();
        let mut mmap_table = mmap::MmapTable::new();

        // Step 1: Open model file
        let model_fd = fd_table
            .alloc(fd::FdEntry::new(
                fd::FdKind::File,
                fd::OpenFlags::new(fd::OpenFlags::O_RDONLY),
                42,
            ))
            .unwrap();
        assert!(model_fd >= 3);

        // Step 2: Dup stdout
        let log_fd = fd_table.dup(fd::STDOUT_FD).unwrap();
        assert_ne!(log_fd, model_fd);

        // Step 3: mmap for tensor allocation (validated but stub)
        let _ = mmap_table.mmap(
            0,
            1024 * 1024, // 1 MiB
            mmap::PROT_READ | mmap::PROT_WRITE,
            mmap::MAP_ANONYMOUS | mmap::MAP_PRIVATE,
            -1,
            0,
        );

        // Step 4: Cleanup
        fd_table.close(model_fd).unwrap();
        fd_table.close(log_fd).unwrap();

        // Verify cleanup
        assert!(!fd_table.is_open(model_fd));
        assert!(fd_table.is_open(fd::STDIN_FD));
        assert!(fd_table.is_open(fd::STDOUT_FD));
        assert!(fd_table.is_open(fd::STDERR_FD));
    }
}
