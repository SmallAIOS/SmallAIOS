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
