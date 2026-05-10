// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Pluggable JSONL flush sink for [`super::ring::Ring`].
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-audit-log/spec.md`
//! Requirement "In-memory ring with periodic flush".
//!
//! The default production flusher writes append-only JSONL to
//! `/data/audit/log.jsonl`. Tests substitute [`MemoryFlusher`] which
//! captures every flushed line in a `Vec<String>` for assertions.
//!
//! Auth events bypass the 1-second timer — `Ring::append` calls
//! `write_line(line, immediate=true)` and the flusher is expected to
//! `fsync` synchronously before returning.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt;

/// Flush-sink errors. Surfaces map their concrete IO errors to one of
/// these variants.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FlusherError {
    /// Underlying IO failed (block device timeout, permission denied,
    /// disk full prior to failsafe eviction, etc.).
    Io(&'static str),
    /// The flusher rejected the line because it carries a non-UTF-8
    /// byte sequence — JSONL is text-only.
    NotUtf8,
}

impl fmt::Display for FlusherError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(reason) => write!(f, "audit io: {}", reason),
            Self::NotUtf8 => f.write_str("audit line not utf-8"),
        }
    }
}

/// Pluggable flush sink. The production wiring (a future
/// `kernel::audit::F2fsFlusher`) writes to `/data/audit/log.jsonl`;
/// `MemoryFlusher` captures lines in-process for tests.
pub trait Flusher {
    /// Write one JSONL line. The flusher SHALL append a `\n`
    /// terminator. When `immediate` is `true` the flusher SHALL
    /// `fsync` before returning so the spec's "auth event flushes
    /// before the next non-auth syscall completes" invariant holds.
    fn write_line(&mut self, line: &str, immediate: bool) -> Result<(), FlusherError>;

    /// Flush any buffered data. Called from `Ring::tick` once a
    /// second.
    fn flush(&mut self) -> Result<(), FlusherError>;
}

// ─── In-memory test fixture ──────────────────────────────────────────────────

/// Captures every flushed line in an internal `Vec<String>`. Used by
/// every `Ring::in_memory()` test. The interior `RefCell` lets
/// callers borrow the line list immutably from `flusher()` while the
/// ring still holds a mutable reference via `flusher_mut()`.
#[derive(Debug, Default)]
pub struct MemoryFlusher {
    lines: RefCell<Vec<String>>,
    /// Number of `flush()` invocations.
    flush_count: RefCell<u32>,
    /// Number of immediate-flush `write_line` invocations (auth
    /// events).
    immediate_count: RefCell<u32>,
    /// When set, the next `write_line` returns this error and clears
    /// the slot. Used by the rotation tests to simulate a transient
    /// IO failure.
    pending_error: RefCell<Option<FlusherError>>,
}

impl MemoryFlusher {
    /// Construct an empty memory flusher.
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow the captured JSONL lines (immutable snapshot).
    pub fn written_lines(&self) -> Vec<String> {
        self.lines.borrow().clone()
    }

    /// Number of `flush()` calls observed.
    pub fn flush_count(&self) -> u32 {
        *self.flush_count.borrow()
    }

    /// Number of immediate-flush `write_line` calls observed.
    pub fn immediate_count(&self) -> u32 {
        *self.immediate_count.borrow()
    }

    /// Inject an error to be returned on the next `write_line` call.
    pub fn set_next_error(&self, err: FlusherError) {
        *self.pending_error.borrow_mut() = Some(err);
    }

    /// Clear all captured state.
    pub fn clear(&self) {
        self.lines.borrow_mut().clear();
        *self.flush_count.borrow_mut() = 0;
        *self.immediate_count.borrow_mut() = 0;
    }
}

impl Flusher for MemoryFlusher {
    fn write_line(&mut self, line: &str, immediate: bool) -> Result<(), FlusherError> {
        if let Some(err) = self.pending_error.borrow_mut().take() {
            return Err(err);
        }
        self.lines.borrow_mut().push(line.to_string());
        if immediate {
            *self.immediate_count.borrow_mut() += 1;
            *self.flush_count.borrow_mut() += 1;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), FlusherError> {
        *self.flush_count.borrow_mut() += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flusher_records_lines() {
        let mut f = MemoryFlusher::new();
        f.write_line("a", false).unwrap();
        f.write_line("b", true).unwrap();
        let lines = f.written_lines();
        assert_eq!(lines, alloc::vec!["a".to_string(), "b".to_string()]);
        assert_eq!(f.immediate_count(), 1);
        assert_eq!(f.flush_count(), 1);
    }

    #[test]
    fn flush_increments_count() {
        let mut f = MemoryFlusher::new();
        f.flush().unwrap();
        f.flush().unwrap();
        assert_eq!(f.flush_count(), 2);
    }

    #[test]
    fn pending_error_taken_once() {
        let mut f = MemoryFlusher::new();
        f.set_next_error(FlusherError::Io("simulated"));
        let err = f.write_line("x", true).unwrap_err();
        assert_eq!(err, FlusherError::Io("simulated"));
        // Next write succeeds.
        f.write_line("x", true).unwrap();
        assert_eq!(f.written_lines().len(), 1);
    }

    #[test]
    fn clear_resets_counts() {
        let mut f = MemoryFlusher::new();
        f.write_line("a", true).unwrap();
        f.flush().unwrap();
        assert!(f.flush_count() > 0);
        f.clear();
        assert_eq!(f.flush_count(), 0);
        assert_eq!(f.immediate_count(), 0);
        assert!(f.written_lines().is_empty());
    }

    #[test]
    fn flusher_error_display() {
        assert_eq!(
            alloc::format!("{}", FlusherError::Io("disk full")),
            "audit io: disk full"
        );
        assert_eq!(
            alloc::format!("{}", FlusherError::NotUtf8),
            "audit line not utf-8"
        );
    }

    #[test]
    fn immediate_flag_increments_immediate_count() {
        let mut f = MemoryFlusher::new();
        for _ in 0..5 {
            f.write_line("x", true).unwrap();
        }
        for _ in 0..3 {
            f.write_line("x", false).unwrap();
        }
        assert_eq!(f.immediate_count(), 5);
        assert_eq!(f.flush_count(), 5);
        assert_eq!(f.written_lines().len(), 8);
    }
}
