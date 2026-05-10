// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Streaming-upload adapter for `model_add` over Zenoh.
//!
//! Spec:
//! `openspec/changes/embedded-overlay-v1/specs/fs-overlay-syscalls/spec.md`,
//! Phase 4 task 4.5 ("Streaming upload over Zenoh queryable for
//! `model_add` payloads").
//!
//! ## Wire model
//!
//! For an upload, the operator-side client opens a queryable
//! subscription on
//!
//! ```text
//! smallaios/admin/model/add/<request_id>/chunks
//! ```
//!
//! and pushes a sequence of chunk frames terminated by an empty frame
//! (or an explicit `done = true` flag — both are accepted; see
//! [`ChunkFrame`]). Chunks are fixed-size 4 KiB by convention but the
//! adapter treats every frame as opaque bytes.
//!
//! On the kernel side, [`ZenohContentsFd`] implements the kernel's
//! [`ContentSource`] trait and pulls one frame at a time as the kernel
//! consumes bytes. Backpressure is provided by Zenoh's natural
//! queryable flow control — the client cannot push faster than the
//! server consumes.
//!
//! ## Mock for tests
//!
//! [`MockChunkSource`] simulates the wire path with a `Vec<Vec<u8>>` of
//! pre-staged chunks and an optional disconnect-after-N-bytes flag
//! used by the partial-upload tests.

use alloc::collections::VecDeque;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use smallaios_kernel::syscall::model::ContentSource;

/// Default chunk size used by the wire (4 KiB). Documented here for
/// the operator CLI / Zenoh shim — the adapter itself does not enforce
/// any specific chunk size.
pub const ZENOH_CHUNK_SIZE: usize = 4096;

/// Hard cap on the per-upload chunk count. Matches the overlay
/// capacity cap of 1 GiB at 4 KiB chunks (~262 144 frames). Beyond
/// this the adapter SHALL reject further frames with
/// [`ChunkPushError::TooManyChunks`].
pub const MAX_CHUNKS_PER_UPLOAD: usize = 1 << 18;

/// One chunk on the wire.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkFrame {
    /// Chunk index, monotonic from `0`. Used only for diagnostics —
    /// the adapter does NOT reorder out-of-order frames; Zenoh
    /// queryables are ordered.
    pub seq: u32,
    /// Chunk payload bytes (may be empty as a sentinel for `done`).
    pub bytes: Vec<u8>,
    /// `true` iff this is the final frame for the upload.
    pub done: bool,
}

/// Errors raised when a producer tries to push a chunk to a
/// [`MockChunkSource`] that has already been closed or filled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkPushError {
    /// The source has already been marked `done` (or disconnected).
    Closed,
    /// `MAX_CHUNKS_PER_UPLOAD` frames already buffered.
    TooManyChunks,
}

/// Trait defining a chunk-by-chunk streaming source (the queryable
/// callback wraps one of these). Splits responsibilities between
/// "give me the next chunk" (sync, since the kernel-side caller is
/// itself synchronous) and "tell me whether you're done".
pub trait ChunkSource {
    /// Pull the next chunk. Returns `Ok(None)` on EOF, `Ok(Some(bytes))`
    /// on a successful pull, `Err(_)` on a transport-level disconnect.
    #[allow(clippy::result_unit_err)]
    fn pull(&mut self) -> Result<Option<Vec<u8>>, ChunkPullError>;
}

/// Errors raised by [`ChunkSource::pull`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChunkPullError {
    /// The Zenoh peer disconnected mid-upload.
    PeerDisconnected,
    /// Transport reported a corrupt frame.
    CorruptFrame,
    /// Per-upload chunk-count cap exceeded.
    TooManyChunks,
    /// Implementation-specific transport error.
    Transport(&'static str),
}

/// Mock chunk source backed by a deque, for unit tests + the local
/// CLI's `--remote` path which loads the whole file into memory.
#[derive(Debug, Default)]
pub struct MockChunkSource {
    queue: VecDeque<Vec<u8>>,
    closed: bool,
    /// If `Some(n)`, the `n`th `pull` call SHALL return
    /// [`ChunkPullError::PeerDisconnected`] regardless of remaining
    /// queued chunks. Used by the disconnect-mid-stream tests.
    disconnect_after_pulls: Option<usize>,
    pulls: usize,
}

impl MockChunkSource {
    /// Construct an empty source. Use [`Self::push`] to enqueue chunks
    /// and [`Self::close`] to mark EOF.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct a source pre-populated with the chunks of `bytes`,
    /// each `chunk_size` bytes (last chunk may be shorter).
    pub fn from_bytes(bytes: &[u8], chunk_size: usize) -> Self {
        let mut s = Self::new();
        if chunk_size == 0 {
            s.close();
            return s;
        }
        let mut i = 0;
        while i < bytes.len() {
            let end = core::cmp::min(i + chunk_size, bytes.len());
            // Push directly without going through the cap check —
            // the test fixture is in control of its own size.
            s.queue.push_back(bytes[i..end].to_vec());
            i = end;
        }
        s.close();
        s
    }

    /// Schedule a peer-disconnect after the given number of `pull`
    /// calls (0 ≡ disconnect on the very first pull).
    pub fn schedule_disconnect_after(&mut self, after_pulls: usize) {
        self.disconnect_after_pulls = Some(after_pulls);
    }

    /// Push one chunk into the queue. `Err` if already closed or the
    /// per-upload cap is reached.
    pub fn push(&mut self, chunk: Vec<u8>) -> Result<(), ChunkPushError> {
        if self.closed {
            return Err(ChunkPushError::Closed);
        }
        if self.queue.len() >= MAX_CHUNKS_PER_UPLOAD {
            return Err(ChunkPushError::TooManyChunks);
        }
        self.queue.push_back(chunk);
        Ok(())
    }

    /// Mark the source closed — subsequent pulls drain the queue then
    /// return `Ok(None)`.
    pub fn close(&mut self) {
        self.closed = true;
    }

    /// Number of chunks remaining in the queue.
    pub fn remaining(&self) -> usize {
        self.queue.len()
    }

    /// `true` iff [`Self::close`] was called.
    pub fn is_closed(&self) -> bool {
        self.closed
    }
}

impl ChunkSource for MockChunkSource {
    fn pull(&mut self) -> Result<Option<Vec<u8>>, ChunkPullError> {
        let n = self.pulls;
        self.pulls = self.pulls.saturating_add(1);
        if let Some(after) = self.disconnect_after_pulls {
            if n >= after {
                return Err(ChunkPullError::PeerDisconnected);
            }
        }
        match self.queue.pop_front() {
            Some(bytes) => Ok(Some(bytes)),
            None => {
                if self.closed {
                    Ok(None)
                } else {
                    // The producer has not yet pushed the next chunk
                    // and has not closed — surface a transport stall
                    // as a corrupt-frame for the kernel-side abort
                    // path. Real Zenoh queryables block on pull, so
                    // this branch is reached only for the unit-test
                    // "starved producer" fixture.
                    Err(ChunkPullError::Transport("starved producer"))
                }
            }
        }
    }
}

/// Adapter implementing the kernel's [`ContentSource`] trait over a
/// [`ChunkSource`]. Holds a small carry buffer for the case where the
/// kernel's read buffer is smaller than the current chunk.
pub struct ZenohContentsFd<S: ChunkSource> {
    source: S,
    /// Bytes carried over from a previously-pulled chunk that did not
    /// fit into the kernel's read buffer.
    carry: Vec<u8>,
    /// Carry read cursor.
    carry_pos: usize,
    /// Sticky EOF flag — once set, every read returns 0.
    eof: bool,
    /// Total bytes streamed (for diagnostics).
    bytes_streamed: u64,
    /// Last error captured from the underlying source. Surfaced via
    /// [`Self::last_error`] so the verb handler can record it in audit
    /// after the kernel returns `-EIO`.
    last_error: Option<ChunkPullError>,
}

impl<S: ChunkSource> ZenohContentsFd<S> {
    /// Wrap a [`ChunkSource`] into a kernel-compatible [`ContentSource`].
    pub fn new(source: S) -> Self {
        Self {
            source,
            carry: Vec::new(),
            carry_pos: 0,
            eof: false,
            bytes_streamed: 0,
            last_error: None,
        }
    }

    /// Total bytes successfully streamed to the kernel.
    pub fn bytes_streamed(&self) -> u64 {
        self.bytes_streamed
    }

    /// Last [`ChunkPullError`] observed (or `None` if upload was
    /// clean). Cleared by `Self::clear_error` for re-use.
    pub fn last_error(&self) -> Option<&ChunkPullError> {
        self.last_error.as_ref()
    }

    /// Clear the captured last error.
    pub fn clear_error(&mut self) {
        self.last_error = None;
    }

    /// Borrow the wrapped source (for diagnostics in tests).
    pub fn source(&self) -> &S {
        &self.source
    }
}

impl<S: ChunkSource> ContentSource for ZenohContentsFd<S> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, ()> {
        if self.eof {
            return Ok(0);
        }
        if buf.is_empty() {
            return Ok(0);
        }
        // Drain carry first.
        if self.carry_pos < self.carry.len() {
            let avail = self.carry.len() - self.carry_pos;
            let n = core::cmp::min(avail, buf.len());
            buf[..n].copy_from_slice(&self.carry[self.carry_pos..self.carry_pos + n]);
            self.carry_pos += n;
            self.bytes_streamed = self.bytes_streamed.saturating_add(n as u64);
            return Ok(n);
        }
        // Carry empty — pull the next chunk.
        match self.source.pull() {
            Ok(Some(chunk)) => {
                if chunk.is_empty() {
                    // Empty frame is a no-op; loop once via recursion.
                    return self.read(buf);
                }
                let n = core::cmp::min(chunk.len(), buf.len());
                buf[..n].copy_from_slice(&chunk[..n]);
                if n < chunk.len() {
                    self.carry = chunk;
                    self.carry_pos = n;
                } else {
                    self.carry.clear();
                    self.carry_pos = 0;
                }
                self.bytes_streamed = self.bytes_streamed.saturating_add(n as u64);
                Ok(n)
            }
            Ok(None) => {
                self.eof = true;
                Ok(0)
            }
            Err(e) => {
                self.last_error = Some(e);
                Err(())
            }
        }
    }
}

/// Helper used by the verb-handler's audit-on-failure path: render a
/// [`ChunkPullError`] into a stable string suitable for the audit
/// reason field.
pub fn chunk_pull_reason(e: &ChunkPullError) -> String {
    match e {
        ChunkPullError::PeerDisconnected => "peer disconnected".to_string(),
        ChunkPullError::CorruptFrame => "corrupt chunk frame".to_string(),
        ChunkPullError::TooManyChunks => "per-upload chunk cap exceeded".to_string(),
        ChunkPullError::Transport(s) => {
            let mut out = String::from("transport: ");
            out.push_str(s);
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bytes(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i as u8).wrapping_mul(31)).collect()
    }

    #[test]
    fn mock_source_drains_in_order() {
        let mut s = MockChunkSource::new();
        s.push(b"abc".to_vec()).unwrap();
        s.push(b"def".to_vec()).unwrap();
        s.close();
        assert_eq!(s.pull().unwrap(), Some(b"abc".to_vec()));
        assert_eq!(s.pull().unwrap(), Some(b"def".to_vec()));
        assert_eq!(s.pull().unwrap(), None);
    }

    #[test]
    fn mock_source_rejects_push_after_close() {
        let mut s = MockChunkSource::new();
        s.close();
        assert_eq!(s.push(b"x".to_vec()), Err(ChunkPushError::Closed));
    }

    #[test]
    fn mock_source_from_bytes_chunks_evenly() {
        let s = MockChunkSource::from_bytes(b"AAAABBBBCC", 4);
        assert_eq!(s.remaining(), 3);
        assert!(s.is_closed());
    }

    #[test]
    fn mock_source_from_bytes_handles_zero_chunk_size() {
        let s = MockChunkSource::from_bytes(b"abc", 0);
        assert_eq!(s.remaining(), 0);
        assert!(s.is_closed());
    }

    #[test]
    fn mock_source_starvation_returns_transport_error() {
        let mut s = MockChunkSource::new();
        // Don't close; pull immediately.
        let err = s.pull().unwrap_err();
        assert_eq!(err, ChunkPullError::Transport("starved producer"));
    }

    #[test]
    fn mock_source_disconnect_after_n_pulls_fires() {
        let mut s = MockChunkSource::new();
        s.push(b"abc".to_vec()).unwrap();
        s.push(b"def".to_vec()).unwrap();
        s.close();
        s.schedule_disconnect_after(1);
        assert_eq!(s.pull().unwrap(), Some(b"abc".to_vec()));
        assert_eq!(s.pull().unwrap_err(), ChunkPullError::PeerDisconnected);
    }

    #[test]
    fn zenoh_contents_fd_streams_aligned_buffers() {
        let bytes = make_bytes(16);
        let src = MockChunkSource::from_bytes(&bytes, 4);
        let mut fd = ZenohContentsFd::new(src);
        let mut out = Vec::new();
        let mut buf = [0u8; 4];
        loop {
            let n = ContentSource::read(&mut fd, &mut buf).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
        }
        assert_eq!(out, bytes);
        assert_eq!(fd.bytes_streamed(), 16);
    }

    #[test]
    fn zenoh_contents_fd_streams_misaligned_buffers() {
        let bytes = make_bytes(13);
        let src = MockChunkSource::from_bytes(&bytes, 4);
        let mut fd = ZenohContentsFd::new(src);
        let mut out = Vec::new();
        let mut buf = [0u8; 3];
        loop {
            let n = ContentSource::read(&mut fd, &mut buf).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&buf[..n]);
        }
        assert_eq!(out, bytes);
        assert_eq!(fd.bytes_streamed(), 13);
    }

    #[test]
    fn zenoh_contents_fd_handles_carry_when_buf_smaller_than_chunk() {
        let src = MockChunkSource::from_bytes(b"abcdefgh", 8);
        let mut fd = ZenohContentsFd::new(src);
        let mut buf = [0u8; 3];
        // First read draws 3 bytes from the 8-byte chunk; carry holds
        // the rest.
        assert_eq!(ContentSource::read(&mut fd, &mut buf).unwrap(), 3);
        assert_eq!(&buf, b"abc");
        assert_eq!(ContentSource::read(&mut fd, &mut buf).unwrap(), 3);
        assert_eq!(&buf, b"def");
        assert_eq!(ContentSource::read(&mut fd, &mut buf).unwrap(), 2);
        assert_eq!(&buf[..2], b"gh");
        assert_eq!(ContentSource::read(&mut fd, &mut buf).unwrap(), 0);
    }

    #[test]
    fn zenoh_contents_fd_propagates_disconnect_as_io_error() {
        let mut src = MockChunkSource::new();
        src.push(b"abc".to_vec()).unwrap();
        src.schedule_disconnect_after(1);
        // Don't close — disconnect fires after the first pull.
        let mut fd = ZenohContentsFd::new(src);
        let mut buf = [0u8; 3];
        // First read drains the queued chunk.
        assert_eq!(ContentSource::read(&mut fd, &mut buf).unwrap(), 3);
        // Second read triggers disconnect.
        let r = ContentSource::read(&mut fd, &mut buf);
        assert_eq!(r, Err(()));
        assert_eq!(fd.last_error(), Some(&ChunkPullError::PeerDisconnected));
    }

    #[test]
    fn zenoh_contents_fd_eof_is_sticky() {
        let src = MockChunkSource::from_bytes(b"a", 1);
        let mut fd = ZenohContentsFd::new(src);
        let mut buf = [0u8; 4];
        assert_eq!(ContentSource::read(&mut fd, &mut buf).unwrap(), 1);
        assert_eq!(ContentSource::read(&mut fd, &mut buf).unwrap(), 0);
        assert_eq!(ContentSource::read(&mut fd, &mut buf).unwrap(), 0);
    }

    #[test]
    fn zenoh_contents_fd_empty_buf_is_zero() {
        let src = MockChunkSource::from_bytes(b"abc", 1);
        let mut fd = ZenohContentsFd::new(src);
        let mut empty: [u8; 0] = [];
        assert_eq!(ContentSource::read(&mut fd, &mut empty).unwrap(), 0);
    }

    #[test]
    fn zenoh_contents_fd_skips_empty_frames() {
        let mut src = MockChunkSource::new();
        src.push(Vec::new()).unwrap();
        src.push(b"abc".to_vec()).unwrap();
        src.push(Vec::new()).unwrap();
        src.close();
        let mut fd = ZenohContentsFd::new(src);
        let mut buf = [0u8; 4];
        assert_eq!(ContentSource::read(&mut fd, &mut buf).unwrap(), 3);
        assert_eq!(&buf[..3], b"abc");
        assert_eq!(ContentSource::read(&mut fd, &mut buf).unwrap(), 0);
    }

    #[test]
    fn zenoh_contents_fd_clear_error_resets() {
        let mut src = MockChunkSource::new();
        src.schedule_disconnect_after(0);
        let mut fd = ZenohContentsFd::new(src);
        let mut buf = [0u8; 4];
        let _ = ContentSource::read(&mut fd, &mut buf);
        assert!(fd.last_error().is_some());
        fd.clear_error();
        assert!(fd.last_error().is_none());
    }

    #[test]
    fn chunk_pull_reason_renders_each_variant() {
        assert_eq!(
            chunk_pull_reason(&ChunkPullError::PeerDisconnected),
            "peer disconnected"
        );
        assert_eq!(
            chunk_pull_reason(&ChunkPullError::CorruptFrame),
            "corrupt chunk frame"
        );
        assert_eq!(
            chunk_pull_reason(&ChunkPullError::TooManyChunks),
            "per-upload chunk cap exceeded"
        );
        assert_eq!(
            chunk_pull_reason(&ChunkPullError::Transport("X")),
            "transport: X"
        );
    }

    #[test]
    fn chunk_frame_constructs_with_seq_and_done() {
        let frame = ChunkFrame {
            seq: 1,
            bytes: b"abc".to_vec(),
            done: true,
        };
        assert_eq!(frame.seq, 1);
        assert_eq!(frame.bytes, b"abc");
        assert!(frame.done);
    }
}
