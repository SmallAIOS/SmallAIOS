// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! UART line discipline — per-read echo / raw-mode options.
//!
//! Spec: `openspec/changes/management-login-v1/specs/peripheral-uart/spec.md`
//!
//! This module implements the in-line `read_line` API required by the
//! `console-login` flow (`management-login-v1` Phase 4). The hardware UART
//! driver reads bytes from the wire (via [`ByteSource`]) and echoes
//! processed characters back through [`ByteSink`]; the line discipline
//! sits between them and:
//!
//! - Buffers a single line until newline (`\r` or `\n`).
//! - Honors backspace (`0x7F` or `0x08`) by trimming the buffer.
//! - Honors Ctrl-U (`0x15`, "kill line") by clearing the buffer.
//! - Honors Ctrl-C (`0x03`, "abort") by returning [`LineError::Aborted`].
//! - Honors Ctrl-D (`0x04`, EOF) at an empty line by returning
//!   [`LineError::Eof`].
//! - When `echo=false`, suppresses ALL output — not even a `*` placeholder
//!   — to avoid leaking password length.
//! - When `raw=true`, delivers control characters verbatim into the buffer
//!   instead of interpreting them.
//!
//! ## Stateless across reads
//!
//! [`ReadOptions`] is passed by value on every call. The driver SHALL
//! NOT cache mode state between reads — a default read after an
//! echo-off read MUST re-echo. This is verified by
//! [`tests::echo_state_does_not_leak_across_reads`].
//!
//! ## Why a free function
//!
//! The spec hands us:
//!
//! ```rust,ignore
//! pub fn read_line(buf: &mut [u8], opts: ReadOptions) -> Result<usize, Error>;
//! ```
//!
//! We implement that as a thin wrapper over [`LineDiscipline::run`] which
//! takes any [`ByteSource`] / [`ByteSink`] pair. Tests plug in
//! [`MockSource`] / [`MockSink`]; production wiring (Phase 5+) will plug
//! the [`super::pl011::Pl011Uart`] / [`super::ns16550a`] / etc. drivers
//! into the trait.

#![allow(dead_code)]

use alloc::vec::Vec;

// ─── Public API ──────────────────────────────────────────────────────────────

/// Per-read options.
///
/// Defaults are `echo=true, raw=false, max_len=usize::MAX` — byte-identical
/// to the legacy line-buffered behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadOptions {
    /// When `false`, no character (not even a placeholder) SHALL be
    /// echoed for any keypress. Default: `true`.
    pub echo: bool,
    /// When `true`, control characters (backspace, Ctrl-U, Ctrl-C,
    /// Ctrl-D) SHALL be delivered verbatim into the buffer. Default:
    /// `false`.
    pub raw: bool,
    /// Hard cap on bytes accepted. Once reached, additional non-newline
    /// keypresses are ignored (and not echoed). Default: `usize::MAX`,
    /// which effectively disables the cap.
    pub max_len: usize,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            echo: true,
            raw: false,
            max_len: usize::MAX,
        }
    }
}

impl ReadOptions {
    /// Helper: line-edited password prompt (`echo=false, raw=false`,
    /// 1024-byte cap).
    pub const fn password() -> Self {
        Self {
            echo: false,
            raw: false,
            max_len: 1024,
        }
    }

    /// Helper: raw single-keypress read (`echo=false, raw=true`, 1-byte
    /// cap). Used by the idle-keypress reset path so any keystroke
    /// (including arrow keys, ^L) refreshes the timer.
    pub const fn raw_one() -> Self {
        Self {
            echo: false,
            raw: true,
            max_len: 1,
        }
    }
}

/// Errors raised by [`LineDiscipline::run`] / [`read_line`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineError {
    /// The operator pressed Ctrl-C; the caller SHALL discard any partial
    /// buffer and re-prompt as appropriate. This is **not** a "failed
    /// attempt" — see `console-login` spec scenario "^C aborts the
    /// prompt".
    Aborted,
    /// EOF (Ctrl-D pressed at an empty line). For the shell prompt this
    /// behaves identically to `logout`. For the password prompt it is
    /// equivalent to abort.
    Eof,
    /// The underlying [`ByteSource`] errored.
    Io(&'static str),
    /// The caller-supplied destination buffer was too small to hold the
    /// completed line.
    BufferTooSmall,
}

/// Read source — yields one byte at a time. `Ok(None)` means EOF on the
/// wire (distinct from Ctrl-D-as-byte). `Err` is a hardware fault.
pub trait ByteSource {
    /// Read one byte; block until available or return `Ok(None)` on EOF.
    fn read_byte(&mut self) -> Result<Option<u8>, &'static str>;
}

/// Echo sink — accepts a byte to display.
pub trait ByteSink {
    /// Write one byte to the sink.
    fn write_byte(&mut self, b: u8) -> Result<(), &'static str>;
}

/// Convenience: read a complete line from `src`, echoing through `sink`,
/// per the spec's `read_line(buf, opts)` shape. Returns the number of
/// bytes deposited in `buf`.
pub fn read_line<S: ByteSource, K: ByteSink>(
    buf: &mut [u8],
    opts: ReadOptions,
    src: &mut S,
    sink: &mut K,
) -> Result<usize, LineError> {
    let mut disc = LineDiscipline::new(opts);
    disc.run(src, sink)?;
    let line = disc.into_inner();
    if line.len() > buf.len() {
        return Err(LineError::BufferTooSmall);
    }
    buf[..line.len()].copy_from_slice(&line);
    Ok(line.len())
}

// ─── Line discipline state machine ───────────────────────────────────────────

/// Backspace (DEL, RFC 854 erase-character).
pub const KEY_DEL: u8 = 0x7F;
/// Backspace (BS).
pub const KEY_BS: u8 = 0x08;
/// Newline.
pub const KEY_LF: u8 = b'\n';
/// Carriage return.
pub const KEY_CR: u8 = b'\r';
/// Ctrl-C (abort prompt).
pub const KEY_CTRL_C: u8 = 0x03;
/// Ctrl-D (EOF).
pub const KEY_CTRL_D: u8 = 0x04;
/// Ctrl-U (kill line).
pub const KEY_CTRL_U: u8 = 0x15;

/// Stateful line discipline. The state lives on the stack of
/// [`run`](LineDiscipline::run); a fresh instance is constructed per
/// read, which is precisely the "stateless across reads" guarantee the
/// spec demands.
#[derive(Debug)]
pub struct LineDiscipline {
    opts: ReadOptions,
    buf: Vec<u8>,
}

impl LineDiscipline {
    /// Construct a discipline configured for `opts`.
    pub fn new(opts: ReadOptions) -> Self {
        Self {
            opts,
            buf: Vec::new(),
        }
    }

    /// Run until the operator submits a line (newline) or aborts.
    pub fn run<S: ByteSource, K: ByteSink>(
        &mut self,
        src: &mut S,
        sink: &mut K,
    ) -> Result<(), LineError> {
        loop {
            let b = match src.read_byte().map_err(LineError::Io)? {
                Some(b) => b,
                // Wire-EOF — treat as Ctrl-D (logout).
                None => return Err(LineError::Eof),
            };

            if self.opts.raw {
                // Raw mode: deliver every byte verbatim. Newline still
                // terminates the read so the caller can dispatch.
                if b == KEY_LF || b == KEY_CR {
                    return Ok(());
                }
                if self.buf.len() < self.opts.max_len {
                    self.buf.push(b);
                    self.echo(sink, b)?;
                }
                continue;
            }

            match b {
                KEY_LF | KEY_CR => {
                    // Echo a newline so the next prompt starts on a
                    // clean line. Skip when echo is suppressed.
                    if self.opts.echo {
                        sink.write_byte(b'\r').map_err(LineError::Io)?;
                        sink.write_byte(b'\n').map_err(LineError::Io)?;
                    }
                    return Ok(());
                }
                KEY_DEL | KEY_BS => {
                    if self.buf.pop().is_some() && self.opts.echo {
                        // Visual erase: BS, space, BS — clears the
                        // glyph on a real terminal.
                        sink.write_byte(0x08).map_err(LineError::Io)?;
                        sink.write_byte(b' ').map_err(LineError::Io)?;
                        sink.write_byte(0x08).map_err(LineError::Io)?;
                    }
                }
                KEY_CTRL_U => {
                    let n = self.buf.len();
                    self.buf.clear();
                    if self.opts.echo {
                        // Visually erase n chars.
                        for _ in 0..n {
                            sink.write_byte(0x08).map_err(LineError::Io)?;
                            sink.write_byte(b' ').map_err(LineError::Io)?;
                            sink.write_byte(0x08).map_err(LineError::Io)?;
                        }
                    }
                }
                KEY_CTRL_C => {
                    self.buf.clear();
                    return Err(LineError::Aborted);
                }
                KEY_CTRL_D => {
                    if self.buf.is_empty() {
                        return Err(LineError::Eof);
                    }
                    // Ctrl-D mid-line: ignore (matches readline /
                    // bash behaviour). The spec only mandates Ctrl-D
                    // at an empty prompt; non-empty Ctrl-D is a
                    // discard.
                }
                _ => {
                    if self.buf.len() < self.opts.max_len {
                        self.buf.push(b);
                        self.echo(sink, b)?;
                    }
                    // Past the cap: silently drop, no echo.
                }
            }
        }
    }

    /// Inner echo helper.
    fn echo<K: ByteSink>(&self, sink: &mut K, b: u8) -> Result<(), LineError> {
        if self.opts.echo {
            sink.write_byte(b).map_err(LineError::Io)?;
        }
        Ok(())
    }

    /// Borrow the accumulated line.
    pub fn buffer(&self) -> &[u8] {
        &self.buf
    }

    /// Consume the discipline and yield the accumulated line.
    pub fn into_inner(self) -> Vec<u8> {
        self.buf
    }
}

// ─── Test mocks ──────────────────────────────────────────────────────────────

/// In-memory [`ByteSource`] that yields a fixed script of bytes.
pub struct MockSource<'a> {
    script: &'a [u8],
    cursor: usize,
}

impl<'a> MockSource<'a> {
    /// Construct a mock source returning bytes from `script`.
    pub fn new(script: &'a [u8]) -> Self {
        Self { script, cursor: 0 }
    }
}

impl ByteSource for MockSource<'_> {
    fn read_byte(&mut self) -> Result<Option<u8>, &'static str> {
        if self.cursor >= self.script.len() {
            return Ok(None);
        }
        let b = self.script[self.cursor];
        self.cursor += 1;
        Ok(Some(b))
    }
}

/// In-memory [`ByteSink`] that records every echoed byte.
#[derive(Default)]
pub struct MockSink {
    out: Vec<u8>,
}

impl MockSink {
    /// Construct an empty sink.
    pub fn new() -> Self {
        Self::default()
    }

    /// Borrow the recorded echo trace.
    pub fn output(&self) -> &[u8] {
        &self.out
    }
}

impl ByteSink for MockSink {
    fn write_byte(&mut self, b: u8) -> Result<(), &'static str> {
        self.out.push(b);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run_read(opts: ReadOptions, script: &[u8]) -> (Vec<u8>, Vec<u8>) {
        let mut buf = [0u8; 256];
        let mut src = MockSource::new(script);
        let mut sink = MockSink::new();
        let n = read_line(&mut buf, opts, &mut src, &mut sink).unwrap();
        (buf[..n].to_vec(), sink.output().to_vec())
    }

    fn run_read_err(opts: ReadOptions, script: &[u8]) -> (LineError, Vec<u8>) {
        let mut buf = [0u8; 256];
        let mut src = MockSource::new(script);
        let mut sink = MockSink::new();
        let err = read_line(&mut buf, opts, &mut src, &mut sink).unwrap_err();
        (err, sink.output().to_vec())
    }

    #[test]
    fn default_options_round_trip_with_echo() {
        let (line, echo) = run_read(ReadOptions::default(), b"hello\n");
        assert_eq!(line, b"hello");
        // Each char echoed, then \r\n on submit.
        assert_eq!(echo, b"hello\r\n");
    }

    #[test]
    fn echo_off_password_emits_nothing() {
        // Spec scenario: "Password prompt suppresses echo"
        let (line, echo) = run_read(ReadOptions::password(), b"hunter2\n");
        assert_eq!(line, b"hunter2");
        assert!(
            echo.is_empty(),
            "echo-off SHALL NOT emit any byte, got {echo:?}"
        );
    }

    #[test]
    fn backspace_in_default_mode_visually_erases() {
        let (line, echo) = run_read(ReadOptions::default(), b"ab\x7Fc\n");
        assert_eq!(line, b"ac");
        // 'a' echoed, 'b' echoed, BS-space-BS, 'c' echoed, \r\n.
        assert_eq!(echo, b"ab\x08 \x08c\r\n");
    }

    #[test]
    fn backspace_in_echo_off_password_mode() {
        // Spec scenario: "Backspace honored in echo-off password mode"
        let (line, echo) = run_read(ReadOptions::password(), b"a\x7Fb\n");
        assert_eq!(line, b"b");
        assert!(echo.is_empty(), "no character SHALL be echoed");
    }

    #[test]
    fn backspace_at_start_of_buffer_is_noop() {
        let (line, echo) = run_read(ReadOptions::default(), b"\x7Fa\n");
        assert_eq!(line, b"a");
        // Just 'a' + newline — no BS-space-BS triplet.
        assert_eq!(echo, b"a\r\n");
    }

    #[test]
    fn ctrl_h_alias_for_backspace() {
        // Some terminals send 0x08 (BS) instead of 0x7F (DEL).
        let (line, _) = run_read(ReadOptions::default(), b"ab\x08c\n");
        assert_eq!(line, b"ac");
    }

    #[test]
    fn ctrl_u_kills_entire_line() {
        let (line, echo) = run_read(ReadOptions::default(), b"hello\x15world\n");
        assert_eq!(line, b"world");
        // Five chars erased visually (BS-space-BS each).
        assert!(echo.windows(3).filter(|w| *w == [0x08, b' ', 0x08]).count() >= 5);
    }

    #[test]
    fn ctrl_u_in_echo_off_mode_clears_buffer_silently() {
        let (line, echo) = run_read(ReadOptions::password(), b"abc\x15def\n");
        assert_eq!(line, b"def");
        assert!(echo.is_empty());
    }

    #[test]
    fn ctrl_c_aborts_with_aborted_error() {
        // Spec scenario: "^C aborts the prompt"
        let (err, echo) = run_read_err(ReadOptions::password(), b"halfway\x03");
        assert_eq!(err, LineError::Aborted);
        assert!(echo.is_empty(), "echo-off mode echoes nothing");
    }

    #[test]
    fn ctrl_c_in_default_mode_returns_aborted() {
        let (err, _) = run_read_err(ReadOptions::default(), b"\x03");
        assert_eq!(err, LineError::Aborted);
    }

    #[test]
    fn ctrl_d_at_empty_buffer_returns_eof() {
        let (err, _) = run_read_err(ReadOptions::default(), b"\x04");
        assert_eq!(err, LineError::Eof);
    }

    #[test]
    fn ctrl_d_with_buffered_chars_is_dropped() {
        // Spec note: Ctrl-D mid-line is a no-op (matches readline).
        let (line, _) = run_read(ReadOptions::default(), b"hi\x04\n");
        assert_eq!(line, b"hi");
    }

    #[test]
    fn raw_mode_delivers_control_chars_verbatim() {
        let opts = ReadOptions {
            echo: false,
            raw: true,
            max_len: 16,
        };
        let (line, _) = run_read(opts, b"a\x03b\x7Fc\n");
        // Newline terminates; everything before is buffered as-is.
        assert_eq!(line, b"a\x03b\x7Fc");
    }

    #[test]
    fn raw_mode_with_max_len_one_returns_after_first_byte() {
        // Used by idle-reset polling path.
        let mut buf = [0u8; 4];
        let mut src = MockSource::new(b"x\n");
        let mut sink = MockSink::new();
        let n = read_line(&mut buf, ReadOptions::raw_one(), &mut src, &mut sink).unwrap();
        assert_eq!(n, 1);
        assert_eq!(&buf[..n], b"x");
    }

    #[test]
    fn cr_terminates_just_like_lf() {
        let (line, _) = run_read(ReadOptions::default(), b"alpha\r");
        assert_eq!(line, b"alpha");
    }

    #[test]
    fn max_len_silently_drops_overflow() {
        let opts = ReadOptions {
            echo: true,
            raw: false,
            max_len: 3,
        };
        let (line, echo) = run_read(opts, b"abcdef\n");
        assert_eq!(line, b"abc");
        // Only "abc" echoed plus newline.
        assert_eq!(echo, b"abc\r\n");
    }

    #[test]
    fn echo_state_does_not_leak_across_reads() {
        // Spec scenario: "Stateless across reads"
        // First read with echo=false, then a default read — the
        // second SHALL behave with echo=true.
        let mut buf = [0u8; 32];
        let mut src = MockSource::new(b"secret\nhello\n");
        let mut sink = MockSink::new();
        // Echo-off pass.
        let _ = read_line(&mut buf, ReadOptions::password(), &mut src, &mut sink).unwrap();
        let after_pw_len = sink.output().len();
        assert_eq!(after_pw_len, 0, "password read echoed bytes");
        // Default pass — echo SHALL be active again.
        let n = read_line(&mut buf, ReadOptions::default(), &mut src, &mut sink).unwrap();
        assert_eq!(&buf[..n], b"hello");
        assert!(
            sink.output().ends_with(b"hello\r\n"),
            "default read SHALL echo: {:?}",
            sink.output()
        );
    }

    #[test]
    fn buffer_too_small_returns_buffer_too_small() {
        let mut buf = [0u8; 2];
        let mut src = MockSource::new(b"abcdef\n");
        let mut sink = MockSink::new();
        let err = read_line(&mut buf, ReadOptions::default(), &mut src, &mut sink).unwrap_err();
        assert_eq!(err, LineError::BufferTooSmall);
    }

    #[test]
    fn read_options_default_matches_legacy_behavior() {
        let opts = ReadOptions::default();
        assert!(opts.echo);
        assert!(!opts.raw);
        assert_eq!(opts.max_len, usize::MAX);
    }

    #[test]
    fn read_options_password_helper() {
        let opts = ReadOptions::password();
        assert!(!opts.echo);
        assert!(!opts.raw);
        assert_eq!(opts.max_len, 1024);
    }

    #[test]
    fn empty_line_just_newline() {
        let (line, echo) = run_read(ReadOptions::default(), b"\n");
        assert!(line.is_empty());
        assert_eq!(echo, b"\r\n");
    }

    #[test]
    fn io_error_propagates() {
        struct ErrSrc;
        impl ByteSource for ErrSrc {
            fn read_byte(&mut self) -> Result<Option<u8>, &'static str> {
                Err("hw fault")
            }
        }
        let mut buf = [0u8; 8];
        let mut sink = MockSink::new();
        let err = read_line(&mut buf, ReadOptions::default(), &mut ErrSrc, &mut sink).unwrap_err();
        match err {
            LineError::Io(s) => assert_eq!(s, "hw fault"),
            other => panic!("expected Io, got {other:?}"),
        }
    }

    #[test]
    fn ctrl_c_does_not_count_partial_buffer() {
        // After Ctrl-C the discipline is dropped — no buffer state
        // bleeds forward.
        let mut src = MockSource::new(b"halfway\x03restart\n");
        let mut sink = MockSink::new();
        let mut buf = [0u8; 32];

        let err = read_line(&mut buf, ReadOptions::password(), &mut src, &mut sink).unwrap_err();
        assert_eq!(err, LineError::Aborted);
        // Caller re-prompts; buffer of next read is independent.
        let n = read_line(&mut buf, ReadOptions::default(), &mut src, &mut sink).unwrap();
        assert_eq!(&buf[..n], b"restart");
    }

    #[test]
    fn eof_on_wire_returns_eof() {
        // Source exhausts before newline.
        let (err, _) = run_read_err(ReadOptions::default(), b"halfway");
        assert_eq!(err, LineError::Eof);
    }

    #[test]
    fn raw_mode_max_len_caps_buffer() {
        let opts = ReadOptions {
            echo: false,
            raw: true,
            max_len: 2,
        };
        let (line, _) = run_read(opts, b"abcd\n");
        assert_eq!(line, b"ab");
    }

    #[test]
    fn raw_mode_carriage_return_terminates() {
        let opts = ReadOptions {
            echo: false,
            raw: true,
            max_len: 16,
        };
        let (line, _) = run_read(opts, b"abc\r");
        assert_eq!(line, b"abc");
    }

    #[test]
    fn echo_off_backspace_at_empty_is_noop() {
        let (line, echo) = run_read(ReadOptions::password(), b"\x7Fab\n");
        assert_eq!(line, b"ab");
        assert!(echo.is_empty());
    }
}
