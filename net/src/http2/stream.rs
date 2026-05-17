// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Client-side HTTP/2 stream state machine (RFC 7540 § 5.1).
//!
//! We model only the states reachable by a client-initiated
//! odd-numbered stream issuing a single unary request:
//!
//! ```text
//!     Idle  ── send HEADERS ────►  Open
//!                                    │
//!                                    │  send END_STREAM (DATA or HEADERS)
//!                                    ▼
//!                              HalfClosedLocal
//!                                    │
//!                                    │  recv END_STREAM (HEADERS trailers)
//!                                    ▼
//!                                  Closed
//! ```
//!
//! `RST_STREAM` may close the stream from any non-Idle state.
//! Server-initiated transitions (push streams) are unreachable
//! because the connection refuses `PUSH_PROMISE`.

use super::{Http2Error, Result};

/// Client-stream lifecycle states we recognize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Idle,
    Open,
    HalfClosedLocal,
    Closed,
}

/// Reason a stream entered the Closed state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseReason {
    /// Both sides observed END_STREAM.
    Complete,
    /// Local side sent RST_STREAM.
    LocalReset,
    /// Remote side sent RST_STREAM.
    RemoteReset,
}

/// One client-initiated stream.
#[derive(Debug, Clone, Copy)]
pub struct Stream {
    pub id: u32,
    state: State,
    close_reason: Option<CloseReason>,
}

impl Stream {
    /// Validate that `id` is a legal client-initiated stream id and
    /// construct a stream in [`State::Idle`].
    pub fn new(id: u32) -> Result<Self> {
        // RFC 7540 § 5.1.1: client-initiated streams use odd ids; 0 is
        // reserved for connection-level frames.
        if id == 0 || id.is_multiple_of(2) {
            return Err(Http2Error::BadStreamId);
        }
        Ok(Self {
            id,
            state: State::Idle,
            close_reason: None,
        })
    }

    /// Current state.
    pub fn state(&self) -> State {
        self.state
    }

    /// Close reason if the stream is in [`State::Closed`].
    pub fn close_reason(&self) -> Option<CloseReason> {
        self.close_reason
    }

    /// Send a HEADERS frame from the client.
    pub fn send_headers(&mut self, end_stream: bool) -> Result<()> {
        match self.state {
            State::Idle => {
                self.state = if end_stream {
                    State::HalfClosedLocal
                } else {
                    State::Open
                };
                Ok(())
            }
            State::Open if end_stream => {
                self.state = State::HalfClosedLocal;
                Ok(())
            }
            _ => Err(Http2Error::BadStreamState),
        }
    }

    /// Send a DATA frame from the client.
    pub fn send_data(&mut self, end_stream: bool) -> Result<()> {
        match self.state {
            State::Open if !end_stream => Ok(()),
            State::Open if end_stream => {
                self.state = State::HalfClosedLocal;
                Ok(())
            }
            _ => Err(Http2Error::BadStreamState),
        }
    }

    /// Receive a HEADERS frame from the server. `end_stream` is the
    /// END_STREAM flag on the inbound frame (e.g. trailers carry it).
    pub fn recv_headers(&mut self, end_stream: bool) -> Result<()> {
        match self.state {
            State::Open | State::HalfClosedLocal => {
                if end_stream {
                    self.state = State::Closed;
                    self.close_reason = Some(CloseReason::Complete);
                }
                Ok(())
            }
            _ => Err(Http2Error::BadStreamState),
        }
    }

    /// Receive a DATA frame from the server.
    pub fn recv_data(&mut self, end_stream: bool) -> Result<()> {
        match self.state {
            State::Open | State::HalfClosedLocal => {
                if end_stream {
                    self.state = State::Closed;
                    self.close_reason = Some(CloseReason::Complete);
                }
                Ok(())
            }
            _ => Err(Http2Error::BadStreamState),
        }
    }

    /// Apply a locally-initiated RST_STREAM.
    pub fn local_reset(&mut self) {
        self.state = State::Closed;
        self.close_reason = Some(CloseReason::LocalReset);
    }

    /// Apply a remotely-initiated RST_STREAM.
    pub fn remote_reset(&mut self) {
        self.state = State::Closed;
        self.close_reason = Some(CloseReason::RemoteReset);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn id_zero_rejected() {
        assert_eq!(Stream::new(0).unwrap_err(), Http2Error::BadStreamId);
    }

    #[test]
    fn even_id_rejected() {
        assert_eq!(Stream::new(2).unwrap_err(), Http2Error::BadStreamId);
    }

    #[test]
    fn happy_path_unary() {
        let mut s = Stream::new(1).unwrap();
        s.send_headers(false).unwrap();
        s.send_data(true).unwrap();
        assert_eq!(s.state(), State::HalfClosedLocal);
        s.recv_headers(false).unwrap();
        s.recv_data(true).unwrap();
        assert_eq!(s.state(), State::Closed);
        assert_eq!(s.close_reason(), Some(CloseReason::Complete));
    }

    #[test]
    fn headers_with_end_stream_skips_open() {
        let mut s = Stream::new(3).unwrap();
        s.send_headers(true).unwrap();
        assert_eq!(s.state(), State::HalfClosedLocal);
    }

    #[test]
    fn double_send_after_end_rejected() {
        let mut s = Stream::new(1).unwrap();
        s.send_headers(true).unwrap();
        assert_eq!(s.send_data(false).unwrap_err(), Http2Error::BadStreamState);
    }

    #[test]
    fn recv_in_idle_rejected() {
        let mut s = Stream::new(1).unwrap();
        assert_eq!(
            s.recv_headers(false).unwrap_err(),
            Http2Error::BadStreamState
        );
    }

    #[test]
    fn local_reset_marks_closed() {
        let mut s = Stream::new(1).unwrap();
        s.send_headers(false).unwrap();
        s.local_reset();
        assert_eq!(s.state(), State::Closed);
        assert_eq!(s.close_reason(), Some(CloseReason::LocalReset));
    }
}
