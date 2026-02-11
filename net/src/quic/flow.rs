// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QUIC flow control (RFC 9000 Section 4).
//!
//! Implements connection-level and stream-level flow control.
//! - Connection level: MAX_DATA limits total bytes across all streams.
//! - Stream level: MAX_STREAM_DATA limits bytes on a single stream.

use crate::NetError;

/// Connection-level flow control state.
#[derive(Debug)]
pub struct ConnectionFlowControl {
    /// Maximum data the peer allows us to send (their advertised limit).
    max_send_data: u64,
    /// Total data we have sent across all streams.
    total_sent: u64,
    /// Maximum data we allow the peer to send to us.
    max_recv_data: u64,
    /// Total data received across all streams.
    total_received: u64,
    /// Threshold for auto-updating receive window (when half consumed).
    auto_update_threshold: u64,
    /// Whether we need to send a MAX_DATA frame.
    max_data_needed: bool,
}

impl ConnectionFlowControl {
    /// Create new connection-level flow control.
    pub fn new(initial_max_send: u64, initial_max_recv: u64) -> Self {
        Self {
            max_send_data: initial_max_send,
            total_sent: 0,
            max_recv_data: initial_max_recv,
            total_received: 0,
            auto_update_threshold: initial_max_recv / 2,
            max_data_needed: false,
        }
    }

    /// Check if we can send `bytes` more data.
    pub fn can_send(&self, bytes: u64) -> bool {
        self.total_sent + bytes <= self.max_send_data
    }

    /// Available send window.
    pub fn send_window(&self) -> u64 {
        self.max_send_data.saturating_sub(self.total_sent)
    }

    /// Record that we sent data.
    pub fn on_data_sent(&mut self, bytes: u64) -> Result<(), NetError> {
        if self.total_sent + bytes > self.max_send_data {
            return Err(NetError::BufferTooSmall);
        }
        self.total_sent += bytes;
        Ok(())
    }

    /// Record that we received data.
    pub fn on_data_received(&mut self, bytes: u64) -> Result<(), NetError> {
        if self.total_received + bytes > self.max_recv_data {
            return Err(NetError::BufferTooSmall);
        }
        self.total_received += bytes;

        // Auto-update window if half consumed
        if self.total_received > self.auto_update_threshold {
            self.max_recv_data += self.total_received;
            self.auto_update_threshold = self.max_recv_data / 2;
            self.max_data_needed = true;
        }

        Ok(())
    }

    /// Update the peer's MAX_DATA limit (received in MAX_DATA frame).
    pub fn update_max_send_data(&mut self, max: u64) {
        if max > self.max_send_data {
            self.max_send_data = max;
        }
    }

    /// Get the current max recv data to advertise.
    pub fn max_recv_data(&self) -> u64 {
        self.max_recv_data
    }

    /// Check if we need to send a MAX_DATA update.
    pub fn needs_max_data_update(&self) -> bool {
        self.max_data_needed
    }

    /// Clear the MAX_DATA update flag.
    pub fn clear_max_data_needed(&mut self) {
        self.max_data_needed = false;
    }

    /// Total sent bytes.
    pub fn total_sent(&self) -> u64 {
        self.total_sent
    }

    /// Total received bytes.
    pub fn total_received(&self) -> u64 {
        self.total_received
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let fc = ConnectionFlowControl::new(1024, 2048);
        assert_eq!(fc.send_window(), 1024);
        assert_eq!(fc.total_sent(), 0);
        assert_eq!(fc.total_received(), 0);
    }

    #[test]
    fn test_can_send() {
        let fc = ConnectionFlowControl::new(100, 100);
        assert!(fc.can_send(100));
        assert!(!fc.can_send(101));
    }

    #[test]
    fn test_on_data_sent() {
        let mut fc = ConnectionFlowControl::new(100, 100);
        fc.on_data_sent(50).unwrap();
        assert_eq!(fc.send_window(), 50);
        assert_eq!(fc.total_sent(), 50);
        fc.on_data_sent(50).unwrap();
        assert_eq!(fc.send_window(), 0);
        assert_eq!(fc.on_data_sent(1).err(), Some(NetError::BufferTooSmall));
    }

    #[test]
    fn test_on_data_received() {
        let mut fc = ConnectionFlowControl::new(100, 100);
        fc.on_data_received(40).unwrap();
        assert_eq!(fc.total_received(), 40);
    }

    #[test]
    fn test_receive_over_limit() {
        let mut fc = ConnectionFlowControl::new(100, 100);
        assert_eq!(
            fc.on_data_received(101).err(),
            Some(NetError::BufferTooSmall)
        );
    }

    #[test]
    fn test_update_max_send_data() {
        let mut fc = ConnectionFlowControl::new(100, 100);
        fc.on_data_sent(100).unwrap();
        assert_eq!(fc.send_window(), 0);
        fc.update_max_send_data(200);
        assert_eq!(fc.send_window(), 100);
    }

    #[test]
    fn test_update_max_send_data_no_decrease() {
        let mut fc = ConnectionFlowControl::new(100, 100);
        fc.update_max_send_data(50); // Should not decrease
        assert_eq!(fc.send_window(), 100);
    }

    #[test]
    fn test_auto_update_recv_window() {
        let mut fc = ConnectionFlowControl::new(100, 100);
        // Receive more than half (threshold = 50)
        fc.on_data_received(60).unwrap();
        assert!(fc.needs_max_data_update());
        assert!(fc.max_recv_data() > 100);
    }

    #[test]
    fn test_clear_max_data_needed() {
        let mut fc = ConnectionFlowControl::new(100, 100);
        fc.on_data_received(60).unwrap();
        assert!(fc.needs_max_data_update());
        fc.clear_max_data_needed();
        assert!(!fc.needs_max_data_update());
    }
}
