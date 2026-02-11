// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QUIC stream management (RFC 9000 Sections 2-3).
//!
//! Implements bidirectional and unidirectional streams with:
//! - Stream ID allocation (client/server, bidi/uni)
//! - Per-stream flow control (MAX_STREAM_DATA)
//! - Stream state machine (Ready, Send, DataSent, ResetSent, DataRecvd)
//! - Concurrent stream limits (MAX_STREAMS)

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use crate::NetError;

/// Stream type (direction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamType {
    /// Bidirectional stream.
    Bidirectional,
    /// Unidirectional stream.
    Unidirectional,
}

/// Stream initiator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StreamInitiator {
    /// Client-initiated (even stream IDs).
    Client,
    /// Server-initiated (odd stream IDs).
    Server,
}

/// Stream send state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendState {
    /// Ready to send.
    Ready,
    /// Sending data.
    Send,
    /// All data sent, waiting for ACK.
    DataSent,
    /// Reset sent.
    ResetSent,
    /// All data acknowledged.
    DataRecvd,
}

/// Stream receive state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecvState {
    /// Receiving data.
    Recv,
    /// All data received (FIN received).
    SizeKnown,
    /// All data read by application.
    DataRead,
    /// Reset received from peer.
    ResetRecvd,
}

/// A single QUIC stream.
#[derive(Debug)]
pub struct Stream {
    /// Stream ID.
    id: u64,
    /// Stream type.
    stream_type: StreamType,
    /// Send state (None for receive-only unidirectional).
    send_state: Option<SendState>,
    /// Receive state (None for send-only unidirectional).
    recv_state: Option<RecvState>,
    /// Send buffer.
    send_buf: Vec<u8>,
    /// Send offset (how much has been sent).
    send_offset: u64,
    /// Receive buffer (offset -> data).
    recv_buf: BTreeMap<u64, Vec<u8>>,
    /// Next receive offset to deliver to application.
    recv_offset: u64,
    /// Whether FIN has been set on send side.
    send_fin: bool,
    /// Whether FIN has been received.
    recv_fin: bool,
    /// Final size (known after FIN received).
    final_size: Option<u64>,
    /// Max send data (peer's flow control limit for this stream).
    max_send_data: u64,
    /// Max recv data (our flow control limit we advertise).
    max_recv_data: u64,
    /// Total bytes received on this stream.
    total_recv: u64,
}

impl Stream {
    /// Create a new stream.
    fn new(id: u64, stream_type: StreamType, is_local: bool, initial_max_send: u64, initial_max_recv: u64) -> Self {
        let (send_state, recv_state) = match stream_type {
            StreamType::Bidirectional => (Some(SendState::Ready), Some(RecvState::Recv)),
            StreamType::Unidirectional => {
                if is_local {
                    // We initiated: we can send, peer receives
                    (Some(SendState::Ready), None)
                } else {
                    // Peer initiated: peer sends, we receive
                    (None, Some(RecvState::Recv))
                }
            }
        };

        Self {
            id,
            stream_type,
            send_state,
            recv_state,
            send_buf: Vec::new(),
            send_offset: 0,
            recv_buf: BTreeMap::new(),
            recv_offset: 0,
            send_fin: false,
            recv_fin: false,
            final_size: None,
            max_send_data: initial_max_send,
            max_recv_data: initial_max_recv,
            total_recv: 0,
        }
    }

    /// Stream ID.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Stream type.
    pub fn stream_type(&self) -> StreamType {
        self.stream_type
    }

    /// Whether this stream can send data.
    pub fn can_send(&self) -> bool {
        matches!(self.send_state, Some(SendState::Ready | SendState::Send))
    }

    /// Whether this stream can receive data.
    pub fn can_recv(&self) -> bool {
        matches!(self.recv_state, Some(RecvState::Recv | RecvState::SizeKnown))
    }

    /// Write data to the send buffer.
    pub fn write(&mut self, data: &[u8]) -> Result<usize, NetError> {
        if !self.can_send() {
            return Err(NetError::InvalidProtocol);
        }
        self.send_buf.extend_from_slice(data);
        if let Some(ref mut state) = self.send_state {
            if *state == SendState::Ready {
                *state = SendState::Send;
            }
        }
        Ok(data.len())
    }

    /// Mark the send side as finished (FIN).
    pub fn finish(&mut self) -> Result<(), NetError> {
        if !self.can_send() {
            return Err(NetError::InvalidProtocol);
        }
        self.send_fin = true;
        Ok(())
    }

    /// Get pending send data up to the flow control limit.
    pub fn pending_send(&self) -> Option<(u64, &[u8], bool)> {
        if !self.can_send() && !matches!(self.send_state, Some(SendState::DataSent)) {
            return None;
        }
        let available = self.max_send_data.saturating_sub(self.send_offset) as usize;
        let buf_remaining = self.send_buf.len() - self.send_offset as usize;
        if buf_remaining == 0 && !self.send_fin {
            return None;
        }
        let len = buf_remaining.min(available);
        let offset = self.send_offset;
        let data = &self.send_buf[offset as usize..offset as usize + len];
        let fin = self.send_fin && len == buf_remaining;
        Some((offset, data, fin))
    }

    /// Record that data was sent.
    pub fn on_data_sent(&mut self, bytes: u64, fin: bool) {
        self.send_offset += bytes;
        if fin {
            self.send_state = Some(SendState::DataSent);
        }
    }

    /// Record that sent data was acknowledged.
    pub fn on_data_acked(&mut self) {
        if self.send_state == Some(SendState::DataSent) {
            self.send_state = Some(SendState::DataRecvd);
        }
    }

    /// Receive data at the given offset.
    pub fn recv_data(&mut self, offset: u64, data: &[u8], fin: bool) -> Result<(), NetError> {
        if !self.can_recv() {
            return Err(NetError::InvalidProtocol);
        }

        let end = offset + data.len() as u64;

        // Check flow control
        if end > self.max_recv_data {
            return Err(NetError::BufferTooSmall);
        }

        if !data.is_empty() {
            self.recv_buf.insert(offset, Vec::from(data));
            if end > self.total_recv {
                self.total_recv = end;
            }
        }

        if fin {
            self.recv_fin = true;
            self.final_size = Some(end);
            if let Some(ref mut state) = self.recv_state {
                *state = RecvState::SizeKnown;
            }
        }

        Ok(())
    }

    /// Read received data (in order). Returns bytes read.
    pub fn read(&mut self, buf: &mut [u8]) -> Result<usize, NetError> {
        if !self.can_recv() && self.recv_state != Some(RecvState::SizeKnown) {
            return Err(NetError::InvalidProtocol);
        }

        let mut total = 0;

        // Collect contiguous data from recv_offset
        let keys: Vec<u64> = self.recv_buf.keys().copied().collect();
        for key in keys {
            if key > self.recv_offset {
                break; // Gap
            }
            let data = self.recv_buf.get(&key).unwrap();
            let start = (self.recv_offset - key) as usize;
            if start >= data.len() {
                // Already consumed
                continue;
            }
            let available = data.len() - start;
            let to_copy = available.min(buf.len() - total);
            buf[total..total + to_copy].copy_from_slice(&data[start..start + to_copy]);
            total += to_copy;
            self.recv_offset = key + start as u64 + to_copy as u64;

            if total >= buf.len() {
                break;
            }
        }

        // Clean up fully consumed entries
        let recv_offset = self.recv_offset;
        self.recv_buf.retain(|&offset, data| offset + data.len() as u64 > recv_offset);

        // Check if all data has been read
        if self.recv_fin && Some(self.recv_offset) == self.final_size {
            self.recv_state = Some(RecvState::DataRead);
        }

        Ok(total)
    }

    /// Reset the send side.
    pub fn reset_send(&mut self, _error_code: u64) {
        self.send_state = Some(SendState::ResetSent);
        self.send_buf.clear();
    }

    /// Handle peer's reset.
    pub fn recv_reset(&mut self, final_size: u64) {
        self.recv_state = Some(RecvState::ResetRecvd);
        self.final_size = Some(final_size);
    }

    /// Update the peer's flow control limit for sending.
    pub fn update_max_send_data(&mut self, max: u64) {
        if max > self.max_send_data {
            self.max_send_data = max;
        }
    }

    /// Get the current max receive data limit.
    pub fn max_recv_data(&self) -> u64 {
        self.max_recv_data
    }

    /// Increase our receive flow control limit.
    pub fn update_max_recv_data(&mut self, max: u64) {
        if max > self.max_recv_data {
            self.max_recv_data = max;
        }
    }

    /// Send state.
    pub fn send_state(&self) -> Option<SendState> {
        self.send_state
    }

    /// Receive state.
    pub fn recv_state(&self) -> Option<RecvState> {
        self.recv_state
    }

    /// Whether the stream is fully closed (both sides done).
    pub fn is_finished(&self) -> bool {
        let send_done = matches!(
            self.send_state,
            Some(SendState::DataRecvd | SendState::ResetSent) | None
        );
        let recv_done = matches!(
            self.recv_state,
            Some(RecvState::DataRead | RecvState::ResetRecvd) | None
        );
        send_done && recv_done
    }
}

/// Stream manager — handles all streams for a connection.
#[derive(Debug)]
pub struct StreamManager {
    /// All streams indexed by stream ID.
    streams: BTreeMap<u64, Stream>,
    /// Our role (determines which stream IDs we can initiate).
    is_client: bool,
    /// Next local bidi stream ID.
    next_bidi_id: u64,
    /// Next local uni stream ID.
    next_uni_id: u64,
    /// Max peer-allowed local bidi streams.
    max_local_bidi: u64,
    /// Max peer-allowed local uni streams.
    max_local_uni: u64,
    /// Max remote bidi streams we allow.
    max_remote_bidi: u64,
    /// Max remote uni streams we allow.
    max_remote_uni: u64,
    /// Current count of open local bidi streams.
    open_local_bidi: u64,
    /// Current count of open local uni streams.
    open_local_uni: u64,
    /// Current count of open remote bidi streams.
    open_remote_bidi: u64,
    /// Current count of open remote uni streams.
    open_remote_uni: u64,
    /// Default initial max stream send data.
    initial_max_stream_data_bidi_local: u64,
    initial_max_stream_data_bidi_remote: u64,
    initial_max_stream_data_uni: u64,
}

/// Decode stream ID into (initiator, type).
pub fn stream_id_type(id: u64) -> (StreamInitiator, StreamType) {
    let initiator = if id & 0x01 == 0 {
        StreamInitiator::Client
    } else {
        StreamInitiator::Server
    };
    let stype = if id & 0x02 == 0 {
        StreamType::Bidirectional
    } else {
        StreamType::Unidirectional
    };
    (initiator, stype)
}

impl StreamManager {
    /// Create a new stream manager.
    pub fn new(is_client: bool, max_remote_bidi: u64, max_remote_uni: u64) -> Self {
        let (first_bidi, first_uni) = if is_client {
            (0, 2)
        } else {
            (1, 3)
        };
        Self {
            streams: BTreeMap::new(),
            is_client,
            next_bidi_id: first_bidi,
            next_uni_id: first_uni,
            max_local_bidi: 0,
            max_local_uni: 0,
            max_remote_bidi,
            max_remote_uni,
            open_local_bidi: 0,
            open_local_uni: 0,
            open_remote_bidi: 0,
            open_remote_uni: 0,
            initial_max_stream_data_bidi_local: 262_144,
            initial_max_stream_data_bidi_remote: 262_144,
            initial_max_stream_data_uni: 262_144,
        }
    }

    /// Set the peer's stream limits (from transport parameters).
    pub fn set_peer_max_streams(&mut self, bidi: u64, uni: u64) {
        self.max_local_bidi = bidi;
        self.max_local_uni = uni;
    }

    /// Set initial max stream data values.
    pub fn set_initial_max_stream_data(
        &mut self,
        bidi_local: u64,
        bidi_remote: u64,
        uni: u64,
    ) {
        self.initial_max_stream_data_bidi_local = bidi_local;
        self.initial_max_stream_data_bidi_remote = bidi_remote;
        self.initial_max_stream_data_uni = uni;
    }

    /// Open a new local bidirectional stream.
    pub fn open_bidi(&mut self) -> Result<u64, NetError> {
        if self.open_local_bidi >= self.max_local_bidi {
            return Err(NetError::TableFull);
        }
        let id = self.next_bidi_id;
        self.next_bidi_id += 4;
        let stream = Stream::new(
            id,
            StreamType::Bidirectional,
            true,
            self.initial_max_stream_data_bidi_remote,
            self.initial_max_stream_data_bidi_local,
        );
        self.streams.insert(id, stream);
        self.open_local_bidi += 1;
        Ok(id)
    }

    /// Open a new local unidirectional stream.
    pub fn open_uni(&mut self) -> Result<u64, NetError> {
        if self.open_local_uni >= self.max_local_uni {
            return Err(NetError::TableFull);
        }
        let id = self.next_uni_id;
        self.next_uni_id += 4;
        let stream = Stream::new(
            id,
            StreamType::Unidirectional,
            true,
            self.initial_max_stream_data_uni,
            0,
        );
        self.streams.insert(id, stream);
        self.open_local_uni += 1;
        Ok(id)
    }

    /// Accept a remotely-initiated stream (creates it if new).
    pub fn accept_stream(&mut self, id: u64) -> Result<&mut Stream, NetError> {
        if self.streams.contains_key(&id) {
            return Ok(self.streams.get_mut(&id).unwrap());
        }

        let (initiator, stype) = stream_id_type(id);
        let is_remote = match (self.is_client, initiator) {
            (true, StreamInitiator::Server) | (false, StreamInitiator::Client) => true,
            _ => return Err(NetError::InvalidProtocol),
        };

        if !is_remote {
            return Err(NetError::InvalidProtocol);
        }

        match stype {
            StreamType::Bidirectional => {
                if self.open_remote_bidi >= self.max_remote_bidi {
                    return Err(NetError::TableFull);
                }
                self.open_remote_bidi += 1;
            }
            StreamType::Unidirectional => {
                if self.open_remote_uni >= self.max_remote_uni {
                    return Err(NetError::TableFull);
                }
                self.open_remote_uni += 1;
            }
        }

        let (max_send, max_recv) = match stype {
            StreamType::Bidirectional => (
                self.initial_max_stream_data_bidi_local,
                self.initial_max_stream_data_bidi_remote,
            ),
            StreamType::Unidirectional => (0, self.initial_max_stream_data_uni),
        };

        let stream = Stream::new(id, stype, false, max_send, max_recv);
        self.streams.insert(id, stream);
        Ok(self.streams.get_mut(&id).unwrap())
    }

    /// Get a stream by ID.
    pub fn get(&self, id: u64) -> Option<&Stream> {
        self.streams.get(&id)
    }

    /// Get a mutable stream by ID.
    pub fn get_mut(&mut self, id: u64) -> Option<&mut Stream> {
        self.streams.get_mut(&id)
    }

    /// Update peer's MAX_STREAMS bidi limit.
    pub fn update_max_streams_bidi(&mut self, max: u64) {
        if max > self.max_local_bidi {
            self.max_local_bidi = max;
        }
    }

    /// Update peer's MAX_STREAMS uni limit.
    pub fn update_max_streams_uni(&mut self, max: u64) {
        if max > self.max_local_uni {
            self.max_local_uni = max;
        }
    }

    /// Number of open streams.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// Remove finished streams.
    pub fn garbage_collect(&mut self) {
        let finished: Vec<u64> = self
            .streams
            .iter()
            .filter(|(_, s)| s.is_finished())
            .map(|(&id, _)| id)
            .collect();

        for id in finished {
            if let Some(stream) = self.streams.remove(&id) {
                let (initiator, stype) = stream_id_type(id);
                let is_local = matches!((self.is_client, initiator), (true, StreamInitiator::Client) | (false, StreamInitiator::Server));
                match (is_local, stype) {
                    (true, StreamType::Bidirectional) => {
                        self.open_local_bidi = self.open_local_bidi.saturating_sub(1);
                    }
                    (true, StreamType::Unidirectional) => {
                        self.open_local_uni = self.open_local_uni.saturating_sub(1);
                    }
                    (false, StreamType::Bidirectional) => {
                        self.open_remote_bidi = self.open_remote_bidi.saturating_sub(1);
                    }
                    (false, StreamType::Unidirectional) => {
                        self.open_remote_uni = self.open_remote_uni.saturating_sub(1);
                    }
                }
                drop(stream);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_id_type() {
        assert_eq!(stream_id_type(0), (StreamInitiator::Client, StreamType::Bidirectional));
        assert_eq!(stream_id_type(1), (StreamInitiator::Server, StreamType::Bidirectional));
        assert_eq!(stream_id_type(2), (StreamInitiator::Client, StreamType::Unidirectional));
        assert_eq!(stream_id_type(3), (StreamInitiator::Server, StreamType::Unidirectional));
        assert_eq!(stream_id_type(4), (StreamInitiator::Client, StreamType::Bidirectional));
    }

    #[test]
    fn test_open_bidi_stream() {
        let mut mgr = StreamManager::new(true, 10, 10);
        mgr.set_peer_max_streams(5, 5);
        let id = mgr.open_bidi().unwrap();
        assert_eq!(id, 0); // Client bidi: 0, 4, 8, ...
        let id2 = mgr.open_bidi().unwrap();
        assert_eq!(id2, 4);
    }

    #[test]
    fn test_open_uni_stream() {
        let mut mgr = StreamManager::new(true, 10, 10);
        mgr.set_peer_max_streams(5, 5);
        let id = mgr.open_uni().unwrap();
        assert_eq!(id, 2); // Client uni: 2, 6, 10, ...
        let id2 = mgr.open_uni().unwrap();
        assert_eq!(id2, 6);
    }

    #[test]
    fn test_server_stream_ids() {
        let mut mgr = StreamManager::new(false, 10, 10);
        mgr.set_peer_max_streams(5, 5);
        let bidi = mgr.open_bidi().unwrap();
        assert_eq!(bidi, 1); // Server bidi: 1, 5, 9, ...
        let uni = mgr.open_uni().unwrap();
        assert_eq!(uni, 3); // Server uni: 3, 7, 11, ...
    }

    #[test]
    fn test_stream_limit_bidi() {
        let mut mgr = StreamManager::new(true, 10, 10);
        mgr.set_peer_max_streams(2, 0);
        mgr.open_bidi().unwrap();
        mgr.open_bidi().unwrap();
        assert_eq!(mgr.open_bidi().err(), Some(NetError::TableFull));
    }

    #[test]
    fn test_stream_limit_uni() {
        let mut mgr = StreamManager::new(true, 10, 10);
        mgr.set_peer_max_streams(0, 1);
        mgr.open_uni().unwrap();
        assert_eq!(mgr.open_uni().err(), Some(NetError::TableFull));
    }

    #[test]
    fn test_stream_write_read() {
        let mut mgr = StreamManager::new(true, 10, 10);
        mgr.set_peer_max_streams(5, 5);
        let id = mgr.open_bidi().unwrap();

        let stream = mgr.get_mut(id).unwrap();
        stream.write(b"hello").unwrap();
        assert_eq!(stream.send_state(), Some(SendState::Send));

        // Simulate receiving data
        stream.recv_data(0, b"world", false).unwrap();
        let mut buf = [0u8; 32];
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"world");
    }

    #[test]
    fn test_stream_finish() {
        let mut stream = Stream::new(0, StreamType::Bidirectional, true, 65536, 65536);
        stream.write(b"data").unwrap();
        stream.finish().unwrap();
        assert!(stream.send_fin);

        let (offset, data, fin) = stream.pending_send().unwrap();
        assert_eq!(offset, 0);
        assert_eq!(data, b"data");
        assert!(fin);
    }

    #[test]
    fn test_stream_recv_with_fin() {
        let mut stream = Stream::new(1, StreamType::Bidirectional, false, 65536, 65536);
        stream.recv_data(0, b"abc", true).unwrap();
        assert_eq!(stream.recv_state(), Some(RecvState::SizeKnown));
        assert_eq!(stream.final_size, Some(3));

        let mut buf = [0u8; 10];
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(n, 3);
        assert_eq!(&buf[..3], b"abc");
        assert_eq!(stream.recv_state(), Some(RecvState::DataRead));
    }

    #[test]
    fn test_stream_flow_control_limit() {
        let mut stream = Stream::new(0, StreamType::Bidirectional, true, 10, 10);
        // Receive more than max_recv_data
        assert_eq!(
            stream.recv_data(0, &[0u8; 11], false).err(),
            Some(NetError::BufferTooSmall)
        );
    }

    #[test]
    fn test_stream_reset_send() {
        let mut stream = Stream::new(0, StreamType::Bidirectional, true, 65536, 65536);
        stream.write(b"data").unwrap();
        stream.reset_send(0x01);
        assert_eq!(stream.send_state(), Some(SendState::ResetSent));
        assert!(!stream.can_send());
    }

    #[test]
    fn test_stream_recv_reset() {
        let mut stream = Stream::new(1, StreamType::Bidirectional, false, 65536, 65536);
        stream.recv_reset(100);
        assert_eq!(stream.recv_state(), Some(RecvState::ResetRecvd));
        assert!(!stream.can_recv());
    }

    #[test]
    fn test_stream_is_finished() {
        let mut stream = Stream::new(0, StreamType::Bidirectional, true, 65536, 65536);
        assert!(!stream.is_finished());

        stream.reset_send(0);
        stream.recv_reset(0);
        assert!(stream.is_finished());
    }

    #[test]
    fn test_uni_stream_send_only() {
        let stream = Stream::new(2, StreamType::Unidirectional, true, 65536, 0);
        assert!(stream.can_send());
        assert!(!stream.can_recv());
    }

    #[test]
    fn test_uni_stream_recv_only() {
        let stream = Stream::new(3, StreamType::Unidirectional, false, 0, 65536);
        assert!(!stream.can_send());
        assert!(stream.can_recv());
    }

    #[test]
    fn test_accept_remote_stream() {
        let mut mgr = StreamManager::new(true, 10, 10);
        // Accept server-initiated bidi stream (ID=1)
        let stream = mgr.accept_stream(1).unwrap();
        assert_eq!(stream.id(), 1);
        assert_eq!(stream.stream_type(), StreamType::Bidirectional);
    }

    #[test]
    fn test_accept_own_stream_fails() {
        let mut mgr = StreamManager::new(true, 10, 10);
        // Client trying to accept client-initiated stream
        assert_eq!(mgr.accept_stream(0).err(), Some(NetError::InvalidProtocol));
    }

    #[test]
    fn test_update_max_streams() {
        let mut mgr = StreamManager::new(true, 10, 10);
        mgr.set_peer_max_streams(1, 0);
        mgr.open_bidi().unwrap();
        assert_eq!(mgr.open_bidi().err(), Some(NetError::TableFull));

        mgr.update_max_streams_bidi(2);
        mgr.open_bidi().unwrap(); // Should succeed now
    }

    #[test]
    fn test_update_max_send_data() {
        let mut stream = Stream::new(0, StreamType::Bidirectional, true, 100, 65536);
        assert_eq!(stream.max_send_data, 100);
        stream.update_max_send_data(200);
        assert_eq!(stream.max_send_data, 200);
        // Should not decrease
        stream.update_max_send_data(150);
        assert_eq!(stream.max_send_data, 200);
    }

    #[test]
    fn test_stream_count() {
        let mut mgr = StreamManager::new(true, 10, 10);
        mgr.set_peer_max_streams(5, 5);
        assert_eq!(mgr.stream_count(), 0);
        mgr.open_bidi().unwrap();
        mgr.open_uni().unwrap();
        assert_eq!(mgr.stream_count(), 2);
    }

    #[test]
    fn test_garbage_collect() {
        let mut mgr = StreamManager::new(true, 10, 10);
        mgr.set_peer_max_streams(5, 5);
        let id = mgr.open_bidi().unwrap();

        // Mark both sides as done
        let stream = mgr.get_mut(id).unwrap();
        stream.reset_send(0);
        stream.recv_reset(0);

        mgr.garbage_collect();
        assert_eq!(mgr.stream_count(), 0);
    }

    #[test]
    fn test_out_of_order_recv() {
        let mut stream = Stream::new(1, StreamType::Bidirectional, false, 65536, 65536);
        // Receive second chunk first
        stream.recv_data(5, b"world", false).unwrap();
        // Then first chunk
        stream.recv_data(0, b"hello", false).unwrap();

        let mut buf = [0u8; 32];
        let n = stream.read(&mut buf).unwrap();
        assert_eq!(n, 10);
        assert_eq!(&buf[..10], b"helloworld");
    }

    #[test]
    fn test_pending_send_respects_flow_control() {
        let mut stream = Stream::new(0, StreamType::Bidirectional, true, 5, 65536);
        stream.write(b"helloworld").unwrap();
        let (offset, data, fin) = stream.pending_send().unwrap();
        assert_eq!(offset, 0);
        assert_eq!(data.len(), 5); // Clamped to max_send_data
        assert!(!fin);
    }
}
