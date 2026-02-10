// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! TCP state machine for SmallAIOS.
//!
//! Provides [`TcpState`] for tracking connection lifecycle, [`TcpHeader`] for
//! parsing and serializing TCP segment headers, and [`TcpConnection`] for
//! driving the connection state machine.

use alloc::vec::Vec;

use crate::NetError;

// ---------------------------------------------------------------------------
// TcpState
// ---------------------------------------------------------------------------

/// TCP connection states per RFC 793.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcpState {
    /// No connection exists.
    Closed,
    /// Waiting for an incoming SYN.
    Listen,
    /// SYN sent, awaiting SYN-ACK.
    SynSent,
    /// SYN received, awaiting ACK of our SYN.
    SynReceived,
    /// Connection is open and data can flow.
    Established,
    /// First FIN sent, awaiting ACK.
    FinWait1,
    /// First FIN acknowledged, awaiting peer FIN.
    FinWait2,
    /// Peer sent FIN, awaiting local close.
    CloseWait,
    /// Both sides sent FIN, awaiting final ACK.
    Closing,
    /// Peer FIN acknowledged, awaiting ACK of our FIN.
    LastAck,
    /// Waiting for enough time to ensure remote received ACK of FIN.
    TimeWait,
}

impl TcpState {
    /// Returns `true` if the connection is fully established and data can flow
    /// in at least one direction.
    pub fn is_connected(&self) -> bool {
        matches!(self, TcpState::Established | TcpState::CloseWait)
    }

    /// Returns `true` if the local side is permitted to send data.
    pub fn can_send(&self) -> bool {
        matches!(self, TcpState::Established | TcpState::CloseWait)
    }

    /// Returns `true` if the local side is permitted to receive data.
    pub fn can_receive(&self) -> bool {
        matches!(
            self,
            TcpState::Established | TcpState::FinWait1 | TcpState::FinWait2
        )
    }
}

// ---------------------------------------------------------------------------
// TcpFlags
// ---------------------------------------------------------------------------

/// TCP header flag constants.
pub struct TcpFlags;

impl TcpFlags {
    /// FIN — no more data from sender.
    pub const FIN: u8 = 0x01;
    /// SYN — synchronize sequence numbers.
    pub const SYN: u8 = 0x02;
    /// RST — reset the connection.
    pub const RST: u8 = 0x04;
    /// PSH — push buffered data to the application.
    pub const PSH: u8 = 0x08;
    /// ACK — acknowledgment field is significant.
    pub const ACK: u8 = 0x10;
    /// URG — urgent pointer field is significant.
    pub const URG: u8 = 0x20;
}

// ---------------------------------------------------------------------------
// TcpHeader
// ---------------------------------------------------------------------------

/// Parsed TCP segment header (fixed 20-byte portion).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TcpHeader {
    /// Source port number.
    pub src_port: u16,
    /// Destination port number.
    pub dst_port: u16,
    /// Sequence number.
    pub seq_num: u32,
    /// Acknowledgment number.
    pub ack_num: u32,
    /// Data offset in 32-bit words (minimum 5).
    pub data_offset: u8,
    /// TCP flags (combination of [`TcpFlags`] constants).
    pub flags: u8,
    /// Receive window size.
    pub window: u16,
    /// Header checksum.
    pub checksum: u16,
    /// Urgent pointer.
    pub urgent_ptr: u16,
}

impl TcpHeader {
    /// Minimum TCP header length in bytes (no options).
    pub const HEADER_MIN_LEN: usize = 20;

    /// Parse a TCP header from raw bytes.
    ///
    /// Returns the parsed header and a slice of the payload (everything past
    /// the header including any options). Returns [`NetError::PacketTooShort`]
    /// if `data` is shorter than 20 bytes or shorter than the data offset
    /// indicates. Returns [`NetError::InvalidHeader`] if the data offset is
    /// less than 5.
    pub fn parse(data: &[u8]) -> Result<(TcpHeader, &[u8]), NetError> {
        if data.len() < Self::HEADER_MIN_LEN {
            return Err(NetError::PacketTooShort);
        }

        let src_port = u16::from_be_bytes([data[0], data[1]]);
        let dst_port = u16::from_be_bytes([data[2], data[3]]);
        let seq_num = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        let ack_num = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);
        let data_offset = (data[12] >> 4) & 0x0F;
        let flags = data[13];
        let window = u16::from_be_bytes([data[14], data[15]]);
        let checksum = u16::from_be_bytes([data[16], data[17]]);
        let urgent_ptr = u16::from_be_bytes([data[18], data[19]]);

        // Data offset is in 32-bit words; minimum is 5 (20 bytes).
        if data_offset < 5 {
            return Err(NetError::InvalidHeader);
        }

        let header_len = (data_offset as usize) * 4;
        if data.len() < header_len {
            return Err(NetError::PacketTooShort);
        }

        let header = TcpHeader {
            src_port,
            dst_port,
            seq_num,
            ack_num,
            data_offset,
            flags,
            window,
            checksum,
            urgent_ptr,
        };

        Ok((header, &data[header_len..]))
    }

    /// Serialize the 20-byte fixed TCP header to a byte array.
    ///
    /// Options are not included; the caller must append them separately if
    /// required.
    pub fn serialize(&self) -> [u8; 20] {
        let mut buf = [0u8; 20];
        buf[0..2].copy_from_slice(&self.src_port.to_be_bytes());
        buf[2..4].copy_from_slice(&self.dst_port.to_be_bytes());
        buf[4..8].copy_from_slice(&self.seq_num.to_be_bytes());
        buf[8..12].copy_from_slice(&self.ack_num.to_be_bytes());
        buf[12] = (self.data_offset << 4) & 0xF0;
        buf[13] = self.flags;
        buf[14..16].copy_from_slice(&self.window.to_be_bytes());
        buf[16..18].copy_from_slice(&self.checksum.to_be_bytes());
        buf[18..20].copy_from_slice(&self.urgent_ptr.to_be_bytes());
        buf
    }
}

// ---------------------------------------------------------------------------
// TcpAction
// ---------------------------------------------------------------------------

/// Action to take after processing an incoming TCP segment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TcpAction {
    /// Send the contained raw segment bytes to the remote peer.
    Send(Vec<u8>),
    /// Close the connection (no more data to send).
    Close,
    /// No action required.
    Nothing,
}

// ---------------------------------------------------------------------------
// TcpConnection
// ---------------------------------------------------------------------------

/// A single TCP connection tracked by the state machine.
pub struct TcpConnection {
    /// Current connection state.
    state: TcpState,
    /// Local port number.
    pub local_port: u16,
    /// Remote port number.
    pub remote_port: u16,
    /// Next sequence number to send (SND.NXT).
    pub send_next: u32,
    /// Oldest unacknowledged sequence number (SND.UNA).
    pub send_una: u32,
    /// Next expected receive sequence number (RCV.NXT).
    pub recv_next: u32,
    /// Send window advertised by the remote peer.
    pub send_window: u16,
    /// Receive window advertised to the remote peer.
    pub recv_window: u16,
}

impl TcpConnection {
    /// Default receive window size (64 KiB minus one).
    const DEFAULT_RECV_WINDOW: u16 = 65535;

    /// Create a new TCP connection bound to `local_port` in the [`TcpState::Closed`]
    /// state.
    pub fn new(local_port: u16) -> Self {
        TcpConnection {
            state: TcpState::Closed,
            local_port,
            remote_port: 0,
            send_next: 0,
            send_una: 0,
            recv_next: 0,
            send_window: 0,
            recv_window: Self::DEFAULT_RECV_WINDOW,
        }
    }

    /// Return the current connection state.
    pub fn state(&self) -> TcpState {
        self.state
    }

    /// Transition the connection to the [`TcpState::Listen`] state.
    ///
    /// Only valid from [`TcpState::Closed`].
    pub fn listen(&mut self) -> Result<(), NetError> {
        if self.state != TcpState::Closed {
            return Err(NetError::InvalidProtocol);
        }
        self.state = TcpState::Listen;
        Ok(())
    }

    /// Initiate an active open (client connect).
    ///
    /// Transitions from [`TcpState::Closed`] to [`TcpState::SynSent`] and
    /// returns a SYN segment to send.
    pub fn connect(&mut self, remote_port: u16) -> Result<TcpAction, NetError> {
        if self.state != TcpState::Closed {
            return Err(NetError::InvalidProtocol);
        }
        self.remote_port = remote_port;
        self.send_next = 1000; // simplified ISN
        self.send_una = self.send_next;

        let syn_header = TcpHeader {
            src_port: self.local_port,
            dst_port: self.remote_port,
            seq_num: self.send_next,
            ack_num: 0,
            data_offset: 5,
            flags: TcpFlags::SYN,
            window: self.recv_window,
            checksum: 0,
            urgent_ptr: 0,
        };

        self.send_next = self.send_next.wrapping_add(1);
        self.state = TcpState::SynSent;
        Ok(TcpAction::Send(Vec::from(syn_header.serialize())))
    }

    /// Process an incoming TCP segment and advance the state machine.
    ///
    /// Returns the action to take in response, or [`NetError::NotImplemented`]
    /// for state transitions that are not yet wired up.
    pub fn on_segment(
        &mut self,
        header: &TcpHeader,
        _payload: &[u8],
    ) -> Result<Option<TcpAction>, NetError> {
        match self.state {
            // -----------------------------------------------------------
            // Listen + SYN → SynReceived (send SYN-ACK)
            // -----------------------------------------------------------
            TcpState::Listen => {
                if header.flags & TcpFlags::SYN != 0 {
                    self.remote_port = header.src_port;
                    self.recv_next = header.seq_num.wrapping_add(1);
                    self.send_window = header.window;
                    self.send_next = 2000; // simplified ISN
                    self.send_una = self.send_next;

                    let syn_ack = TcpHeader {
                        src_port: self.local_port,
                        dst_port: self.remote_port,
                        seq_num: self.send_next,
                        ack_num: self.recv_next,
                        data_offset: 5,
                        flags: TcpFlags::SYN | TcpFlags::ACK,
                        window: self.recv_window,
                        checksum: 0,
                        urgent_ptr: 0,
                    };

                    self.send_next = self.send_next.wrapping_add(1);
                    self.state = TcpState::SynReceived;
                    Ok(Some(TcpAction::Send(Vec::from(syn_ack.serialize()))))
                } else {
                    Ok(Some(TcpAction::Nothing))
                }
            }

            // -----------------------------------------------------------
            // SynSent + SYN-ACK → Established (send ACK)
            // -----------------------------------------------------------
            TcpState::SynSent => {
                if header.flags & (TcpFlags::SYN | TcpFlags::ACK) == (TcpFlags::SYN | TcpFlags::ACK)
                {
                    self.recv_next = header.seq_num.wrapping_add(1);
                    self.send_una = header.ack_num;
                    self.send_window = header.window;

                    let ack = TcpHeader {
                        src_port: self.local_port,
                        dst_port: self.remote_port,
                        seq_num: self.send_next,
                        ack_num: self.recv_next,
                        data_offset: 5,
                        flags: TcpFlags::ACK,
                        window: self.recv_window,
                        checksum: 0,
                        urgent_ptr: 0,
                    };

                    self.state = TcpState::Established;
                    Ok(Some(TcpAction::Send(Vec::from(ack.serialize()))))
                } else {
                    Ok(Some(TcpAction::Nothing))
                }
            }

            // -----------------------------------------------------------
            // SynReceived + ACK → Established
            // -----------------------------------------------------------
            TcpState::SynReceived => {
                if header.flags & TcpFlags::ACK != 0 {
                    self.send_una = header.ack_num;
                    self.state = TcpState::Established;
                    Ok(Some(TcpAction::Nothing))
                } else {
                    Ok(Some(TcpAction::Nothing))
                }
            }

            // All other state transitions are not yet implemented.
            _ => Err(NetError::NotImplemented),
        }
    }
}

// ---------------------------------------------------------------------------
// Pseudo-header checksum helper
// ---------------------------------------------------------------------------

/// Compute the partial checksum for the TCP pseudo-header.
///
/// The pseudo-header (per RFC 793) consists of source IP, destination IP, a
/// zero byte, the protocol number (6 for TCP), and the TCP segment length.
/// This function sums those fields as 16-bit words and returns the partial
/// 32-bit accumulator so the caller can continue folding in the TCP
/// header and payload.
pub fn tcp_checksum_pseudo_header(src_ip: [u8; 4], dst_ip: [u8; 4], tcp_len: u16) -> u32 {
    let mut sum: u32 = 0;

    // Source IP (two 16-bit words)
    sum += u16::from_be_bytes([src_ip[0], src_ip[1]]) as u32;
    sum += u16::from_be_bytes([src_ip[2], src_ip[3]]) as u32;

    // Destination IP (two 16-bit words)
    sum += u16::from_be_bytes([dst_ip[0], dst_ip[1]]) as u32;
    sum += u16::from_be_bytes([dst_ip[2], dst_ip[3]]) as u32;

    // Zero + Protocol (TCP = 6)
    sum += 6u32;

    // TCP segment length
    sum += tcp_len as u32;

    sum
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- TcpState methods ---------------------------------------------------

    #[test]
    fn test_state_is_connected() {
        assert!(TcpState::Established.is_connected());
        assert!(TcpState::CloseWait.is_connected());
        assert!(!TcpState::Closed.is_connected());
        assert!(!TcpState::Listen.is_connected());
        assert!(!TcpState::SynSent.is_connected());
        assert!(!TcpState::TimeWait.is_connected());
    }

    #[test]
    fn test_state_can_send() {
        assert!(TcpState::Established.can_send());
        assert!(TcpState::CloseWait.can_send());
        assert!(!TcpState::Closed.can_send());
        assert!(!TcpState::FinWait1.can_send());
        assert!(!TcpState::FinWait2.can_send());
    }

    #[test]
    fn test_state_can_receive() {
        assert!(TcpState::Established.can_receive());
        assert!(TcpState::FinWait1.can_receive());
        assert!(TcpState::FinWait2.can_receive());
        assert!(!TcpState::Closed.can_receive());
        assert!(!TcpState::CloseWait.can_receive());
        assert!(!TcpState::LastAck.can_receive());
    }

    // -- TcpFlags constants -------------------------------------------------

    #[test]
    fn test_flag_values() {
        assert_eq!(TcpFlags::FIN, 0x01);
        assert_eq!(TcpFlags::SYN, 0x02);
        assert_eq!(TcpFlags::RST, 0x04);
        assert_eq!(TcpFlags::PSH, 0x08);
        assert_eq!(TcpFlags::ACK, 0x10);
        assert_eq!(TcpFlags::URG, 0x20);
    }

    #[test]
    fn test_flag_combinations() {
        let syn_ack = TcpFlags::SYN | TcpFlags::ACK;
        assert_eq!(syn_ack, 0x12);

        let fin_ack = TcpFlags::FIN | TcpFlags::ACK;
        assert_eq!(fin_ack, 0x11);
    }

    // -- TcpHeader parse / serialize ----------------------------------------

    #[test]
    fn test_parse_valid_header() {
        let mut data = [0u8; 24];
        // src_port = 8080
        data[0..2].copy_from_slice(&8080u16.to_be_bytes());
        // dst_port = 80
        data[2..4].copy_from_slice(&80u16.to_be_bytes());
        // seq_num = 1000
        data[4..8].copy_from_slice(&1000u32.to_be_bytes());
        // ack_num = 2000
        data[8..12].copy_from_slice(&2000u32.to_be_bytes());
        // data_offset = 5 (20 bytes), flags = SYN
        data[12] = 5 << 4;
        data[13] = TcpFlags::SYN;
        // window = 65535
        data[14..16].copy_from_slice(&65535u16.to_be_bytes());
        // checksum = 0xABCD
        data[16..18].copy_from_slice(&0xABCDu16.to_be_bytes());
        // urgent = 0
        data[18..20].copy_from_slice(&0u16.to_be_bytes());
        // 4 bytes payload
        data[20..24].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);

        let (hdr, payload) = TcpHeader::parse(&data).unwrap();
        assert_eq!(hdr.src_port, 8080);
        assert_eq!(hdr.dst_port, 80);
        assert_eq!(hdr.seq_num, 1000);
        assert_eq!(hdr.ack_num, 2000);
        assert_eq!(hdr.data_offset, 5);
        assert_eq!(hdr.flags, TcpFlags::SYN);
        assert_eq!(hdr.window, 65535);
        assert_eq!(hdr.checksum, 0xABCD);
        assert_eq!(hdr.urgent_ptr, 0);
        assert_eq!(payload, &[0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn test_parse_too_short() {
        let data = [0u8; 19];
        assert_eq!(TcpHeader::parse(&data), Err(NetError::PacketTooShort));
    }

    #[test]
    fn test_parse_invalid_data_offset() {
        let mut data = [0u8; 20];
        // data_offset = 3 (invalid, minimum is 5)
        data[12] = 3 << 4;
        assert_eq!(TcpHeader::parse(&data), Err(NetError::InvalidHeader));
    }

    #[test]
    fn test_serialize_roundtrip() {
        let original = TcpHeader {
            src_port: 12345,
            dst_port: 443,
            seq_num: 0xDEADBEEF,
            ack_num: 0xCAFEBABE,
            data_offset: 5,
            flags: TcpFlags::SYN | TcpFlags::ACK,
            window: 32768,
            checksum: 0x1234,
            urgent_ptr: 0,
        };

        let bytes = original.serialize();
        assert_eq!(bytes.len(), 20);

        let (parsed, payload) = TcpHeader::parse(&bytes).unwrap();
        assert_eq!(parsed, original);
        assert!(payload.is_empty());
    }

    #[test]
    fn test_header_min_len_constant() {
        assert_eq!(TcpHeader::HEADER_MIN_LEN, 20);
    }

    // -- TcpConnection creation and basic states ----------------------------

    #[test]
    fn test_connection_new() {
        let conn = TcpConnection::new(8080);
        assert_eq!(conn.state(), TcpState::Closed);
        assert_eq!(conn.local_port, 8080);
        assert_eq!(conn.remote_port, 0);
        assert_eq!(conn.recv_window, 65535);
    }

    #[test]
    fn test_connection_listen() {
        let mut conn = TcpConnection::new(80);
        assert!(conn.listen().is_ok());
        assert_eq!(conn.state(), TcpState::Listen);
    }

    #[test]
    fn test_connection_listen_from_non_closed() {
        let mut conn = TcpConnection::new(80);
        conn.listen().unwrap();
        // Cannot listen again from Listen state.
        assert_eq!(conn.listen(), Err(NetError::InvalidProtocol));
    }

    #[test]
    fn test_connection_connect() {
        let mut conn = TcpConnection::new(5000);
        let action = conn.connect(80).unwrap();
        assert_eq!(conn.state(), TcpState::SynSent);
        assert_eq!(conn.remote_port, 80);
        // Should produce a Send action with a SYN segment.
        match action {
            TcpAction::Send(data) => {
                assert_eq!(data.len(), 20);
                let (hdr, _) = TcpHeader::parse(&data).unwrap();
                assert_eq!(hdr.flags, TcpFlags::SYN);
                assert_eq!(hdr.src_port, 5000);
                assert_eq!(hdr.dst_port, 80);
            }
            _ => panic!("expected TcpAction::Send"),
        }
    }

    // -- State transitions via on_segment -----------------------------------

    #[test]
    fn test_listen_to_syn_received() {
        let mut conn = TcpConnection::new(80);
        conn.listen().unwrap();

        let syn = TcpHeader {
            src_port: 5000,
            dst_port: 80,
            seq_num: 100,
            ack_num: 0,
            data_offset: 5,
            flags: TcpFlags::SYN,
            window: 65535,
            checksum: 0,
            urgent_ptr: 0,
        };

        let result = conn.on_segment(&syn, &[]).unwrap();
        assert_eq!(conn.state(), TcpState::SynReceived);
        assert_eq!(conn.remote_port, 5000);
        assert_eq!(conn.recv_next, 101);
        // Should return a SYN-ACK segment.
        match result {
            Some(TcpAction::Send(data)) => {
                let (hdr, _) = TcpHeader::parse(&data).unwrap();
                assert_eq!(hdr.flags, TcpFlags::SYN | TcpFlags::ACK);
                assert_eq!(hdr.ack_num, 101);
            }
            _ => panic!("expected Some(TcpAction::Send)"),
        }
    }

    #[test]
    fn test_syn_received_to_established() {
        let mut conn = TcpConnection::new(80);
        conn.listen().unwrap();

        // Receive SYN first.
        let syn = TcpHeader {
            src_port: 5000,
            dst_port: 80,
            seq_num: 100,
            ack_num: 0,
            data_offset: 5,
            flags: TcpFlags::SYN,
            window: 65535,
            checksum: 0,
            urgent_ptr: 0,
        };
        conn.on_segment(&syn, &[]).unwrap();
        assert_eq!(conn.state(), TcpState::SynReceived);

        // Receive ACK to complete handshake.
        let ack = TcpHeader {
            src_port: 5000,
            dst_port: 80,
            seq_num: 101,
            ack_num: 2001,
            data_offset: 5,
            flags: TcpFlags::ACK,
            window: 65535,
            checksum: 0,
            urgent_ptr: 0,
        };
        let result = conn.on_segment(&ack, &[]).unwrap();
        assert_eq!(conn.state(), TcpState::Established);
        assert!(matches!(result, Some(TcpAction::Nothing)));
    }

    #[test]
    fn test_established_returns_not_implemented() {
        let mut conn = TcpConnection::new(80);
        conn.listen().unwrap();

        // Drive through SYN → SYN-RCV → Established.
        let syn = TcpHeader {
            src_port: 5000,
            dst_port: 80,
            seq_num: 100,
            ack_num: 0,
            data_offset: 5,
            flags: TcpFlags::SYN,
            window: 65535,
            checksum: 0,
            urgent_ptr: 0,
        };
        conn.on_segment(&syn, &[]).unwrap();

        let ack = TcpHeader {
            src_port: 5000,
            dst_port: 80,
            seq_num: 101,
            ack_num: 2001,
            data_offset: 5,
            flags: TcpFlags::ACK,
            window: 65535,
            checksum: 0,
            urgent_ptr: 0,
        };
        conn.on_segment(&ack, &[]).unwrap();
        assert_eq!(conn.state(), TcpState::Established);

        // Data segment in Established — not yet implemented.
        let data_seg = TcpHeader {
            src_port: 5000,
            dst_port: 80,
            seq_num: 101,
            ack_num: 2001,
            data_offset: 5,
            flags: TcpFlags::ACK | TcpFlags::PSH,
            window: 65535,
            checksum: 0,
            urgent_ptr: 0,
        };
        assert_eq!(
            conn.on_segment(&data_seg, b"hello"),
            Err(NetError::NotImplemented)
        );
    }

    // -- Pseudo-header checksum ---------------------------------------------

    #[test]
    fn test_pseudo_header_checksum() {
        let src = [192, 168, 1, 1];
        let dst = [10, 0, 0, 1];
        let tcp_len: u16 = 40;
        let sum = tcp_checksum_pseudo_header(src, dst, tcp_len);

        // Manual calculation:
        // 192.168 = 0xC0A8, 1.1 = 0x0101 → 0xC0A8 + 0x0101
        // 10.0 = 0x0A00, 0.1 = 0x0001 → 0x0A00 + 0x0001
        // protocol = 6, tcp_len = 40
        let expected: u32 = 0xC0A8 + 0x0101 + 0x0A00 + 0x0001 + 6 + 40;
        assert_eq!(sum, expected);
    }
}
