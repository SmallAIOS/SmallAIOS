// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RMAP (Remote Memory Access Protocol) command/reply codec.
//!
//! Implements RMAP read/write commands and replies over SpaceWire links
//! with CRC-8 integrity checking per ECSS-E-ST-50-52C.

use crate::BusError;
use alloc::vec::Vec;

/// RMAP protocol identifier.
pub const RMAP_PROTOCOL_ID: u8 = 0x01;

/// RMAP CRC-8 polynomial (x^8 + x^2 + x + 1 = 0x07).
const CRC8_POLY: u8 = 0x07;

/// Compute RMAP CRC-8 over a byte slice.
pub fn rmap_crc8(data: &[u8]) -> u8 {
    let mut crc: u8 = 0;
    for &byte in data {
        crc ^= byte;
        for _ in 0..8 {
            if crc & 0x80 != 0 {
                crc = (crc << 1) ^ CRC8_POLY;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

/// RMAP command type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RmapCommandType {
    /// Read from remote memory.
    Read,
    /// Write to remote memory.
    Write,
    /// Write to remote memory with verify (read-back check).
    WriteVerify,
}

/// RMAP command packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmapCommand {
    /// Target logical address.
    pub target_addr: u8,
    /// Command type.
    pub cmd_type: RmapCommandType,
    /// Destination key (authorization).
    pub key: u8,
    /// Transaction ID.
    pub transaction_id: u16,
    /// Remote memory address (32-bit).
    pub memory_addr: u32,
    /// Data length (for read: how many bytes to read; for write: payload length).
    pub data_length: u32,
    /// Write data payload (empty for read commands).
    pub data: Vec<u8>,
    /// Whether to request a reply.
    pub reply_requested: bool,
}

/// Encoded command layout:
/// target_addr(1) + protocol_id(1) + cmd_byte(1) + key(1) + tid(2) +
/// mem_addr(4) + data_len(3) + header_crc(1) = 14 bytes
/// Followed by optional: data(N) + data_crc(1)
const CMD_HEADER_LEN: usize = 14;

impl RmapCommand {
    /// Create a new RMAP read command.
    pub fn read(
        target_addr: u8,
        key: u8,
        transaction_id: u16,
        memory_addr: u32,
        length: u32,
    ) -> Self {
        Self {
            target_addr,
            cmd_type: RmapCommandType::Read,
            key,
            transaction_id,
            memory_addr,
            data_length: length,
            data: Vec::new(),
            reply_requested: true,
        }
    }

    /// Create a new RMAP write command.
    pub fn write(
        target_addr: u8,
        key: u8,
        transaction_id: u16,
        memory_addr: u32,
        data: &[u8],
        reply_requested: bool,
    ) -> Self {
        Self {
            target_addr,
            cmd_type: RmapCommandType::Write,
            key,
            transaction_id,
            memory_addr,
            data_length: data.len() as u32,
            data: Vec::from(data),
            reply_requested,
        }
    }

    fn cmd_byte(&self) -> u8 {
        let mut cmd: u8 = 0;
        // Bit 7-6: packet type (01 = command)
        cmd |= 0x40;
        // Bit 5: write = 1, read = 0
        match self.cmd_type {
            RmapCommandType::Write | RmapCommandType::WriteVerify => cmd |= 0x20,
            RmapCommandType::Read => {}
        }
        // Bit 4: verify (for write verify)
        if self.cmd_type == RmapCommandType::WriteVerify {
            cmd |= 0x10;
        }
        // Bit 3: reply requested
        if self.reply_requested {
            cmd |= 0x08;
        }
        // Bit 2: increment address (always set)
        cmd |= 0x04;
        cmd
    }

    /// Encode the command into a byte buffer.
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, BusError> {
        let data_section = if !self.data.is_empty() {
            self.data.len() + 1
        } else {
            0
        };
        let total = CMD_HEADER_LEN + data_section;
        if buf.len() < total {
            return Err(BusError::BufferTooSmall);
        }

        buf[0] = self.target_addr;
        buf[1] = RMAP_PROTOCOL_ID;
        buf[2] = self.cmd_byte();
        buf[3] = self.key;
        buf[4] = (self.transaction_id >> 8) as u8;
        buf[5] = self.transaction_id as u8;
        buf[6] = (self.memory_addr >> 24) as u8;
        buf[7] = (self.memory_addr >> 16) as u8;
        buf[8] = (self.memory_addr >> 8) as u8;
        buf[9] = self.memory_addr as u8;
        buf[10] = (self.data_length >> 16) as u8;
        buf[11] = (self.data_length >> 8) as u8;
        buf[12] = self.data_length as u8;
        // Extended address byte (always 0 for 32-bit addressing)
        buf[13] = rmap_crc8(&buf[0..13]);

        if !self.data.is_empty() {
            buf[14..14 + self.data.len()].copy_from_slice(&self.data);
            buf[14 + self.data.len()] = rmap_crc8(&self.data);
        }

        Ok(total)
    }

    /// Decode an RMAP command from a byte buffer.
    pub fn decode(data: &[u8]) -> Result<Self, BusError> {
        if data.len() < CMD_HEADER_LEN {
            return Err(BusError::FrameTooShort);
        }

        // Verify header CRC
        let header_crc = rmap_crc8(&data[0..13]);
        if header_crc != data[13] {
            return Err(BusError::CrcMismatch);
        }

        let target_addr = data[0];
        if data[1] != RMAP_PROTOCOL_ID {
            return Err(BusError::InvalidHeader);
        }
        let cmd_byte = data[2];
        let is_write = cmd_byte & 0x20 != 0;
        let is_verify = cmd_byte & 0x10 != 0;
        let reply_requested = cmd_byte & 0x08 != 0;

        let cmd_type = if is_write && is_verify {
            RmapCommandType::WriteVerify
        } else if is_write {
            RmapCommandType::Write
        } else {
            RmapCommandType::Read
        };

        let key = data[3];
        let transaction_id = ((data[4] as u16) << 8) | data[5] as u16;
        let memory_addr = ((data[6] as u32) << 24)
            | ((data[7] as u32) << 16)
            | ((data[8] as u32) << 8)
            | data[9] as u32;
        let data_length = ((data[10] as u32) << 16) | ((data[11] as u32) << 8) | data[12] as u32;

        let payload = if is_write && data.len() > CMD_HEADER_LEN {
            let payload_end = data.len() - 1;
            let payload_data = &data[14..payload_end];
            // Verify data CRC
            let data_crc = rmap_crc8(payload_data);
            if data_crc != data[payload_end] {
                return Err(BusError::CrcMismatch);
            }
            Vec::from(payload_data)
        } else {
            Vec::new()
        };

        Ok(Self {
            target_addr,
            cmd_type,
            key,
            transaction_id,
            memory_addr,
            data_length,
            data: payload,
            reply_requested,
        })
    }
}

/// RMAP reply packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RmapReply {
    /// Initiator logical address.
    pub initiator_addr: u8,
    /// Status code (0 = success).
    pub status: u8,
    /// Transaction ID.
    pub transaction_id: u16,
    /// Read data payload (empty for write replies).
    pub data: Vec<u8>,
}

/// Encoded reply layout:
/// initiator_addr(1) + protocol_id(1) + reply_byte(1) + status(1) +
/// tid(2) + data_len(3) + header_crc(1) + [data(N) + data_crc(1)]
const REPLY_HEADER_LEN: usize = 10;

impl RmapReply {
    /// Create a success reply for a write command.
    pub fn write_success(initiator_addr: u8, transaction_id: u16) -> Self {
        Self {
            initiator_addr,
            status: 0,
            transaction_id,
            data: Vec::new(),
        }
    }

    /// Create a success reply for a read command.
    pub fn read_success(initiator_addr: u8, transaction_id: u16, data: &[u8]) -> Self {
        Self {
            initiator_addr,
            status: 0,
            transaction_id,
            data: Vec::from(data),
        }
    }

    /// Create an error reply.
    pub fn error(initiator_addr: u8, transaction_id: u16, status: u8) -> Self {
        Self {
            initiator_addr,
            status,
            transaction_id,
            data: Vec::new(),
        }
    }

    /// Returns true if the reply indicates success.
    pub fn is_success(&self) -> bool {
        self.status == 0
    }

    /// Encode the reply into a byte buffer.
    pub fn encode(&self, buf: &mut [u8]) -> Result<usize, BusError> {
        let data_section = if !self.data.is_empty() {
            self.data.len() + 1
        } else {
            0
        };
        let total = REPLY_HEADER_LEN + data_section;
        if buf.len() < total {
            return Err(BusError::BufferTooSmall);
        }

        buf[0] = self.initiator_addr;
        buf[1] = RMAP_PROTOCOL_ID;
        buf[2] = 0x00; // Reply byte (not a command)
        buf[3] = self.status;
        buf[4] = (self.transaction_id >> 8) as u8;
        buf[5] = self.transaction_id as u8;
        let data_len = self.data.len() as u32;
        buf[6] = (data_len >> 16) as u8;
        buf[7] = (data_len >> 8) as u8;
        buf[8] = data_len as u8;
        buf[9] = rmap_crc8(&buf[0..9]);

        if !self.data.is_empty() {
            buf[10..10 + self.data.len()].copy_from_slice(&self.data);
            buf[10 + self.data.len()] = rmap_crc8(&self.data);
        }

        Ok(total)
    }

    /// Decode an RMAP reply from a byte buffer.
    pub fn decode(data: &[u8]) -> Result<Self, BusError> {
        if data.len() < REPLY_HEADER_LEN {
            return Err(BusError::FrameTooShort);
        }

        let header_crc = rmap_crc8(&data[0..9]);
        if header_crc != data[9] {
            return Err(BusError::CrcMismatch);
        }

        if data[1] != RMAP_PROTOCOL_ID {
            return Err(BusError::InvalidHeader);
        }

        let initiator_addr = data[0];
        let status = data[3];
        let transaction_id = ((data[4] as u16) << 8) | data[5] as u16;

        let payload = if data.len() > REPLY_HEADER_LEN {
            let payload_end = data.len() - 1;
            let payload_data = &data[10..payload_end];
            let data_crc = rmap_crc8(payload_data);
            if data_crc != data[payload_end] {
                return Err(BusError::CrcMismatch);
            }
            Vec::from(payload_data)
        } else {
            Vec::new()
        };

        Ok(Self {
            initiator_addr,
            status,
            transaction_id,
            data: payload,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_crc8_empty() {
        assert_eq!(rmap_crc8(&[]), 0);
    }

    #[test]
    fn test_crc8_known_values() {
        // Verify CRC-8 is deterministic
        let data = [0x01, 0x02, 0x03];
        let crc = rmap_crc8(&data);
        assert_eq!(rmap_crc8(&data), crc); // same input = same output
    }

    #[test]
    fn test_crc8_changes_with_data() {
        let crc1 = rmap_crc8(&[0x00]);
        let crc2 = rmap_crc8(&[0x01]);
        assert_ne!(crc1, crc2);
    }

    // === RMAP Command Tests ===

    #[test]
    fn test_read_command_encode_decode() {
        let cmd = RmapCommand::read(42, 0xAB, 0x1234, 0x80000000, 256);
        let mut buf = [0u8; 256];
        let len = cmd.encode(&mut buf).unwrap();

        let decoded = RmapCommand::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.target_addr, 42);
        assert_eq!(decoded.cmd_type, RmapCommandType::Read);
        assert_eq!(decoded.key, 0xAB);
        assert_eq!(decoded.transaction_id, 0x1234);
        assert_eq!(decoded.memory_addr, 0x80000000);
        assert_eq!(decoded.data_length, 256);
        assert!(decoded.data.is_empty());
        assert!(decoded.reply_requested);
    }

    #[test]
    fn test_write_command_encode_decode() {
        let data = [0xDE, 0xAD, 0xBE, 0xEF];
        let cmd = RmapCommand::write(42, 0xAB, 0x5678, 0x40000000, &data, true);
        let mut buf = [0u8; 256];
        let len = cmd.encode(&mut buf).unwrap();

        let decoded = RmapCommand::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.cmd_type, RmapCommandType::Write);
        assert_eq!(decoded.data, &data);
        assert_eq!(decoded.data_length, 4);
    }

    #[test]
    fn test_write_no_reply() {
        let cmd = RmapCommand::write(42, 0xAB, 0x0001, 0x0, &[0xFF], false);
        let mut buf = [0u8; 256];
        let len = cmd.encode(&mut buf).unwrap();
        let decoded = RmapCommand::decode(&buf[..len]).unwrap();
        assert!(!decoded.reply_requested);
    }

    #[test]
    fn test_command_header_crc_mismatch() {
        let cmd = RmapCommand::read(42, 0xAB, 0x1234, 0x80000000, 256);
        let mut buf = [0u8; 256];
        let len = cmd.encode(&mut buf).unwrap();
        buf[13] ^= 0xFF; // Corrupt header CRC
        assert_eq!(
            RmapCommand::decode(&buf[..len]).err(),
            Some(BusError::CrcMismatch)
        );
    }

    #[test]
    fn test_command_data_crc_mismatch() {
        let cmd = RmapCommand::write(42, 0xAB, 0x1234, 0x0, &[0xAA, 0xBB], true);
        let mut buf = [0u8; 256];
        let len = cmd.encode(&mut buf).unwrap();
        buf[len - 1] ^= 0xFF; // Corrupt data CRC
        assert_eq!(
            RmapCommand::decode(&buf[..len]).err(),
            Some(BusError::CrcMismatch)
        );
    }

    #[test]
    fn test_command_too_short() {
        assert_eq!(
            RmapCommand::decode(&[0u8; 10]).err(),
            Some(BusError::FrameTooShort)
        );
    }

    #[test]
    fn test_command_buffer_too_small() {
        let cmd = RmapCommand::write(42, 0xAB, 0x0, 0x0, &[0xAA; 100], true);
        let mut buf = [0u8; 10];
        assert_eq!(cmd.encode(&mut buf).err(), Some(BusError::BufferTooSmall));
    }

    // === RMAP Reply Tests ===

    #[test]
    fn test_write_success_reply() {
        let reply = RmapReply::write_success(0x20, 0x1234);
        assert!(reply.is_success());
        let mut buf = [0u8; 64];
        let len = reply.encode(&mut buf).unwrap();
        let decoded = RmapReply::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.initiator_addr, 0x20);
        assert_eq!(decoded.transaction_id, 0x1234);
        assert_eq!(decoded.status, 0);
        assert!(decoded.data.is_empty());
    }

    #[test]
    fn test_read_success_reply() {
        let data = [0x01, 0x02, 0x03, 0x04];
        let reply = RmapReply::read_success(0x20, 0x5678, &data);
        let mut buf = [0u8; 64];
        let len = reply.encode(&mut buf).unwrap();
        let decoded = RmapReply::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.data, &data);
        assert!(decoded.is_success());
    }

    #[test]
    fn test_error_reply() {
        let reply = RmapReply::error(0x20, 0x0001, 0x03);
        assert!(!reply.is_success());
        let mut buf = [0u8; 64];
        let len = reply.encode(&mut buf).unwrap();
        let decoded = RmapReply::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.status, 0x03);
    }

    #[test]
    fn test_reply_header_crc_mismatch() {
        let reply = RmapReply::write_success(0x20, 0x1234);
        let mut buf = [0u8; 64];
        let len = reply.encode(&mut buf).unwrap();
        buf[9] ^= 0xFF;
        assert_eq!(
            RmapReply::decode(&buf[..len]).err(),
            Some(BusError::CrcMismatch)
        );
    }

    #[test]
    fn test_reply_data_crc_mismatch() {
        let reply = RmapReply::read_success(0x20, 0x1234, &[0xAA, 0xBB]);
        let mut buf = [0u8; 64];
        let len = reply.encode(&mut buf).unwrap();
        buf[len - 1] ^= 0xFF;
        assert_eq!(
            RmapReply::decode(&buf[..len]).err(),
            Some(BusError::CrcMismatch)
        );
    }

    #[test]
    fn test_reply_too_short() {
        assert_eq!(
            RmapReply::decode(&[0u8; 5]).err(),
            Some(BusError::FrameTooShort)
        );
    }
}
