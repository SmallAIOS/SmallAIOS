// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! MIL-STD-1553B word codecs.
//!
//! All 1553 words are 20 bits: 3-bit sync + 16-bit content + 1-bit parity.
//! We encode into a `u32` with bits [19:0] representing the 20-bit word.
//! Sync type distinguishes command/status words from data words.

use crate::BusError;

/// Maximum RT address (0-30; 31 is broadcast).
pub const RT_BROADCAST: u8 = 31;

/// Maximum word count field value.
pub const MAX_WORD_COUNT: u8 = 31;

/// Mode code subaddresses: 0b00000 and 0b11111.
pub const MODE_CODE_SA_LOW: u8 = 0;
pub const MODE_CODE_SA_HIGH: u8 = 31;

/// Sync pattern type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncType {
    /// Command sync (used for command words).
    Command,
    /// Data sync (used for status and data words).
    Data,
}

/// 3-bit sync pattern that identifies a command word.
const SYNC_COMMAND: u32 = 0b111;

/// 3-bit sync pattern that identifies a status or data word.
const SYNC_DATA: u32 = 0b110;

/// Assemble a 20-bit MIL-STD-1553B word: `sync(3) | bits(16) | parity(1)`.
///
/// MIL-STD-1553B uses odd parity, so the parity bit is set to make the count
/// of `1` bits across the 16-bit content plus the parity bit itself odd.
#[inline]
fn assemble_word(sync: u32, bits: u16) -> u32 {
    let parity: u32 = if bits.count_ones().is_multiple_of(2) {
        1
    } else {
        0
    };
    (sync << 17) | ((bits as u32) << 1) | parity
}

/// Validate sync + odd parity and extract the 16-bit content of a 1553 word.
///
/// Returns [`BusError::InvalidHeader`] if the sync field does not match
/// `expected_sync`, or [`BusError::ParityError`] if the odd parity check fails.
#[inline]
fn parse_word(word: u32, expected_sync: u32) -> Result<u16, BusError> {
    let sync = (word >> 17) & 0x7;
    if sync != expected_sync {
        return Err(BusError::InvalidHeader);
    }
    let bits = ((word >> 1) & 0xFFFF) as u16;
    let parity_bit = (word & 1) as u16;
    // Odd parity: count of ones in content + parity bit must be odd.
    let ones = bits.count_ones() + parity_bit as u32;
    if ones.is_multiple_of(2) {
        return Err(BusError::ParityError);
    }
    Ok(bits)
}

/// MIL-STD-1553B Command Word.
///
/// Layout (MSB first within the 16-bit content):
/// - RT Address [4:0] (5 bits)
/// - T/R bit (1 bit): 1 = transmit (RT->BC), 0 = receive (BC->RT)
/// - Subaddress [4:0] (5 bits)
/// - Word count / Mode code [4:0] (5 bits)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CommandWord {
    /// Remote Terminal address (0-30, 31 = broadcast).
    pub rt_address: u8,
    /// Transmit/Receive bit: true = transmit (RT sends data).
    pub transmit: bool,
    /// Subaddress (0-31). SA 0 or 31 indicates mode code.
    pub subaddress: u8,
    /// Word count (1-32, encoded as 0-31 where 0 means 32) or mode code.
    pub word_count: u8,
}

impl CommandWord {
    /// Create a new command word.
    pub fn new(
        rt_address: u8,
        transmit: bool,
        subaddress: u8,
        word_count: u8,
    ) -> Result<Self, BusError> {
        if rt_address > RT_BROADCAST {
            return Err(BusError::InvalidId);
        }
        if subaddress > 31 {
            return Err(BusError::OutOfRange);
        }
        if word_count > 31 {
            return Err(BusError::OutOfRange);
        }
        Ok(Self {
            rt_address,
            transmit,
            subaddress,
            word_count,
        })
    }

    /// Returns true if this is a mode code command (SA = 0 or 31).
    pub fn is_mode_code(&self) -> bool {
        self.subaddress == MODE_CODE_SA_LOW || self.subaddress == MODE_CODE_SA_HIGH
    }

    /// Returns the actual data word count (1-32).
    /// Word count 0 encodes 32.
    pub fn data_word_count(&self) -> u8 {
        if self.is_mode_code() {
            0 // Mode codes don't have data words (or have 1 for some mode codes)
        } else if self.word_count == 0 {
            32
        } else {
            self.word_count
        }
    }

    /// Encode to a 20-bit value (3-bit sync + 16-bit content + 1-bit parity).
    pub fn encode(&self) -> u32 {
        let mut bits: u16 = 0;
        bits |= (self.rt_address as u16 & 0x1F) << 11;
        bits |= if self.transmit { 1 << 10 } else { 0 };
        bits |= (self.subaddress as u16 & 0x1F) << 5;
        bits |= self.word_count as u16 & 0x1F;
        assemble_word(SYNC_COMMAND, bits)
    }

    /// Decode from a 20-bit value.
    pub fn decode(word: u32) -> Result<Self, BusError> {
        let bits = parse_word(word, SYNC_COMMAND)?;
        Ok(Self {
            rt_address: ((bits >> 11) & 0x1F) as u8,
            transmit: (bits >> 10) & 1 == 1,
            subaddress: ((bits >> 5) & 0x1F) as u8,
            word_count: (bits & 0x1F) as u8,
        })
    }
}

/// MIL-STD-1553B Status Word.
///
/// Layout (MSB first within the 16-bit content):
/// - RT Address [4:0] (5 bits)
/// - Message Error (1 bit)
/// - Instrumentation (1 bit)
/// - Service Request (1 bit)
/// - Reserved (3 bits, must be 0)
/// - Broadcast Received (1 bit)
/// - Busy (1 bit)
/// - Subsystem Flag (1 bit)
/// - Dynamic Bus Control Acceptance (1 bit)
/// - Terminal Flag (1 bit)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatusWord {
    /// RT address (0-30).
    pub rt_address: u8,
    /// Message Error flag.
    pub message_error: bool,
    /// Instrumentation bit.
    pub instrumentation: bool,
    /// Service Request flag.
    pub service_request: bool,
    /// Broadcast Received flag.
    pub broadcast_received: bool,
    /// Busy flag.
    pub busy: bool,
    /// Subsystem Flag.
    pub subsystem_flag: bool,
    /// Dynamic Bus Control Acceptance.
    pub dbc_accept: bool,
    /// Terminal Flag.
    pub terminal_flag: bool,
}

impl StatusWord {
    /// Create a new status word with all flags clear.
    pub fn new(rt_address: u8) -> Result<Self, BusError> {
        if rt_address > 30 {
            return Err(BusError::InvalidId);
        }
        Ok(Self {
            rt_address,
            message_error: false,
            instrumentation: false,
            service_request: false,
            broadcast_received: false,
            busy: false,
            subsystem_flag: false,
            dbc_accept: false,
            terminal_flag: false,
        })
    }

    /// Encode to a 20-bit value (data sync + 16-bit content + parity).
    pub fn encode(&self) -> u32 {
        let mut bits: u16 = 0;
        bits |= (self.rt_address as u16 & 0x1F) << 11;
        if self.message_error {
            bits |= 1 << 10;
        }
        if self.instrumentation {
            bits |= 1 << 9;
        }
        if self.service_request {
            bits |= 1 << 8;
        }
        // bits 7,6,5 reserved = 0
        if self.broadcast_received {
            bits |= 1 << 4;
        }
        if self.busy {
            bits |= 1 << 3;
        }
        if self.subsystem_flag {
            bits |= 1 << 2;
        }
        if self.dbc_accept {
            bits |= 1 << 1;
        }
        if self.terminal_flag {
            bits |= 1;
        }
        assemble_word(SYNC_DATA, bits)
    }

    /// Decode from a 20-bit value.
    pub fn decode(word: u32) -> Result<Self, BusError> {
        let bits = parse_word(word, SYNC_DATA)?;
        Ok(Self {
            rt_address: ((bits >> 11) & 0x1F) as u8,
            message_error: (bits >> 10) & 1 == 1,
            instrumentation: (bits >> 9) & 1 == 1,
            service_request: (bits >> 8) & 1 == 1,
            broadcast_received: (bits >> 4) & 1 == 1,
            busy: (bits >> 3) & 1 == 1,
            subsystem_flag: (bits >> 2) & 1 == 1,
            dbc_accept: (bits >> 1) & 1 == 1,
            terminal_flag: bits & 1 == 1,
        })
    }

    /// Returns true if any error flag is set.
    pub fn has_error(&self) -> bool {
        self.message_error || self.terminal_flag
    }
}

/// MIL-STD-1553B Data Word.
///
/// Contains a 16-bit data payload with data sync and odd parity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DataWord {
    /// 16-bit data payload.
    pub data: u16,
}

impl DataWord {
    /// Create a new data word.
    pub fn new(data: u16) -> Self {
        Self { data }
    }

    /// Encode to a 20-bit value (data sync + 16-bit data + parity).
    pub fn encode(&self) -> u32 {
        assemble_word(SYNC_DATA, self.data)
    }

    /// Decode from a 20-bit value.
    pub fn decode(word: u32) -> Result<Self, BusError> {
        let bits = parse_word(word, SYNC_DATA)?;
        Ok(Self { data: bits })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === Command Word Tests ===

    #[test]
    fn test_command_word_encode_decode() {
        let cmd = CommandWord::new(5, false, 3, 8).unwrap();
        let encoded = cmd.encode();
        let decoded = CommandWord::decode(encoded).unwrap();
        assert_eq!(decoded.rt_address, 5);
        assert!(!decoded.transmit);
        assert_eq!(decoded.subaddress, 3);
        assert_eq!(decoded.word_count, 8);
    }

    #[test]
    fn test_command_word_transmit_bit() {
        let cmd = CommandWord::new(7, true, 2, 4).unwrap();
        let encoded = cmd.encode();
        let decoded = CommandWord::decode(encoded).unwrap();
        assert!(decoded.transmit);
    }

    #[test]
    fn test_command_word_broadcast() {
        let cmd = CommandWord::new(RT_BROADCAST, false, 3, 8).unwrap();
        assert_eq!(cmd.rt_address, 31);
    }

    #[test]
    fn test_command_word_invalid_rt() {
        assert_eq!(
            CommandWord::new(32, false, 0, 0).err(),
            Some(BusError::InvalidId)
        );
    }

    #[test]
    fn test_command_word_invalid_sa() {
        assert_eq!(
            CommandWord::new(0, false, 32, 0).err(),
            Some(BusError::OutOfRange)
        );
    }

    #[test]
    fn test_command_word_invalid_wc() {
        assert_eq!(
            CommandWord::new(0, false, 0, 32).err(),
            Some(BusError::OutOfRange)
        );
    }

    #[test]
    fn test_command_word_mode_code() {
        let cmd = CommandWord::new(5, true, 0, 2).unwrap();
        assert!(cmd.is_mode_code());
        assert_eq!(cmd.data_word_count(), 0);

        let cmd2 = CommandWord::new(5, true, 31, 2).unwrap();
        assert!(cmd2.is_mode_code());
    }

    #[test]
    fn test_command_word_not_mode_code() {
        let cmd = CommandWord::new(5, false, 3, 8).unwrap();
        assert!(!cmd.is_mode_code());
        assert_eq!(cmd.data_word_count(), 8);
    }

    #[test]
    fn test_command_word_count_zero_means_32() {
        let cmd = CommandWord::new(5, false, 3, 0).unwrap();
        assert_eq!(cmd.data_word_count(), 32);
    }

    #[test]
    fn test_command_word_parity_error() {
        let cmd = CommandWord::new(5, false, 3, 8).unwrap();
        let encoded = cmd.encode();
        // Flip the parity bit
        let corrupted = encoded ^ 1;
        assert_eq!(
            CommandWord::decode(corrupted).err(),
            Some(BusError::ParityError)
        );
    }

    #[test]
    fn test_command_word_wrong_sync() {
        let cmd = CommandWord::new(5, false, 3, 8).unwrap();
        let encoded = cmd.encode();
        // Change sync from command (111) to data (110)
        let wrong_sync = (encoded & 0x1FFFF) | (0b110 << 17);
        assert_eq!(
            CommandWord::decode(wrong_sync).err(),
            Some(BusError::InvalidHeader)
        );
    }

    #[test]
    fn test_command_word_all_rt_addresses() {
        for rt in 0..=31 {
            let cmd = CommandWord::new(rt, false, 1, 1).unwrap();
            let decoded = CommandWord::decode(cmd.encode()).unwrap();
            assert_eq!(decoded.rt_address, rt);
        }
    }

    // === Status Word Tests ===

    #[test]
    fn test_status_word_encode_decode() {
        let mut sw = StatusWord::new(5).unwrap();
        sw.service_request = true;
        sw.busy = true;
        let encoded = sw.encode();
        let decoded = StatusWord::decode(encoded).unwrap();
        assert_eq!(decoded.rt_address, 5);
        assert!(decoded.service_request);
        assert!(decoded.busy);
        assert!(!decoded.message_error);
        assert!(!decoded.broadcast_received);
    }

    #[test]
    fn test_status_word_all_flags() {
        let mut sw = StatusWord::new(10).unwrap();
        sw.message_error = true;
        sw.instrumentation = true;
        sw.service_request = true;
        sw.broadcast_received = true;
        sw.busy = true;
        sw.subsystem_flag = true;
        sw.dbc_accept = true;
        sw.terminal_flag = true;
        let decoded = StatusWord::decode(sw.encode()).unwrap();
        assert!(decoded.message_error);
        assert!(decoded.instrumentation);
        assert!(decoded.service_request);
        assert!(decoded.broadcast_received);
        assert!(decoded.busy);
        assert!(decoded.subsystem_flag);
        assert!(decoded.dbc_accept);
        assert!(decoded.terminal_flag);
    }

    #[test]
    fn test_status_word_invalid_rt() {
        assert_eq!(StatusWord::new(31).err(), Some(BusError::InvalidId));
    }

    #[test]
    fn test_status_word_parity_error() {
        let sw = StatusWord::new(5).unwrap();
        let encoded = sw.encode();
        let corrupted = encoded ^ 1;
        assert_eq!(
            StatusWord::decode(corrupted).err(),
            Some(BusError::ParityError)
        );
    }

    #[test]
    fn test_status_word_has_error() {
        let mut sw = StatusWord::new(5).unwrap();
        assert!(!sw.has_error());
        sw.message_error = true;
        assert!(sw.has_error());
    }

    #[test]
    fn test_status_word_busy_not_error() {
        let mut sw = StatusWord::new(5).unwrap();
        sw.busy = true;
        assert!(!sw.has_error());
    }

    // === Data Word Tests ===

    #[test]
    fn test_data_word_encode_decode() {
        let dw = DataWord::new(0xABCD);
        let encoded = dw.encode();
        let decoded = DataWord::decode(encoded).unwrap();
        assert_eq!(decoded.data, 0xABCD);
    }

    #[test]
    fn test_data_word_zero() {
        let dw = DataWord::new(0);
        let decoded = DataWord::decode(dw.encode()).unwrap();
        assert_eq!(decoded.data, 0);
    }

    #[test]
    fn test_data_word_all_ones() {
        let dw = DataWord::new(0xFFFF);
        let decoded = DataWord::decode(dw.encode()).unwrap();
        assert_eq!(decoded.data, 0xFFFF);
    }

    #[test]
    fn test_data_word_parity_error() {
        let dw = DataWord::new(0x1234);
        let encoded = dw.encode();
        let corrupted = encoded ^ 1;
        assert_eq!(
            DataWord::decode(corrupted).err(),
            Some(BusError::ParityError)
        );
    }

    #[test]
    fn test_data_word_wrong_sync() {
        let dw = DataWord::new(0x1234);
        let encoded = dw.encode();
        // Change sync from data (110) to command (111)
        let wrong_sync = (encoded & 0x1FFFF) | (0b111 << 17);
        assert_eq!(
            DataWord::decode(wrong_sync).err(),
            Some(BusError::InvalidHeader)
        );
    }

    #[test]
    fn test_data_word_multi_word_sequence() {
        let words: [u16; 4] = [0x1111, 0x2222, 0x3333, 0x4444];
        let encoded: alloc::vec::Vec<u32> =
            words.iter().map(|&w| DataWord::new(w).encode()).collect();
        for (i, &enc) in encoded.iter().enumerate() {
            let dec = DataWord::decode(enc).unwrap();
            assert_eq!(dec.data, words[i]);
        }
    }

    // === Internal helper tests (assemble_word / parse_word) ===

    #[test]
    fn test_assemble_word_parity_odd_ones() {
        // 0x0001 has one '1' bit (odd) -> parity bit must be 0 for total odd.
        let word = assemble_word(SYNC_DATA, 0x0001);
        assert_eq!((word >> 17) & 0x7, SYNC_DATA);
        assert_eq!(word & 1, 0);
    }

    #[test]
    fn test_assemble_word_parity_even_ones() {
        // 0x0003 has two '1' bits (even) -> parity bit must be 1 for total odd.
        let word = assemble_word(SYNC_DATA, 0x0003);
        assert_eq!(word & 1, 1);
    }

    #[test]
    fn test_parse_word_round_trip() {
        let bits: u16 = 0xABCD;
        let word = assemble_word(SYNC_COMMAND, bits);
        assert_eq!(parse_word(word, SYNC_COMMAND).unwrap(), bits);
    }

    #[test]
    fn test_parse_word_wrong_sync_rejected() {
        let word = assemble_word(SYNC_COMMAND, 0x1234);
        assert_eq!(
            parse_word(word, SYNC_DATA).err(),
            Some(BusError::InvalidHeader)
        );
    }

    #[test]
    fn test_parse_word_parity_flip_rejected() {
        let word = assemble_word(SYNC_DATA, 0x5555) ^ 1;
        assert_eq!(
            parse_word(word, SYNC_DATA).err(),
            Some(BusError::ParityError)
        );
    }

    #[test]
    fn test_sync_type_differentiation() {
        let cmd = CommandWord::new(5, false, 3, 8).unwrap();
        let dw = DataWord::new(0x1234);
        let cmd_sync = (cmd.encode() >> 17) & 0x7;
        let data_sync = (dw.encode() >> 17) & 0x7;
        assert_eq!(cmd_sync, 0b111); // command sync
        assert_eq!(data_sync, 0b110); // data sync
        assert_ne!(cmd_sync, data_sync);
    }
}
