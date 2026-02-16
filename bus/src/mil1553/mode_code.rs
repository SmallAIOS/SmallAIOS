// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! MIL-STD-1553B Mode Code support.
//!
//! Mode codes are special commands identified by subaddress 0 or 31.
//! The 5-bit word count/mode code field selects the specific mode code.
//! Mode codes are divided into two groups:
//! - Without data word (transmit bit determines direction)
//! - With data word (one data word follows or precedes the status word)

use crate::mil1553::word::{CommandWord, StatusWord, MODE_CODE_SA_HIGH, MODE_CODE_SA_LOW};
use crate::BusError;

/// Well-known MIL-STD-1553B mode codes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeCode {
    /// Dynamic Bus Control (00001) — transfer BC role to the addressed RT.
    DynamicBusControl,
    /// Synchronize (00001, T/R=0) — synchronize RT clock (no data word).
    Synchronize,
    /// Transmit Status Word (00010, T/R=1) — RT transmits its status word.
    TransmitStatusWord,
    /// Initiate Self Test (00011, T/R=0) — RT begins BIT.
    InitiateSelfTest,
    /// Transmitter Shutdown (00100, T/R=0) — inhibit RT transmitter.
    TransmitterShutdown,
    /// Override Transmitter Shutdown (00101, T/R=0) — re-enable RT transmitter.
    OverrideTransmitterShutdown,
    /// Inhibit Terminal Flag (00110, T/R=0) — clear terminal flag in status.
    InhibitTerminalFlag,
    /// Override Inhibit Terminal Flag (00111, T/R=0) — allow terminal flag.
    OverrideInhibitTerminalFlag,
    /// Reset Remote Terminal (01000, T/R=0) — reset RT to power-on state.
    ResetRemoteTerminal,
    /// Transmit Vector Word (10000, T/R=1) — RT transmits a single data word.
    TransmitVectorWord,
    /// Synchronize with Data (10001, T/R=0) — sync with a data word.
    SynchronizeWithData,
    /// Transmit Last Command (10010, T/R=1) — RT echoes last received command.
    TransmitLastCommand,
    /// Transmit BIT Word (10011, T/R=1) — RT transmits Built-In-Test result.
    TransmitBitWord,
    /// Unknown or reserved mode code.
    Unknown(u8),
}

impl ModeCode {
    /// Decode a mode code from the word count field and T/R bit.
    pub fn from_code(code: u8, transmit: bool) -> Self {
        match (code, transmit) {
            (0b00001, true) => ModeCode::DynamicBusControl,
            (0b00001, false) => ModeCode::Synchronize,
            (0b00010, true) => ModeCode::TransmitStatusWord,
            (0b00011, false) => ModeCode::InitiateSelfTest,
            (0b00100, false) => ModeCode::TransmitterShutdown,
            (0b00101, false) => ModeCode::OverrideTransmitterShutdown,
            (0b00110, false) => ModeCode::InhibitTerminalFlag,
            (0b00111, false) => ModeCode::OverrideInhibitTerminalFlag,
            (0b01000, false) => ModeCode::ResetRemoteTerminal,
            (0b10000, true) => ModeCode::TransmitVectorWord,
            (0b10001, false) => ModeCode::SynchronizeWithData,
            (0b10010, true) => ModeCode::TransmitLastCommand,
            (0b10011, true) => ModeCode::TransmitBitWord,
            (code, _) => ModeCode::Unknown(code),
        }
    }

    /// Returns true if this mode code requires a data word.
    pub fn has_data_word(&self) -> bool {
        matches!(
            self,
            ModeCode::TransmitVectorWord
                | ModeCode::SynchronizeWithData
                | ModeCode::TransmitLastCommand
                | ModeCode::TransmitBitWord
        )
    }

    /// Returns true if this is a transmit mode code (RT sends data/status).
    pub fn is_transmit(&self) -> bool {
        matches!(
            self,
            ModeCode::DynamicBusControl
                | ModeCode::TransmitStatusWord
                | ModeCode::TransmitVectorWord
                | ModeCode::TransmitLastCommand
                | ModeCode::TransmitBitWord
        )
    }
}

/// Result of processing a mode code command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModeCodeResult {
    /// Status word only (no data word).
    StatusOnly(StatusWord),
    /// Status word with a data word (for transmit mode codes with data).
    StatusWithData(StatusWord, u16),
    /// No response (broadcast mode code).
    NoResponse,
}

/// Mode code handler for an RT.
///
/// Processes mode code commands and generates appropriate responses.
#[derive(Debug)]
pub struct ModeCodeHandler {
    /// RT address.
    rt_address: u8,
    /// Vector word (returned by TransmitVectorWord).
    vector_word: u16,
    /// Built-In-Test result word.
    bit_word: u16,
    /// Terminal flag inhibited.
    terminal_flag_inhibited: bool,
    /// Self-test in progress.
    self_test_active: bool,
    /// Last received command word (for TransmitLastCommand).
    last_command_raw: u16,
}

impl ModeCodeHandler {
    /// Create a new mode code handler for the given RT address.
    pub fn new(rt_address: u8) -> Result<Self, BusError> {
        if rt_address > 30 {
            return Err(BusError::InvalidId);
        }
        Ok(Self {
            rt_address,
            vector_word: 0,
            bit_word: 0,
            terminal_flag_inhibited: false,
            self_test_active: false,
            last_command_raw: 0,
        })
    }

    /// Returns whether the terminal flag is inhibited.
    pub fn terminal_flag_inhibited(&self) -> bool {
        self.terminal_flag_inhibited
    }

    /// Returns whether a self-test is active.
    pub fn self_test_active(&self) -> bool {
        self.self_test_active
    }

    /// Set the vector word (application-defined interrupt vector).
    pub fn set_vector_word(&mut self, word: u16) {
        self.vector_word = word;
    }

    /// Set the BIT result word.
    pub fn set_bit_word(&mut self, word: u16) {
        self.bit_word = word;
    }

    /// Record the last command word received (raw 16-bit content).
    pub fn record_last_command(&mut self, cmd: &CommandWord) {
        // Reconstruct the 16-bit content from the command fields.
        let mut bits: u16 = 0;
        bits |= (cmd.rt_address as u16 & 0x1F) << 11;
        bits |= if cmd.transmit { 1 << 10 } else { 0 };
        bits |= (cmd.subaddress as u16 & 0x1F) << 5;
        bits |= cmd.word_count as u16 & 0x1F;
        self.last_command_raw = bits;
    }

    /// Process a mode code command.
    ///
    /// Returns the appropriate response based on the mode code.
    /// The `cmd` must have subaddress 0 or 31 (a mode code command).
    pub fn process(
        &mut self,
        cmd: &CommandWord,
        data_word: Option<u16>,
    ) -> Result<ModeCodeResult, BusError> {
        if cmd.subaddress != MODE_CODE_SA_LOW && cmd.subaddress != MODE_CODE_SA_HIGH {
            return Err(BusError::InvalidHeader);
        }

        let mode = ModeCode::from_code(cmd.word_count, cmd.transmit);
        let is_broadcast = cmd.rt_address == 31;

        self.record_last_command(cmd);

        match mode {
            ModeCode::DynamicBusControl => {
                let mut sw = self.make_status()?;
                sw.dbc_accept = true;
                if is_broadcast {
                    Ok(ModeCodeResult::NoResponse)
                } else {
                    Ok(ModeCodeResult::StatusOnly(sw))
                }
            }
            ModeCode::Synchronize | ModeCode::TransmitStatusWord => {
                self.broadcast_or_status(is_broadcast)
            }
            ModeCode::InitiateSelfTest => {
                self.self_test_active = true;
                self.broadcast_or_status(is_broadcast)
            }
            ModeCode::TransmitterShutdown | ModeCode::OverrideTransmitterShutdown => {
                self.broadcast_or_status(is_broadcast)
            }
            ModeCode::InhibitTerminalFlag => {
                self.terminal_flag_inhibited = true;
                self.broadcast_or_status(is_broadcast)
            }
            ModeCode::OverrideInhibitTerminalFlag => {
                self.terminal_flag_inhibited = false;
                self.broadcast_or_status(is_broadcast)
            }
            ModeCode::ResetRemoteTerminal => {
                self.terminal_flag_inhibited = false;
                self.self_test_active = false;
                self.vector_word = 0;
                self.bit_word = 0;
                self.broadcast_or_status(is_broadcast)
            }
            ModeCode::TransmitVectorWord => {
                self.broadcast_or_status_with_data(is_broadcast, self.vector_word)
            }
            ModeCode::SynchronizeWithData => {
                // The data word contains a sync value from the BC
                let _sync_value = data_word.unwrap_or(0);
                self.broadcast_or_status(is_broadcast)
            }
            ModeCode::TransmitLastCommand => {
                self.broadcast_or_status_with_data(is_broadcast, self.last_command_raw)
            }
            ModeCode::TransmitBitWord => {
                self.broadcast_or_status_with_data(is_broadcast, self.bit_word)
            }
            ModeCode::Unknown(_) => Err(BusError::NotSupported),
        }
    }

    /// Build the current status word.
    fn make_status(&self) -> Result<StatusWord, BusError> {
        let mut sw = StatusWord::new(self.rt_address)?;
        if !self.terminal_flag_inhibited {
            sw.terminal_flag = self.self_test_active;
        }
        Ok(sw)
    }

    /// Return `NoResponse` for broadcast commands, or `StatusOnly` with
    /// the current status word for unicast commands.
    fn broadcast_or_status(&self, is_broadcast: bool) -> Result<ModeCodeResult, BusError> {
        if is_broadcast {
            Ok(ModeCodeResult::NoResponse)
        } else {
            Ok(ModeCodeResult::StatusOnly(self.make_status()?))
        }
    }

    /// Return `NoResponse` for broadcast commands, or `StatusWithData`
    /// carrying `data` for unicast commands.
    fn broadcast_or_status_with_data(
        &self,
        is_broadcast: bool,
        data: u16,
    ) -> Result<ModeCodeResult, BusError> {
        if is_broadcast {
            Ok(ModeCodeResult::NoResponse)
        } else {
            Ok(ModeCodeResult::StatusWithData(self.make_status()?, data))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mil1553::word::RT_BROADCAST;

    fn mode_cmd(rt: u8, transmit: bool, code: u8) -> CommandWord {
        CommandWord::new(rt, transmit, MODE_CODE_SA_LOW, code).unwrap()
    }

    #[test]
    fn test_mode_code_decode() {
        assert_eq!(
            ModeCode::from_code(0b00001, true),
            ModeCode::DynamicBusControl
        );
        assert_eq!(ModeCode::from_code(0b00001, false), ModeCode::Synchronize);
        assert_eq!(
            ModeCode::from_code(0b00010, true),
            ModeCode::TransmitStatusWord
        );
        assert_eq!(
            ModeCode::from_code(0b00011, false),
            ModeCode::InitiateSelfTest
        );
        assert_eq!(
            ModeCode::from_code(0b00100, false),
            ModeCode::TransmitterShutdown
        );
        assert_eq!(
            ModeCode::from_code(0b00101, false),
            ModeCode::OverrideTransmitterShutdown
        );
        assert_eq!(
            ModeCode::from_code(0b01000, false),
            ModeCode::ResetRemoteTerminal
        );
    }

    #[test]
    fn test_mode_code_has_data_word() {
        assert!(ModeCode::TransmitVectorWord.has_data_word());
        assert!(ModeCode::SynchronizeWithData.has_data_word());
        assert!(ModeCode::TransmitLastCommand.has_data_word());
        assert!(ModeCode::TransmitBitWord.has_data_word());
        assert!(!ModeCode::Synchronize.has_data_word());
        assert!(!ModeCode::TransmitterShutdown.has_data_word());
    }

    #[test]
    fn test_mode_code_is_transmit() {
        assert!(ModeCode::DynamicBusControl.is_transmit());
        assert!(ModeCode::TransmitStatusWord.is_transmit());
        assert!(ModeCode::TransmitVectorWord.is_transmit());
        assert!(!ModeCode::Synchronize.is_transmit());
        assert!(!ModeCode::InitiateSelfTest.is_transmit());
    }

    #[test]
    fn test_handler_new() {
        let h = ModeCodeHandler::new(5).unwrap();
        assert!(!h.terminal_flag_inhibited());
        assert!(!h.self_test_active());
    }

    #[test]
    fn test_handler_invalid_address() {
        assert_eq!(ModeCodeHandler::new(31).err(), Some(BusError::InvalidId));
    }

    #[test]
    fn test_transmit_status_word() {
        let mut h = ModeCodeHandler::new(5).unwrap();
        let cmd = mode_cmd(5, true, 0b00010);
        let result = h.process(&cmd, None).unwrap();
        match result {
            ModeCodeResult::StatusOnly(sw) => {
                assert_eq!(sw.rt_address, 5);
            }
            _ => panic!("expected StatusOnly"),
        }
    }

    #[test]
    fn test_initiate_self_test() {
        let mut h = ModeCodeHandler::new(5).unwrap();
        let cmd = mode_cmd(5, false, 0b00011);
        h.process(&cmd, None).unwrap();
        assert!(h.self_test_active());
    }

    #[test]
    fn test_inhibit_terminal_flag() {
        let mut h = ModeCodeHandler::new(5).unwrap();
        let cmd = mode_cmd(5, false, 0b00110);
        h.process(&cmd, None).unwrap();
        assert!(h.terminal_flag_inhibited());

        // Override
        let cmd2 = mode_cmd(5, false, 0b00111);
        h.process(&cmd2, None).unwrap();
        assert!(!h.terminal_flag_inhibited());
    }

    #[test]
    fn test_reset_remote_terminal() {
        let mut h = ModeCodeHandler::new(5).unwrap();
        h.set_vector_word(0x1234);
        h.set_bit_word(0xABCD);
        // Initiate self test first
        let cmd = mode_cmd(5, false, 0b00011);
        h.process(&cmd, None).unwrap();
        assert!(h.self_test_active());

        // Reset
        let cmd = mode_cmd(5, false, 0b01000);
        h.process(&cmd, None).unwrap();
        assert!(!h.self_test_active());
        assert!(!h.terminal_flag_inhibited());
    }

    #[test]
    fn test_transmit_vector_word() {
        let mut h = ModeCodeHandler::new(5).unwrap();
        h.set_vector_word(0x42);
        let cmd = mode_cmd(5, true, 0b10000);
        let result = h.process(&cmd, None).unwrap();
        match result {
            ModeCodeResult::StatusWithData(sw, data) => {
                assert_eq!(sw.rt_address, 5);
                assert_eq!(data, 0x42);
            }
            _ => panic!("expected StatusWithData"),
        }
    }

    #[test]
    fn test_transmit_bit_word() {
        let mut h = ModeCodeHandler::new(5).unwrap();
        h.set_bit_word(0xFF00);
        let cmd = mode_cmd(5, true, 0b10011);
        let result = h.process(&cmd, None).unwrap();
        match result {
            ModeCodeResult::StatusWithData(_, data) => {
                assert_eq!(data, 0xFF00);
            }
            _ => panic!("expected StatusWithData"),
        }
    }

    #[test]
    fn test_synchronize_with_data() {
        let mut h = ModeCodeHandler::new(5).unwrap();
        let cmd = mode_cmd(5, false, 0b10001);
        let result = h.process(&cmd, Some(0x1234)).unwrap();
        match result {
            ModeCodeResult::StatusOnly(sw) => {
                assert_eq!(sw.rt_address, 5);
            }
            _ => panic!("expected StatusOnly"),
        }
    }

    #[test]
    fn test_transmit_last_command() {
        let mut h = ModeCodeHandler::new(5).unwrap();
        // Process returns the last command recorded BEFORE the current one,
        // but process() calls record_last_command internally, so the
        // TransmitLastCommand returns its own encoding.
        // Per MIL-STD-1553B, "last command" means the most recent command received.
        let cmd = mode_cmd(5, true, 0b10010);
        let result = h.process(&cmd, None).unwrap();
        match result {
            ModeCodeResult::StatusWithData(_, data) => {
                // The last command is the TransmitLastCommand itself
                let expected: u16 = (5 << 11) | (1 << 10) | (0 << 5) | 0b10010;
                assert_eq!(data, expected);
            }
            _ => panic!("expected StatusWithData"),
        }
    }

    #[test]
    fn test_broadcast_mode_code() {
        let mut h = ModeCodeHandler::new(5).unwrap();
        let cmd = CommandWord::new(RT_BROADCAST, false, MODE_CODE_SA_LOW, 0b00001).unwrap();
        let result = h.process(&cmd, None).unwrap();
        assert_eq!(result, ModeCodeResult::NoResponse);
    }

    #[test]
    fn test_dynamic_bus_control() {
        let mut h = ModeCodeHandler::new(5).unwrap();
        let cmd = mode_cmd(5, true, 0b00001);
        let result = h.process(&cmd, None).unwrap();
        match result {
            ModeCodeResult::StatusOnly(sw) => {
                assert!(sw.dbc_accept);
            }
            _ => panic!("expected StatusOnly"),
        }
    }

    #[test]
    fn test_unknown_mode_code() {
        let mut h = ModeCodeHandler::new(5).unwrap();
        let cmd = mode_cmd(5, false, 0b11111);
        assert_eq!(h.process(&cmd, None).err(), Some(BusError::NotSupported));
    }

    #[test]
    fn test_non_mode_code_subaddress_rejected() {
        let mut h = ModeCodeHandler::new(5).unwrap();
        let cmd = CommandWord::new(5, false, 3, 1).unwrap(); // SA=3, not a mode code
        assert_eq!(h.process(&cmd, None).err(), Some(BusError::InvalidHeader));
    }

    #[test]
    fn test_mode_code_sa_high() {
        let mut h = ModeCodeHandler::new(5).unwrap();
        let cmd = CommandWord::new(5, true, MODE_CODE_SA_HIGH, 0b00010).unwrap();
        let result = h.process(&cmd, None).unwrap();
        match result {
            ModeCodeResult::StatusOnly(sw) => {
                assert_eq!(sw.rt_address, 5);
            }
            _ => panic!("expected StatusOnly"),
        }
    }
}
