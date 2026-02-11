// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! MIL-STD-1553B Remote Terminal (RT) response state machine.
//!
//! An RT listens for command words addressed to its configured RT address,
//! processes receive/transmit commands, and generates status word responses.

use crate::BusError;
use crate::mil1553::word::{CommandWord, StatusWord, RT_BROADCAST};

/// Maximum number of subaddresses (0-31, but 0 and 31 are mode codes).
pub const NUM_SUBADDRESSES: usize = 32;

/// Maximum data words per subaddress buffer.
pub const SUBADDRESS_BUF_SIZE: usize = 32;

/// RT operating state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RtState {
    /// RT is ready and listening.
    Ready,
    /// RT transmitter is inhibited on the specified bus.
    TransmitterShutdown,
    /// RT is busy and cannot process commands.
    Busy,
}

/// Remote Terminal state machine.
#[derive(Debug)]
pub struct RtTerminal {
    /// Configured RT address (0-30).
    rt_address: u8,
    /// Current operating state.
    state: RtState,
    /// Per-subaddress data buffers.
    buffers: [[u16; SUBADDRESS_BUF_SIZE]; NUM_SUBADDRESSES],
    /// Per-subaddress valid data word count.
    buf_counts: [u8; NUM_SUBADDRESSES],
    /// Broadcast received flag (set on next status word).
    broadcast_received: bool,
    /// Service request flag.
    service_request: bool,
    /// Subsystem flag.
    subsystem_flag: bool,
    /// Last received command word (for mode code 00010).
    last_command: Option<CommandWord>,
}

impl RtTerminal {
    /// Create a new RT with the given address.
    pub fn new(rt_address: u8) -> Result<Self, BusError> {
        if rt_address > 30 {
            return Err(BusError::InvalidId);
        }
        Ok(Self {
            rt_address,
            state: RtState::Ready,
            buffers: [[0u16; SUBADDRESS_BUF_SIZE]; NUM_SUBADDRESSES],
            buf_counts: [0u8; NUM_SUBADDRESSES],
            broadcast_received: false,
            service_request: false,
            subsystem_flag: false,
            last_command: None,
        })
    }

    /// Returns the RT address.
    pub fn rt_address(&self) -> u8 {
        self.rt_address
    }

    /// Returns the current state.
    pub fn state(&self) -> RtState {
        self.state
    }

    /// Set the RT to busy state.
    pub fn set_busy(&mut self, busy: bool) {
        self.state = if busy { RtState::Busy } else { RtState::Ready };
    }

    /// Set the service request flag.
    pub fn set_service_request(&mut self, flag: bool) {
        self.service_request = flag;
    }

    /// Set the subsystem flag.
    pub fn set_subsystem_flag(&mut self, flag: bool) {
        self.subsystem_flag = flag;
    }

    /// Check whether this RT should respond to the given command.
    pub fn is_addressed(&self, cmd: &CommandWord) -> bool {
        cmd.rt_address == self.rt_address || cmd.rt_address == RT_BROADCAST
    }

    /// Process a receive command (BC writes data to this RT).
    ///
    /// Stores data in the subaddress buffer and returns the status word to send.
    /// Returns `None` for broadcast commands (no status response).
    pub fn process_receive(
        &mut self,
        cmd: &CommandWord,
        data_words: &[u16],
    ) -> Result<Option<StatusWord>, BusError> {
        if !self.is_addressed(cmd) {
            return Err(BusError::InvalidId);
        }
        if cmd.transmit {
            return Err(BusError::InvalidHeader);
        }
        self.last_command = Some(*cmd);
        let sa = cmd.subaddress as usize;
        let count = data_words.len().min(SUBADDRESS_BUF_SIZE);
        self.buffers[sa][..count].copy_from_slice(&data_words[..count]);
        self.buf_counts[sa] = count as u8;

        if cmd.rt_address == RT_BROADCAST {
            self.broadcast_received = true;
            return Ok(None); // No status response for broadcast
        }

        Ok(Some(self.make_status_word()))
    }

    /// Process a transmit command (BC reads data from this RT).
    ///
    /// Returns the status word and data words to send.
    /// Returns `None` for broadcast commands.
    pub fn process_transmit(
        &mut self,
        cmd: &CommandWord,
    ) -> Result<Option<(StatusWord, alloc::vec::Vec<u16>)>, BusError> {
        if !self.is_addressed(cmd) {
            return Err(BusError::InvalidId);
        }
        if !cmd.transmit {
            return Err(BusError::InvalidHeader);
        }
        self.last_command = Some(*cmd);
        let sa = cmd.subaddress as usize;
        let wc = cmd.data_word_count() as usize;

        if cmd.rt_address == RT_BROADCAST {
            self.broadcast_received = true;
            return Ok(None);
        }

        let available = self.buf_counts[sa] as usize;
        let count = wc.min(available);
        let data = alloc::vec::Vec::from(&self.buffers[sa][..count]);

        Ok(Some((self.make_status_word(), data)))
    }

    /// Load data into a subaddress buffer (application-side).
    pub fn load_subaddress(&mut self, sa: u8, data: &[u16]) -> Result<(), BusError> {
        if sa as usize >= NUM_SUBADDRESSES {
            return Err(BusError::OutOfRange);
        }
        if data.len() > SUBADDRESS_BUF_SIZE {
            return Err(BusError::PayloadTooLarge);
        }
        let sa_idx = sa as usize;
        self.buffers[sa_idx][..data.len()].copy_from_slice(data);
        self.buf_counts[sa_idx] = data.len() as u8;
        Ok(())
    }

    /// Read data from a subaddress buffer (application-side).
    pub fn read_subaddress(&self, sa: u8) -> Result<&[u16], BusError> {
        if sa as usize >= NUM_SUBADDRESSES {
            return Err(BusError::OutOfRange);
        }
        let sa_idx = sa as usize;
        let count = self.buf_counts[sa_idx] as usize;
        Ok(&self.buffers[sa_idx][..count])
    }

    /// Inhibit the transmitter (mode code response).
    pub fn shutdown_transmitter(&mut self) {
        self.state = RtState::TransmitterShutdown;
    }

    /// Re-enable the transmitter.
    pub fn enable_transmitter(&mut self) {
        if self.state == RtState::TransmitterShutdown {
            self.state = RtState::Ready;
        }
    }

    /// Build the current status word.
    fn make_status_word(&mut self) -> StatusWord {
        let mut sw = StatusWord::new(self.rt_address).unwrap();
        sw.busy = self.state == RtState::Busy;
        sw.broadcast_received = self.broadcast_received;
        sw.service_request = self.service_request;
        sw.subsystem_flag = self.subsystem_flag;
        // Clear broadcast_received after reporting it
        self.broadcast_received = false;
        sw
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_rt_create() {
        let rt = RtTerminal::new(5).unwrap();
        assert_eq!(rt.rt_address(), 5);
        assert_eq!(rt.state(), RtState::Ready);
    }

    #[test]
    fn test_rt_invalid_address() {
        assert_eq!(RtTerminal::new(31).err(), Some(BusError::InvalidId));
    }

    #[test]
    fn test_rt_is_addressed() {
        let rt = RtTerminal::new(5).unwrap();
        let cmd = CommandWord::new(5, false, 1, 1).unwrap();
        assert!(rt.is_addressed(&cmd));
        let other = CommandWord::new(6, false, 1, 1).unwrap();
        assert!(!rt.is_addressed(&other));
    }

    #[test]
    fn test_rt_is_addressed_broadcast() {
        let rt = RtTerminal::new(5).unwrap();
        let bcast = CommandWord::new(RT_BROADCAST, false, 1, 1).unwrap();
        assert!(rt.is_addressed(&bcast));
    }

    #[test]
    fn test_rt_process_receive() {
        let mut rt = RtTerminal::new(5).unwrap();
        let cmd = CommandWord::new(5, false, 3, 2).unwrap();
        let data = [0xAAAA, 0xBBBB];
        let status = rt.process_receive(&cmd, &data).unwrap().unwrap();
        assert_eq!(status.rt_address, 5);
        assert!(!status.busy);
        // Verify data stored
        assert_eq!(rt.read_subaddress(3).unwrap(), &[0xAAAA, 0xBBBB]);
    }

    #[test]
    fn test_rt_process_receive_broadcast() {
        let mut rt = RtTerminal::new(5).unwrap();
        let cmd = CommandWord::new(RT_BROADCAST, false, 3, 2).unwrap();
        let data = [0x1111, 0x2222];
        let result = rt.process_receive(&cmd, &data).unwrap();
        assert!(result.is_none()); // No status for broadcast

        // Next status should have broadcast_received set
        let cmd2 = CommandWord::new(5, true, 3, 2).unwrap();
        rt.load_subaddress(3, &[0xAAAA, 0xBBBB]).unwrap();
        let (status, _) = rt.process_transmit(&cmd2).unwrap().unwrap();
        assert!(status.broadcast_received);
    }

    #[test]
    fn test_rt_process_transmit() {
        let mut rt = RtTerminal::new(5).unwrap();
        rt.load_subaddress(3, &[0x1111, 0x2222, 0x3333, 0x4444]).unwrap();
        let cmd = CommandWord::new(5, true, 3, 4).unwrap();
        let (status, data) = rt.process_transmit(&cmd).unwrap().unwrap();
        assert_eq!(status.rt_address, 5);
        assert_eq!(data, vec![0x1111, 0x2222, 0x3333, 0x4444]);
    }

    #[test]
    fn test_rt_wrong_address_ignored() {
        let mut rt = RtTerminal::new(5).unwrap();
        let cmd = CommandWord::new(6, false, 1, 1).unwrap();
        assert_eq!(rt.process_receive(&cmd, &[0x1111]).err(), Some(BusError::InvalidId));
    }

    #[test]
    fn test_rt_busy_status() {
        let mut rt = RtTerminal::new(5).unwrap();
        rt.set_busy(true);
        assert_eq!(rt.state(), RtState::Busy);
        let cmd = CommandWord::new(5, false, 1, 1).unwrap();
        let status = rt.process_receive(&cmd, &[0x1111]).unwrap().unwrap();
        assert!(status.busy);
    }

    #[test]
    fn test_rt_load_and_read_subaddress() {
        let mut rt = RtTerminal::new(5).unwrap();
        rt.load_subaddress(10, &[0xAA, 0xBB, 0xCC]).unwrap();
        assert_eq!(rt.read_subaddress(10).unwrap(), &[0xAA, 0xBB, 0xCC]);
    }

    #[test]
    fn test_rt_load_subaddress_too_large() {
        let mut rt = RtTerminal::new(5).unwrap();
        let data = [0u16; 33];
        assert_eq!(rt.load_subaddress(1, &data).err(), Some(BusError::PayloadTooLarge));
    }

    #[test]
    fn test_rt_transmitter_shutdown() {
        let mut rt = RtTerminal::new(5).unwrap();
        rt.shutdown_transmitter();
        assert_eq!(rt.state(), RtState::TransmitterShutdown);
        rt.enable_transmitter();
        assert_eq!(rt.state(), RtState::Ready);
    }

    #[test]
    fn test_rt_service_request() {
        let mut rt = RtTerminal::new(5).unwrap();
        rt.set_service_request(true);
        let cmd = CommandWord::new(5, false, 1, 1).unwrap();
        let status = rt.process_receive(&cmd, &[0x1111]).unwrap().unwrap();
        assert!(status.service_request);
    }

    #[test]
    fn test_rt_broadcast_received_cleared_after_report() {
        let mut rt = RtTerminal::new(5).unwrap();
        // Send broadcast
        let bcast = CommandWord::new(RT_BROADCAST, false, 1, 1).unwrap();
        rt.process_receive(&bcast, &[0x1111]).unwrap();

        // First status should have broadcast_received
        let cmd = CommandWord::new(5, false, 2, 1).unwrap();
        let s1 = rt.process_receive(&cmd, &[0x2222]).unwrap().unwrap();
        assert!(s1.broadcast_received);

        // Second status should NOT have broadcast_received
        let s2 = rt.process_receive(&cmd, &[0x3333]).unwrap().unwrap();
        assert!(!s2.broadcast_received);
    }

    #[test]
    fn test_rt_process_receive_wrong_tr_bit() {
        let mut rt = RtTerminal::new(5).unwrap();
        // transmit=true for a receive operation
        let cmd = CommandWord::new(5, true, 1, 1).unwrap();
        assert_eq!(rt.process_receive(&cmd, &[0x1111]).err(), Some(BusError::InvalidHeader));
    }

    #[test]
    fn test_rt_process_transmit_wrong_tr_bit() {
        let mut rt = RtTerminal::new(5).unwrap();
        // transmit=false for a transmit operation
        let cmd = CommandWord::new(5, false, 1, 1).unwrap();
        assert_eq!(rt.process_transmit(&cmd).err(), Some(BusError::InvalidHeader));
    }
}
