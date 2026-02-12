// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! MIL-STD-1553B Bus Controller (BC) message sequencing.
//!
//! The BC manages the command/response protocol on the bus, scheduling
//! messages, detecting timeouts, and handling retries.

use crate::mil1553::word::{CommandWord, DataWord, StatusWord};
use crate::BusError;
use alloc::vec::Vec;

/// Response timeout in microseconds (14 us per MIL-STD-1553B).
pub const RESPONSE_TIMEOUT_US: u64 = 14;

/// Type of bus message transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    /// BC writes data to RT (BC-to-RT).
    BcToRt,
    /// BC reads data from RT (RT-to-BC).
    RtToBc,
    /// RT-to-RT transfer.
    RtToRt,
    /// Mode code command.
    ModeCode,
}

/// Bus identifier for dual-redundant operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bus {
    /// Bus A (primary).
    A,
    /// Bus B (secondary).
    B,
}

/// Result of a BC message transaction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageResult {
    /// Transaction completed successfully.
    Success { status: StatusWord, data: Vec<u16> },
    /// RT responded with busy flag.
    Busy,
    /// RT did not respond within the timeout.
    Timeout,
    /// RT responded with a message error.
    MessageError,
}

/// A scheduled BC message.
#[derive(Debug, Clone)]
pub struct BcMessage {
    /// Message type.
    pub msg_type: MessageType,
    /// Command word (always present).
    pub command: CommandWord,
    /// Second command word (for RT-to-RT only).
    pub command2: Option<CommandWord>,
    /// Data words to transmit (for BC-to-RT).
    pub tx_data: Vec<u16>,
    /// Target bus.
    pub bus: Bus,
    /// Retry count remaining.
    pub retries: u8,
}

impl BcMessage {
    /// Create a BC-to-RT message (write data to RT).
    pub fn bc_to_rt(rt: u8, subaddress: u8, data: &[u16], bus: Bus) -> Result<Self, BusError> {
        let wc = if data.len() == 32 {
            0
        } else {
            data.len() as u8
        };
        let command = CommandWord::new(rt, false, subaddress, wc)?;
        if data.is_empty() || data.len() > 32 {
            return Err(BusError::OutOfRange);
        }
        Ok(Self {
            msg_type: MessageType::BcToRt,
            command,
            command2: None,
            tx_data: Vec::from(data),
            bus,
            retries: 1,
        })
    }

    /// Create an RT-to-BC message (read data from RT).
    pub fn rt_to_bc(rt: u8, subaddress: u8, word_count: u8, bus: Bus) -> Result<Self, BusError> {
        let command = CommandWord::new(rt, true, subaddress, word_count)?;
        Ok(Self {
            msg_type: MessageType::RtToBc,
            command,
            command2: None,
            tx_data: Vec::new(),
            bus,
            retries: 1,
        })
    }

    /// Create an RT-to-RT transfer message.
    pub fn rt_to_rt(
        rx_rt: u8,
        rx_sa: u8,
        tx_rt: u8,
        tx_sa: u8,
        word_count: u8,
        bus: Bus,
    ) -> Result<Self, BusError> {
        // Receive command to destination RT
        let cmd_rx = CommandWord::new(rx_rt, false, rx_sa, word_count)?;
        // Transmit command to source RT
        let cmd_tx = CommandWord::new(tx_rt, true, tx_sa, word_count)?;
        Ok(Self {
            msg_type: MessageType::RtToRt,
            command: cmd_rx,
            command2: Some(cmd_tx),
            tx_data: Vec::new(),
            bus,
            retries: 1,
        })
    }

    /// Returns the encoded command word(s) for transmission.
    pub fn encode_commands(&self) -> Vec<u32> {
        let mut words = Vec::new();
        words.push(self.command.encode());
        if let Some(ref cmd2) = self.command2 {
            words.push(cmd2.encode());
        }
        words
    }

    /// Returns the encoded data words for BC-to-RT transmission.
    pub fn encode_data(&self) -> Vec<u32> {
        self.tx_data
            .iter()
            .map(|&d| DataWord::new(d).encode())
            .collect()
    }
}

/// Bus Controller message sequencer.
///
/// Manages a queue of scheduled messages and tracks per-bus health.
#[derive(Debug)]
pub struct BcSequencer {
    /// Pending messages to execute.
    queue: Vec<BcMessage>,
    /// Active bus selection.
    active_bus: Bus,
    /// Per-bus timeout counters.
    bus_a_timeouts: u64,
    bus_b_timeouts: u64,
    /// Per-bus error counters.
    bus_a_errors: u64,
    bus_b_errors: u64,
    /// Error threshold for bus failover.
    error_threshold: u64,
}

impl BcSequencer {
    /// Create a new BC sequencer.
    pub fn new(error_threshold: u64) -> Self {
        Self {
            queue: Vec::new(),
            active_bus: Bus::A,
            bus_a_timeouts: 0,
            bus_b_timeouts: 0,
            bus_a_errors: 0,
            bus_b_errors: 0,
            error_threshold,
        }
    }

    /// Schedule a message for execution.
    pub fn schedule(&mut self, msg: BcMessage) {
        self.queue.push(msg);
    }

    /// Take the next message from the queue.
    pub fn next_message(&mut self) -> Option<BcMessage> {
        if self.queue.is_empty() {
            None
        } else {
            Some(self.queue.remove(0))
        }
    }

    /// Record a transaction result and update bus health counters.
    pub fn record_result(&mut self, bus: Bus, result: &MessageResult) {
        match result {
            MessageResult::Timeout => match bus {
                Bus::A => self.bus_a_timeouts += 1,
                Bus::B => self.bus_b_timeouts += 1,
            },
            MessageResult::MessageError => match bus {
                Bus::A => self.bus_a_errors += 1,
                Bus::B => self.bus_b_errors += 1,
            },
            _ => {}
        }
        // Check for bus failover
        if self.bus_a_timeouts + self.bus_a_errors >= self.error_threshold
            && self.active_bus == Bus::A
        {
            self.active_bus = Bus::B;
        }
        if self.bus_b_timeouts + self.bus_b_errors >= self.error_threshold
            && self.active_bus == Bus::B
        {
            self.active_bus = Bus::A;
        }
    }

    /// Returns the currently active bus.
    pub fn active_bus(&self) -> Bus {
        self.active_bus
    }

    /// Returns the number of pending messages.
    pub fn pending_count(&self) -> usize {
        self.queue.len()
    }

    /// Returns timeout count for a bus.
    pub fn timeouts(&self, bus: Bus) -> u64 {
        match bus {
            Bus::A => self.bus_a_timeouts,
            Bus::B => self.bus_b_timeouts,
        }
    }

    /// Returns error count for a bus.
    pub fn errors(&self, bus: Bus) -> u64 {
        match bus {
            Bus::A => self.bus_a_errors,
            Bus::B => self.bus_b_errors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mil1553::word::RT_BROADCAST;

    #[test]
    fn test_bc_to_rt_message() {
        let data = [0x1111, 0x2222, 0x3333];
        let msg = BcMessage::bc_to_rt(7, 2, &data, Bus::A).unwrap();
        assert_eq!(msg.msg_type, MessageType::BcToRt);
        assert_eq!(msg.command.rt_address, 7);
        assert!(!msg.command.transmit);
        assert_eq!(msg.command.subaddress, 2);
        assert_eq!(msg.command.word_count, 3);
        assert_eq!(msg.tx_data.len(), 3);
    }

    #[test]
    fn test_rt_to_bc_message() {
        let msg = BcMessage::rt_to_bc(3, 5, 4, Bus::A).unwrap();
        assert_eq!(msg.msg_type, MessageType::RtToBc);
        assert!(msg.command.transmit);
        assert_eq!(msg.command.word_count, 4);
    }

    #[test]
    fn test_rt_to_rt_message() {
        let msg = BcMessage::rt_to_rt(7, 2, 3, 1, 4, Bus::A).unwrap();
        assert_eq!(msg.msg_type, MessageType::RtToRt);
        assert!(!msg.command.transmit); // receive command
        assert!(msg.command2.unwrap().transmit); // transmit command
    }

    #[test]
    fn test_bc_to_rt_empty_data() {
        assert_eq!(
            BcMessage::bc_to_rt(1, 1, &[], Bus::A).err(),
            Some(BusError::OutOfRange)
        );
    }

    #[test]
    fn test_bc_to_rt_too_much_data() {
        let data = [0u16; 33];
        assert_eq!(
            BcMessage::bc_to_rt(1, 1, &data, Bus::A).err(),
            Some(BusError::OutOfRange)
        );
    }

    #[test]
    fn test_bc_to_rt_32_words() {
        let data = [0u16; 32];
        let msg = BcMessage::bc_to_rt(1, 1, &data, Bus::A).unwrap();
        assert_eq!(msg.command.word_count, 0); // 0 encodes 32
    }

    #[test]
    fn test_encode_commands() {
        let msg = BcMessage::bc_to_rt(5, 3, &[0x1234], Bus::A).unwrap();
        let cmds = msg.encode_commands();
        assert_eq!(cmds.len(), 1);
        let decoded = CommandWord::decode(cmds[0]).unwrap();
        assert_eq!(decoded.rt_address, 5);
    }

    #[test]
    fn test_encode_data() {
        let data = [0xAAAA, 0xBBBB];
        let msg = BcMessage::bc_to_rt(1, 1, &data, Bus::A).unwrap();
        let encoded = msg.encode_data();
        assert_eq!(encoded.len(), 2);
        assert_eq!(DataWord::decode(encoded[0]).unwrap().data, 0xAAAA);
        assert_eq!(DataWord::decode(encoded[1]).unwrap().data, 0xBBBB);
    }

    #[test]
    fn test_sequencer_schedule_and_dequeue() {
        let mut seq = BcSequencer::new(10);
        let msg1 = BcMessage::bc_to_rt(1, 1, &[0x1111], Bus::A).unwrap();
        let msg2 = BcMessage::rt_to_bc(2, 1, 4, Bus::A).unwrap();
        seq.schedule(msg1);
        seq.schedule(msg2);
        assert_eq!(seq.pending_count(), 2);

        let first = seq.next_message().unwrap();
        assert_eq!(first.msg_type, MessageType::BcToRt);
        let second = seq.next_message().unwrap();
        assert_eq!(second.msg_type, MessageType::RtToBc);
        assert!(seq.next_message().is_none());
    }

    #[test]
    fn test_sequencer_bus_failover() {
        let mut seq = BcSequencer::new(3);
        assert_eq!(seq.active_bus(), Bus::A);

        for _ in 0..3 {
            seq.record_result(Bus::A, &MessageResult::Timeout);
        }
        assert_eq!(seq.active_bus(), Bus::B);
    }

    #[test]
    fn test_sequencer_counters() {
        let mut seq = BcSequencer::new(100);
        seq.record_result(Bus::A, &MessageResult::Timeout);
        seq.record_result(Bus::A, &MessageResult::MessageError);
        seq.record_result(Bus::B, &MessageResult::Timeout);
        assert_eq!(seq.timeouts(Bus::A), 1);
        assert_eq!(seq.errors(Bus::A), 1);
        assert_eq!(seq.timeouts(Bus::B), 1);
    }

    #[test]
    fn test_sequencer_success_no_counter_increment() {
        let mut seq = BcSequencer::new(100);
        let sw = StatusWord::new(5).unwrap();
        seq.record_result(
            Bus::A,
            &MessageResult::Success {
                status: sw,
                data: Vec::new(),
            },
        );
        assert_eq!(seq.timeouts(Bus::A), 0);
        assert_eq!(seq.errors(Bus::A), 0);
    }

    #[test]
    fn test_broadcast_bc_to_rt() {
        let msg = BcMessage::bc_to_rt(RT_BROADCAST, 3, &[0x1234], Bus::A).unwrap();
        assert_eq!(msg.command.rt_address, 31);
    }
}
