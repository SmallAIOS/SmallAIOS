// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! MIL-STD-1553B hardware interface abstraction.
//!
//! Defines a trait for dedicated 1553 transceiver ICs (e.g., DDC BU-61580,
//! Holt HI-6130) and FPGA soft-IP controllers. These devices handle the
//! physical Manchester-encoded signaling and provide a register-level
//! interface for command/data word exchange.

use crate::BusError;

/// MIL-STD-1553 bus selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Bus1553 {
    /// Primary bus (Bus A).
    BusA,
    /// Redundant bus (Bus B).
    BusB,
}

/// MIL-STD-1553 terminal role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalRole {
    /// Bus Controller (BC): initiates all transactions.
    BusController,
    /// Remote Terminal (RT): responds to BC commands.
    RemoteTerminal,
    /// Bus Monitor (BM): passive listener.
    BusMonitor,
}

/// A 1553 word (16-bit data + parity, encoded in a 20-bit Manchester waveform).
/// At the register level we see only the 16-bit data portion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Word1553 {
    pub data: u16,
}

/// Hardware-independent MIL-STD-1553 transceiver trait.
///
/// Implementations target specific hardware:
/// - DDC BU-61580 / BU-67118 (SPI or parallel interface)
/// - Holt HI-6130 / HI-6138 (SPI)
/// - FPGA soft-IP (AXI-mapped MIL-STD-1553 controllers)
pub trait Mil1553Transceiver {
    /// Initialize the transceiver hardware.
    fn init(&mut self) -> Result<(), BusError>;

    /// Set the terminal role (BC, RT, or BM).
    fn set_role(&mut self, role: TerminalRole) -> Result<(), BusError>;

    /// Set the RT address (0-30, address 31 is broadcast).
    fn set_rt_address(&mut self, address: u8) -> Result<(), BusError>;

    /// Transmit a command word on the specified bus.
    fn transmit_command(&mut self, bus: Bus1553, word: Word1553) -> Result<(), BusError>;

    /// Transmit a data word on the specified bus.
    fn transmit_data(&mut self, bus: Bus1553, word: Word1553) -> Result<(), BusError>;

    /// Receive the next word from the specified bus.
    ///
    /// Returns `Ok(Some(word))` if a word is available, `Ok(None)` if empty.
    fn receive(&mut self, bus: Bus1553) -> Result<Option<Word1553>, BusError>;

    /// Check if the transceiver detected a message error on the last reception.
    fn message_error(&self) -> bool;

    /// Get the busy status of the transceiver.
    fn is_busy(&self) -> bool;

    /// Select the active bus for transmission (Bus A or Bus B).
    fn select_bus(&mut self, bus: Bus1553) -> Result<(), BusError>;

    /// Reset the transceiver to its initial state.
    fn reset(&mut self) -> Result<(), BusError>;
}

/// Mock MIL-STD-1553 transceiver for testing.
pub struct MockMil1553Transceiver {
    role: TerminalRole,
    rt_address: u8,
    active_bus: Bus1553,
    rx_queue_a: alloc::vec::Vec<Word1553>,
    rx_queue_b: alloc::vec::Vec<Word1553>,
    initialized: bool,
    msg_error: bool,
    busy: bool,
}

impl MockMil1553Transceiver {
    pub fn new() -> Self {
        Self {
            role: TerminalRole::BusController,
            rt_address: 0,
            active_bus: Bus1553::BusA,
            rx_queue_a: alloc::vec::Vec::new(),
            rx_queue_b: alloc::vec::Vec::new(),
            initialized: false,
            msg_error: false,
            busy: false,
        }
    }

    /// Inject a word into the RX queue for a specific bus.
    pub fn inject_rx(&mut self, bus: Bus1553, word: Word1553) {
        match bus {
            Bus1553::BusA => self.rx_queue_a.push(word),
            Bus1553::BusB => self.rx_queue_b.push(word),
        }
    }
}

impl Default for MockMil1553Transceiver {
    fn default() -> Self {
        Self::new()
    }
}

impl Mil1553Transceiver for MockMil1553Transceiver {
    fn init(&mut self) -> Result<(), BusError> {
        self.initialized = true;
        Ok(())
    }

    fn set_role(&mut self, role: TerminalRole) -> Result<(), BusError> {
        self.role = role;
        Ok(())
    }

    fn set_rt_address(&mut self, address: u8) -> Result<(), BusError> {
        if address > 30 {
            return Err(BusError::OutOfRange);
        }
        self.rt_address = address;
        Ok(())
    }

    fn transmit_command(&mut self, _bus: Bus1553, _word: Word1553) -> Result<(), BusError> {
        if !self.initialized {
            return Err(BusError::NotSupported);
        }
        Ok(())
    }

    fn transmit_data(&mut self, _bus: Bus1553, _word: Word1553) -> Result<(), BusError> {
        if !self.initialized {
            return Err(BusError::NotSupported);
        }
        Ok(())
    }

    fn receive(&mut self, bus: Bus1553) -> Result<Option<Word1553>, BusError> {
        let q = match bus {
            Bus1553::BusA => &mut self.rx_queue_a,
            Bus1553::BusB => &mut self.rx_queue_b,
        };
        if q.is_empty() {
            Ok(None)
        } else {
            Ok(Some(q.remove(0)))
        }
    }

    fn message_error(&self) -> bool {
        self.msg_error
    }

    fn is_busy(&self) -> bool {
        self.busy
    }

    fn select_bus(&mut self, bus: Bus1553) -> Result<(), BusError> {
        self.active_bus = bus;
        Ok(())
    }

    fn reset(&mut self) -> Result<(), BusError> {
        self.initialized = false;
        self.rx_queue_a.clear();
        self.rx_queue_b.clear();
        self.msg_error = false;
        self.busy = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_init() {
        let mut hw = MockMil1553Transceiver::new();
        hw.init().unwrap();
        assert!(hw.initialized);
    }

    #[test]
    fn test_mock_set_role() {
        let mut hw = MockMil1553Transceiver::new();
        hw.set_role(TerminalRole::RemoteTerminal).unwrap();
        assert_eq!(hw.role, TerminalRole::RemoteTerminal);
    }

    #[test]
    fn test_mock_rt_address() {
        let mut hw = MockMil1553Transceiver::new();
        hw.set_rt_address(5).unwrap();
        assert_eq!(hw.rt_address, 5);
    }

    #[test]
    fn test_mock_rt_address_invalid() {
        let mut hw = MockMil1553Transceiver::new();
        assert_eq!(hw.set_rt_address(31).err(), Some(BusError::OutOfRange));
    }

    #[test]
    fn test_mock_transmit() {
        let mut hw = MockMil1553Transceiver::new();
        hw.init().unwrap();
        let word = Word1553 { data: 0x1234 };
        hw.transmit_command(Bus1553::BusA, word).unwrap();
        hw.transmit_data(Bus1553::BusB, word).unwrap();
    }

    #[test]
    fn test_mock_transmit_uninit() {
        let mut hw = MockMil1553Transceiver::new();
        let word = Word1553 { data: 0 };
        assert_eq!(
            hw.transmit_command(Bus1553::BusA, word).err(),
            Some(BusError::NotSupported)
        );
    }

    #[test]
    fn test_mock_receive() {
        let mut hw = MockMil1553Transceiver::new();
        hw.inject_rx(Bus1553::BusA, Word1553 { data: 0xABCD });
        let word = hw.receive(Bus1553::BusA).unwrap().unwrap();
        assert_eq!(word.data, 0xABCD);
        assert!(hw.receive(Bus1553::BusA).unwrap().is_none());
    }

    #[test]
    fn test_mock_dual_bus() {
        let mut hw = MockMil1553Transceiver::new();
        hw.inject_rx(Bus1553::BusA, Word1553 { data: 0x1111 });
        hw.inject_rx(Bus1553::BusB, Word1553 { data: 0x2222 });
        assert_eq!(hw.receive(Bus1553::BusA).unwrap().unwrap().data, 0x1111);
        assert_eq!(hw.receive(Bus1553::BusB).unwrap().unwrap().data, 0x2222);
    }

    #[test]
    fn test_mock_select_bus() {
        let mut hw = MockMil1553Transceiver::new();
        hw.select_bus(Bus1553::BusB).unwrap();
        assert_eq!(hw.active_bus, Bus1553::BusB);
    }

    #[test]
    fn test_mock_reset() {
        let mut hw = MockMil1553Transceiver::new();
        hw.init().unwrap();
        hw.inject_rx(Bus1553::BusA, Word1553 { data: 0 });
        hw.reset().unwrap();
        assert!(!hw.initialized);
        assert!(hw.receive(Bus1553::BusA).unwrap().is_none());
    }

    #[test]
    fn test_mock_status_flags() {
        let hw = MockMil1553Transceiver::new();
        assert!(!hw.message_error());
        assert!(!hw.is_busy());
    }
}
