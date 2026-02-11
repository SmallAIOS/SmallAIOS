// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SpaceWire hardware interface abstraction.
//!
//! Defines a trait for LVDS PHY transceivers and FPGA SpaceWire codec IP
//! cores. The physical layer uses LVDS differential signaling with
//! data-strobe (DS) encoding per ECSS-E-ST-50-12C.
//!
//! # References
//!
//! - ECSS-E-ST-50-12C: SpaceWire standard
//! - STAR-Dundee SpaceWire IP cores
//! - AMBA AXI SpaceWire CODEC (e.g., Cobham GR740 SpW)

use crate::BusError;

/// SpaceWire link speed configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpwLinkSpeed {
    /// Transmit clock divisor (link speed = base_clock / divisor).
    /// Divisor of 1 gives the maximum speed.
    pub divisor: u16,
}

impl SpwLinkSpeed {
    /// 10 Mbps (typical SpaceWire default).
    pub fn default_10mbps() -> Self {
        Self { divisor: 10 }
    }

    /// 100 Mbps.
    pub fn fast_100mbps() -> Self {
        Self { divisor: 1 }
    }

    /// 200 Mbps.
    pub fn fast_200mbps() -> Self {
        Self { divisor: 1 }
    }
}

/// SpaceWire link state (per ECSS-E-ST-50-12C Section 8.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpwLinkState {
    /// ErrorReset: link detected error, resetting.
    ErrorReset,
    /// ErrorWait: waiting for a timeout before restart.
    ErrorWait,
    /// Ready: link ready to start.
    Ready,
    /// Started: local end has started, waiting for remote.
    Started,
    /// Connecting: both ends exchanging NULLs.
    Connecting,
    /// Run: link is operational, data can flow.
    Run,
}

/// Hardware-independent SpaceWire link PHY trait.
///
/// Implementations target specific hardware:
/// - LVDS PHY transceivers (e.g., Aeroflex UT200SpW)
/// - FPGA SpaceWire codec IP (STAR-Dundee, Cobham GR740 SpW)
/// - AXI-mapped SpaceWire controllers on SoC FPGAs
pub trait SpwLinkPhy {
    /// Initialize the SpaceWire link hardware and PHY.
    fn init(&mut self) -> Result<(), BusError>;

    /// Set the transmit link speed.
    fn set_speed(&mut self, speed: SpwLinkSpeed) -> Result<(), BusError>;

    /// Start the link (begin the connection sequence).
    fn link_start(&mut self) -> Result<(), BusError>;

    /// Disable the link (enter ErrorReset).
    fn link_disable(&mut self) -> Result<(), BusError>;

    /// Get the current link state.
    fn link_state(&self) -> SpwLinkState;

    /// Check if the link is operational (in Run state).
    fn is_running(&self) -> bool {
        self.link_state() == SpwLinkState::Run
    }

    /// Transmit a SpaceWire packet (destination address + cargo bytes).
    ///
    /// The hardware handles character encoding and EOP/EEP markers.
    fn transmit_packet(&mut self, dest: u8, data: &[u8]) -> Result<(), BusError>;

    /// Receive a SpaceWire packet.
    ///
    /// Returns `Ok(Some((dest, data)))` if a packet is available.
    /// The `dest` byte is the destination address, and `data` is the cargo.
    fn receive_packet(&mut self) -> Result<Option<(u8, alloc::vec::Vec<u8>)>, BusError>;

    /// Transmit a time-code (6-bit counter value).
    fn transmit_timecode(&mut self, counter: u8) -> Result<(), BusError>;

    /// Receive the latest time-code value.
    fn receive_timecode(&self) -> Option<u8>;

    /// Get the number of link errors since last reset.
    fn error_count(&self) -> u32;

    /// Reset the link hardware to its initial state.
    fn reset(&mut self) -> Result<(), BusError>;
}

/// Mock SpaceWire link PHY for testing.
pub struct MockSpwLinkPhy {
    state: SpwLinkState,
    speed: SpwLinkSpeed,
    rx_queue: alloc::vec::Vec<(u8, alloc::vec::Vec<u8>)>,
    last_timecode: Option<u8>,
    error_count: u32,
    initialized: bool,
}

impl MockSpwLinkPhy {
    pub fn new() -> Self {
        Self {
            state: SpwLinkState::ErrorReset,
            speed: SpwLinkSpeed::default_10mbps(),
            rx_queue: alloc::vec::Vec::new(),
            last_timecode: None,
            error_count: 0,
            initialized: false,
        }
    }

    /// Inject a packet into the RX queue.
    pub fn inject_rx(&mut self, dest: u8, data: &[u8]) {
        self.rx_queue.push((dest, alloc::vec::Vec::from(data)));
    }

    /// Set the simulated link state.
    pub fn set_state(&mut self, state: SpwLinkState) {
        self.state = state;
    }
}

impl Default for MockSpwLinkPhy {
    fn default() -> Self {
        Self::new()
    }
}

impl SpwLinkPhy for MockSpwLinkPhy {
    fn init(&mut self) -> Result<(), BusError> {
        self.initialized = true;
        self.state = SpwLinkState::Ready;
        Ok(())
    }

    fn set_speed(&mut self, speed: SpwLinkSpeed) -> Result<(), BusError> {
        self.speed = speed;
        Ok(())
    }

    fn link_start(&mut self) -> Result<(), BusError> {
        if !self.initialized {
            return Err(BusError::NotSupported);
        }
        self.state = SpwLinkState::Run;
        Ok(())
    }

    fn link_disable(&mut self) -> Result<(), BusError> {
        self.state = SpwLinkState::ErrorReset;
        Ok(())
    }

    fn link_state(&self) -> SpwLinkState {
        self.state
    }

    fn transmit_packet(&mut self, _dest: u8, _data: &[u8]) -> Result<(), BusError> {
        if self.state != SpwLinkState::Run {
            return Err(BusError::NotSupported);
        }
        Ok(())
    }

    fn receive_packet(&mut self) -> Result<Option<(u8, alloc::vec::Vec<u8>)>, BusError> {
        if self.rx_queue.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.rx_queue.remove(0)))
        }
    }

    fn transmit_timecode(&mut self, counter: u8) -> Result<(), BusError> {
        if self.state != SpwLinkState::Run {
            return Err(BusError::NotSupported);
        }
        self.last_timecode = Some(counter & 0x3F);
        Ok(())
    }

    fn receive_timecode(&self) -> Option<u8> {
        self.last_timecode
    }

    fn error_count(&self) -> u32 {
        self.error_count
    }

    fn reset(&mut self) -> Result<(), BusError> {
        self.state = SpwLinkState::ErrorReset;
        self.rx_queue.clear();
        self.last_timecode = None;
        self.error_count = 0;
        self.initialized = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_init() {
        let mut hw = MockSpwLinkPhy::new();
        assert_eq!(hw.link_state(), SpwLinkState::ErrorReset);
        hw.init().unwrap();
        assert_eq!(hw.link_state(), SpwLinkState::Ready);
    }

    #[test]
    fn test_mock_link_start() {
        let mut hw = MockSpwLinkPhy::new();
        hw.init().unwrap();
        hw.link_start().unwrap();
        assert_eq!(hw.link_state(), SpwLinkState::Run);
        assert!(hw.is_running());
    }

    #[test]
    fn test_mock_link_start_uninit() {
        let mut hw = MockSpwLinkPhy::new();
        assert_eq!(hw.link_start().err(), Some(BusError::NotSupported));
    }

    #[test]
    fn test_mock_link_disable() {
        let mut hw = MockSpwLinkPhy::new();
        hw.init().unwrap();
        hw.link_start().unwrap();
        hw.link_disable().unwrap();
        assert_eq!(hw.link_state(), SpwLinkState::ErrorReset);
    }

    #[test]
    fn test_mock_set_speed() {
        let mut hw = MockSpwLinkPhy::new();
        hw.set_speed(SpwLinkSpeed::fast_100mbps()).unwrap();
        assert_eq!(hw.speed.divisor, 1);
    }

    #[test]
    fn test_mock_transmit_not_running() {
        let mut hw = MockSpwLinkPhy::new();
        hw.init().unwrap();
        // Still in Ready state, not Run
        hw.set_state(SpwLinkState::Ready);
        assert_eq!(hw.transmit_packet(1, &[0x01]).err(), Some(BusError::NotSupported));
    }

    #[test]
    fn test_mock_transmit_running() {
        let mut hw = MockSpwLinkPhy::new();
        hw.init().unwrap();
        hw.link_start().unwrap();
        hw.transmit_packet(5, &[0x01, 0x02, 0x03]).unwrap();
    }

    #[test]
    fn test_mock_receive() {
        let mut hw = MockSpwLinkPhy::new();
        hw.inject_rx(10, &[0xAA, 0xBB]);
        let (dest, data) = hw.receive_packet().unwrap().unwrap();
        assert_eq!(dest, 10);
        assert_eq!(data, &[0xAA, 0xBB]);
        assert!(hw.receive_packet().unwrap().is_none());
    }

    #[test]
    fn test_mock_timecode() {
        let mut hw = MockSpwLinkPhy::new();
        hw.init().unwrap();
        hw.link_start().unwrap();
        assert!(hw.receive_timecode().is_none());
        hw.transmit_timecode(42).unwrap();
        assert_eq!(hw.receive_timecode(), Some(42));
    }

    #[test]
    fn test_mock_timecode_mask() {
        let mut hw = MockSpwLinkPhy::new();
        hw.init().unwrap();
        hw.link_start().unwrap();
        // Time-code is 6-bit, so 0xFF & 0x3F = 0x3F
        hw.transmit_timecode(0xFF).unwrap();
        assert_eq!(hw.receive_timecode(), Some(0x3F));
    }

    #[test]
    fn test_mock_error_count() {
        let hw = MockSpwLinkPhy::new();
        assert_eq!(hw.error_count(), 0);
    }

    #[test]
    fn test_mock_reset() {
        let mut hw = MockSpwLinkPhy::new();
        hw.init().unwrap();
        hw.link_start().unwrap();
        hw.inject_rx(1, &[0x01]);
        hw.reset().unwrap();
        assert_eq!(hw.link_state(), SpwLinkState::ErrorReset);
        assert!(hw.receive_packet().unwrap().is_none());
        assert!(!hw.initialized);
    }

    #[test]
    fn test_link_speed_presets() {
        let s10 = SpwLinkSpeed::default_10mbps();
        assert_eq!(s10.divisor, 10);
        let s100 = SpwLinkSpeed::fast_100mbps();
        assert_eq!(s100.divisor, 1);
    }
}
