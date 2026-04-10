// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CAN controller driver abstraction trait.
//!
//! Defines a hardware-independent interface for CAN bus controllers,
//! supporting initialization, frame transmission/reception, bitrate
//! configuration, and error state monitoring. Concrete implementations
//! target specific hardware (e.g., MCP2515 SPI, AXI CAN FPGA IP).

use crate::can::frame::CanFrame;
use crate::can::state::BusState;
use crate::BusError;

/// CAN controller operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanMode {
    /// Normal operation — transmit and receive frames.
    Normal,
    /// Listen-only — receive frames but do not transmit or acknowledge.
    ListenOnly,
    /// Loopback — transmitted frames are received back (for self-test).
    Loopback,
    /// Configuration — controller halted, registers accessible.
    Configuration,
    /// Sleep — low-power mode, wake on bus activity.
    Sleep,
}

/// CAN bus bitrate configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanBitTiming {
    /// Baud rate prescaler (divides the peripheral clock).
    pub prescaler: u16,
    /// Time Segment 1 (propagation + phase1), in time quanta.
    pub tseg1: u8,
    /// Time Segment 2 (phase2), in time quanta.
    pub tseg2: u8,
    /// Synchronization Jump Width, in time quanta.
    pub sjw: u8,
}

impl CanBitTiming {
    /// Standard 500 kbps timing (assuming 8 MHz peripheral clock).
    pub fn standard_500kbps() -> Self {
        Self {
            prescaler: 1,
            tseg1: 11,
            tseg2: 4,
            sjw: 1,
        }
    }

    /// Standard 250 kbps timing (assuming 8 MHz peripheral clock).
    pub fn standard_250kbps() -> Self {
        Self {
            prescaler: 2,
            tseg1: 11,
            tseg2: 4,
            sjw: 1,
        }
    }

    /// Standard 125 kbps timing (assuming 8 MHz peripheral clock).
    pub fn standard_125kbps() -> Self {
        Self {
            prescaler: 4,
            tseg1: 11,
            tseg2: 4,
            sjw: 1,
        }
    }

    /// Standard 1 Mbps timing (assuming 8 MHz peripheral clock).
    pub fn standard_1mbps() -> Self {
        Self {
            prescaler: 1,
            tseg1: 5,
            tseg2: 2,
            sjw: 1,
        }
    }
}

/// CAN controller statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CanStats {
    /// Number of frames transmitted successfully.
    pub tx_frames: u64,
    /// Number of frames received successfully.
    pub rx_frames: u64,
    /// Number of transmit errors.
    pub tx_errors: u64,
    /// Number of receive errors.
    pub rx_errors: u64,
    /// Number of bus-off events.
    pub bus_off_count: u32,
    /// Number of error-passive transitions.
    pub error_passive_count: u32,
    /// Current transmit error counter (TEC).
    pub tec: u8,
    /// Current receive error counter (REC).
    pub rec: u8,
}

/// Sink that receives CAN frames asynchronously from a controller.
///
/// Implementors process incoming frames (e.g., the inference adapter
/// batches them into tensors). Multiple sinks can be attached to a
/// single controller and will receive frames in registration order.
///
/// # Integration pattern
///
/// The existing [`CanController`] trait is intentionally not extended
/// to register sinks directly — doing so would require changes to the
/// MCP2515 and AXI CAN drivers. Instead, whoever owns the controller
/// (for example, a dataflow runner) is responsible for calling
/// [`CanController::receive`] in a loop and forwarding each received
/// frame to all attached sinks via [`CanFrameSink::on_frame`]. This
/// "caller pattern" keeps driver implementations unchanged while still
/// allowing higher-level components (e.g., the inference adapter) to
/// observe CAN traffic.
pub trait CanFrameSink: Send {
    /// Called by the caller for each received frame.
    fn on_frame(&mut self, frame: &CanFrame);
}

/// Hardware-independent CAN controller driver trait.
///
/// Implementations provide access to a specific CAN controller (SPI,
/// memory-mapped, or FPGA soft-IP). The trait covers initialization,
/// mode switching, frame TX/RX, and error state monitoring.
pub trait CanController {
    /// Initialize the controller hardware and enter configuration mode.
    fn init(&mut self) -> Result<(), BusError>;

    /// Set the operating mode.
    fn set_mode(&mut self, mode: CanMode) -> Result<(), BusError>;

    /// Get the current operating mode.
    fn mode(&self) -> CanMode;

    /// Configure the bus bitrate timing parameters.
    fn set_bit_timing(&mut self, timing: &CanBitTiming) -> Result<(), BusError>;

    /// Transmit a CAN frame.
    ///
    /// Returns `Ok(())` on successful queuing for transmission.
    /// Returns `Err(BusError::BusOff)` if the controller is in bus-off state.
    fn transmit(&mut self, frame: &CanFrame) -> Result<(), BusError>;

    /// Receive a CAN frame from the hardware RX FIFO.
    ///
    /// Returns `Ok(Some(frame))` if a frame is available, `Ok(None)` if
    /// the RX FIFO is empty.
    fn receive(&mut self) -> Result<Option<CanFrame>, BusError>;

    /// Get the current bus state (error active, passive, bus-off).
    fn bus_state(&self) -> BusState;

    /// Get the current controller statistics.
    fn stats(&self) -> CanStats;

    /// Reset the controller to its initial state.
    fn reset(&mut self) -> Result<(), BusError>;
}

/// Mock CAN controller for testing.
///
/// Implements [`CanController`] using in-memory queues for TX and RX
/// frames, allowing software-only testing without hardware.
#[derive(Debug)]
pub struct MockCanController {
    mode: CanMode,
    bus_state: BusState,
    stats: CanStats,
    bit_timing: Option<CanBitTiming>,
    tx_queue: alloc::vec::Vec<CanFrame>,
    rx_queue: alloc::vec::Vec<CanFrame>,
    initialized: bool,
}

impl Default for MockCanController {
    fn default() -> Self {
        Self::new()
    }
}

impl MockCanController {
    /// Create a new mock CAN controller.
    pub fn new() -> Self {
        Self {
            mode: CanMode::Configuration,
            bus_state: BusState::ErrorActive,
            stats: CanStats::default(),
            bit_timing: None,
            tx_queue: alloc::vec::Vec::new(),
            rx_queue: alloc::vec::Vec::new(),
            initialized: false,
        }
    }

    /// Inject a frame into the RX queue (simulating reception from bus).
    pub fn inject_rx(&mut self, frame: CanFrame) {
        self.rx_queue.push(frame);
    }

    /// Read frames from the TX queue (to verify what was transmitted).
    pub fn drain_tx(&mut self) -> alloc::vec::Vec<CanFrame> {
        core::mem::take(&mut self.tx_queue)
    }

    /// Set the simulated bus state.
    pub fn set_bus_state(&mut self, state: BusState) {
        self.bus_state = state;
    }
}

impl CanController for MockCanController {
    fn init(&mut self) -> Result<(), BusError> {
        self.initialized = true;
        self.mode = CanMode::Configuration;
        Ok(())
    }

    fn set_mode(&mut self, mode: CanMode) -> Result<(), BusError> {
        if !self.initialized {
            return Err(BusError::NotSupported);
        }
        self.mode = mode;
        Ok(())
    }

    fn mode(&self) -> CanMode {
        self.mode
    }

    fn set_bit_timing(&mut self, timing: &CanBitTiming) -> Result<(), BusError> {
        if self.mode != CanMode::Configuration {
            return Err(BusError::NotSupported);
        }
        self.bit_timing = Some(*timing);
        Ok(())
    }

    fn transmit(&mut self, frame: &CanFrame) -> Result<(), BusError> {
        if self.bus_state == BusState::BusOff {
            return Err(BusError::BusOff);
        }
        if self.mode != CanMode::Normal && self.mode != CanMode::Loopback {
            return Err(BusError::NotSupported);
        }
        self.tx_queue.push(frame.clone());
        self.stats.tx_frames += 1;

        // In loopback mode, also inject into RX
        if self.mode == CanMode::Loopback {
            self.rx_queue.push(frame.clone());
        }

        Ok(())
    }

    fn receive(&mut self) -> Result<Option<CanFrame>, BusError> {
        if self.mode == CanMode::Configuration || self.mode == CanMode::Sleep {
            return Err(BusError::NotSupported);
        }
        if self.rx_queue.is_empty() {
            Ok(None)
        } else {
            self.stats.rx_frames += 1;
            Ok(Some(self.rx_queue.remove(0)))
        }
    }

    fn bus_state(&self) -> BusState {
        self.bus_state
    }

    fn stats(&self) -> CanStats {
        self.stats
    }

    fn reset(&mut self) -> Result<(), BusError> {
        self.mode = CanMode::Configuration;
        self.bus_state = BusState::ErrorActive;
        self.stats = CanStats::default();
        self.bit_timing = None;
        self.tx_queue.clear();
        self.rx_queue.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mock_init() {
        let mut ctrl = MockCanController::new();
        assert_eq!(ctrl.mode(), CanMode::Configuration);
        ctrl.init().unwrap();
        assert_eq!(ctrl.mode(), CanMode::Configuration);
    }

    #[test]
    fn test_mock_set_mode() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_mode(CanMode::Normal).unwrap();
        assert_eq!(ctrl.mode(), CanMode::Normal);
    }

    #[test]
    fn test_mock_set_mode_uninit() {
        let mut ctrl = MockCanController::new();
        assert_eq!(
            ctrl.set_mode(CanMode::Normal).err(),
            Some(BusError::NotSupported)
        );
    }

    #[test]
    fn test_mock_bit_timing() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        let timing = CanBitTiming::standard_500kbps();
        ctrl.set_bit_timing(&timing).unwrap();
    }

    #[test]
    fn test_mock_bit_timing_wrong_mode() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_mode(CanMode::Normal).unwrap();
        let timing = CanBitTiming::standard_500kbps();
        assert_eq!(
            ctrl.set_bit_timing(&timing).err(),
            Some(BusError::NotSupported)
        );
    }

    #[test]
    fn test_mock_transmit() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_mode(CanMode::Normal).unwrap();
        let frame = CanFrame::new_standard(0x123, &[0x01, 0x02]).unwrap();
        ctrl.transmit(&frame).unwrap();
        let sent = ctrl.drain_tx();
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].id, 0x123);
        assert_eq!(ctrl.stats().tx_frames, 1);
    }

    #[test]
    fn test_mock_transmit_bus_off() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_mode(CanMode::Normal).unwrap();
        ctrl.set_bus_state(BusState::BusOff);
        let frame = CanFrame::new_standard(0x100, &[0x01]).unwrap();
        assert_eq!(ctrl.transmit(&frame).err(), Some(BusError::BusOff));
    }

    #[test]
    fn test_mock_transmit_wrong_mode() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        // Still in configuration mode
        let frame = CanFrame::new_standard(0x100, &[0x01]).unwrap();
        assert_eq!(ctrl.transmit(&frame).err(), Some(BusError::NotSupported));
    }

    #[test]
    fn test_mock_receive() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_mode(CanMode::Normal).unwrap();
        assert!(ctrl.receive().unwrap().is_none());

        let frame = CanFrame::new_standard(0x200, &[0xAA]).unwrap();
        ctrl.inject_rx(frame.clone());
        let received = ctrl.receive().unwrap().unwrap();
        assert_eq!(received.id, 0x200);
        assert_eq!(ctrl.stats().rx_frames, 1);
    }

    #[test]
    fn test_mock_receive_wrong_mode() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        // Still in configuration mode
        assert_eq!(ctrl.receive().err(), Some(BusError::NotSupported));
    }

    #[test]
    fn test_mock_loopback() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_mode(CanMode::Loopback).unwrap();
        let frame = CanFrame::new_standard(0x100, &[0x01, 0x02]).unwrap();
        ctrl.transmit(&frame).unwrap();
        // In loopback, TX frames are also received
        let received = ctrl.receive().unwrap().unwrap();
        assert_eq!(received.id, 0x100);
        assert_eq!(received.data, &[0x01, 0x02]);
    }

    #[test]
    fn test_mock_reset() {
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_mode(CanMode::Normal).unwrap();
        let frame = CanFrame::new_standard(0x100, &[0x01]).unwrap();
        ctrl.transmit(&frame).unwrap();
        ctrl.inject_rx(frame);

        ctrl.reset().unwrap();
        assert_eq!(ctrl.mode(), CanMode::Configuration);
        assert_eq!(ctrl.bus_state(), BusState::ErrorActive);
        assert_eq!(ctrl.stats().tx_frames, 0);
        assert!(ctrl.drain_tx().is_empty());
    }

    #[test]
    fn test_mock_bus_state() {
        let mut ctrl = MockCanController::new();
        assert_eq!(ctrl.bus_state(), BusState::ErrorActive);
        ctrl.set_bus_state(BusState::ErrorPassive);
        assert_eq!(ctrl.bus_state(), BusState::ErrorPassive);
    }

    #[test]
    fn test_bit_timing_presets() {
        let t500 = CanBitTiming::standard_500kbps();
        assert_eq!(t500.prescaler, 1);

        let t250 = CanBitTiming::standard_250kbps();
        assert_eq!(t250.prescaler, 2);

        let t125 = CanBitTiming::standard_125kbps();
        assert_eq!(t125.prescaler, 4);

        let t1m = CanBitTiming::standard_1mbps();
        assert_eq!(t1m.prescaler, 1);
        assert!(t1m.tseg1 < t500.tseg1); // Faster = shorter segments
    }

    /// Test sink that counts how many frames it has observed.
    struct CountingSink {
        count: usize,
        last_id: u32,
    }

    impl CanFrameSink for CountingSink {
        fn on_frame(&mut self, frame: &CanFrame) {
            self.count += 1;
            self.last_id = frame.id;
        }
    }

    #[test]
    fn test_can_frame_sink_counts_frames() {
        let mut sink = CountingSink {
            count: 0,
            last_id: 0,
        };
        let f1 = CanFrame::new_standard(0x100, &[0x01]).unwrap();
        let f2 = CanFrame::new_standard(0x200, &[0x02]).unwrap();
        let f3 = CanFrame::new_extended(0x1ABCD, &[0x03]).unwrap();

        sink.on_frame(&f1);
        sink.on_frame(&f2);
        sink.on_frame(&f3);

        assert_eq!(sink.count, 3);
        assert_eq!(sink.last_id, 0x1ABCD);
    }

    #[test]
    fn test_can_frame_sink_caller_pattern() {
        // Exercise the "caller drains controller, forwards to sink" pattern
        // described in the CanFrameSink trait docs.
        let mut ctrl = MockCanController::new();
        ctrl.init().unwrap();
        ctrl.set_mode(CanMode::Normal).unwrap();

        ctrl.inject_rx(CanFrame::new_standard(0x111, &[0xAA]).unwrap());
        ctrl.inject_rx(CanFrame::new_standard(0x222, &[0xBB]).unwrap());

        let mut sink = CountingSink {
            count: 0,
            last_id: 0,
        };

        while let Some(frame) = ctrl.receive().unwrap() {
            sink.on_frame(&frame);
        }

        assert_eq!(sink.count, 2);
        assert_eq!(sink.last_id, 0x222);
    }

    #[test]
    fn test_stats_default() {
        let stats = CanStats::default();
        assert_eq!(stats.tx_frames, 0);
        assert_eq!(stats.rx_frames, 0);
        assert_eq!(stats.tx_errors, 0);
        assert_eq!(stats.rx_errors, 0);
        assert_eq!(stats.bus_off_count, 0);
        assert_eq!(stats.tec, 0);
        assert_eq!(stats.rec, 0);
    }
}
