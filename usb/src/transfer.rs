// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! USB transfer management — control and bulk transfer helpers.

use crate::UsbError;
use smallaios_kernel::hal::{UsbHostController, UsbSetupPacket, UsbTransferResult};

// ---------------------------------------------------------------------------
// Vendor control transfer helper
// ---------------------------------------------------------------------------

/// Submit a vendor control transfer (commonly used by SDR devices).
///
/// Builds and sends a vendor-type, device-recipient control transfer.
pub fn vendor_control_out(
    hc: &mut dyn UsbHostController,
    slot: u8,
    request: u8,
    value: u16,
    index: u16,
    data: &[u8],
) -> Result<UsbTransferResult, UsbError> {
    let setup = UsbSetupPacket::new(
        crate::request_type::DIR_OUT
            | crate::request_type::TYPE_VENDOR
            | crate::request_type::RECIP_DEVICE,
        request,
        value,
        index,
        data.len() as u16,
    );
    // For OUT transfers with data, we need a mutable copy to satisfy the trait.
    // The host controller reads from this buffer; the data isn't modified.
    if data.is_empty() {
        Ok(hc.control_transfer(slot, &setup, None)?)
    } else {
        let mut buf = [0u8; 4096];
        let len = data.len().min(buf.len());
        buf[..len].copy_from_slice(&data[..len]);
        Ok(hc.control_transfer(slot, &setup, Some(&mut buf[..len]))?)
    }
}

/// Submit a vendor control IN transfer (read data from device).
pub fn vendor_control_in(
    hc: &mut dyn UsbHostController,
    slot: u8,
    request: u8,
    value: u16,
    index: u16,
    buf: &mut [u8],
) -> Result<UsbTransferResult, UsbError> {
    let setup = UsbSetupPacket::new(
        crate::request_type::DIR_IN
            | crate::request_type::TYPE_VENDOR
            | crate::request_type::RECIP_DEVICE,
        request,
        value,
        index,
        buf.len() as u16,
    );
    Ok(hc.control_transfer(slot, &setup, Some(buf))?)
}

// ---------------------------------------------------------------------------
// Bulk transfer helpers
// ---------------------------------------------------------------------------

/// Submit a bulk IN transfer (device → host).
pub fn bulk_in(
    hc: &mut dyn UsbHostController,
    slot: u8,
    endpoint: u8,
    buf: &mut [u8],
) -> Result<UsbTransferResult, UsbError> {
    // Ensure endpoint address has IN direction bit set.
    let ep_addr = endpoint | 0x80;
    Ok(hc.bulk_transfer(slot, ep_addr, buf)?)
}

/// Submit a bulk OUT transfer (host → device).
pub fn bulk_out(
    hc: &mut dyn UsbHostController,
    slot: u8,
    endpoint: u8,
    data: &mut [u8],
) -> Result<UsbTransferResult, UsbError> {
    // Ensure endpoint address has OUT direction (bit 7 clear).
    let ep_addr = endpoint & 0x7F;
    Ok(hc.bulk_transfer(slot, ep_addr, data)?)
}

// ---------------------------------------------------------------------------
// Endpoint state management
// ---------------------------------------------------------------------------

/// USB endpoint state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointState {
    /// Endpoint is idle, ready for transfers.
    Idle,
    /// Endpoint has an active (pending) transfer.
    Active,
    /// Endpoint is halted due to a STALL or error.
    Halted,
}

/// Tracks the state of an endpoint.
#[derive(Debug, Clone, Copy)]
pub struct EndpointTracker {
    /// Endpoint address.
    pub address: u8,
    /// Current state.
    pub state: EndpointState,
    /// Number of successfully completed transfers.
    pub completed_count: u32,
    /// Number of failed transfers.
    pub error_count: u32,
}

impl EndpointTracker {
    /// Create a new tracker for the given endpoint address.
    pub fn new(address: u8) -> Self {
        Self {
            address,
            state: EndpointState::Idle,
            completed_count: 0,
            error_count: 0,
        }
    }

    /// Record a successful transfer completion.
    pub fn on_complete(&mut self) {
        self.state = EndpointState::Idle;
        self.completed_count = self.completed_count.saturating_add(1);
    }

    /// Record a transfer error.
    pub fn on_error(&mut self) {
        self.error_count = self.error_count.saturating_add(1);
    }

    /// Mark endpoint as halted.
    pub fn on_halt(&mut self) {
        self.state = EndpointState::Halted;
        self.error_count = self.error_count.saturating_add(1);
    }

    /// Clear halt condition (after CLEAR_FEATURE succeeds).
    pub fn on_clear_halt(&mut self) {
        self.state = EndpointState::Idle;
    }

    /// Mark endpoint as having an active transfer.
    pub fn on_submit(&mut self) {
        self.state = EndpointState::Active;
    }
}

/// Clear the halt condition on an endpoint via CLEAR_FEATURE(ENDPOINT_HALT).
pub fn clear_endpoint_halt(
    hc: &mut dyn UsbHostController,
    slot: u8,
    endpoint: u8,
) -> Result<(), UsbError> {
    let setup = crate::enumeration::make_clear_endpoint_halt(endpoint);
    let result = hc.control_transfer(slot, &setup, None)?;
    if result.success {
        Ok(())
    } else if result.stalled {
        Err(UsbError::Stall)
    } else {
        Err(UsbError::TransferError)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_kernel::hal::{
        HalError, UsbHostController, UsbPortStatus, UsbSetupPacket, UsbSpeed, UsbTransferResult,
        UsbTransferType,
    };

    // -----------------------------------------------------------------------
    // Mock USB host controller
    // -----------------------------------------------------------------------

    struct MockHc {
        control_result: Result<UsbTransferResult, HalError>,
        bulk_result: Result<UsbTransferResult, HalError>,
    }

    impl MockHc {
        fn success() -> Self {
            Self {
                control_result: Ok(UsbTransferResult {
                    bytes_transferred: 0,
                    success: true,
                    stalled: false,
                    token: 0,
                }),
                bulk_result: Ok(UsbTransferResult {
                    bytes_transferred: 64,
                    success: true,
                    stalled: false,
                    token: 0,
                }),
            }
        }
    }

    impl UsbHostController for MockHc {
        fn init(&mut self) -> Result<(), HalError> {
            Ok(())
        }
        fn port_count(&self) -> u8 {
            1
        }
        fn port_status(&self, _port: u8) -> Result<UsbPortStatus, HalError> {
            Ok(UsbPortStatus {
                connected: true,
                enabled: true,
                reset_active: false,
                speed: UsbSpeed::High,
                port: 0,
            })
        }
        fn port_reset(&mut self, _port: u8) -> Result<(), HalError> {
            Ok(())
        }
        fn device_attach(&mut self, _port: u8) -> Result<u8, HalError> {
            Ok(1)
        }
        fn device_detach(&mut self, _slot: u8) -> Result<(), HalError> {
            Ok(())
        }
        fn control_transfer(
            &mut self,
            _slot: u8,
            _setup: &UsbSetupPacket,
            _data: Option<&mut [u8]>,
        ) -> Result<UsbTransferResult, HalError> {
            self.control_result
        }
        fn bulk_transfer(
            &mut self,
            _slot: u8,
            _endpoint: u8,
            _data: &mut [u8],
        ) -> Result<UsbTransferResult, HalError> {
            self.bulk_result
        }
        fn configure_endpoint(
            &mut self,
            _slot: u8,
            _endpoint: u8,
            _transfer_type: UsbTransferType,
            _max_packet_size: u16,
        ) -> Result<(), HalError> {
            Ok(())
        }
        fn poll_events(&mut self) -> Result<bool, HalError> {
            Ok(false)
        }
    }

    // -----------------------------------------------------------------------
    // EndpointTracker tests (existing)
    // -----------------------------------------------------------------------

    #[test]
    fn test_endpoint_tracker_lifecycle() {
        let mut tracker = EndpointTracker::new(0x81);
        assert_eq!(tracker.state, EndpointState::Idle);
        assert_eq!(tracker.completed_count, 0);

        tracker.on_submit();
        assert_eq!(tracker.state, EndpointState::Active);

        tracker.on_complete();
        assert_eq!(tracker.state, EndpointState::Idle);
        assert_eq!(tracker.completed_count, 1);
    }

    #[test]
    fn test_endpoint_tracker_halt_clear() {
        let mut tracker = EndpointTracker::new(0x02);
        tracker.on_halt();
        assert_eq!(tracker.state, EndpointState::Halted);
        assert_eq!(tracker.error_count, 1);

        tracker.on_clear_halt();
        assert_eq!(tracker.state, EndpointState::Idle);
    }

    #[test]
    fn test_endpoint_tracker_error_count() {
        let mut tracker = EndpointTracker::new(0x83);
        tracker.on_error();
        tracker.on_error();
        tracker.on_error();
        assert_eq!(tracker.error_count, 3);
    }

    #[test]
    fn test_endpoint_state_variants() {
        assert_ne!(EndpointState::Idle, EndpointState::Active);
        assert_ne!(EndpointState::Active, EndpointState::Halted);
    }

    // -----------------------------------------------------------------------
    // vendor_control_out tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vendor_control_out_empty_data() {
        let mut hc = MockHc::success();
        let result = vendor_control_out(&mut hc, 1, 0x01, 0x1234, 0x0000, &[]);
        assert!(result.is_ok());
        let r = result.unwrap();
        assert!(r.success);
    }

    #[test]
    fn test_vendor_control_out_with_data() {
        let mut hc = MockHc {
            control_result: Ok(UsbTransferResult {
                bytes_transferred: 4,
                success: true,
                stalled: false,
                token: 0,
            }),
            bulk_result: Ok(UsbTransferResult {
                bytes_transferred: 0,
                success: true,
                stalled: false,
                token: 0,
            }),
        };
        let data = [0xDE, 0xAD, 0xBE, 0xEF];
        let result = vendor_control_out(&mut hc, 1, 0x09, 0x0100, 0x0000, &data);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().bytes_transferred, 4);
    }

    #[test]
    fn test_vendor_control_out_hal_error() {
        let mut hc = MockHc {
            control_result: Err(HalError::Timeout),
            bulk_result: Ok(UsbTransferResult {
                bytes_transferred: 0,
                success: true,
                stalled: false,
                token: 0,
            }),
        };
        let result = vendor_control_out(&mut hc, 1, 0x01, 0, 0, &[]);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), UsbError::HalError(HalError::Timeout));
    }

    // -----------------------------------------------------------------------
    // vendor_control_in tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vendor_control_in() {
        let mut hc = MockHc {
            control_result: Ok(UsbTransferResult {
                bytes_transferred: 8,
                success: true,
                stalled: false,
                token: 0,
            }),
            bulk_result: Ok(UsbTransferResult {
                bytes_transferred: 0,
                success: true,
                stalled: false,
                token: 0,
            }),
        };
        let mut buf = [0u8; 64];
        let result = vendor_control_in(&mut hc, 1, 0x01, 0x0000, 0x0000, &mut buf);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().bytes_transferred, 8);
    }

    #[test]
    fn test_vendor_control_in_hal_error() {
        let mut hc = MockHc {
            control_result: Err(HalError::UsbTransferError),
            bulk_result: Ok(UsbTransferResult {
                bytes_transferred: 0,
                success: true,
                stalled: false,
                token: 0,
            }),
        };
        let mut buf = [0u8; 64];
        let result = vendor_control_in(&mut hc, 1, 0x01, 0, 0, &mut buf);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            UsbError::HalError(HalError::UsbTransferError)
        );
    }

    // -----------------------------------------------------------------------
    // bulk_in / bulk_out tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bulk_in() {
        let mut hc = MockHc::success();
        let mut buf = [0u8; 512];
        let result = bulk_in(&mut hc, 1, 0x01, &mut buf);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().bytes_transferred, 64);
    }

    #[test]
    fn test_bulk_in_hal_error() {
        let mut hc = MockHc {
            control_result: Ok(UsbTransferResult {
                bytes_transferred: 0,
                success: true,
                stalled: false,
                token: 0,
            }),
            bulk_result: Err(HalError::Timeout),
        };
        let mut buf = [0u8; 512];
        let result = bulk_in(&mut hc, 1, 0x01, &mut buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_bulk_out() {
        let mut hc = MockHc::success();
        let mut data = [0xAA; 128];
        let result = bulk_out(&mut hc, 1, 0x02, &mut data);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().bytes_transferred, 64);
    }

    #[test]
    fn test_bulk_out_hal_error() {
        let mut hc = MockHc {
            control_result: Ok(UsbTransferResult {
                bytes_transferred: 0,
                success: true,
                stalled: false,
                token: 0,
            }),
            bulk_result: Err(HalError::UsbTransferError),
        };
        let mut data = [0xAA; 128];
        let result = bulk_out(&mut hc, 1, 0x02, &mut data);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // clear_endpoint_halt tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_clear_endpoint_halt_success() {
        let mut hc = MockHc {
            control_result: Ok(UsbTransferResult {
                bytes_transferred: 0,
                success: true,
                stalled: false,
                token: 0,
            }),
            bulk_result: Ok(UsbTransferResult {
                bytes_transferred: 0,
                success: true,
                stalled: false,
                token: 0,
            }),
        };
        let result = clear_endpoint_halt(&mut hc, 1, 0x81);
        assert!(result.is_ok());
    }

    #[test]
    fn test_clear_endpoint_halt_stalled() {
        let mut hc = MockHc {
            control_result: Ok(UsbTransferResult {
                bytes_transferred: 0,
                success: false,
                stalled: true,
                token: 0,
            }),
            bulk_result: Ok(UsbTransferResult {
                bytes_transferred: 0,
                success: true,
                stalled: false,
                token: 0,
            }),
        };
        let result = clear_endpoint_halt(&mut hc, 1, 0x81);
        assert_eq!(result.unwrap_err(), UsbError::Stall);
    }

    #[test]
    fn test_clear_endpoint_halt_transfer_error() {
        let mut hc = MockHc {
            control_result: Ok(UsbTransferResult {
                bytes_transferred: 0,
                success: false,
                stalled: false,
                token: 0,
            }),
            bulk_result: Ok(UsbTransferResult {
                bytes_transferred: 0,
                success: true,
                stalled: false,
                token: 0,
            }),
        };
        let result = clear_endpoint_halt(&mut hc, 1, 0x81);
        assert_eq!(result.unwrap_err(), UsbError::TransferError);
    }

    #[test]
    fn test_clear_endpoint_halt_hal_error() {
        let mut hc = MockHc {
            control_result: Err(HalError::UsbEndpointHalted),
            bulk_result: Ok(UsbTransferResult {
                bytes_transferred: 0,
                success: true,
                stalled: false,
                token: 0,
            }),
        };
        let result = clear_endpoint_halt(&mut hc, 1, 0x81);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            UsbError::HalError(HalError::UsbEndpointHalted)
        );
    }
}
