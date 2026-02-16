// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! USB device enumeration state machine.
//!
//! Handles the sequence: port reset → SET_ADDRESS → GET_DESCRIPTOR(Device) →
//! GET_DESCRIPTOR(Configuration) → SET_CONFIGURATION.

use crate::descriptor::{
    ConfigDescriptor, DeviceDescriptor, CONFIG_DESCRIPTOR_MIN_LEN, DEVICE_DESCRIPTOR_LEN,
};
use crate::{descriptor_type, request, request_type, UsbError};
use smallaios_kernel::hal::{UsbHostController, UsbSetupPacket};

/// Maximum USB device address (7-bit addressing, 1-127).
pub const MAX_USB_ADDRESS: u8 = 127;

/// Timeout for enumeration operations in milliseconds.
pub const ENUMERATION_TIMEOUT_MS: u32 = 5000;

/// Maximum configuration descriptor size we'll accept.
pub const MAX_CONFIG_DESC_SIZE: usize = 512;

/// Enumeration state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnumerationState {
    /// Waiting for port reset to complete.
    WaitingForReset,
    /// Port reset done, ready to assign address.
    SettingAddress,
    /// Address assigned, reading device descriptor.
    ReadingDeviceDescriptor,
    /// Device descriptor read, reading configuration descriptor.
    ReadingConfigDescriptor,
    /// Configuration descriptor read, setting configuration.
    SettingConfiguration,
    /// Enumeration complete.
    Complete,
    /// Enumeration failed.
    Failed,
}

/// Result of a successful enumeration.
#[derive(Debug)]
pub struct EnumeratedDevice {
    /// Assigned device slot (from host controller).
    pub slot: u8,
    /// Parsed device descriptor.
    pub device_desc: DeviceDescriptor,
    /// Raw configuration descriptor chain.
    pub config_data: [u8; MAX_CONFIG_DESC_SIZE],
    /// Valid bytes in `config_data`.
    pub config_data_len: usize,
    /// Parsed configuration descriptor header.
    pub config_desc: ConfigDescriptor,
    /// Root hub port number.
    pub port: u8,
}

/// Address allocator: tracks which USB addresses (1-127) are in use.
pub struct AddressAllocator {
    /// Bitmap: bit N is set if address N is in use.
    used: [u32; 4], // 128 bits
}

impl Default for AddressAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl AddressAllocator {
    /// Create a new allocator with all addresses free.
    pub fn new() -> Self {
        Self { used: [0; 4] }
    }

    /// Allocate the next free address (1-127).
    pub fn allocate(&mut self) -> Option<u8> {
        for addr in 1..=MAX_USB_ADDRESS {
            let word = (addr / 32) as usize;
            let bit = addr % 32;
            if self.used[word] & (1 << bit) == 0 {
                self.used[word] |= 1 << bit;
                return Some(addr);
            }
        }
        None
    }

    /// Release an address back to the pool.
    pub fn release(&mut self, addr: u8) {
        if (1..=MAX_USB_ADDRESS).contains(&addr) {
            let word = (addr / 32) as usize;
            let bit = addr % 32;
            self.used[word] &= !(1 << bit);
        }
    }

    /// Check if an address is in use.
    pub fn is_used(&self, addr: u8) -> bool {
        if addr == 0 || addr > MAX_USB_ADDRESS {
            return false;
        }
        let word = (addr / 32) as usize;
        let bit = addr % 32;
        self.used[word] & (1 << bit) != 0
    }
}

/// Build a SET_ADDRESS setup packet.
pub fn make_set_address(addr: u8) -> UsbSetupPacket {
    UsbSetupPacket::new(
        request_type::DIR_OUT | request_type::TYPE_STANDARD | request_type::RECIP_DEVICE,
        request::SET_ADDRESS,
        addr as u16,
        0,
        0,
    )
}

/// Build a GET_DESCRIPTOR(Device) setup packet.
pub fn make_get_device_descriptor() -> UsbSetupPacket {
    UsbSetupPacket::new(
        request_type::DIR_IN | request_type::TYPE_STANDARD | request_type::RECIP_DEVICE,
        request::GET_DESCRIPTOR,
        (descriptor_type::DEVICE as u16) << 8,
        0,
        DEVICE_DESCRIPTOR_LEN as u16,
    )
}

/// Build a GET_DESCRIPTOR(Configuration) setup packet.
pub fn make_get_config_descriptor(length: u16) -> UsbSetupPacket {
    UsbSetupPacket::new(
        request_type::DIR_IN | request_type::TYPE_STANDARD | request_type::RECIP_DEVICE,
        request::GET_DESCRIPTOR,
        (descriptor_type::CONFIGURATION as u16) << 8,
        0,
        length,
    )
}

/// Build a SET_CONFIGURATION setup packet.
pub fn make_set_configuration(config_value: u8) -> UsbSetupPacket {
    UsbSetupPacket::new(
        request_type::DIR_OUT | request_type::TYPE_STANDARD | request_type::RECIP_DEVICE,
        request::SET_CONFIGURATION,
        config_value as u16,
        0,
        0,
    )
}

/// Build a CLEAR_FEATURE(ENDPOINT_HALT) setup packet.
pub fn make_clear_endpoint_halt(endpoint: u8) -> UsbSetupPacket {
    UsbSetupPacket::new(
        request_type::DIR_OUT | request_type::TYPE_STANDARD | request_type::RECIP_ENDPOINT,
        request::CLEAR_FEATURE,
        crate::feature::ENDPOINT_HALT,
        endpoint as u16,
        0,
    )
}

/// Enumerate a newly connected USB device on the given port.
///
/// Performs: port reset → device_attach → SET_ADDRESS → GET_DESCRIPTOR(Device)
/// → GET_DESCRIPTOR(Configuration) → SET_CONFIGURATION.
pub fn enumerate_device(
    hc: &mut dyn UsbHostController,
    port: u8,
) -> Result<EnumeratedDevice, UsbError> {
    // Step 1: Reset port.
    hc.port_reset(port)?;

    // Step 2: Attach device (allocates slot, assigns address).
    let slot = hc.device_attach(port)?;

    // Step 3: GET_DESCRIPTOR(Device).
    let setup = make_get_device_descriptor();
    let mut dev_buf = [0u8; DEVICE_DESCRIPTOR_LEN];
    let result = hc.control_transfer(slot, &setup, Some(&mut dev_buf))?;
    if !result.success || (result.bytes_transferred as usize) < DEVICE_DESCRIPTOR_LEN {
        return Err(UsbError::TransferError);
    }
    let device_desc = DeviceDescriptor::parse(&dev_buf)?;

    // Step 4: GET_DESCRIPTOR(Configuration) — first read header for total_length.
    let setup = make_get_config_descriptor(CONFIG_DESCRIPTOR_MIN_LEN as u16);
    let mut config_buf = [0u8; MAX_CONFIG_DESC_SIZE];
    let result = hc.control_transfer(
        slot,
        &setup,
        Some(&mut config_buf[..CONFIG_DESCRIPTOR_MIN_LEN]),
    )?;
    if !result.success || (result.bytes_transferred as usize) < CONFIG_DESCRIPTOR_MIN_LEN {
        return Err(UsbError::TransferError);
    }
    let config_header = ConfigDescriptor::parse(&config_buf)?;
    let total_len = config_header.total_length as usize;
    let total_len = total_len.min(MAX_CONFIG_DESC_SIZE);

    // Read full configuration descriptor chain.
    if total_len > CONFIG_DESCRIPTOR_MIN_LEN {
        let setup = make_get_config_descriptor(total_len as u16);
        let result = hc.control_transfer(slot, &setup, Some(&mut config_buf[..total_len]))?;
        if !result.success {
            return Err(UsbError::TransferError);
        }
    }

    // Re-parse the full config descriptor.
    let config_desc = ConfigDescriptor::parse(&config_buf)?;

    // Step 5: SET_CONFIGURATION.
    let setup = make_set_configuration(config_desc.configuration_value);
    let result = hc.control_transfer(slot, &setup, None)?;
    if !result.success {
        return Err(UsbError::TransferError);
    }

    Ok(EnumeratedDevice {
        slot,
        device_desc,
        config_data: config_buf,
        config_data_len: total_len,
        config_desc,
        port,
    })
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use smallaios_kernel::hal::{
        HalError, UsbHostController, UsbPortStatus, UsbSetupPacket, UsbSpeed, UsbTransferResult,
        UsbTransferType,
    };

    // -----------------------------------------------------------------------
    // Test data helpers
    // -----------------------------------------------------------------------

    fn valid_device_descriptor() -> [u8; 18] {
        [
            18,   // bLength
            0x01, // bDescriptorType = DEVICE
            0x10, 0x02, // bcdUSB = 2.10
            0xFF, // bDeviceClass = Vendor
            0x00, // bDeviceSubClass
            0x00, // bDeviceProtocol
            64,   // bMaxPacketSize0
            0xAD, 0xDE, // idVendor = 0xDEAD
            0xEF, 0xBE, // idProduct = 0xBEEF
            0x00, 0x01, // bcdDevice = 1.00
            1,    // iManufacturer
            2,    // iProduct
            3,    // iSerialNumber
            1,    // bNumConfigurations
        ]
    }

    fn valid_config_descriptor() -> [u8; 9] {
        [
            9,    // bLength
            0x02, // bDescriptorType = CONFIGURATION
            9, 0,    // wTotalLength = 9 (header only)
            1,    // bNumInterfaces
            1,    // bConfigurationValue
            0,    // iConfiguration
            0x80, // bmAttributes = bus-powered
            50,   // bMaxPower = 100mA
        ]
    }

    // -----------------------------------------------------------------------
    // Scripted mock USB host controller
    // -----------------------------------------------------------------------

    /// A scripted control transfer response: (result, optional data to fill).
    type ControlResponse = (Result<UsbTransferResult, HalError>, Option<Vec<u8>>);

    /// Tracks which step in the enumeration sequence we are at and provides
    /// scripted responses.
    struct MockHc {
        /// Scripted control transfer responses in order.
        control_responses: Vec<ControlResponse>,
        control_call_index: usize,
        port_reset_result: Result<(), HalError>,
        device_attach_result: Result<u8, HalError>,
    }

    impl MockHc {
        /// Create a mock that succeeds through the full enumeration sequence.
        fn success() -> Self {
            let dev_desc = valid_device_descriptor();
            let cfg_desc = valid_config_descriptor();

            let ok_transfer = |bytes: u32, data: Option<Vec<u8>>| {
                (
                    Ok(UsbTransferResult {
                        bytes_transferred: bytes,
                        success: true,
                        stalled: false,
                        token: 0,
                    }),
                    data,
                )
            };

            Self {
                control_responses: alloc::vec![
                    // Step 3: GET_DESCRIPTOR(Device) — returns 18-byte device descriptor
                    ok_transfer(18, Some(dev_desc.to_vec())),
                    // Step 4a: GET_DESCRIPTOR(Config header) — returns 9-byte config descriptor
                    ok_transfer(9, Some(cfg_desc.to_vec())),
                    // Step 5: SET_CONFIGURATION — no data
                    ok_transfer(0, None),
                ],
                control_call_index: 0,
                port_reset_result: Ok(()),
                device_attach_result: Ok(1),
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
            self.port_reset_result
        }
        fn device_attach(&mut self, _port: u8) -> Result<u8, HalError> {
            self.device_attach_result
        }
        fn device_detach(&mut self, _slot: u8) -> Result<(), HalError> {
            Ok(())
        }
        fn control_transfer(
            &mut self,
            _slot: u8,
            _setup: &UsbSetupPacket,
            data: Option<&mut [u8]>,
        ) -> Result<UsbTransferResult, HalError> {
            if self.control_call_index >= self.control_responses.len() {
                return Err(HalError::UsbTransferError);
            }
            let (result, fill_data) = &self.control_responses[self.control_call_index];
            self.control_call_index += 1;

            // Fill the data buffer if requested and data is provided.
            if let (Some(buf), Some(fill)) = (data, fill_data) {
                let len = buf.len().min(fill.len());
                buf[..len].copy_from_slice(&fill[..len]);
            }

            *result
        }
        fn bulk_transfer(
            &mut self,
            _slot: u8,
            _endpoint: u8,
            _data: &mut [u8],
        ) -> Result<UsbTransferResult, HalError> {
            Ok(UsbTransferResult {
                bytes_transferred: 0,
                success: true,
                stalled: false,
                token: 0,
            })
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
    // AddressAllocator tests (existing + new)
    // -----------------------------------------------------------------------

    #[test]
    fn test_address_allocator_basic() {
        let mut alloc = AddressAllocator::new();
        assert!(!alloc.is_used(1));
        let addr = alloc.allocate().unwrap();
        assert_eq!(addr, 1);
        assert!(alloc.is_used(1));
    }

    #[test]
    fn test_address_allocator_sequential() {
        let mut alloc = AddressAllocator::new();
        for expected in 1..=10u8 {
            let addr = alloc.allocate().unwrap();
            assert_eq!(addr, expected);
        }
    }

    #[test]
    fn test_address_allocator_release() {
        let mut alloc = AddressAllocator::new();
        let addr = alloc.allocate().unwrap();
        assert!(alloc.is_used(addr));
        alloc.release(addr);
        assert!(!alloc.is_used(addr));
        // Re-allocate should return the same address.
        let addr2 = alloc.allocate().unwrap();
        assert_eq!(addr2, addr);
    }

    #[test]
    fn test_address_allocator_exhaustion() {
        let mut alloc = AddressAllocator::new();
        for _ in 1..=127 {
            assert!(alloc.allocate().is_some());
        }
        assert!(alloc.allocate().is_none());
    }

    #[test]
    fn test_address_allocator_zero_not_used() {
        let alloc = AddressAllocator::new();
        assert!(!alloc.is_used(0));
    }

    #[test]
    fn test_address_allocator_default() {
        let alloc = AddressAllocator::default();
        // Default should behave identically to new().
        assert!(!alloc.is_used(1));
        assert!(!alloc.is_used(0));
        assert!(!alloc.is_used(127));
    }

    #[test]
    fn test_address_allocator_release_zero() {
        // Releasing address 0 should be a no-op (no panic).
        let mut alloc = AddressAllocator::new();
        alloc.release(0);
        // State should be unchanged.
        assert!(!alloc.is_used(0));
        assert!(!alloc.is_used(1));
    }

    #[test]
    fn test_address_allocator_release_out_of_range() {
        // Releasing address 128+ should be a no-op (no panic).
        let mut alloc = AddressAllocator::new();
        alloc.release(128);
        alloc.release(255);
        // State should be unchanged.
        assert!(!alloc.is_used(1));
    }

    #[test]
    fn test_address_allocator_is_used_out_of_range() {
        let alloc = AddressAllocator::new();
        // Out-of-range addresses should return false.
        assert!(!alloc.is_used(128));
        assert!(!alloc.is_used(255));
    }

    // -----------------------------------------------------------------------
    // Setup packet builder tests (existing)
    // -----------------------------------------------------------------------

    #[test]
    fn test_make_set_address() {
        let pkt = make_set_address(5);
        assert_eq!(pkt.bm_request_type, 0x00);
        assert_eq!(pkt.b_request, request::SET_ADDRESS);
        assert_eq!(pkt.w_value, 5);
        assert_eq!(pkt.w_length, 0);
    }

    #[test]
    fn test_make_get_device_descriptor() {
        let pkt = make_get_device_descriptor();
        assert_eq!(pkt.bm_request_type, 0x80);
        assert_eq!(pkt.b_request, request::GET_DESCRIPTOR);
        assert_eq!(pkt.w_value, 0x0100); // DEVICE descriptor type << 8
        assert_eq!(pkt.w_length, 18);
    }

    #[test]
    fn test_make_get_config_descriptor() {
        let pkt = make_get_config_descriptor(255);
        assert_eq!(pkt.bm_request_type, 0x80);
        assert_eq!(pkt.b_request, request::GET_DESCRIPTOR);
        assert_eq!(pkt.w_value, 0x0200); // CONFIGURATION descriptor type << 8
        assert_eq!(pkt.w_length, 255);
    }

    #[test]
    fn test_make_set_configuration() {
        let pkt = make_set_configuration(1);
        assert_eq!(pkt.bm_request_type, 0x00);
        assert_eq!(pkt.b_request, request::SET_CONFIGURATION);
        assert_eq!(pkt.w_value, 1);
    }

    #[test]
    fn test_make_clear_endpoint_halt() {
        let pkt = make_clear_endpoint_halt(0x81);
        assert_eq!(pkt.bm_request_type, 0x02); // OUT | STANDARD | ENDPOINT
        assert_eq!(pkt.b_request, request::CLEAR_FEATURE);
        assert_eq!(pkt.w_value, 0x0000); // ENDPOINT_HALT
        assert_eq!(pkt.w_index, 0x0081);
    }

    // -----------------------------------------------------------------------
    // EnumerationState tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_enumeration_state_variants() {
        assert_ne!(
            EnumerationState::WaitingForReset,
            EnumerationState::Complete
        );
        assert_eq!(EnumerationState::Failed, EnumerationState::Failed);
    }

    #[test]
    fn test_enumeration_state_all_variants() {
        // Exercise all EnumerationState variants for coverage of Debug/PartialEq.
        let states = [
            EnumerationState::WaitingForReset,
            EnumerationState::SettingAddress,
            EnumerationState::ReadingDeviceDescriptor,
            EnumerationState::ReadingConfigDescriptor,
            EnumerationState::SettingConfiguration,
            EnumerationState::Complete,
            EnumerationState::Failed,
        ];
        // Each state should be equal to itself.
        for s in &states {
            assert_eq!(*s, *s);
        }
        // All states should be distinct.
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                assert_ne!(states[i], states[j]);
            }
        }
        // Debug output should work for each.
        for s in &states {
            let debug = alloc::format!("{:?}", s);
            assert!(!debug.is_empty());
        }
    }

    // -----------------------------------------------------------------------
    // enumerate_device success path
    // -----------------------------------------------------------------------

    #[test]
    fn test_enumerate_device_success() {
        let mut hc = MockHc::success();
        let result = enumerate_device(&mut hc, 0);
        assert!(result.is_ok());
        let dev = result.unwrap();
        assert_eq!(dev.slot, 1);
        assert_eq!(dev.port, 0);
        assert_eq!(dev.device_desc.vendor_id, 0xDEAD);
        assert_eq!(dev.device_desc.product_id, 0xBEEF);
        assert_eq!(dev.device_desc.device_class, 0xFF);
        assert_eq!(dev.config_desc.configuration_value, 1);
        assert_eq!(dev.config_desc.num_interfaces, 1);
        assert_eq!(dev.config_data_len, 9);
    }

    // -----------------------------------------------------------------------
    // enumerate_device with total_length > 9 (triggers full config read)
    // -----------------------------------------------------------------------

    #[test]
    fn test_enumerate_device_large_config_descriptor() {
        let dev_desc = valid_device_descriptor();
        // Config descriptor with total_length = 32 (larger than header).
        let mut cfg_desc_header = valid_config_descriptor();
        cfg_desc_header[2] = 32; // wTotalLength low byte
        cfg_desc_header[3] = 0; // wTotalLength high byte

        // Build a full 32-byte config descriptor chain:
        // config(9) + interface(9) + endpoint(7) + endpoint(7) = 32
        let mut full_config = [0u8; 32];
        full_config[..9].copy_from_slice(&cfg_desc_header);
        // Interface descriptor
        full_config[9] = 9;
        full_config[10] = 0x04; // INTERFACE
        full_config[11] = 0; // interface number
        full_config[13] = 2; // num_endpoints
        full_config[14] = 0xFF; // class
                                // Endpoint 1 (bulk IN 0x81)
        full_config[18] = 7;
        full_config[19] = 0x05; // ENDPOINT
        full_config[20] = 0x81;
        full_config[21] = 0x02; // bulk
        full_config[22] = 0x00;
        full_config[23] = 0x02; // 512 max pkt
                                // Endpoint 2 (bulk OUT 0x02)
        full_config[25] = 7;
        full_config[26] = 0x05; // ENDPOINT
        full_config[27] = 0x02;
        full_config[28] = 0x02; // bulk
        full_config[29] = 0x00;
        full_config[30] = 0x02; // 512 max pkt

        let ok = |bytes: u32, data: Option<Vec<u8>>| {
            (
                Ok(UsbTransferResult {
                    bytes_transferred: bytes,
                    success: true,
                    stalled: false,
                    token: 0,
                }),
                data,
            )
        };

        let mut hc = MockHc {
            control_responses: alloc::vec![
                ok(18, Some(dev_desc.to_vec())),
                ok(9, Some(cfg_desc_header.to_vec())),
                ok(32, Some(full_config.to_vec())),
                ok(0, None),
            ],
            control_call_index: 0,
            port_reset_result: Ok(()),
            device_attach_result: Ok(1),
        };

        let result = enumerate_device(&mut hc, 0);
        assert!(result.is_ok());
        let dev = result.unwrap();
        assert_eq!(dev.config_data_len, 32);
        assert_eq!(dev.config_desc.total_length, 32);
        assert_eq!(dev.config_desc.num_interfaces, 1);
    }

    // -----------------------------------------------------------------------
    // enumerate_device failure paths
    // -----------------------------------------------------------------------

    #[test]
    fn test_enumerate_device_port_reset_failure() {
        let mut hc = MockHc {
            control_responses: alloc::vec![],
            control_call_index: 0,
            port_reset_result: Err(HalError::Timeout),
            device_attach_result: Ok(1),
        };
        let result = enumerate_device(&mut hc, 0);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), UsbError::HalError(HalError::Timeout));
    }

    #[test]
    fn test_enumerate_device_attach_failure() {
        let mut hc = MockHc {
            control_responses: alloc::vec![],
            control_call_index: 0,
            port_reset_result: Ok(()),
            device_attach_result: Err(HalError::UsbDeviceNotFound),
        };
        let result = enumerate_device(&mut hc, 0);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            UsbError::HalError(HalError::UsbDeviceNotFound)
        );
    }

    #[test]
    fn test_enumerate_device_get_device_desc_transfer_error() {
        // Control transfer returns not-success for device descriptor.
        let mut hc = MockHc {
            control_responses: alloc::vec![(
                Ok(UsbTransferResult {
                    bytes_transferred: 0,
                    success: false,
                    stalled: false,
                    token: 0,
                }),
                None,
            )],
            control_call_index: 0,
            port_reset_result: Ok(()),
            device_attach_result: Ok(1),
        };
        let result = enumerate_device(&mut hc, 0);
        assert_eq!(result.unwrap_err(), UsbError::TransferError);
    }

    #[test]
    fn test_enumerate_device_get_device_desc_hal_error() {
        // Control transfer returns HalError for device descriptor.
        let mut hc = MockHc {
            control_responses: alloc::vec![(Err(HalError::UsbTransferError), None)],
            control_call_index: 0,
            port_reset_result: Ok(()),
            device_attach_result: Ok(1),
        };
        let result = enumerate_device(&mut hc, 0);
        assert_eq!(
            result.unwrap_err(),
            UsbError::HalError(HalError::UsbTransferError)
        );
    }

    #[test]
    fn test_enumerate_device_get_device_desc_short_transfer() {
        // Control transfer succeeds but returns too few bytes.
        let dev_desc = valid_device_descriptor();
        let mut hc = MockHc {
            control_responses: alloc::vec![(
                Ok(UsbTransferResult {
                    bytes_transferred: 8, // Too short (need 18)
                    success: true,
                    stalled: false,
                    token: 0,
                }),
                Some(dev_desc.to_vec()),
            )],
            control_call_index: 0,
            port_reset_result: Ok(()),
            device_attach_result: Ok(1),
        };
        let result = enumerate_device(&mut hc, 0);
        assert_eq!(result.unwrap_err(), UsbError::TransferError);
    }

    #[test]
    fn test_enumerate_device_get_config_header_failure() {
        let dev_desc = valid_device_descriptor();
        let ok = |bytes: u32, data: Option<Vec<u8>>| {
            (
                Ok(UsbTransferResult {
                    bytes_transferred: bytes,
                    success: true,
                    stalled: false,
                    token: 0,
                }),
                data,
            )
        };
        let mut hc = MockHc {
            control_responses: alloc::vec![
                ok(18, Some(dev_desc.to_vec())),
                // Config header read fails
                (
                    Ok(UsbTransferResult {
                        bytes_transferred: 0,
                        success: false,
                        stalled: false,
                        token: 0,
                    }),
                    None,
                ),
            ],
            control_call_index: 0,
            port_reset_result: Ok(()),
            device_attach_result: Ok(1),
        };
        let result = enumerate_device(&mut hc, 0);
        assert_eq!(result.unwrap_err(), UsbError::TransferError);
    }

    #[test]
    fn test_enumerate_device_get_config_header_short() {
        let dev_desc = valid_device_descriptor();
        let cfg_desc = valid_config_descriptor();
        let ok = |bytes: u32, data: Option<Vec<u8>>| {
            (
                Ok(UsbTransferResult {
                    bytes_transferred: bytes,
                    success: true,
                    stalled: false,
                    token: 0,
                }),
                data,
            )
        };
        let mut hc = MockHc {
            control_responses: alloc::vec![
                ok(18, Some(dev_desc.to_vec())),
                // Config header returns too few bytes
                (
                    Ok(UsbTransferResult {
                        bytes_transferred: 4, // Need at least 9
                        success: true,
                        stalled: false,
                        token: 0,
                    }),
                    Some(cfg_desc.to_vec()),
                ),
            ],
            control_call_index: 0,
            port_reset_result: Ok(()),
            device_attach_result: Ok(1),
        };
        let result = enumerate_device(&mut hc, 0);
        assert_eq!(result.unwrap_err(), UsbError::TransferError);
    }

    #[test]
    fn test_enumerate_device_set_configuration_failure() {
        let dev_desc = valid_device_descriptor();
        let cfg_desc = valid_config_descriptor();
        let ok = |bytes: u32, data: Option<Vec<u8>>| {
            (
                Ok(UsbTransferResult {
                    bytes_transferred: bytes,
                    success: true,
                    stalled: false,
                    token: 0,
                }),
                data,
            )
        };
        let mut hc = MockHc {
            control_responses: alloc::vec![
                ok(18, Some(dev_desc.to_vec())),
                ok(9, Some(cfg_desc.to_vec())),
                // SET_CONFIGURATION fails
                (
                    Ok(UsbTransferResult {
                        bytes_transferred: 0,
                        success: false,
                        stalled: false,
                        token: 0,
                    }),
                    None,
                ),
            ],
            control_call_index: 0,
            port_reset_result: Ok(()),
            device_attach_result: Ok(1),
        };
        let result = enumerate_device(&mut hc, 0);
        assert_eq!(result.unwrap_err(), UsbError::TransferError);
    }

    #[test]
    fn test_enumerate_device_full_config_read_failure() {
        let dev_desc = valid_device_descriptor();
        // Config with total_length > 9 to trigger full read
        let mut cfg_desc_header = valid_config_descriptor();
        cfg_desc_header[2] = 32;
        cfg_desc_header[3] = 0;

        let ok = |bytes: u32, data: Option<Vec<u8>>| {
            (
                Ok(UsbTransferResult {
                    bytes_transferred: bytes,
                    success: true,
                    stalled: false,
                    token: 0,
                }),
                data,
            )
        };
        let mut hc = MockHc {
            control_responses: alloc::vec![
                ok(18, Some(dev_desc.to_vec())),
                ok(9, Some(cfg_desc_header.to_vec())),
                // Full config read fails
                (
                    Ok(UsbTransferResult {
                        bytes_transferred: 32,
                        success: false,
                        stalled: false,
                        token: 0,
                    }),
                    None,
                ),
            ],
            control_call_index: 0,
            port_reset_result: Ok(()),
            device_attach_result: Ok(1),
        };
        let result = enumerate_device(&mut hc, 0);
        assert_eq!(result.unwrap_err(), UsbError::TransferError);
    }

    #[test]
    fn test_enumerate_device_full_config_read_hal_error() {
        let dev_desc = valid_device_descriptor();
        let mut cfg_desc_header = valid_config_descriptor();
        cfg_desc_header[2] = 32;
        cfg_desc_header[3] = 0;

        let ok = |bytes: u32, data: Option<Vec<u8>>| {
            (
                Ok(UsbTransferResult {
                    bytes_transferred: bytes,
                    success: true,
                    stalled: false,
                    token: 0,
                }),
                data,
            )
        };
        let mut hc = MockHc {
            control_responses: alloc::vec![
                ok(18, Some(dev_desc.to_vec())),
                ok(9, Some(cfg_desc_header.to_vec())),
                // Full config read: HAL error
                (Err(HalError::Timeout), None),
            ],
            control_call_index: 0,
            port_reset_result: Ok(()),
            device_attach_result: Ok(1),
        };
        let result = enumerate_device(&mut hc, 0);
        assert_eq!(result.unwrap_err(), UsbError::HalError(HalError::Timeout));
    }

    // -----------------------------------------------------------------------
    // Constants tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_enumeration_constants() {
        assert_eq!(MAX_USB_ADDRESS, 127);
        assert_eq!(ENUMERATION_TIMEOUT_MS, 5000);
        assert_eq!(MAX_CONFIG_DESC_SIZE, 512);
    }
}
