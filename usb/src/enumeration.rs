// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! USB device enumeration state machine.
//!
//! Handles the sequence: port reset → SET_ADDRESS → GET_DESCRIPTOR(Device) →
//! GET_DESCRIPTOR(Configuration) → SET_CONFIGURATION.

use crate::descriptor::{ConfigDescriptor, DeviceDescriptor, CONFIG_DESCRIPTOR_MIN_LEN, DEVICE_DESCRIPTOR_LEN};
use crate::{request, request_type, descriptor_type, UsbError};
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
        if addr >= 1 && addr <= MAX_USB_ADDRESS {
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
    let result = hc.control_transfer(slot, &setup, Some(&mut config_buf[..CONFIG_DESCRIPTOR_MIN_LEN]))?;
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

    #[test]
    fn test_enumeration_state_variants() {
        assert_ne!(EnumerationState::WaitingForReset, EnumerationState::Complete);
        assert_eq!(EnumerationState::Failed, EnumerationState::Failed);
    }
}
