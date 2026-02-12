// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! USB gadget (device-mode) framework.
//!
//! Provides gadget function registration, USB descriptor composition,
//! and EP0 standard request handling for presenting SmallAIOS as a
//! USB peripheral device.

use crate::{descriptor_type, request, UsbError};
use smallaios_kernel::hal::UsbSetupPacket;

/// Maximum number of gadget functions that can be registered.
pub const MAX_GADGET_FUNCTIONS: usize = 4;

/// Maximum number of interfaces across all functions.
pub const MAX_INTERFACES: usize = 8;

/// Maximum number of endpoints across all functions.
pub const MAX_ENDPOINTS: usize = 16;

/// Maximum length of a composed configuration descriptor.
pub const MAX_CONFIG_DESC_LEN: usize = 256;

/// Maximum string descriptor length (UTF-16LE + header).
pub const MAX_STRING_DESC_LEN: usize = 126;

// ---------------------------------------------------------------------------
// Gadget function trait
// ---------------------------------------------------------------------------

/// Descriptor set provided by a gadget function.
#[derive(Debug, Clone)]
pub struct GadgetFunctionDescriptors {
    /// Interface descriptor bytes (9 bytes per interface).
    pub interfaces: [[u8; 9]; 2],
    /// Number of interfaces.
    pub num_interfaces: u8,
    /// Endpoint descriptor bytes (7 bytes per endpoint).
    pub endpoints: [[u8; 7]; 4],
    /// Number of endpoints.
    pub num_endpoints: u8,
    /// Interface class code (for the primary interface).
    pub interface_class: u8,
    /// Interface subclass code.
    pub interface_sub_class: u8,
    /// Interface protocol code.
    pub interface_protocol: u8,
}

// ---------------------------------------------------------------------------
// Gadget device configuration
// ---------------------------------------------------------------------------

/// USB gadget device identity.
#[derive(Debug, Clone, Copy)]
pub struct GadgetDeviceConfig {
    /// Vendor ID.
    pub vendor_id: u16,
    /// Product ID.
    pub product_id: u16,
    /// Device class (0xEF for composite with IAD, 0xFF for vendor-specific).
    pub device_class: u8,
    /// Device subclass.
    pub device_sub_class: u8,
    /// Device protocol.
    pub device_protocol: u8,
    /// BCD device version.
    pub bcd_device: u16,
}

impl Default for GadgetDeviceConfig {
    fn default() -> Self {
        Self {
            vendor_id: 0x1209, // pid.codes test VID
            product_id: 0x0001,
            device_class: 0xFF,
            device_sub_class: 0x00,
            device_protocol: 0x00,
            bcd_device: 0x0100,
        }
    }
}

// ---------------------------------------------------------------------------
// Descriptor composer
// ---------------------------------------------------------------------------

/// Compose a USB device descriptor for the gadget.
pub fn compose_device_descriptor(config: &GadgetDeviceConfig) -> [u8; 18] {
    let mut d = [0u8; 18];
    d[0] = 18;
    d[1] = descriptor_type::DEVICE;
    d[2] = 0x00;
    d[3] = 0x02; // USB 2.0
    d[4] = config.device_class;
    d[5] = config.device_sub_class;
    d[6] = config.device_protocol;
    d[7] = 64; // EP0 max packet size
    d[8] = (config.vendor_id & 0xFF) as u8;
    d[9] = (config.vendor_id >> 8) as u8;
    d[10] = (config.product_id & 0xFF) as u8;
    d[11] = (config.product_id >> 8) as u8;
    d[12] = (config.bcd_device & 0xFF) as u8;
    d[13] = (config.bcd_device >> 8) as u8;
    d[14] = 1; // iManufacturer
    d[15] = 2; // iProduct
    d[16] = 3; // iSerialNumber
    d[17] = 1; // bNumConfigurations
    d
}

/// Compose a configuration descriptor from registered gadget function descriptors.
///
/// Returns the number of valid bytes written to `out`.
pub fn compose_config_descriptor(
    functions: &[GadgetFunctionDescriptors],
    out: &mut [u8; MAX_CONFIG_DESC_LEN],
) -> usize {
    let mut offset = 9; // Reserve space for config descriptor header.
    let mut total_interfaces = 0u8;

    for func in functions {
        // Write interface descriptors.
        for i in 0..func.num_interfaces as usize {
            if offset + 9 > MAX_CONFIG_DESC_LEN {
                break;
            }
            let mut iface = func.interfaces[i];
            iface[2] = total_interfaces + i as u8; // Assign global interface number
            out[offset..offset + 9].copy_from_slice(&iface);
            offset += 9;

            // Write endpoint descriptors for this interface.
            // In a simple model, all endpoints belong to the last interface.
        }

        // Write endpoint descriptors.
        for i in 0..func.num_endpoints as usize {
            if offset + 7 > MAX_CONFIG_DESC_LEN {
                break;
            }
            out[offset..offset + 7].copy_from_slice(&func.endpoints[i]);
            offset += 7;
        }

        total_interfaces += func.num_interfaces;
    }

    // Fill in the config descriptor header.
    out[0] = 9; // bLength
    out[1] = descriptor_type::CONFIGURATION;
    out[2] = (offset & 0xFF) as u8;
    out[3] = (offset >> 8) as u8;
    out[4] = total_interfaces;
    out[5] = 1; // bConfigurationValue
    out[6] = 0; // iConfiguration
    out[7] = 0x80; // bmAttributes: bus-powered
    out[8] = 250; // MaxPower: 500mA

    offset
}

/// Encode a string as a USB string descriptor (UTF-16LE).
///
/// Returns the number of valid bytes written to `out`.
pub fn encode_string_descriptor(s: &str, out: &mut [u8; MAX_STRING_DESC_LEN]) -> usize {
    let max_chars = (MAX_STRING_DESC_LEN - 2) / 2;
    let mut offset = 2; // Skip bLength and bDescriptorType.

    for (i, ch) in s.chars().enumerate() {
        if i >= max_chars || offset + 2 > MAX_STRING_DESC_LEN {
            break;
        }
        let code = ch as u16;
        out[offset] = (code & 0xFF) as u8;
        out[offset + 1] = (code >> 8) as u8;
        offset += 2;
    }

    out[0] = offset as u8;
    out[1] = descriptor_type::STRING;
    offset
}

/// Encode the language ID string descriptor (string index 0).
pub fn encode_language_descriptor(out: &mut [u8; 4]) {
    out[0] = 4;
    out[1] = descriptor_type::STRING;
    out[2] = 0x09; // English (US) low byte
    out[3] = 0x04; // English (US) high byte
}

// ---------------------------------------------------------------------------
// EP0 request handler
// ---------------------------------------------------------------------------

/// Handle a standard USB device request on EP0.
///
/// Returns the response data to send, or an error if the request should be stalled.
pub fn handle_standard_request(
    setup: &UsbSetupPacket,
    device_desc: &[u8; 18],
    config_desc: &[u8; MAX_CONFIG_DESC_LEN],
    config_desc_len: usize,
    response_buf: &mut [u8],
) -> Result<usize, UsbError> {
    match setup.b_request {
        request::GET_DESCRIPTOR => {
            let desc_type = (setup.w_value >> 8) as u8;
            let desc_index = (setup.w_value & 0xFF) as u8;
            let max_len = setup.w_length as usize;

            match desc_type {
                descriptor_type::DEVICE => {
                    let len = max_len.min(18).min(response_buf.len());
                    response_buf[..len].copy_from_slice(&device_desc[..len]);
                    Ok(len)
                }
                descriptor_type::CONFIGURATION => {
                    let len = max_len.min(config_desc_len).min(response_buf.len());
                    response_buf[..len].copy_from_slice(&config_desc[..len]);
                    Ok(len)
                }
                descriptor_type::STRING => {
                    // String descriptors are handled separately.
                    // Return empty for now — caller should handle string indices.
                    if desc_index == 0 {
                        let mut lang = [0u8; 4];
                        encode_language_descriptor(&mut lang);
                        let len = max_len.min(4).min(response_buf.len());
                        response_buf[..len].copy_from_slice(&lang[..len]);
                        Ok(len)
                    } else {
                        Err(UsbError::Stall) // Unknown string index
                    }
                }
                _ => Err(UsbError::Stall),
            }
        }
        request::SET_ADDRESS => {
            // Handled at the HAL level; just ACK.
            Ok(0)
        }
        request::SET_CONFIGURATION => {
            // Accept configuration 1.
            Ok(0)
        }
        request::GET_CONFIGURATION => {
            if !response_buf.is_empty() {
                response_buf[0] = 1; // We only have configuration 1.
                Ok(1)
            } else {
                Err(UsbError::BufferTooSmall)
            }
        }
        request::GET_STATUS => {
            if response_buf.len() >= 2 {
                response_buf[0] = 0; // Not self-powered, no remote wakeup
                response_buf[1] = 0;
                Ok(2)
            } else {
                Err(UsbError::BufferTooSmall)
            }
        }
        _ => Err(UsbError::Stall), // Unsupported request
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compose_device_descriptor() {
        let config = GadgetDeviceConfig {
            vendor_id: 0x1234,
            product_id: 0x5678,
            device_class: 0xFF,
            device_sub_class: 0x01,
            device_protocol: 0x02,
            bcd_device: 0x0200,
        };
        let desc = compose_device_descriptor(&config);
        assert_eq!(desc[0], 18);
        assert_eq!(desc[1], descriptor_type::DEVICE);
        assert_eq!(u16::from_le_bytes([desc[8], desc[9]]), 0x1234);
        assert_eq!(u16::from_le_bytes([desc[10], desc[11]]), 0x5678);
        assert_eq!(desc[4], 0xFF);
    }

    #[test]
    fn test_compose_config_descriptor_single_function() {
        let mut iface = [0u8; 9];
        iface[0] = 9;
        iface[1] = descriptor_type::INTERFACE;
        iface[4] = 2; // num endpoints
        iface[5] = 0xFF;

        let mut ep_in = [0u8; 7];
        ep_in[0] = 7;
        ep_in[1] = descriptor_type::ENDPOINT;
        ep_in[2] = 0x81;
        ep_in[3] = 0x02;
        ep_in[4] = 0x00;
        ep_in[5] = 0x02; // 512

        let mut ep_out = [0u8; 7];
        ep_out[0] = 7;
        ep_out[1] = descriptor_type::ENDPOINT;
        ep_out[2] = 0x02;
        ep_out[3] = 0x02;
        ep_out[4] = 0x00;
        ep_out[5] = 0x02;

        let func = GadgetFunctionDescriptors {
            interfaces: [iface, [0u8; 9]],
            num_interfaces: 1,
            endpoints: [ep_in, ep_out, [0u8; 7], [0u8; 7]],
            num_endpoints: 2,
            interface_class: 0xFF,
            interface_sub_class: 0,
            interface_protocol: 0,
        };

        let mut out = [0u8; MAX_CONFIG_DESC_LEN];
        let len = compose_config_descriptor(&[func], &mut out);
        assert_eq!(len, 9 + 9 + 7 + 7); // config + 1 interface + 2 endpoints
        assert_eq!(out[0], 9);
        assert_eq!(out[1], descriptor_type::CONFIGURATION);
        assert_eq!(u16::from_le_bytes([out[2], out[3]]), len as u16);
        assert_eq!(out[4], 1); // 1 interface
    }

    #[test]
    fn test_encode_string_descriptor() {
        let mut out = [0u8; MAX_STRING_DESC_LEN];
        let len = encode_string_descriptor("SmallAIOS", &mut out);
        assert_eq!(out[0], len as u8);
        assert_eq!(out[1], descriptor_type::STRING);
        // 'S' = 0x0053
        assert_eq!(out[2], 0x53);
        assert_eq!(out[3], 0x00);
        assert_eq!(len, 2 + 9 * 2); // header + 9 chars * 2 bytes
    }

    #[test]
    fn test_encode_language_descriptor() {
        let mut out = [0u8; 4];
        encode_language_descriptor(&mut out);
        assert_eq!(out[0], 4);
        assert_eq!(out[1], descriptor_type::STRING);
        assert_eq!(u16::from_le_bytes([out[2], out[3]]), 0x0409);
    }

    #[test]
    fn test_handle_get_device_descriptor() {
        let config = GadgetDeviceConfig::default();
        let device_desc = compose_device_descriptor(&config);
        let config_desc = [0u8; MAX_CONFIG_DESC_LEN];
        let mut response = [0u8; 64];

        let setup = UsbSetupPacket::new(0x80, request::GET_DESCRIPTOR, 0x0100, 0, 18);
        let len =
            handle_standard_request(&setup, &device_desc, &config_desc, 9, &mut response).unwrap();
        assert_eq!(len, 18);
        assert_eq!(response[0], 18);
        assert_eq!(response[1], descriptor_type::DEVICE);
    }

    #[test]
    fn test_handle_get_config_descriptor() {
        let config = GadgetDeviceConfig::default();
        let device_desc = compose_device_descriptor(&config);
        let mut config_desc = [0u8; MAX_CONFIG_DESC_LEN];
        config_desc[0] = 9;
        config_desc[1] = descriptor_type::CONFIGURATION;
        config_desc[2] = 9;
        let mut response = [0u8; 64];

        let setup = UsbSetupPacket::new(0x80, request::GET_DESCRIPTOR, 0x0200, 0, 9);
        let len =
            handle_standard_request(&setup, &device_desc, &config_desc, 9, &mut response).unwrap();
        assert_eq!(len, 9);
    }

    #[test]
    fn test_handle_set_address() {
        let config = GadgetDeviceConfig::default();
        let device_desc = compose_device_descriptor(&config);
        let config_desc = [0u8; MAX_CONFIG_DESC_LEN];
        let mut response = [0u8; 64];

        let setup = UsbSetupPacket::new(0x00, request::SET_ADDRESS, 5, 0, 0);
        let len =
            handle_standard_request(&setup, &device_desc, &config_desc, 9, &mut response).unwrap();
        assert_eq!(len, 0);
    }

    #[test]
    fn test_handle_unsupported_request() {
        let config = GadgetDeviceConfig::default();
        let device_desc = compose_device_descriptor(&config);
        let config_desc = [0u8; MAX_CONFIG_DESC_LEN];
        let mut response = [0u8; 64];

        let setup = UsbSetupPacket::new(0x80, 0xFF, 0, 0, 0); // Unknown request
        assert_eq!(
            handle_standard_request(&setup, &device_desc, &config_desc, 9, &mut response),
            Err(UsbError::Stall)
        );
    }
}
