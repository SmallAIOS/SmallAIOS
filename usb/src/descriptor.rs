// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! USB descriptor parsing — zero-copy views over raw byte buffers.
//!
//! Supports device, configuration, interface, endpoint, and string descriptors
//! per USB 2.0 specification chapter 9.

use crate::UsbError;

// ---------------------------------------------------------------------------
// Descriptor type constants
// ---------------------------------------------------------------------------

/// Expected length of a USB device descriptor.
pub const DEVICE_DESCRIPTOR_LEN: usize = 18;
/// Minimum length of a USB configuration descriptor header.
pub const CONFIG_DESCRIPTOR_MIN_LEN: usize = 9;
/// Expected length of a USB interface descriptor.
pub const INTERFACE_DESCRIPTOR_LEN: usize = 9;
/// Expected length of a USB endpoint descriptor.
pub const ENDPOINT_DESCRIPTOR_LEN: usize = 7;
/// Minimum length of a USB string descriptor.
pub const STRING_DESCRIPTOR_MIN_LEN: usize = 2;

// ---------------------------------------------------------------------------
// Device Descriptor
// ---------------------------------------------------------------------------

/// Parsed USB device descriptor (18 bytes, USB 2.0 Table 9-8).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceDescriptor {
    /// USB specification release number (BCD, e.g. 0x0200 = USB 2.0).
    pub bcd_usb: u16,
    /// Device class code.
    pub device_class: u8,
    /// Device subclass code.
    pub device_sub_class: u8,
    /// Device protocol code.
    pub device_protocol: u8,
    /// Maximum packet size for EP0 (8, 16, 32, or 64).
    pub max_packet_size_0: u8,
    /// Vendor ID.
    pub vendor_id: u16,
    /// Product ID.
    pub product_id: u16,
    /// Device release number (BCD).
    pub bcd_device: u16,
    /// Index of manufacturer string descriptor.
    pub manufacturer_index: u8,
    /// Index of product string descriptor.
    pub product_index: u8,
    /// Index of serial number string descriptor.
    pub serial_number_index: u8,
    /// Number of possible configurations.
    pub num_configurations: u8,
}

impl DeviceDescriptor {
    /// Parse a device descriptor from a raw byte buffer.
    pub fn parse(data: &[u8]) -> Result<Self, UsbError> {
        if data.len() < DEVICE_DESCRIPTOR_LEN {
            return Err(UsbError::DescriptorTooShort);
        }
        let b_length = data[0];
        if b_length != DEVICE_DESCRIPTOR_LEN as u8 {
            return Err(UsbError::InvalidDescriptorLength);
        }
        let b_descriptor_type = data[1];
        if b_descriptor_type != crate::descriptor_type::DEVICE {
            return Err(UsbError::InvalidDescriptorType);
        }
        Ok(Self {
            bcd_usb: u16::from_le_bytes([data[2], data[3]]),
            device_class: data[4],
            device_sub_class: data[5],
            device_protocol: data[6],
            max_packet_size_0: data[7],
            vendor_id: u16::from_le_bytes([data[8], data[9]]),
            product_id: u16::from_le_bytes([data[10], data[11]]),
            bcd_device: u16::from_le_bytes([data[12], data[13]]),
            manufacturer_index: data[14],
            product_index: data[15],
            serial_number_index: data[16],
            num_configurations: data[17],
        })
    }

    /// Returns true if the device uses vendor-specific class (0xFF).
    pub fn is_vendor_specific(&self) -> bool {
        self.device_class == 0xFF
    }
}

// ---------------------------------------------------------------------------
// Configuration Descriptor
// ---------------------------------------------------------------------------

/// Parsed USB configuration descriptor header (9 bytes, USB 2.0 Table 9-10).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConfigDescriptor {
    /// Total length of data returned for this configuration.
    pub total_length: u16,
    /// Number of interfaces supported by this configuration.
    pub num_interfaces: u8,
    /// Value to use as argument for SET_CONFIGURATION.
    pub configuration_value: u8,
    /// Index of string descriptor describing this configuration.
    pub configuration_index: u8,
    /// Configuration attributes (bit 6 = self-powered, bit 5 = remote wakeup).
    pub attributes: u8,
    /// Maximum power consumption in 2mA units.
    pub max_power: u8,
}

impl ConfigDescriptor {
    /// Parse a configuration descriptor header from a raw byte buffer.
    pub fn parse(data: &[u8]) -> Result<Self, UsbError> {
        if data.len() < CONFIG_DESCRIPTOR_MIN_LEN {
            return Err(UsbError::DescriptorTooShort);
        }
        let b_length = data[0];
        if b_length < CONFIG_DESCRIPTOR_MIN_LEN as u8 {
            return Err(UsbError::InvalidDescriptorLength);
        }
        let b_descriptor_type = data[1];
        if b_descriptor_type != crate::descriptor_type::CONFIGURATION {
            return Err(UsbError::InvalidDescriptorType);
        }
        Ok(Self {
            total_length: u16::from_le_bytes([data[2], data[3]]),
            num_interfaces: data[4],
            configuration_value: data[5],
            configuration_index: data[6],
            attributes: data[7],
            max_power: data[8],
        })
    }
}

// ---------------------------------------------------------------------------
// Interface Descriptor
// ---------------------------------------------------------------------------

/// Parsed USB interface descriptor (9 bytes, USB 2.0 Table 9-12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterfaceDescriptor {
    /// Number of this interface (0-based).
    pub interface_number: u8,
    /// Alternate setting value.
    pub alternate_setting: u8,
    /// Number of endpoints used by this interface (excluding EP0).
    pub num_endpoints: u8,
    /// Interface class code.
    pub interface_class: u8,
    /// Interface subclass code.
    pub interface_sub_class: u8,
    /// Interface protocol code.
    pub interface_protocol: u8,
    /// Index of string descriptor describing this interface.
    pub interface_index: u8,
}

impl InterfaceDescriptor {
    /// Parse an interface descriptor from a raw byte buffer.
    pub fn parse(data: &[u8]) -> Result<Self, UsbError> {
        if data.len() < INTERFACE_DESCRIPTOR_LEN {
            return Err(UsbError::DescriptorTooShort);
        }
        let b_length = data[0];
        if b_length < INTERFACE_DESCRIPTOR_LEN as u8 {
            return Err(UsbError::InvalidDescriptorLength);
        }
        let b_descriptor_type = data[1];
        if b_descriptor_type != crate::descriptor_type::INTERFACE {
            return Err(UsbError::InvalidDescriptorType);
        }
        Ok(Self {
            interface_number: data[2],
            alternate_setting: data[3],
            num_endpoints: data[4],
            interface_class: data[5],
            interface_sub_class: data[6],
            interface_protocol: data[7],
            interface_index: data[8],
        })
    }

    /// Returns true if this is a vendor-specific interface (class 0xFF).
    pub fn is_vendor_specific(&self) -> bool {
        self.interface_class == 0xFF
    }
}

// ---------------------------------------------------------------------------
// Endpoint Descriptor
// ---------------------------------------------------------------------------

/// Parsed USB endpoint descriptor (7 bytes, USB 2.0 Table 9-13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EndpointDescriptor {
    /// Endpoint address (bits 0-3 = number, bit 7 = direction: 1=IN, 0=OUT).
    pub endpoint_address: u8,
    /// Attributes (bits 0-1 = transfer type).
    pub attributes: u8,
    /// Maximum packet size this endpoint can send/receive.
    pub max_packet_size: u16,
    /// Polling interval (in frames/microframes).
    pub interval: u8,
}

impl EndpointDescriptor {
    /// Parse an endpoint descriptor from a raw byte buffer.
    pub fn parse(data: &[u8]) -> Result<Self, UsbError> {
        if data.len() < ENDPOINT_DESCRIPTOR_LEN {
            return Err(UsbError::DescriptorTooShort);
        }
        let b_length = data[0];
        if b_length < ENDPOINT_DESCRIPTOR_LEN as u8 {
            return Err(UsbError::InvalidDescriptorLength);
        }
        let b_descriptor_type = data[1];
        if b_descriptor_type != crate::descriptor_type::ENDPOINT {
            return Err(UsbError::InvalidDescriptorType);
        }
        Ok(Self {
            endpoint_address: data[2],
            attributes: data[3],
            max_packet_size: u16::from_le_bytes([data[4], data[5]]),
            interval: data[6],
        })
    }

    /// Endpoint number (0-15).
    pub fn number(&self) -> u8 {
        self.endpoint_address & 0x0F
    }

    /// Returns true if this is an IN endpoint (device-to-host).
    pub fn is_in(&self) -> bool {
        self.endpoint_address & 0x80 != 0
    }

    /// Returns true if this is an OUT endpoint (host-to-device).
    pub fn is_out(&self) -> bool {
        self.endpoint_address & 0x80 == 0
    }

    /// Transfer type from attributes field.
    pub fn transfer_type(&self) -> smallaios_kernel::hal::UsbTransferType {
        match self.attributes & 0x03 {
            0 => smallaios_kernel::hal::UsbTransferType::Control,
            1 => smallaios_kernel::hal::UsbTransferType::Isochronous,
            2 => smallaios_kernel::hal::UsbTransferType::Bulk,
            3 => smallaios_kernel::hal::UsbTransferType::Interrupt,
            _ => unreachable!(),
        }
    }
}

// ---------------------------------------------------------------------------
// String Descriptor
// ---------------------------------------------------------------------------

/// Parse a USB string descriptor (UTF-16LE) into a fixed-size buffer.
///
/// Returns the number of valid UTF-8 bytes written to `out`.
pub fn parse_string_descriptor(data: &[u8], out: &mut [u8]) -> Result<usize, UsbError> {
    if data.len() < STRING_DESCRIPTOR_MIN_LEN {
        return Err(UsbError::DescriptorTooShort);
    }
    let b_length = data[0] as usize;
    if b_length < 2 || b_length > data.len() {
        return Err(UsbError::InvalidDescriptorLength);
    }
    if data[1] != crate::descriptor_type::STRING {
        return Err(UsbError::InvalidDescriptorType);
    }

    // String data starts at offset 2, encoded as UTF-16LE.
    let utf16_bytes = &data[2..b_length];
    let mut written = 0usize;

    let mut i = 0;
    while i + 1 < utf16_bytes.len() {
        let code_unit = u16::from_le_bytes([utf16_bytes[i], utf16_bytes[i + 1]]);
        i += 2;

        // Simple BMP-only UTF-16 to UTF-8 conversion.
        if code_unit < 0x80 {
            if written >= out.len() {
                break;
            }
            out[written] = code_unit as u8;
            written += 1;
        } else if code_unit < 0x800 {
            if written + 1 >= out.len() {
                break;
            }
            out[written] = 0xC0 | ((code_unit >> 6) as u8);
            out[written + 1] = 0x80 | ((code_unit & 0x3F) as u8);
            written += 2;
        } else {
            if written + 2 >= out.len() {
                break;
            }
            out[written] = 0xE0 | ((code_unit >> 12) as u8);
            out[written + 1] = 0x80 | (((code_unit >> 6) & 0x3F) as u8);
            out[written + 2] = 0x80 | ((code_unit & 0x3F) as u8);
            written += 3;
        }
    }

    Ok(written)
}

// ---------------------------------------------------------------------------
// Configuration descriptor chain iterator
// ---------------------------------------------------------------------------

/// Iterator over descriptors in a configuration descriptor chain.
///
/// Walks the descriptor chain using bLength to advance, yielding raw slices
/// for each sub-descriptor (interface, endpoint, etc.).
pub struct DescriptorIter<'a> {
    data: &'a [u8],
    offset: usize,
}

impl<'a> DescriptorIter<'a> {
    /// Create a new iterator starting after the configuration descriptor header.
    pub fn new(config_data: &'a [u8]) -> Self {
        // Skip the config descriptor header itself.
        let start = if config_data.len() >= CONFIG_DESCRIPTOR_MIN_LEN {
            config_data[0] as usize
        } else {
            config_data.len()
        };
        Self {
            data: config_data,
            offset: start,
        }
    }
}

impl<'a> Iterator for DescriptorIter<'a> {
    type Item = &'a [u8];

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.data.len() {
            return None;
        }
        // bLength is the first byte.
        let remaining = &self.data[self.offset..];
        if remaining.is_empty() {
            return None;
        }
        let b_length = remaining[0] as usize;
        if b_length < 2 || self.offset + b_length > self.data.len() {
            return None; // Malformed: stop iteration.
        }
        let desc = &self.data[self.offset..self.offset + b_length];
        self.offset += b_length;
        Some(desc)
    }
}

// ---------------------------------------------------------------------------
// Helper: find interfaces matching class/subclass/protocol
// ---------------------------------------------------------------------------

/// Search a configuration descriptor chain for interfaces matching the given
/// class, subclass, and protocol. Returns matching `InterfaceDescriptor`s.
pub fn find_interfaces(
    config_data: &[u8],
    class: u8,
    subclass: u8,
    protocol: u8,
    out: &mut [InterfaceDescriptor],
) -> usize {
    let mut count = 0;
    for desc_bytes in DescriptorIter::new(config_data) {
        if desc_bytes.len() >= INTERFACE_DESCRIPTOR_LEN
            && desc_bytes[1] == crate::descriptor_type::INTERFACE
        {
            if let Ok(iface) = InterfaceDescriptor::parse(desc_bytes) {
                if iface.interface_class == class
                    && iface.interface_sub_class == subclass
                    && iface.interface_protocol == protocol
                    && count < out.len()
                {
                    out[count] = iface;
                    count += 1;
                }
            }
        }
    }
    count
}

/// Collect endpoint descriptors belonging to a specific interface.
///
/// Walks the descriptor chain from the start, finds the matching interface
/// descriptor, then collects subsequent endpoint descriptors until the next
/// interface descriptor or end of chain.
pub fn find_endpoints_for_interface(
    config_data: &[u8],
    interface_number: u8,
    out: &mut [EndpointDescriptor],
) -> usize {
    let mut count = 0;
    let mut in_target_interface = false;

    for desc_bytes in DescriptorIter::new(config_data) {
        let desc_type = if desc_bytes.len() >= 2 {
            desc_bytes[1]
        } else {
            continue;
        };

        if desc_type == crate::descriptor_type::INTERFACE {
            if let Ok(iface) = InterfaceDescriptor::parse(desc_bytes) {
                in_target_interface = iface.interface_number == interface_number;
            }
        } else if desc_type == crate::descriptor_type::ENDPOINT && in_target_interface {
            if let Ok(ep) = EndpointDescriptor::parse(desc_bytes) {
                if count < out.len() {
                    out[count] = ep;
                    count += 1;
                }
            }
        }
    }
    count
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Device Descriptor --------------------------------------------------

    /// Build a valid USB device descriptor byte array.
    fn make_device_desc(vid: u16, pid: u16, class: u8) -> [u8; 18] {
        let mut d = [0u8; 18];
        d[0] = 18; // bLength
        d[1] = crate::descriptor_type::DEVICE;
        d[2] = 0x00;
        d[3] = 0x02; // bcdUSB = 2.0
        d[4] = class;
        d[7] = 64; // bMaxPacketSize0
        d[8] = (vid & 0xFF) as u8;
        d[9] = (vid >> 8) as u8;
        d[10] = (pid & 0xFF) as u8;
        d[11] = (pid >> 8) as u8;
        d[12] = 0x00;
        d[13] = 0x01; // bcdDevice = 1.0
        d[14] = 1; // iManufacturer
        d[15] = 2; // iProduct
        d[16] = 3; // iSerialNumber
        d[17] = 1; // bNumConfigurations
        d
    }

    #[test]
    fn test_parse_device_descriptor_valid() {
        let data = make_device_desc(0x1D50, 0x6089, 0xFF);
        let desc = DeviceDescriptor::parse(&data).unwrap();
        assert_eq!(desc.vendor_id, 0x1D50);
        assert_eq!(desc.product_id, 0x6089);
        assert_eq!(desc.device_class, 0xFF);
        assert!(desc.is_vendor_specific());
        assert_eq!(desc.bcd_usb, 0x0200);
        assert_eq!(desc.max_packet_size_0, 64);
        assert_eq!(desc.num_configurations, 1);
    }

    #[test]
    fn test_parse_device_descriptor_truncated() {
        let data = [0u8; 10];
        assert_eq!(
            DeviceDescriptor::parse(&data),
            Err(UsbError::DescriptorTooShort)
        );
    }

    #[test]
    fn test_parse_device_descriptor_bad_length() {
        let mut data = make_device_desc(0, 0, 0);
        data[0] = 12; // Wrong bLength
        assert_eq!(
            DeviceDescriptor::parse(&data),
            Err(UsbError::InvalidDescriptorLength)
        );
    }

    #[test]
    fn test_parse_device_descriptor_bad_type() {
        let mut data = make_device_desc(0, 0, 0);
        data[1] = 0x02; // Configuration instead of Device
        assert_eq!(
            DeviceDescriptor::parse(&data),
            Err(UsbError::InvalidDescriptorType)
        );
    }

    #[test]
    fn test_device_descriptor_not_vendor_specific() {
        let data = make_device_desc(0x0456, 0xb673, 0xEF);
        let desc = DeviceDescriptor::parse(&data).unwrap();
        assert!(!desc.is_vendor_specific());
    }

    // -- Configuration Descriptor -------------------------------------------

    fn make_config_desc(total_len: u16, num_ifaces: u8) -> [u8; 9] {
        let mut d = [0u8; 9];
        d[0] = 9;
        d[1] = crate::descriptor_type::CONFIGURATION;
        d[2] = (total_len & 0xFF) as u8;
        d[3] = (total_len >> 8) as u8;
        d[4] = num_ifaces;
        d[5] = 1; // bConfigurationValue
        d[8] = 250; // MaxPower = 500mA
        d
    }

    #[test]
    fn test_parse_config_descriptor_valid() {
        let data = make_config_desc(32, 2);
        let desc = ConfigDescriptor::parse(&data).unwrap();
        assert_eq!(desc.total_length, 32);
        assert_eq!(desc.num_interfaces, 2);
        assert_eq!(desc.configuration_value, 1);
        assert_eq!(desc.max_power, 250);
    }

    #[test]
    fn test_parse_config_descriptor_truncated() {
        let data = [0u8; 5];
        assert_eq!(
            ConfigDescriptor::parse(&data),
            Err(UsbError::DescriptorTooShort)
        );
    }

    // -- Interface Descriptor -----------------------------------------------

    fn make_interface_desc(num: u8, class: u8, sub: u8, proto: u8, num_ep: u8) -> [u8; 9] {
        let mut d = [0u8; 9];
        d[0] = 9;
        d[1] = crate::descriptor_type::INTERFACE;
        d[2] = num;
        d[4] = num_ep;
        d[5] = class;
        d[6] = sub;
        d[7] = proto;
        d
    }

    #[test]
    fn test_parse_interface_descriptor_valid() {
        let data = make_interface_desc(0, 0xFF, 0x01, 0x02, 2);
        let desc = InterfaceDescriptor::parse(&data).unwrap();
        assert_eq!(desc.interface_number, 0);
        assert_eq!(desc.interface_class, 0xFF);
        assert_eq!(desc.interface_sub_class, 0x01);
        assert_eq!(desc.interface_protocol, 0x02);
        assert_eq!(desc.num_endpoints, 2);
        assert!(desc.is_vendor_specific());
    }

    #[test]
    fn test_parse_interface_descriptor_truncated() {
        let data = [0u8; 5];
        assert_eq!(
            InterfaceDescriptor::parse(&data),
            Err(UsbError::DescriptorTooShort)
        );
    }

    // -- Endpoint Descriptor ------------------------------------------------

    fn make_endpoint_desc(addr: u8, attrs: u8, max_pkt: u16) -> [u8; 7] {
        let mut d = [0u8; 7];
        d[0] = 7;
        d[1] = crate::descriptor_type::ENDPOINT;
        d[2] = addr;
        d[3] = attrs;
        d[4] = (max_pkt & 0xFF) as u8;
        d[5] = (max_pkt >> 8) as u8;
        d[6] = 0; // interval
        d
    }

    #[test]
    fn test_parse_endpoint_descriptor_bulk_in() {
        let data = make_endpoint_desc(0x81, 0x02, 512);
        let desc = EndpointDescriptor::parse(&data).unwrap();
        assert_eq!(desc.endpoint_address, 0x81);
        assert!(desc.is_in());
        assert!(!desc.is_out());
        assert_eq!(desc.number(), 1);
        assert_eq!(desc.max_packet_size, 512);
        assert_eq!(
            desc.transfer_type(),
            smallaios_kernel::hal::UsbTransferType::Bulk
        );
    }

    #[test]
    fn test_parse_endpoint_descriptor_bulk_out() {
        let data = make_endpoint_desc(0x02, 0x02, 512);
        let desc = EndpointDescriptor::parse(&data).unwrap();
        assert!(desc.is_out());
        assert!(!desc.is_in());
        assert_eq!(desc.number(), 2);
    }

    #[test]
    fn test_parse_endpoint_descriptor_interrupt() {
        let data = make_endpoint_desc(0x83, 0x03, 8);
        let desc = EndpointDescriptor::parse(&data).unwrap();
        assert_eq!(
            desc.transfer_type(),
            smallaios_kernel::hal::UsbTransferType::Interrupt
        );
    }

    #[test]
    fn test_parse_endpoint_descriptor_truncated() {
        let data = [0u8; 4];
        assert_eq!(
            EndpointDescriptor::parse(&data),
            Err(UsbError::DescriptorTooShort)
        );
    }

    // -- String Descriptor --------------------------------------------------

    #[test]
    fn test_parse_string_descriptor_ascii() {
        // "Hi" in UTF-16LE: H=0x0048, i=0x0069
        let data = [6, crate::descriptor_type::STRING, 0x48, 0x00, 0x69, 0x00];
        let mut out = [0u8; 32];
        let len = parse_string_descriptor(&data, &mut out).unwrap();
        assert_eq!(len, 2);
        assert_eq!(&out[..len], b"Hi");
    }

    #[test]
    fn test_parse_string_descriptor_empty() {
        let data = [2, crate::descriptor_type::STRING];
        let mut out = [0u8; 32];
        let len = parse_string_descriptor(&data, &mut out).unwrap();
        assert_eq!(len, 0);
    }

    #[test]
    fn test_parse_string_descriptor_truncated() {
        let data = [1];
        let mut out = [0u8; 32];
        assert_eq!(
            parse_string_descriptor(&data, &mut out),
            Err(UsbError::DescriptorTooShort)
        );
    }

    #[test]
    fn test_parse_string_descriptor_bad_type() {
        let data = [4, 0x01, 0x48, 0x00]; // type = DEVICE instead of STRING
        let mut out = [0u8; 32];
        assert_eq!(
            parse_string_descriptor(&data, &mut out),
            Err(UsbError::InvalidDescriptorType)
        );
    }

    // -- Descriptor Iterator ------------------------------------------------

    #[test]
    fn test_descriptor_iter_composite() {
        // Config(9) + Interface(9) + Endpoint(7) + Endpoint(7) = 32 bytes total
        let mut data = [0u8; 32];
        // Config header
        data[0] = 9;
        data[1] = crate::descriptor_type::CONFIGURATION;
        data[2] = 32;
        data[3] = 0; // total length
        data[4] = 1;
        data[5] = 1;
        // Interface
        data[9] = 9;
        data[10] = crate::descriptor_type::INTERFACE;
        data[11] = 0;
        data[13] = 2;
        data[14] = 0xFF;
        // Endpoint 1 (bulk IN)
        data[18] = 7;
        data[19] = crate::descriptor_type::ENDPOINT;
        data[20] = 0x81;
        data[21] = 0x02;
        data[22] = 0x00;
        data[23] = 0x02;
        // Endpoint 2 (bulk OUT)
        data[25] = 7;
        data[26] = crate::descriptor_type::ENDPOINT;
        data[27] = 0x02;
        data[28] = 0x02;
        data[29] = 0x00;
        data[30] = 0x02;

        let descs: alloc::vec::Vec<&[u8]> = DescriptorIter::new(&data).collect();
        assert_eq!(descs.len(), 3); // interface + 2 endpoints
        assert_eq!(descs[0][1], crate::descriptor_type::INTERFACE);
        assert_eq!(descs[1][1], crate::descriptor_type::ENDPOINT);
        assert_eq!(descs[2][1], crate::descriptor_type::ENDPOINT);
    }

    // -- find_interfaces / find_endpoints_for_interface ----------------------

    #[test]
    fn test_find_interfaces() {
        let mut data = [0u8; 27];
        // Config header
        data[0] = 9;
        data[1] = crate::descriptor_type::CONFIGURATION;
        data[2] = 27;
        data[4] = 2;
        // Interface 0: class 0xFF
        data[9] = 9;
        data[10] = crate::descriptor_type::INTERFACE;
        data[11] = 0;
        data[14] = 0xFF;
        data[15] = 0x01;
        data[16] = 0x02;
        // Interface 1: class 0x02 (CDC)
        data[18] = 9;
        data[19] = crate::descriptor_type::INTERFACE;
        data[20] = 1;
        data[23] = 0x02;

        let mut out = [InterfaceDescriptor {
            interface_number: 0,
            alternate_setting: 0,
            num_endpoints: 0,
            interface_class: 0,
            interface_sub_class: 0,
            interface_protocol: 0,
            interface_index: 0,
        }; 4];
        let count = find_interfaces(&data, 0xFF, 0x01, 0x02, &mut out);
        assert_eq!(count, 1);
        assert_eq!(out[0].interface_number, 0);
    }

    #[test]
    fn test_find_endpoints_for_interface() {
        let mut data = [0u8; 32];
        // Config header
        data[0] = 9;
        data[1] = crate::descriptor_type::CONFIGURATION;
        data[2] = 32;
        // Interface 0
        data[9] = 9;
        data[10] = crate::descriptor_type::INTERFACE;
        data[11] = 0;
        data[13] = 2;
        // EP 0x81 bulk IN
        data[18] = 7;
        data[19] = crate::descriptor_type::ENDPOINT;
        data[20] = 0x81;
        data[21] = 0x02;
        data[22] = 0x00;
        data[23] = 0x02;
        // EP 0x02 bulk OUT
        data[25] = 7;
        data[26] = crate::descriptor_type::ENDPOINT;
        data[27] = 0x02;
        data[28] = 0x02;
        data[29] = 0x00;
        data[30] = 0x02;

        let mut out = [EndpointDescriptor {
            endpoint_address: 0,
            attributes: 0,
            max_packet_size: 0,
            interval: 0,
        }; 4];
        let count = find_endpoints_for_interface(&data, 0, &mut out);
        assert_eq!(count, 2);
        assert_eq!(out[0].endpoint_address, 0x81);
        assert_eq!(out[1].endpoint_address, 0x02);
    }
}
