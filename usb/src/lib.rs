// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS USB stack.
//!
//! Provides a `#![no_std]` USB implementation for SmallAIOS, including:
//!
//! - **Core protocol**: descriptor parsing, device enumeration, transfer management
//! - **xHCI host controller** (feature `xhci`): PCIe-based USB 3.x host driver
//! - **Gadget framework** (feature `gadget`): USB device-mode support
//! - **Inference gadget** (feature `inference-gadget`): USB inference appliance

#![no_std]
extern crate alloc;

use core::fmt;

pub mod descriptor;
pub mod enumeration;
pub mod registry;
pub mod transfer;

#[cfg(feature = "xhci")]
pub mod xhci;

#[cfg(feature = "gadget")]
pub mod gadget;

#[cfg(feature = "inference-gadget")]
pub mod inference_gadget;

// ---------------------------------------------------------------------------
// USB constants
// ---------------------------------------------------------------------------

/// USB standard request types (bmRequestType direction bit).
pub mod request_type {
    /// Host-to-device direction.
    pub const DIR_OUT: u8 = 0x00;
    /// Device-to-host direction.
    pub const DIR_IN: u8 = 0x80;

    /// Standard request type.
    pub const TYPE_STANDARD: u8 = 0x00;
    /// Class request type.
    pub const TYPE_CLASS: u8 = 0x20;
    /// Vendor request type.
    pub const TYPE_VENDOR: u8 = 0x40;

    /// Device recipient.
    pub const RECIP_DEVICE: u8 = 0x00;
    /// Interface recipient.
    pub const RECIP_INTERFACE: u8 = 0x01;
    /// Endpoint recipient.
    pub const RECIP_ENDPOINT: u8 = 0x02;
}

/// USB standard request codes (bRequest).
pub mod request {
    pub const GET_STATUS: u8 = 0x00;
    pub const CLEAR_FEATURE: u8 = 0x01;
    pub const SET_FEATURE: u8 = 0x03;
    pub const SET_ADDRESS: u8 = 0x05;
    pub const GET_DESCRIPTOR: u8 = 0x06;
    pub const SET_DESCRIPTOR: u8 = 0x07;
    pub const GET_CONFIGURATION: u8 = 0x08;
    pub const SET_CONFIGURATION: u8 = 0x09;
}

/// USB descriptor type codes.
pub mod descriptor_type {
    pub const DEVICE: u8 = 0x01;
    pub const CONFIGURATION: u8 = 0x02;
    pub const STRING: u8 = 0x03;
    pub const INTERFACE: u8 = 0x04;
    pub const ENDPOINT: u8 = 0x05;
    pub const INTERFACE_ASSOCIATION: u8 = 0x0B;
}

/// USB endpoint feature selectors.
pub mod feature {
    pub const ENDPOINT_HALT: u16 = 0x00;
}

// ---------------------------------------------------------------------------
// USB error types
// ---------------------------------------------------------------------------

/// Errors returned by the USB stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UsbError {
    /// Descriptor data is too short or truncated.
    DescriptorTooShort,
    /// Descriptor type field does not match expected type.
    InvalidDescriptorType,
    /// Descriptor bLength field is invalid.
    InvalidDescriptorLength,
    /// Device did not respond within the timeout period.
    Timeout,
    /// Device returned a STALL handshake.
    Stall,
    /// Transfer failed (CRC error, babble, etc.).
    TransferError,
    /// No free device address or slot available.
    NoFreeSlot,
    /// Device not found in the registry.
    DeviceNotFound,
    /// Interface already claimed by another driver.
    InterfaceAlreadyClaimed,
    /// Endpoint is in the halted state.
    EndpointHalted,
    /// Buffer too small for the requested operation.
    BufferTooSmall,
    /// Invalid endpoint address.
    InvalidEndpoint,
    /// Maximum number of devices reached.
    TooManyDevices,
    /// Hardware (HAL) error.
    HalError(smallaios_kernel::hal::HalError),
}

impl fmt::Display for UsbError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UsbError::DescriptorTooShort => write!(f, "descriptor too short"),
            UsbError::InvalidDescriptorType => write!(f, "invalid descriptor type"),
            UsbError::InvalidDescriptorLength => write!(f, "invalid descriptor length"),
            UsbError::Timeout => write!(f, "device timeout"),
            UsbError::Stall => write!(f, "endpoint stall"),
            UsbError::TransferError => write!(f, "transfer error"),
            UsbError::NoFreeSlot => write!(f, "no free device slot"),
            UsbError::DeviceNotFound => write!(f, "device not found"),
            UsbError::InterfaceAlreadyClaimed => write!(f, "interface already claimed"),
            UsbError::EndpointHalted => write!(f, "endpoint halted"),
            UsbError::BufferTooSmall => write!(f, "buffer too small"),
            UsbError::InvalidEndpoint => write!(f, "invalid endpoint"),
            UsbError::TooManyDevices => write!(f, "too many devices"),
            UsbError::HalError(e) => write!(f, "HAL error: {e}"),
        }
    }
}

impl From<smallaios_kernel::hal::HalError> for UsbError {
    fn from(e: smallaios_kernel::hal::HalError) -> Self {
        UsbError::HalError(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn test_usb_error_display_all_variants() {
        // Test every UsbError Display variant for full coverage.
        assert_eq!(
            format!("{}", UsbError::DescriptorTooShort),
            "descriptor too short"
        );
        assert_eq!(
            format!("{}", UsbError::InvalidDescriptorType),
            "invalid descriptor type"
        );
        assert_eq!(
            format!("{}", UsbError::InvalidDescriptorLength),
            "invalid descriptor length"
        );
        assert_eq!(format!("{}", UsbError::Timeout), "device timeout");
        assert_eq!(format!("{}", UsbError::Stall), "endpoint stall");
        assert_eq!(format!("{}", UsbError::TransferError), "transfer error");
        assert_eq!(format!("{}", UsbError::NoFreeSlot), "no free device slot");
        assert_eq!(format!("{}", UsbError::DeviceNotFound), "device not found");
        assert_eq!(
            format!("{}", UsbError::InterfaceAlreadyClaimed),
            "interface already claimed"
        );
        assert_eq!(format!("{}", UsbError::EndpointHalted), "endpoint halted");
        assert_eq!(format!("{}", UsbError::BufferTooSmall), "buffer too small");
        assert_eq!(format!("{}", UsbError::InvalidEndpoint), "invalid endpoint");
        assert_eq!(format!("{}", UsbError::TooManyDevices), "too many devices");
        assert_eq!(
            format!(
                "{}",
                UsbError::HalError(smallaios_kernel::hal::HalError::Timeout)
            ),
            "HAL error: timeout"
        );
    }

    #[test]
    fn test_usb_error_from_hal() {
        let hal_err = smallaios_kernel::hal::HalError::Timeout;
        let usb_err: UsbError = hal_err.into();
        assert_eq!(
            usb_err,
            UsbError::HalError(smallaios_kernel::hal::HalError::Timeout)
        );
    }

    #[test]
    fn test_usb_error_from_hal_all_variants() {
        // Verify From<HalError> works for several HalError variants.
        let variants = [
            smallaios_kernel::hal::HalError::InitFailed,
            smallaios_kernel::hal::HalError::UsbTransferError,
            smallaios_kernel::hal::HalError::UsbStall,
            smallaios_kernel::hal::HalError::UsbDeviceNotFound,
            smallaios_kernel::hal::HalError::UsbEndpointHalted,
            smallaios_kernel::hal::HalError::HardwareError,
        ];
        for hal_err in variants {
            let usb_err: UsbError = hal_err.into();
            assert_eq!(usb_err, UsbError::HalError(hal_err));
        }
    }

    #[test]
    fn test_usb_error_debug() {
        // Ensure Debug impl works (derived).
        let err = UsbError::Timeout;
        let debug_str = format!("{:?}", err);
        assert!(debug_str.contains("Timeout"));
    }

    #[test]
    fn test_usb_error_clone_copy_eq() {
        let err1 = UsbError::Stall;
        let err2 = err1; // Copy
        let err3 = err1; // Copy (also tests Clone since Copy implies Clone)
        assert_eq!(err1, err2);
        assert_eq!(err1, err3);
        assert_ne!(UsbError::Stall, UsbError::Timeout);
    }

    #[test]
    fn test_request_type_constants() {
        assert_eq!(
            request_type::DIR_IN | request_type::TYPE_STANDARD | request_type::RECIP_DEVICE,
            0x80
        );
        assert_eq!(
            request_type::DIR_OUT | request_type::TYPE_VENDOR | request_type::RECIP_DEVICE,
            0x40
        );
    }

    #[test]
    fn test_request_type_all_constants() {
        assert_eq!(request_type::DIR_OUT, 0x00);
        assert_eq!(request_type::DIR_IN, 0x80);
        assert_eq!(request_type::TYPE_STANDARD, 0x00);
        assert_eq!(request_type::TYPE_CLASS, 0x20);
        assert_eq!(request_type::TYPE_VENDOR, 0x40);
        assert_eq!(request_type::RECIP_DEVICE, 0x00);
        assert_eq!(request_type::RECIP_INTERFACE, 0x01);
        assert_eq!(request_type::RECIP_ENDPOINT, 0x02);
    }

    #[test]
    fn test_request_type_combinations() {
        // Class request to interface (e.g. HID SET_REPORT)
        assert_eq!(
            request_type::DIR_OUT | request_type::TYPE_CLASS | request_type::RECIP_INTERFACE,
            0x21
        );
        // Vendor IN from endpoint
        assert_eq!(
            request_type::DIR_IN | request_type::TYPE_VENDOR | request_type::RECIP_ENDPOINT,
            0xC2
        );
    }

    #[test]
    fn test_request_constants() {
        assert_eq!(request::GET_STATUS, 0x00);
        assert_eq!(request::CLEAR_FEATURE, 0x01);
        assert_eq!(request::SET_FEATURE, 0x03);
        assert_eq!(request::SET_ADDRESS, 0x05);
        assert_eq!(request::GET_DESCRIPTOR, 0x06);
        assert_eq!(request::SET_DESCRIPTOR, 0x07);
        assert_eq!(request::GET_CONFIGURATION, 0x08);
        assert_eq!(request::SET_CONFIGURATION, 0x09);
    }

    #[test]
    fn test_descriptor_type_constants() {
        assert_eq!(descriptor_type::DEVICE, 1);
        assert_eq!(descriptor_type::CONFIGURATION, 2);
        assert_eq!(descriptor_type::STRING, 3);
        assert_eq!(descriptor_type::INTERFACE, 4);
        assert_eq!(descriptor_type::ENDPOINT, 5);
        assert_eq!(descriptor_type::INTERFACE_ASSOCIATION, 0x0B);
    }

    #[test]
    fn test_feature_constants() {
        assert_eq!(feature::ENDPOINT_HALT, 0x00);
    }
}
