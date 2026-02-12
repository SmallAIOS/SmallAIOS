// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS Peripheral Drivers
//!
//! Platform drivers for I2C, SPI, GPIO, UART, MIPI CSI-2 camera, and I2S audio.
//! All modules are feature-gated — when disabled, zero code is compiled.
//! HAL traits live in `smallaios-kernel::hal`; this crate provides concrete
//! implementations for ARM, RISC-V, FPGA, and x86 platforms.

#![no_std]

extern crate alloc;

pub mod discovery;

#[cfg(feature = "i2c")]
pub mod i2c;

#[cfg(feature = "spi")]
pub mod spi;

#[cfg(feature = "gpio")]
pub mod gpio;

#[cfg(feature = "uart")]
pub mod uart;

#[cfg(feature = "camera-csi")]
pub mod camera;

#[cfg(feature = "audio-i2s")]
pub mod audio;

/// Peripheral device type identifiers used for device enumeration and capability checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PeripheralType {
    /// I2C bus controller.
    I2c = 0,
    /// SPI bus controller.
    Spi = 1,
    /// GPIO controller.
    Gpio = 2,
    /// UART serial port.
    Uart = 3,
    /// MIPI CSI-2 camera.
    Camera = 4,
    /// I2S/TDM audio.
    Audio = 5,
}

impl PeripheralType {
    /// Convert from u8.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::I2c),
            1 => Some(Self::Spi),
            2 => Some(Self::Gpio),
            3 => Some(Self::Uart),
            4 => Some(Self::Camera),
            5 => Some(Self::Audio),
            _ => None,
        }
    }
}

/// Peripheral device descriptor returned by device enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PeripheralDescriptor {
    /// Device type.
    pub peripheral_type: PeripheralType,
    /// Instance index (e.g., I2C bus 0, UART port 1).
    pub instance: u8,
    /// MMIO base address.
    pub base_addr: u64,
    /// IRQ number.
    pub irq: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_peripheral_type_from_u8() {
        assert_eq!(PeripheralType::from_u8(0), Some(PeripheralType::I2c));
        assert_eq!(PeripheralType::from_u8(1), Some(PeripheralType::Spi));
        assert_eq!(PeripheralType::from_u8(2), Some(PeripheralType::Gpio));
        assert_eq!(PeripheralType::from_u8(3), Some(PeripheralType::Uart));
        assert_eq!(PeripheralType::from_u8(4), Some(PeripheralType::Camera));
        assert_eq!(PeripheralType::from_u8(5), Some(PeripheralType::Audio));
        assert_eq!(PeripheralType::from_u8(6), None);
    }

    #[test]
    fn test_peripheral_descriptor() {
        let desc = PeripheralDescriptor {
            peripheral_type: PeripheralType::I2c,
            instance: 0,
            base_addr: 0x4000_5400,
            irq: 31,
        };
        assert_eq!(desc.peripheral_type, PeripheralType::I2c);
        assert_eq!(desc.instance, 0);
    }
}
