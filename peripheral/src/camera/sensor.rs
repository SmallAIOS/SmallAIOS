// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Camera sensor auto-detection via I2C.
//!
//! Probes known I2C addresses and reads chip ID registers to identify
//! the connected image sensor. Supports OV5640, IMX219, and IMX477.

#![allow(dead_code)]

use smallaios_kernel::hal::{HalError, I2cController};

/// Known camera sensor types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorType {
    /// OmniVision OV5640 (5 MP, parallel/MIPI).
    Ov5640,
    /// Sony IMX219 (8 MP, Raspberry Pi Camera Module v2).
    Imx219,
    /// Sony IMX477 (12.3 MP, Raspberry Pi HQ Camera).
    Imx477,
}

/// Result of sensor detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SensorInfo {
    /// Detected sensor type.
    pub sensor_type: SensorType,
    /// I2C address where the sensor was found.
    pub i2c_addr: u16,
    /// Chip ID read from the sensor.
    pub chip_id: u16,
    /// Maximum resolution width.
    pub max_width: u16,
    /// Maximum resolution height.
    pub max_height: u16,
}

// ---------------------------------------------------------------------------
// Sensor I2C addresses and chip IDs
// ---------------------------------------------------------------------------

/// OV5640 default I2C address (7-bit: 0x3C).
pub const OV5640_I2C_ADDR: u16 = 0x3C;

/// OV5640 chip ID register address (16-bit register).
pub const OV5640_CHIP_ID_HIGH: u16 = 0x300A;
pub const OV5640_CHIP_ID_LOW: u16 = 0x300B;

/// OV5640 expected chip ID.
pub const OV5640_CHIP_ID: u16 = 0x5640;

/// IMX219 default I2C address (7-bit: 0x10).
pub const IMX219_I2C_ADDR: u16 = 0x10;

/// IMX219 chip ID register address (16-bit register).
pub const IMX219_CHIP_ID_HIGH: u16 = 0x0000;
pub const IMX219_CHIP_ID_LOW: u16 = 0x0001;

/// IMX219 expected chip ID.
pub const IMX219_CHIP_ID: u16 = 0x0219;

/// IMX477 default I2C address (7-bit: 0x1A).
pub const IMX477_I2C_ADDR: u16 = 0x1A;

/// IMX477 chip ID register address (16-bit register).
pub const IMX477_CHIP_ID_HIGH: u16 = 0x0016;
pub const IMX477_CHIP_ID_LOW: u16 = 0x0017;

/// IMX477 expected chip ID.
pub const IMX477_CHIP_ID: u16 = 0x0477;

// ---------------------------------------------------------------------------
// I2C register access helpers
// ---------------------------------------------------------------------------

/// Read a single 8-bit register from a 16-bit register address (big-endian addr).
///
/// This is the standard pattern for camera sensors which use 16-bit register
/// addresses: write the 2-byte address, then read 1 byte.
pub fn read_reg8<I: I2cController>(
    i2c: &mut I,
    device_addr: u16,
    reg_addr: u16,
) -> Result<u8, HalError> {
    let addr_bytes = [(reg_addr >> 8) as u8, (reg_addr & 0xFF) as u8];
    let mut buf = [0u8; 1];
    i2c.write_read(device_addr, &addr_bytes, &mut buf)?;
    Ok(buf[0])
}

/// Write a single 8-bit register at a 16-bit register address.
pub fn write_reg8<I: I2cController>(
    i2c: &mut I,
    device_addr: u16,
    reg_addr: u16,
    value: u8,
) -> Result<(), HalError> {
    let data = [(reg_addr >> 8) as u8, (reg_addr & 0xFF) as u8, value];
    i2c.write(device_addr, &data)
}

/// Read a 16-bit chip ID from two consecutive 8-bit registers.
fn read_chip_id<I: I2cController>(
    i2c: &mut I,
    device_addr: u16,
    reg_high: u16,
    reg_low: u16,
) -> Result<u16, HalError> {
    let high = read_reg8(i2c, device_addr, reg_high)?;
    let low = read_reg8(i2c, device_addr, reg_low)?;
    Ok((high as u16) << 8 | low as u16)
}

// ---------------------------------------------------------------------------
// Detection logic
// ---------------------------------------------------------------------------

/// Sensor probe table entry.
struct ProbeEntry {
    sensor_type: SensorType,
    i2c_addr: u16,
    chip_id_high_reg: u16,
    chip_id_low_reg: u16,
    expected_id: u16,
    max_width: u16,
    max_height: u16,
}

/// Table of known sensors and their probe parameters.
const PROBE_TABLE: &[ProbeEntry] = &[
    ProbeEntry {
        sensor_type: SensorType::Ov5640,
        i2c_addr: OV5640_I2C_ADDR,
        chip_id_high_reg: OV5640_CHIP_ID_HIGH,
        chip_id_low_reg: OV5640_CHIP_ID_LOW,
        expected_id: OV5640_CHIP_ID,
        max_width: 2592,
        max_height: 1944,
    },
    ProbeEntry {
        sensor_type: SensorType::Imx219,
        i2c_addr: IMX219_I2C_ADDR,
        chip_id_high_reg: IMX219_CHIP_ID_HIGH,
        chip_id_low_reg: IMX219_CHIP_ID_LOW,
        expected_id: IMX219_CHIP_ID,
        max_width: 3280,
        max_height: 2464,
    },
    ProbeEntry {
        sensor_type: SensorType::Imx477,
        i2c_addr: IMX477_I2C_ADDR,
        chip_id_high_reg: IMX477_CHIP_ID_HIGH,
        chip_id_low_reg: IMX477_CHIP_ID_LOW,
        expected_id: IMX477_CHIP_ID,
        max_width: 4056,
        max_height: 3040,
    },
];

/// Attempt to detect a connected camera sensor by probing known I2C addresses.
///
/// Iterates through the probe table and reads the chip ID register at each
/// known address. Returns information about the first sensor found.
///
/// Returns `Err(HalError::HardwareError)` if no sensor is detected.
pub fn detect_sensor<I: I2cController>(i2c: &mut I) -> Result<SensorInfo, HalError> {
    for entry in PROBE_TABLE.iter() {
        if let Ok(chip_id) = read_chip_id(
            i2c,
            entry.i2c_addr,
            entry.chip_id_high_reg,
            entry.chip_id_low_reg,
        ) {
            if chip_id == entry.expected_id {
                return Ok(SensorInfo {
                    sensor_type: entry.sensor_type,
                    i2c_addr: entry.i2c_addr,
                    chip_id,
                    max_width: entry.max_width,
                    max_height: entry.max_height,
                });
            }
        }
    }
    Err(HalError::HardwareError)
}

/// Probe a specific sensor type at its default I2C address.
///
/// Returns the chip ID if the sensor is present, or an error if not found.
pub fn probe_sensor<I: I2cController>(
    i2c: &mut I,
    sensor_type: SensorType,
) -> Result<u16, HalError> {
    let entry = PROBE_TABLE
        .iter()
        .find(|e| e.sensor_type == sensor_type)
        .ok_or(HalError::InvalidConfig)?;
    let chip_id = read_chip_id(
        i2c,
        entry.i2c_addr,
        entry.chip_id_high_reg,
        entry.chip_id_low_reg,
    )?;
    if chip_id == entry.expected_id {
        Ok(chip_id)
    } else {
        Err(HalError::HardwareError)
    }
}

/// Get the maximum resolution for a known sensor type.
pub fn sensor_max_resolution(sensor_type: SensorType) -> (u16, u16) {
    match sensor_type {
        SensorType::Ov5640 => (2592, 1944),
        SensorType::Imx219 => (3280, 2464),
        SensorType::Imx477 => (4056, 3040),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_kernel::hal::I2cController;

    /// Mock I2C controller that responds to sensor ID probes.
    struct MockI2c {
        /// Stores responses for write_read calls.
        responses: alloc::vec::Vec<(u16, u16, u8)>, // (device_addr, reg_addr, value)
    }

    impl MockI2c {
        fn new() -> Self {
            Self {
                responses: alloc::vec::Vec::new(),
            }
        }

        /// Register a response: when reading reg_addr from device_addr, return value.
        fn add_response(&mut self, device_addr: u16, reg_addr: u16, value: u8) {
            self.responses.push((device_addr, reg_addr, value));
        }

        /// Simulate an OV5640 connected at its default address.
        fn with_ov5640(mut self) -> Self {
            self.add_response(OV5640_I2C_ADDR, OV5640_CHIP_ID_HIGH, 0x56);
            self.add_response(OV5640_I2C_ADDR, OV5640_CHIP_ID_LOW, 0x40);
            self
        }

        /// Simulate an IMX219 connected at its default address.
        fn with_imx219(mut self) -> Self {
            self.add_response(IMX219_I2C_ADDR, IMX219_CHIP_ID_HIGH, 0x02);
            self.add_response(IMX219_I2C_ADDR, IMX219_CHIP_ID_LOW, 0x19);
            self
        }

        /// Simulate an IMX477 connected at its default address.
        fn with_imx477(mut self) -> Self {
            self.add_response(IMX477_I2C_ADDR, IMX477_CHIP_ID_HIGH, 0x04);
            self.add_response(IMX477_I2C_ADDR, IMX477_CHIP_ID_LOW, 0x77);
            self
        }

        fn find_response(&self, device_addr: u16, reg_addr: u16) -> Option<u8> {
            self.responses
                .iter()
                .find(|(da, ra, _)| *da == device_addr && *ra == reg_addr)
                .map(|&(_, _, v)| v)
        }
    }

    impl I2cController for MockI2c {
        fn init(&mut self, _config: smallaios_kernel::hal::I2cConfig) -> Result<(), HalError> {
            Ok(())
        }

        fn write(&mut self, _addr: u16, _data: &[u8]) -> Result<(), HalError> {
            Ok(())
        }

        fn read(&mut self, _addr: u16, _buf: &mut [u8]) -> Result<(), HalError> {
            Err(HalError::HardwareError)
        }

        fn write_read(
            &mut self,
            addr: u16,
            write_data: &[u8],
            read_buf: &mut [u8],
        ) -> Result<(), HalError> {
            if write_data.len() < 2 || read_buf.is_empty() {
                return Err(HalError::InvalidConfig);
            }
            let reg_addr = (write_data[0] as u16) << 8 | write_data[1] as u16;
            if let Some(val) = self.find_response(addr, reg_addr) {
                read_buf[0] = val;
                Ok(())
            } else {
                Err(HalError::NackReceived)
            }
        }

        fn bus_recovery(&mut self) -> Result<(), HalError> {
            Ok(())
        }

        fn reset(&mut self) -> Result<(), HalError> {
            Ok(())
        }
    }

    #[test]
    fn test_detect_ov5640() {
        let mut i2c = MockI2c::new().with_ov5640();
        let info = detect_sensor(&mut i2c).unwrap();
        assert_eq!(info.sensor_type, SensorType::Ov5640);
        assert_eq!(info.i2c_addr, OV5640_I2C_ADDR);
        assert_eq!(info.chip_id, OV5640_CHIP_ID);
        assert_eq!(info.max_width, 2592);
        assert_eq!(info.max_height, 1944);
    }

    #[test]
    fn test_detect_imx219() {
        let mut i2c = MockI2c::new().with_imx219();
        let info = detect_sensor(&mut i2c).unwrap();
        assert_eq!(info.sensor_type, SensorType::Imx219);
        assert_eq!(info.chip_id, IMX219_CHIP_ID);
    }

    #[test]
    fn test_detect_imx477() {
        let mut i2c = MockI2c::new().with_imx477();
        let info = detect_sensor(&mut i2c).unwrap();
        assert_eq!(info.sensor_type, SensorType::Imx477);
        assert_eq!(info.chip_id, IMX477_CHIP_ID);
    }

    #[test]
    fn test_detect_no_sensor() {
        let mut i2c = MockI2c::new(); // No sensors connected.
        let result = detect_sensor(&mut i2c);
        assert_eq!(result, Err(HalError::HardwareError));
    }

    #[test]
    fn test_probe_specific_sensor() {
        let mut i2c = MockI2c::new().with_ov5640();
        let chip_id = probe_sensor(&mut i2c, SensorType::Ov5640).unwrap();
        assert_eq!(chip_id, OV5640_CHIP_ID);
    }

    #[test]
    fn test_probe_wrong_sensor() {
        let mut i2c = MockI2c::new().with_ov5640();
        // Probing for IMX219 should fail since only OV5640 is connected.
        let result = probe_sensor(&mut i2c, SensorType::Imx219);
        assert!(result.is_err());
    }

    #[test]
    fn test_sensor_max_resolution() {
        assert_eq!(sensor_max_resolution(SensorType::Ov5640), (2592, 1944));
        assert_eq!(sensor_max_resolution(SensorType::Imx219), (3280, 2464));
        assert_eq!(sensor_max_resolution(SensorType::Imx477), (4056, 3040));
    }

    #[test]
    fn test_read_reg8_formats_address_correctly() {
        let mut i2c = MockI2c::new();
        i2c.add_response(0x3C, 0x300A, 0x56);
        let val = read_reg8(&mut i2c, 0x3C, 0x300A).unwrap();
        assert_eq!(val, 0x56);
    }
}
