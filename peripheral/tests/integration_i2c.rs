// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the I2C peripheral subsystem.
//!
//! These tests exercise a mock I2C controller that implements the
//! `smallaios_kernel::hal::I2cController` trait, verifying configuration,
//! write/read transactions, write-read combined transfers, bus recovery,
//! and error handling.

#![cfg(feature = "i2c")]

extern crate alloc;

use smallaios_kernel::hal::{HalError, I2cAddressMode, I2cConfig, I2cController, I2cSpeed};
use smallaios_peripheral::PeripheralType;

// ---------------------------------------------------------------------------
// Mock I2C controller
// ---------------------------------------------------------------------------

/// Mock I2C controller for testing.
struct MockI2c {
    initialized: bool,
    config: Option<I2cConfig>,
    /// Simulated register file: address -> data.
    registers: alloc::collections::BTreeMap<u16, alloc::vec::Vec<u8>>,
    /// Last written data (address, data).
    last_write: Option<(u16, alloc::vec::Vec<u8>)>,
    /// Whether to simulate a NACK on the next operation.
    force_nack: bool,
    /// Bus recovery call count.
    recovery_count: u32,
}

impl MockI2c {
    fn new() -> Self {
        Self {
            initialized: false,
            config: None,
            registers: alloc::collections::BTreeMap::new(),
            last_write: None,
            force_nack: false,
            recovery_count: 0,
        }
    }

    /// Pre-load data that will be returned on a read from the given address.
    #[allow(dead_code)]
    fn preload_register(&mut self, addr: u16, data: &[u8]) {
        self.registers.insert(addr, data.to_vec());
    }
}

impl I2cController for MockI2c {
    fn init(&mut self, config: I2cConfig) -> Result<(), HalError> {
        self.config = Some(config);
        self.initialized = true;
        Ok(())
    }

    fn write(&mut self, addr: u16, data: &[u8]) -> Result<(), HalError> {
        if !self.initialized {
            return Err(HalError::InitFailed);
        }
        if self.force_nack {
            self.force_nack = false;
            return Err(HalError::NackReceived);
        }
        self.last_write = Some((addr, data.to_vec()));
        self.registers.insert(addr, data.to_vec());
        Ok(())
    }

    fn read(&mut self, addr: u16, buf: &mut [u8]) -> Result<(), HalError> {
        if !self.initialized {
            return Err(HalError::InitFailed);
        }
        if self.force_nack {
            self.force_nack = false;
            return Err(HalError::NackReceived);
        }
        if let Some(data) = self.registers.get(&addr) {
            let copy_len = buf.len().min(data.len());
            buf[..copy_len].copy_from_slice(&data[..copy_len]);
            Ok(())
        } else {
            Err(HalError::NackReceived)
        }
    }

    fn write_read(
        &mut self,
        addr: u16,
        write_data: &[u8],
        read_buf: &mut [u8],
    ) -> Result<(), HalError> {
        // Write the register pointer, then read back.
        self.write(addr, write_data)?;
        self.read(addr, read_buf)
    }

    fn bus_recovery(&mut self) -> Result<(), HalError> {
        self.recovery_count += 1;
        Ok(())
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.initialized = false;
        self.config = None;
        self.registers.clear();
        self.last_write = None;
        self.force_nack = false;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_i2c_init_standard_mode() {
    let mut i2c = MockI2c::new();
    let config = I2cConfig {
        speed: I2cSpeed::Standard,
        address_mode: I2cAddressMode::SevenBit,
        clock_stretching: false,
        timeout_us: 1000,
    };
    assert!(i2c.init(config).is_ok());
    assert!(i2c.initialized);
    assert_eq!(i2c.config.unwrap().speed, I2cSpeed::Standard);
}

#[test]
fn test_i2c_init_fast_mode_10bit() {
    let mut i2c = MockI2c::new();
    let config = I2cConfig {
        speed: I2cSpeed::Fast,
        address_mode: I2cAddressMode::TenBit,
        clock_stretching: true,
        timeout_us: 5000,
    };
    assert!(i2c.init(config).is_ok());
    let cfg = i2c.config.unwrap();
    assert_eq!(cfg.speed, I2cSpeed::Fast);
    assert_eq!(cfg.address_mode, I2cAddressMode::TenBit);
    assert!(cfg.clock_stretching);
}

#[test]
fn test_i2c_write_then_read() {
    let mut i2c = MockI2c::new();
    i2c.init(I2cConfig {
        speed: I2cSpeed::Fast,
        address_mode: I2cAddressMode::SevenBit,
        clock_stretching: false,
        timeout_us: 1000,
    })
    .unwrap();

    let sensor_addr: u16 = 0x48;
    let data = [0x01, 0x02, 0x03];
    i2c.write(sensor_addr, &data).unwrap();

    let (addr, written) = i2c.last_write.as_ref().unwrap();
    assert_eq!(*addr, sensor_addr);
    assert_eq!(written.as_slice(), &data);

    let mut buf = [0u8; 3];
    i2c.read(sensor_addr, &mut buf).unwrap();
    assert_eq!(buf, data);
}

#[test]
fn test_i2c_write_read_combined() {
    let mut i2c = MockI2c::new();
    i2c.init(I2cConfig {
        speed: I2cSpeed::FastPlus,
        address_mode: I2cAddressMode::SevenBit,
        clock_stretching: false,
        timeout_us: 2000,
    })
    .unwrap();

    let addr: u16 = 0x68; // IMU address
    let reg_ptr = [0x3B]; // ACCEL_XOUT_H register
    let mut read_buf = [0u8; 1];
    i2c.write_read(addr, &reg_ptr, &mut read_buf).unwrap();
    // After write_read, the register should contain what we wrote.
    assert_eq!(read_buf[0], 0x3B);
}

#[test]
fn test_i2c_nack_error() {
    let mut i2c = MockI2c::new();
    i2c.init(I2cConfig {
        speed: I2cSpeed::Standard,
        address_mode: I2cAddressMode::SevenBit,
        clock_stretching: false,
        timeout_us: 1000,
    })
    .unwrap();

    i2c.force_nack = true;
    let result = i2c.write(0x50, &[0x00]);
    assert_eq!(result, Err(HalError::NackReceived));
}

#[test]
fn test_i2c_read_uninitialized() {
    let mut i2c = MockI2c::new();
    let mut buf = [0u8; 2];
    let result = i2c.read(0x48, &mut buf);
    assert_eq!(result, Err(HalError::InitFailed));
}

#[test]
fn test_i2c_bus_recovery() {
    let mut i2c = MockI2c::new();
    i2c.init(I2cConfig {
        speed: I2cSpeed::Standard,
        address_mode: I2cAddressMode::SevenBit,
        clock_stretching: false,
        timeout_us: 1000,
    })
    .unwrap();

    i2c.bus_recovery().unwrap();
    i2c.bus_recovery().unwrap();
    assert_eq!(i2c.recovery_count, 2);
}

#[test]
fn test_i2c_reset_clears_state() {
    let mut i2c = MockI2c::new();
    i2c.init(I2cConfig {
        speed: I2cSpeed::Fast,
        address_mode: I2cAddressMode::SevenBit,
        clock_stretching: false,
        timeout_us: 1000,
    })
    .unwrap();
    i2c.write(0x48, &[0xAA, 0xBB]).unwrap();
    assert!(i2c.initialized);

    i2c.reset().unwrap();
    assert!(!i2c.initialized);
    assert!(i2c.config.is_none());
    assert!(i2c.last_write.is_none());
}

#[test]
fn test_i2c_peripheral_type() {
    // Verify that I2C is represented correctly in the peripheral type enum.
    assert_eq!(PeripheralType::from_u8(0), Some(PeripheralType::I2c));
    assert_eq!(PeripheralType::I2c as u8, 0);
}

#[test]
fn test_i2c_multiple_devices() {
    let mut i2c = MockI2c::new();
    i2c.init(I2cConfig {
        speed: I2cSpeed::Fast,
        address_mode: I2cAddressMode::SevenBit,
        clock_stretching: false,
        timeout_us: 1000,
    })
    .unwrap();

    // Write to two different devices on the same bus.
    let temp_sensor: u16 = 0x48;
    let eeprom: u16 = 0x50;

    i2c.write(temp_sensor, &[0x00, 0x1A]).unwrap();
    i2c.write(eeprom, &[0x00, 0x00, 0xFF]).unwrap();

    // Read back from each.
    let mut temp_buf = [0u8; 2];
    i2c.read(temp_sensor, &mut temp_buf).unwrap();
    assert_eq!(temp_buf, [0x00, 0x1A]);

    let mut eeprom_buf = [0u8; 3];
    i2c.read(eeprom, &mut eeprom_buf).unwrap();
    assert_eq!(eeprom_buf, [0x00, 0x00, 0xFF]);
}
