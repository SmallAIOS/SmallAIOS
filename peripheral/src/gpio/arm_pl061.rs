// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARM PL061 GPIO MMIO controller driver.
//!
//! Implements the [`GpioController`] trait for ARM platforms using the
//! PrimeCell PL061 GPIO IP. Each PL061 instance manages 8 GPIO pins.
//!
//! # Key feature: address-masked data access
//!
//! The PL061 uses a unique address-masking scheme for atomic data access.
//! GPIODATA occupies addresses 0x000-0x3FC. The address bits [9:2] serve
//! as a byte mask: only pins whose corresponding address bit is set are
//! affected by the read/write.
//!
//! # References
//!
//! - ARM PL061 TRM (DDI0190)

#![allow(dead_code)]

use smallaios_kernel::hal::{GpioConfig, GpioController, GpioInterruptEdge, GpioPinMode, HalError};

const GPIODATA_BASE: u32 = 0x000;
const GPIODATA_ALL: u32 = 0x3FC;
const GPIODIR: u32 = 0x400;
const GPIOIS: u32 = 0x404;
const GPIOIBE: u32 = 0x408;
const GPIOIEV: u32 = 0x40C;
const GPIOIE: u32 = 0x410;
const GPIORIS: u32 = 0x414;
const GPIOMIS: u32 = 0x418;
const GPIOIC: u32 = 0x41C;
const GPIOAFSEL: u32 = 0x420;

const PINS_PER_INSTANCE: u8 = 8;

/// ARM PL061 GPIO controller driver.
pub struct Pl061Gpio {
    base_addr: u64,
    irq: u32,
    dir_shadow: u8,
    output_shadow: u8,
    ie_shadow: u8,
}

impl Pl061Gpio {
    pub fn new(base_addr: u64, irq: u32) -> Self {
        Self {
            base_addr,
            irq,
            dir_shadow: 0,
            output_shadow: 0,
            ie_shadow: 0,
        }
    }

    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }

    fn data_addr_for_mask(mask: u8) -> u32 {
        GPIODATA_BASE + ((mask as u32) << 2)
    }

    fn read_reg(&self, _offset: u32) -> Result<u32, HalError> {
        Err(HalError::MmioError)
    }
    fn write_reg(&mut self, _offset: u32, _value: u32) -> Result<(), HalError> {
        Err(HalError::MmioError)
    }

    fn configure_interrupt(&mut self, pin: u8, edge: GpioInterruptEdge) -> Result<(), HalError> {
        let mask = crate::gpio::pin_mask(pin);
        let is = self.read_reg(GPIOIS)? & !mask;
        let ibe = self.read_reg(GPIOIBE)? & !mask;
        let iev = self.read_reg(GPIOIEV)? & !mask;

        let (is_val, ibe_val, iev_val) = match edge {
            GpioInterruptEdge::Rising => (0, 0, mask),
            GpioInterruptEdge::Falling => (0, 0, 0),
            GpioInterruptEdge::Both => (0, mask, 0),
            GpioInterruptEdge::LevelHigh => (mask, 0, mask),
            GpioInterruptEdge::LevelLow => (mask, 0, 0),
        };

        self.write_reg(GPIOIS, is | is_val)?;
        self.write_reg(GPIOIBE, ibe | ibe_val)?;
        self.write_reg(GPIOIEV, iev | iev_val)?;
        Ok(())
    }
}

impl GpioController for Pl061Gpio {
    fn configure(&mut self, pin: u8, config: GpioConfig) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_INSTANCE)?;
        let mask = 1u8 << pin;

        match config.mode {
            GpioPinMode::Output | GpioPinMode::OpenDrain => {
                self.dir_shadow |= mask;
            }
            GpioPinMode::Input => {
                self.dir_shadow &= !mask;
            }
            GpioPinMode::AlternateFunction(_) => {
                let afsel = self.read_reg(GPIOAFSEL)?;
                self.write_reg(GPIOAFSEL, afsel | crate::gpio::pin_mask(pin))?;
                return Ok(());
            }
            GpioPinMode::Analog => {
                return Err(HalError::NotSupported);
            }
        }

        let afsel = self.read_reg(GPIOAFSEL)?;
        self.write_reg(GPIOAFSEL, afsel & !crate::gpio::pin_mask(pin))?;
        self.write_reg(GPIODIR, self.dir_shadow as u32)?;

        if let Some(edge) = config.interrupt {
            self.configure_interrupt(pin, edge)?;
        }
        Ok(())
    }

    fn read(&self, pin: u8) -> Result<bool, HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_INSTANCE)?;
        let addr = Self::data_addr_for_mask(1 << pin);
        let val = self.read_reg(addr)?;
        Ok(val & crate::gpio::pin_mask(pin) != 0)
    }

    fn set_high(&mut self, pin: u8) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_INSTANCE)?;
        let mask = 1u8 << pin;
        self.output_shadow |= mask;
        self.write_reg(Self::data_addr_for_mask(mask), 0xFF)
    }

    fn set_low(&mut self, pin: u8) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_INSTANCE)?;
        let mask = 1u8 << pin;
        self.output_shadow &= !mask;
        self.write_reg(Self::data_addr_for_mask(mask), 0x00)
    }

    fn toggle(&mut self, pin: u8) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_INSTANCE)?;
        let mask = 1u8 << pin;
        self.output_shadow ^= mask;
        let val = if self.output_shadow & mask != 0 {
            0xFF
        } else {
            0x00
        };
        self.write_reg(Self::data_addr_for_mask(mask), val)
    }

    fn set_mask(&mut self, set_mask: u32, clear_mask: u32) -> Result<(), HalError> {
        let set = (set_mask & 0xFF) as u8;
        let clear = (clear_mask & 0xFF) as u8;
        self.output_shadow = (self.output_shadow | set) & !clear;
        self.write_reg(GPIODATA_ALL, self.output_shadow as u32)
    }

    fn read_all(&self) -> Result<u32, HalError> {
        self.read_reg(GPIODATA_ALL)
    }

    fn enable_interrupt(&mut self, pin: u8) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_INSTANCE)?;
        self.ie_shadow |= 1 << pin;
        self.write_reg(GPIOIE, self.ie_shadow as u32)
    }

    fn disable_interrupt(&mut self, pin: u8) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_INSTANCE)?;
        self.ie_shadow &= !(1 << pin);
        self.write_reg(GPIOIE, self.ie_shadow as u32)
    }

    fn irq_handler(&mut self) -> Result<u8, HalError> {
        let mis = self.read_reg(GPIOMIS)?;
        if mis == 0 {
            return Err(HalError::InterruptError);
        }
        let pin = (mis as u8).trailing_zeros() as u8;
        self.write_reg(GPIOIC, crate::gpio::pin_mask(pin))?;
        Ok(pin)
    }

    fn pin_count(&self) -> u8 {
        PINS_PER_INSTANCE
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.dir_shadow = 0;
        self.output_shadow = 0;
        self.ie_shadow = 0;
        self.write_reg(GPIODIR, 0)?;
        self.write_reg(GPIOIE, 0)?;
        self.write_reg(GPIOIC, 0xFF)?;
        self.write_reg(GPIOAFSEL, 0)?;
        self.write_reg(GPIODATA_ALL, 0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_kernel::hal::GpioPull;

    #[test]
    fn test_pl061_new() {
        let drv = Pl061Gpio::new(0x4000_A000, 10);
        assert_eq!(drv.base_addr(), 0x4000_A000);
        assert_eq!(drv.pin_count(), 8);
    }

    #[test]
    fn test_data_addr_for_mask() {
        assert_eq!(Pl061Gpio::data_addr_for_mask(0x01), 0x004);
        assert_eq!(Pl061Gpio::data_addr_for_mask(0x80), 0x200);
        assert_eq!(Pl061Gpio::data_addr_for_mask(0xFF), 0x3FC);
    }

    #[test]
    fn test_configure_pin_out_of_range() {
        let mut drv = Pl061Gpio::new(0x4000_A000, 10);
        let config = GpioConfig {
            mode: GpioPinMode::Input,
            pull: GpioPull::None,
            interrupt: None,
            debounce_us: 0,
        };
        assert_eq!(drv.configure(8, config), Err(HalError::OutOfRange));
    }

    #[test]
    fn test_configure_analog_not_supported() {
        let mut drv = Pl061Gpio::new(0x4000_A000, 10);
        let config = GpioConfig {
            mode: GpioPinMode::Analog,
            pull: GpioPull::None,
            interrupt: None,
            debounce_us: 0,
        };
        assert_eq!(drv.configure(0, config), Err(HalError::NotSupported));
    }

    #[test]
    fn test_toggle_shadow_tracking() {
        let mut drv = Pl061Gpio::new(0x4000_A000, 10);
        assert_eq!(drv.output_shadow & 0x01, 0);
        let _ = drv.toggle(0);
        assert_eq!(drv.output_shadow & 0x01, 1);
        let _ = drv.toggle(0);
        assert_eq!(drv.output_shadow & 0x01, 0);
    }

    #[test]
    fn test_dir_shadow_tracking() {
        let mut drv = Pl061Gpio::new(0x4000_A000, 10);
        let config = GpioConfig {
            mode: GpioPinMode::Output,
            pull: GpioPull::None,
            interrupt: None,
            debounce_us: 0,
        };
        let _ = drv.configure(3, config);
        assert_ne!(drv.dir_shadow & (1 << 3), 0);
    }

    #[test]
    fn test_constructor_fields() {
        let drv = Pl061Gpio::new(0xBEEF_0000, 99);
        assert_eq!(drv.base_addr(), 0xBEEF_0000);
        assert_eq!(drv.irq, 99);
        assert_eq!(drv.dir_shadow, 0);
        assert_eq!(drv.output_shadow, 0);
        assert_eq!(drv.ie_shadow, 0);
    }

    #[test]
    fn test_pin_validation_8_pins() {
        let drv = Pl061Gpio::new(0x4000_A000, 10);
        // Pins 0..7 are valid, pin 8+ is out of range.
        assert_eq!(drv.read(0), Err(HalError::MmioError));
        assert_eq!(drv.read(7), Err(HalError::MmioError));
        assert_eq!(drv.read(8), Err(HalError::OutOfRange));
        assert_eq!(drv.read(31), Err(HalError::OutOfRange));
    }

    #[test]
    fn test_dir_shadow_input_clears_bit() {
        let mut drv = Pl061Gpio::new(0x4000_A000, 10);
        // Set pin 5 as output (sets dir bit).
        let out_config = GpioConfig {
            mode: GpioPinMode::Output,
            pull: GpioPull::None,
            interrupt: None,
            debounce_us: 0,
        };
        let _ = drv.configure(5, out_config);
        assert_ne!(drv.dir_shadow & (1 << 5), 0);

        // Set pin 5 as input (clears dir bit).
        let in_config = GpioConfig {
            mode: GpioPinMode::Input,
            pull: GpioPull::None,
            interrupt: None,
            debounce_us: 0,
        };
        let _ = drv.configure(5, in_config);
        assert_eq!(drv.dir_shadow & (1 << 5), 0);
    }

    #[test]
    fn test_output_shadow_set_high_low() {
        let mut drv = Pl061Gpio::new(0x4000_A000, 10);
        let _ = drv.set_high(3);
        assert_ne!(drv.output_shadow & (1 << 3), 0);
        let _ = drv.set_low(3);
        assert_eq!(drv.output_shadow & (1 << 3), 0);
    }

    #[test]
    fn test_output_shadow_set_high_out_of_range() {
        let mut drv = Pl061Gpio::new(0x4000_A000, 10);
        assert_eq!(drv.set_high(8), Err(HalError::OutOfRange));
    }

    #[test]
    fn test_set_mask_shadow() {
        let mut drv = Pl061Gpio::new(0x4000_A000, 10);
        // set bits 0 and 2
        let _ = drv.set_mask(0x05, 0x00);
        assert_eq!(drv.output_shadow, 0x05);
        // clear bit 0
        let _ = drv.set_mask(0x00, 0x01);
        assert_eq!(drv.output_shadow, 0x04);
    }

    #[test]
    fn test_set_mask_truncates_to_8_bits() {
        let mut drv = Pl061Gpio::new(0x4000_A000, 10);
        // Bits above 7 should be masked off for the 8-pin PL061.
        let _ = drv.set_mask(0xFF00, 0x0000);
        assert_eq!(drv.output_shadow, 0x00); // Upper bits are masked away
    }

    #[test]
    fn test_ie_shadow_enable_disable() {
        let mut drv = Pl061Gpio::new(0x4000_A000, 10);
        let _ = drv.enable_interrupt(5);
        assert_ne!(drv.ie_shadow & (1 << 5), 0);

        let _ = drv.disable_interrupt(5);
        assert_eq!(drv.ie_shadow & (1 << 5), 0);
    }

    #[test]
    fn test_ie_shadow_multiple_pins() {
        let mut drv = Pl061Gpio::new(0x4000_A000, 10);
        let _ = drv.enable_interrupt(0);
        let _ = drv.enable_interrupt(7);
        assert_eq!(drv.ie_shadow, (1 << 0) | (1 << 7));

        let _ = drv.disable_interrupt(0);
        assert_eq!(drv.ie_shadow, 1 << 7);
    }

    #[test]
    fn test_enable_interrupt_out_of_range() {
        let mut drv = Pl061Gpio::new(0x4000_A000, 10);
        assert_eq!(drv.enable_interrupt(8), Err(HalError::OutOfRange));
    }

    #[test]
    fn test_reset_clears_all_shadows() {
        let mut drv = Pl061Gpio::new(0x4000_A000, 10);
        drv.dir_shadow = 0xFF;
        drv.output_shadow = 0xFF;
        drv.ie_shadow = 0xFF;

        let _ = drv.reset();
        assert_eq!(drv.dir_shadow, 0);
        assert_eq!(drv.output_shadow, 0);
        assert_eq!(drv.ie_shadow, 0);
    }

    #[test]
    fn test_open_drain_sets_dir_as_output() {
        let mut drv = Pl061Gpio::new(0x4000_A000, 10);
        let config = GpioConfig {
            mode: GpioPinMode::OpenDrain,
            pull: GpioPull::None,
            interrupt: None,
            debounce_us: 0,
        };
        let _ = drv.configure(2, config);
        assert_ne!(drv.dir_shadow & (1 << 2), 0);
    }

    #[test]
    fn test_data_addr_for_mask_single_pins() {
        // Each pin mask should produce a unique address.
        for pin in 0..8u8 {
            let mask = 1u8 << pin;
            let addr = Pl061Gpio::data_addr_for_mask(mask);
            assert_eq!(addr, (mask as u32) << 2);
        }
    }

    #[test]
    fn test_toggle_out_of_range() {
        let mut drv = Pl061Gpio::new(0x4000_A000, 10);
        assert_eq!(drv.toggle(8), Err(HalError::OutOfRange));
    }
}
