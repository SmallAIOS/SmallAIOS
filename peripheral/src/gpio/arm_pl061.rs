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
}
