// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SiFive GPIO MMIO controller driver for RISC-V platforms.
//!
//! Each instance manages up to 32 GPIO pins.
//! Reference: SiFive FU740-C000 Manual, Chapter 17

#![allow(dead_code)]

use smallaios_kernel::hal::{
    GpioConfig, GpioController, GpioInterruptEdge, GpioPinMode, GpioPull, HalError,
};

const INPUT_VAL: u32 = 0x00;
const INPUT_EN: u32 = 0x04;
const OUTPUT_EN: u32 = 0x08;
const OUTPUT_VAL: u32 = 0x0C;
const PUE: u32 = 0x10;
const DS: u32 = 0x14;
const RISE_IE: u32 = 0x18;
const RISE_IP: u32 = 0x1C;
const FALL_IE: u32 = 0x20;
const FALL_IP: u32 = 0x24;
const HIGH_IE: u32 = 0x28;
const HIGH_IP: u32 = 0x2C;
const LOW_IE: u32 = 0x30;
const LOW_IP: u32 = 0x34;
const IOF_EN: u32 = 0x38;
const IOF_SEL: u32 = 0x3C;
const OUT_XOR: u32 = 0x40;

const PINS_PER_INSTANCE: u8 = 32;

pub struct SiFiveGpio {
    base_addr: u64,
    irq_base: u32,
    output_en_shadow: u32,
    output_val_shadow: u32,
    rise_ie_shadow: u32,
    fall_ie_shadow: u32,
    high_ie_shadow: u32,
    low_ie_shadow: u32,
}

impl SiFiveGpio {
    pub fn new(base_addr: u64, irq_base: u32) -> Self {
        Self {
            base_addr,
            irq_base,
            output_en_shadow: 0,
            output_val_shadow: 0,
            rise_ie_shadow: 0,
            fall_ie_shadow: 0,
            high_ie_shadow: 0,
            low_ie_shadow: 0,
        }
    }

    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }

    fn read_reg(&self, _offset: u32) -> Result<u32, HalError> {
        Err(HalError::MmioError)
    }
    fn write_reg(&mut self, _offset: u32, _value: u32) -> Result<(), HalError> {
        Err(HalError::MmioError)
    }

    fn clear_interrupt_config(&mut self, pin: u8) {
        let mask = !crate::gpio::pin_mask(pin);
        self.rise_ie_shadow &= mask;
        self.fall_ie_shadow &= mask;
        self.high_ie_shadow &= mask;
        self.low_ie_shadow &= mask;
    }

    fn set_interrupt_config(&mut self, pin: u8, edge: GpioInterruptEdge) {
        let mask = crate::gpio::pin_mask(pin);
        self.clear_interrupt_config(pin);
        match edge {
            GpioInterruptEdge::Rising => {
                self.rise_ie_shadow |= mask;
            }
            GpioInterruptEdge::Falling => {
                self.fall_ie_shadow |= mask;
            }
            GpioInterruptEdge::Both => {
                self.rise_ie_shadow |= mask;
                self.fall_ie_shadow |= mask;
            }
            GpioInterruptEdge::LevelHigh => {
                self.high_ie_shadow |= mask;
            }
            GpioInterruptEdge::LevelLow => {
                self.low_ie_shadow |= mask;
            }
        }
    }
}

impl GpioController for SiFiveGpio {
    fn configure(&mut self, pin: u8, config: GpioConfig) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_INSTANCE)?;
        let mask = crate::gpio::pin_mask(pin);
        match config.mode {
            GpioPinMode::Output | GpioPinMode::OpenDrain => {
                self.output_en_shadow |= mask;
                self.write_reg(OUTPUT_EN, self.output_en_shadow)?;
                let input_en = self.read_reg(INPUT_EN)? & !mask;
                self.write_reg(INPUT_EN, input_en)?;
                let iof = self.read_reg(IOF_EN)? & !mask;
                self.write_reg(IOF_EN, iof)?;
            }
            GpioPinMode::Input => {
                self.output_en_shadow &= !mask;
                self.write_reg(OUTPUT_EN, self.output_en_shadow)?;
                let input_en = self.read_reg(INPUT_EN)? | mask;
                self.write_reg(INPUT_EN, input_en)?;
                let iof = self.read_reg(IOF_EN)? & !mask;
                self.write_reg(IOF_EN, iof)?;
            }
            GpioPinMode::AlternateFunction(sel) => {
                let iof = self.read_reg(IOF_EN)? | mask;
                self.write_reg(IOF_EN, iof)?;
                let iof_sel = self.read_reg(IOF_SEL)?;
                if sel == 0 {
                    self.write_reg(IOF_SEL, iof_sel & !mask)?;
                } else {
                    self.write_reg(IOF_SEL, iof_sel | mask)?;
                }
                return Ok(());
            }
            GpioPinMode::Analog => {
                return Err(HalError::NotSupported);
            }
        }
        let pue = self.read_reg(PUE)?;
        match config.pull {
            GpioPull::PullUp => self.write_reg(PUE, pue | mask)?,
            _ => self.write_reg(PUE, pue & !mask)?,
        }
        if let Some(edge) = config.interrupt {
            self.set_interrupt_config(pin, edge);
            self.write_reg(RISE_IE, self.rise_ie_shadow)?;
            self.write_reg(FALL_IE, self.fall_ie_shadow)?;
            self.write_reg(HIGH_IE, self.high_ie_shadow)?;
            self.write_reg(LOW_IE, self.low_ie_shadow)?;
        } else {
            self.clear_interrupt_config(pin);
        }
        Ok(())
    }

    fn read(&self, pin: u8) -> Result<bool, HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_INSTANCE)?;
        let val = self.read_reg(INPUT_VAL)?;
        Ok(val & crate::gpio::pin_mask(pin) != 0)
    }

    fn set_high(&mut self, pin: u8) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_INSTANCE)?;
        self.output_val_shadow |= crate::gpio::pin_mask(pin);
        self.write_reg(OUTPUT_VAL, self.output_val_shadow)
    }

    fn set_low(&mut self, pin: u8) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_INSTANCE)?;
        self.output_val_shadow &= !crate::gpio::pin_mask(pin);
        self.write_reg(OUTPUT_VAL, self.output_val_shadow)
    }

    fn toggle(&mut self, pin: u8) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_INSTANCE)?;
        self.output_val_shadow ^= crate::gpio::pin_mask(pin);
        self.write_reg(OUTPUT_VAL, self.output_val_shadow)
    }

    fn set_mask(&mut self, set_mask: u32, clear_mask: u32) -> Result<(), HalError> {
        self.output_val_shadow = (self.output_val_shadow | set_mask) & !clear_mask;
        self.write_reg(OUTPUT_VAL, self.output_val_shadow)
    }

    fn read_all(&self) -> Result<u32, HalError> {
        self.read_reg(INPUT_VAL)
    }

    fn enable_interrupt(&mut self, pin: u8) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_INSTANCE)?;
        self.write_reg(RISE_IE, self.rise_ie_shadow)?;
        self.write_reg(FALL_IE, self.fall_ie_shadow)?;
        self.write_reg(HIGH_IE, self.high_ie_shadow)?;
        self.write_reg(LOW_IE, self.low_ie_shadow)
    }

    fn disable_interrupt(&mut self, pin: u8) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_INSTANCE)?;
        self.clear_interrupt_config(pin);
        self.write_reg(RISE_IE, self.rise_ie_shadow)?;
        self.write_reg(FALL_IE, self.fall_ie_shadow)?;
        self.write_reg(HIGH_IE, self.high_ie_shadow)?;
        self.write_reg(LOW_IE, self.low_ie_shadow)
    }

    fn irq_handler(&mut self) -> Result<u8, HalError> {
        let rise = self.read_reg(RISE_IP)?;
        let fall = self.read_reg(FALL_IP)?;
        let high = self.read_reg(HIGH_IP)?;
        let low = self.read_reg(LOW_IP)?;
        let pending = rise | fall | high | low;
        if pending == 0 {
            return Err(HalError::InterruptError);
        }
        let pin = pending.trailing_zeros() as u8;
        let mask = crate::gpio::pin_mask(pin);
        if rise & mask != 0 {
            self.write_reg(RISE_IP, mask)?;
        }
        if fall & mask != 0 {
            self.write_reg(FALL_IP, mask)?;
        }
        if high & mask != 0 {
            self.write_reg(HIGH_IP, mask)?;
        }
        if low & mask != 0 {
            self.write_reg(LOW_IP, mask)?;
        }
        Ok(pin)
    }

    fn pin_count(&self) -> u8 {
        PINS_PER_INSTANCE
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.output_en_shadow = 0;
        self.output_val_shadow = 0;
        self.rise_ie_shadow = 0;
        self.fall_ie_shadow = 0;
        self.high_ie_shadow = 0;
        self.low_ie_shadow = 0;
        self.write_reg(OUTPUT_EN, 0)?;
        self.write_reg(OUTPUT_VAL, 0)?;
        self.write_reg(RISE_IE, 0)?;
        self.write_reg(FALL_IE, 0)?;
        self.write_reg(HIGH_IE, 0)?;
        self.write_reg(LOW_IE, 0)?;
        self.write_reg(IOF_EN, 0)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sifive_gpio_new() {
        let drv = SiFiveGpio::new(0x1006_0000, 7);
        assert_eq!(drv.base_addr(), 0x1006_0000);
        assert_eq!(drv.pin_count(), 32);
    }

    #[test]
    fn test_read_pin_out_of_range() {
        let drv = SiFiveGpio::new(0x1006_0000, 7);
        assert_eq!(drv.read(32), Err(HalError::OutOfRange));
    }

    #[test]
    fn test_set_interrupt_config_shadows() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        drv.set_interrupt_config(5, GpioInterruptEdge::Rising);
        assert_ne!(drv.rise_ie_shadow & (1 << 5), 0);
        drv.set_interrupt_config(5, GpioInterruptEdge::Both);
        assert_ne!(drv.fall_ie_shadow & (1 << 5), 0);
    }

    #[test]
    fn test_output_shadow_tracking() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        let _ = drv.set_high(5);
        assert_ne!(drv.output_val_shadow & (1 << 5), 0);
        let _ = drv.set_low(5);
        assert_eq!(drv.output_val_shadow & (1 << 5), 0);
    }

    #[test]
    fn test_toggle_shadow() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        let _ = drv.toggle(10);
        assert_ne!(drv.output_val_shadow & (1 << 10), 0);
        let _ = drv.toggle(10);
        assert_eq!(drv.output_val_shadow & (1 << 10), 0);
    }

    #[test]
    fn test_pin_count_returns_32() {
        let drv = SiFiveGpio::new(0x1006_0000, 7);
        assert_eq!(drv.pin_count(), 32);
    }

    #[test]
    fn test_validate_pin_0_first_valid() {
        let drv = SiFiveGpio::new(0x1006_0000, 7);
        // Pin 0 is a valid pin and should not return OutOfRange.
        // read_reg will fail with MmioError, but not OutOfRange.
        assert_eq!(drv.read(0), Err(HalError::MmioError));
    }

    #[test]
    fn test_validate_pin_31_last_valid() {
        let drv = SiFiveGpio::new(0x1006_0000, 7);
        // Pin 31 is the last valid pin (0..31 for 32 pins).
        assert_eq!(drv.read(31), Err(HalError::MmioError));
    }

    #[test]
    fn test_clear_interrupt_config_clears_all_edges() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        // Set all four interrupt shadow registers for pin 3.
        drv.rise_ie_shadow |= 1 << 3;
        drv.fall_ie_shadow |= 1 << 3;
        drv.high_ie_shadow |= 1 << 3;
        drv.low_ie_shadow |= 1 << 3;

        drv.clear_interrupt_config(3);

        assert_eq!(drv.rise_ie_shadow & (1 << 3), 0);
        assert_eq!(drv.fall_ie_shadow & (1 << 3), 0);
        assert_eq!(drv.high_ie_shadow & (1 << 3), 0);
        assert_eq!(drv.low_ie_shadow & (1 << 3), 0);
    }

    #[test]
    fn test_clear_interrupt_config_preserves_other_pins() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        drv.rise_ie_shadow = 0xFFFF_FFFF;
        drv.fall_ie_shadow = 0xFFFF_FFFF;

        drv.clear_interrupt_config(5);

        // Pin 5 should be cleared, all others remain set.
        assert_eq!(drv.rise_ie_shadow & (1 << 5), 0);
        assert_ne!(drv.rise_ie_shadow & (1 << 4), 0);
        assert_ne!(drv.rise_ie_shadow & (1 << 6), 0);
    }

    #[test]
    fn test_set_interrupt_config_rising() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        drv.set_interrupt_config(10, GpioInterruptEdge::Rising);
        assert_ne!(drv.rise_ie_shadow & (1 << 10), 0);
        assert_eq!(drv.fall_ie_shadow & (1 << 10), 0);
        assert_eq!(drv.high_ie_shadow & (1 << 10), 0);
        assert_eq!(drv.low_ie_shadow & (1 << 10), 0);
    }

    #[test]
    fn test_set_interrupt_config_falling() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        drv.set_interrupt_config(10, GpioInterruptEdge::Falling);
        assert_eq!(drv.rise_ie_shadow & (1 << 10), 0);
        assert_ne!(drv.fall_ie_shadow & (1 << 10), 0);
        assert_eq!(drv.high_ie_shadow & (1 << 10), 0);
        assert_eq!(drv.low_ie_shadow & (1 << 10), 0);
    }

    #[test]
    fn test_set_interrupt_config_both() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        drv.set_interrupt_config(10, GpioInterruptEdge::Both);
        assert_ne!(drv.rise_ie_shadow & (1 << 10), 0);
        assert_ne!(drv.fall_ie_shadow & (1 << 10), 0);
        assert_eq!(drv.high_ie_shadow & (1 << 10), 0);
        assert_eq!(drv.low_ie_shadow & (1 << 10), 0);
    }

    #[test]
    fn test_set_interrupt_config_level_high() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        drv.set_interrupt_config(10, GpioInterruptEdge::LevelHigh);
        assert_eq!(drv.rise_ie_shadow & (1 << 10), 0);
        assert_eq!(drv.fall_ie_shadow & (1 << 10), 0);
        assert_ne!(drv.high_ie_shadow & (1 << 10), 0);
        assert_eq!(drv.low_ie_shadow & (1 << 10), 0);
    }

    #[test]
    fn test_set_interrupt_config_level_low() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        drv.set_interrupt_config(10, GpioInterruptEdge::LevelLow);
        assert_eq!(drv.rise_ie_shadow & (1 << 10), 0);
        assert_eq!(drv.fall_ie_shadow & (1 << 10), 0);
        assert_eq!(drv.high_ie_shadow & (1 << 10), 0);
        assert_ne!(drv.low_ie_shadow & (1 << 10), 0);
    }

    #[test]
    fn test_set_interrupt_config_clears_previous() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        // First set Rising, then switch to Falling — Rising should be cleared.
        drv.set_interrupt_config(7, GpioInterruptEdge::Rising);
        assert_ne!(drv.rise_ie_shadow & (1 << 7), 0);

        drv.set_interrupt_config(7, GpioInterruptEdge::Falling);
        assert_eq!(drv.rise_ie_shadow & (1 << 7), 0);
        assert_ne!(drv.fall_ie_shadow & (1 << 7), 0);
    }

    #[test]
    fn test_set_mask_operations() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        // Set bits 0, 1, 2.
        let _ = drv.set_mask(0x07, 0x00);
        assert_eq!(drv.output_val_shadow & 0x07, 0x07);

        // Clear bit 1, keep 0 and 2.
        let _ = drv.set_mask(0x00, 0x02);
        assert_eq!(drv.output_val_shadow & 0x07, 0x05);
    }

    #[test]
    fn test_set_mask_simultaneous_set_and_clear() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        // Set bits 0..3 and clear bits 2..5 simultaneously.
        // Result: set first → 0x0F, then clear → 0x0F & !0x3C = 0x03.
        let _ = drv.set_mask(0x0F, 0x3C);
        assert_eq!(drv.output_val_shadow, 0x03);
    }

    #[test]
    fn test_reset_clears_all_shadows() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        // Pollute all shadow registers.
        drv.output_en_shadow = 0xAAAA_AAAA;
        drv.output_val_shadow = 0x5555_5555;
        drv.rise_ie_shadow = 0xFFFF_FFFF;
        drv.fall_ie_shadow = 0xFFFF_FFFF;
        drv.high_ie_shadow = 0xFFFF_FFFF;
        drv.low_ie_shadow = 0xFFFF_FFFF;

        // reset will fail at write_reg (MmioError) but shadows are cleared first.
        let _ = drv.reset();
        assert_eq!(drv.output_en_shadow, 0);
        assert_eq!(drv.output_val_shadow, 0);
        assert_eq!(drv.rise_ie_shadow, 0);
        assert_eq!(drv.fall_ie_shadow, 0);
        assert_eq!(drv.high_ie_shadow, 0);
        assert_eq!(drv.low_ie_shadow, 0);
    }

    #[test]
    fn test_constructor_fields() {
        let drv = SiFiveGpio::new(0xDEAD_BEEF, 42);
        assert_eq!(drv.base_addr(), 0xDEAD_BEEF);
        assert_eq!(drv.irq_base, 42);
        assert_eq!(drv.output_en_shadow, 0);
        assert_eq!(drv.output_val_shadow, 0);
        assert_eq!(drv.rise_ie_shadow, 0);
        assert_eq!(drv.fall_ie_shadow, 0);
        assert_eq!(drv.high_ie_shadow, 0);
        assert_eq!(drv.low_ie_shadow, 0);
    }

    #[test]
    fn test_set_high_out_of_range() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        assert_eq!(drv.set_high(32), Err(HalError::OutOfRange));
        assert_eq!(drv.set_high(255), Err(HalError::OutOfRange));
    }

    #[test]
    fn test_set_low_out_of_range() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        assert_eq!(drv.set_low(32), Err(HalError::OutOfRange));
    }

    #[test]
    fn test_toggle_out_of_range() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        assert_eq!(drv.toggle(32), Err(HalError::OutOfRange));
    }

    #[test]
    fn test_enable_interrupt_out_of_range() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        assert_eq!(drv.enable_interrupt(32), Err(HalError::OutOfRange));
    }

    #[test]
    fn test_disable_interrupt_out_of_range() {
        let mut drv = SiFiveGpio::new(0x1006_0000, 7);
        assert_eq!(drv.disable_interrupt(32), Err(HalError::OutOfRange));
    }
}
