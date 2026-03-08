// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Xilinx AXI GPIO FPGA controller driver (PG144).
//! Dual-channel, up to 32 pins per channel. This driver uses channel 1.

#![allow(dead_code)]

use smallaios_kernel::hal::{FpgaFabric, GpioConfig, GpioController, GpioPinMode, HalError};

const GPIO_DATA: u32 = 0x0000;
const GPIO_TRI: u32 = 0x0004;
const GPIO2_DATA: u32 = 0x0008;
const GPIO2_TRI: u32 = 0x000C;
const GIER: u32 = 0x011C;
const IP_ISR: u32 = 0x0120;
const IP_IER: u32 = 0x0128;
const GIER_GIE: u32 = 1 << 31;
const IP_CH1: u32 = 1 << 0;
const PINS_PER_CHANNEL: u8 = 32;

pub struct AxiGpio<F: FpgaFabric> {
    fabric: F,
    base_addr: u64,
    tri_shadow: u32,
    data_shadow: u32,
    irq_mask: u32,
    last_input: u32,
}

impl<F: FpgaFabric> AxiGpio<F> {
    pub fn new(fabric: F, base_addr: u64) -> Self {
        Self {
            fabric,
            base_addr,
            tri_shadow: 0xFFFF_FFFF,
            data_shadow: 0,
            irq_mask: 0,
            last_input: 0,
        }
    }
    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }
    fn read_reg(&self, offset: u32) -> Result<u32, HalError> {
        self.fabric.read_reg(self.base_addr, offset)
    }
    fn write_reg(&mut self, offset: u32, value: u32) -> Result<(), HalError> {
        self.fabric.write_reg(self.base_addr, offset, value)
    }
}

impl<F: FpgaFabric> GpioController for AxiGpio<F> {
    fn configure(&mut self, pin: u8, config: GpioConfig) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_CHANNEL)?;
        let mask = crate::gpio::pin_mask(pin);
        match config.mode {
            GpioPinMode::Output | GpioPinMode::OpenDrain => {
                self.tri_shadow &= !mask;
            }
            GpioPinMode::Input => {
                self.tri_shadow |= mask;
            }
            GpioPinMode::AlternateFunction(_) | GpioPinMode::Analog => {
                return Err(HalError::NotSupported);
            }
        }
        self.write_reg(GPIO_TRI, self.tri_shadow)?;
        if config.interrupt.is_some() {
            self.irq_mask |= mask;
        } else {
            self.irq_mask &= !mask;
        }
        Ok(())
    }

    fn read(&self, pin: u8) -> Result<bool, HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_CHANNEL)?;
        let data = self.read_reg(GPIO_DATA)?;
        Ok(data & crate::gpio::pin_mask(pin) != 0)
    }

    fn set_high(&mut self, pin: u8) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_CHANNEL)?;
        self.data_shadow |= crate::gpio::pin_mask(pin);
        self.write_reg(GPIO_DATA, self.data_shadow)
    }

    fn set_low(&mut self, pin: u8) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_CHANNEL)?;
        self.data_shadow &= !crate::gpio::pin_mask(pin);
        self.write_reg(GPIO_DATA, self.data_shadow)
    }

    fn toggle(&mut self, pin: u8) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_CHANNEL)?;
        self.data_shadow ^= crate::gpio::pin_mask(pin);
        self.write_reg(GPIO_DATA, self.data_shadow)
    }

    fn set_mask(&mut self, set_mask: u32, clear_mask: u32) -> Result<(), HalError> {
        self.data_shadow = (self.data_shadow | set_mask) & !clear_mask;
        self.write_reg(GPIO_DATA, self.data_shadow)
    }

    fn read_all(&self) -> Result<u32, HalError> {
        self.read_reg(GPIO_DATA)
    }

    fn enable_interrupt(&mut self, pin: u8) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_CHANNEL)?;
        self.irq_mask |= crate::gpio::pin_mask(pin);
        self.write_reg(GIER, GIER_GIE)?;
        self.write_reg(IP_IER, IP_CH1)
    }

    fn disable_interrupt(&mut self, pin: u8) -> Result<(), HalError> {
        crate::gpio::validate_pin(pin, PINS_PER_CHANNEL)?;
        self.irq_mask &= !crate::gpio::pin_mask(pin);
        if self.irq_mask == 0 {
            self.write_reg(GIER, 0)?;
            self.write_reg(IP_IER, 0)?;
        }
        Ok(())
    }

    fn irq_handler(&mut self) -> Result<u8, HalError> {
        let isr = self.read_reg(IP_ISR)?;
        if isr & IP_CH1 == 0 {
            return Err(HalError::InterruptError);
        }
        self.write_reg(IP_ISR, IP_CH1)?;
        let current = self.read_reg(GPIO_DATA)?;
        let changed = (current ^ self.last_input) & self.irq_mask;
        self.last_input = current;
        if changed == 0 {
            return Err(HalError::InterruptError);
        }
        Ok(changed.trailing_zeros() as u8)
    }

    fn pin_count(&self) -> u8 {
        PINS_PER_CHANNEL
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.tri_shadow = 0xFFFF_FFFF;
        self.data_shadow = 0;
        self.irq_mask = 0;
        self.last_input = 0;
        self.write_reg(GPIO_TRI, self.tri_shadow)?;
        self.write_reg(GPIO_DATA, 0)?;
        self.write_reg(GIER, 0)?;
        self.write_reg(IP_IER, 0)?;
        let isr = self.read_reg(IP_ISR)?;
        self.write_reg(IP_ISR, isr)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_kernel::hal::{DmaDescriptor, DmaToken, GpioInterruptEdge, GpioPull};

    struct MockFpgaFabric {
        regs: [u32; 128],
    }
    impl MockFpgaFabric {
        fn new() -> Self {
            Self { regs: [0u32; 128] }
        }
    }
    impl FpgaFabric for MockFpgaFabric {
        fn read_reg(&self, _base_addr: u64, offset: u32) -> Result<u32, HalError> {
            let idx = (offset / 4) as usize;
            if idx < self.regs.len() {
                Ok(self.regs[idx])
            } else {
                Err(HalError::OutOfRange)
            }
        }
        fn write_reg(&mut self, _base_addr: u64, offset: u32, value: u32) -> Result<(), HalError> {
            let idx = (offset / 4) as usize;
            if idx < self.regs.len() {
                self.regs[idx] = value;
                Ok(())
            } else {
                Err(HalError::OutOfRange)
            }
        }
        fn dma_start(&mut self, _desc: DmaDescriptor) -> Result<DmaToken, HalError> {
            Err(HalError::NotSupported)
        }
        fn dma_poll(&self, token: DmaToken) -> Result<DmaToken, HalError> {
            Ok(token)
        }
        fn dma_irq_ack(&mut self) -> Result<u8, HalError> {
            Ok(0)
        }
    }

    #[test]
    fn test_axi_gpio_new() {
        let drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        assert_eq!(drv.pin_count(), 32);
    }

    #[test]
    fn test_axi_gpio_configure_output() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        let config = GpioConfig {
            mode: GpioPinMode::Output,
            pull: GpioPull::None,
            interrupt: None,
            debounce_us: 0,
        };
        assert!(drv.configure(5, config).is_ok());
        assert_eq!(drv.tri_shadow & (1 << 5), 0);
    }

    #[test]
    fn test_axi_gpio_set_high_low() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        drv.set_high(3).unwrap();
        assert_ne!(drv.data_shadow & (1 << 3), 0);
        drv.set_low(3).unwrap();
        assert_eq!(drv.data_shadow & (1 << 3), 0);
    }

    #[test]
    fn test_axi_gpio_reset() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        drv.set_high(0).unwrap();
        drv.reset().unwrap();
        assert_eq!(drv.data_shadow, 0);
    }

    #[test]
    fn test_axi_gpio_pin_out_of_range() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        assert_eq!(drv.set_high(32), Err(HalError::OutOfRange));
    }

    #[test]
    fn test_axi_gpio_constructor_fields() {
        let drv = AxiGpio::new(MockFpgaFabric::new(), 0xABCD_0000);
        assert_eq!(drv.base_addr(), 0xABCD_0000);
        assert_eq!(drv.pin_count(), 32);
        assert_eq!(drv.tri_shadow, 0xFFFF_FFFF); // All inputs by default
        assert_eq!(drv.data_shadow, 0);
        assert_eq!(drv.irq_mask, 0);
        assert_eq!(drv.last_input, 0);
    }

    #[test]
    fn test_axi_gpio_configure_input() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        // First set as output to clear tri bit.
        let out_config = GpioConfig {
            mode: GpioPinMode::Output,
            pull: GpioPull::None,
            interrupt: None,
            debounce_us: 0,
        };
        drv.configure(10, out_config).unwrap();
        assert_eq!(drv.tri_shadow & (1 << 10), 0);

        // Then set back to input.
        let in_config = GpioConfig {
            mode: GpioPinMode::Input,
            pull: GpioPull::None,
            interrupt: None,
            debounce_us: 0,
        };
        drv.configure(10, in_config).unwrap();
        assert_ne!(drv.tri_shadow & (1 << 10), 0);
    }

    #[test]
    fn test_axi_gpio_alternate_function_not_supported() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        let config = GpioConfig {
            mode: GpioPinMode::AlternateFunction(0),
            pull: GpioPull::None,
            interrupt: None,
            debounce_us: 0,
        };
        assert_eq!(drv.configure(0, config), Err(HalError::NotSupported));
    }

    #[test]
    fn test_axi_gpio_analog_not_supported() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        let config = GpioConfig {
            mode: GpioPinMode::Analog,
            pull: GpioPull::None,
            interrupt: None,
            debounce_us: 0,
        };
        assert_eq!(drv.configure(0, config), Err(HalError::NotSupported));
    }

    #[test]
    fn test_axi_gpio_toggle() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        drv.toggle(7).unwrap();
        assert_ne!(drv.data_shadow & (1 << 7), 0);
        drv.toggle(7).unwrap();
        assert_eq!(drv.data_shadow & (1 << 7), 0);
    }

    #[test]
    fn test_axi_gpio_set_mask() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        drv.set_mask(0x0F, 0x00).unwrap();
        assert_eq!(drv.data_shadow, 0x0F);
        drv.set_mask(0x00, 0x03).unwrap();
        assert_eq!(drv.data_shadow, 0x0C);
    }

    #[test]
    fn test_axi_gpio_interrupt_enable_sets_irq_mask() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        // enable_interrupt writes to GIER and IP_IER registers.
        drv.enable_interrupt(4).unwrap();
        assert_ne!(drv.irq_mask & (1 << 4), 0);
    }

    #[test]
    fn test_axi_gpio_interrupt_disable_clears_irq_mask() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        drv.enable_interrupt(4).unwrap();
        assert_ne!(drv.irq_mask & (1 << 4), 0);
        // disable_interrupt clears the bit; if all bits zero, it writes to regs.
        drv.disable_interrupt(4).unwrap();
        assert_eq!(drv.irq_mask & (1 << 4), 0);
    }

    #[test]
    fn test_axi_gpio_interrupt_config_via_configure() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        let config = GpioConfig {
            mode: GpioPinMode::Input,
            pull: GpioPull::None,
            interrupt: Some(GpioInterruptEdge::Rising),
            debounce_us: 0,
        };
        drv.configure(2, config).unwrap();
        assert_ne!(drv.irq_mask & (1 << 2), 0);

        // Configure without interrupt clears the irq_mask bit.
        let config_no_irq = GpioConfig {
            mode: GpioPinMode::Input,
            pull: GpioPull::None,
            interrupt: None,
            debounce_us: 0,
        };
        drv.configure(2, config_no_irq).unwrap();
        assert_eq!(drv.irq_mask & (1 << 2), 0);
    }

    #[test]
    fn test_axi_gpio_channel1_data_register() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        // set_high writes to GPIO_DATA (offset 0x0000) — channel 1.
        drv.set_high(0).unwrap();
        assert_eq!(drv.data_shadow & 1, 1);
        // Verify the fabric register at offset 0x0000 was written.
        let val = drv.read_reg(GPIO_DATA).unwrap();
        assert_eq!(val & 1, 1);
    }

    #[test]
    fn test_axi_gpio_tri_state_register() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        // Default: all inputs (tri = 0xFFFF_FFFF).
        assert_eq!(drv.tri_shadow, 0xFFFF_FFFF);

        // Configure pin 0 as output.
        let config = GpioConfig {
            mode: GpioPinMode::Output,
            pull: GpioPull::None,
            interrupt: None,
            debounce_us: 0,
        };
        drv.configure(0, config).unwrap();
        // Verify tri register written via fabric.
        let tri = drv.read_reg(GPIO_TRI).unwrap();
        assert_eq!(tri & 1, 0); // Pin 0 is now output (tri bit = 0).
    }

    #[test]
    fn test_axi_gpio_reset_restores_defaults() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        drv.set_high(5).unwrap();
        drv.enable_interrupt(3).unwrap();
        let config = GpioConfig {
            mode: GpioPinMode::Output,
            pull: GpioPull::None,
            interrupt: None,
            debounce_us: 0,
        };
        drv.configure(5, config).unwrap();

        drv.reset().unwrap();
        assert_eq!(drv.tri_shadow, 0xFFFF_FFFF);
        assert_eq!(drv.data_shadow, 0);
        assert_eq!(drv.irq_mask, 0);
        assert_eq!(drv.last_input, 0);
    }

    #[test]
    fn test_axi_gpio_read_pin() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        // Write a value into the fabric's GPIO_DATA register directly, then read.
        drv.write_reg(GPIO_DATA, 0x0000_0010).unwrap();
        assert!(drv.read(4).unwrap());
        assert!(!drv.read(3).unwrap());
    }

    #[test]
    fn test_axi_gpio_open_drain_sets_output() {
        let mut drv = AxiGpio::new(MockFpgaFabric::new(), 0x4012_0000);
        let config = GpioConfig {
            mode: GpioPinMode::OpenDrain,
            pull: GpioPull::None,
            interrupt: None,
            debounce_us: 0,
        };
        drv.configure(15, config).unwrap();
        assert_eq!(drv.tri_shadow & (1 << 15), 0); // output
    }
}
