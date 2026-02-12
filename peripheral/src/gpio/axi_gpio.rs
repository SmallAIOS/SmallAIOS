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
    use smallaios_kernel::hal::{DmaDescriptor, DmaToken, GpioPull};

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
}
