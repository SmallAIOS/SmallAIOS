// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the GPIO peripheral subsystem.
//!
//! Tests exercise a mock GPIO controller implementing the
//! `smallaios_kernel::hal::GpioController` trait, covering pin
//! configuration, digital read/write, toggle, bitmask operations,
//! interrupt enable/disable, and interrupt acknowledgement.

#![cfg(feature = "gpio")]

extern crate alloc;

use smallaios_kernel::hal::{
    GpioConfig, GpioController, GpioInterruptEdge, GpioPinMode, GpioPull, HalError,
};
use smallaios_peripheral::PeripheralType;

// ---------------------------------------------------------------------------
// Mock GPIO controller
// ---------------------------------------------------------------------------

const NUM_PINS: usize = 32;

/// Mock GPIO controller for testing.
#[allow(dead_code)]
struct MockGpio {
    initialized: bool,
    pin_modes: [GpioPinMode; NUM_PINS],
    pin_pulls: [GpioPull; NUM_PINS],
    pin_interrupt: [Option<GpioInterruptEdge>; NUM_PINS],
    pin_values: u32,
    interrupt_enabled: u32,
    pending_interrupt: Option<u8>,
}

impl MockGpio {
    fn new() -> Self {
        Self {
            initialized: true,
            pin_modes: [GpioPinMode::Input; NUM_PINS],
            pin_pulls: [GpioPull::None; NUM_PINS],
            pin_interrupt: [None; NUM_PINS],
            pin_values: 0,
            interrupt_enabled: 0,
            pending_interrupt: None,
        }
    }

    /// Simulate an external signal driving a pin high or low.
    fn drive_pin(&mut self, pin: u8, high: bool) {
        if high {
            self.pin_values |= 1 << pin;
        } else {
            self.pin_values &= !(1 << pin);
        }
    }

    /// Simulate a hardware interrupt on a pin.
    fn simulate_interrupt(&mut self, pin: u8) {
        if pin < NUM_PINS as u8 && (self.interrupt_enabled & (1 << pin)) != 0 {
            self.pending_interrupt = Some(pin);
        }
    }
}

impl GpioController for MockGpio {
    fn configure(&mut self, pin: u8, config: GpioConfig) -> Result<(), HalError> {
        if pin >= NUM_PINS as u8 {
            return Err(HalError::OutOfRange);
        }
        self.pin_modes[pin as usize] = config.mode;
        self.pin_pulls[pin as usize] = config.pull;
        self.pin_interrupt[pin as usize] = config.interrupt;
        if config.interrupt.is_some() {
            self.interrupt_enabled |= 1 << pin;
        }
        Ok(())
    }

    fn read(&self, pin: u8) -> Result<bool, HalError> {
        if pin >= NUM_PINS as u8 {
            return Err(HalError::OutOfRange);
        }
        Ok((self.pin_values & (1 << pin)) != 0)
    }

    fn set_high(&mut self, pin: u8) -> Result<(), HalError> {
        if pin >= NUM_PINS as u8 {
            return Err(HalError::OutOfRange);
        }
        self.pin_values |= 1 << pin;
        Ok(())
    }

    fn set_low(&mut self, pin: u8) -> Result<(), HalError> {
        if pin >= NUM_PINS as u8 {
            return Err(HalError::OutOfRange);
        }
        self.pin_values &= !(1 << pin);
        Ok(())
    }

    fn toggle(&mut self, pin: u8) -> Result<(), HalError> {
        if pin >= NUM_PINS as u8 {
            return Err(HalError::OutOfRange);
        }
        self.pin_values ^= 1 << pin;
        Ok(())
    }

    fn set_mask(&mut self, set_mask: u32, clear_mask: u32) -> Result<(), HalError> {
        self.pin_values |= set_mask;
        self.pin_values &= !clear_mask;
        Ok(())
    }

    fn read_all(&self) -> Result<u32, HalError> {
        Ok(self.pin_values)
    }

    fn enable_interrupt(&mut self, pin: u8) -> Result<(), HalError> {
        if pin >= NUM_PINS as u8 {
            return Err(HalError::OutOfRange);
        }
        self.interrupt_enabled |= 1 << pin;
        Ok(())
    }

    fn disable_interrupt(&mut self, pin: u8) -> Result<(), HalError> {
        if pin >= NUM_PINS as u8 {
            return Err(HalError::OutOfRange);
        }
        self.interrupt_enabled &= !(1 << pin);
        Ok(())
    }

    fn irq_handler(&mut self) -> Result<u8, HalError> {
        if let Some(pin) = self.pending_interrupt.take() {
            Ok(pin)
        } else {
            Err(HalError::InterruptError)
        }
    }

    fn pin_count(&self) -> u8 {
        NUM_PINS as u8
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.pin_modes = [GpioPinMode::Input; NUM_PINS];
        self.pin_pulls = [GpioPull::None; NUM_PINS];
        self.pin_interrupt = [None; NUM_PINS];
        self.pin_values = 0;
        self.interrupt_enabled = 0;
        self.pending_interrupt = None;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_gpio_configure_input_pullup() {
    let mut gpio = MockGpio::new();
    let config = GpioConfig {
        mode: GpioPinMode::Input,
        pull: GpioPull::PullUp,
        interrupt: None,
        debounce_us: 0,
    };
    assert!(gpio.configure(5, config).is_ok());
    assert_eq!(gpio.pin_modes[5], GpioPinMode::Input);
    assert_eq!(gpio.pin_pulls[5], GpioPull::PullUp);
}

#[test]
fn test_gpio_configure_output() {
    let mut gpio = MockGpio::new();
    let config = GpioConfig {
        mode: GpioPinMode::Output,
        pull: GpioPull::None,
        interrupt: None,
        debounce_us: 0,
    };
    gpio.configure(10, config).unwrap();
    assert_eq!(gpio.pin_modes[10], GpioPinMode::Output);
}

#[test]
fn test_gpio_set_and_read() {
    let mut gpio = MockGpio::new();
    let config = GpioConfig {
        mode: GpioPinMode::Output,
        pull: GpioPull::None,
        interrupt: None,
        debounce_us: 0,
    };
    gpio.configure(0, config).unwrap();

    gpio.set_high(0).unwrap();
    assert_eq!(gpio.read(0).unwrap(), true);

    gpio.set_low(0).unwrap();
    assert_eq!(gpio.read(0).unwrap(), false);
}

#[test]
fn test_gpio_toggle() {
    let mut gpio = MockGpio::new();
    let config = GpioConfig {
        mode: GpioPinMode::Output,
        pull: GpioPull::None,
        interrupt: None,
        debounce_us: 0,
    };
    gpio.configure(7, config).unwrap();

    assert_eq!(gpio.read(7).unwrap(), false);
    gpio.toggle(7).unwrap();
    assert_eq!(gpio.read(7).unwrap(), true);
    gpio.toggle(7).unwrap();
    assert_eq!(gpio.read(7).unwrap(), false);
}

#[test]
fn test_gpio_read_all_and_set_mask() {
    let mut gpio = MockGpio::new();

    gpio.set_high(0).unwrap();
    gpio.set_high(4).unwrap();
    gpio.set_high(8).unwrap();

    let all = gpio.read_all().unwrap();
    assert_eq!(all, (1 << 0) | (1 << 4) | (1 << 8));

    // Set pins 1,2 and clear pin 0 atomically.
    gpio.set_mask(0b0110, 0b0001).unwrap();
    let all = gpio.read_all().unwrap();
    assert!((all & 0b0001) == 0); // Pin 0 cleared.
    assert!((all & 0b0110) == 0b0110); // Pins 1,2 set.
}

#[test]
fn test_gpio_interrupt_enable_and_handler() {
    let mut gpio = MockGpio::new();
    let config = GpioConfig {
        mode: GpioPinMode::Input,
        pull: GpioPull::PullDown,
        interrupt: Some(GpioInterruptEdge::Rising),
        debounce_us: 50,
    };
    gpio.configure(3, config).unwrap();

    // Simulate interrupt on pin 3.
    gpio.simulate_interrupt(3);
    let pin = gpio.irq_handler().unwrap();
    assert_eq!(pin, 3);

    // No pending interrupt now.
    assert_eq!(gpio.irq_handler(), Err(HalError::InterruptError));
}

#[test]
fn test_gpio_interrupt_disabled_no_trigger() {
    let mut gpio = MockGpio::new();
    let config = GpioConfig {
        mode: GpioPinMode::Input,
        pull: GpioPull::None,
        interrupt: None,
        debounce_us: 0,
    };
    gpio.configure(3, config).unwrap();
    gpio.disable_interrupt(3).unwrap();

    // Simulate interrupt; should not register because interrupt is disabled.
    gpio.simulate_interrupt(3);
    assert_eq!(gpio.irq_handler(), Err(HalError::InterruptError));
}

#[test]
fn test_gpio_external_drive() {
    let mut gpio = MockGpio::new();
    let config = GpioConfig {
        mode: GpioPinMode::Input,
        pull: GpioPull::None,
        interrupt: None,
        debounce_us: 0,
    };
    gpio.configure(15, config).unwrap();

    assert_eq!(gpio.read(15).unwrap(), false);
    gpio.drive_pin(15, true);
    assert_eq!(gpio.read(15).unwrap(), true);
    gpio.drive_pin(15, false);
    assert_eq!(gpio.read(15).unwrap(), false);
}

#[test]
fn test_gpio_out_of_range() {
    let mut gpio = MockGpio::new();
    assert_eq!(
        gpio.configure(
            32,
            GpioConfig {
                mode: GpioPinMode::Input,
                pull: GpioPull::None,
                interrupt: None,
                debounce_us: 0,
            },
        ),
        Err(HalError::OutOfRange)
    );
    assert_eq!(gpio.read(32), Err(HalError::OutOfRange));
    assert_eq!(gpio.set_high(32), Err(HalError::OutOfRange));
}

#[test]
fn test_gpio_pin_count() {
    let gpio = MockGpio::new();
    assert_eq!(gpio.pin_count(), 32);
}

#[test]
fn test_gpio_reset() {
    let mut gpio = MockGpio::new();
    gpio.set_high(0).unwrap();
    gpio.set_high(1).unwrap();
    gpio.enable_interrupt(0).unwrap();

    gpio.reset().unwrap();
    assert_eq!(gpio.read_all().unwrap(), 0);
    assert_eq!(gpio.interrupt_enabled, 0);
}

#[test]
fn test_gpio_peripheral_type() {
    assert_eq!(PeripheralType::from_u8(2), Some(PeripheralType::Gpio));
    assert_eq!(PeripheralType::Gpio as u8, 2);
}
