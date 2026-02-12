// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPIO bitbang I2C fallback driver.
//!
//! Implements the [`I2cController`] trait by bit-banging the I2C protocol
//! over two GPIO pins (SDA, SCL). Supports standard mode (100 kHz) only.
//!
//! This driver is useful as a last-resort fallback when no dedicated I2C
//! hardware is available, or for bus recovery when the hardware controller
//! cannot recover a stuck bus.
//!
//! # Protocol implementation
//!
//! The driver implements the full I2C master protocol:
//! - START condition: SDA goes low while SCL is high
//! - STOP condition:  SDA goes high while SCL is high
//! - Data transfer:   MSB first, 8 bits per byte
//! - ACK/NACK:        9th clock pulse for acknowledgment
//! - Repeated START:  for write-then-read transactions

#![allow(dead_code)]

use smallaios_kernel::hal::{
    GpioConfig, GpioController, GpioPinMode, GpioPull, HalError, I2cConfig, I2cController, I2cSpeed,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Half-period delay count for standard mode (100 kHz).
/// At 100 kHz, the period is 10 us, half-period = 5 us.
/// The actual delay depends on the platform's busy-loop speed.
const STD_HALF_PERIOD_US: u32 = 5;

/// Number of clock pulses for bus recovery.
const BUS_RECOVERY_CLOCKS: u8 = 9;

// ---------------------------------------------------------------------------
// Driver implementation
// ---------------------------------------------------------------------------

/// GPIO bitbang I2C driver.
///
/// Generic over a [`GpioController`] implementation that provides
/// pin read/write access for the SDA and SCL lines.
pub struct BitbangI2c<G: GpioController> {
    /// GPIO controller.
    gpio: G,
    /// SDA pin number.
    sda_pin: u8,
    /// SCL pin number.
    scl_pin: u8,
    /// Whether the driver has been initialized.
    configured: bool,
    /// Stored configuration.
    config: Option<I2cConfig>,
    /// Half-period delay in microseconds.
    half_period_us: u32,
}

impl<G: GpioController> BitbangI2c<G> {
    /// Create a new bitbang I2C driver.
    ///
    /// `sda_pin` and `scl_pin` are GPIO pin numbers on the given controller.
    pub fn new(gpio: G, sda_pin: u8, scl_pin: u8) -> Self {
        Self {
            gpio,
            sda_pin,
            scl_pin,
            configured: false,
            config: None,
            half_period_us: STD_HALF_PERIOD_US,
        }
    }

    /// Return the SDA pin number.
    pub fn sda_pin(&self) -> u8 {
        self.sda_pin
    }

    /// Return the SCL pin number.
    pub fn scl_pin(&self) -> u8 {
        self.scl_pin
    }

    /// Return whether the driver is configured.
    pub fn is_configured(&self) -> bool {
        self.configured
    }

    /// Busy-loop delay for half an SCL period.
    ///
    /// In a real kernel, this would use a calibrated timer or cycle counter.
    /// Here it's a no-op for host testing.
    fn delay_half_period(&self) {
        // Platform-specific delay would go here.
        // For host testing, this is instantaneous.
        let _ = self.half_period_us;
    }

    /// Set SDA high (release — open-drain with pull-up).
    fn sda_high(&mut self) -> Result<(), HalError> {
        // Configure as input (high-Z) to let the pull-up pull it high.
        self.gpio.configure(
            self.sda_pin,
            GpioConfig {
                mode: GpioPinMode::Input,
                pull: GpioPull::PullUp,
                interrupt: None,
                debounce_us: 0,
            },
        )
    }

    /// Set SDA low (drive low — open-drain).
    fn sda_low(&mut self) -> Result<(), HalError> {
        self.gpio.configure(
            self.sda_pin,
            GpioConfig {
                mode: GpioPinMode::OpenDrain,
                pull: GpioPull::None,
                interrupt: None,
                debounce_us: 0,
            },
        )?;
        self.gpio.set_low(self.sda_pin)
    }

    /// Read the SDA pin level.
    fn sda_read(&self) -> Result<bool, HalError> {
        self.gpio.read(self.sda_pin)
    }

    /// Set SCL high (release — open-drain with pull-up).
    fn scl_high(&mut self) -> Result<(), HalError> {
        self.gpio.configure(
            self.scl_pin,
            GpioConfig {
                mode: GpioPinMode::Input,
                pull: GpioPull::PullUp,
                interrupt: None,
                debounce_us: 0,
            },
        )
    }

    /// Set SCL low (drive low).
    fn scl_low(&mut self) -> Result<(), HalError> {
        self.gpio.configure(
            self.scl_pin,
            GpioConfig {
                mode: GpioPinMode::OpenDrain,
                pull: GpioPull::None,
                interrupt: None,
                debounce_us: 0,
            },
        )?;
        self.gpio.set_low(self.scl_pin)
    }

    /// Generate a START condition.
    ///
    /// Precondition: SCL and SDA are both high (bus idle).
    /// Sequence: SDA goes low while SCL is high, then SCL goes low.
    fn start(&mut self) -> Result<(), HalError> {
        self.sda_high()?;
        self.scl_high()?;
        self.delay_half_period();

        self.sda_low()?;
        self.delay_half_period();

        self.scl_low()?;
        self.delay_half_period();
        Ok(())
    }

    /// Generate a STOP condition.
    ///
    /// Sequence: SCL goes high while SDA is low, then SDA goes high.
    fn stop(&mut self) -> Result<(), HalError> {
        self.sda_low()?;
        self.delay_half_period();

        self.scl_high()?;
        self.delay_half_period();

        self.sda_high()?;
        self.delay_half_period();
        Ok(())
    }

    /// Transmit one byte MSB-first and return the ACK bit.
    ///
    /// Returns `true` if ACK was received (SDA low on 9th clock).
    fn send_byte(&mut self, byte: u8) -> Result<bool, HalError> {
        for bit_idx in (0..8).rev() {
            let bit = (byte >> bit_idx) & 1;
            if bit != 0 {
                self.sda_high()?;
            } else {
                self.sda_low()?;
            }
            self.delay_half_period();

            self.scl_high()?;
            self.delay_half_period();

            self.scl_low()?;
        }

        // Release SDA for ACK.
        self.sda_high()?;
        self.delay_half_period();

        // Clock in the ACK bit.
        self.scl_high()?;
        self.delay_half_period();
        let ack = !self.sda_read()?; // ACK = SDA low
        self.scl_low()?;
        self.delay_half_period();

        Ok(ack)
    }

    /// Receive one byte MSB-first.
    ///
    /// Sends ACK if `ack` is true, NACK if false.
    fn recv_byte(&mut self, ack: bool) -> Result<u8, HalError> {
        // Release SDA so the target can drive it.
        self.sda_high()?;

        let mut byte: u8 = 0;
        for _ in 0..8 {
            self.delay_half_period();
            self.scl_high()?;
            self.delay_half_period();

            let bit = self.sda_read()?;
            byte = (byte << 1) | (bit as u8);

            self.scl_low()?;
        }

        // Send ACK or NACK.
        if ack {
            self.sda_low()?; // ACK
        } else {
            self.sda_high()?; // NACK
        }
        self.delay_half_period();

        self.scl_high()?;
        self.delay_half_period();
        self.scl_low()?;
        self.delay_half_period();

        // Release SDA.
        self.sda_high()?;

        Ok(byte)
    }

    /// Send the address byte (7-bit address << 1 | R/W bit).
    fn send_address(&mut self, addr: u16, read: bool) -> Result<(), HalError> {
        let addr_byte = ((addr as u8) << 1) | (read as u8);
        let ack = self.send_byte(addr_byte)?;
        if !ack {
            self.stop()?;
            return Err(HalError::NackReceived);
        }
        Ok(())
    }
}

impl<G: GpioController> I2cController for BitbangI2c<G> {
    fn init(&mut self, config: I2cConfig) -> Result<(), HalError> {
        if config.timeout_us == 0 {
            return Err(HalError::InvalidConfig);
        }

        // Only standard mode is supported for bitbang.
        if config.speed != I2cSpeed::Standard {
            return Err(HalError::NotSupported);
        }

        // Configure both pins as inputs with pull-ups (bus idle state).
        self.sda_high()?;
        self.scl_high()?;

        self.config = Some(config);
        self.configured = true;
        Ok(())
    }

    fn write(&mut self, addr: u16, data: &[u8]) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        let config = self.config.as_ref().unwrap();
        crate::i2c::validate_addr(addr, config.address_mode)?;

        self.start()?;
        self.send_address(addr, false)?;

        for &byte in data {
            let ack = self.send_byte(byte)?;
            if !ack {
                self.stop()?;
                return Err(HalError::NackReceived);
            }
        }

        self.stop()
    }

    fn read(&mut self, addr: u16, buf: &mut [u8]) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        let config = self.config.as_ref().unwrap();
        crate::i2c::validate_addr(addr, config.address_mode)?;

        self.start()?;
        self.send_address(addr, true)?;

        for i in 0..buf.len() {
            let last = i == buf.len() - 1;
            buf[i] = self.recv_byte(!last)?;
        }

        self.stop()
    }

    fn write_read(
        &mut self,
        addr: u16,
        write_data: &[u8],
        read_buf: &mut [u8],
    ) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        let config = self.config.as_ref().unwrap();
        crate::i2c::validate_addr(addr, config.address_mode)?;

        // Write phase.
        self.start()?;
        self.send_address(addr, false)?;

        for &byte in write_data {
            let ack = self.send_byte(byte)?;
            if !ack {
                self.stop()?;
                return Err(HalError::NackReceived);
            }
        }

        // Repeated START for read phase.
        self.start()?;
        self.send_address(addr, true)?;

        for i in 0..read_buf.len() {
            let last = i == read_buf.len() - 1;
            read_buf[i] = self.recv_byte(!last)?;
        }

        self.stop()
    }

    fn bus_recovery(&mut self) -> Result<(), HalError> {
        // Release SDA, then clock SCL 9 times.
        self.sda_high()?;

        for _ in 0..BUS_RECOVERY_CLOCKS {
            self.scl_high()?;
            self.delay_half_period();

            // Check if SDA is released.
            if self.sda_read()? {
                // SDA is high — generate STOP to reset bus.
                self.stop()?;
                return Ok(());
            }

            self.scl_low()?;
            self.delay_half_period();
        }

        // After 9 clocks, try a STOP anyway.
        self.stop()
    }

    fn reset(&mut self) -> Result<(), HalError> {
        // Release both lines.
        self.sda_high()?;
        self.scl_high()?;
        self.configured = false;
        self.config = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_kernel::hal::{GpioConfig, I2cAddressMode};

    /// Mock GPIO controller for testing bitbang I2C.
    struct MockGpio {
        /// Pin states: true = high, false = low.
        pins: [bool; 32],
        /// Pin modes.
        modes: [GpioPinMode; 32],
        /// Number of configure calls (for verification).
        configure_count: u32,
        /// Simulated ACK response for send_byte tests.
        ack_response: bool,
    }

    impl MockGpio {
        fn new() -> Self {
            Self {
                pins: [true; 32], // All pins start high (pull-up)
                modes: [GpioPinMode::Input; 32],
                configure_count: 0,
                ack_response: true,
            }
        }

        fn with_ack(ack: bool) -> Self {
            Self {
                pins: [true; 32],
                modes: [GpioPinMode::Input; 32],
                configure_count: 0,
                ack_response: ack,
            }
        }
    }

    impl GpioController for MockGpio {
        fn configure(&mut self, pin: u8, config: GpioConfig) -> Result<(), HalError> {
            if pin >= 32 {
                return Err(HalError::OutOfRange);
            }
            self.modes[pin as usize] = config.mode;
            self.configure_count += 1;

            // Simulate pull-up behavior: input with pull-up → pin goes high.
            if matches!(config.mode, GpioPinMode::Input) && config.pull == GpioPull::PullUp {
                self.pins[pin as usize] = true;
            }

            Ok(())
        }

        fn read(&self, pin: u8) -> Result<bool, HalError> {
            if pin >= 32 {
                return Err(HalError::OutOfRange);
            }
            // For ACK simulation: when reading SDA during ACK phase,
            // return !ack_response (ACK = SDA low = false).
            Ok(self.pins[pin as usize])
        }

        fn set_high(&mut self, pin: u8) -> Result<(), HalError> {
            if pin >= 32 {
                return Err(HalError::OutOfRange);
            }
            self.pins[pin as usize] = true;
            Ok(())
        }

        fn set_low(&mut self, pin: u8) -> Result<(), HalError> {
            if pin >= 32 {
                return Err(HalError::OutOfRange);
            }
            self.pins[pin as usize] = false;
            Ok(())
        }

        fn toggle(&mut self, pin: u8) -> Result<(), HalError> {
            if pin >= 32 {
                return Err(HalError::OutOfRange);
            }
            self.pins[pin as usize] = !self.pins[pin as usize];
            Ok(())
        }

        fn set_mask(&mut self, set_mask: u32, clear_mask: u32) -> Result<(), HalError> {
            for i in 0..32 {
                if set_mask & (1 << i) != 0 {
                    self.pins[i] = true;
                }
                if clear_mask & (1 << i) != 0 {
                    self.pins[i] = false;
                }
            }
            Ok(())
        }

        fn read_all(&self) -> Result<u32, HalError> {
            let mut val = 0u32;
            for i in 0..32 {
                if self.pins[i] {
                    val |= 1 << i;
                }
            }
            Ok(val)
        }

        fn enable_interrupt(&mut self, _pin: u8) -> Result<(), HalError> {
            Ok(())
        }

        fn disable_interrupt(&mut self, _pin: u8) -> Result<(), HalError> {
            Ok(())
        }

        fn irq_handler(&mut self) -> Result<u8, HalError> {
            Ok(0)
        }

        fn pin_count(&self) -> u8 {
            32
        }

        fn reset(&mut self) -> Result<(), HalError> {
            self.pins = [true; 32];
            self.modes = [GpioPinMode::Input; 32];
            Ok(())
        }
    }

    fn make_config() -> I2cConfig {
        I2cConfig {
            speed: I2cSpeed::Standard,
            address_mode: I2cAddressMode::SevenBit,
            clock_stretching: false,
            timeout_us: 10_000,
        }
    }

    #[test]
    fn test_bitbang_new() {
        let gpio = MockGpio::new();
        let drv = BitbangI2c::new(gpio, 2, 3);
        assert_eq!(drv.sda_pin(), 2);
        assert_eq!(drv.scl_pin(), 3);
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_bitbang_init() {
        let gpio = MockGpio::new();
        let mut drv = BitbangI2c::new(gpio, 2, 3);
        assert!(drv.init(make_config()).is_ok());
        assert!(drv.is_configured());
    }

    #[test]
    fn test_bitbang_init_fast_mode_unsupported() {
        let gpio = MockGpio::new();
        let mut drv = BitbangI2c::new(gpio, 2, 3);
        let mut config = make_config();
        config.speed = I2cSpeed::Fast;
        assert_eq!(drv.init(config), Err(HalError::NotSupported));
    }

    #[test]
    fn test_bitbang_init_zero_timeout() {
        let gpio = MockGpio::new();
        let mut drv = BitbangI2c::new(gpio, 2, 3);
        let mut config = make_config();
        config.timeout_us = 0;
        assert_eq!(drv.init(config), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_bitbang_write_not_configured() {
        let gpio = MockGpio::new();
        let mut drv = BitbangI2c::new(gpio, 2, 3);
        assert_eq!(drv.write(0x50, &[0x00]), Err(HalError::InitFailed));
    }

    #[test]
    fn test_bitbang_start_stop_sequence() {
        let gpio = MockGpio::new();
        let mut drv = BitbangI2c::new(gpio, 2, 3);
        // Start should work even without init (low-level helper).
        assert!(drv.start().is_ok());
        assert!(drv.stop().is_ok());
    }

    #[test]
    fn test_bitbang_send_byte_ack() {
        let gpio = MockGpio::new();
        let mut drv = BitbangI2c::new(gpio, 2, 3);
        drv.init(make_config()).unwrap();

        // MockGpio returns true for SDA read (NACK: SDA high).
        // In real hardware this means NACK.
        let ack = drv.send_byte(0xA0).unwrap();
        assert!(!ack); // SDA high = NACK
    }

    #[test]
    fn test_bitbang_recv_byte() {
        let gpio = MockGpio::new();
        let mut drv = BitbangI2c::new(gpio, 2, 3);
        drv.init(make_config()).unwrap();

        // MockGpio returns true for all SDA reads → all bits are 1 → byte = 0xFF.
        let byte = drv.recv_byte(true).unwrap();
        assert_eq!(byte, 0xFF);
    }

    #[test]
    fn test_bitbang_bus_recovery() {
        let gpio = MockGpio::new();
        let mut drv = BitbangI2c::new(gpio, 2, 3);
        drv.init(make_config()).unwrap();
        // SDA reads as high (bus idle), so recovery should succeed quickly.
        assert!(drv.bus_recovery().is_ok());
    }

    #[test]
    fn test_bitbang_reset() {
        let gpio = MockGpio::new();
        let mut drv = BitbangI2c::new(gpio, 2, 3);
        drv.init(make_config()).unwrap();
        assert!(drv.is_configured());
        assert!(drv.reset().is_ok());
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_bitbang_write_invalid_addr() {
        let gpio = MockGpio::new();
        let mut drv = BitbangI2c::new(gpio, 2, 3);
        drv.init(make_config()).unwrap();
        assert_eq!(drv.write(0x00, &[0xAB]), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_bitbang_pin_out_of_range() {
        let gpio = MockGpio::new();
        let mut drv = BitbangI2c::new(gpio, 32, 33);
        // Init will fail because pin 32 is out of range for MockGpio.
        assert_eq!(drv.init(make_config()), Err(HalError::OutOfRange));
    }
}
