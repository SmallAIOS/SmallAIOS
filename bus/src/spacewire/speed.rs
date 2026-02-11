// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SpaceWire link speed configuration (2-400 Mbps).
//!
//! SpaceWire links operate at configurable data rates from 2 Mbps to
//! 400 Mbps. The speed is determined by the transmit clock frequency.
//! During link initialization, a low startup rate (typically 10 Mbps)
//! is used, then the link transitions to the configured operational rate.

use crate::BusError;

/// Minimum link speed in Mbps.
pub const MIN_SPEED_MBPS: u32 = 2;

/// Maximum link speed in Mbps.
pub const MAX_SPEED_MBPS: u32 = 400;

/// Default startup speed in Mbps (used during initialization).
pub const DEFAULT_STARTUP_SPEED_MBPS: u32 = 10;

/// SpaceWire link speed configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinkSpeed {
    /// Operational data rate in Mbps.
    operational_mbps: u32,
    /// Startup data rate in Mbps (used during link initialization).
    startup_mbps: u32,
    /// Transmit clock divisor (derived from system clock and target speed).
    tx_divisor: u16,
}

impl LinkSpeed {
    /// Create a new link speed configuration.
    ///
    /// `operational_mbps`: desired operational speed (2-400 Mbps).
    /// `sys_clock_mhz`: system clock frequency in MHz.
    pub fn new(operational_mbps: u32, sys_clock_mhz: u32) -> Result<Self, BusError> {
        if !(MIN_SPEED_MBPS..=MAX_SPEED_MBPS).contains(&operational_mbps) {
            return Err(BusError::OutOfRange);
        }
        if sys_clock_mhz == 0 {
            return Err(BusError::OutOfRange);
        }

        // SpaceWire uses DDR signaling, so the clock frequency is half the data rate.
        // tx_divisor = sys_clock / (data_rate / 2) = 2 * sys_clock / data_rate
        let divisor = (2 * sys_clock_mhz) / operational_mbps;
        let divisor = if divisor == 0 { 1 } else { divisor };

        Ok(Self {
            operational_mbps,
            startup_mbps: DEFAULT_STARTUP_SPEED_MBPS.min(operational_mbps),
            tx_divisor: divisor as u16,
        })
    }

    /// Create a speed configuration with a custom startup speed.
    pub fn with_startup(
        operational_mbps: u32,
        startup_mbps: u32,
        sys_clock_mhz: u32,
    ) -> Result<Self, BusError> {
        if !(MIN_SPEED_MBPS..=MAX_SPEED_MBPS).contains(&startup_mbps) {
            return Err(BusError::OutOfRange);
        }
        if startup_mbps > operational_mbps {
            return Err(BusError::OutOfRange);
        }
        let mut speed = Self::new(operational_mbps, sys_clock_mhz)?;
        speed.startup_mbps = startup_mbps;
        Ok(speed)
    }

    /// Returns the operational data rate in Mbps.
    pub fn operational_mbps(&self) -> u32 {
        self.operational_mbps
    }

    /// Returns the startup data rate in Mbps.
    pub fn startup_mbps(&self) -> u32 {
        self.startup_mbps
    }

    /// Returns the transmit clock divisor.
    pub fn tx_divisor(&self) -> u16 {
        self.tx_divisor
    }

    /// Compute the actual data rate given the system clock and divisor.
    pub fn actual_rate_mbps(&self, sys_clock_mhz: u32) -> u32 {
        if self.tx_divisor == 0 {
            return 0;
        }
        (2 * sys_clock_mhz) / self.tx_divisor as u32
    }

    /// Compute the startup clock divisor for the system clock.
    pub fn startup_divisor(&self, sys_clock_mhz: u32) -> u16 {
        let divisor = (2 * sys_clock_mhz) / self.startup_mbps;
        if divisor == 0 { 1 } else { divisor as u16 }
    }

    /// Returns the character transmission time in nanoseconds.
    ///
    /// A SpaceWire character is 10 bits (data char) or 4 bits (control char).
    /// This returns the time for a 10-bit data character.
    pub fn char_time_ns(&self) -> u64 {
        // Time per bit = 1000 / rate_mbps (in ns)
        // Data character = 10 bits
        (10 * 1000) / self.operational_mbps as u64
    }
}

/// Common speed presets.
impl LinkSpeed {
    /// 10 Mbps (common startup rate).
    pub fn preset_10mbps(sys_clock_mhz: u32) -> Result<Self, BusError> {
        Self::new(10, sys_clock_mhz)
    }

    /// 100 Mbps (common operational rate).
    pub fn preset_100mbps(sys_clock_mhz: u32) -> Result<Self, BusError> {
        Self::new(100, sys_clock_mhz)
    }

    /// 200 Mbps (high-speed operational rate).
    pub fn preset_200mbps(sys_clock_mhz: u32) -> Result<Self, BusError> {
        Self::new(200, sys_clock_mhz)
    }

    /// 400 Mbps (maximum operational rate).
    pub fn preset_400mbps(sys_clock_mhz: u32) -> Result<Self, BusError> {
        Self::new(400, sys_clock_mhz)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_speed_new_valid() {
        let speed = LinkSpeed::new(100, 200).unwrap();
        assert_eq!(speed.operational_mbps(), 100);
        assert_eq!(speed.startup_mbps(), 10);
        assert_eq!(speed.tx_divisor(), 4); // 2 * 200 / 100 = 4
    }

    #[test]
    fn test_speed_minimum() {
        let speed = LinkSpeed::new(MIN_SPEED_MBPS, 100).unwrap();
        assert_eq!(speed.operational_mbps(), 2);
    }

    #[test]
    fn test_speed_maximum() {
        let speed = LinkSpeed::new(MAX_SPEED_MBPS, 400).unwrap();
        assert_eq!(speed.operational_mbps(), 400);
    }

    #[test]
    fn test_speed_below_minimum() {
        assert_eq!(LinkSpeed::new(1, 100).err(), Some(BusError::OutOfRange));
    }

    #[test]
    fn test_speed_above_maximum() {
        assert_eq!(LinkSpeed::new(401, 400).err(), Some(BusError::OutOfRange));
    }

    #[test]
    fn test_speed_zero_clock() {
        assert_eq!(LinkSpeed::new(100, 0).err(), Some(BusError::OutOfRange));
    }

    #[test]
    fn test_speed_with_startup() {
        let speed = LinkSpeed::with_startup(200, 5, 200).unwrap();
        assert_eq!(speed.operational_mbps(), 200);
        assert_eq!(speed.startup_mbps(), 5);
    }

    #[test]
    fn test_speed_startup_exceeds_operational() {
        assert_eq!(
            LinkSpeed::with_startup(100, 200, 200).err(),
            Some(BusError::OutOfRange)
        );
    }

    #[test]
    fn test_speed_startup_below_minimum() {
        assert_eq!(
            LinkSpeed::with_startup(100, 1, 200).err(),
            Some(BusError::OutOfRange)
        );
    }

    #[test]
    fn test_actual_rate() {
        let speed = LinkSpeed::new(100, 200).unwrap();
        assert_eq!(speed.actual_rate_mbps(200), 100);
    }

    #[test]
    fn test_startup_divisor() {
        let speed = LinkSpeed::new(200, 200).unwrap();
        let startup_div = speed.startup_divisor(200);
        // Startup at 10 Mbps: 2 * 200 / 10 = 40
        assert_eq!(startup_div, 40);
    }

    #[test]
    fn test_char_time_ns() {
        let speed = LinkSpeed::new(100, 200).unwrap();
        // 10 * 1000 / 100 = 100 ns per data character
        assert_eq!(speed.char_time_ns(), 100);
    }

    #[test]
    fn test_char_time_ns_10mbps() {
        let speed = LinkSpeed::new(10, 200).unwrap();
        // 10 * 1000 / 10 = 1000 ns per data character
        assert_eq!(speed.char_time_ns(), 1000);
    }

    #[test]
    fn test_preset_10mbps() {
        let speed = LinkSpeed::preset_10mbps(200).unwrap();
        assert_eq!(speed.operational_mbps(), 10);
    }

    #[test]
    fn test_preset_100mbps() {
        let speed = LinkSpeed::preset_100mbps(200).unwrap();
        assert_eq!(speed.operational_mbps(), 100);
    }

    #[test]
    fn test_preset_200mbps() {
        let speed = LinkSpeed::preset_200mbps(200).unwrap();
        assert_eq!(speed.operational_mbps(), 200);
    }

    #[test]
    fn test_preset_400mbps() {
        let speed = LinkSpeed::preset_400mbps(400).unwrap();
        assert_eq!(speed.operational_mbps(), 400);
    }

    #[test]
    fn test_startup_capped_at_operational() {
        // If operational < default startup (10), startup should be capped
        let speed = LinkSpeed::new(5, 100).unwrap();
        assert_eq!(speed.startup_mbps(), 5);
    }
}
