// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tegra X1 CAR clock/reset control and GPCPLL configuration.
//!
//! Manages the GPU clock domain through the Clock and Reset (CAR) controller
//! and configures the GPCPLL for the desired GPU frequency step.
//!
//! The GPCPLL programming sequence follows the GM20B bring-up order:
//! 1. Exit IDDQ (power-down) mode
//! 2. Set output divider and bypass mode
//! 3. Program M/N/P coefficients
//! 4. Enable PLL and poll for lock
//! 5. Switch clock source to VCO
//! 6. Restore output divider

#![allow(dead_code)]

use super::power::MmioAccess;
use super::regs::*;
use crate::GpuError;

// ---------------------------------------------------------------------------
// Platform base addresses
// ---------------------------------------------------------------------------

/// CAR (Clock And Reset) controller MMIO base address.
const CAR_BASE: u64 = 0x6000_6000;

/// GPU BAR0 MMIO base address (for GPCPLL registers).
const GPU_BAR0_BASE: u64 = 0x5700_0000;

// ---------------------------------------------------------------------------
// ClockState
// ---------------------------------------------------------------------------

/// GPU clock domain state machine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClockState {
    /// Clocks are disabled, reset is not asserted.
    Disabled,
    /// Reset is asserted to the GPU (held in reset).
    ResetAsserted,
    /// Clock is enabled and reset is de-asserted.
    ClockEnabled,
    /// GPCPLL is locked and GPU is clocked from VCO.
    GpcpllLocked,
}

// ---------------------------------------------------------------------------
// GpuClock
// ---------------------------------------------------------------------------

/// Maximum poll iterations for GPCPLL lock.
const GPCPLL_LOCK_POLL_LIMIT: u32 = 10_000;

/// GPU clock controller for CAR and GPCPLL management.
pub struct GpuClock {
    state: ClockState,
    current_step: usize,
}

impl Default for GpuClock {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuClock {
    /// Create a new GPU clock controller in the Disabled state.
    pub fn new() -> Self {
        Self {
            state: ClockState::Disabled,
            current_step: 0,
        }
    }

    /// Return the current clock state.
    pub fn state(&self) -> ClockState {
        self.state
    }

    /// Return the current GPCPLL step index.
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Return the current GPU clock frequency in MHz.
    pub fn current_frequency_mhz(&self) -> u32 {
        GPCPLL_STEPS[self.current_step].freq_mhz
    }

    /// Enable GPU clocks through the CAR controller.
    ///
    /// Sequence:
    /// 1. Assert reset via CAR_RST_DEV_X_SET
    /// 2. Enable clock via CAR_CLK_OUT_ENB_X_SET
    /// 3. Brief delay (spin loop)
    /// 4. De-assert reset via CAR_RST_DEV_X_CLR
    pub fn enable_clocks(&mut self, mmio: &impl MmioAccess) -> Result<(), GpuError> {
        if self.state == ClockState::ClockEnabled || self.state == ClockState::GpcpllLocked {
            return Ok(());
        }

        // Step 1: Assert reset
        mmio.write32(CAR_BASE + CAR_RST_DEV_X_SET_OFFSET as u64, CAR_GPU_BIT);
        self.state = ClockState::ResetAsserted;

        // Step 2: Enable clock
        mmio.write32(CAR_BASE + CAR_CLK_OUT_ENB_X_SET_OFFSET as u64, CAR_GPU_BIT);

        // Step 3: Small delay for clock to stabilize
        for _ in 0..100 {
            core::hint::spin_loop();
        }

        // Step 4: De-assert reset
        mmio.write32(CAR_BASE + CAR_RST_DEV_X_CLR_OFFSET as u64, CAR_GPU_BIT);
        self.state = ClockState::ClockEnabled;

        Ok(())
    }

    /// Disable GPU clocks through the CAR controller.
    ///
    /// Asserts reset and disables the clock output.
    pub fn disable_clocks(&mut self, mmio: &impl MmioAccess) -> Result<(), GpuError> {
        if self.state == ClockState::Disabled {
            return Ok(());
        }

        // Assert reset
        mmio.write32(CAR_BASE + CAR_RST_DEV_X_SET_OFFSET as u64, CAR_GPU_BIT);

        // Disable clock
        mmio.write32(CAR_BASE + CAR_CLK_OUT_ENB_X_CLR_OFFSET as u64, CAR_GPU_BIT);

        self.state = ClockState::Disabled;
        Ok(())
    }

    /// Configure the GPCPLL to the given frequency step.
    ///
    /// The full PLL programming sequence:
    /// 1. Validate step index
    /// 2. Exit IDDQ (clear power-down bit)
    /// 3. Set output divider to safe value
    /// 4. Enable bypass mode
    /// 5. Program M/N/P coefficients
    /// 6. Enable PLL and poll for lock
    /// 7. Switch clock source to VCO
    /// 8. Restore output divider
    pub fn configure_gpcpll(
        &mut self,
        step: usize,
        mmio: &impl MmioAccess,
    ) -> Result<(), GpuError> {
        if step >= GPCPLL_STEPS.len() {
            return Err(GpuError::InvalidConfig);
        }

        if self.state != ClockState::ClockEnabled && self.state != ClockState::GpcpllLocked {
            return Err(GpuError::InvalidState);
        }

        let coeff = &GPCPLL_STEPS[step];

        // Step 1: Exit IDDQ — clear the IDDQ bit, keep PLL disabled
        let cfg = mmio.read32(GPU_BAR0_BASE + GPCPLL_CFG as u64);
        let cfg = cfg & !GPCPLL_CFG_IDDQ;
        mmio.write32(GPU_BAR0_BASE + GPCPLL_CFG as u64, cfg);

        // Small delay for IDDQ exit
        for _ in 0..100 {
            core::hint::spin_loop();
        }

        // Step 2: Set output divider to safe value (divide by max)
        mmio.write32(GPU_BAR0_BASE + GPC2CLK_OUT as u64, 0);

        // Step 3: Enable bypass mode
        mmio.write32(GPU_BAR0_BASE + BYPASSCTRL as u64, 1);

        // Step 4: Program M/N/P coefficients
        let coeff_val = ((coeff.m as u32) << COEFF_M_SHIFT)
            | ((coeff.n as u32) << COEFF_N_SHIFT)
            | ((coeff.pl as u32) << COEFF_P_SHIFT);
        mmio.write32(GPU_BAR0_BASE + GPCPLL_COEFF as u64, coeff_val);

        // Step 5: Enable PLL
        let cfg = mmio.read32(GPU_BAR0_BASE + GPCPLL_CFG as u64);
        let cfg = cfg | GPCPLL_CFG_ENABLE;
        mmio.write32(GPU_BAR0_BASE + GPCPLL_CFG as u64, cfg);

        // Step 6: Poll for PLL lock
        let mut locked = false;
        for _ in 0..GPCPLL_LOCK_POLL_LIMIT {
            let cfg = mmio.read32(GPU_BAR0_BASE + GPCPLL_CFG as u64);
            if cfg & GPCPLL_CFG_LOCK != 0 {
                locked = true;
                break;
            }
        }
        if !locked {
            return Err(GpuError::InitFailed);
        }

        // Step 7: Switch clock source to VCO
        mmio.write32(GPU_BAR0_BASE + SEL_VCO as u64, 1);

        // Step 8: Disable bypass
        mmio.write32(GPU_BAR0_BASE + BYPASSCTRL as u64, 0);

        // Step 9: Restore output divider
        mmio.write32(GPU_BAR0_BASE + GPC2CLK_OUT as u64, 1);

        self.current_step = step;
        self.state = ClockState::GpcpllLocked;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::cell::RefCell;

    /// Record of a single MMIO write operation.
    #[derive(Clone, Debug, PartialEq)]
    struct WriteRecord {
        addr: u64,
        val: u32,
    }

    /// Mock MMIO that records writes and returns configurable read values.
    struct MockMmio {
        reads: RefCell<Vec<u32>>,
        writes: RefCell<Vec<WriteRecord>>,
    }

    impl MockMmio {
        fn new(reads: Vec<u32>) -> Self {
            Self {
                reads: RefCell::new(reads),
                writes: RefCell::new(Vec::new()),
            }
        }

        fn writes(&self) -> Vec<WriteRecord> {
            self.writes.borrow().clone()
        }
    }

    impl MmioAccess for MockMmio {
        fn read32(&self, _addr: u64) -> u32 {
            let mut reads = self.reads.borrow_mut();
            if reads.is_empty() {
                0
            } else {
                reads.remove(0)
            }
        }

        fn write32(&self, addr: u64, val: u32) {
            self.writes.borrow_mut().push(WriteRecord { addr, val });
        }
    }

    #[test]
    fn initial_state_is_disabled() {
        let clock = GpuClock::new();
        assert_eq!(clock.state(), ClockState::Disabled);
        assert_eq!(clock.current_step(), 0);
    }

    #[test]
    fn initial_frequency() {
        let clock = GpuClock::new();
        assert_eq!(clock.current_frequency_mhz(), GPCPLL_STEPS[0].freq_mhz);
    }

    #[test]
    fn enable_clocks_sequence() {
        let mmio = MockMmio::new(vec![]);
        let mut clock = GpuClock::new();

        assert!(clock.enable_clocks(&mmio).is_ok());
        assert_eq!(clock.state(), ClockState::ClockEnabled);

        let writes = mmio.writes();
        assert!(writes.len() >= 3);

        // First write: assert reset
        assert_eq!(writes[0].addr, CAR_BASE + CAR_RST_DEV_X_SET_OFFSET as u64);
        assert_eq!(writes[0].val, CAR_GPU_BIT);

        // Second write: enable clock
        assert_eq!(
            writes[1].addr,
            CAR_BASE + CAR_CLK_OUT_ENB_X_SET_OFFSET as u64
        );
        assert_eq!(writes[1].val, CAR_GPU_BIT);

        // Third write: de-assert reset
        assert_eq!(writes[2].addr, CAR_BASE + CAR_RST_DEV_X_CLR_OFFSET as u64);
        assert_eq!(writes[2].val, CAR_GPU_BIT);
    }

    #[test]
    fn enable_clocks_idempotent() {
        let mmio = MockMmio::new(vec![]);
        let mut clock = GpuClock::new();

        clock.enable_clocks(&mmio).unwrap();
        let write_count_first = mmio.writes().len();

        clock.enable_clocks(&mmio).unwrap();
        let write_count_second = mmio.writes().len();

        // No additional writes on second call
        assert_eq!(write_count_first, write_count_second);
    }

    #[test]
    fn disable_clocks() {
        let mmio = MockMmio::new(vec![]);
        let mut clock = GpuClock::new();

        clock.enable_clocks(&mmio).unwrap();
        assert!(clock.disable_clocks(&mmio).is_ok());
        assert_eq!(clock.state(), ClockState::Disabled);

        let writes = mmio.writes();
        let last_two = &writes[writes.len() - 2..];

        // Assert reset
        assert_eq!(last_two[0].addr, CAR_BASE + CAR_RST_DEV_X_SET_OFFSET as u64);
        assert_eq!(last_two[0].val, CAR_GPU_BIT);

        // Disable clock
        assert_eq!(
            last_two[1].addr,
            CAR_BASE + CAR_CLK_OUT_ENB_X_CLR_OFFSET as u64
        );
        assert_eq!(last_two[1].val, CAR_GPU_BIT);
    }

    #[test]
    fn disable_clocks_already_disabled() {
        let mmio = MockMmio::new(vec![]);
        let mut clock = GpuClock::new();

        assert!(clock.disable_clocks(&mmio).is_ok());
        assert_eq!(clock.state(), ClockState::Disabled);
        assert!(mmio.writes().is_empty());
    }

    #[test]
    fn configure_gpcpll_invalid_step() {
        let mmio = MockMmio::new(vec![]);
        let mut clock = GpuClock::new();
        clock.enable_clocks(&mmio).unwrap();

        assert_eq!(
            clock.configure_gpcpll(12, &mmio),
            Err(GpuError::InvalidConfig)
        );
        assert_eq!(
            clock.configure_gpcpll(100, &mmio),
            Err(GpuError::InvalidConfig)
        );
    }

    #[test]
    fn configure_gpcpll_wrong_state() {
        let mmio = MockMmio::new(vec![]);
        let mut clock = GpuClock::new();

        // Cannot configure GPCPLL when clocks are disabled
        assert_eq!(
            clock.configure_gpcpll(0, &mmio),
            Err(GpuError::InvalidState)
        );
    }

    #[test]
    fn configure_gpcpll_success() {
        // Reads needed during configure_gpcpll:
        // 1. read GPCPLL_CFG for IDDQ clear
        // 2. read GPCPLL_CFG for enable
        // 3. read GPCPLL_CFG for lock poll (return locked)
        let cfg_with_lock = GPCPLL_CFG_ENABLE | GPCPLL_CFG_LOCK;
        let mmio = MockMmio::new(vec![
            0,             // GPCPLL_CFG read (IDDQ clear)
            0,             // GPCPLL_CFG read (enable)
            cfg_with_lock, // GPCPLL_CFG read (lock poll — locked)
        ]);
        let mut clock = GpuClock::new();
        clock.enable_clocks(&mmio).unwrap();

        assert!(clock.configure_gpcpll(5, &mmio).is_ok());
        assert_eq!(clock.state(), ClockState::GpcpllLocked);
        assert_eq!(clock.current_step(), 5);
        assert_eq!(clock.current_frequency_mhz(), 460);
    }

    #[test]
    fn configure_gpcpll_coefficients() {
        let cfg_with_lock = GPCPLL_CFG_ENABLE | GPCPLL_CFG_LOCK;
        let mmio = MockMmio::new(vec![0, 0, cfg_with_lock]);
        let mut clock = GpuClock::new();
        clock.enable_clocks(&mmio).unwrap();

        // Step 7: freq=691 MHz, m=1, n=18, pl=1
        clock.configure_gpcpll(8, &mmio).unwrap();

        let writes = mmio.writes();
        let coeff_addr = GPU_BAR0_BASE + GPCPLL_COEFF as u64;
        let coeff_write = writes.iter().find(|w| w.addr == coeff_addr).unwrap();

        let expected = (1u32 << COEFF_M_SHIFT) | (18u32 << COEFF_N_SHIFT) | (1u32 << COEFF_P_SHIFT);
        assert_eq!(coeff_write.val, expected);
    }

    #[test]
    fn configure_gpcpll_lock_timeout() {
        // All GPCPLL_CFG reads return 0 (never locked)
        let mut reads = vec![0u32; (GPCPLL_LOCK_POLL_LIMIT + 3) as usize];
        // First read: IDDQ clear, second: enable — both return 0
        reads[0] = 0;
        reads[1] = 0;
        // Remaining reads: lock poll — all 0 (no lock)
        let mmio = MockMmio::new(reads);
        let mut clock = GpuClock::new();
        clock.enable_clocks(&mmio).unwrap();

        assert_eq!(clock.configure_gpcpll(0, &mmio), Err(GpuError::InitFailed));
    }

    #[test]
    fn configure_gpcpll_bypass_sequence() {
        let cfg_with_lock = GPCPLL_CFG_ENABLE | GPCPLL_CFG_LOCK;
        let mmio = MockMmio::new(vec![0, 0, cfg_with_lock]);
        let mut clock = GpuClock::new();
        clock.enable_clocks(&mmio).unwrap();

        clock.configure_gpcpll(0, &mmio).unwrap();

        let writes = mmio.writes();
        let bypass_addr = GPU_BAR0_BASE + BYPASSCTRL as u64;
        let bypass_writes: Vec<_> = writes.iter().filter(|w| w.addr == bypass_addr).collect();

        // Bypass enabled (1) then disabled (0)
        assert_eq!(bypass_writes.len(), 2);
        assert_eq!(bypass_writes[0].val, 1);
        assert_eq!(bypass_writes[1].val, 0);
    }

    #[test]
    fn configure_gpcpll_vco_select() {
        let cfg_with_lock = GPCPLL_CFG_ENABLE | GPCPLL_CFG_LOCK;
        let mmio = MockMmio::new(vec![0, 0, cfg_with_lock]);
        let mut clock = GpuClock::new();
        clock.enable_clocks(&mmio).unwrap();

        clock.configure_gpcpll(0, &mmio).unwrap();

        let writes = mmio.writes();
        let vco_addr = GPU_BAR0_BASE + SEL_VCO as u64;
        let vco_write = writes.iter().find(|w| w.addr == vco_addr).unwrap();
        assert_eq!(vco_write.val, 1);
    }

    #[test]
    fn reconfigure_gpcpll_from_locked() {
        let cfg_with_lock = GPCPLL_CFG_ENABLE | GPCPLL_CFG_LOCK;
        // Two configure calls: 3 reads each
        let mmio = MockMmio::new(vec![0, 0, cfg_with_lock, 0, 0, cfg_with_lock]);
        let mut clock = GpuClock::new();
        clock.enable_clocks(&mmio).unwrap();

        clock.configure_gpcpll(3, &mmio).unwrap();
        assert_eq!(clock.current_frequency_mhz(), 307);

        clock.configure_gpcpll(10, &mmio).unwrap();
        assert_eq!(clock.current_frequency_mhz(), 844);
        assert_eq!(clock.state(), ClockState::GpcpllLocked);
    }

    #[test]
    fn all_steps_produce_valid_coefficients() {
        for (i, step) in GPCPLL_STEPS.iter().enumerate() {
            let coeff_val = ((step.m as u32) << COEFF_M_SHIFT)
                | ((step.n as u32) << COEFF_N_SHIFT)
                | ((step.pl as u32) << COEFF_P_SHIFT);

            // Verify round-trip through masks
            let m = (coeff_val & COEFF_M_MASK) >> COEFF_M_SHIFT;
            let n = (coeff_val & COEFF_N_MASK) >> COEFF_N_SHIFT;
            let p = (coeff_val & COEFF_P_MASK) >> COEFF_P_SHIFT;

            assert_eq!(m, step.m as u32, "Step {}: M mismatch", i);
            assert_eq!(n, step.n as u32, "Step {}: N mismatch", i);
            assert_eq!(p, step.pl as u32, "Step {}: P mismatch", i);
        }
    }

    #[test]
    fn output_divider_sequence() {
        let cfg_with_lock = GPCPLL_CFG_ENABLE | GPCPLL_CFG_LOCK;
        let mmio = MockMmio::new(vec![0, 0, cfg_with_lock]);
        let mut clock = GpuClock::new();
        clock.enable_clocks(&mmio).unwrap();

        clock.configure_gpcpll(0, &mmio).unwrap();

        let writes = mmio.writes();
        let out_addr = GPU_BAR0_BASE + GPC2CLK_OUT as u64;
        let out_writes: Vec<_> = writes.iter().filter(|w| w.addr == out_addr).collect();

        // Output divider set to 0 (safe), then restored to 1
        assert_eq!(out_writes.len(), 2);
        assert_eq!(out_writes[0].val, 0);
        assert_eq!(out_writes[1].val, 1);
    }
}
