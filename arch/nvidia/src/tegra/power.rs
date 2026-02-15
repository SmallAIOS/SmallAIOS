// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tegra X1 PMC GPU power control.
//!
//! Manages the GPU power partition via the PMC (Power Management Controller):
//! - Power gate toggle and status polling
//! - Rail clamp assertion / de-assertion
//! - State machine tracking for safe power transitions

#![allow(dead_code)]

use super::regs::*;
use crate::GpuError;

// ---------------------------------------------------------------------------
// Platform base addresses
// ---------------------------------------------------------------------------

/// PMC (Power Management Controller) MMIO base address.
const PMC_BASE: u64 = 0x7000_E000;

// ---------------------------------------------------------------------------
// MMIO access trait
// ---------------------------------------------------------------------------

/// Trait for memory-mapped I/O access, allowing mock implementations in tests.
pub trait MmioAccess {
    /// Read a 32-bit value from the given MMIO address.
    fn read32(&self, addr: u64) -> u32;

    /// Write a 32-bit value to the given MMIO address.
    fn write32(&self, addr: u64, val: u32);
}

/// Hardware MMIO implementation using volatile reads/writes.
pub struct HwMmio;

impl MmioAccess for HwMmio {
    fn read32(&self, addr: u64) -> u32 {
        #[cfg(not(test))]
        unsafe {
            core::ptr::read_volatile(addr as *const u32)
        }
        #[cfg(test)]
        {
            let _ = addr;
            0
        }
    }

    fn write32(&self, addr: u64, val: u32) {
        #[cfg(not(test))]
        unsafe {
            core::ptr::write_volatile(addr as *mut u32, val);
        }
        #[cfg(test)]
        {
            let _ = (addr, val);
        }
    }
}

// ---------------------------------------------------------------------------
// GpuPowerState
// ---------------------------------------------------------------------------

/// GPU power domain state.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GpuPowerState {
    /// GPU partition is power-gated (off).
    Off,
    /// GPU is powered but rail clamp is still asserted.
    RailClamped,
    /// GPU is fully powered and rail clamp is released.
    Powered,
}

// ---------------------------------------------------------------------------
// GpuPower
// ---------------------------------------------------------------------------

/// Maximum number of poll iterations when waiting for power gate status.
const PWRGATE_POLL_LIMIT: u32 = 10_000;

/// Toggle command: START bit (bit 8) | partition ID in bits [4:0].
const PWRGATE_TOGGLE_START: u32 = 1 << 8;

/// GPU power controller backed by MMIO access.
pub struct GpuPower<M: MmioAccess> {
    state: GpuPowerState,
    mmio: M,
}

impl<M: MmioAccess> GpuPower<M> {
    /// Create a new GPU power controller in the Off state.
    pub fn new(mmio: M) -> Self {
        Self {
            state: GpuPowerState::Off,
            mmio,
        }
    }

    /// Return the current power state.
    pub fn state(&self) -> GpuPowerState {
        self.state
    }

    /// Power on the GPU partition.
    ///
    /// Sequence:
    /// 1. Check PMC_PWRGATE_STATUS bit for partition 14
    /// 2. If not powered, toggle PMC_PWRGATE_TOGGLE and poll for power-up
    /// 3. Clear rail clamp via PMC_GPU_RG_CNTRL
    pub fn power_on(&mut self) -> Result<(), GpuError> {
        if self.state == GpuPowerState::Powered {
            return Ok(());
        }

        let partition_mask = 1u32 << PMC_GPU_PARTITION;

        // Step 1: Check if GPU partition is already powered
        let status = self
            .mmio
            .read32(PMC_BASE + PMC_PWRGATE_STATUS_OFFSET as u64);
        if status & partition_mask == 0 {
            // GPU partition is off — toggle power gate to power it on
            let toggle_val = PWRGATE_TOGGLE_START | PMC_GPU_PARTITION;
            self.mmio
                .write32(PMC_BASE + PMC_PWRGATE_TOGGLE_OFFSET as u64, toggle_val);

            // Poll for power-on (bit 14 set in status register)
            let mut powered = false;
            for _ in 0..PWRGATE_POLL_LIMIT {
                let st = self
                    .mmio
                    .read32(PMC_BASE + PMC_PWRGATE_STATUS_OFFSET as u64);
                if st & partition_mask != 0 {
                    powered = true;
                    break;
                }
            }
            if !powered {
                return Err(GpuError::InitFailed);
            }
        }

        self.state = GpuPowerState::RailClamped;

        // Step 2: Release rail clamp — write 0 to PMC_GPU_RG_CNTRL
        self.mmio
            .write32(PMC_BASE + PMC_GPU_RG_CNTRL_OFFSET as u64, 0);
        self.state = GpuPowerState::Powered;

        Ok(())
    }

    /// Power off the GPU partition.
    ///
    /// Sequence:
    /// 1. Assert rail clamp via PMC_GPU_RG_CNTRL (bit 0 = 1)
    /// 2. Toggle power gate to power down partition 14
    /// 3. Poll for power-off confirmation
    pub fn power_off(&mut self) -> Result<(), GpuError> {
        if self.state == GpuPowerState::Off {
            return Ok(());
        }

        let partition_mask = 1u32 << PMC_GPU_PARTITION;

        // Step 1: Assert rail clamp
        self.mmio
            .write32(PMC_BASE + PMC_GPU_RG_CNTRL_OFFSET as u64, 1);
        self.state = GpuPowerState::RailClamped;

        // Step 2: Toggle power gate off
        let toggle_val = PWRGATE_TOGGLE_START | PMC_GPU_PARTITION;
        self.mmio
            .write32(PMC_BASE + PMC_PWRGATE_TOGGLE_OFFSET as u64, toggle_val);

        // Poll for power-off (bit 14 cleared in status register)
        let mut gated = false;
        for _ in 0..PWRGATE_POLL_LIMIT {
            let st = self
                .mmio
                .read32(PMC_BASE + PMC_PWRGATE_STATUS_OFFSET as u64);
            if st & partition_mask == 0 {
                gated = true;
                break;
            }
        }
        if !gated {
            return Err(GpuError::InitFailed);
        }

        self.state = GpuPowerState::Off;
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
        /// Queued read values (consumed in order).
        reads: RefCell<Vec<u32>>,
        /// Recorded writes.
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
    fn initial_state_is_off() {
        let mmio = MockMmio::new(vec![]);
        let power = GpuPower::new(mmio);
        assert_eq!(power.state(), GpuPowerState::Off);
    }

    #[test]
    fn power_on_already_powered_partition() {
        // PWRGATE_STATUS already has bit 14 set
        let partition_mask = 1u32 << PMC_GPU_PARTITION;
        let mmio = MockMmio::new(vec![partition_mask]);
        let mut power = GpuPower::new(mmio);

        assert!(power.power_on().is_ok());
        assert_eq!(power.state(), GpuPowerState::Powered);
    }

    #[test]
    fn power_on_needs_toggle() {
        let partition_mask = 1u32 << PMC_GPU_PARTITION;
        // First read: status=0 (not powered), second read: status=partition_mask (powered)
        let mmio = MockMmio::new(vec![0, partition_mask]);
        let mut power = GpuPower::new(mmio);

        assert!(power.power_on().is_ok());
        assert_eq!(power.state(), GpuPowerState::Powered);

        // Verify toggle was written
        let writes = power.mmio.writes();
        let toggle_addr = PMC_BASE + PMC_PWRGATE_TOGGLE_OFFSET as u64;
        let toggle_write = writes.iter().find(|w| w.addr == toggle_addr);
        assert!(toggle_write.is_some());
        assert_eq!(
            toggle_write.unwrap().val,
            PWRGATE_TOGGLE_START | PMC_GPU_PARTITION,
        );

        // Verify rail clamp was cleared
        let rg_addr = PMC_BASE + PMC_GPU_RG_CNTRL_OFFSET as u64;
        let rg_write = writes.iter().find(|w| w.addr == rg_addr);
        assert!(rg_write.is_some());
        assert_eq!(rg_write.unwrap().val, 0);
    }

    #[test]
    fn power_on_timeout() {
        // All status reads return 0 (never powered)
        let reads = vec![0u32; (PWRGATE_POLL_LIMIT + 1) as usize];
        let mmio = MockMmio::new(reads);
        let mut power = GpuPower::new(mmio);

        assert_eq!(power.power_on(), Err(GpuError::InitFailed));
    }

    #[test]
    fn power_on_idempotent() {
        let partition_mask = 1u32 << PMC_GPU_PARTITION;
        let mmio = MockMmio::new(vec![partition_mask]);
        let mut power = GpuPower::new(mmio);

        assert!(power.power_on().is_ok());
        assert_eq!(power.state(), GpuPowerState::Powered);

        // Second call should be a no-op
        assert!(power.power_on().is_ok());
        assert_eq!(power.state(), GpuPowerState::Powered);
    }

    #[test]
    fn power_off_from_powered() {
        let partition_mask = 1u32 << PMC_GPU_PARTITION;
        // power_on: status already set
        // power_off: status returns partition_mask first (still on), then 0 (gated)
        let mmio = MockMmio::new(vec![partition_mask, partition_mask, 0]);
        let mut power = GpuPower::new(mmio);

        power.power_on().unwrap();
        assert_eq!(power.state(), GpuPowerState::Powered);

        assert!(power.power_off().is_ok());
        assert_eq!(power.state(), GpuPowerState::Off);

        // Verify rail clamp was asserted (val=1)
        let writes = power.mmio.writes();
        let rg_addr = PMC_BASE + PMC_GPU_RG_CNTRL_OFFSET as u64;
        let rg_writes: Vec<_> = writes.iter().filter(|w| w.addr == rg_addr).collect();
        // First write: clear (0) from power_on, second write: assert (1) from power_off
        assert_eq!(rg_writes.len(), 2);
        assert_eq!(rg_writes[0].val, 0);
        assert_eq!(rg_writes[1].val, 1);
    }

    #[test]
    fn power_off_already_off() {
        let mmio = MockMmio::new(vec![]);
        let mut power = GpuPower::new(mmio);

        assert!(power.power_off().is_ok());
        assert_eq!(power.state(), GpuPowerState::Off);
    }

    #[test]
    fn power_off_timeout() {
        let partition_mask = 1u32 << PMC_GPU_PARTITION;
        // power_on: status already set
        // power_off: all reads return partition_mask (never goes off)
        let mut reads = vec![partition_mask]; // for power_on status check
        reads.extend(vec![partition_mask; (PWRGATE_POLL_LIMIT + 1) as usize]);
        let mmio = MockMmio::new(reads);
        let mut power = GpuPower::new(mmio);

        power.power_on().unwrap();
        assert_eq!(power.power_off(), Err(GpuError::InitFailed));
    }

    #[test]
    fn power_cycle() {
        let partition_mask = 1u32 << PMC_GPU_PARTITION;
        // power_on: status=set, power_off: status=set then 0, power_on again: status=0 then set
        let mmio = MockMmio::new(vec![
            partition_mask, // power_on: already powered
            partition_mask, // power_off: still on
            0,              // power_off: now gated
            0,              // power_on: not powered
            partition_mask, // power_on: now powered
        ]);
        let mut power = GpuPower::new(mmio);

        power.power_on().unwrap();
        assert_eq!(power.state(), GpuPowerState::Powered);

        power.power_off().unwrap();
        assert_eq!(power.state(), GpuPowerState::Off);

        power.power_on().unwrap();
        assert_eq!(power.state(), GpuPowerState::Powered);
    }
}
