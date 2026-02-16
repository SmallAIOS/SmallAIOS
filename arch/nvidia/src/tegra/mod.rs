// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tegra T210 GPU HAL — SoC-integrated GM20B (Maxwell, CC 5.3).
//!
//! This module provides bare-metal GPU initialization for the NVIDIA Tegra X1
//! SoC as found in the Jetson Nano. Unlike discrete PCIe GPUs, the GM20B is
//! memory-mapped at fixed addresses and shares system DRAM.
//!
//! # Initialization sequence
//!
//! 1. **Power** — Enable VDD_GPU regulator, remove PMC rail clamp
//! 2. **Clocks** — Enable GPU clock via CAR, configure GPCPLL
//! 3. **Interrupts** — Enable GICv2 SPI 189/190
//! 4. **Firmware** — Load ACR, FECS, GPCCS via Falcon DMA
//! 5. **Engines** — Initialize GR, FIFO, GMMU
//!
//! # Licensing
//!
//! All code in this module is Apache-2.0. Register definitions in `regs.rs`
//! document their provenance from the MIT-licensed nvgpu driver.
//! See `LICENSES/MIT-nvgpu.txt` for the MIT license text.

#![allow(dead_code)]

pub mod clock;
pub mod falcon;
pub mod fifo;
pub mod gmmu;
pub mod gr;
pub mod power;
pub mod regs;

/// Top-level Tegra GPU state machine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TegraGpuState {
    /// GPU is completely off.
    Off,
    /// Power domain enabled, clocks running.
    PlatformReady,
    /// Firmware loaded, engines initialized.
    Ready,
    /// GPU encountered an error.
    Error,
}

/// Platform-level orchestrator for GPU power, clocks, and GPCPLL.
///
/// `TegraGpuPlatform` manages the bring-up and teardown sequence that takes the
/// GPU from `Off` to `PlatformReady` (power on, clocks enabled, GPCPLL locked).
/// It tracks per-subsystem status so callers can inspect exactly which steps
/// have completed.
pub struct TegraGpuPlatform<M: power::MmioAccess> {
    /// GPU power controller (owns the MMIO accessor).
    power: power::GpuPower<M>,
    /// GPU clock / GPCPLL controller.
    clock: clock::GpuClock,
    /// Composite platform state.
    state: TegraGpuState,
    /// True after power_on succeeds.
    powered: bool,
    /// True after enable_clocks succeeds.
    clocks_enabled: bool,
    /// Current GPCPLL frequency in MHz (0 when unconfigured).
    gpcpll_freq: u32,
}

impl<M: power::MmioAccess> TegraGpuPlatform<M> {
    /// Create a new platform orchestrator in the Off state.
    pub fn new(mmio: M) -> Self {
        Self {
            power: power::GpuPower::new(mmio),
            clock: clock::GpuClock::new(),
            state: TegraGpuState::Off,
            powered: false,
            clocks_enabled: false,
            gpcpll_freq: 0,
        }
    }

    /// Current platform state.
    pub fn state(&self) -> TegraGpuState {
        self.state
    }

    /// Whether the GPU power domain is active.
    pub fn powered(&self) -> bool {
        self.powered
    }

    /// Whether CAR clocks are enabled.
    pub fn clocks_enabled(&self) -> bool {
        self.clocks_enabled
    }

    /// Current GPCPLL frequency in MHz (0 if not configured).
    pub fn gpcpll_freq(&self) -> u32 {
        self.gpcpll_freq
    }

    /// Full platform init: power on, enable clocks, configure GPCPLL.
    ///
    /// Uses GPCPLL step 7 (614 MHz) as a sensible default boot frequency.
    /// On failure the platform transitions to `Error` and partial state is
    /// preserved so the caller can inspect how far init progressed.
    pub fn init(&mut self) -> Result<(), crate::GpuError> {
        self.init_with_step(7)
    }

    /// Platform init with an explicit GPCPLL frequency step index.
    pub fn init_with_step(&mut self, gpcpll_step: usize) -> Result<(), crate::GpuError> {
        // Step 1: Power on
        if let Err(e) = self.power.power_on() {
            self.state = TegraGpuState::Error;
            return Err(e);
        }
        self.powered = true;

        // Step 2: Enable clocks via CAR
        if let Err(e) = self.clock.enable_clocks(self.power.mmio()) {
            self.state = TegraGpuState::Error;
            return Err(e);
        }
        self.clocks_enabled = true;

        // Step 3: Configure GPCPLL
        if let Err(e) = self.clock.configure_gpcpll(gpcpll_step, self.power.mmio()) {
            self.state = TegraGpuState::Error;
            return Err(e);
        }
        self.gpcpll_freq = self.clock.current_frequency_mhz();

        self.state = TegraGpuState::PlatformReady;
        Ok(())
    }

    /// Shutdown: disable GPCPLL/clocks, then power off (reverse of init).
    pub fn shutdown(&mut self) -> Result<(), crate::GpuError> {
        // Step 1: Disable clocks (also implicitly bypasses GPCPLL)
        if let Err(e) = self.clock.disable_clocks(self.power.mmio()) {
            self.state = TegraGpuState::Error;
            return Err(e);
        }
        self.clocks_enabled = false;
        self.gpcpll_freq = 0;

        // Step 2: Power off
        if let Err(e) = self.power.power_off() {
            self.state = TegraGpuState::Error;
            return Err(e);
        }
        self.powered = false;

        self.state = TegraGpuState::Off;
        Ok(())
    }
}

/// Tegra T210 GPU context — orchestrates the full initialization sequence.
pub struct TegraGpu {
    state: TegraGpuState,
    bar0_base: u64,
}

impl TegraGpu {
    /// Create a new Tegra GPU context.
    ///
    /// `bar0_base` is the GPU register base (0x5700_0000 on Tegra X1).
    pub fn new(bar0_base: u64) -> Self {
        Self {
            state: TegraGpuState::Off,
            bar0_base,
        }
    }

    /// Current GPU state.
    pub fn state(&self) -> TegraGpuState {
        self.state
    }

    /// GPU BAR0 base address.
    pub fn bar0_base(&self) -> u64 {
        self.bar0_base
    }

    /// Returns true if the GPU is fully initialized and ready for compute.
    pub fn is_ready(&self) -> bool {
        self.state == TegraGpuState::Ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
    use core::cell::RefCell;

    use regs::*;

    // -----------------------------------------------------------------
    // MockMmio — same pattern as power.rs / clock.rs tests
    // -----------------------------------------------------------------

    struct MockMmio {
        reads: RefCell<Vec<u32>>,
    }

    impl MockMmio {
        fn new(reads: Vec<u32>) -> Self {
            Self {
                reads: RefCell::new(reads),
            }
        }
    }

    impl power::MmioAccess for MockMmio {
        fn read32(&self, _addr: u64) -> u32 {
            let mut reads = self.reads.borrow_mut();
            if reads.is_empty() {
                0
            } else {
                reads.remove(0)
            }
        }

        fn write32(&self, _addr: u64, _val: u32) {}
    }

    /// Build mock reads for a successful init sequence.
    ///
    /// power_on: 1 read (partition already powered)
    /// enable_clocks: 0 reads
    /// configure_gpcpll: 3 reads (IDDQ clear, enable, lock poll)
    fn mock_reads_for_init() -> Vec<u32> {
        let partition_mask = 1u32 << PMC_GPU_PARTITION;
        let cfg_locked = GPCPLL_CFG_ENABLE | GPCPLL_CFG_LOCK;
        vec![
            partition_mask, // power_on: partition already on
            0,              // configure_gpcpll: GPCPLL_CFG read (IDDQ)
            0,              // configure_gpcpll: GPCPLL_CFG read (enable)
            cfg_locked,     // configure_gpcpll: GPCPLL_CFG read (lock)
        ]
    }

    /// Build mock reads for a successful init + shutdown round-trip.
    fn mock_reads_for_init_and_shutdown() -> Vec<u32> {
        let partition_mask = 1u32 << PMC_GPU_PARTITION;
        let cfg_locked = GPCPLL_CFG_ENABLE | GPCPLL_CFG_LOCK;
        vec![
            partition_mask, // power_on: partition already on
            0,              // configure_gpcpll: IDDQ
            0,              // configure_gpcpll: enable
            cfg_locked,     // configure_gpcpll: lock
            // shutdown:
            // disable_clocks: no reads
            // power_off: poll for gate
            partition_mask, // power_off: still on first poll
            0,              // power_off: gated
        ]
    }

    // -----------------------------------------------------------------
    // TegraGpu tests (existing)
    // -----------------------------------------------------------------

    #[test]
    fn new_tegra_gpu_starts_off() {
        let gpu = TegraGpu::new(0x5700_0000);
        assert_eq!(gpu.state(), TegraGpuState::Off);
        assert!(!gpu.is_ready());
    }

    #[test]
    fn bar0_base_is_stored() {
        let gpu = TegraGpu::new(0x5700_0000);
        assert_eq!(gpu.bar0_base(), 0x5700_0000);
    }

    // -----------------------------------------------------------------
    // TegraGpuPlatform tests
    // -----------------------------------------------------------------

    #[test]
    fn platform_new_starts_off() {
        let mmio = MockMmio::new(vec![]);
        let plat = TegraGpuPlatform::new(mmio);

        assert_eq!(plat.state(), TegraGpuState::Off);
        assert!(!plat.powered());
        assert!(!plat.clocks_enabled());
        assert_eq!(plat.gpcpll_freq(), 0);
    }

    #[test]
    fn platform_init_success() {
        let mmio = MockMmio::new(mock_reads_for_init());
        let mut plat = TegraGpuPlatform::new(mmio);

        assert!(plat.init().is_ok());
        assert_eq!(plat.state(), TegraGpuState::PlatformReady);
        assert!(plat.powered());
        assert!(plat.clocks_enabled());
        // Default step 7 = 614 MHz
        assert_eq!(plat.gpcpll_freq(), 614);
    }

    #[test]
    fn platform_init_with_step() {
        let mmio = MockMmio::new(mock_reads_for_init());
        let mut plat = TegraGpuPlatform::new(mmio);

        assert!(plat.init_with_step(0).is_ok());
        assert_eq!(plat.state(), TegraGpuState::PlatformReady);
        assert_eq!(plat.gpcpll_freq(), 76); // step 0 = 76 MHz
    }

    #[test]
    fn platform_shutdown_success() {
        let mmio = MockMmio::new(mock_reads_for_init_and_shutdown());
        let mut plat = TegraGpuPlatform::new(mmio);

        plat.init().unwrap();
        assert!(plat.shutdown().is_ok());

        assert_eq!(plat.state(), TegraGpuState::Off);
        assert!(!plat.powered());
        assert!(!plat.clocks_enabled());
        assert_eq!(plat.gpcpll_freq(), 0);
    }

    #[test]
    fn platform_init_power_failure_sets_error() {
        // All status reads return 0 — power_on will timeout
        let reads = vec![0u32; 10_001];
        let mmio = MockMmio::new(reads);
        let mut plat = TegraGpuPlatform::new(mmio);

        assert!(plat.init().is_err());
        assert_eq!(plat.state(), TegraGpuState::Error);
        // Power never succeeded
        assert!(!plat.powered());
        assert!(!plat.clocks_enabled());
        assert_eq!(plat.gpcpll_freq(), 0);
    }

    #[test]
    fn platform_init_gpcpll_failure_preserves_partial_state() {
        let partition_mask = 1u32 << PMC_GPU_PARTITION;
        // power_on succeeds, enable_clocks succeeds, but GPCPLL never locks
        let mut reads = vec![partition_mask]; // power_on
        // configure_gpcpll: IDDQ read, enable read, then 10_000 lock polls all 0
        reads.push(0); // IDDQ
        reads.push(0); // enable
        reads.extend(vec![0u32; 10_000]); // lock poll — never locks
        let mmio = MockMmio::new(reads);
        let mut plat = TegraGpuPlatform::new(mmio);

        assert!(plat.init().is_err());
        assert_eq!(plat.state(), TegraGpuState::Error);
        // Power and clocks succeeded before GPCPLL failed
        assert!(plat.powered());
        assert!(plat.clocks_enabled());
        assert_eq!(plat.gpcpll_freq(), 0);
    }

    #[test]
    fn platform_init_invalid_step() {
        let mmio = MockMmio::new(mock_reads_for_init());
        let mut plat = TegraGpuPlatform::new(mmio);

        // Step 99 is out of range — fails at configure_gpcpll (InvalidConfig)
        // but power and clocks still succeed first
        assert!(plat.init_with_step(99).is_err());
        assert_eq!(plat.state(), TegraGpuState::Error);
        assert!(plat.powered());
        assert!(plat.clocks_enabled());
    }

    #[test]
    fn platform_full_cycle() {
        let partition_mask = 1u32 << PMC_GPU_PARTITION;
        let cfg_locked = GPCPLL_CFG_ENABLE | GPCPLL_CFG_LOCK;
        let mmio = MockMmio::new(vec![
            // init:
            partition_mask, // power_on
            0,              // GPCPLL IDDQ
            0,              // GPCPLL enable
            cfg_locked,     // GPCPLL lock
            // shutdown:
            partition_mask, // power_off poll (still on)
            0,              // power_off poll (gated)
            // re-init:
            0,              // power_on: not powered
            partition_mask, // power_on: now powered
            0,              // GPCPLL IDDQ
            0,              // GPCPLL enable
            cfg_locked,     // GPCPLL lock
        ]);
        let mut plat = TegraGpuPlatform::new(mmio);

        // Init
        plat.init().unwrap();
        assert_eq!(plat.state(), TegraGpuState::PlatformReady);
        assert_eq!(plat.gpcpll_freq(), 614);

        // Shutdown
        plat.shutdown().unwrap();
        assert_eq!(plat.state(), TegraGpuState::Off);
        assert_eq!(plat.gpcpll_freq(), 0);

        // Re-init
        plat.init().unwrap();
        assert_eq!(plat.state(), TegraGpuState::PlatformReady);
        assert!(plat.powered());
    }

    // -----------------------------------------------------------------
    // Mock MMIO adapters for GR and GMMU engines
    // -----------------------------------------------------------------

    /// Mock MMIO implementing gr::MmioAccess for GR engine tests.
    struct GrMockMmio;

    impl gr::MmioAccess for GrMockMmio {
        fn read32(&self, _addr: u64) -> u32 {
            0
        }
        fn write32(&self, _addr: u64, _val: u32) {}
    }

    /// Mock MMIO implementing gmmu::MmioAccess for GMMU page table tests.
    struct GmmuMockMmio;

    impl gmmu::MmioAccess for GmmuMockMmio {
        fn read32(&self, _addr: u64) -> u32 {
            0
        }
        fn write32(&self, _addr: u64, _val: u32) {}
    }

    // -----------------------------------------------------------------
    // Integration test: full Phase A–C init sequence
    // -----------------------------------------------------------------

    /// End-to-end test exercising the complete GPU init sequence:
    /// Phase A: Platform power-on (power, clocks, GPCPLL)
    /// Phase B: GR engine init + FIFO channel allocation
    /// Phase C: GMMU page table setup
    #[test]
    fn integration_full_init_sequence() {
        let bar0: u64 = 0x5700_0000;

        // --- Phase A: Platform init (power on, clocks, GPCPLL) ---
        let mmio = MockMmio::new(mock_reads_for_init());
        let mut platform = TegraGpuPlatform::new(mmio);

        assert_eq!(platform.state(), TegraGpuState::Off);
        platform.init().unwrap();
        assert_eq!(platform.state(), TegraGpuState::PlatformReady);
        assert!(platform.powered());
        assert!(platform.clocks_enabled());
        assert_eq!(platform.gpcpll_freq(), 614);

        // --- Phase B: GR engine init ---
        let gr_mmio = GrMockMmio;
        let mut gr = gr::GrEngine::new();

        assert_eq!(gr.state(), gr::GrState::Uninitialized);
        assert!(!gr.is_ready());
        gr.init(bar0, &gr_mmio).unwrap();
        assert_eq!(gr.state(), gr::GrState::Ready);
        assert!(gr.is_ready());
        assert_eq!(gr.topology().total_cuda_cores(), 256);

        // --- Phase B (cont): FIFO channel allocation, bind to GR, activate ---
        let mut fifo = fifo::FifoEngine::new();

        let ch_id = fifo.allocate_channel().unwrap();
        assert_eq!(ch_id, 0);

        fifo.bind_to_gr(ch_id).unwrap();
        fifo.activate(ch_id).unwrap();
        assert_eq!(fifo.active_channel_count(), 1);

        // Verify command submission works on the active channel
        let entry = fifo::GpfifoEntry::new(0x1000_0000, 64);
        assert!(entry.is_valid());
        fifo.submit(ch_id, &entry).unwrap();

        // --- Phase C: GMMU page table setup ---
        let gmmu_mmio = GmmuMockMmio;
        let mut gmmu = gmmu::GmmuPageTable::new(gmmu::PageSize::Small);

        assert_eq!(gmmu.state(), gmmu::GmmuState::Uninitialized);

        // Identity-map 1 MiB of system DRAM for GPU access
        gmmu.create_identity_map(0x8000_0000, 0x10_0000).unwrap();
        assert_eq!(gmmu.state(), gmmu::GmmuState::PageTablesCreated);
        assert_eq!(gmmu.mapped_size(), 0x10_0000);

        gmmu.activate(bar0, &gmmu_mmio).unwrap();
        assert!(gmmu.is_active());
        assert_eq!(gmmu.state(), gmmu::GmmuState::Active);

        // --- Final assertions: all subsystems ready ---
        assert_eq!(platform.state(), TegraGpuState::PlatformReady);
        assert!(gr.is_ready());
        assert_eq!(fifo.active_channel_count(), 1);
        assert!(gmmu.is_active());
    }
}
