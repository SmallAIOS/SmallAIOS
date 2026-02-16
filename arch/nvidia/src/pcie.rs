// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! PCIe enumeration and BAR mapping for NVIDIA GPUs.
//!
//! Scans the PCI configuration space for NVIDIA display/3D controllers,
//! decodes Base Address Registers, and provides device filtering. On real
//! hardware the scan reads I/O ports 0x0CF8/0x0CFC; under `#[cfg(test)]`
//! the scanner pushes mock devices instead.

#![allow(dead_code)]

use alloc::vec::Vec;

use crate::GpuError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// NVIDIA PCI vendor ID.
const NVIDIA_VENDOR_ID: u16 = 0x10DE;

/// x86 PCI configuration address port.
const PCI_CONFIG_ADDR: u32 = 0x0CF8;

/// x86 PCI configuration data port.
const PCI_CONFIG_DATA: u32 = 0x0CFC;

/// Number of PCI buses to scan (first 8 covers most single-segment systems).
const MAX_BUSES: u8 = 8;

/// Maximum devices per bus (PCI spec).
const MAX_DEVICES: u8 = 32;

/// Maximum functions per device (PCI spec, multi-function).
const MAX_FUNCTIONS: u8 = 8;

/// Hard cap on the number of PCI devices we store.
const MAX_PCI_DEVICES: usize = 32;

// ---------------------------------------------------------------------------
// PciAddress
// ---------------------------------------------------------------------------

/// Bus/device/function triplet that identifies a PCI function.
#[derive(Clone, Debug, PartialEq)]
pub struct PciAddress {
    pub bus: u8,
    pub device: u8,
    pub function: u8,
}

impl PciAddress {
    /// Encode this address + a register offset into the 32-bit value written
    /// to `PCI_CONFIG_ADDR`.
    ///
    /// Layout (bit 31 = enable):
    /// ```text
    /// 31   | 30..24 | 23..16 | 15..11   | 10..8      | 7..2 | 1..0
    /// 1    | 0      | bus    | device   | function   | reg  | 00
    /// ```
    pub fn config_address(&self, reg: u8) -> u32 {
        (1u32 << 31)
            | ((self.bus as u32) << 16)
            | (((self.device & 0x1F) as u32) << 11)
            | (((self.function & 0x07) as u32) << 8)
            | ((reg & 0xFC) as u32)
    }
}

// ---------------------------------------------------------------------------
// BarType / BaseAddressRegister
// ---------------------------------------------------------------------------

/// PCI Base Address Register type.
#[derive(Clone, Debug, PartialEq)]
pub enum BarType {
    /// 32-bit memory-mapped region.
    Memory32,
    /// 64-bit memory-mapped region (consumes two BAR slots).
    Memory64,
    /// I/O port region.
    Io,
}

/// Decoded information about a single PCI BAR.
#[derive(Clone, Debug, PartialEq)]
pub struct BaseAddressRegister {
    pub bar_type: BarType,
    pub address: u64,
    pub size: u64,
    pub prefetchable: bool,
}

// ---------------------------------------------------------------------------
// PciDevice
// ---------------------------------------------------------------------------

/// A single PCI device (function) discovered during enumeration.
#[derive(Clone, Debug, PartialEq)]
pub struct PciDevice {
    pub address: PciAddress,
    pub vendor_id: u16,
    pub device_id: u16,
    pub class_code: u8,
    pub subclass: u8,
    pub revision: u8,
    pub bars: Vec<BaseAddressRegister>,
    pub irq_line: u8,
}

impl PciDevice {
    /// Returns `true` when the vendor is NVIDIA (0x10DE).
    pub fn is_nvidia(&self) -> bool {
        self.vendor_id == NVIDIA_VENDOR_ID
    }

    /// Returns `true` when the device belongs to PCI display-controller class
    /// (class code 0x03).
    pub fn is_display_controller(&self) -> bool {
        self.class_code == 0x03
    }

    /// Returns `true` for a PCI 3D controller (class 0x03, subclass 0x02).
    /// Most modern NVIDIA compute GPUs report this subclass.
    pub fn is_3d_controller(&self) -> bool {
        self.class_code == 0x03 && self.subclass == 0x02
    }
}

// ---------------------------------------------------------------------------
// PciScanner
// ---------------------------------------------------------------------------

/// Enumerates PCI buses and stores discovered devices.
#[derive(Clone, Debug, PartialEq)]
pub struct PciScanner {
    devices: Vec<PciDevice>,
}

impl Default for PciScanner {
    fn default() -> Self {
        Self::new()
    }
}

impl PciScanner {
    /// Create a new, empty scanner.
    pub fn new() -> Self {
        Self {
            devices: Vec::new(),
        }
    }

    /// Scan the PCI configuration space for NVIDIA display/3D controllers.
    ///
    /// On real hardware this reads I/O ports; in test builds it pushes a set
    /// of mock devices so the rest of the stack can be exercised without
    /// actual hardware.
    pub fn scan(&mut self) -> Result<(), GpuError> {
        #[cfg(test)]
        {
            self.scan_mock()
        }
        #[cfg(not(test))]
        {
            self.scan_hardware()
        }
    }

    /// Hardware scan (placeholder — real implementation reads I/O ports).
    #[cfg(not(test))]
    fn scan_hardware(&mut self) -> Result<(), GpuError> {
        // Iterate over buses / devices / functions.
        // For each function, read vendor_id; skip 0xFFFF.
        // If vendor == NVIDIA_VENDOR_ID and class == 0x03, record device.
        //
        // Actual I/O port access will be implemented once we have the
        // port-I/O abstraction from the x86-64 arch crate.
        for _bus in 0..MAX_BUSES {
            for _device in 0..MAX_DEVICES {
                for _function in 0..MAX_FUNCTIONS {
                    // Stub: no-op on real hardware until port I/O ready.
                }
            }
        }
        Ok(())
    }

    /// Mock scan used in unit tests — pushes representative NVIDIA devices.
    #[cfg(test)]
    fn scan_mock(&mut self) -> Result<(), GpuError> {
        // Tesla T4 (Turing)
        self.push_device(PciDevice {
            address: PciAddress {
                bus: 0,
                device: 3,
                function: 0,
            },
            vendor_id: NVIDIA_VENDOR_ID,
            device_id: 0x1B80,
            class_code: 0x03,
            subclass: 0x02,
            revision: 0xA1,
            bars: alloc::vec![
                BaseAddressRegister {
                    bar_type: BarType::Memory64,
                    address: 0xFB00_0000,
                    size: 16 * 1024 * 1024,
                    prefetchable: false,
                },
                BaseAddressRegister {
                    bar_type: BarType::Memory64,
                    address: 0x3800_0000_0000,
                    size: 256 * 1024 * 1024,
                    prefetchable: true,
                },
            ],
            irq_line: 11,
        });

        // A100 (Ampere)
        self.push_device(PciDevice {
            address: PciAddress {
                bus: 0,
                device: 4,
                function: 0,
            },
            vendor_id: NVIDIA_VENDOR_ID,
            device_id: 0x20B0,
            class_code: 0x03,
            subclass: 0x02,
            revision: 0xA1,
            bars: alloc::vec![
                BaseAddressRegister {
                    bar_type: BarType::Memory64,
                    address: 0xFC00_0000,
                    size: 32 * 1024 * 1024,
                    prefetchable: false,
                },
                BaseAddressRegister {
                    bar_type: BarType::Memory64,
                    address: 0x3900_0000_0000,
                    size: 512 * 1024 * 1024,
                    prefetchable: true,
                },
            ],
            irq_line: 12,
        });

        // Non-NVIDIA device (Intel integrated, for filter testing)
        self.push_device(PciDevice {
            address: PciAddress {
                bus: 0,
                device: 2,
                function: 0,
            },
            vendor_id: 0x8086,
            device_id: 0x5917,
            class_code: 0x03,
            subclass: 0x00,
            revision: 0x04,
            bars: alloc::vec![BaseAddressRegister {
                bar_type: BarType::Memory64,
                address: 0xDE00_0000,
                size: 16 * 1024 * 1024,
                prefetchable: false,
            },],
            irq_line: 10,
        });

        Ok(())
    }

    /// Add a device to the internal list (capped at `MAX_PCI_DEVICES`).
    fn push_device(&mut self, device: PciDevice) {
        if self.devices.len() < MAX_PCI_DEVICES {
            self.devices.push(device);
        }
    }

    /// Return references to only the NVIDIA-vendor devices.
    pub fn nvidia_devices(&self) -> Vec<&PciDevice> {
        self.devices.iter().filter(|d| d.is_nvidia()).collect()
    }

    /// Total number of discovered PCI devices (all vendors).
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// Return a reference to the full device list.
    pub fn devices(&self) -> &[PciDevice] {
        &self.devices
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- PciAddress --------------------------------------------------------

    #[test]
    fn config_address_enable_bit_set() {
        let addr = PciAddress {
            bus: 0,
            device: 0,
            function: 0,
        };
        let cfg = addr.config_address(0);
        assert!(cfg & (1 << 31) != 0, "enable bit must be set");
    }

    #[test]
    fn config_address_bus_encoding() {
        let addr = PciAddress {
            bus: 5,
            device: 0,
            function: 0,
        };
        let cfg = addr.config_address(0);
        let bus_field = (cfg >> 16) & 0xFF;
        assert_eq!(bus_field, 5);
    }

    #[test]
    fn config_address_device_encoding() {
        let addr = PciAddress {
            bus: 0,
            device: 18,
            function: 0,
        };
        let cfg = addr.config_address(0);
        let dev_field = (cfg >> 11) & 0x1F;
        assert_eq!(dev_field, 18);
    }

    #[test]
    fn config_address_function_encoding() {
        let addr = PciAddress {
            bus: 0,
            device: 0,
            function: 7,
        };
        let cfg = addr.config_address(0);
        let fn_field = (cfg >> 8) & 0x07;
        assert_eq!(fn_field, 7);
    }

    #[test]
    fn config_address_register_encoding() {
        let addr = PciAddress {
            bus: 0,
            device: 0,
            function: 0,
        };
        let cfg = addr.config_address(0x3C);
        let reg_field = cfg & 0xFC;
        assert_eq!(reg_field, 0x3C);
    }

    #[test]
    fn config_address_full_encoding() {
        // bus=1, device=3, function=2, register=0x10
        let addr = PciAddress {
            bus: 1,
            device: 3,
            function: 2,
        };
        let cfg = addr.config_address(0x10);
        let expected = (1u32 << 31) | (1u32 << 16) | (3u32 << 11) | (2u32 << 8) | 0x10u32;
        assert_eq!(cfg, expected);
    }

    // -- BarType -----------------------------------------------------------

    #[test]
    fn bar_type_memory32() {
        let bar = BaseAddressRegister {
            bar_type: BarType::Memory32,
            address: 0xF000_0000,
            size: 1024,
            prefetchable: false,
        };
        assert_eq!(bar.bar_type, BarType::Memory32);
        assert!(!bar.prefetchable);
    }

    #[test]
    fn bar_type_memory64_prefetchable() {
        let bar = BaseAddressRegister {
            bar_type: BarType::Memory64,
            address: 0x3800_0000_0000,
            size: 256 * 1024 * 1024,
            prefetchable: true,
        };
        assert_eq!(bar.bar_type, BarType::Memory64);
        assert!(bar.prefetchable);
    }

    #[test]
    fn bar_type_io() {
        let bar = BaseAddressRegister {
            bar_type: BarType::Io,
            address: 0x1000,
            size: 256,
            prefetchable: false,
        };
        assert_eq!(bar.bar_type, BarType::Io);
    }

    // -- PciDevice classification ------------------------------------------

    #[test]
    fn device_is_nvidia() {
        let dev = PciDevice {
            address: PciAddress {
                bus: 0,
                device: 0,
                function: 0,
            },
            vendor_id: NVIDIA_VENDOR_ID,
            device_id: 0x1B80,
            class_code: 0x03,
            subclass: 0x02,
            revision: 0xA1,
            bars: alloc::vec![],
            irq_line: 11,
        };
        assert!(dev.is_nvidia());
    }

    #[test]
    fn device_is_not_nvidia() {
        let dev = PciDevice {
            address: PciAddress {
                bus: 0,
                device: 0,
                function: 0,
            },
            vendor_id: 0x8086,
            device_id: 0x5917,
            class_code: 0x03,
            subclass: 0x00,
            revision: 0x04,
            bars: alloc::vec![],
            irq_line: 10,
        };
        assert!(!dev.is_nvidia());
    }

    #[test]
    fn device_is_display_controller() {
        let dev = PciDevice {
            address: PciAddress {
                bus: 0,
                device: 0,
                function: 0,
            },
            vendor_id: NVIDIA_VENDOR_ID,
            device_id: 0x1B80,
            class_code: 0x03,
            subclass: 0x00,
            revision: 0,
            bars: alloc::vec![],
            irq_line: 0,
        };
        assert!(dev.is_display_controller());
    }

    #[test]
    fn device_is_3d_controller() {
        let dev = PciDevice {
            address: PciAddress {
                bus: 0,
                device: 0,
                function: 0,
            },
            vendor_id: NVIDIA_VENDOR_ID,
            device_id: 0x20B0,
            class_code: 0x03,
            subclass: 0x02,
            revision: 0,
            bars: alloc::vec![],
            irq_line: 0,
        };
        assert!(dev.is_3d_controller());
    }

    #[test]
    fn device_not_3d_if_wrong_subclass() {
        let dev = PciDevice {
            address: PciAddress {
                bus: 0,
                device: 0,
                function: 0,
            },
            vendor_id: NVIDIA_VENDOR_ID,
            device_id: 0x20B0,
            class_code: 0x03,
            subclass: 0x00,
            revision: 0,
            bars: alloc::vec![],
            irq_line: 0,
        };
        assert!(!dev.is_3d_controller());
    }

    // -- PciScanner --------------------------------------------------------

    #[test]
    fn scanner_new_is_empty() {
        let scanner = PciScanner::new();
        assert_eq!(scanner.device_count(), 0);
    }

    #[test]
    fn scanner_mock_scan_finds_devices() {
        let mut scanner = PciScanner::new();
        scanner.scan().expect("mock scan should succeed");
        assert_eq!(scanner.device_count(), 3);
    }

    #[test]
    fn scanner_nvidia_filter() {
        let mut scanner = PciScanner::new();
        scanner.scan().unwrap();
        let nvidia = scanner.nvidia_devices();
        assert_eq!(nvidia.len(), 2, "mock scan has exactly 2 NVIDIA devices");
        for dev in &nvidia {
            assert!(dev.is_nvidia());
        }
    }

    #[test]
    fn scanner_non_nvidia_present() {
        let mut scanner = PciScanner::new();
        scanner.scan().unwrap();
        let non_nvidia: Vec<_> = scanner
            .devices()
            .iter()
            .filter(|d| !d.is_nvidia())
            .collect();
        assert_eq!(non_nvidia.len(), 1, "mock scan has 1 non-NVIDIA device");
        assert_eq!(non_nvidia[0].vendor_id, 0x8086);
    }

    #[test]
    fn scanner_device_ids_for_known_families() {
        let mut scanner = PciScanner::new();
        scanner.scan().unwrap();
        let nvidia = scanner.nvidia_devices();
        let ids: Vec<u16> = nvidia.iter().map(|d| d.device_id).collect();
        assert!(ids.contains(&0x1B80), "should contain Tesla T4 (Turing)");
        assert!(ids.contains(&0x20B0), "should contain A100 (Ampere)");
    }

    #[test]
    fn scanner_max_capacity() {
        let mut scanner = PciScanner::new();
        for i in 0..MAX_PCI_DEVICES + 10 {
            scanner.push_device(PciDevice {
                address: PciAddress {
                    bus: 0,
                    device: (i % 32) as u8,
                    function: 0,
                },
                vendor_id: NVIDIA_VENDOR_ID,
                device_id: 0x1B80,
                class_code: 0x03,
                subclass: 0x02,
                revision: 0,
                bars: alloc::vec![],
                irq_line: 0,
            });
        }
        assert_eq!(scanner.device_count(), MAX_PCI_DEVICES);
    }

    #[test]
    fn pci_address_ranges_valid() {
        // Verify that device and function fields are masked properly.
        let addr = PciAddress {
            bus: 255,
            device: 31,
            function: 7,
        };
        let cfg = addr.config_address(0xFC);
        assert_eq!((cfg >> 16) & 0xFF, 255);
        assert_eq!((cfg >> 11) & 0x1F, 31);
        assert_eq!((cfg >> 8) & 0x07, 7);
        assert_eq!(cfg & 0xFC, 0xFC);
    }

    #[test]
    fn mock_devices_have_bars() {
        let mut scanner = PciScanner::new();
        scanner.scan().unwrap();
        let nvidia = scanner.nvidia_devices();
        for dev in &nvidia {
            assert!(
                !dev.bars.is_empty(),
                "NVIDIA mock devices should have at least one BAR"
            );
        }
    }
}
