// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! PCIe enumeration and BAR mapping for Intel GPUs.
//!
//! Scans the PCI configuration space for Intel display/3D controllers
//! (vendor 0x8086), decodes Base Address Registers, and provides device
//! filtering. On real hardware the scan reads I/O ports 0x0CF8/0x0CFC;
//! under `#[cfg(test)]` the scanner pushes mock devices instead.

use alloc::vec::Vec;

use crate::GpuError;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Intel PCI vendor ID.
const INTEL_VENDOR_ID: u16 = 0x8086;

/// x86 PCI configuration address port.
const PCI_CONFIG_ADDR: u32 = 0x0CF8;

/// x86 PCI configuration data port.
const PCI_CONFIG_DATA: u32 = 0x0CFC;

/// Number of PCI buses to scan.
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
    /// Returns `true` when the vendor is Intel (0x8086).
    pub fn is_intel(&self) -> bool {
        self.vendor_id == INTEL_VENDOR_ID
    }

    /// Returns `true` when the device belongs to PCI display-controller class
    /// (class code 0x03).
    pub fn is_display_controller(&self) -> bool {
        self.class_code == 0x03
    }

    /// Returns `true` for a PCI VGA-compatible controller (class 0x03, subclass 0x00).
    /// Most Intel integrated GPUs report this subclass.
    pub fn is_vga_controller(&self) -> bool {
        self.class_code == 0x03 && self.subclass == 0x00
    }

    /// Returns `true` for a PCI 3D controller (class 0x03, subclass 0x02).
    /// Intel discrete GPUs (Arc) may report this subclass.
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

    /// Scan the PCI configuration space for Intel display/3D controllers.
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

    /// Hardware scan (placeholder).
    #[cfg(not(test))]
    fn scan_hardware(&mut self) -> Result<(), GpuError> {
        for _bus in 0..MAX_BUSES {
            for _device in 0..MAX_DEVICES {
                for _function in 0..MAX_FUNCTIONS {
                    // Stub: no-op until port I/O ready.
                }
            }
        }
        Ok(())
    }

    /// Mock scan used in unit tests.
    #[cfg(test)]
    fn scan_mock(&mut self) -> Result<(), GpuError> {
        // Intel Xe-LP integrated GPU (Tiger Lake)
        self.push_device(PciDevice {
            address: PciAddress {
                bus: 0,
                device: 2,
                function: 0,
            },
            vendor_id: INTEL_VENDOR_ID,
            device_id: 0x9A49,
            class_code: 0x03,
            subclass: 0x00,
            revision: 0x01,
            bars: alloc::vec![
                BaseAddressRegister {
                    bar_type: BarType::Memory64,
                    address: 0x6000_0000,
                    size: 16 * 1024 * 1024,
                    prefetchable: false,
                },
                BaseAddressRegister {
                    bar_type: BarType::Memory64,
                    address: 0x4000_0000_0000,
                    size: 256 * 1024 * 1024,
                    prefetchable: true,
                },
            ],
            irq_line: 10,
        });

        // Intel Arc A770 (Xe-HPG discrete)
        self.push_device(PciDevice {
            address: PciAddress {
                bus: 3,
                device: 0,
                function: 0,
            },
            vendor_id: INTEL_VENDOR_ID,
            device_id: 0x56A0,
            class_code: 0x03,
            subclass: 0x00,
            revision: 0x08,
            bars: alloc::vec![
                BaseAddressRegister {
                    bar_type: BarType::Memory64,
                    address: 0xFB00_0000,
                    size: 32 * 1024 * 1024,
                    prefetchable: false,
                },
                BaseAddressRegister {
                    bar_type: BarType::Memory64,
                    address: 0x3800_0000_0000,
                    size: 16 * 1024 * 1024 * 1024, // 16 GiB aperture
                    prefetchable: true,
                },
            ],
            irq_line: 11,
        });

        // Intel Data Center GPU Max 1550 (Xe-HPC)
        self.push_device(PciDevice {
            address: PciAddress {
                bus: 4,
                device: 0,
                function: 0,
            },
            vendor_id: INTEL_VENDOR_ID,
            device_id: 0x0BD5,
            class_code: 0x03,
            subclass: 0x02,
            revision: 0x05,
            bars: alloc::vec![
                BaseAddressRegister {
                    bar_type: BarType::Memory64,
                    address: 0xFC00_0000,
                    size: 64 * 1024 * 1024,
                    prefetchable: false,
                },
                BaseAddressRegister {
                    bar_type: BarType::Memory64,
                    address: 0x3900_0000_0000,
                    size: 128u64 * 1024 * 1024 * 1024, // 128 GiB aperture
                    prefetchable: true,
                },
            ],
            irq_line: 12,
        });

        // Non-Intel device (NVIDIA, for filter testing)
        self.push_device(PciDevice {
            address: PciAddress {
                bus: 1,
                device: 0,
                function: 0,
            },
            vendor_id: 0x10DE,
            device_id: 0x1B80,
            class_code: 0x03,
            subclass: 0x02,
            revision: 0xA1,
            bars: alloc::vec![BaseAddressRegister {
                bar_type: BarType::Memory64,
                address: 0xDE00_0000,
                size: 16 * 1024 * 1024,
                prefetchable: false,
            }],
            irq_line: 13,
        });

        Ok(())
    }

    /// Add a device to the internal list (capped at `MAX_PCI_DEVICES`).
    fn push_device(&mut self, device: PciDevice) {
        if self.devices.len() < MAX_PCI_DEVICES {
            self.devices.push(device);
        }
    }

    /// Return references to only the Intel-vendor devices.
    pub fn intel_devices(&self) -> Vec<&PciDevice> {
        self.devices.iter().filter(|d| d.is_intel()).collect()
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
            address: 0x4000_0000_0000,
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
    fn device_is_intel() {
        let dev = PciDevice {
            address: PciAddress {
                bus: 0,
                device: 2,
                function: 0,
            },
            vendor_id: INTEL_VENDOR_ID,
            device_id: 0x9A49,
            class_code: 0x03,
            subclass: 0x00,
            revision: 0x01,
            bars: alloc::vec![],
            irq_line: 10,
        };
        assert!(dev.is_intel());
    }

    #[test]
    fn device_is_not_intel() {
        let dev = PciDevice {
            address: PciAddress {
                bus: 0,
                device: 0,
                function: 0,
            },
            vendor_id: 0x10DE,
            device_id: 0x1B80,
            class_code: 0x03,
            subclass: 0x02,
            revision: 0xA1,
            bars: alloc::vec![],
            irq_line: 11,
        };
        assert!(!dev.is_intel());
    }

    #[test]
    fn device_is_display_controller() {
        let dev = PciDevice {
            address: PciAddress {
                bus: 0,
                device: 2,
                function: 0,
            },
            vendor_id: INTEL_VENDOR_ID,
            device_id: 0x9A49,
            class_code: 0x03,
            subclass: 0x00,
            revision: 0,
            bars: alloc::vec![],
            irq_line: 0,
        };
        assert!(dev.is_display_controller());
    }

    #[test]
    fn device_is_vga_controller() {
        let dev = PciDevice {
            address: PciAddress {
                bus: 0,
                device: 2,
                function: 0,
            },
            vendor_id: INTEL_VENDOR_ID,
            device_id: 0x9A49,
            class_code: 0x03,
            subclass: 0x00,
            revision: 0,
            bars: alloc::vec![],
            irq_line: 0,
        };
        assert!(dev.is_vga_controller());
    }

    #[test]
    fn device_is_3d_controller() {
        let dev = PciDevice {
            address: PciAddress {
                bus: 0,
                device: 0,
                function: 0,
            },
            vendor_id: INTEL_VENDOR_ID,
            device_id: 0x0BD5,
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
            vendor_id: INTEL_VENDOR_ID,
            device_id: 0x9A49,
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
        assert_eq!(scanner.device_count(), 4);
    }

    #[test]
    fn scanner_intel_filter() {
        let mut scanner = PciScanner::new();
        scanner.scan().unwrap();
        let intel = scanner.intel_devices();
        assert_eq!(intel.len(), 3, "mock scan has exactly 3 Intel devices");
        for dev in &intel {
            assert!(dev.is_intel());
        }
    }

    #[test]
    fn scanner_non_intel_present() {
        let mut scanner = PciScanner::new();
        scanner.scan().unwrap();
        let non_intel: Vec<_> = scanner.devices().iter().filter(|d| !d.is_intel()).collect();
        assert_eq!(non_intel.len(), 1, "mock scan has 1 non-Intel device");
        assert_eq!(non_intel[0].vendor_id, 0x10DE);
    }

    #[test]
    fn scanner_device_ids_for_known_families() {
        let mut scanner = PciScanner::new();
        scanner.scan().unwrap();
        let intel = scanner.intel_devices();
        let ids: Vec<u16> = intel.iter().map(|d| d.device_id).collect();
        assert!(ids.contains(&0x9A49), "should contain Tiger Lake iGPU");
        assert!(ids.contains(&0x56A0), "should contain Arc A770");
        assert!(ids.contains(&0x0BD5), "should contain DC GPU Max 1550");
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
                vendor_id: INTEL_VENDOR_ID,
                device_id: 0x9A49,
                class_code: 0x03,
                subclass: 0x00,
                revision: 0,
                bars: alloc::vec![],
                irq_line: 0,
            });
        }
        assert_eq!(scanner.device_count(), MAX_PCI_DEVICES);
    }

    #[test]
    fn pci_address_ranges_valid() {
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
        let intel = scanner.intel_devices();
        for dev in &intel {
            assert!(
                !dev.bars.is_empty(),
                "Intel mock devices should have at least one BAR"
            );
        }
    }
}
