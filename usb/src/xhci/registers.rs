// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! xHCI register definitions and capability parsing.
//!
//! Defines the memory-mapped register layout for xHCI controllers,
//! including Capability, Operational, Runtime, and Doorbell registers.

// ---------------------------------------------------------------------------
// PCIe class codes for xHCI detection
// ---------------------------------------------------------------------------

/// PCI class code for USB controller.
pub const PCI_CLASS_SERIAL_BUS: u8 = 0x0C;
/// PCI subclass for USB.
pub const PCI_SUBCLASS_USB: u8 = 0x03;
/// PCI programming interface for xHCI.
pub const PCI_PROGIF_XHCI: u8 = 0x30;

// ---------------------------------------------------------------------------
// Capability Register offsets (xHCI Section 5.3)
// ---------------------------------------------------------------------------

/// CAPLENGTH — Capability Register Length (offset 0x00, 1 byte).
pub const CAP_CAPLENGTH: usize = 0x00;
/// HCIVERSION — Host Controller Interface Version (offset 0x02, 2 bytes).
pub const CAP_HCIVERSION: usize = 0x02;
/// HCSPARAMS1 — Structural Parameters 1 (offset 0x04, 4 bytes).
pub const CAP_HCSPARAMS1: usize = 0x04;
/// HCSPARAMS2 — Structural Parameters 2 (offset 0x08, 4 bytes).
pub const CAP_HCSPARAMS2: usize = 0x08;
/// HCSPARAMS3 — Structural Parameters 3 (offset 0x0C, 4 bytes).
pub const CAP_HCSPARAMS3: usize = 0x0C;
/// HCCPARAMS1 — Capability Parameters 1 (offset 0x10, 4 bytes).
pub const CAP_HCCPARAMS1: usize = 0x10;
/// DBOFF — Doorbell Offset (offset 0x14, 4 bytes).
pub const CAP_DBOFF: usize = 0x14;
/// RTSOFF — Runtime Register Space Offset (offset 0x18, 4 bytes).
pub const CAP_RTSOFF: usize = 0x18;
/// HCCPARAMS2 — Capability Parameters 2 (offset 0x1C, 4 bytes).
pub const CAP_HCCPARAMS2: usize = 0x1C;

// ---------------------------------------------------------------------------
// Operational Register offsets (relative to CAPLENGTH)
// ---------------------------------------------------------------------------

/// USBCMD — USB Command Register.
pub const OP_USBCMD: usize = 0x00;
/// USBSTS — USB Status Register.
pub const OP_USBSTS: usize = 0x04;
/// PAGESIZE — Page Size Register.
pub const OP_PAGESIZE: usize = 0x08;
/// DNCTRL — Device Notification Control.
pub const OP_DNCTRL: usize = 0x14;
/// CRCR — Command Ring Control Register (64-bit).
pub const OP_CRCR: usize = 0x18;
/// DCBAAP — Device Context Base Address Array Pointer (64-bit).
pub const OP_DCBAAP: usize = 0x30;
/// CONFIG — Configure Register.
pub const OP_CONFIG: usize = 0x38;

/// Port register set base offset (relative to operational registers).
pub const OP_PORT_BASE: usize = 0x400;
/// Size of each port register set.
pub const PORT_REG_SET_SIZE: usize = 0x10;

// ---------------------------------------------------------------------------
// USBCMD bits
// ---------------------------------------------------------------------------

/// Run/Stop bit.
pub const USBCMD_RS: u32 = 1 << 0;
/// Host Controller Reset.
pub const USBCMD_HCRST: u32 = 1 << 1;
/// Interrupter Enable.
pub const USBCMD_INTE: u32 = 1 << 2;

// ---------------------------------------------------------------------------
// USBSTS bits
// ---------------------------------------------------------------------------

/// Host Controller Halted.
pub const USBSTS_HCH: u32 = 1 << 0;
/// Controller Not Ready.
pub const USBSTS_CNR: u32 = 1 << 11;
/// Event Interrupt.
pub const USBSTS_EINT: u32 = 1 << 3;
/// Port Change Detect.
pub const USBSTS_PCD: u32 = 1 << 4;

// ---------------------------------------------------------------------------
// PORTSC bits (Port Status and Control)
// ---------------------------------------------------------------------------

/// Port offset for port N (0-based).
pub fn port_sc_offset(port: u8) -> usize {
    OP_PORT_BASE + (port as usize) * PORT_REG_SET_SIZE
}

/// Current Connect Status.
pub const PORTSC_CCS: u32 = 1 << 0;
/// Port Enabled/Disabled.
pub const PORTSC_PED: u32 = 1 << 1;
/// Port Reset.
pub const PORTSC_PR: u32 = 1 << 4;
/// Port Power.
pub const PORTSC_PP: u32 = 1 << 9;
/// Port Speed (bits 13:10).
pub const PORTSC_SPEED_MASK: u32 = 0xF << 10;
/// Port Speed shift.
pub const PORTSC_SPEED_SHIFT: u32 = 10;
/// Connect Status Change.
pub const PORTSC_CSC: u32 = 1 << 17;
/// Port Reset Change.
pub const PORTSC_PRC: u32 = 1 << 21;

/// Port speed values.
pub mod port_speed {
    /// Full-speed (12 Mbps).
    pub const FULL_SPEED: u32 = 1;
    /// Low-speed (1.5 Mbps).
    pub const LOW_SPEED: u32 = 2;
    /// High-speed (480 Mbps).
    pub const HIGH_SPEED: u32 = 3;
    /// SuperSpeed (5 Gbps).
    pub const SUPER_SPEED: u32 = 4;
    /// SuperSpeedPlus (10+ Gbps).
    pub const SUPER_SPEED_PLUS: u32 = 5;
}

// ---------------------------------------------------------------------------
// Parsed capability parameters
// ---------------------------------------------------------------------------

/// Parsed xHCI capability parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XhciCapabilities {
    /// Operational register offset from BAR0.
    pub op_reg_offset: u8,
    /// HCI version (e.g., 0x0100 = 1.0, 0x0110 = 1.1).
    pub hci_version: u16,
    /// Maximum number of device slots.
    pub max_slots: u8,
    /// Maximum number of interrupters.
    pub max_intrs: u16,
    /// Maximum number of ports.
    pub max_ports: u8,
    /// Number of scratchpad buffers required.
    pub scratchpad_bufs: u16,
    /// Doorbell array offset from BAR0.
    pub doorbell_offset: u32,
    /// Runtime register offset from BAR0.
    pub runtime_offset: u32,
    /// Context size: true = 64-byte contexts, false = 32-byte contexts.
    pub context_size_64: bool,
}

impl XhciCapabilities {
    /// Parse capabilities from raw register values.
    ///
    /// `caplength_hciversion`: raw 32-bit value at offset 0x00.
    /// `hcsparams1`: raw HCSPARAMS1 value.
    /// `hcsparams2`: raw HCSPARAMS2 value.
    /// `hccparams1`: raw HCCPARAMS1 value.
    /// `dboff`: raw DBOFF value.
    /// `rtsoff`: raw RTSOFF value.
    pub fn parse(
        caplength_hciversion: u32,
        hcsparams1: u32,
        hcsparams2: u32,
        hccparams1: u32,
        dboff: u32,
        rtsoff: u32,
    ) -> Self {
        let caplength = (caplength_hciversion & 0xFF) as u8;
        let hci_version = ((caplength_hciversion >> 16) & 0xFFFF) as u16;

        let max_slots = (hcsparams1 & 0xFF) as u8;
        let max_intrs = ((hcsparams1 >> 8) & 0x7FF) as u16;
        let max_ports = ((hcsparams1 >> 24) & 0xFF) as u8;

        let scratchpad_hi = ((hcsparams2 >> 21) & 0x1F) as u16;
        let scratchpad_lo = ((hcsparams2 >> 27) & 0x1F) as u16;
        let scratchpad_bufs = (scratchpad_hi << 5) | scratchpad_lo;

        let context_size_64 = hccparams1 & (1 << 2) != 0; // CSZ bit

        Self {
            op_reg_offset: caplength,
            hci_version,
            max_slots,
            max_intrs,
            max_ports,
            scratchpad_bufs,
            doorbell_offset: dboff & !0x3, // Must be 4-byte aligned
            runtime_offset: rtsoff & !0x1F, // Must be 32-byte aligned
            context_size_64,
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_capabilities() {
        // Simulate a typical xHCI controller.
        let caplength_hciversion = 0x0100_0020; // version 1.0, caplength = 0x20
        let hcsparams1 = (4u32 << 24) | (1 << 8) | 16; // 4 ports, 1 interrupter, 16 slots
        let hcsparams2 = 0; // no scratchpad
        let hccparams1 = 0; // 32-byte contexts
        let dboff = 0x2000;
        let rtsoff = 0x3000;

        let caps = XhciCapabilities::parse(
            caplength_hciversion, hcsparams1, hcsparams2, hccparams1, dboff, rtsoff,
        );

        assert_eq!(caps.op_reg_offset, 0x20);
        assert_eq!(caps.hci_version, 0x0100);
        assert_eq!(caps.max_slots, 16);
        assert_eq!(caps.max_intrs, 1);
        assert_eq!(caps.max_ports, 4);
        assert_eq!(caps.scratchpad_bufs, 0);
        assert_eq!(caps.doorbell_offset, 0x2000);
        assert_eq!(caps.runtime_offset, 0x3000);
        assert!(!caps.context_size_64);
    }

    #[test]
    fn test_parse_capabilities_64bit_context() {
        let caps = XhciCapabilities::parse(
            0x0110_0020, // version 1.1
            0x0200_0100 | 32, // 2 ports, 1 intr, 32 slots
            0,
            0x04, // CSZ bit set
            0x800,
            0xA00,
        );

        assert!(caps.context_size_64);
        assert_eq!(caps.hci_version, 0x0110);
        assert_eq!(caps.max_slots, 32);
        assert_eq!(caps.max_ports, 2);
    }

    #[test]
    fn test_port_sc_offset() {
        assert_eq!(port_sc_offset(0), 0x400);
        assert_eq!(port_sc_offset(1), 0x410);
        assert_eq!(port_sc_offset(3), 0x430);
    }

    #[test]
    fn test_pci_class_constants() {
        assert_eq!(PCI_CLASS_SERIAL_BUS, 0x0C);
        assert_eq!(PCI_SUBCLASS_USB, 0x03);
        assert_eq!(PCI_PROGIF_XHCI, 0x30);
    }

    #[test]
    fn test_usbcmd_bits() {
        assert_eq!(USBCMD_RS, 1);
        assert_eq!(USBCMD_HCRST, 2);
        assert_eq!(USBCMD_INTE, 4);
    }

    #[test]
    fn test_portsc_speed_extraction() {
        // Simulate PORTSC with High Speed (3).
        let portsc = (port_speed::HIGH_SPEED << PORTSC_SPEED_SHIFT) | PORTSC_CCS | PORTSC_PED;
        let speed = (portsc & PORTSC_SPEED_MASK) >> PORTSC_SPEED_SHIFT;
        assert_eq!(speed, port_speed::HIGH_SPEED);
        assert!(portsc & PORTSC_CCS != 0);
        assert!(portsc & PORTSC_PED != 0);
    }
}
