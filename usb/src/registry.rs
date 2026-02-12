// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! USB device registry — tracks enumerated devices and driver bindings.

use crate::descriptor::DeviceDescriptor;
use crate::UsbError;

/// Maximum number of concurrently tracked USB devices.
pub const MAX_DEVICES: usize = 16;

/// Maximum number of interfaces per device that can be tracked.
pub const MAX_INTERFACES_PER_DEVICE: usize = 8;

/// Device match criteria for driver binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceMatch {
    /// Match by vendor ID and product ID.
    VidPid { vendor_id: u16, product_id: u16 },
    /// Match by interface class, subclass, and protocol.
    InterfaceClass {
        class: u8,
        subclass: u8,
        protocol: u8,
    },
}

/// Information about an enumerated USB device.
#[derive(Debug, Clone)]
pub struct DeviceEntry {
    /// Host controller slot ID.
    pub slot: u8,
    /// Root hub port number.
    pub port: u8,
    /// Vendor ID.
    pub vendor_id: u16,
    /// Product ID.
    pub product_id: u16,
    /// Device class.
    pub device_class: u8,
    /// Interface claim bitmap (bit N = interface N is claimed).
    pub claimed_interfaces: u8,
    /// Whether this entry is active (device connected).
    pub active: bool,
}

impl Default for DeviceEntry {
    fn default() -> Self {
        Self {
            slot: 0,
            port: 0,
            vendor_id: 0,
            product_id: 0,
            device_class: 0,
            claimed_interfaces: 0,
            active: false,
        }
    }
}

/// USB device registry.
///
/// Maintains a list of connected devices and tracks which interfaces
/// have been claimed by drivers.
pub struct DeviceRegistry {
    devices: [DeviceEntry; MAX_DEVICES],
    count: usize,
}

impl Default for DeviceRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl DeviceRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            devices: core::array::from_fn(|_| DeviceEntry::default()),
            count: 0,
        }
    }

    /// Register a newly enumerated device. Returns its index in the registry.
    pub fn register(
        &mut self,
        slot: u8,
        port: u8,
        desc: &DeviceDescriptor,
    ) -> Result<usize, UsbError> {
        // Look for an existing inactive slot to reuse.
        for i in 0..MAX_DEVICES {
            if !self.devices[i].active {
                self.devices[i] = DeviceEntry {
                    slot,
                    port,
                    vendor_id: desc.vendor_id,
                    product_id: desc.product_id,
                    device_class: desc.device_class,
                    claimed_interfaces: 0,
                    active: true,
                };
                if i >= self.count {
                    self.count = i + 1;
                }
                return Ok(i);
            }
        }
        Err(UsbError::TooManyDevices)
    }

    /// Unregister a device (on disconnection). Clears the entry.
    pub fn unregister(&mut self, slot: u8) {
        for entry in self.devices.iter_mut() {
            if entry.active && entry.slot == slot {
                entry.active = false;
                entry.claimed_interfaces = 0;
                return;
            }
        }
    }

    /// Find a device by VID/PID match.
    pub fn find_by_vid_pid(&self, vendor_id: u16, product_id: u16) -> Option<&DeviceEntry> {
        self.devices.iter().find(|d| {
            d.active && d.vendor_id == vendor_id && d.product_id == product_id
        })
    }

    /// Find a device by slot ID.
    pub fn find_by_slot(&self, slot: u8) -> Option<&DeviceEntry> {
        self.devices.iter().find(|d| d.active && d.slot == slot)
    }

    /// Claim an interface on a device. Returns error if already claimed.
    pub fn claim_interface(&mut self, slot: u8, interface: u8) -> Result<(), UsbError> {
        if interface >= MAX_INTERFACES_PER_DEVICE as u8 {
            return Err(UsbError::InvalidEndpoint);
        }
        for entry in self.devices.iter_mut() {
            if entry.active && entry.slot == slot {
                let mask = 1 << interface;
                if entry.claimed_interfaces & mask != 0 {
                    return Err(UsbError::InterfaceAlreadyClaimed);
                }
                entry.claimed_interfaces |= mask;
                return Ok(());
            }
        }
        Err(UsbError::DeviceNotFound)
    }

    /// Release an interface claim.
    pub fn release_interface(&mut self, slot: u8, interface: u8) {
        if interface >= MAX_INTERFACES_PER_DEVICE as u8 {
            return;
        }
        for entry in self.devices.iter_mut() {
            if entry.active && entry.slot == slot {
                entry.claimed_interfaces &= !(1 << interface);
                return;
            }
        }
    }

    /// Return the number of active devices.
    pub fn active_count(&self) -> usize {
        self.devices.iter().filter(|d| d.active).count()
    }

    /// Check if a device matches the given criteria.
    pub fn matches(entry: &DeviceEntry, criteria: &DeviceMatch) -> bool {
        match criteria {
            DeviceMatch::VidPid { vendor_id, product_id } => {
                entry.vendor_id == *vendor_id && entry.product_id == *product_id
            }
            DeviceMatch::InterfaceClass { class, .. } => {
                // Device-level class check; interface-level matching requires
                // looking at the configuration descriptor.
                entry.device_class == *class
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::descriptor::DeviceDescriptor;

    fn make_desc(vid: u16, pid: u16) -> DeviceDescriptor {
        DeviceDescriptor {
            bcd_usb: 0x0200,
            device_class: 0xFF,
            device_sub_class: 0,
            device_protocol: 0,
            max_packet_size_0: 64,
            vendor_id: vid,
            product_id: pid,
            bcd_device: 0x0100,
            manufacturer_index: 0,
            product_index: 0,
            serial_number_index: 0,
            num_configurations: 1,
        }
    }

    #[test]
    fn test_registry_register_find() {
        let mut reg = DeviceRegistry::new();
        let desc = make_desc(0x1D50, 0x6089);
        let idx = reg.register(1, 0, &desc).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(reg.active_count(), 1);

        let found = reg.find_by_vid_pid(0x1D50, 0x6089).unwrap();
        assert_eq!(found.slot, 1);
        assert_eq!(found.port, 0);
    }

    #[test]
    fn test_registry_find_not_found() {
        let reg = DeviceRegistry::new();
        assert!(reg.find_by_vid_pid(0x1D50, 0x6089).is_none());
    }

    #[test]
    fn test_registry_unregister() {
        let mut reg = DeviceRegistry::new();
        let desc = make_desc(0x0456, 0xb673);
        reg.register(2, 1, &desc).unwrap();
        assert_eq!(reg.active_count(), 1);

        reg.unregister(2);
        assert_eq!(reg.active_count(), 0);
        assert!(reg.find_by_slot(2).is_none());
    }

    #[test]
    fn test_registry_claim_interface() {
        let mut reg = DeviceRegistry::new();
        let desc = make_desc(0x1D50, 0x6089);
        reg.register(1, 0, &desc).unwrap();

        reg.claim_interface(1, 0).unwrap();
        // Double claim should fail.
        assert_eq!(reg.claim_interface(1, 0), Err(UsbError::InterfaceAlreadyClaimed));
    }

    #[test]
    fn test_registry_release_interface() {
        let mut reg = DeviceRegistry::new();
        let desc = make_desc(0x1D50, 0x6089);
        reg.register(1, 0, &desc).unwrap();
        reg.claim_interface(1, 0).unwrap();
        reg.release_interface(1, 0);
        // Should be able to claim again after release.
        reg.claim_interface(1, 0).unwrap();
    }

    #[test]
    fn test_registry_claim_nonexistent_device() {
        let mut reg = DeviceRegistry::new();
        assert_eq!(reg.claim_interface(99, 0), Err(UsbError::DeviceNotFound));
    }

    #[test]
    fn test_registry_too_many_devices() {
        let mut reg = DeviceRegistry::new();
        for i in 0..MAX_DEVICES {
            let desc = make_desc(i as u16, 0);
            reg.register(i as u8 + 1, 0, &desc).unwrap();
        }
        let desc = make_desc(0xFF, 0xFF);
        assert_eq!(reg.register(100, 0, &desc), Err(UsbError::TooManyDevices));
    }

    #[test]
    fn test_registry_reuse_slot() {
        let mut reg = DeviceRegistry::new();
        let desc = make_desc(0x1D50, 0x6089);
        reg.register(1, 0, &desc).unwrap();
        reg.unregister(1);
        // Should reuse the freed slot.
        let desc2 = make_desc(0x0456, 0xb673);
        let idx = reg.register(2, 1, &desc2).unwrap();
        assert_eq!(idx, 0); // Reused slot 0
    }

    #[test]
    fn test_device_match_vid_pid() {
        let entry = DeviceEntry {
            slot: 1, port: 0, vendor_id: 0x1D50, product_id: 0x6089,
            device_class: 0xFF, claimed_interfaces: 0, active: true,
        };
        let m = DeviceMatch::VidPid { vendor_id: 0x1D50, product_id: 0x6089 };
        assert!(DeviceRegistry::matches(&entry, &m));

        let m2 = DeviceMatch::VidPid { vendor_id: 0x0456, product_id: 0xb673 };
        assert!(!DeviceRegistry::matches(&entry, &m2));
    }
}
