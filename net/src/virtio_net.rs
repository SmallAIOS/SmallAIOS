// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! virtio-net device driver (MMIO transport) for SmallAIOS.
//!
//! Provides [`VirtioNetDevice`] for initializing and operating a virtio-net
//! NIC.  The current implementation stubs out actual MMIO register access and
//! virtqueue operations; the public API is ready for integration once the
//! MMIO transport layer is available.

use crate::ethernet::MacAddress;
use crate::NetError;

// ---------------------------------------------------------------------------
// VirtioFeatures
// ---------------------------------------------------------------------------

/// Feature bits negotiated during virtio-net device initialization.
pub struct VirtioFeatures;

impl VirtioFeatures {
    /// Device has a valid MAC address in its configuration space.
    pub const MAC: u32 = 0x20;
    /// Device reports link status in the `status` field.
    pub const STATUS: u32 = 0x10000;
    /// Driver can merge receive buffers.
    pub const MRG_RXBUF: u32 = 0x8000;
}

// ---------------------------------------------------------------------------
// VirtioNetConfig
// ---------------------------------------------------------------------------

/// Configuration read from the virtio-net device.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VirtioNetConfig {
    /// MAC address provisioned by the hypervisor.
    pub mac: MacAddress,
    /// Link status (bit 0: link up).
    pub status: u16,
    /// Maximum number of transmit/receive queue pairs.
    pub max_virtqueue_pairs: u16,
}

// ---------------------------------------------------------------------------
// VirtioNetDevice
// ---------------------------------------------------------------------------

/// A virtio-net network device (MMIO transport).
pub struct VirtioNetDevice {
    /// Device configuration.
    config: VirtioNetConfig,
    /// Whether [`VirtioNetDevice::initialize`] has completed.
    is_initialized: bool,
    /// Number of descriptors in the receive virtqueue.
    rx_queue_size: u16,
    /// Number of descriptors in the transmit virtqueue.
    tx_queue_size: u16,
}

impl VirtioNetDevice {
    /// Default virtqueue size for both RX and TX.
    const DEFAULT_QUEUE_SIZE: u16 = 256;

    /// Create a new [`VirtioNetDevice`] with the given MAC address.
    ///
    /// The device is in an uninitialized state; call [`VirtioNetDevice::initialize`]
    /// before sending or receiving.
    pub fn new(mac: MacAddress) -> Self {
        VirtioNetDevice {
            config: VirtioNetConfig {
                mac,
                status: 0,
                max_virtqueue_pairs: 1,
            },
            is_initialized: false,
            rx_queue_size: Self::DEFAULT_QUEUE_SIZE,
            tx_queue_size: Self::DEFAULT_QUEUE_SIZE,
        }
    }

    /// Initialize the virtio-net device.
    ///
    /// Performs feature negotiation, virtqueue setup, and driver-ok signalling.
    /// Currently a stub that marks the device as initialized.
    pub fn initialize(&mut self) -> Result<(), NetError> {
        // TODO: actual MMIO register access and virtqueue setup.
        self.is_initialized = true;
        self.config.status = 1; // link up
        Ok(())
    }

    /// Transmit an Ethernet frame.
    ///
    /// `frame` should be a complete Ethernet frame (header + payload).
    /// Returns [`NetError::InvalidProtocol`] if the device is not initialized.
    /// The actual transmit path is not yet implemented.
    pub fn send(&self, _frame: &[u8]) -> Result<(), NetError> {
        if !self.is_initialized {
            return Err(NetError::InvalidProtocol);
        }
        Err(NetError::NotImplemented)
    }

    /// Receive an Ethernet frame into `buf`.
    ///
    /// Returns the number of bytes written to `buf`.
    /// Returns [`NetError::InvalidProtocol`] if the device is not initialized.
    /// The actual receive path is not yet implemented.
    pub fn receive(&self, _buf: &mut [u8]) -> Result<usize, NetError> {
        if !self.is_initialized {
            return Err(NetError::InvalidProtocol);
        }
        Err(NetError::NotImplemented)
    }

    /// Return the device's MAC address.
    pub fn mac_address(&self) -> MacAddress {
        self.config.mac
    }

    /// Return `true` if the device has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.is_initialized
    }

    /// Return the current device configuration.
    pub fn config(&self) -> &VirtioNetConfig {
        &self.config
    }

    /// Return the receive queue size.
    pub fn rx_queue_size(&self) -> u16 {
        self.rx_queue_size
    }

    /// Return the transmit queue size.
    pub fn tx_queue_size(&self) -> u16 {
        self.tx_queue_size
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_mac() -> MacAddress {
        MacAddress::new(0x52, 0x54, 0x00, 0x12, 0x34, 0x56)
    }

    // -- Device creation ----------------------------------------------------

    #[test]
    fn test_device_creation() {
        let dev = VirtioNetDevice::new(test_mac());
        assert!(!dev.is_initialized());
        assert_eq!(dev.mac_address(), test_mac());
        assert_eq!(dev.rx_queue_size(), 256);
        assert_eq!(dev.tx_queue_size(), 256);
    }

    #[test]
    fn test_mac_address() {
        let mac = MacAddress::new(0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF);
        let dev = VirtioNetDevice::new(mac);
        assert_eq!(dev.mac_address(), mac);
        assert_eq!(
            dev.mac_address().as_bytes(),
            &[0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF]
        );
    }

    #[test]
    fn test_config_after_creation() {
        let dev = VirtioNetDevice::new(test_mac());
        let cfg = dev.config();
        assert_eq!(cfg.mac, test_mac());
        assert_eq!(cfg.status, 0);
        assert_eq!(cfg.max_virtqueue_pairs, 1);
    }

    // -- Initialization -----------------------------------------------------

    #[test]
    fn test_initialize() {
        let mut dev = VirtioNetDevice::new(test_mac());
        assert!(!dev.is_initialized());
        dev.initialize().unwrap();
        assert!(dev.is_initialized());
        assert_eq!(dev.config().status, 1);
    }

    // -- Send / receive stubs -----------------------------------------------

    #[test]
    fn test_send_before_init() {
        let dev = VirtioNetDevice::new(test_mac());
        let frame = [0u8; 64];
        assert_eq!(dev.send(&frame), Err(NetError::InvalidProtocol));
    }

    #[test]
    fn test_send_after_init() {
        let mut dev = VirtioNetDevice::new(test_mac());
        dev.initialize().unwrap();
        let frame = [0u8; 64];
        assert_eq!(dev.send(&frame), Err(NetError::NotImplemented));
    }

    #[test]
    fn test_receive_before_init() {
        let dev = VirtioNetDevice::new(test_mac());
        let mut buf = [0u8; 1514];
        assert_eq!(dev.receive(&mut buf), Err(NetError::InvalidProtocol));
    }

    #[test]
    fn test_receive_after_init() {
        let mut dev = VirtioNetDevice::new(test_mac());
        dev.initialize().unwrap();
        let mut buf = [0u8; 1514];
        assert_eq!(dev.receive(&mut buf), Err(NetError::NotImplemented));
    }

    // -- Feature constants --------------------------------------------------

    #[test]
    fn test_feature_constants() {
        assert_eq!(VirtioFeatures::MAC, 0x20);
        assert_eq!(VirtioFeatures::STATUS, 0x10000);
        assert_eq!(VirtioFeatures::MRG_RXBUF, 0x8000);
    }

    #[test]
    fn test_default_queue_sizes() {
        let dev = VirtioNetDevice::new(test_mac());
        assert_eq!(dev.rx_queue_size(), VirtioNetDevice::DEFAULT_QUEUE_SIZE);
        assert_eq!(dev.tx_queue_size(), VirtioNetDevice::DEFAULT_QUEUE_SIZE);
    }
}
