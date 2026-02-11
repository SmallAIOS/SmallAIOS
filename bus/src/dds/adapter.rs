// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! DDS-to-Zenoh transport adapter.
//!
//! Maps DDS topics to Zenoh key expressions: `dds/{domain_id}/{topic}`
//!
//! Also supports ROS 2 topic name mangling with `rt/`, `rq/`, `rr/` prefixes.
//!
//! # Key Expression Mapping
//!
//! - DDS topic "SensorData" in domain 0 -> `dds/0/SensorData`
//! - ROS 2 topic "/robot/cmd_vel" -> `dds/0/rt/robot/cmd_vel`
//! - ROS 2 service request "/robot/get_state" -> `dds/0/rq/robot/get_state`
//! - ROS 2 service reply "/robot/get_state" -> `dds/0/rr/robot/get_state`

use alloc::collections::VecDeque;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use crate::transport::{BusSample, ZenohTransport};
use crate::BusError;
use super::types::MAX_DOMAIN_ID;

/// Maximum number of queued receive samples.
const MAX_RX_QUEUE: usize = 256;

/// ROS 2 topic name prefix for data topics.
pub const ROS2_TOPIC_PREFIX: &str = "rt/";

/// ROS 2 topic name prefix for service requests.
pub const ROS2_REQUEST_PREFIX: &str = "rq/";

/// ROS 2 topic name prefix for service replies.
pub const ROS2_REPLY_PREFIX: &str = "rr/";

/// DDS-to-Zenoh transport adapter.
pub struct DdsZenohAdapter {
    /// DDS domain ID.
    domain_id: u32,
    /// Key expression prefix (e.g., "dds/0").
    prefix: String,
    /// Receive buffer for DDS samples.
    rx_queue: VecDeque<(String, Vec<u8>)>,
}

impl DdsZenohAdapter {
    /// Create a new DDS-to-Zenoh adapter for the given domain.
    pub fn new(domain_id: u32) -> Result<Self, BusError> {
        if domain_id > MAX_DOMAIN_ID {
            return Err(BusError::InvalidDomain);
        }
        Ok(Self {
            domain_id,
            prefix: format!("dds/{}", domain_id),
            rx_queue: VecDeque::new(),
        })
    }

    /// Returns the DDS domain ID.
    pub fn domain_id(&self) -> u32 {
        self.domain_id
    }

    /// Build a key expression for a DDS topic.
    pub fn key_for_topic(&self, topic_name: &str) -> String {
        format!("{}/{}", self.prefix, topic_name)
    }

    /// Convert a ROS 2 topic name to its DDS mangled form.
    ///
    /// ROS 2 convention:
    /// - Data topic "/foo/bar" -> topic name "rt/foo/bar"
    /// - Service request "/foo/bar" -> topic name "rq/foo/bar"
    /// - Service reply "/foo/bar" -> topic name "rr/foo/bar"
    pub fn ros2_topic_to_dds(ros2_topic: &str, prefix: &str) -> String {
        let cleaned = ros2_topic.strip_prefix('/').unwrap_or(ros2_topic);
        format!("{}{}", prefix, cleaned)
    }

    /// Build a key expression for a ROS 2 data topic.
    pub fn key_for_ros2_topic(&self, ros2_topic: &str) -> String {
        let dds_name = Self::ros2_topic_to_dds(ros2_topic, ROS2_TOPIC_PREFIX);
        self.key_for_topic(&dds_name)
    }

    /// Build a key expression for a ROS 2 service request topic.
    pub fn key_for_ros2_request(&self, service_name: &str) -> String {
        let dds_name = Self::ros2_topic_to_dds(service_name, ROS2_REQUEST_PREFIX);
        self.key_for_topic(&dds_name)
    }

    /// Build a key expression for a ROS 2 service reply topic.
    pub fn key_for_ros2_reply(&self, service_name: &str) -> String {
        let dds_name = Self::ros2_topic_to_dds(service_name, ROS2_REPLY_PREFIX);
        self.key_for_topic(&dds_name)
    }

    /// Check if a key expression represents a ROS 2 topic.
    pub fn is_ros2_key(key_expr: &str) -> bool {
        // After stripping the dds/{domain_id}/ prefix, check for rt/, rq/, rr/
        if let Some(remainder) = key_expr.strip_prefix("dds/") {
            if let Some(after_domain) = remainder.split_once('/').map(|(_, rest)| rest) {
                return after_domain.starts_with("rt/")
                    || after_domain.starts_with("rq/")
                    || after_domain.starts_with("rr/");
            }
        }
        false
    }

    /// Extract the DDS topic name from a key expression.
    fn extract_topic<'a>(&self, key_expr: &'a str) -> Result<&'a str, BusError> {
        key_expr
            .strip_prefix(&self.prefix)
            .and_then(|s| s.strip_prefix('/'))
            .ok_or(BusError::InvalidDomain)
    }

    /// Inject a received DDS sample into the adapter's RX queue.
    pub fn inject_rx(&mut self, topic_name: &str, data: &[u8]) {
        if self.rx_queue.len() >= MAX_RX_QUEUE {
            return;
        }
        let key = self.key_for_topic(topic_name);
        self.rx_queue.push_back((key, Vec::from(data)));
    }

    /// Return the number of queued receive samples.
    pub fn rx_count(&self) -> usize {
        self.rx_queue.len()
    }
}

impl ZenohTransport for DdsZenohAdapter {
    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn transmit(&mut self, key_expr: &str, payload: &[u8]) -> Result<(), BusError> {
        let _topic = self.extract_topic(key_expr)?;

        if payload.is_empty() {
            return Err(BusError::FrameTooShort);
        }

        // In a real implementation, this would publish through the DDS DataWriter.
        Ok(())
    }

    fn receive<'a>(
        &mut self,
        buf: &'a mut [u8],
    ) -> Result<Option<BusSample<'a>>, BusError> {
        if let Some((key, data)) = self.rx_queue.pop_front() {
            let key_bytes = key.as_bytes();
            let total = key_bytes.len() + 1 + data.len();
            if buf.len() < total {
                return Err(BusError::BufferTooSmall);
            }
            buf[..key_bytes.len()].copy_from_slice(key_bytes);
            buf[key_bytes.len()] = 0;
            let payload_start = key_bytes.len() + 1;
            buf[payload_start..payload_start + data.len()].copy_from_slice(&data);

            Ok(Some(BusSample {
                key_expr: core::str::from_utf8(&buf[..key_bytes.len()])
                    .map_err(|_| BusError::InvalidHeader)?,
                payload: &buf[payload_start..payload_start + data.len()],
                timestamp_us: 0,
            }))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_prefix() {
        let adapter = DdsZenohAdapter::new(0).unwrap();
        assert_eq!(adapter.prefix(), "dds/0");
    }

    #[test]
    fn test_adapter_domain_42() {
        let adapter = DdsZenohAdapter::new(42).unwrap();
        assert_eq!(adapter.prefix(), "dds/42");
        assert_eq!(adapter.domain_id(), 42);
    }

    #[test]
    fn test_adapter_invalid_domain() {
        assert!(DdsZenohAdapter::new(233).is_err());
    }

    #[test]
    fn test_key_for_topic() {
        let adapter = DdsZenohAdapter::new(0).unwrap();
        assert_eq!(adapter.key_for_topic("SensorData"), "dds/0/SensorData");
        assert_eq!(
            adapter.key_for_topic("vehicle/speed"),
            "dds/0/vehicle/speed"
        );
    }

    // -- ROS 2 name mangling tests (21.19) --

    #[test]
    fn test_ros2_topic_to_dds() {
        assert_eq!(
            DdsZenohAdapter::ros2_topic_to_dds("/robot/cmd_vel", ROS2_TOPIC_PREFIX),
            "rt/robot/cmd_vel"
        );
        assert_eq!(
            DdsZenohAdapter::ros2_topic_to_dds("robot/cmd_vel", ROS2_TOPIC_PREFIX),
            "rt/robot/cmd_vel"
        );
    }

    #[test]
    fn test_ros2_service_request() {
        assert_eq!(
            DdsZenohAdapter::ros2_topic_to_dds("/robot/get_state", ROS2_REQUEST_PREFIX),
            "rq/robot/get_state"
        );
    }

    #[test]
    fn test_ros2_service_reply() {
        assert_eq!(
            DdsZenohAdapter::ros2_topic_to_dds("/robot/get_state", ROS2_REPLY_PREFIX),
            "rr/robot/get_state"
        );
    }

    #[test]
    fn test_key_for_ros2_topic() {
        let adapter = DdsZenohAdapter::new(0).unwrap();
        assert_eq!(
            adapter.key_for_ros2_topic("/robot/cmd_vel"),
            "dds/0/rt/robot/cmd_vel"
        );
    }

    #[test]
    fn test_key_for_ros2_request() {
        let adapter = DdsZenohAdapter::new(0).unwrap();
        assert_eq!(
            adapter.key_for_ros2_request("/robot/get_state"),
            "dds/0/rq/robot/get_state"
        );
    }

    #[test]
    fn test_key_for_ros2_reply() {
        let adapter = DdsZenohAdapter::new(0).unwrap();
        assert_eq!(
            adapter.key_for_ros2_reply("/robot/get_state"),
            "dds/0/rr/robot/get_state"
        );
    }

    #[test]
    fn test_is_ros2_key() {
        assert!(DdsZenohAdapter::is_ros2_key("dds/0/rt/robot/cmd_vel"));
        assert!(DdsZenohAdapter::is_ros2_key("dds/0/rq/robot/get_state"));
        assert!(DdsZenohAdapter::is_ros2_key("dds/0/rr/robot/get_state"));
        assert!(!DdsZenohAdapter::is_ros2_key("dds/0/SensorData"));
        assert!(!DdsZenohAdapter::is_ros2_key("can/0/0x100"));
    }

    // -- Core adapter tests --

    #[test]
    fn test_inject_and_receive() {
        let mut adapter = DdsZenohAdapter::new(0).unwrap();
        adapter.inject_rx("SensorData", &[0x01, 0x02, 0x03]);
        assert_eq!(adapter.rx_count(), 1);

        let mut buf = [0u8; 256];
        let sample = adapter.receive(&mut buf).unwrap().unwrap();
        assert_eq!(sample.key_expr, "dds/0/SensorData");
        assert_eq!(sample.payload, &[0x01, 0x02, 0x03]);
    }

    #[test]
    fn test_receive_empty() {
        let mut adapter = DdsZenohAdapter::new(0).unwrap();
        let mut buf = [0u8; 256];
        assert!(adapter.receive(&mut buf).unwrap().is_none());
    }

    #[test]
    fn test_transmit_valid() {
        let mut adapter = DdsZenohAdapter::new(0).unwrap();
        adapter
            .transmit("dds/0/SensorData", &[0x01, 0x02])
            .unwrap();
    }

    #[test]
    fn test_transmit_ros2_topic() {
        let mut adapter = DdsZenohAdapter::new(0).unwrap();
        adapter
            .transmit("dds/0/rt/robot/cmd_vel", &[0x01])
            .unwrap();
    }

    #[test]
    fn test_transmit_empty_payload() {
        let mut adapter = DdsZenohAdapter::new(0).unwrap();
        assert_eq!(
            adapter.transmit("dds/0/SensorData", &[]),
            Err(BusError::FrameTooShort)
        );
    }

    #[test]
    fn test_transmit_invalid_key() {
        let mut adapter = DdsZenohAdapter::new(0).unwrap();
        assert!(adapter.transmit("can/0/0x100", &[0x01]).is_err());
    }

    #[test]
    fn test_multiple_topics() {
        let mut adapter = DdsZenohAdapter::new(0).unwrap();
        adapter.inject_rx("SensorA", &[0x01]);
        adapter.inject_rx("SensorB", &[0x02]);
        adapter.inject_rx("rt/robot/cmd_vel", &[0x03]);
        assert_eq!(adapter.rx_count(), 3);

        let mut buf = [0u8; 256];
        let s1 = adapter.receive(&mut buf).unwrap().unwrap();
        assert_eq!(s1.key_expr, "dds/0/SensorA");

        let s2 = adapter.receive(&mut buf).unwrap().unwrap();
        assert_eq!(s2.key_expr, "dds/0/SensorB");

        let s3 = adapter.receive(&mut buf).unwrap().unwrap();
        assert_eq!(s3.key_expr, "dds/0/rt/robot/cmd_vel");
    }
}
