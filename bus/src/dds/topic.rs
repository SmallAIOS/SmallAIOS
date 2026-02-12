// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! DDS Topic type system.
//!
//! A Topic defines a named data type within a domain. Type consistency is
//! enforced: DataWriters and DataReaders on the same Topic name must agree
//! on the type name.

use crate::dds::types::DomainId;
use crate::BusError;
use alloc::string::String;

/// Maximum topic name length.
pub const MAX_TOPIC_NAME_LEN: usize = 256;

/// Maximum type name length.
pub const MAX_TYPE_NAME_LEN: usize = 256;

/// DDS Topic definition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topic {
    /// Domain ID this topic belongs to.
    domain_id: DomainId,
    /// Topic name.
    name: String,
    /// Type name.
    type_name: String,
}

impl Topic {
    /// Create a new Topic.
    pub fn new(domain_id: DomainId, name: &str, type_name: &str) -> Result<Self, BusError> {
        if name.is_empty() || name.len() > MAX_TOPIC_NAME_LEN {
            return Err(BusError::OutOfRange);
        }
        if type_name.is_empty() || type_name.len() > MAX_TYPE_NAME_LEN {
            return Err(BusError::OutOfRange);
        }
        Ok(Self {
            domain_id,
            name: String::from(name),
            type_name: String::from(type_name),
        })
    }

    /// Returns the domain ID.
    pub fn domain_id(&self) -> DomainId {
        self.domain_id
    }

    /// Returns the topic name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the type name.
    pub fn type_name(&self) -> &str {
        &self.type_name
    }

    /// Check type consistency with another topic.
    ///
    /// Two topics are type-consistent if they have the same name and type_name
    /// within the same domain.
    pub fn is_type_consistent(&self, other: &Topic) -> bool {
        self.domain_id == other.domain_id
            && self.name == other.name
            && self.type_name == other.type_name
    }

    /// Check if another topic has the same name but different type (mismatch).
    pub fn is_type_mismatch(&self, other: &Topic) -> bool {
        self.domain_id == other.domain_id
            && self.name == other.name
            && self.type_name != other.type_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topic_create() {
        let topic = Topic::new(0, "SensorData", "SensorMsg").unwrap();
        assert_eq!(topic.domain_id(), 0);
        assert_eq!(topic.name(), "SensorData");
        assert_eq!(topic.type_name(), "SensorMsg");
    }

    #[test]
    fn test_topic_empty_name() {
        assert_eq!(
            Topic::new(0, "", "SensorMsg").err(),
            Some(BusError::OutOfRange)
        );
    }

    #[test]
    fn test_topic_empty_type() {
        assert_eq!(
            Topic::new(0, "SensorData", "").err(),
            Some(BusError::OutOfRange)
        );
    }

    #[test]
    fn test_topic_type_consistent() {
        let t1 = Topic::new(0, "SensorData", "SensorMsg").unwrap();
        let t2 = Topic::new(0, "SensorData", "SensorMsg").unwrap();
        assert!(t1.is_type_consistent(&t2));
    }

    #[test]
    fn test_topic_type_mismatch() {
        let t1 = Topic::new(0, "SensorData", "SensorMsg").unwrap();
        let t2 = Topic::new(0, "SensorData", "OtherMsg").unwrap();
        assert!(t1.is_type_mismatch(&t2));
        assert!(!t1.is_type_consistent(&t2));
    }

    #[test]
    fn test_topic_different_names_not_mismatch() {
        let t1 = Topic::new(0, "SensorData", "SensorMsg").unwrap();
        let t2 = Topic::new(0, "OtherTopic", "OtherMsg").unwrap();
        assert!(!t1.is_type_mismatch(&t2));
        assert!(!t1.is_type_consistent(&t2));
    }

    #[test]
    fn test_topic_different_domains() {
        let t1 = Topic::new(0, "SensorData", "SensorMsg").unwrap();
        let t2 = Topic::new(1, "SensorData", "SensorMsg").unwrap();
        assert!(!t1.is_type_consistent(&t2));
    }

    #[test]
    fn test_topic_equality() {
        let t1 = Topic::new(0, "SensorData", "SensorMsg").unwrap();
        let t2 = Topic::new(0, "SensorData", "SensorMsg").unwrap();
        assert_eq!(t1, t2);
    }
}
