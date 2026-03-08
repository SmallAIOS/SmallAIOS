// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration test: container boot sequence.

use smallaios_container::boot::BootSequence;
use smallaios_container::config::ContainerConfig;

#[test]
fn test_boot_sequence_creation() {
    let config = ContainerConfig::default();
    let seq = BootSequence::new(config);
    assert!(!seq.is_ready());
}

#[test]
fn test_boot_sequence_advance() {
    let config = ContainerConfig::default();
    let mut seq = BootSequence::new(config);
    let status = seq.advance().unwrap();
    assert!(!status.message.is_empty());
}

#[test]
fn test_boot_sequence_run_all() {
    let config = ContainerConfig::default();
    let mut seq = BootSequence::new(config);
    let results = seq.run_all().unwrap();
    assert!(!results.is_empty());
    assert!(seq.is_ready());
}

#[test]
fn test_config_validate() {
    let config = ContainerConfig::default();
    assert!(config.validate().is_ok());
}

#[test]
fn test_config_with_gpu() {
    let mut config = ContainerConfig::default();
    config.with_gpu();
    assert!(config.validate().is_ok());
}
