// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration test: capability-gated access enforcement.

use smallaios_security::capability::{CapRegistry, Permissions, ResourceRef, ResourceType};

#[test]
fn test_capability_create_and_check() {
    let mut registry = CapRegistry::new();

    let task_id = 1;
    let resource = ResourceRef::new(ResourceType::TensorBuffer, 0);
    let perms = Permissions::READ.union(Permissions::WRITE);

    let _cap_id = registry.create(task_id, resource, perms, 0).unwrap();
    assert!(registry
        .check(task_id, &resource, Permissions::READ, 0)
        .is_ok());
    assert!(registry
        .check(task_id, &resource, Permissions::WRITE, 0)
        .is_ok());

    // Task without capability gets denied
    let other_task = 2;
    assert!(registry
        .check(other_task, &resource, Permissions::READ, 0)
        .is_err());
}

#[test]
fn test_capability_revocation() {
    let mut registry = CapRegistry::new();

    let task_id = 1;
    let resource = ResourceRef::new(ResourceType::TensorBuffer, 0);
    let perms = Permissions::READ;

    let cap_id = registry.create(task_id, resource, perms, 0).unwrap();
    assert!(registry
        .check(task_id, &resource, Permissions::READ, 0)
        .is_ok());

    registry.revoke(cap_id).unwrap();
    assert!(registry
        .check(task_id, &resource, Permissions::READ, 0)
        .is_err());
}

#[test]
fn test_capability_expiration() {
    let mut registry = CapRegistry::new();

    let task_id = 1;
    let resource = ResourceRef::new(ResourceType::NetworkSocket, 0);

    let _cap = registry
        .create(task_id, resource, Permissions::READ, 100)
        .unwrap();

    // Valid before expiry
    assert!(registry
        .check(task_id, &resource, Permissions::READ, 50)
        .is_ok());
}

#[test]
fn test_capability_insufficient_permissions() {
    let mut registry = CapRegistry::new();

    let task_id = 1;
    let resource = ResourceRef::new(ResourceType::IpcEndpoint, 0);

    // Grant READ only
    let _cap = registry
        .create(task_id, resource, Permissions::READ, 0)
        .unwrap();

    // READ succeeds
    assert!(registry
        .check(task_id, &resource, Permissions::READ, 0)
        .is_ok());
    // WRITE denied
    assert!(registry
        .check(task_id, &resource, Permissions::WRITE, 0)
        .is_err());
}
