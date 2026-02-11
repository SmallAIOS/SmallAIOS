// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Capability syscalls (0x60-0x6F).
//!
//! - `cap_create(resource_type, instance_id, permissions, expires) -> CapId`
//! - `cap_revoke(cap_id)`
//! - `cap_delegate(cap_id, to_task, permissions, expires) -> CapId`
//! - `cap_check(cap_id, resource_type, instance_id, permissions) -> 0 | error`
//! - `cap_list(task_id, buf_ptr, buf_len) -> count`
//!
//! Wired to the CapRegistry in kernel state. The current task is used as
//! the owner for create operations. Delegation and revocation go through
//! the registry's cascading logic.

use super::{SyscallArgs, SyscallError, SyscallResult};
use crate::state;
use crate::syscall::task::current_task_id;
use smallaios_security::capability::{Permissions, ResourceRef, ResourceType, MAX_CAPS_PER_TASK};

/// Create a new capability (root grant).
///
/// # Arguments
/// - args[0]: resource_type (u8 as usize)
/// - args[1]: instance_id (u64)
/// - args[2]: permissions (u32 bitmask)
/// - args[3]: expires (u64 timestamp, 0 = no expiry)
///
/// # Returns
/// Capability ID on success, negative error on failure.
pub fn sys_cap_create(args: &SyscallArgs) -> SyscallResult {
    let resource_type_raw = args.args[0] as u8;
    let instance_id = args.args[1] as u64;
    let permissions_raw = args.args[2] as u32;
    let expires = args.args[3] as u64;

    // Validate resource type (0-7 valid)
    let resource_type = match ResourceType::from_u8(resource_type_raw) {
        Some(rt) => rt,
        None => return SyscallError::InvalidArgument.as_i64(),
    };

    // Validate permissions (only lower 4 bits)
    if permissions_raw & !0b1111 != 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // Must grant at least one permission
    if permissions_raw == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    let permissions = Permissions::from_bits(permissions_raw);
    let resource = ResourceRef::new(resource_type, instance_id);
    let owner = current_task_id();

    // SAFETY: Syscall handlers run with interrupts masked.
    let result = unsafe {
        state::with_cap_registry(|registry| registry.create(owner, resource, permissions, expires))
    };

    match result {
        Ok(cap_id) => cap_id as i64,
        Err(err) => state::cap_error_to_syscall(err),
    }
}

/// Revoke a capability and all delegated children.
///
/// # Arguments
/// - args[0]: cap_id (u64)
///
/// # Returns
/// 0 on success, negative error on failure.
pub fn sys_cap_revoke(args: &SyscallArgs) -> SyscallResult {
    let cap_id = args.args[0] as u64;

    if cap_id == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }

    // SAFETY: Syscall handlers run with interrupts masked.
    let result = unsafe { state::with_cap_registry(|registry| registry.revoke(cap_id)) };

    match result {
        Ok(()) => SyscallError::Success.as_i64(),
        Err(err) => state::cap_error_to_syscall(err),
    }
}

/// Delegate a capability to another task with reduced permissions.
///
/// # Arguments
/// - args[0]: cap_id (u64) — capability to delegate from
/// - args[1]: to_task (u64) — target task ID
/// - args[2]: permissions (u32 bitmask) — must be subset of parent
/// - args[3]: expires (u64 timestamp, 0 = no expiry)
///
/// # Returns
/// New capability ID on success, negative error on failure.
pub fn sys_cap_delegate(args: &SyscallArgs) -> SyscallResult {
    let cap_id = args.args[0] as u64;
    let to_task = args.args[1] as u64;
    let permissions_raw = args.args[2] as u32;
    let expires = args.args[3] as u64;

    if cap_id == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }

    if to_task == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // Validate permissions
    if permissions_raw & !0b1111 != 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    if permissions_raw == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    let permissions = Permissions::from_bits(permissions_raw);

    // SAFETY: Syscall handlers run with interrupts masked.
    // now=0 means no expiry checking against current time.
    let result = unsafe {
        state::with_cap_registry(|registry| {
            registry.delegate(cap_id, to_task, permissions, expires, 0)
        })
    };

    match result {
        Ok(new_cap_id) => new_cap_id as i64,
        Err(err) => state::cap_error_to_syscall(err),
    }
}

/// Check whether a capability grants the requested permissions.
///
/// # Arguments
/// - args[0]: cap_id (u64)
/// - args[1]: resource_type (u8 as usize)
/// - args[2]: instance_id (u64)
/// - args[3]: permissions (u32 bitmask)
///
/// # Returns
/// 0 if the check passes, negative error if it fails.
pub fn sys_cap_check(args: &SyscallArgs) -> SyscallResult {
    let cap_id = args.args[0] as u64;
    let resource_type_raw = args.args[1] as u8;
    let instance_id = args.args[2] as u64;
    let permissions_raw = args.args[3] as u32;

    if cap_id == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }

    let resource_type = match ResourceType::from_u8(resource_type_raw) {
        Some(rt) => rt,
        None => return SyscallError::InvalidArgument.as_i64(),
    };

    if permissions_raw & !0b1111 != 0 || permissions_raw == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    let permissions = Permissions::from_bits(permissions_raw);
    let resource = ResourceRef::new(resource_type, instance_id);

    // Look up the capability and check it
    let result = state::with_cap_registry_ref(|registry| {
        let cap = registry.lookup(cap_id, 0)?;
        cap.check(&resource, permissions, 0)
    });

    match result {
        Ok(()) => SyscallError::Success.as_i64(),
        Err(err) => state::cap_error_to_syscall(err),
    }
}

/// List capabilities held by a task.
///
/// # Arguments
/// - args[0]: task_id (u64, 0 = current task)
/// - args[1]: buf_ptr (pointer to CapId array)
/// - args[2]: buf_len (number of entries in buffer)
///
/// # Returns
/// Number of capabilities on success, negative error on failure.
pub fn sys_cap_list(args: &SyscallArgs) -> SyscallResult {
    let task_id_raw = args.args[0] as u64;
    let buf_ptr = args.args[1];
    let buf_len = args.args[2];

    // Validate buffer pointer (0 is acceptable if buf_len is 0 — count-only mode)
    if buf_ptr == 0 && buf_len > 0 {
        return SyscallError::BadAddress.as_i64();
    }

    let task_id = if task_id_raw == 0 {
        current_task_id()
    } else {
        task_id_raw
    };

    // Clamp buf_len to MAX_CAPS_PER_TASK to prevent absurd slice creation
    let clamped_len = if buf_len > MAX_CAPS_PER_TASK {
        MAX_CAPS_PER_TASK
    } else {
        buf_len
    };

    // Count caps first (always safe, no pointer dereference)
    let total = state::with_cap_registry_ref(|registry| registry.count_for_task(task_id));

    if buf_ptr == 0 || clamped_len == 0 || total == 0 {
        // Count-only mode, or nothing to write
        total as i64
    } else {
        // Fill buffer mode
        // SAFETY: In unikernel mode, the caller is in the same address space.
        // We only reach here when total > 0 and the pointer is non-null.
        let write_len = if clamped_len < total {
            clamped_len
        } else {
            total
        };
        let buf = unsafe { core::slice::from_raw_parts_mut(buf_ptr as *mut u64, write_len) };
        let count = state::with_cap_registry_ref(|registry| registry.list_for_task(task_id, buf));
        count as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syscall::SyscallArgs;

    #[test]
    fn test_cap_create_valid() {
        // resource_type=0 (TensorBuffer), instance=42, perms=READ(0b0001), no expiry
        let args = SyscallArgs::new(0x60, [0, 42, 0b0001, 0, 0, 0]);
        let result = sys_cap_create(&args);
        assert!(result > 0, "cap_create should return a positive cap ID");
    }

    #[test]
    fn test_cap_create_invalid_resource_type() {
        let args = SyscallArgs::new(0x60, [99, 1, 0b0001, 0, 0, 0]);
        assert_eq!(
            sys_cap_create(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_cap_create_invalid_permissions() {
        // Permissions with high bits set
        let args = SyscallArgs::new(0x60, [0, 1, 0xFF, 0, 0, 0]);
        assert_eq!(
            sys_cap_create(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_cap_create_zero_permissions() {
        let args = SyscallArgs::new(0x60, [0, 1, 0, 0, 0, 0]);
        assert_eq!(
            sys_cap_create(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_cap_revoke_zero_id() {
        let args = SyscallArgs::new(0x61, [0, 0, 0, 0, 0, 0]);
        assert_eq!(sys_cap_revoke(&args), SyscallError::InvalidHandle.as_i64());
    }

    #[test]
    fn test_cap_revoke_nonexistent() {
        // ID 999999 doesn't exist in registry
        let args = SyscallArgs::new(0x61, [999999, 0, 0, 0, 0, 0]);
        let result = sys_cap_revoke(&args);
        // Should return NotFound mapped to SyscallError
        assert!(result < 0);
    }

    #[test]
    fn test_cap_delegate_zero_cap() {
        let args = SyscallArgs::new(0x62, [0, 2, 0b0001, 0, 0, 0]);
        assert_eq!(
            sys_cap_delegate(&args),
            SyscallError::InvalidHandle.as_i64()
        );
    }

    #[test]
    fn test_cap_delegate_zero_task() {
        let args = SyscallArgs::new(0x62, [1, 0, 0b0001, 0, 0, 0]);
        assert_eq!(
            sys_cap_delegate(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_cap_delegate_zero_perms() {
        let args = SyscallArgs::new(0x62, [1, 2, 0, 0, 0, 0]);
        assert_eq!(
            sys_cap_delegate(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_cap_check_zero_cap() {
        let args = SyscallArgs::new(0x63, [0, 0, 1, 0b0001, 0, 0]);
        assert_eq!(sys_cap_check(&args), SyscallError::InvalidHandle.as_i64());
    }

    #[test]
    fn test_cap_check_invalid_resource() {
        let args = SyscallArgs::new(0x63, [1, 99, 1, 0b0001, 0, 0]);
        assert_eq!(sys_cap_check(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_cap_check_zero_perms() {
        let args = SyscallArgs::new(0x63, [1, 0, 1, 0, 0, 0]);
        assert_eq!(sys_cap_check(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_cap_list_count_mode() {
        // task_id=1, buf_ptr=0, buf_len=0 — count-only mode
        let args = SyscallArgs::new(0x64, [1, 0, 0, 0, 0, 0]);
        let count = sys_cap_list(&args);
        assert!(count >= 0, "cap_list count-only should return >= 0");
    }

    #[test]
    fn test_cap_list_null_buf_nonzero_len() {
        let args = SyscallArgs::new(0x64, [1, 0, 10, 0, 0, 0]);
        assert_eq!(sys_cap_list(&args), SyscallError::BadAddress.as_i64());
    }
}
