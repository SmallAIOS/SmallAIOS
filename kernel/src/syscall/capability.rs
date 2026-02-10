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
//! These are stubs that validate arguments and return placeholder results.
//! Full integration with the capability registry happens when the security
//! crate is wired into the kernel.

use super::{SyscallArgs, SyscallError, SyscallResult};

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
    let resource_type = args.args[0] as u8;
    let _instance_id = args.args[1] as u64;
    let permissions = args.args[2] as u32;

    // Validate resource type (0-7 valid)
    if resource_type > 7 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // Validate permissions (only lower 4 bits)
    if permissions & !0b1111 != 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // Must grant at least one permission
    if permissions == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // Stub: return a placeholder capability ID
    1i64
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

    // Stub: success
    SyscallError::Success.as_i64()
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
    let permissions = args.args[2] as u32;

    if cap_id == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }

    if to_task == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // Validate permissions
    if permissions & !0b1111 != 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    if permissions == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // Stub: return a placeholder delegated capability ID
    2i64
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
    let resource_type = args.args[1] as u8;
    let _instance_id = args.args[2] as u64;
    let permissions = args.args[3] as u32;

    if cap_id == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }

    if resource_type > 7 {
        return SyscallError::InvalidArgument.as_i64();
    }

    if permissions & !0b1111 != 0 || permissions == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // Stub: success (in production, this checks the registry)
    SyscallError::Success.as_i64()
}

/// List capabilities held by a task.
///
/// # Arguments
/// - args[0]: task_id (u64, 0 = current task)
/// - args[1]: buf_ptr (pointer to CapId array)
/// - args[2]: buf_len (number of entries in buffer)
///
/// # Returns
/// Number of capabilities written on success, negative error on failure.
pub fn sys_cap_list(args: &SyscallArgs) -> SyscallResult {
    let _task_id = args.args[0] as u64;
    let buf_ptr = args.args[1];
    let buf_len = args.args[2];

    // Validate buffer pointer (0 is acceptable if buf_len is 0 — count-only mode)
    if buf_ptr == 0 && buf_len > 0 {
        return SyscallError::BadAddress.as_i64();
    }

    // Stub: no capabilities to list
    0i64
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
    fn test_cap_revoke_valid() {
        let args = SyscallArgs::new(0x61, [1, 0, 0, 0, 0, 0]);
        assert_eq!(sys_cap_revoke(&args), SyscallError::Success.as_i64());
    }

    #[test]
    fn test_cap_revoke_zero_id() {
        let args = SyscallArgs::new(0x61, [0, 0, 0, 0, 0, 0]);
        assert_eq!(sys_cap_revoke(&args), SyscallError::InvalidHandle.as_i64());
    }

    #[test]
    fn test_cap_delegate_valid() {
        // delegate cap 1 to task 2, READ permission
        let args = SyscallArgs::new(0x62, [1, 2, 0b0001, 0, 0, 0]);
        let result = sys_cap_delegate(&args);
        assert!(result > 0, "delegate should return a positive cap ID");
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
    fn test_cap_check_valid() {
        // cap 1, resource_type=0, instance=42, perms=READ
        let args = SyscallArgs::new(0x63, [1, 0, 42, 0b0001, 0, 0]);
        assert_eq!(sys_cap_check(&args), SyscallError::Success.as_i64());
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

    #[test]
    fn test_cap_list_with_buffer() {
        // Non-null buffer pointer with length
        let args = SyscallArgs::new(0x64, [1, 0x1000, 10, 0, 0, 0]);
        let count = sys_cap_list(&args);
        assert_eq!(count, 0, "stub should return 0 capabilities");
    }
}
