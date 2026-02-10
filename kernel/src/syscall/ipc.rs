// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! IPC syscalls (0x20-0x2F).
//!
//! Zenoh-inspired pub/sub and request/reply messaging:
//! - `ipc_publish(key, data, len)`
//! - `ipc_subscribe(key_expr) -> SubHandle`
//! - `ipc_recv(handle, buf, len) -> usize`
//! - `ipc_query(key, data, len, reply_buf, reply_len) -> usize`
//! - `ipc_channel_create() -> (SendHandle, RecvHandle)`
//! - `ipc_channel_send(handle, data, len)`
//! - `ipc_channel_recv(handle, buf, len) -> usize`
//! - `ipc_channel_close(handle)`

use super::{SyscallArgs, SyscallError, SyscallResult};

/// Maximum IPC message size: 64 MiB.
pub const MAX_IPC_MESSAGE_SIZE: usize = 64 * 1024 * 1024;

/// Maximum key expression length: 256 bytes.
pub const MAX_KEY_EXPR_LEN: usize = 256;

/// Publish a message to a key expression.
///
/// Args: [key_ptr, key_len, data_ptr, data_len, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_ipc_publish(args: &SyscallArgs) -> SyscallResult {
    let key_ptr = args.args[0];
    let key_len = args.args[1];
    let data_ptr = args.args[2];
    let data_len = args.args[3];

    if key_ptr == 0 || key_len == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if key_len > MAX_KEY_EXPR_LEN {
        return SyscallError::InvalidArgument.as_i64();
    }
    if data_len > MAX_IPC_MESSAGE_SIZE {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with IPC router
    let _ = (key_ptr, key_len, data_ptr, data_len);
    SyscallError::NotSupported.as_i64()
}

/// Subscribe to a key expression.
///
/// Args: [key_expr_ptr, key_expr_len, 0, 0, 0, 0]
/// Returns: subscription handle on success, negative error code on failure.
pub fn sys_ipc_subscribe(args: &SyscallArgs) -> SyscallResult {
    let key_ptr = args.args[0];
    let key_len = args.args[1];

    if key_ptr == 0 || key_len == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if key_len > MAX_KEY_EXPR_LEN {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with IPC router
    let _ = (key_ptr, key_len);
    SyscallError::NotSupported.as_i64()
}

/// Receive a message from a subscription.
///
/// Args: [handle, buf_ptr, buf_len, 0, 0, 0]
/// Returns: bytes received on success, negative error code on failure.
pub fn sys_ipc_recv(args: &SyscallArgs) -> SyscallResult {
    let handle = args.args[0];
    let buf_ptr = args.args[1];
    let buf_len = args.args[2];

    if handle == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }
    if buf_ptr == 0 || buf_len == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with IPC router
    let _ = (handle, buf_ptr, buf_len);
    SyscallError::NotSupported.as_i64()
}

/// Query a key expression (request/reply pattern).
///
/// Args: [key_ptr, key_len, data_ptr, data_len, reply_buf_ptr, reply_buf_len]
/// Returns: reply bytes on success, negative error code on failure.
pub fn sys_ipc_query(args: &SyscallArgs) -> SyscallResult {
    let key_ptr = args.args[0];
    let key_len = args.args[1];
    let data_ptr = args.args[2];
    let data_len = args.args[3];
    let reply_ptr = args.args[4];
    let reply_len = args.args[5];

    if key_ptr == 0 || key_len == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if key_len > MAX_KEY_EXPR_LEN {
        return SyscallError::InvalidArgument.as_i64();
    }
    if data_len > MAX_IPC_MESSAGE_SIZE {
        return SyscallError::InvalidArgument.as_i64();
    }
    if reply_ptr == 0 || reply_len == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with IPC router
    let _ = (key_ptr, key_len, data_ptr, data_len, reply_ptr, reply_len);
    SyscallError::NotSupported.as_i64()
}

/// Create a point-to-point channel.
///
/// Args: [send_handle_ptr, recv_handle_ptr, 0, 0, 0, 0]
/// Returns: 0 on success (handles written to pointers), negative error code on failure.
pub fn sys_ipc_channel_create(args: &SyscallArgs) -> SyscallResult {
    let send_handle_ptr = args.args[0];
    let recv_handle_ptr = args.args[1];

    if send_handle_ptr == 0 || recv_handle_ptr == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with IPC channel subsystem
    let _ = (send_handle_ptr, recv_handle_ptr);
    SyscallError::NotSupported.as_i64()
}

/// Send data on a channel.
///
/// Args: [handle, data_ptr, data_len, 0, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_ipc_channel_send(args: &SyscallArgs) -> SyscallResult {
    let handle = args.args[0];
    let data_ptr = args.args[1];
    let data_len = args.args[2];

    if handle == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }
    if data_len > MAX_IPC_MESSAGE_SIZE {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with IPC channel subsystem
    let _ = (handle, data_ptr, data_len);
    SyscallError::NotSupported.as_i64()
}

/// Receive data from a channel.
///
/// Args: [handle, buf_ptr, buf_len, 0, 0, 0]
/// Returns: bytes received on success, negative error code on failure.
pub fn sys_ipc_channel_recv(args: &SyscallArgs) -> SyscallResult {
    let handle = args.args[0];
    let buf_ptr = args.args[1];
    let buf_len = args.args[2];

    if handle == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }
    if buf_ptr == 0 || buf_len == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with IPC channel subsystem
    let _ = (handle, buf_ptr, buf_len);
    SyscallError::NotSupported.as_i64()
}

/// Close a channel handle.
///
/// Args: [handle, 0, 0, 0, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_ipc_channel_close(args: &SyscallArgs) -> SyscallResult {
    let handle = args.args[0];

    if handle == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }

    // TODO: Integrate with IPC channel subsystem
    let _ = handle;
    SyscallError::NotSupported.as_i64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_publish_null_key() {
        let args = SyscallArgs::new(0x20, [0, 5, 0x1000, 100, 0, 0]);
        assert_eq!(
            sys_ipc_publish(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_publish_zero_key_len() {
        let args = SyscallArgs::new(0x20, [0x1000, 0, 0x2000, 100, 0, 0]);
        assert_eq!(
            sys_ipc_publish(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_publish_key_too_long() {
        let args = SyscallArgs::new(0x20, [0x1000, MAX_KEY_EXPR_LEN + 1, 0x2000, 100, 0, 0]);
        assert_eq!(
            sys_ipc_publish(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_publish_data_too_large() {
        let args = SyscallArgs::new(0x20, [0x1000, 10, 0x2000, MAX_IPC_MESSAGE_SIZE + 1, 0, 0]);
        assert_eq!(
            sys_ipc_publish(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_subscribe_null_key() {
        let args = SyscallArgs::new(0x21, [0, 10, 0, 0, 0, 0]);
        assert_eq!(
            sys_ipc_subscribe(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_recv_zero_handle() {
        let args = SyscallArgs::new(0x22, [0, 0x1000, 256, 0, 0, 0]);
        assert_eq!(sys_ipc_recv(&args), SyscallError::InvalidHandle.as_i64());
    }

    #[test]
    fn test_recv_null_buffer() {
        let args = SyscallArgs::new(0x22, [1, 0, 256, 0, 0, 0]);
        assert_eq!(
            sys_ipc_recv(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_query_null_key() {
        let args = SyscallArgs::new(0x23, [0, 10, 0x1000, 100, 0x2000, 256]);
        assert_eq!(sys_ipc_query(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_query_null_reply_buf() {
        let args = SyscallArgs::new(0x23, [0x1000, 10, 0x2000, 100, 0, 256]);
        assert_eq!(sys_ipc_query(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_channel_create_null_ptrs() {
        let args = SyscallArgs::new(0x24, [0, 0x1000, 0, 0, 0, 0]);
        assert_eq!(
            sys_ipc_channel_create(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_channel_send_zero_handle() {
        let args = SyscallArgs::new(0x25, [0, 0x1000, 100, 0, 0, 0]);
        assert_eq!(
            sys_ipc_channel_send(&args),
            SyscallError::InvalidHandle.as_i64()
        );
    }

    #[test]
    fn test_channel_send_oversized() {
        let args = SyscallArgs::new(0x25, [1, 0x1000, MAX_IPC_MESSAGE_SIZE + 1, 0, 0, 0]);
        assert_eq!(
            sys_ipc_channel_send(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_channel_recv_zero_handle() {
        let args = SyscallArgs::new(0x26, [0, 0x1000, 256, 0, 0, 0]);
        assert_eq!(
            sys_ipc_channel_recv(&args),
            SyscallError::InvalidHandle.as_i64()
        );
    }

    #[test]
    fn test_channel_close_zero_handle() {
        let args = SyscallArgs::new(0x27, [0, 0, 0, 0, 0, 0]);
        assert_eq!(
            sys_ipc_channel_close(&args),
            SyscallError::InvalidHandle.as_i64()
        );
    }
}
