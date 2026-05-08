// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ONNX syscalls (0x30-0x3F).
//!
//! - `onnx_load(data, len) -> ModelHandle`
//! - `onnx_unload(handle)`
//! - `onnx_create_session(model, opts) -> SessionHandle`
//! - `onnx_run(session, inputs, num_inputs, outputs, num_outputs)`
//! - `onnx_get_metadata(model, buf, len) -> usize`
//! - `onnx_list_providers() -> ProviderList`

use super::{SyscallArgs, SyscallError, SyscallResult};

/// Maximum ONNX model size: 2 GiB.
pub const MAX_MODEL_SIZE: usize = 2 * 1024 * 1024 * 1024;

/// Maximum number of inputs/outputs per inference call.
pub const MAX_IO_TENSORS: usize = 64;

/// Load an ONNX model from memory.
///
/// Args: [data_ptr, data_len, 0, 0, 0, 0]
/// Returns: model handle on success, negative error code on failure.
pub fn sys_onnx_load(args: &SyscallArgs) -> SyscallResult {
    let data_ptr = args.args[0];
    let data_len = args.args[1];

    if data_ptr == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if data_len == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if data_len > MAX_MODEL_SIZE {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with ONNX runtime — parse protobuf, validate model
    let _ = (data_ptr, data_len);
    SyscallError::NotSupported.as_i64()
}

/// Unload a previously loaded ONNX model.
///
/// Args: [handle, 0, 0, 0, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_onnx_unload(args: &SyscallArgs) -> SyscallResult {
    let handle = args.args[0];

    if handle == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }

    // TODO: Integrate with ONNX runtime — free model resources
    let _ = handle;
    SyscallError::NotSupported.as_i64()
}

/// Create an inference session for a loaded model.
///
/// Args: [model_handle, opts_ptr, 0, 0, 0, 0]
/// Returns: session handle on success, negative error code on failure.
pub fn sys_onnx_create_session(args: &SyscallArgs) -> SyscallResult {
    let model_handle = args.args[0];

    if model_handle == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }

    // opts_ptr may be 0 (use defaults)
    let _ = model_handle;
    SyscallError::NotSupported.as_i64()
}

/// Run inference on a session.
///
/// Args: [session_handle, inputs_ptr, num_inputs, outputs_ptr, num_outputs, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_onnx_run(args: &SyscallArgs) -> SyscallResult {
    let session = args.args[0];
    let inputs_ptr = args.args[1];
    let num_inputs = args.args[2];
    let outputs_ptr = args.args[3];
    let num_outputs = args.args[4];

    if session == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }
    if inputs_ptr == 0 || num_inputs == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if outputs_ptr == 0 || num_outputs == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if num_inputs > MAX_IO_TENSORS || num_outputs > MAX_IO_TENSORS {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with ONNX runtime — execute graph, manage tensor I/O
    let _ = (session, inputs_ptr, num_inputs, outputs_ptr, num_outputs);
    SyscallError::NotSupported.as_i64()
}

/// Get metadata for a loaded model.
///
/// Args: [model_handle, buf_ptr, buf_len, 0, 0, 0]
/// Returns: bytes written on success, negative error code on failure.
pub fn sys_onnx_get_metadata(args: &SyscallArgs) -> SyscallResult {
    let model_handle = args.args[0];
    let buf_ptr = args.args[1];
    let buf_len = args.args[2];

    if model_handle == 0 {
        return SyscallError::InvalidHandle.as_i64();
    }
    if buf_ptr == 0 || buf_len == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with ONNX runtime — serialize model metadata
    let _ = (model_handle, buf_ptr, buf_len);
    SyscallError::NotSupported.as_i64()
}

/// List available execution providers.
///
/// Args: [buf_ptr, buf_len, 0, 0, 0, 0]
/// Returns: bytes written on success (provider list), negative error code on failure.
pub fn sys_onnx_list_providers(args: &SyscallArgs) -> SyscallResult {
    let buf_ptr = args.args[0];
    let buf_len = args.args[1];

    // Allow zero buf_ptr to query required size
    if buf_ptr == 0 && buf_len == 0 {
        // Return required buffer size
        // TODO: Compute actual required size
        return SyscallError::NotSupported.as_i64();
    }

    if buf_ptr == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with ONNX runtime — enumerate providers
    let _ = (buf_ptr, buf_len);
    SyscallError::NotSupported.as_i64()
}

/// `onnx_model_add(name_ptr, name_len, blob_ptr, blob_len, sig_ptr, sig_len)` -> 0 | -errno
///
/// Wave-0 stub. The real handler ships in `embedded-overlay-v1` Phase 3
/// (operator-uploads, writes to the F2FS overlay upper layer). Until
/// then this returns `-ENOSYS`. Routed here so the dispatch table has a
/// stable entry point at 0x36; the overlay agent will swap in the real
/// handler without touching the dispatcher.
pub fn sys_onnx_model_add(_args: &SyscallArgs) -> SyscallResult {
    -38 // -ENOSYS
}

/// `onnx_model_remove(name_ptr, name_len)` -> 0 | -errno
///
/// Wave-0 stub. Root-only model removal; the overlay agent's Phase 3
/// fills the body. Returns `-ENOSYS` until then.
pub fn sys_onnx_model_remove(_args: &SyscallArgs) -> SyscallResult {
    -38 // -ENOSYS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_onnx_load_null_data() {
        let args = SyscallArgs::new(0x30, [0, 1024, 0, 0, 0, 0]);
        assert_eq!(sys_onnx_load(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_onnx_load_zero_len() {
        let args = SyscallArgs::new(0x30, [0x1000, 0, 0, 0, 0, 0]);
        assert_eq!(sys_onnx_load(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_onnx_load_too_large() {
        let args = SyscallArgs::new(0x30, [0x1000, MAX_MODEL_SIZE + 1, 0, 0, 0, 0]);
        assert_eq!(sys_onnx_load(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_onnx_unload_zero_handle() {
        let args = SyscallArgs::new(0x31, [0, 0, 0, 0, 0, 0]);
        assert_eq!(sys_onnx_unload(&args), SyscallError::InvalidHandle.as_i64());
    }

    #[test]
    fn test_onnx_create_session_zero_handle() {
        let args = SyscallArgs::new(0x32, [0, 0, 0, 0, 0, 0]);
        assert_eq!(
            sys_onnx_create_session(&args),
            SyscallError::InvalidHandle.as_i64()
        );
    }

    #[test]
    fn test_onnx_run_zero_session() {
        let args = SyscallArgs::new(0x33, [0, 0x1000, 1, 0x2000, 1, 0]);
        assert_eq!(sys_onnx_run(&args), SyscallError::InvalidHandle.as_i64());
    }

    #[test]
    fn test_onnx_run_zero_inputs() {
        let args = SyscallArgs::new(0x33, [1, 0, 0, 0x2000, 1, 0]);
        assert_eq!(sys_onnx_run(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_onnx_run_zero_outputs() {
        let args = SyscallArgs::new(0x33, [1, 0x1000, 1, 0, 0, 0]);
        assert_eq!(sys_onnx_run(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_onnx_run_too_many_inputs() {
        let args = SyscallArgs::new(0x33, [1, 0x1000, MAX_IO_TENSORS + 1, 0x2000, 1, 0]);
        assert_eq!(sys_onnx_run(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_onnx_get_metadata_zero_handle() {
        let args = SyscallArgs::new(0x34, [0, 0x1000, 256, 0, 0, 0]);
        assert_eq!(
            sys_onnx_get_metadata(&args),
            SyscallError::InvalidHandle.as_i64()
        );
    }

    #[test]
    fn test_onnx_get_metadata_null_buf() {
        let args = SyscallArgs::new(0x34, [1, 0, 256, 0, 0, 0]);
        assert_eq!(
            sys_onnx_get_metadata(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_onnx_list_providers_null_zero() {
        // Both zero: query size mode
        let args = SyscallArgs::new(0x35, [0, 0, 0, 0, 0, 0]);
        assert_eq!(
            sys_onnx_list_providers(&args),
            SyscallError::NotSupported.as_i64()
        );
    }

    #[test]
    fn test_onnx_list_providers_null_nonzero_len() {
        let args = SyscallArgs::new(0x35, [0, 256, 0, 0, 0, 0]);
        assert_eq!(
            sys_onnx_list_providers(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }
}
