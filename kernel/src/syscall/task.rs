// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Task syscalls (0x10-0x1F).
//!
//! - `task_spawn(entry, arg) -> TaskId`
//! - `task_yield()`
//! - `task_exit(code)`
//! - `task_join(id) -> ExitCode`
//! - `task_set_priority(id, priority)`
//! - `task_set_class(id, class)` — set scheduling class
//! - `task_current() -> TaskId`

use super::{SyscallArgs, SyscallError, SyscallResult};

/// Maximum scheduling class value (Inference = 2).
const MAX_SCHEDULING_CLASS: usize = 2;

/// Spawn a new task.
///
/// Args: [entry_fn_ptr, arg, task_type, 0, 0, 0]
/// Returns: task ID on success, negative error code on failure.
pub fn sys_task_spawn(args: &SyscallArgs) -> SyscallResult {
    let entry = args.args[0];
    let arg = args.args[1];
    let task_type = args.args[2];

    if entry == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with scheduler to spawn tasks
    let _ = (entry, arg, task_type);
    SyscallError::NotSupported.as_i64()
}

/// Yield the current task's time slice.
///
/// Args: [0, 0, 0, 0, 0, 0]
/// Returns: 0 always.
pub fn sys_task_yield(_args: &SyscallArgs) -> SyscallResult {
    // TODO: Integrate with scheduler — enqueue current task, switch
    SyscallError::Success.as_i64()
}

/// Exit the current task.
///
/// Args: [exit_code, 0, 0, 0, 0, 0]
/// Returns: does not return (but signature requires i64).
pub fn sys_task_exit(args: &SyscallArgs) -> SyscallResult {
    let _exit_code = args.args[0];

    // TODO: Integrate with scheduler — mark task Done, notify joiners
    SyscallError::Success.as_i64()
}

/// Join (wait for) a task to complete.
///
/// Args: [task_id, 0, 0, 0, 0, 0]
/// Returns: exit code of the joined task, or negative error code.
pub fn sys_task_join(args: &SyscallArgs) -> SyscallResult {
    let task_id = args.args[0];

    if task_id == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with scheduler — block until target task completes
    let _ = task_id;
    SyscallError::NotSupported.as_i64()
}

/// Set task priority within its scheduling class.
///
/// Args: [task_id, priority, 0, 0, 0, 0]
/// Returns: 0 on success, negative error code on failure.
pub fn sys_task_set_priority(args: &SyscallArgs) -> SyscallResult {
    let task_id = args.args[0];
    let priority = args.args[1];

    if task_id == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    // Priority is 0-255 within a scheduling class
    if priority > 255 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with scheduler — update task priority
    let _ = (task_id, priority);
    SyscallError::NotSupported.as_i64()
}

/// Set task scheduling class.
///
/// Args: [task_id, class, 0, 0, 0, 0]
/// class: 0=System, 1=Ipc, 2=Inference
/// Returns: 0 on success, negative error code on failure.
pub fn sys_task_set_class(args: &SyscallArgs) -> SyscallResult {
    let task_id = args.args[0];
    let class = args.args[1];

    if task_id == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    if class > MAX_SCHEDULING_CLASS {
        return SyscallError::InvalidArgument.as_i64();
    }

    // TODO: Integrate with scheduler — update scheduling class
    let _ = (task_id, class);
    SyscallError::NotSupported.as_i64()
}

/// Get the current task's ID.
///
/// Args: [0, 0, 0, 0, 0, 0]
/// Returns: current task ID on success, negative error code on failure.
pub fn sys_task_current(_args: &SyscallArgs) -> SyscallResult {
    // TODO: Read from per-CPU current task pointer
    // For now, return 0 (no current task in stub mode)
    SyscallError::Success.as_i64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_spawn_null_entry() {
        let args = SyscallArgs::new(0x10, [0, 0, 0, 0, 0, 0]);
        assert_eq!(
            sys_task_spawn(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_task_spawn_valid() {
        let args = SyscallArgs::new(0x10, [0x1000, 42, 0, 0, 0, 0]);
        assert_eq!(
            sys_task_spawn(&args),
            SyscallError::NotSupported.as_i64()
        );
    }

    #[test]
    fn test_task_yield_returns_success() {
        let args = SyscallArgs::zero(0x11);
        assert_eq!(sys_task_yield(&args), SyscallError::Success.as_i64());
    }

    #[test]
    fn test_task_exit_returns_success() {
        let args = SyscallArgs::new(0x12, [0, 0, 0, 0, 0, 0]);
        assert_eq!(sys_task_exit(&args), SyscallError::Success.as_i64());
    }

    #[test]
    fn test_task_join_zero_id() {
        let args = SyscallArgs::new(0x13, [0, 0, 0, 0, 0, 0]);
        assert_eq!(
            sys_task_join(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_task_set_priority_zero_id() {
        let args = SyscallArgs::new(0x14, [0, 5, 0, 0, 0, 0]);
        assert_eq!(
            sys_task_set_priority(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_task_set_priority_too_high() {
        let args = SyscallArgs::new(0x14, [1, 256, 0, 0, 0, 0]);
        assert_eq!(
            sys_task_set_priority(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_task_set_priority_max_valid() {
        let args = SyscallArgs::new(0x14, [1, 255, 0, 0, 0, 0]);
        assert_eq!(
            sys_task_set_priority(&args),
            SyscallError::NotSupported.as_i64()
        );
    }

    #[test]
    fn test_task_set_class_zero_id() {
        let args = SyscallArgs::new(0x15, [0, 0, 0, 0, 0, 0]);
        assert_eq!(
            sys_task_set_class(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_task_set_class_invalid_class() {
        let args = SyscallArgs::new(0x15, [1, 3, 0, 0, 0, 0]);
        assert_eq!(
            sys_task_set_class(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_task_set_class_all_valid() {
        for class in 0..=MAX_SCHEDULING_CLASS {
            let args = SyscallArgs::new(0x15, [1, class, 0, 0, 0, 0]);
            assert_eq!(
                sys_task_set_class(&args),
                SyscallError::NotSupported.as_i64()
            );
        }
    }

    #[test]
    fn test_task_current_returns_success() {
        let args = SyscallArgs::zero(0x16);
        assert_eq!(sys_task_current(&args), SyscallError::Success.as_i64());
    }
}
