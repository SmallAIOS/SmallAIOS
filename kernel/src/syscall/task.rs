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
use crate::sched::task::TaskId;
use core::sync::atomic::{AtomicU64, Ordering};

/// Maximum scheduling class value (Inference = 2).
const MAX_SCHEDULING_CLASS: usize = 2;

/// Current task ID per-CPU placeholder.
/// In production this would be per-CPU storage; for now a single atomic
/// suffices for the unikernel's cooperative scheduler.
static CURRENT_TASK: AtomicU64 = AtomicU64::new(0);

/// Set the current task ID (called by the scheduler when switching tasks).
pub fn set_current_task(id: u64) {
    CURRENT_TASK.store(id, Ordering::Release);
}

/// Get the current task ID (used by other syscall modules, e.g., capability).
pub fn current_task_id() -> u64 {
    CURRENT_TASK.load(Ordering::Acquire)
}

/// Spawn a new task.
///
/// Args: [entry_fn_ptr, arg, task_type, 0, 0, 0]
/// Returns: task ID on success, negative error code on failure.
pub fn sys_task_spawn(args: &SyscallArgs) -> SyscallResult {
    let entry = args.args[0];
    let _arg = args.args[1];
    let task_type = args.args[2];

    if entry == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }

    // Validate task_type maps to a valid scheduling class
    if task_type > MAX_SCHEDULING_CLASS {
        return SyscallError::InvalidArgument.as_i64();
    }

    // Allocate a new task ID from the scheduler
    let id = TaskId::new();

    // The task will be fully created and enqueued by the executor
    // when the async runtime integration is complete. For now,
    // we return the allocated ID to confirm the syscall path works.
    id.as_u64() as i64
}

/// Yield the current task's time slice.
///
/// Args: [0, 0, 0, 0, 0, 0]
/// Returns: 0 always.
pub fn sys_task_yield(_args: &SyscallArgs) -> SyscallResult {
    // In the cooperative scheduler, yield enqueues the current task
    // at the back of its run queue and switches to the next ready task.
    // The actual context switch happens in the executor's poll loop.
    SyscallError::Success.as_i64()
}

/// Exit the current task.
///
/// Args: [exit_code, 0, 0, 0, 0, 0]
/// Returns: does not return (but signature requires i64).
pub fn sys_task_exit(args: &SyscallArgs) -> SyscallResult {
    let _exit_code = args.args[0];

    // Mark the current task as Done with the given exit code.
    // Wake any tasks blocked on join for this task.
    // The scheduler will not reschedule a Done task.
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

    // Look up the target task. If it's already Done, return its exit code.
    // If still running, block the current task until the target completes.
    // This requires the executor's waker integration (async join handle).
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

    // Look up task and update its priority in the run queue.
    // This may cause a re-sort of the priority queue.
    SyscallError::Success.as_i64()
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

    // Look up task and move it to the appropriate scheduling class queue.
    SyscallError::Success.as_i64()
}

/// Get the current task's ID.
///
/// Args: [0, 0, 0, 0, 0, 0]
/// Returns: current task ID on success.
pub fn sys_task_current(_args: &SyscallArgs) -> SyscallResult {
    let id = CURRENT_TASK.load(Ordering::Acquire);
    id as i64
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
        let result = sys_task_spawn(&args);
        assert!(result > 0, "spawn should return a positive task ID");
    }

    #[test]
    fn test_task_spawn_invalid_class() {
        let args = SyscallArgs::new(0x10, [0x1000, 0, 3, 0, 0, 0]);
        assert_eq!(
            sys_task_spawn(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_task_spawn_allocates_unique_ids() {
        let args1 = SyscallArgs::new(0x10, [0x1000, 0, 0, 0, 0, 0]);
        let args2 = SyscallArgs::new(0x10, [0x2000, 0, 1, 0, 0, 0]);
        let id1 = sys_task_spawn(&args1);
        let id2 = sys_task_spawn(&args2);
        assert!(id1 > 0);
        assert!(id2 > 0);
        assert_ne!(id1, id2, "each spawn must return a unique ID");
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
        assert_eq!(sys_task_join(&args), SyscallError::InvalidArgument.as_i64());
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
        assert_eq!(sys_task_set_priority(&args), SyscallError::Success.as_i64());
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
            assert_eq!(sys_task_set_class(&args), SyscallError::Success.as_i64());
        }
    }

    #[test]
    fn test_task_current_returns_id() {
        set_current_task(42);
        let args = SyscallArgs::zero(0x16);
        assert_eq!(sys_task_current(&args), 42);
    }
}
