// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Task definition — the fundamental unit of scheduling.
//!
//! Each task has:
//! - Unique ID
//! - Scheduling class (SYSTEM, IPC, INFERENCE)
//! - Task type (WatchdogTask, InferenceTask, etc.)
//! - State machine (Created → Ready → Running → Waiting/Done)
//! - Optional CPU affinity
//! - Waker for the async executor

use core::future::Future;
use core::pin::Pin;
use core::sync::atomic::{AtomicU64, Ordering};
use core::task::{Context, Poll};

/// Monotonically increasing task ID counter.
static NEXT_TASK_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for a task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct TaskId(pub u64);

impl Default for TaskId {
    fn default() -> Self {
        Self::new()
    }
}

impl TaskId {
    /// Allocate a new unique task ID.
    pub fn new() -> Self {
        Self(NEXT_TASK_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value.
    pub fn as_u64(self) -> u64 {
        self.0
    }
}

/// Scheduling class — determines priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum SchedulingClass {
    /// Highest priority. Watchdog servicing, health probes, syslog, timers.
    /// Hard-RT: must complete within 1 ms.
    System = 0,
    /// Medium priority. IPC routing, TCP, model reload.
    /// Soft-RT: target < 10 ms. Preempts INFERENCE at operator boundaries.
    Ipc = 1,
    /// Normal priority. ONNX execution, GPU commands.
    /// Best-effort with per-operator time budgets.
    Inference = 2,
}

impl SchedulingClass {
    /// Get the numeric priority (lower = higher priority).
    pub fn priority(self) -> u8 {
        self as u8
    }
}

/// Task type — determines the scheduling class and behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskType {
    // SYSTEM class
    Watchdog = 0,
    Health = 1,
    Syslog = 2,
    Timer = 3,
    // IPC class
    IpcRouter = 10,
    Tcp = 11,
    ModelReload = 12,
    // INFERENCE class
    Inference = 20,
    Gpu = 21,
}

impl TaskType {
    /// Get the scheduling class for this task type.
    pub fn scheduling_class(self) -> SchedulingClass {
        match self {
            Self::Watchdog | Self::Health | Self::Syslog | Self::Timer => SchedulingClass::System,
            Self::IpcRouter | Self::Tcp | Self::ModelReload => SchedulingClass::Ipc,
            Self::Inference | Self::Gpu => SchedulingClass::Inference,
        }
    }
}

/// Task state machine.
///
/// ```text
/// Created → Ready → Running → Done
///                ↗          ↘
///           Waiting ←────────
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskState {
    /// Task has been created but not yet enqueued.
    Created = 0,
    /// Task is in a run queue, waiting to be scheduled.
    Ready = 1,
    /// Task is actively executing on a CPU core.
    Running = 2,
    /// Task is blocked, waiting for an event (I/O, timer, etc.).
    Waiting = 3,
    /// Task has completed execution.
    Done = 4,
}

/// CPU affinity — which core(s) a task may run on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuAffinity {
    /// Task may run on any core.
    Any,
    /// Task is pinned to a specific core.
    Pinned(u32),
    /// Task prefers a specific core but may migrate.
    Preferred(u32),
}

/// A task wrapping an async future for the cooperative executor.
pub struct Task {
    /// Unique task identifier.
    pub id: TaskId,
    /// Scheduling class.
    pub class: SchedulingClass,
    /// Task type.
    pub task_type: TaskType,
    /// Current state.
    pub state: TaskState,
    /// CPU affinity.
    pub affinity: CpuAffinity,
    /// The async future being driven to completion.
    pub future: Pin<&'static mut (dyn Future<Output = ()> + Send)>,
}

impl Task {
    /// Create a new task from a pinned future.
    pub fn new(
        task_type: TaskType,
        future: Pin<&'static mut (dyn Future<Output = ()> + Send)>,
    ) -> Self {
        Self {
            id: TaskId::new(),
            class: task_type.scheduling_class(),
            task_type,
            state: TaskState::Created,
            affinity: CpuAffinity::Any,
            future,
        }
    }

    /// Create a task with a specific CPU affinity.
    pub fn with_affinity(
        task_type: TaskType,
        affinity: CpuAffinity,
        future: Pin<&'static mut (dyn Future<Output = ()> + Send)>,
    ) -> Self {
        Self {
            id: TaskId::new(),
            class: task_type.scheduling_class(),
            task_type,
            state: TaskState::Created,
            affinity,
            future,
        }
    }

    /// Poll the task's future. Returns true if the task completed.
    pub fn poll(&mut self, cx: &mut Context<'_>) -> bool {
        match self.future.as_mut().poll(cx) {
            Poll::Ready(()) => {
                self.state = TaskState::Done;
                true
            }
            Poll::Pending => false,
        }
    }

    /// Transition task to Ready state.
    pub fn make_ready(&mut self) {
        debug_assert!(
            self.state == TaskState::Created
                || self.state == TaskState::Waiting
                || self.state == TaskState::Running,
            "invalid state transition to Ready from {:?}",
            self.state
        );
        self.state = TaskState::Ready;
    }

    /// Transition task to Running state.
    pub fn make_running(&mut self) {
        debug_assert_eq!(self.state, TaskState::Ready);
        self.state = TaskState::Running;
    }

    /// Transition task to Waiting state.
    pub fn make_waiting(&mut self) {
        debug_assert_eq!(self.state, TaskState::Running);
        self.state = TaskState::Waiting;
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::boxed::Box;

    use super::*;

    #[test]
    fn test_task_id_uniqueness() {
        let id1 = TaskId::new();
        let id2 = TaskId::new();
        assert_ne!(id1, id2);
        assert!(id2.as_u64() > id1.as_u64());
    }

    #[test]
    fn test_scheduling_class_priority() {
        assert!(SchedulingClass::System < SchedulingClass::Ipc);
        assert!(SchedulingClass::Ipc < SchedulingClass::Inference);
        assert_eq!(SchedulingClass::System.priority(), 0);
        assert_eq!(SchedulingClass::Ipc.priority(), 1);
        assert_eq!(SchedulingClass::Inference.priority(), 2);
    }

    #[test]
    fn test_task_type_class() {
        assert_eq!(
            TaskType::Watchdog.scheduling_class(),
            SchedulingClass::System
        );
        assert_eq!(TaskType::Health.scheduling_class(), SchedulingClass::System);
        assert_eq!(TaskType::IpcRouter.scheduling_class(), SchedulingClass::Ipc);
        assert_eq!(TaskType::Tcp.scheduling_class(), SchedulingClass::Ipc);
        assert_eq!(
            TaskType::Inference.scheduling_class(),
            SchedulingClass::Inference
        );
        assert_eq!(TaskType::Gpu.scheduling_class(), SchedulingClass::Inference);
    }

    #[test]
    fn test_task_state_transitions() {
        // We can't easily construct a real Future in no_std tests,
        // but we can verify the state enum values
        assert_eq!(TaskState::Created as u8, 0);
        assert_eq!(TaskState::Ready as u8, 1);
        assert_eq!(TaskState::Running as u8, 2);
        assert_eq!(TaskState::Waiting as u8, 3);
        assert_eq!(TaskState::Done as u8, 4);
    }

    #[test]
    fn test_cpu_affinity() {
        assert_eq!(CpuAffinity::Any, CpuAffinity::Any);
        assert_eq!(CpuAffinity::Pinned(0), CpuAffinity::Pinned(0));
        assert_ne!(CpuAffinity::Pinned(0), CpuAffinity::Pinned(1));
        assert_ne!(CpuAffinity::Any, CpuAffinity::Pinned(0));
        assert_eq!(CpuAffinity::Preferred(3), CpuAffinity::Preferred(3));
        assert_ne!(CpuAffinity::Preferred(3), CpuAffinity::Pinned(3));
    }

    #[test]
    fn test_all_task_types_scheduling_class() {
        // Cover every arm of TaskType::scheduling_class()
        assert_eq!(
            TaskType::Watchdog.scheduling_class(),
            SchedulingClass::System
        );
        assert_eq!(TaskType::Health.scheduling_class(), SchedulingClass::System);
        assert_eq!(TaskType::Syslog.scheduling_class(), SchedulingClass::System);
        assert_eq!(TaskType::Timer.scheduling_class(), SchedulingClass::System);
        assert_eq!(TaskType::IpcRouter.scheduling_class(), SchedulingClass::Ipc);
        assert_eq!(TaskType::Tcp.scheduling_class(), SchedulingClass::Ipc);
        assert_eq!(
            TaskType::ModelReload.scheduling_class(),
            SchedulingClass::Ipc
        );
        assert_eq!(
            TaskType::Inference.scheduling_class(),
            SchedulingClass::Inference
        );
        assert_eq!(TaskType::Gpu.scheduling_class(), SchedulingClass::Inference);
    }

    // A simple future that completes immediately.
    struct ImmediateFuture;
    impl Future for ImmediateFuture {
        type Output = ();
        fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            Poll::Ready(())
        }
    }

    // A future that returns Pending on first poll, Ready on second.
    struct TwoPollFuture {
        polled: bool,
    }
    impl Future for TwoPollFuture {
        type Output = ();
        fn poll(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<()> {
            if self.polled {
                Poll::Ready(())
            } else {
                self.polled = true;
                Poll::Pending
            }
        }
    }

    fn make_noop_waker() -> core::task::Waker {
        fn noop(_: *const ()) {}
        fn clone_fn(p: *const ()) -> core::task::RawWaker {
            core::task::RawWaker::new(p, &VTABLE)
        }
        static VTABLE: core::task::RawWakerVTable =
            core::task::RawWakerVTable::new(clone_fn, noop, noop, noop);
        unsafe {
            core::task::Waker::from_raw(core::task::RawWaker::new(core::ptr::null(), &VTABLE))
        }
    }

    #[test]
    fn test_task_poll_immediate() {
        // Task that completes on first poll
        let fut = Box::leak(Box::new(ImmediateFuture));
        let fut_pin = Pin::new(fut);
        let mut task = Task::new(TaskType::Inference, fut_pin);
        assert_eq!(task.state, TaskState::Created);

        task.make_ready();
        assert_eq!(task.state, TaskState::Ready);

        task.make_running();
        assert_eq!(task.state, TaskState::Running);

        let waker = make_noop_waker();
        let mut cx = Context::from_waker(&waker);
        let done = task.poll(&mut cx);
        assert!(done);
        assert_eq!(task.state, TaskState::Done);
    }

    #[test]
    fn test_task_poll_pending_then_ready() {
        // `Task` requires a `'static` future; allocate manually so the
        // test can reclaim it at the end (Miri's leak checker reports
        // `Box::leak` allocations at process exit).
        let fut_raw: *mut TwoPollFuture = Box::into_raw(Box::new(TwoPollFuture { polled: false }));
        // SAFETY: `fut_raw` came from `Box::into_raw` above — non-null,
        // aligned, and uniquely owned until `Box::from_raw` below.
        let fut: &'static mut TwoPollFuture = unsafe { &mut *fut_raw };
        let fut_pin = Pin::new(fut);
        let mut task = Task::new(TaskType::IpcRouter, fut_pin);

        task.make_ready();
        task.make_running();

        let waker = make_noop_waker();
        let mut cx = Context::from_waker(&waker);

        // First poll returns Pending
        let done = task.poll(&mut cx);
        assert!(!done);
        assert_eq!(task.state, TaskState::Running); // state unchanged by pending

        // Second poll returns Ready
        let done = task.poll(&mut cx);
        assert!(done);
        assert_eq!(task.state, TaskState::Done);

        // Reclaim the future: `task` holds the only reference to it.
        // (`Task` has no `Drop` impl; the drop is still semantically
        // load-bearing — it ends the borrow of the future.)
        #[allow(clippy::drop_non_drop)]
        drop(task);
        // SAFETY: `task` was dropped above, so no reference into the
        // allocation remains and `fut_raw` still uniquely owns it.
        unsafe { drop(Box::from_raw(fut_raw)) };
    }

    #[test]
    fn test_task_with_affinity() {
        let fut = Box::leak(Box::new(ImmediateFuture));
        let fut_pin = Pin::new(fut);
        let task = Task::with_affinity(TaskType::Gpu, CpuAffinity::Pinned(2), fut_pin);
        assert_eq!(task.affinity, CpuAffinity::Pinned(2));
        assert_eq!(task.class, SchedulingClass::Inference);
        assert_eq!(task.task_type, TaskType::Gpu);
    }

    #[test]
    fn test_task_state_waiting_transition() {
        // Manually managed allocation — see test_task_poll_pending_then_ready.
        let fut_raw: *mut TwoPollFuture = Box::into_raw(Box::new(TwoPollFuture { polled: false }));
        // SAFETY: `fut_raw` came from `Box::into_raw` above — non-null,
        // aligned, and uniquely owned until `Box::from_raw` below.
        let fut: &'static mut TwoPollFuture = unsafe { &mut *fut_raw };
        let fut_pin = Pin::new(fut);
        let mut task = Task::new(TaskType::Tcp, fut_pin);

        task.make_ready();
        task.make_running();
        task.make_waiting(); // Running → Waiting
        assert_eq!(task.state, TaskState::Waiting);

        task.make_ready(); // Waiting → Ready
        assert_eq!(task.state, TaskState::Ready);

        // Reclaim the future: `task` holds the only reference to it.
        // (`Task` has no `Drop` impl; the drop is still semantically
        // load-bearing — it ends the borrow of the future.)
        #[allow(clippy::drop_non_drop)]
        drop(task);
        // SAFETY: `task` was dropped above, so no reference into the
        // allocation remains and `fut_raw` still uniquely owns it.
        unsafe { drop(Box::from_raw(fut_raw)) };
    }
}
