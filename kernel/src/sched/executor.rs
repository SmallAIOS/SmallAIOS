// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cooperative async executor with priority scheduling and work-stealing.
//!
//! Design:
//! - Per-core run queues organized by scheduling class
//! - SYSTEM queue always drained first, then IPC, then INFERENCE
//! - Lock-free work-stealing: idle cores steal from busy cores' INFERENCE queues
//! - Waker integration: tasks wake via run queue insertion
//!
//! The executor is cooperative — tasks yield voluntarily (at `await` points
//! or ONNX operator boundaries). Priority preemption happens at yield points.

use super::task::{SchedulingClass, TaskId};
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use core::task::{RawWaker, RawWakerVTable, Waker};

/// Maximum number of CPU cores supported.
pub const MAX_CORES: usize = 128;

/// Maximum tasks per priority queue per core.
const MAX_QUEUE_SIZE: usize = 256;

/// Per-core priority run queue.
///
/// Three queues: SYSTEM (0), IPC (1), INFERENCE (2).
/// Each is a simple circular buffer with head/tail indices.
pub struct RunQueue {
    /// Task IDs in each priority queue.
    queues: [[TaskId; MAX_QUEUE_SIZE]; 3],
    /// Head index for each queue (next to dequeue).
    heads: [usize; 3],
    /// Tail index for each queue (next insert position).
    tails: [usize; 3],
    /// Count of tasks in each queue.
    counts: [usize; 3],
    /// Core ID that owns this queue.
    core_id: u32,
}

impl RunQueue {
    /// Create an empty run queue for a core.
    pub const fn new(core_id: u32) -> Self {
        Self {
            queues: [[TaskId(0); MAX_QUEUE_SIZE]; 3],
            heads: [0; 3],
            tails: [0; 3],
            counts: [0; 3],
            core_id,
        }
    }

    /// Push a task ID into the appropriate priority queue.
    /// Returns false if the queue is full.
    pub fn push(&mut self, class: SchedulingClass, task_id: TaskId) -> bool {
        let q = class.priority() as usize;
        if self.counts[q] >= MAX_QUEUE_SIZE {
            return false;
        }
        self.queues[q][self.tails[q]] = task_id;
        self.tails[q] = (self.tails[q] + 1) % MAX_QUEUE_SIZE;
        self.counts[q] += 1;
        true
    }

    /// Pop the highest-priority task from the queues.
    /// Checks SYSTEM first, then IPC, then INFERENCE.
    pub fn pop(&mut self) -> Option<(SchedulingClass, TaskId)> {
        for q in 0..3 {
            if self.counts[q] > 0 {
                let task_id = self.queues[q][self.heads[q]];
                self.heads[q] = (self.heads[q] + 1) % MAX_QUEUE_SIZE;
                self.counts[q] -= 1;
                let class = match q {
                    0 => SchedulingClass::System,
                    1 => SchedulingClass::Ipc,
                    _ => SchedulingClass::Inference,
                };
                return Some((class, task_id));
            }
        }
        None
    }

    /// Steal a task from the INFERENCE queue (work-stealing).
    /// Takes from the tail (FIFO steal) for better locality.
    pub fn steal_inference(&mut self) -> Option<TaskId> {
        let q = SchedulingClass::Inference.priority() as usize;
        if self.counts[q] == 0 {
            return None;
        }
        // Steal from the tail (opposite end from pop)
        self.tails[q] = if self.tails[q] == 0 {
            MAX_QUEUE_SIZE - 1
        } else {
            self.tails[q] - 1
        };
        let task_id = self.queues[q][self.tails[q]];
        self.counts[q] -= 1;
        Some(task_id)
    }

    /// Total number of tasks across all queues.
    pub fn total_count(&self) -> usize {
        self.counts[0] + self.counts[1] + self.counts[2]
    }

    /// Number of tasks in a specific priority queue.
    pub fn count(&self, class: SchedulingClass) -> usize {
        self.counts[class.priority() as usize]
    }

    /// Check if there are any higher-priority tasks pending.
    /// Used at operator yield points.
    pub fn has_higher_priority_than(&self, class: SchedulingClass) -> bool {
        for q in 0..(class.priority() as usize) {
            if self.counts[q] > 0 {
                return true;
            }
        }
        false
    }

    /// Check if there are any SYSTEM tasks pending.
    pub fn has_system_tasks(&self) -> bool {
        self.counts[0] > 0
    }

    /// Check if there are any IPC tasks pending.
    pub fn has_ipc_tasks(&self) -> bool {
        self.counts[1] > 0
    }

    /// Core ID.
    pub fn core_id(&self) -> u32 {
        self.core_id
    }
}

pub use smallaios_sched_types::{BudgetResult, OperatorBudget, OperatorClass};

/// Watchdog state managed by the executor.
pub struct WatchdogState {
    /// Timeout in abstract ticks (scheduler steps).
    pub timeout_ticks: u64,
    /// Current countdown.
    pub remaining_ticks: AtomicU64,
    /// Whether the watchdog is enabled.
    pub enabled: AtomicBool,
}

impl WatchdogState {
    /// Create a new watchdog with the given timeout.
    pub const fn new(timeout_ticks: u64) -> Self {
        Self {
            timeout_ticks,
            remaining_ticks: AtomicU64::new(timeout_ticks),
            enabled: AtomicBool::new(false),
        }
    }

    /// Enable the watchdog.
    pub fn enable(&self) {
        self.enabled.store(true, Ordering::Release);
        self.remaining_ticks
            .store(self.timeout_ticks, Ordering::Release);
    }

    /// Pet (service) the watchdog — reset the countdown.
    pub fn pet(&self) {
        self.remaining_ticks
            .store(self.timeout_ticks, Ordering::Release);
    }

    /// Tick the watchdog — decrement the countdown.
    /// Returns true if the watchdog has fired (system should reset).
    pub fn tick(&self) -> bool {
        if !self.enabled.load(Ordering::Acquire) {
            return false;
        }
        let prev = self.remaining_ticks.fetch_sub(1, Ordering::AcqRel);
        prev <= 1
    }

    /// Get remaining ticks.
    pub fn remaining(&self) -> u64 {
        self.remaining_ticks.load(Ordering::Acquire)
    }

    /// Check if enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }
}

/// Executor statistics.
pub struct ExecutorStats {
    /// Total tasks scheduled.
    pub tasks_scheduled: AtomicU64,
    /// Total context switches (task changes).
    pub context_switches: AtomicU64,
    /// Total work-steal events.
    pub steals: AtomicU64,
    /// Total idle cycles.
    pub idle_cycles: AtomicU64,
    /// Total operator budget warnings.
    pub budget_warnings: AtomicU64,
    /// Total operator budget soft limits.
    pub budget_soft_limits: AtomicU64,
    /// Total operator budget hard limits (aborts).
    pub budget_hard_limits: AtomicU64,
}

impl Default for ExecutorStats {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutorStats {
    pub const fn new() -> Self {
        Self {
            tasks_scheduled: AtomicU64::new(0),
            context_switches: AtomicU64::new(0),
            steals: AtomicU64::new(0),
            idle_cycles: AtomicU64::new(0),
            budget_warnings: AtomicU64::new(0),
            budget_soft_limits: AtomicU64::new(0),
            budget_hard_limits: AtomicU64::new(0),
        }
    }
}

/// Create a no-op waker for polling futures.
///
/// In the cooperative model, tasks are re-enqueued when they yield
/// (return Poll::Pending), so the waker doesn't need to do anything
/// special — the task is already in the run queue.
pub fn noop_waker() -> Waker {
    fn noop(_: *const ()) {}
    fn clone(p: *const ()) -> RawWaker {
        RawWaker::new(p, &VTABLE)
    }

    static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);

    unsafe { Waker::from_raw(RawWaker::new(core::ptr::null(), &VTABLE)) }
}

/// Yield point — called between ONNX operators.
///
/// This is the primary preemption mechanism in SmallAIOS.
/// Returns true if the current inference task should continue,
/// false if it should yield to higher-priority work.
pub fn should_continue_inference(queue: &RunQueue) -> bool {
    // If SYSTEM or IPC tasks are pending, inference should yield
    !queue.has_higher_priority_than(SchedulingClass::Inference)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_queue_basic() {
        let mut q = RunQueue::new(0);
        assert_eq!(q.total_count(), 0);

        let t1 = TaskId::new();
        let t2 = TaskId::new();
        assert!(q.push(SchedulingClass::Inference, t1));
        assert!(q.push(SchedulingClass::Inference, t2));
        assert_eq!(q.total_count(), 2);

        let (class, id) = q.pop().unwrap();
        assert_eq!(class, SchedulingClass::Inference);
        assert_eq!(id, t1);
    }

    #[test]
    fn test_priority_ordering() {
        let mut q = RunQueue::new(0);

        // Enqueue in reverse priority order
        let inference = TaskId::new();
        let ipc = TaskId::new();
        let system = TaskId::new();

        q.push(SchedulingClass::Inference, inference);
        q.push(SchedulingClass::Ipc, ipc);
        q.push(SchedulingClass::System, system);

        // Should dequeue in priority order: SYSTEM, IPC, INFERENCE
        let (c1, id1) = q.pop().unwrap();
        assert_eq!(c1, SchedulingClass::System);
        assert_eq!(id1, system);

        let (c2, id2) = q.pop().unwrap();
        assert_eq!(c2, SchedulingClass::Ipc);
        assert_eq!(id2, ipc);

        let (c3, id3) = q.pop().unwrap();
        assert_eq!(c3, SchedulingClass::Inference);
        assert_eq!(id3, inference);

        assert!(q.pop().is_none());
    }

    #[test]
    fn test_work_stealing() {
        let mut q = RunQueue::new(0);

        let t1 = TaskId::new();
        let t2 = TaskId::new();
        let t3 = TaskId::new();

        q.push(SchedulingClass::Inference, t1);
        q.push(SchedulingClass::Inference, t2);
        q.push(SchedulingClass::Inference, t3);

        // Steal from tail
        let stolen = q.steal_inference().unwrap();
        assert_eq!(stolen, t3); // Last pushed
        assert_eq!(q.count(SchedulingClass::Inference), 2);

        // Pop from head
        let (_, popped) = q.pop().unwrap();
        assert_eq!(popped, t1); // First pushed
    }

    #[test]
    fn test_has_higher_priority() {
        let mut q = RunQueue::new(0);
        assert!(!q.has_higher_priority_than(SchedulingClass::Inference));

        q.push(SchedulingClass::System, TaskId::new());
        assert!(q.has_higher_priority_than(SchedulingClass::Inference));
        assert!(q.has_higher_priority_than(SchedulingClass::Ipc));
        assert!(!q.has_higher_priority_than(SchedulingClass::System));
    }

    #[test]
    fn test_operator_budget_check() {
        let budget = OperatorBudget::DEFAULT;

        // Within budget
        assert_eq!(
            budget.check(OperatorClass::Elementwise, 500),
            BudgetResult::Ok
        );

        // Warning (1x-2x)
        assert_eq!(
            budget.check(OperatorClass::Elementwise, 1_500),
            BudgetResult::Warning
        );

        // Soft limit (2x-10x)
        assert_eq!(
            budget.check(OperatorClass::Elementwise, 5_000),
            BudgetResult::SoftLimit
        );

        // Hard limit (>10x)
        assert_eq!(
            budget.check(OperatorClass::Elementwise, 15_000),
            BudgetResult::HardLimit
        );
    }

    #[test]
    fn test_watchdog() {
        let wd = WatchdogState::new(10);
        assert!(!wd.is_enabled());

        wd.enable();
        assert!(wd.is_enabled());
        assert_eq!(wd.remaining(), 10);

        // Tick down
        assert!(!wd.tick()); // 9
        assert!(!wd.tick()); // 8
        assert_eq!(wd.remaining(), 8);

        // Pet resets
        wd.pet();
        assert_eq!(wd.remaining(), 10);

        // Tick to zero
        for _ in 0..9 {
            assert!(!wd.tick());
        }
        assert_eq!(wd.remaining(), 1);
        assert!(wd.tick()); // fires at 0
    }

    #[test]
    fn test_should_continue_inference() {
        let mut q = RunQueue::new(0);
        assert!(should_continue_inference(&q)); // No pending work

        q.push(SchedulingClass::Inference, TaskId::new());
        assert!(should_continue_inference(&q)); // Only inference

        q.push(SchedulingClass::Ipc, TaskId::new());
        assert!(!should_continue_inference(&q)); // IPC pending

        // Drain IPC
        q.pop(); // system queue empty, ipc has 1
        q.pop(); // now IPC popped
        assert!(should_continue_inference(&q)); // Only inference again

        q.push(SchedulingClass::System, TaskId::new());
        assert!(!should_continue_inference(&q)); // System pending
    }

    #[test]
    fn test_noop_waker() {
        let waker = noop_waker();
        waker.wake_by_ref(); // Should not panic
        let _cloned = waker.clone();
        drop(waker);
    }

    #[test]
    fn test_executor_stats() {
        let stats = ExecutorStats::new();
        assert_eq!(stats.tasks_scheduled.load(Ordering::Relaxed), 0);
        stats.tasks_scheduled.fetch_add(1, Ordering::Relaxed);
        assert_eq!(stats.tasks_scheduled.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_queue_full() {
        let mut q = RunQueue::new(0);
        for _ in 0..MAX_QUEUE_SIZE {
            assert!(q.push(SchedulingClass::System, TaskId::new()));
        }
        // Should fail when full
        assert!(!q.push(SchedulingClass::System, TaskId::new()));
    }

    #[test]
    fn test_steal_empty() {
        let mut q = RunQueue::new(0);
        assert!(q.steal_inference().is_none());
    }

    #[test]
    fn test_pop_empty() {
        let mut q = RunQueue::new(0);
        assert!(q.pop().is_none());
    }

    #[test]
    fn test_core_id() {
        let q = RunQueue::new(42);
        assert_eq!(q.core_id(), 42);
    }

    #[test]
    fn test_count_per_class() {
        let mut q = RunQueue::new(0);
        assert_eq!(q.count(SchedulingClass::System), 0);
        assert_eq!(q.count(SchedulingClass::Ipc), 0);
        assert_eq!(q.count(SchedulingClass::Inference), 0);

        q.push(SchedulingClass::System, TaskId::new());
        q.push(SchedulingClass::System, TaskId::new());
        q.push(SchedulingClass::Ipc, TaskId::new());
        assert_eq!(q.count(SchedulingClass::System), 2);
        assert_eq!(q.count(SchedulingClass::Ipc), 1);
        assert_eq!(q.count(SchedulingClass::Inference), 0);
    }

    #[test]
    fn test_has_system_and_ipc() {
        let mut q = RunQueue::new(0);
        assert!(!q.has_system_tasks());
        assert!(!q.has_ipc_tasks());

        q.push(SchedulingClass::Ipc, TaskId::new());
        assert!(!q.has_system_tasks());
        assert!(q.has_ipc_tasks());

        q.push(SchedulingClass::System, TaskId::new());
        assert!(q.has_system_tasks());
        assert!(q.has_ipc_tasks());
    }

    #[test]
    fn test_fifo_within_priority() {
        let mut q = RunQueue::new(0);
        let t1 = TaskId::new();
        let t2 = TaskId::new();
        let t3 = TaskId::new();

        q.push(SchedulingClass::Ipc, t1);
        q.push(SchedulingClass::Ipc, t2);
        q.push(SchedulingClass::Ipc, t3);

        let (_, id1) = q.pop().unwrap();
        let (_, id2) = q.pop().unwrap();
        let (_, id3) = q.pop().unwrap();
        assert_eq!(id1, t1);
        assert_eq!(id2, t2);
        assert_eq!(id3, t3);
    }

    #[test]
    fn test_operator_budget_all_classes() {
        let budget = OperatorBudget::DEFAULT;

        // Each operator class within budget
        assert_eq!(
            budget.check(OperatorClass::Reduction, 5_000),
            BudgetResult::Ok
        );
        assert_eq!(budget.check(OperatorClass::Gemm, 50_000), BudgetResult::Ok);
        assert_eq!(
            budget.check(OperatorClass::Attention, 400_000),
            BudgetResult::Ok
        );
        assert_eq!(
            budget.check(OperatorClass::GpuKernel, 900_000),
            BudgetResult::Ok
        );

        // Each operator class at warning level
        assert_eq!(
            budget.check(OperatorClass::Reduction, 15_000),
            BudgetResult::Warning
        );
        assert_eq!(
            budget.check(OperatorClass::Gemm, 150_000),
            BudgetResult::Warning
        );
        assert_eq!(
            budget.check(OperatorClass::Attention, 800_000),
            BudgetResult::Warning
        );
        assert_eq!(
            budget.check(OperatorClass::GpuKernel, 1_500_000),
            BudgetResult::Warning
        );

        // Each operator class at soft limit
        assert_eq!(
            budget.check(OperatorClass::Reduction, 50_000),
            BudgetResult::SoftLimit
        );
        assert_eq!(
            budget.check(OperatorClass::Gemm, 500_000),
            BudgetResult::SoftLimit
        );

        // Each operator class at hard limit
        assert_eq!(
            budget.check(OperatorClass::Reduction, 200_000),
            BudgetResult::HardLimit
        );
        assert_eq!(
            budget.check(OperatorClass::Gemm, 2_000_000),
            BudgetResult::HardLimit
        );
        assert_eq!(
            budget.check(OperatorClass::Attention, 10_000_000),
            BudgetResult::HardLimit
        );
        assert_eq!(
            budget.check(OperatorClass::GpuKernel, 20_000_000),
            BudgetResult::HardLimit
        );
    }

    #[test]
    fn test_budget_boundary_values() {
        let budget = OperatorBudget::DEFAULT;

        // Exactly at budget = Ok
        assert_eq!(
            budget.check(OperatorClass::Elementwise, 1_000),
            BudgetResult::Ok
        );
        // 1 us over budget = Warning
        assert_eq!(
            budget.check(OperatorClass::Elementwise, 1_001),
            BudgetResult::Warning
        );
        // Exactly at soft limit = Ok (not exceeded)
        assert_eq!(
            budget.check(OperatorClass::Elementwise, 2_000),
            BudgetResult::Warning
        );
        // 1 us over soft limit = SoftLimit
        assert_eq!(
            budget.check(OperatorClass::Elementwise, 2_001),
            BudgetResult::SoftLimit
        );
        // Exactly at hard limit = SoftLimit (not exceeded)
        assert_eq!(
            budget.check(OperatorClass::Elementwise, 10_000),
            BudgetResult::SoftLimit
        );
        // 1 us over hard limit = HardLimit
        assert_eq!(
            budget.check(OperatorClass::Elementwise, 10_001),
            BudgetResult::HardLimit
        );
    }

    #[test]
    fn test_watchdog_disabled() {
        let wd = WatchdogState::new(5);
        assert!(!wd.is_enabled());

        // Tick while disabled should return false (not fire)
        assert!(!wd.tick());
        assert!(!wd.tick());
        assert!(!wd.tick());
    }

    #[test]
    fn test_watchdog_pet_then_fire() {
        let wd = WatchdogState::new(3);
        wd.enable();

        assert!(!wd.tick()); // 2
        wd.pet(); // back to 3
        assert_eq!(wd.remaining(), 3);

        assert!(!wd.tick()); // 2
        assert!(!wd.tick()); // 1
        assert!(wd.tick()); // fires at 0
    }

    #[test]
    fn test_executor_stats_all_counters() {
        let stats = ExecutorStats::new();
        stats.context_switches.fetch_add(3, Ordering::Relaxed);
        stats.steals.fetch_add(2, Ordering::Relaxed);
        stats.idle_cycles.fetch_add(100, Ordering::Relaxed);
        stats.budget_warnings.fetch_add(1, Ordering::Relaxed);
        stats.budget_soft_limits.fetch_add(1, Ordering::Relaxed);
        stats.budget_hard_limits.fetch_add(1, Ordering::Relaxed);

        assert_eq!(stats.context_switches.load(Ordering::Relaxed), 3);
        assert_eq!(stats.steals.load(Ordering::Relaxed), 2);
        assert_eq!(stats.idle_cycles.load(Ordering::Relaxed), 100);
        assert_eq!(stats.budget_warnings.load(Ordering::Relaxed), 1);
        assert_eq!(stats.budget_soft_limits.load(Ordering::Relaxed), 1);
        assert_eq!(stats.budget_hard_limits.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_steal_after_partial_pop() {
        let mut q = RunQueue::new(0);
        let t1 = TaskId::new();
        let t2 = TaskId::new();
        let t3 = TaskId::new();

        q.push(SchedulingClass::Inference, t1);
        q.push(SchedulingClass::Inference, t2);
        q.push(SchedulingClass::Inference, t3);

        // Pop one from head
        let (_, popped) = q.pop().unwrap();
        assert_eq!(popped, t1);

        // Steal one from tail
        let stolen = q.steal_inference().unwrap();
        assert_eq!(stolen, t3);

        // Only t2 remains
        assert_eq!(q.count(SchedulingClass::Inference), 1);
        let (_, last) = q.pop().unwrap();
        assert_eq!(last, t2);
    }

    #[test]
    fn test_queue_wrap_around() {
        let mut q = RunQueue::new(0);

        // Fill and drain several times to wrap around indices
        for _ in 0..3 {
            for _ in 0..MAX_QUEUE_SIZE {
                assert!(q.push(SchedulingClass::Inference, TaskId::new()));
            }
            for _ in 0..MAX_QUEUE_SIZE {
                assert!(q.pop().is_some());
            }
        }
        assert_eq!(q.total_count(), 0);

        // Should still work after wrapping
        let t = TaskId::new();
        assert!(q.push(SchedulingClass::Inference, t));
        let (_, id) = q.pop().unwrap();
        assert_eq!(id, t);
    }

    #[test]
    fn test_mixed_priority_interleaving() {
        let mut q = RunQueue::new(0);

        // Add inference tasks
        let inf1 = TaskId::new();
        let inf2 = TaskId::new();
        q.push(SchedulingClass::Inference, inf1);
        q.push(SchedulingClass::Inference, inf2);

        // Pop one inference (only inference available)
        let (c, _) = q.pop().unwrap();
        assert_eq!(c, SchedulingClass::Inference);

        // Now add system task — should be popped next despite remaining inference
        let sys = TaskId::new();
        q.push(SchedulingClass::System, sys);
        let (c, id) = q.pop().unwrap();
        assert_eq!(c, SchedulingClass::System);
        assert_eq!(id, sys);
    }
}
