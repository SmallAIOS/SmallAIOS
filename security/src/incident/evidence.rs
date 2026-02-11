// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Evidence preservation for incident response.
//!
//! Captures system state at incident time per NIST SP 800-61:
//! - Audit log batch export
//! - Task state snapshot (running, blocked, terminated + capabilities)
//! - Memory allocation statistics (per-region usage, fragmentation, failures)
//! - Capability registry snapshot (active capabilities, owners, permissions)
//!
//! Evidence is timestamped and bundled into an [`EvidenceSnapshot`].

use alloc::vec::Vec;

/// Maximum number of task state entries in a single snapshot.
const MAX_TASK_ENTRIES: usize = 128;

/// Maximum number of capability entries in a single snapshot.
const MAX_CAP_ENTRIES: usize = 256;

/// A complete evidence snapshot captured at incident time.
#[derive(Debug, Clone)]
pub struct EvidenceSnapshot {
    /// Nanosecond timestamp when the snapshot was captured.
    pub timestamp_ns: u64,
    /// Incident ID this evidence relates to.
    pub incident_id: u64,
    /// Task state at time of capture.
    pub task_states: Vec<TaskStateEntry>,
    /// Memory allocation statistics at time of capture.
    pub memory_stats: MemorySnapshot,
    /// Capability registry snapshot.
    pub capability_snapshot: Vec<CapabilityEntry>,
    /// Number of audit batches available for export.
    pub audit_batch_count: u32,
}

impl EvidenceSnapshot {
    /// Create a new evidence snapshot with the given incident and timestamp.
    pub fn new(incident_id: u64, timestamp_ns: u64) -> Self {
        Self {
            timestamp_ns,
            incident_id,
            task_states: Vec::new(),
            memory_stats: MemorySnapshot::empty(),
            capability_snapshot: Vec::new(),
            audit_batch_count: 0,
        }
    }

    /// Add a task state entry.
    pub fn add_task_state(&mut self, entry: TaskStateEntry) -> bool {
        if self.task_states.len() >= MAX_TASK_ENTRIES {
            return false;
        }
        self.task_states.push(entry);
        true
    }

    /// Add a capability entry.
    pub fn add_capability(&mut self, entry: CapabilityEntry) -> bool {
        if self.capability_snapshot.len() >= MAX_CAP_ENTRIES {
            return false;
        }
        self.capability_snapshot.push(entry);
        true
    }

    /// Set memory statistics.
    pub fn set_memory_stats(&mut self, stats: MemorySnapshot) {
        self.memory_stats = stats;
    }

    /// Set the number of audit batches available.
    pub fn set_audit_batch_count(&mut self, count: u32) {
        self.audit_batch_count = count;
    }

    /// Total number of evidence items captured.
    pub fn total_items(&self) -> usize {
        self.task_states.len() + self.capability_snapshot.len() + 1 // +1 for memory stats
    }
}

/// Task execution state at time of snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskState {
    /// Task is currently running.
    Running = 0,
    /// Task is blocked waiting for a resource.
    Blocked = 1,
    /// Task has been terminated.
    Terminated = 2,
    /// Task is suspended (containment or voluntary).
    Suspended = 3,
}

impl TaskState {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Running),
            1 => Some(Self::Blocked),
            2 => Some(Self::Terminated),
            3 => Some(Self::Suspended),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Blocked => "blocked",
            Self::Terminated => "terminated",
            Self::Suspended => "suspended",
        }
    }
}

/// A single task's state at evidence capture time.
#[derive(Debug, Clone)]
pub struct TaskStateEntry {
    /// Task identifier.
    pub task_id: u64,
    /// Current execution state.
    pub state: TaskState,
    /// Number of capabilities held by this task.
    pub capability_count: u32,
    /// Whether the task is under containment.
    pub under_containment: bool,
    /// Task creation timestamp (nanoseconds).
    pub created_ns: u64,
}

/// Memory allocation statistics snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemorySnapshot {
    /// Buddy allocator statistics.
    pub buddy: RegionStats,
    /// Slab allocator statistics.
    pub slab: RegionStats,
    /// Tensor buffer statistics.
    pub tensor: RegionStats,
    /// Total allocation failures since boot.
    pub total_alloc_failures: u64,
    /// Recent allocation failures (last 60 seconds).
    pub recent_alloc_failures: u32,
}

impl MemorySnapshot {
    /// Create an empty memory snapshot (all zeros).
    pub fn empty() -> Self {
        Self {
            buddy: RegionStats::empty(),
            slab: RegionStats::empty(),
            tensor: RegionStats::empty(),
            total_alloc_failures: 0,
            recent_alloc_failures: 0,
        }
    }

    /// Total memory used across all regions.
    pub fn total_used(&self) -> u64 {
        self.buddy.used_bytes + self.slab.used_bytes + self.tensor.used_bytes
    }

    /// Total memory capacity across all regions.
    pub fn total_capacity(&self) -> u64 {
        self.buddy.total_bytes + self.slab.total_bytes + self.tensor.total_bytes
    }
}

/// Per-region memory statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegionStats {
    /// Total bytes available in this region.
    pub total_bytes: u64,
    /// Bytes currently in use.
    pub used_bytes: u64,
    /// Number of active allocations.
    pub active_allocations: u32,
    /// Fragmentation percentage (0-100).
    pub fragmentation_pct: u8,
}

impl RegionStats {
    /// Create empty region statistics.
    pub fn empty() -> Self {
        Self {
            total_bytes: 0,
            used_bytes: 0,
            active_allocations: 0,
            fragmentation_pct: 0,
        }
    }

    /// Usage percentage (0-100).
    pub fn usage_pct(&self) -> u8 {
        if self.total_bytes == 0 {
            return 0;
        }
        ((self.used_bytes * 100) / self.total_bytes).min(100) as u8
    }
}

/// A capability entry in the evidence snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityEntry {
    /// Capability ID.
    pub cap_id: u64,
    /// Owning task ID.
    pub owner_task_id: u64,
    /// Resource type (from capability system).
    pub resource_type: u8,
    /// Resource instance.
    pub resource_instance: u32,
    /// Permission bits.
    pub permissions: u32,
    /// Creation timestamp (nanoseconds).
    pub created_ns: u64,
}

/// Serialized size of the evidence header for transport.
///
/// Layout:
/// - [0..8]   timestamp_ns (u64 BE)
/// - [8..16]  incident_id (u64 BE)
/// - [16..18] task_count (u16 BE)
/// - [18..20] cap_count (u16 BE)
/// - [20..24] audit_batch_count (u32 BE)
pub const EVIDENCE_HEADER_SIZE: usize = 24;

impl EvidenceSnapshot {
    /// Serialize the evidence header to bytes.
    pub fn serialize_header(&self, buf: &mut [u8; EVIDENCE_HEADER_SIZE]) {
        buf[0..8].copy_from_slice(&self.timestamp_ns.to_be_bytes());
        buf[8..16].copy_from_slice(&self.incident_id.to_be_bytes());
        let task_count = self.task_states.len().min(u16::MAX as usize) as u16;
        buf[16..18].copy_from_slice(&task_count.to_be_bytes());
        let cap_count = self.capability_snapshot.len().min(u16::MAX as usize) as u16;
        buf[18..20].copy_from_slice(&cap_count.to_be_bytes());
        buf[20..24].copy_from_slice(&self.audit_batch_count.to_be_bytes());
    }
}

/// Builder for constructing evidence snapshots from live system state.
pub struct EvidenceCollector {
    snapshot: EvidenceSnapshot,
}

impl EvidenceCollector {
    /// Start collecting evidence for an incident.
    pub fn new(incident_id: u64, timestamp_ns: u64) -> Self {
        Self {
            snapshot: EvidenceSnapshot::new(incident_id, timestamp_ns),
        }
    }

    /// Record a task's state.
    pub fn record_task(
        &mut self,
        task_id: u64,
        state: TaskState,
        capability_count: u32,
        under_containment: bool,
        created_ns: u64,
    ) -> bool {
        self.snapshot.add_task_state(TaskStateEntry {
            task_id,
            state,
            capability_count,
            under_containment,
            created_ns,
        })
    }

    /// Record a capability.
    pub fn record_capability(
        &mut self,
        cap_id: u64,
        owner_task_id: u64,
        resource_type: u8,
        resource_instance: u32,
        permissions: u32,
        created_ns: u64,
    ) -> bool {
        self.snapshot.add_capability(CapabilityEntry {
            cap_id,
            owner_task_id,
            resource_type,
            resource_instance,
            permissions,
            created_ns,
        })
    }

    /// Record memory statistics.
    pub fn record_memory(&mut self, stats: MemorySnapshot) {
        self.snapshot.set_memory_stats(stats);
    }

    /// Record audit batch count.
    pub fn record_audit_batches(&mut self, count: u32) {
        self.snapshot.set_audit_batch_count(count);
    }

    /// Finalize and return the evidence snapshot.
    pub fn finalize(self) -> EvidenceSnapshot {
        self.snapshot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_snapshot() {
        let snap = EvidenceSnapshot::new(1, 1_000_000);
        assert_eq!(snap.incident_id, 1);
        assert_eq!(snap.timestamp_ns, 1_000_000);
        assert!(snap.task_states.is_empty());
        assert!(snap.capability_snapshot.is_empty());
        assert_eq!(snap.audit_batch_count, 0);
        assert_eq!(snap.total_items(), 1); // memory stats always counted
    }

    #[test]
    fn add_task_states() {
        let mut snap = EvidenceSnapshot::new(1, 1000);
        assert!(snap.add_task_state(TaskStateEntry {
            task_id: 42,
            state: TaskState::Running,
            capability_count: 3,
            under_containment: false,
            created_ns: 500,
        }));
        assert_eq!(snap.task_states.len(), 1);
        assert_eq!(snap.task_states[0].task_id, 42);
        assert_eq!(snap.task_states[0].state, TaskState::Running);
    }

    #[test]
    fn add_capabilities() {
        let mut snap = EvidenceSnapshot::new(1, 1000);
        assert!(snap.add_capability(CapabilityEntry {
            cap_id: 10,
            owner_task_id: 42,
            resource_type: 1,
            resource_instance: 0,
            permissions: 0x07,
            created_ns: 500,
        }));
        assert_eq!(snap.capability_snapshot.len(), 1);
    }

    #[test]
    fn task_state_limit() {
        let mut snap = EvidenceSnapshot::new(1, 1000);
        for i in 0..MAX_TASK_ENTRIES {
            assert!(snap.add_task_state(TaskStateEntry {
                task_id: i as u64,
                state: TaskState::Running,
                capability_count: 0,
                under_containment: false,
                created_ns: 0,
            }));
        }
        // Should fail at limit
        assert!(!snap.add_task_state(TaskStateEntry {
            task_id: 999,
            state: TaskState::Running,
            capability_count: 0,
            under_containment: false,
            created_ns: 0,
        }));
    }

    #[test]
    fn capability_limit() {
        let mut snap = EvidenceSnapshot::new(1, 1000);
        for i in 0..MAX_CAP_ENTRIES {
            assert!(snap.add_capability(CapabilityEntry {
                cap_id: i as u64,
                owner_task_id: 0,
                resource_type: 0,
                resource_instance: 0,
                permissions: 0,
                created_ns: 0,
            }));
        }
        assert!(!snap.add_capability(CapabilityEntry {
            cap_id: 999,
            owner_task_id: 0,
            resource_type: 0,
            resource_instance: 0,
            permissions: 0,
            created_ns: 0,
        }));
    }

    #[test]
    fn memory_snapshot_totals() {
        let mem = MemorySnapshot {
            buddy: RegionStats {
                total_bytes: 1024,
                used_bytes: 512,
                active_allocations: 10,
                fragmentation_pct: 5,
            },
            slab: RegionStats {
                total_bytes: 256,
                used_bytes: 128,
                active_allocations: 20,
                fragmentation_pct: 2,
            },
            tensor: RegionStats {
                total_bytes: 2048,
                used_bytes: 1024,
                active_allocations: 5,
                fragmentation_pct: 10,
            },
            total_alloc_failures: 3,
            recent_alloc_failures: 1,
        };
        assert_eq!(mem.total_used(), 512 + 128 + 1024);
        assert_eq!(mem.total_capacity(), 1024 + 256 + 2048);
    }

    #[test]
    fn region_usage_pct() {
        let r = RegionStats {
            total_bytes: 1000,
            used_bytes: 750,
            active_allocations: 0,
            fragmentation_pct: 0,
        };
        assert_eq!(r.usage_pct(), 75);

        let empty = RegionStats::empty();
        assert_eq!(empty.usage_pct(), 0);
    }

    #[test]
    fn task_state_roundtrip() {
        for i in 0..=3u8 {
            let s = TaskState::from_u8(i).unwrap();
            assert_eq!(s as u8, i);
        }
        assert!(TaskState::from_u8(4).is_none());
    }

    #[test]
    fn task_state_as_str() {
        assert_eq!(TaskState::Running.as_str(), "running");
        assert_eq!(TaskState::Blocked.as_str(), "blocked");
        assert_eq!(TaskState::Terminated.as_str(), "terminated");
        assert_eq!(TaskState::Suspended.as_str(), "suspended");
    }

    #[test]
    fn serialize_header() {
        let mut snap = EvidenceSnapshot::new(42, 1_000_000_000);
        snap.add_task_state(TaskStateEntry {
            task_id: 1,
            state: TaskState::Running,
            capability_count: 0,
            under_containment: false,
            created_ns: 0,
        });
        snap.add_capability(CapabilityEntry {
            cap_id: 10,
            owner_task_id: 1,
            resource_type: 0,
            resource_instance: 0,
            permissions: 0,
            created_ns: 0,
        });
        snap.set_audit_batch_count(5);

        let mut buf = [0u8; EVIDENCE_HEADER_SIZE];
        snap.serialize_header(&mut buf);

        assert_eq!(
            u64::from_be_bytes(buf[0..8].try_into().unwrap()),
            1_000_000_000
        );
        assert_eq!(u64::from_be_bytes(buf[8..16].try_into().unwrap()), 42);
        assert_eq!(u16::from_be_bytes(buf[16..18].try_into().unwrap()), 1); // 1 task
        assert_eq!(u16::from_be_bytes(buf[18..20].try_into().unwrap()), 1); // 1 cap
        assert_eq!(u32::from_be_bytes(buf[20..24].try_into().unwrap()), 5); // 5 batches
    }

    #[test]
    fn evidence_collector_workflow() {
        let mut collector = EvidenceCollector::new(1, 5_000_000);

        assert!(collector.record_task(10, TaskState::Running, 3, false, 1000));
        assert!(collector.record_task(20, TaskState::Blocked, 1, true, 2000));

        assert!(collector.record_capability(100, 10, 1, 0, 0x07, 1000));

        collector.record_memory(MemorySnapshot {
            buddy: RegionStats {
                total_bytes: 1024,
                used_bytes: 512,
                active_allocations: 5,
                fragmentation_pct: 3,
            },
            slab: RegionStats::empty(),
            tensor: RegionStats::empty(),
            total_alloc_failures: 0,
            recent_alloc_failures: 0,
        });

        collector.record_audit_batches(10);

        let snap = collector.finalize();
        assert_eq!(snap.incident_id, 1);
        assert_eq!(snap.task_states.len(), 2);
        assert_eq!(snap.capability_snapshot.len(), 1);
        assert_eq!(snap.memory_stats.buddy.used_bytes, 512);
        assert_eq!(snap.audit_batch_count, 10);
        assert_eq!(snap.total_items(), 4); // 2 tasks + 1 cap + 1 mem
    }
}
