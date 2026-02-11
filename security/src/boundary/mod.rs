// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Security boundary and information flow enforcement.
//!
//! Enforces mandatory information flow rules via the capability system:
//! - ONNX runtime SHALL NOT access network resources
//! - IPC router SHALL NOT access GPU resources
//! - Bus protocol handlers SHALL NOT access ONNX model data
//!
//! The flow matrix is defined at build time and is immutable at runtime.
//! Every capability grant is checked against the matrix before approval.

pub mod attack_surface;
pub mod data_flow_auth;
pub mod trust_boundaries;

use crate::capability::ResourceType;

/// Task types in the system for information flow classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TaskType {
    /// ONNX inference runtime.
    OnnxRuntime = 0,
    /// IPC router (Zenoh pub/sub).
    IpcRouter = 1,
    /// Bus protocol handler (CAN, ARINC, 1553, SpaceWire, CCSDS).
    BusHandler = 2,
    /// Network stack (TCP/IP, TLS).
    NetworkStack = 3,
    /// GPU compute engine.
    GpuEngine = 4,
    /// System management / operator.
    SystemManager = 5,
    /// Audit subsystem.
    AuditSubsystem = 6,
}

impl TaskType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::OnnxRuntime),
            1 => Some(Self::IpcRouter),
            2 => Some(Self::BusHandler),
            3 => Some(Self::NetworkStack),
            4 => Some(Self::GpuEngine),
            5 => Some(Self::SystemManager),
            6 => Some(Self::AuditSubsystem),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::OnnxRuntime => "onnx-runtime",
            Self::IpcRouter => "ipc-router",
            Self::BusHandler => "bus-handler",
            Self::NetworkStack => "network-stack",
            Self::GpuEngine => "gpu-engine",
            Self::SystemManager => "system-manager",
            Self::AuditSubsystem => "audit-subsystem",
        }
    }
}

/// Number of task types.
const NUM_TASK_TYPES: usize = 7;

/// Number of resource types (from capability.rs).
const NUM_RESOURCE_TYPES: usize = 8;

/// Result of an information flow check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowCheckResult {
    /// Access is allowed by the flow matrix.
    Allowed,
    /// Access is denied by the flow matrix.
    Denied {
        task_type: TaskType,
        resource_type: ResourceType,
    },
}

impl FlowCheckResult {
    pub fn is_allowed(&self) -> bool {
        matches!(self, Self::Allowed)
    }
}

/// Information flow enforcement matrix.
///
/// `matrix[task_type][resource_type]` = true if the task type
/// is allowed to hold capabilities for that resource type.
///
/// The matrix is immutable after initialization (build-time policy).
pub struct FlowMatrix {
    /// Access matrix: [task_type][resource_type] = allowed.
    matrix: [[bool; NUM_RESOURCE_TYPES]; NUM_TASK_TYPES],
    /// Total flow checks performed.
    total_checks: u64,
    /// Total denials.
    total_denials: u64,
}

impl FlowMatrix {
    /// Create a new flow matrix with default policy.
    ///
    /// Default policy per spec:
    /// - ONNX runtime: TensorBuffer, OnnxModel, IpcEndpoint (NO network, NO device)
    /// - IPC router: IpcEndpoint, NetworkSocket (NO GPU, NO OnnxModel)
    /// - Bus handler: Device, IpcEndpoint (NO OnnxModel, NO TensorBuffer, NO GPU)
    /// - Network stack: NetworkSocket, IpcEndpoint
    /// - GPU engine: GpuDevice, TensorBuffer
    /// - System manager: all resources (full access)
    /// - Audit subsystem: IpcEndpoint, SystemConfig
    pub fn with_default_policy() -> Self {
        let mut matrix = [[false; NUM_RESOURCE_TYPES]; NUM_TASK_TYPES];

        // ONNX runtime: tensor buffers, ONNX models, IPC
        matrix[TaskType::OnnxRuntime as usize][ResourceType::TensorBuffer as usize] = true;
        matrix[TaskType::OnnxRuntime as usize][ResourceType::OnnxModel as usize] = true;
        matrix[TaskType::OnnxRuntime as usize][ResourceType::IpcEndpoint as usize] = true;
        matrix[TaskType::OnnxRuntime as usize][ResourceType::GpuDevice as usize] = true;
        // ONNX runtime: NO NetworkSocket, NO SystemConfig, NO SystemControl, NO Device

        // IPC router: IPC endpoints, network
        matrix[TaskType::IpcRouter as usize][ResourceType::IpcEndpoint as usize] = true;
        matrix[TaskType::IpcRouter as usize][ResourceType::NetworkSocket as usize] = true;
        // IPC router: NO GpuDevice, NO OnnxModel, NO TensorBuffer

        // Bus handler: device, IPC
        matrix[TaskType::BusHandler as usize][ResourceType::Device as usize] = true;
        matrix[TaskType::BusHandler as usize][ResourceType::IpcEndpoint as usize] = true;
        // Bus handler: NO OnnxModel, NO TensorBuffer, NO GpuDevice, NO NetworkSocket

        // Network stack: network, IPC
        matrix[TaskType::NetworkStack as usize][ResourceType::NetworkSocket as usize] = true;
        matrix[TaskType::NetworkStack as usize][ResourceType::IpcEndpoint as usize] = true;

        // GPU engine: GPU device, tensor buffers
        matrix[TaskType::GpuEngine as usize][ResourceType::GpuDevice as usize] = true;
        matrix[TaskType::GpuEngine as usize][ResourceType::TensorBuffer as usize] = true;

        // System manager: full access (all resource types)
        for slot in matrix[TaskType::SystemManager as usize].iter_mut() {
            *slot = true;
        }

        // Audit subsystem: IPC, system config
        matrix[TaskType::AuditSubsystem as usize][ResourceType::IpcEndpoint as usize] = true;
        matrix[TaskType::AuditSubsystem as usize][ResourceType::SystemConfig as usize] = true;

        Self {
            matrix,
            total_checks: 0,
            total_denials: 0,
        }
    }

    /// Check whether a task type is allowed to access a resource type.
    pub fn check_flow(
        &mut self,
        task_type: TaskType,
        resource_type: ResourceType,
    ) -> FlowCheckResult {
        self.total_checks += 1;

        let allowed = self.matrix[task_type as usize][resource_type as usize];
        if allowed {
            FlowCheckResult::Allowed
        } else {
            self.total_denials += 1;
            FlowCheckResult::Denied {
                task_type,
                resource_type,
            }
        }
    }

    /// Check without mutating counters (for read-only queries).
    pub fn is_allowed(&self, task_type: TaskType, resource_type: ResourceType) -> bool {
        self.matrix[task_type as usize][resource_type as usize]
    }

    /// Total flow checks performed.
    pub fn total_checks(&self) -> u64 {
        self.total_checks
    }

    /// Total denials.
    pub fn total_denials(&self) -> u64 {
        self.total_denials
    }

    /// List all denied flows for a task type.
    pub fn denied_resources(&self, task_type: TaskType) -> DeniedResourceIter {
        DeniedResourceIter {
            matrix_row: self.matrix[task_type as usize],
            pos: 0,
        }
    }
}

impl Default for FlowMatrix {
    fn default() -> Self {
        Self::with_default_policy()
    }
}

/// Iterator over denied resource types for a task type.
pub struct DeniedResourceIter {
    matrix_row: [bool; NUM_RESOURCE_TYPES],
    pos: usize,
}

impl Iterator for DeniedResourceIter {
    type Item = ResourceType;

    fn next(&mut self) -> Option<Self::Item> {
        while self.pos < NUM_RESOURCE_TYPES {
            let idx = self.pos;
            self.pos += 1;
            if !self.matrix_row[idx] {
                if let Some(rt) = ResourceType::from_u8(idx as u8) {
                    return Some(rt);
                }
            }
        }
        None
    }
}

/// Task registration for flow enforcement.
///
/// Maps task IDs to task types so the flow matrix can be applied
/// to capability grant requests.
pub struct TaskRegistry {
    /// (task_id, task_type) pairs.
    entries: [(u64, TaskType, bool); MAX_TASKS],
}

const MAX_TASKS: usize = 128;

impl TaskRegistry {
    pub fn new() -> Self {
        Self {
            entries: [(0, TaskType::OnnxRuntime, false); MAX_TASKS],
        }
    }

    /// Register a task with its type.
    pub fn register(&mut self, task_id: u64, task_type: TaskType) -> bool {
        if let Some(slot) = self.entries.iter_mut().find(|e| !e.2) {
            *slot = (task_id, task_type, true);
            true
        } else {
            false
        }
    }

    /// Unregister a task.
    pub fn unregister(&mut self, task_id: u64) -> bool {
        if let Some(slot) = self.entries.iter_mut().find(|e| e.2 && e.0 == task_id) {
            slot.2 = false;
            true
        } else {
            false
        }
    }

    /// Look up the task type for a task ID.
    pub fn task_type(&self, task_id: u64) -> Option<TaskType> {
        self.entries
            .iter()
            .find(|e| e.2 && e.0 == task_id)
            .map(|e| e.1)
    }

    /// Number of registered tasks.
    pub fn count(&self) -> usize {
        self.entries.iter().filter(|e| e.2).count()
    }
}

impl Default for TaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn task_type_roundtrip() {
        for i in 0..=6u8 {
            let tt = TaskType::from_u8(i).unwrap();
            assert_eq!(tt as u8, i);
        }
        assert!(TaskType::from_u8(7).is_none());
    }

    #[test]
    fn task_type_as_str() {
        assert_eq!(TaskType::OnnxRuntime.as_str(), "onnx-runtime");
        assert_eq!(TaskType::IpcRouter.as_str(), "ipc-router");
        assert_eq!(TaskType::BusHandler.as_str(), "bus-handler");
        assert_eq!(TaskType::SystemManager.as_str(), "system-manager");
    }

    // ── ONNX runtime isolation ──

    #[test]
    fn onnx_cannot_access_network() {
        let mut fm = FlowMatrix::with_default_policy();
        let result = fm.check_flow(TaskType::OnnxRuntime, ResourceType::NetworkSocket);
        assert!(!result.is_allowed());
        assert_eq!(fm.total_denials(), 1);
    }

    #[test]
    fn onnx_can_access_tensor() {
        let mut fm = FlowMatrix::with_default_policy();
        let result = fm.check_flow(TaskType::OnnxRuntime, ResourceType::TensorBuffer);
        assert!(result.is_allowed());
    }

    #[test]
    fn onnx_can_access_model() {
        let mut fm = FlowMatrix::with_default_policy();
        let result = fm.check_flow(TaskType::OnnxRuntime, ResourceType::OnnxModel);
        assert!(result.is_allowed());
    }

    #[test]
    fn onnx_can_access_gpu() {
        let mut fm = FlowMatrix::with_default_policy();
        let result = fm.check_flow(TaskType::OnnxRuntime, ResourceType::GpuDevice);
        assert!(result.is_allowed());
    }

    #[test]
    fn onnx_cannot_access_device() {
        let mut fm = FlowMatrix::with_default_policy();
        let result = fm.check_flow(TaskType::OnnxRuntime, ResourceType::Device);
        assert!(!result.is_allowed());
    }

    // ── IPC router isolation ──

    #[test]
    fn ipc_cannot_access_gpu() {
        let mut fm = FlowMatrix::with_default_policy();
        let result = fm.check_flow(TaskType::IpcRouter, ResourceType::GpuDevice);
        assert!(!result.is_allowed());
    }

    #[test]
    fn ipc_cannot_access_onnx() {
        let mut fm = FlowMatrix::with_default_policy();
        let result = fm.check_flow(TaskType::IpcRouter, ResourceType::OnnxModel);
        assert!(!result.is_allowed());
    }

    #[test]
    fn ipc_can_access_network() {
        let mut fm = FlowMatrix::with_default_policy();
        let result = fm.check_flow(TaskType::IpcRouter, ResourceType::NetworkSocket);
        assert!(result.is_allowed());
    }

    #[test]
    fn ipc_can_access_ipc() {
        let mut fm = FlowMatrix::with_default_policy();
        let result = fm.check_flow(TaskType::IpcRouter, ResourceType::IpcEndpoint);
        assert!(result.is_allowed());
    }

    // ── Bus handler isolation ──

    #[test]
    fn bus_cannot_access_onnx() {
        let mut fm = FlowMatrix::with_default_policy();
        let result = fm.check_flow(TaskType::BusHandler, ResourceType::OnnxModel);
        assert!(!result.is_allowed());
    }

    #[test]
    fn bus_cannot_access_tensor() {
        let mut fm = FlowMatrix::with_default_policy();
        let result = fm.check_flow(TaskType::BusHandler, ResourceType::TensorBuffer);
        assert!(!result.is_allowed());
    }

    #[test]
    fn bus_cannot_access_gpu() {
        let mut fm = FlowMatrix::with_default_policy();
        let result = fm.check_flow(TaskType::BusHandler, ResourceType::GpuDevice);
        assert!(!result.is_allowed());
    }

    #[test]
    fn bus_can_access_device() {
        let mut fm = FlowMatrix::with_default_policy();
        let result = fm.check_flow(TaskType::BusHandler, ResourceType::Device);
        assert!(result.is_allowed());
    }

    #[test]
    fn bus_can_access_ipc() {
        let mut fm = FlowMatrix::with_default_policy();
        let result = fm.check_flow(TaskType::BusHandler, ResourceType::IpcEndpoint);
        assert!(result.is_allowed());
    }

    // ── System manager full access ──

    #[test]
    fn system_manager_full_access() {
        let mut fm = FlowMatrix::with_default_policy();
        for rt_val in 0..=7u8 {
            if let Some(rt) = ResourceType::from_u8(rt_val) {
                let result = fm.check_flow(TaskType::SystemManager, rt);
                assert!(result.is_allowed(), "SystemManager should access {:?}", rt);
            }
        }
    }

    // ── Denied resource iterator ──

    #[test]
    fn denied_resources_for_onnx() {
        let fm = FlowMatrix::with_default_policy();
        let denied: alloc::vec::Vec<_> = fm.denied_resources(TaskType::OnnxRuntime).collect();
        assert!(denied.contains(&ResourceType::NetworkSocket));
        assert!(denied.contains(&ResourceType::Device));
        assert!(!denied.contains(&ResourceType::TensorBuffer));
        assert!(!denied.contains(&ResourceType::OnnxModel));
    }

    #[test]
    fn denied_resources_for_system_manager() {
        let fm = FlowMatrix::with_default_policy();
        let denied: alloc::vec::Vec<_> = fm.denied_resources(TaskType::SystemManager).collect();
        assert!(denied.is_empty());
    }

    // ── Counter tracking ──

    #[test]
    fn check_counters() {
        let mut fm = FlowMatrix::with_default_policy();
        fm.check_flow(TaskType::OnnxRuntime, ResourceType::TensorBuffer);
        fm.check_flow(TaskType::OnnxRuntime, ResourceType::NetworkSocket);
        fm.check_flow(TaskType::IpcRouter, ResourceType::GpuDevice);
        assert_eq!(fm.total_checks(), 3);
        assert_eq!(fm.total_denials(), 2);
    }

    // ── Task registry ──

    #[test]
    fn register_and_lookup() {
        let mut reg = TaskRegistry::new();
        assert!(reg.register(42, TaskType::OnnxRuntime));
        assert_eq!(reg.task_type(42), Some(TaskType::OnnxRuntime));
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn unregister_task() {
        let mut reg = TaskRegistry::new();
        reg.register(42, TaskType::OnnxRuntime);
        assert!(reg.unregister(42));
        assert!(reg.task_type(42).is_none());
        assert_eq!(reg.count(), 0);
    }

    #[test]
    fn unregister_nonexistent() {
        let mut reg = TaskRegistry::new();
        assert!(!reg.unregister(99));
    }

    #[test]
    fn lookup_nonexistent() {
        let reg = TaskRegistry::new();
        assert!(reg.task_type(99).is_none());
    }

    #[test]
    fn multiple_tasks() {
        let mut reg = TaskRegistry::new();
        reg.register(1, TaskType::OnnxRuntime);
        reg.register(2, TaskType::IpcRouter);
        reg.register(3, TaskType::BusHandler);
        assert_eq!(reg.count(), 3);
        assert_eq!(reg.task_type(1), Some(TaskType::OnnxRuntime));
        assert_eq!(reg.task_type(2), Some(TaskType::IpcRouter));
        assert_eq!(reg.task_type(3), Some(TaskType::BusHandler));
    }

    // ── Integration: registry + flow matrix ──

    #[test]
    fn registry_with_flow_check() {
        let mut reg = TaskRegistry::new();
        let mut fm = FlowMatrix::with_default_policy();

        reg.register(100, TaskType::OnnxRuntime);
        reg.register(200, TaskType::BusHandler);

        // ONNX runtime task tries to access network -> denied
        if let Some(tt) = reg.task_type(100) {
            let result = fm.check_flow(tt, ResourceType::NetworkSocket);
            assert!(!result.is_allowed());
        }

        // Bus handler tries to access ONNX model -> denied
        if let Some(tt) = reg.task_type(200) {
            let result = fm.check_flow(tt, ResourceType::OnnxModel);
            assert!(!result.is_allowed());
        }

        // ONNX runtime accesses tensor -> allowed
        if let Some(tt) = reg.task_type(100) {
            let result = fm.check_flow(tt, ResourceType::TensorBuffer);
            assert!(result.is_allowed());
        }
    }
}
