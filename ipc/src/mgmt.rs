// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS Management API — Zenoh endpoints for Kubernetes integration.
//!
//! Defines the management API that a Virtual Kubelet provider uses to deploy
//! ONNX inference workloads onto SmallAIOS instances. Endpoints follow the
//! Zenoh key-expression convention under `sai/mgmt/`.
//!
//! Endpoints:
//! - `sai/mgmt/deploy`   — deploy an ONNX model
//! - `sai/mgmt/undeploy` — undeploy a running model
//! - `sai/mgmt/status`   — query workload status
//! - `sai/mgmt/config`   — update runtime configuration
//! - `sai/mgmt/resources`— report node resources (CPU, memory, GPU)

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

use crate::IpcError;
use crate::key_expr::KeyExpr;

/// Maximum number of deployed workloads.
const MAX_WORKLOADS: usize = 32;

/// Maximum number of resource entries.
const MAX_RESOURCES: usize = 16;

// ---------------------------------------------------------------------------
// Management API key expressions
// ---------------------------------------------------------------------------

/// Base prefix for all management endpoints.
pub const MGMT_PREFIX: &str = "sai/mgmt";

/// Key expression for model deployment.
pub const MGMT_DEPLOY: &str = "sai/mgmt/deploy";

/// Key expression for model undeployment.
pub const MGMT_UNDEPLOY: &str = "sai/mgmt/undeploy";

/// Key expression for workload status queries.
pub const MGMT_STATUS: &str = "sai/mgmt/status";

/// Key expression for runtime configuration updates.
pub const MGMT_CONFIG: &str = "sai/mgmt/config";

/// Key expression for node resource reporting.
pub const MGMT_RESOURCES: &str = "sai/mgmt/resources";

// ---------------------------------------------------------------------------
// Deploy request/response
// ---------------------------------------------------------------------------

/// Request to deploy an ONNX model workload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeployRequest {
    /// Unique workload identifier (maps to K8s pod UID).
    pub workload_id: String,
    /// URL or path to the ONNX model.
    pub model_url: String,
    /// Human-readable workload name (maps to K8s pod name).
    pub name: String,
    /// Namespace (maps to K8s namespace).
    pub namespace: String,
    /// Environment variables / configuration key-value pairs.
    pub env: Vec<(String, String)>,
    /// Requested CPU millicores (e.g. 1000 = 1 core).
    pub cpu_millis: u32,
    /// Requested memory in bytes.
    pub memory_bytes: u64,
    /// Whether a GPU is required.
    pub gpu_required: bool,
}

impl DeployRequest {
    /// Encode the deploy request into a minimal JSON-like wire format.
    ///
    /// Format: `workload_id\0model_url\0name\0namespace\0cpu_millis\0memory_bytes\0gpu_required`
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        Self::push_str(&mut buf, &self.workload_id);
        Self::push_str(&mut buf, &self.model_url);
        Self::push_str(&mut buf, &self.name);
        Self::push_str(&mut buf, &self.namespace);
        buf.extend_from_slice(&self.cpu_millis.to_le_bytes());
        buf.extend_from_slice(&self.memory_bytes.to_le_bytes());
        buf.push(if self.gpu_required { 1 } else { 0 });
        // Encode env count + key/value pairs
        buf.extend_from_slice(&(self.env.len() as u16).to_le_bytes());
        for (k, v) in &self.env {
            Self::push_str(&mut buf, k);
            Self::push_str(&mut buf, v);
        }
        buf
    }

    /// Decode a deploy request from wire format.
    pub fn decode(data: &[u8]) -> Result<Self, IpcError> {
        let mut pos = 0;
        let workload_id = Self::read_str(data, &mut pos)?;
        let model_url = Self::read_str(data, &mut pos)?;
        let name = Self::read_str(data, &mut pos)?;
        let namespace = Self::read_str(data, &mut pos)?;

        if pos + 4 > data.len() {
            return Err(IpcError::PacketTooShort);
        }
        let cpu_millis = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
        pos += 4;

        if pos + 8 > data.len() {
            return Err(IpcError::PacketTooShort);
        }
        let memory_bytes = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        if pos >= data.len() {
            return Err(IpcError::PacketTooShort);
        }
        let gpu_required = data[pos] != 0;
        pos += 1;

        let mut env = Vec::new();
        if pos + 2 <= data.len() {
            let env_count = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
            pos += 2;
            for _ in 0..env_count {
                let k = Self::read_str(data, &mut pos)?;
                let v = Self::read_str(data, &mut pos)?;
                env.push((k, v));
            }
        }

        Ok(Self {
            workload_id,
            model_url,
            name,
            namespace,
            env,
            cpu_millis,
            memory_bytes,
            gpu_required,
        })
    }

    fn push_str(buf: &mut Vec<u8>, s: &str) {
        let len = s.len() as u16;
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn read_str(data: &[u8], pos: &mut usize) -> Result<String, IpcError> {
        if *pos + 2 > data.len() {
            return Err(IpcError::PacketTooShort);
        }
        let len = u16::from_le_bytes(data[*pos..*pos + 2].try_into().unwrap()) as usize;
        *pos += 2;
        if *pos + len > data.len() {
            return Err(IpcError::PacketTooShort);
        }
        let s = core::str::from_utf8(&data[*pos..*pos + len])
            .map_err(|_| IpcError::InvalidHeader)?;
        *pos += len;
        Ok(String::from(s))
    }
}

/// Response to a deploy request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeployResponse {
    /// Deployment accepted.
    Accepted { workload_id: String },
    /// Deployment rejected with reason.
    Rejected { workload_id: String, reason: String },
}

// ---------------------------------------------------------------------------
// Undeploy request/response
// ---------------------------------------------------------------------------

/// Request to undeploy a workload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UndeployRequest {
    /// Workload ID to undeploy.
    pub workload_id: String,
    /// Whether to force-kill (true) or gracefully stop (false).
    pub force: bool,
}

impl UndeployRequest {
    /// Encode.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        let len = self.workload_id.len() as u16;
        buf.extend_from_slice(&len.to_le_bytes());
        buf.extend_from_slice(self.workload_id.as_bytes());
        buf.push(if self.force { 1 } else { 0 });
        buf
    }

    /// Decode.
    pub fn decode(data: &[u8]) -> Result<Self, IpcError> {
        let mut pos = 0;
        if pos + 2 > data.len() {
            return Err(IpcError::PacketTooShort);
        }
        let len = u16::from_le_bytes(data[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if pos + len > data.len() {
            return Err(IpcError::PacketTooShort);
        }
        let workload_id = core::str::from_utf8(&data[pos..pos + len])
            .map_err(|_| IpcError::InvalidHeader)?;
        pos += len;
        if pos >= data.len() {
            return Err(IpcError::PacketTooShort);
        }
        let force = data[pos] != 0;
        Ok(Self {
            workload_id: String::from(workload_id),
            force,
        })
    }
}

// ---------------------------------------------------------------------------
// Workload status
// ---------------------------------------------------------------------------

/// Workload lifecycle phase (maps to K8s pod phases).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadPhase {
    /// Workload is accepted but not yet running.
    Pending,
    /// Model is being downloaded / loaded.
    Loading,
    /// Workload is actively serving inference requests.
    Running,
    /// Workload completed successfully (batch inference).
    Succeeded,
    /// Workload terminated with an error.
    Failed,
    /// Workload is being torn down.
    Terminating,
}

impl WorkloadPhase {
    /// Map to a K8s-compatible pod phase string.
    pub fn as_k8s_phase(&self) -> &'static str {
        match self {
            WorkloadPhase::Pending => "Pending",
            WorkloadPhase::Loading => "Pending",
            WorkloadPhase::Running => "Running",
            WorkloadPhase::Succeeded => "Succeeded",
            WorkloadPhase::Failed => "Failed",
            WorkloadPhase::Terminating => "Running",
        }
    }
}

/// Status of a single deployed workload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkloadStatus {
    /// Workload ID.
    pub workload_id: String,
    /// Human-readable name.
    pub name: String,
    /// Namespace.
    pub namespace: String,
    /// Current phase.
    pub phase: WorkloadPhase,
    /// Status message (human-readable, e.g. error details).
    pub message: String,
    /// Total inference requests served.
    pub inference_count: u64,
    /// Uptime in milliseconds since deployment.
    pub uptime_ms: u64,
}

// ---------------------------------------------------------------------------
// Node resource reporting
// ---------------------------------------------------------------------------

/// A single resource capacity entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceEntry {
    /// Resource name (e.g. "cpu", "memory", "nvidia.com/gpu").
    pub name: String,
    /// Total capacity (in resource-specific units).
    pub capacity: u64,
    /// Currently allocatable (capacity - used).
    pub allocatable: u64,
}

/// Node resource report for Kubernetes node advertisement.
#[derive(Debug, Clone)]
pub struct NodeResources {
    /// Resource entries.
    entries: Vec<ResourceEntry>,
}

impl Default for NodeResources {
    fn default() -> Self {
        Self::new()
    }
}

impl NodeResources {
    /// Create a new empty resource report.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Add a resource entry.
    pub fn add(&mut self, name: &str, capacity: u64, allocatable: u64) -> Result<(), IpcError> {
        if self.entries.len() >= MAX_RESOURCES {
            return Err(IpcError::TableFull);
        }
        // Update if already exists
        for entry in &mut self.entries {
            if entry.name == name {
                entry.capacity = capacity;
                entry.allocatable = allocatable;
                return Ok(());
            }
        }
        self.entries.push(ResourceEntry {
            name: String::from(name),
            capacity,
            allocatable,
        });
        Ok(())
    }

    /// Get a resource entry by name.
    pub fn get(&self, name: &str) -> Option<&ResourceEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Returns the number of resource entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if there are no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns all entries.
    pub fn entries(&self) -> &[ResourceEntry] {
        &self.entries
    }

    /// Create a standard node resource report with CPU, memory, and optional GPU.
    pub fn standard(cpu_millis: u64, memory_bytes: u64, gpu_count: u64) -> Self {
        let mut res = Self::new();
        // CPU in millicores
        let _ = res.add("cpu", cpu_millis, cpu_millis);
        // Memory in bytes
        let _ = res.add("memory", memory_bytes, memory_bytes);
        // GPU as extended resource
        if gpu_count > 0 {
            let _ = res.add("nvidia.com/gpu", gpu_count, gpu_count);
        }
        res
    }

    /// Produce a minimal JSON resource report.
    ///
    /// Example: `{"cpu":{"capacity":4000,"allocatable":3500},"memory":{...}}`
    pub fn to_json(&self) -> String {
        let mut s = String::from("{");
        for (i, entry) in self.entries.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push('"');
            s.push_str(&entry.name);
            s.push_str("\":{\"capacity\":");
            push_u64(&mut s, entry.capacity);
            s.push_str(",\"allocatable\":");
            push_u64(&mut s, entry.allocatable);
            s.push('}');
        }
        s.push('}');
        s
    }
}

// ---------------------------------------------------------------------------
// Management API handler
// ---------------------------------------------------------------------------

/// Management API handler — processes requests on `sai/mgmt/*` endpoints.
///
/// This handler maintains the workload registry and node resources, and
/// provides the Rust-side implementation that a Virtual Kubelet provider
/// communicates with over Zenoh.
#[derive(Debug)]
pub struct ManagementApi {
    /// Deployed workloads.
    workloads: Vec<WorkloadStatus>,
    /// Node resource report.
    resources: NodeResources,
    /// Key expressions for the management endpoints.
    deploy_key: KeyExpr,
    undeploy_key: KeyExpr,
    status_key: KeyExpr,
    config_key: KeyExpr,
    resources_key: KeyExpr,
}

impl ManagementApi {
    /// Create a new management API handler.
    pub fn new(resources: NodeResources) -> Result<Self, IpcError> {
        Ok(Self {
            workloads: Vec::new(),
            resources,
            deploy_key: KeyExpr::new(MGMT_DEPLOY).map_err(|_| IpcError::InvalidState)?,
            undeploy_key: KeyExpr::new(MGMT_UNDEPLOY).map_err(|_| IpcError::InvalidState)?,
            status_key: KeyExpr::new(MGMT_STATUS).map_err(|_| IpcError::InvalidState)?,
            config_key: KeyExpr::new(MGMT_CONFIG).map_err(|_| IpcError::InvalidState)?,
            resources_key: KeyExpr::new(MGMT_RESOURCES).map_err(|_| IpcError::InvalidState)?,
        })
    }

    /// Handle a deploy request.
    pub fn handle_deploy(&mut self, req: &DeployRequest) -> DeployResponse {
        // Check capacity
        if self.workloads.len() >= MAX_WORKLOADS {
            return DeployResponse::Rejected {
                workload_id: req.workload_id.clone(),
                reason: String::from("maximum workload capacity reached"),
            };
        }

        // Check for duplicate
        if self.workloads.iter().any(|w| w.workload_id == req.workload_id) {
            return DeployResponse::Rejected {
                workload_id: req.workload_id.clone(),
                reason: String::from("workload already deployed"),
            };
        }

        // Check resource availability
        if req.gpu_required {
            if let Some(gpu) = self.resources.get("nvidia.com/gpu") {
                if gpu.allocatable == 0 {
                    return DeployResponse::Rejected {
                        workload_id: req.workload_id.clone(),
                        reason: String::from("no GPU available"),
                    };
                }
            } else {
                return DeployResponse::Rejected {
                    workload_id: req.workload_id.clone(),
                    reason: String::from("no GPU resource on node"),
                };
            }
        }

        // Create workload entry
        self.workloads.push(WorkloadStatus {
            workload_id: req.workload_id.clone(),
            name: req.name.clone(),
            namespace: req.namespace.clone(),
            phase: WorkloadPhase::Pending,
            message: String::from("accepted, awaiting model load"),
            inference_count: 0,
            uptime_ms: 0,
        });

        DeployResponse::Accepted {
            workload_id: req.workload_id.clone(),
        }
    }

    /// Handle an undeploy request.
    pub fn handle_undeploy(&mut self, req: &UndeployRequest) -> Result<(), IpcError> {
        let pos = self.workloads.iter().position(|w| w.workload_id == req.workload_id)
            .ok_or(IpcError::NotFound)?;
        if req.force {
            self.workloads.remove(pos);
        } else {
            self.workloads[pos].phase = WorkloadPhase::Terminating;
            self.workloads[pos].message = String::from("graceful shutdown initiated");
        }
        Ok(())
    }

    /// Query status of a specific workload (or all if workload_id is empty).
    pub fn query_status(&self, workload_id: &str) -> Vec<&WorkloadStatus> {
        if workload_id.is_empty() {
            self.workloads.iter().collect()
        } else {
            self.workloads
                .iter()
                .filter(|w| w.workload_id == workload_id)
                .collect()
        }
    }

    /// Update a workload's phase.
    pub fn update_phase(
        &mut self,
        workload_id: &str,
        phase: WorkloadPhase,
        message: &str,
    ) -> Result<(), IpcError> {
        let w = self.workloads.iter_mut()
            .find(|w| w.workload_id == workload_id)
            .ok_or(IpcError::NotFound)?;
        w.phase = phase;
        w.message = String::from(message);
        Ok(())
    }

    /// Increment inference count for a workload.
    pub fn record_inference(&mut self, workload_id: &str) -> Result<(), IpcError> {
        let w = self.workloads.iter_mut()
            .find(|w| w.workload_id == workload_id)
            .ok_or(IpcError::NotFound)?;
        w.inference_count += 1;
        Ok(())
    }

    /// Update workload uptime.
    pub fn update_uptime(&mut self, workload_id: &str, uptime_ms: u64) -> Result<(), IpcError> {
        let w = self.workloads.iter_mut()
            .find(|w| w.workload_id == workload_id)
            .ok_or(IpcError::NotFound)?;
        w.uptime_ms = uptime_ms;
        Ok(())
    }

    /// Finalize and remove terminated workloads.
    pub fn cleanup_terminated(&mut self) -> usize {
        let before = self.workloads.len();
        self.workloads.retain(|w| w.phase != WorkloadPhase::Terminating);
        before - self.workloads.len()
    }

    /// Returns the number of deployed workloads.
    pub fn workload_count(&self) -> usize {
        self.workloads.len()
    }

    /// Returns the number of running workloads.
    pub fn running_count(&self) -> usize {
        self.workloads.iter().filter(|w| w.phase == WorkloadPhase::Running).count()
    }

    /// Returns node resources.
    pub fn resources(&self) -> &NodeResources {
        &self.resources
    }

    /// Returns a mutable reference to node resources (for updating allocatable).
    pub fn resources_mut(&mut self) -> &mut NodeResources {
        &mut self.resources
    }

    /// Returns the deploy key expression.
    pub fn deploy_key(&self) -> &KeyExpr {
        &self.deploy_key
    }

    /// Returns the undeploy key expression.
    pub fn undeploy_key(&self) -> &KeyExpr {
        &self.undeploy_key
    }

    /// Returns the status key expression.
    pub fn status_key(&self) -> &KeyExpr {
        &self.status_key
    }

    /// Returns the config key expression.
    pub fn config_key(&self) -> &KeyExpr {
        &self.config_key
    }

    /// Returns the resources key expression.
    pub fn resources_key(&self) -> &KeyExpr {
        &self.resources_key
    }

    /// Produce a JSON status summary for all workloads.
    pub fn to_status_json(&self) -> String {
        let mut s = String::from("[");
        for (i, w) in self.workloads.iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            s.push_str("{\"id\":\"");
            s.push_str(&w.workload_id);
            s.push_str("\",\"name\":\"");
            s.push_str(&w.name);
            s.push_str("\",\"phase\":\"");
            s.push_str(w.phase.as_k8s_phase());
            s.push_str("\",\"inferences\":");
            push_u64(&mut s, w.inference_count);
            s.push('}');
        }
        s.push(']');
        s
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn push_u64(s: &mut String, n: u64) {
    if n == 0 {
        s.push('0');
        return;
    }
    let mut buf = [0u8; 20];
    let mut pos = buf.len();
    let mut val = n;
    while val > 0 {
        pos -= 1;
        buf[pos] = b'0' + (val % 10) as u8;
        val /= 10;
    }
    for &b in &buf[pos..] {
        s.push(b as char);
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_api() -> ManagementApi {
        let resources = NodeResources::standard(4000, 8 * 1024 * 1024 * 1024, 1);
        ManagementApi::new(resources).unwrap()
    }

    fn sample_deploy_req() -> DeployRequest {
        DeployRequest {
            workload_id: String::from("pod-uid-123"),
            model_url: String::from("https://models.example.com/resnet50.onnx"),
            name: String::from("resnet50-inference"),
            namespace: String::from("default"),
            env: alloc::vec![
                (String::from("BATCH_SIZE"), String::from("4")),
                (String::from("PRECISION"), String::from("fp16")),
            ],
            cpu_millis: 1000,
            memory_bytes: 512 * 1024 * 1024,
            gpu_required: false,
        }
    }

    // --- Key expressions ---

    #[test]
    fn test_mgmt_key_expressions() {
        let api = make_api();
        assert_eq!(api.deploy_key().as_str(), "sai/mgmt/deploy");
        assert_eq!(api.undeploy_key().as_str(), "sai/mgmt/undeploy");
        assert_eq!(api.status_key().as_str(), "sai/mgmt/status");
        assert_eq!(api.config_key().as_str(), "sai/mgmt/config");
        assert_eq!(api.resources_key().as_str(), "sai/mgmt/resources");
    }

    // --- Deploy ---

    #[test]
    fn test_deploy_success() {
        let mut api = make_api();
        let req = sample_deploy_req();
        let resp = api.handle_deploy(&req);
        assert_eq!(resp, DeployResponse::Accepted { workload_id: String::from("pod-uid-123") });
        assert_eq!(api.workload_count(), 1);
    }

    #[test]
    fn test_deploy_duplicate_rejected() {
        let mut api = make_api();
        let req = sample_deploy_req();
        api.handle_deploy(&req);
        let resp = api.handle_deploy(&req);
        match resp {
            DeployResponse::Rejected { reason, .. } => {
                assert!(reason.contains("already deployed"));
            }
            _ => panic!("expected rejection"),
        }
    }

    #[test]
    fn test_deploy_capacity_exceeded() {
        let mut api = make_api();
        for i in 0..MAX_WORKLOADS {
            let req = DeployRequest {
                workload_id: alloc::format!("wl-{}", i),
                model_url: String::from("model.onnx"),
                name: alloc::format!("wl-{}", i),
                namespace: String::from("default"),
                env: Vec::new(),
                cpu_millis: 100,
                memory_bytes: 1024,
                gpu_required: false,
            };
            let resp = api.handle_deploy(&req);
            assert!(matches!(resp, DeployResponse::Accepted { .. }));
        }
        let req = DeployRequest {
            workload_id: String::from("overflow"),
            model_url: String::from("model.onnx"),
            name: String::from("overflow"),
            namespace: String::from("default"),
            env: Vec::new(),
            cpu_millis: 100,
            memory_bytes: 1024,
            gpu_required: false,
        };
        let resp = api.handle_deploy(&req);
        match resp {
            DeployResponse::Rejected { reason, .. } => {
                assert!(reason.contains("capacity"));
            }
            _ => panic!("expected rejection"),
        }
    }

    #[test]
    fn test_deploy_gpu_required_no_gpu() {
        let resources = NodeResources::standard(4000, 8_000_000_000, 0);
        let mut api = ManagementApi::new(resources).unwrap();
        let mut req = sample_deploy_req();
        req.gpu_required = true;
        let resp = api.handle_deploy(&req);
        match resp {
            DeployResponse::Rejected { reason, .. } => {
                assert!(reason.contains("GPU"));
            }
            _ => panic!("expected GPU rejection"),
        }
    }

    #[test]
    fn test_deploy_gpu_required_available() {
        let mut api = make_api();
        let mut req = sample_deploy_req();
        req.gpu_required = true;
        let resp = api.handle_deploy(&req);
        assert!(matches!(resp, DeployResponse::Accepted { .. }));
    }

    // --- Undeploy ---

    #[test]
    fn test_undeploy_graceful() {
        let mut api = make_api();
        api.handle_deploy(&sample_deploy_req());
        let req = UndeployRequest {
            workload_id: String::from("pod-uid-123"),
            force: false,
        };
        api.handle_undeploy(&req).unwrap();
        let status = api.query_status("pod-uid-123");
        assert_eq!(status[0].phase, WorkloadPhase::Terminating);
    }

    #[test]
    fn test_undeploy_force() {
        let mut api = make_api();
        api.handle_deploy(&sample_deploy_req());
        let req = UndeployRequest {
            workload_id: String::from("pod-uid-123"),
            force: true,
        };
        api.handle_undeploy(&req).unwrap();
        assert_eq!(api.workload_count(), 0);
    }

    #[test]
    fn test_undeploy_not_found() {
        let mut api = make_api();
        let req = UndeployRequest {
            workload_id: String::from("nonexistent"),
            force: false,
        };
        assert_eq!(api.handle_undeploy(&req), Err(IpcError::NotFound));
    }

    // --- Status ---

    #[test]
    fn test_query_status_all() {
        let mut api = make_api();
        api.handle_deploy(&sample_deploy_req());
        let mut req2 = sample_deploy_req();
        req2.workload_id = String::from("wl-2");
        req2.name = String::from("model-b");
        api.handle_deploy(&req2);
        let all = api.query_status("");
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_query_status_specific() {
        let mut api = make_api();
        api.handle_deploy(&sample_deploy_req());
        let status = api.query_status("pod-uid-123");
        assert_eq!(status.len(), 1);
        assert_eq!(status[0].phase, WorkloadPhase::Pending);
    }

    #[test]
    fn test_query_status_not_found() {
        let api = make_api();
        let status = api.query_status("nonexistent");
        assert!(status.is_empty());
    }

    // --- Phase updates ---

    #[test]
    fn test_update_phase() {
        let mut api = make_api();
        api.handle_deploy(&sample_deploy_req());
        api.update_phase("pod-uid-123", WorkloadPhase::Loading, "downloading model").unwrap();
        assert_eq!(api.query_status("pod-uid-123")[0].phase, WorkloadPhase::Loading);
        api.update_phase("pod-uid-123", WorkloadPhase::Running, "serving").unwrap();
        assert_eq!(api.query_status("pod-uid-123")[0].phase, WorkloadPhase::Running);
    }

    #[test]
    fn test_update_phase_not_found() {
        let mut api = make_api();
        assert_eq!(
            api.update_phase("nonexistent", WorkloadPhase::Running, ""),
            Err(IpcError::NotFound)
        );
    }

    // --- Inference tracking ---

    #[test]
    fn test_record_inference() {
        let mut api = make_api();
        api.handle_deploy(&sample_deploy_req());
        api.update_phase("pod-uid-123", WorkloadPhase::Running, "serving").unwrap();
        for _ in 0..10 {
            api.record_inference("pod-uid-123").unwrap();
        }
        assert_eq!(api.query_status("pod-uid-123")[0].inference_count, 10);
    }

    // --- Uptime ---

    #[test]
    fn test_update_uptime() {
        let mut api = make_api();
        api.handle_deploy(&sample_deploy_req());
        api.update_uptime("pod-uid-123", 5000).unwrap();
        assert_eq!(api.query_status("pod-uid-123")[0].uptime_ms, 5000);
    }

    // --- Cleanup ---

    #[test]
    fn test_cleanup_terminated() {
        let mut api = make_api();
        api.handle_deploy(&sample_deploy_req());
        api.update_phase("pod-uid-123", WorkloadPhase::Terminating, "shutting down").unwrap();
        let cleaned = api.cleanup_terminated();
        assert_eq!(cleaned, 1);
        assert_eq!(api.workload_count(), 0);
    }

    // --- Running count ---

    #[test]
    fn test_running_count() {
        let mut api = make_api();
        api.handle_deploy(&sample_deploy_req());
        assert_eq!(api.running_count(), 0);
        api.update_phase("pod-uid-123", WorkloadPhase::Running, "ok").unwrap();
        assert_eq!(api.running_count(), 1);
    }

    // --- Node resources ---

    #[test]
    fn test_node_resources_standard() {
        let res = NodeResources::standard(4000, 8_000_000_000, 1);
        assert_eq!(res.len(), 3);
        assert_eq!(res.get("cpu").unwrap().capacity, 4000);
        assert_eq!(res.get("memory").unwrap().capacity, 8_000_000_000);
        assert_eq!(res.get("nvidia.com/gpu").unwrap().capacity, 1);
    }

    #[test]
    fn test_node_resources_no_gpu() {
        let res = NodeResources::standard(2000, 4_000_000_000, 0);
        assert_eq!(res.len(), 2);
        assert!(res.get("nvidia.com/gpu").is_none());
    }

    #[test]
    fn test_node_resources_add_update() {
        let mut res = NodeResources::new();
        res.add("cpu", 4000, 4000).unwrap();
        assert_eq!(res.get("cpu").unwrap().allocatable, 4000);
        // Update existing
        res.add("cpu", 4000, 3000).unwrap();
        assert_eq!(res.get("cpu").unwrap().allocatable, 3000);
        assert_eq!(res.len(), 1); // Not duplicated
    }

    #[test]
    fn test_node_resources_json() {
        let res = NodeResources::standard(4000, 8000, 1);
        let json = res.to_json();
        assert!(json.contains("\"cpu\":{\"capacity\":4000,\"allocatable\":4000}"));
        assert!(json.contains("\"memory\":{\"capacity\":8000,\"allocatable\":8000}"));
        assert!(json.contains("\"nvidia.com/gpu\":{\"capacity\":1,\"allocatable\":1}"));
    }

    #[test]
    fn test_node_resources_empty_json() {
        let res = NodeResources::new();
        assert_eq!(res.to_json(), "{}");
    }

    // --- Deploy request encode/decode ---

    #[test]
    fn test_deploy_request_roundtrip() {
        let req = sample_deploy_req();
        let encoded = req.encode();
        let decoded = DeployRequest::decode(&encoded).unwrap();
        assert_eq!(decoded.workload_id, req.workload_id);
        assert_eq!(decoded.model_url, req.model_url);
        assert_eq!(decoded.name, req.name);
        assert_eq!(decoded.namespace, req.namespace);
        assert_eq!(decoded.cpu_millis, req.cpu_millis);
        assert_eq!(decoded.memory_bytes, req.memory_bytes);
        assert_eq!(decoded.gpu_required, req.gpu_required);
        assert_eq!(decoded.env, req.env);
    }

    #[test]
    fn test_deploy_request_decode_too_short() {
        assert_eq!(DeployRequest::decode(&[0u8; 2]).err(), Some(IpcError::PacketTooShort));
    }

    // --- Undeploy request encode/decode ---

    #[test]
    fn test_undeploy_request_roundtrip() {
        let req = UndeployRequest {
            workload_id: String::from("wl-abc"),
            force: true,
        };
        let encoded = req.encode();
        let decoded = UndeployRequest::decode(&encoded).unwrap();
        assert_eq!(decoded.workload_id, "wl-abc");
        assert!(decoded.force);
    }

    #[test]
    fn test_undeploy_request_decode_too_short() {
        assert_eq!(UndeployRequest::decode(&[]).err(), Some(IpcError::PacketTooShort));
    }

    // --- WorkloadPhase ---

    #[test]
    fn test_workload_phase_k8s_mapping() {
        assert_eq!(WorkloadPhase::Pending.as_k8s_phase(), "Pending");
        assert_eq!(WorkloadPhase::Loading.as_k8s_phase(), "Pending");
        assert_eq!(WorkloadPhase::Running.as_k8s_phase(), "Running");
        assert_eq!(WorkloadPhase::Succeeded.as_k8s_phase(), "Succeeded");
        assert_eq!(WorkloadPhase::Failed.as_k8s_phase(), "Failed");
        assert_eq!(WorkloadPhase::Terminating.as_k8s_phase(), "Running");
    }

    // --- Status JSON ---

    #[test]
    fn test_status_json_empty() {
        let api = make_api();
        assert_eq!(api.to_status_json(), "[]");
    }

    #[test]
    fn test_status_json_with_workloads() {
        let mut api = make_api();
        api.handle_deploy(&sample_deploy_req());
        api.update_phase("pod-uid-123", WorkloadPhase::Running, "ok").unwrap();
        api.record_inference("pod-uid-123").unwrap();
        let json = api.to_status_json();
        assert!(json.contains("\"id\":\"pod-uid-123\""));
        assert!(json.contains("\"phase\":\"Running\""));
        assert!(json.contains("\"inferences\":1"));
    }

    // --- Resources table full ---

    #[test]
    fn test_resources_table_full() {
        let mut res = NodeResources::new();
        for i in 0..MAX_RESOURCES {
            res.add(&alloc::format!("res-{}", i), 100, 100).unwrap();
        }
        assert_eq!(res.add("overflow", 1, 1), Err(IpcError::TableFull));
    }
}
