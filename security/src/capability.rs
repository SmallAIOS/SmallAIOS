// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Capability-based access control.
//!
//! Every resource access requires an explicit capability token. No ambient
//! authority exists — tasks start with zero capabilities and receive them
//! through explicit grants. Capabilities can be delegated (with reduced
//! permissions) and revoked.

/// Unique capability identifier.
pub type CapId = u64;

/// Task identifier (matches scheduler task ID).
pub type TaskId = u64;

/// Maximum number of capabilities per task.
pub const MAX_CAPS_PER_TASK: usize = 256;

/// Maximum total capabilities system-wide.
pub const MAX_TOTAL_CAPS: usize = 4096;

/// Resource types that capabilities can protect.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ResourceType {
    /// Tensor buffer (READ, WRITE).
    TensorBuffer = 0,
    /// ONNX model (READ, EXECUTE).
    OnnxModel = 1,
    /// IPC endpoint — pub/sub key expression (READ, WRITE).
    IpcEndpoint = 2,
    /// GPU device (READ, WRITE, EXECUTE).
    GpuDevice = 3,
    /// Network socket (READ, WRITE).
    NetworkSocket = 4,
    /// System configuration (READ).
    SystemConfig = 5,
    /// System control — shutdown, watchdog (EXECUTE).
    SystemControl = 6,
    /// Device — bus, FPGA, DMA (READ, WRITE, EXECUTE).
    Device = 7,
}

impl ResourceType {
    /// Convert from u8.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::TensorBuffer),
            1 => Some(Self::OnnxModel),
            2 => Some(Self::IpcEndpoint),
            3 => Some(Self::GpuDevice),
            4 => Some(Self::NetworkSocket),
            5 => Some(Self::SystemConfig),
            6 => Some(Self::SystemControl),
            7 => Some(Self::Device),
            _ => None,
        }
    }
}

/// Peripheral device subtype for fine-grained device capability checks.
///
/// Used with `ResourceType::Device` to differentiate peripheral types.
/// The `instance_id` in `ResourceRef` encodes `(subtype << 8) | instance`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PeripheralCapability {
    /// I2C bus controller access.
    I2c = 0x10,
    /// SPI bus controller access.
    Spi = 0x11,
    /// GPIO controller access.
    Gpio = 0x12,
    /// UART serial port access.
    Uart = 0x13,
    /// MIPI CSI-2 camera access (privacy-sensitive).
    Camera = 0x14,
    /// I2S/TDM audio capture access (privacy-sensitive).
    Audio = 0x15,
}

impl PeripheralCapability {
    /// Convert from u8.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0x10 => Some(Self::I2c),
            0x11 => Some(Self::Spi),
            0x12 => Some(Self::Gpio),
            0x13 => Some(Self::Uart),
            0x14 => Some(Self::Camera),
            0x15 => Some(Self::Audio),
            _ => None,
        }
    }

    /// Encode a peripheral capability + instance into a `ResourceRef` instance_id.
    pub const fn encode_instance_id(self, instance: u8) -> u64 {
        ((self as u64) << 8) | (instance as u64)
    }

    /// Decode a `ResourceRef` instance_id into (PeripheralCapability, instance).
    pub fn decode_instance_id(instance_id: u64) -> Option<(Self, u8)> {
        let subtype = ((instance_id >> 8) & 0xFF) as u8;
        let instance = (instance_id & 0xFF) as u8;
        Self::from_u8(subtype).map(|cap| (cap, instance))
    }

    /// Create a `ResourceRef` for this peripheral type and instance.
    pub const fn resource_ref(self, instance: u8) -> ResourceRef {
        ResourceRef::new(ResourceType::Device, self.encode_instance_id(instance))
    }
}

/// Reference to a specific resource instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceRef {
    /// Type of resource.
    pub resource_type: ResourceType,
    /// Instance identifier (resource-type-specific).
    /// For tensors: tensor handle. For models: model handle.
    /// For IPC: key expression hash. For devices: `(subtype << 8) | instance`.
    pub instance_id: u64,
}

impl ResourceRef {
    pub const fn new(resource_type: ResourceType, instance_id: u64) -> Self {
        Self {
            resource_type,
            instance_id,
        }
    }
}

/// Permission bitmask for capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Permissions(u32);

impl Permissions {
    pub const READ: Self = Self(0b0001);
    pub const WRITE: Self = Self(0b0010);
    pub const EXECUTE: Self = Self(0b0100);
    pub const GRANT: Self = Self(0b1000);
    pub const NONE: Self = Self(0);
    pub const ALL: Self = Self(0b1111);

    pub const fn bits(self) -> u32 {
        self.0
    }

    pub const fn from_bits(bits: u32) -> Self {
        Self(bits & 0b1111)
    }

    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    pub const fn intersect(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Check if this permission set is a subset of another.
    pub const fn is_subset_of(self, other: Self) -> bool {
        (self.0 & !other.0) == 0
    }
}

/// A capability token granting access to a resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capability {
    /// Unique identifier for this capability.
    pub id: CapId,
    /// The resource this capability grants access to.
    pub resource: ResourceRef,
    /// Permissions granted by this capability.
    pub permissions: Permissions,
    /// Optional expiry timestamp (0 = no expiry).
    pub expires: u64,
    /// Task that owns this capability.
    pub owner: TaskId,
    /// Capability this was delegated from (0 = root grant).
    pub parent: CapId,
}

impl Capability {
    /// Check whether this capability has expired.
    pub fn is_expired(&self, now: u64) -> bool {
        self.expires != 0 && now >= self.expires
    }

    /// Check whether this capability grants the requested permissions on the resource.
    pub fn check(
        &self,
        resource: &ResourceRef,
        required: Permissions,
        now: u64,
    ) -> Result<(), CapError> {
        if self.is_expired(now) {
            return Err(CapError::Expired);
        }
        if self.resource != *resource {
            return Err(CapError::WrongResource);
        }
        if !self.permissions.contains(required) {
            return Err(CapError::InsufficientPermissions);
        }
        Ok(())
    }
}

/// Errors from capability operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapError {
    /// Capability not found.
    NotFound,
    /// Capability has expired.
    Expired,
    /// Capability is for a different resource.
    WrongResource,
    /// Insufficient permissions.
    InsufficientPermissions,
    /// Cannot delegate more permissions than held.
    DelegationExceedsPermissions,
    /// Capability has been revoked.
    Revoked,
    /// Too many capabilities for this task.
    TooManyCaps,
    /// System-wide capability limit reached.
    SystemLimitReached,
    /// Invalid capability ID.
    InvalidId,
}

/// Capability registry — tracks all capabilities system-wide.
///
/// Uses a fixed-size array to avoid heap allocation. In production this would
/// be a more sophisticated data structure; for the kernel's needs (bounded
/// capability count) a flat array with linear scan is sufficient and avoids
/// allocator dependencies.
pub struct CapRegistry {
    /// All capabilities in the system. Slot is `None` if free.
    caps: [Option<Capability>; MAX_TOTAL_CAPS],
    /// Next capability ID to assign.
    next_id: CapId,
    /// Set of revoked capability IDs (stored as a bitmap for space).
    /// Index = cap_id % REVOKED_BITMAP_SIZE, so this is approximate
    /// for IDs > REVOKED_BITMAP_SIZE. Production would use a proper set.
    revoked: [u64; REVOKED_BITMAP_WORDS],
}

const REVOKED_BITMAP_SIZE: usize = 4096;
const REVOKED_BITMAP_WORDS: usize = REVOKED_BITMAP_SIZE / 64;

impl CapRegistry {
    /// Create an empty capability registry.
    pub const fn new() -> Self {
        Self {
            caps: [None; MAX_TOTAL_CAPS],
            next_id: 1, // 0 is reserved (no parent)
            revoked: [0; REVOKED_BITMAP_WORDS],
        }
    }

    /// Create a new capability (root grant — no parent).
    pub fn create(
        &mut self,
        owner: TaskId,
        resource: ResourceRef,
        permissions: Permissions,
        expires: u64,
    ) -> Result<CapId, CapError> {
        self.insert(owner, resource, permissions, expires, 0)
    }

    /// Delegate a capability to another task with reduced permissions.
    pub fn delegate(
        &mut self,
        cap_id: CapId,
        to_task: TaskId,
        new_perms: Permissions,
        new_expires: u64,
        now: u64,
    ) -> Result<CapId, CapError> {
        // Find the parent capability
        let parent = self.find(cap_id).ok_or(CapError::NotFound)?;

        if self.is_revoked(cap_id) {
            return Err(CapError::Revoked);
        }
        if parent.is_expired(now) {
            return Err(CapError::Expired);
        }
        // Must hold GRANT permission to delegate
        if !parent.permissions.contains(Permissions::GRANT) {
            return Err(CapError::InsufficientPermissions);
        }
        // Delegated permissions must be a subset of parent's (minus GRANT unless explicitly included)
        if !new_perms.is_subset_of(parent.permissions) {
            return Err(CapError::DelegationExceedsPermissions);
        }
        // If parent expires, delegated must expire no later
        let effective_expires =
            if parent.expires != 0 && (new_expires == 0 || new_expires > parent.expires) {
                parent.expires
            } else {
                new_expires
            };

        let resource = parent.resource;
        self.insert(to_task, resource, new_perms, effective_expires, cap_id)
    }

    /// Revoke a capability and all capabilities delegated from it.
    pub fn revoke(&mut self, cap_id: CapId) -> Result<(), CapError> {
        // Check it exists
        if self.find(cap_id).is_none() {
            return Err(CapError::NotFound);
        }

        // Mark as revoked
        self.mark_revoked(cap_id);

        // Remove from slots
        for slot in self.caps.iter_mut() {
            if let Some(cap) = slot {
                if cap.id == cap_id {
                    *slot = None;
                    break;
                }
            }
        }

        // Cascade: revoke all children
        // We need to collect children first to avoid borrow issues
        let mut children = [0u64; 64];
        let mut child_count = 0;
        for cap in self.caps.iter().flatten() {
            if cap.parent == cap_id && child_count < 64 {
                children[child_count] = cap.id;
                child_count += 1;
            }
        }
        for child in children.iter().take(child_count) {
            let _ = self.revoke(*child);
        }

        Ok(())
    }

    /// Look up a capability by ID.
    pub fn lookup(&self, cap_id: CapId, now: u64) -> Result<&Capability, CapError> {
        if self.is_revoked(cap_id) {
            return Err(CapError::Revoked);
        }
        let cap = self.find(cap_id).ok_or(CapError::NotFound)?;
        if cap.is_expired(now) {
            return Err(CapError::Expired);
        }
        Ok(cap)
    }

    /// Check whether a task has the required permissions on a resource.
    pub fn check(
        &self,
        task_id: TaskId,
        resource: &ResourceRef,
        required: Permissions,
        now: u64,
    ) -> Result<CapId, CapError> {
        for cap in self.caps.iter().flatten() {
            if cap.owner == task_id
                && !self.is_revoked(cap.id)
                && cap.check(resource, required, now).is_ok()
            {
                return Ok(cap.id);
            }
        }
        Err(CapError::InsufficientPermissions)
    }

    /// List all capabilities held by a task.
    pub fn list_for_task(&self, task_id: TaskId, buf: &mut [CapId]) -> usize {
        let mut count = 0;
        for cap in self.caps.iter().flatten() {
            if cap.owner == task_id && !self.is_revoked(cap.id) && count < buf.len() {
                buf[count] = cap.id;
                count += 1;
            }
        }
        count
    }

    /// Count capabilities held by a task.
    pub fn count_for_task(&self, task_id: TaskId) -> usize {
        self.caps
            .iter()
            .filter(|slot| {
                slot.as_ref()
                    .map(|c| c.owner == task_id && !self.is_revoked(c.id))
                    .unwrap_or(false)
            })
            .count()
    }

    /// Total number of active capabilities.
    pub fn total_count(&self) -> usize {
        self.caps.iter().filter(|s| s.is_some()).count()
    }

    // --- Internal helpers ---

    fn insert(
        &mut self,
        owner: TaskId,
        resource: ResourceRef,
        permissions: Permissions,
        expires: u64,
        parent: CapId,
    ) -> Result<CapId, CapError> {
        // Check per-task limit
        if self.count_for_task(owner) >= MAX_CAPS_PER_TASK {
            return Err(CapError::TooManyCaps);
        }

        // Find a free slot
        let slot = self
            .caps
            .iter_mut()
            .find(|s| s.is_none())
            .ok_or(CapError::SystemLimitReached)?;

        let id = self.next_id;
        self.next_id += 1;

        *slot = Some(Capability {
            id,
            resource,
            permissions,
            expires,
            owner,
            parent,
        });

        Ok(id)
    }

    fn find(&self, cap_id: CapId) -> Option<&Capability> {
        self.caps
            .iter()
            .filter_map(|s| s.as_ref())
            .find(|c| c.id == cap_id)
    }

    fn is_revoked(&self, cap_id: CapId) -> bool {
        let idx = (cap_id as usize) % REVOKED_BITMAP_SIZE;
        let word = idx / 64;
        let bit = idx % 64;
        (self.revoked[word] & (1u64 << bit)) != 0
    }

    fn mark_revoked(&mut self, cap_id: CapId) {
        let idx = (cap_id as usize) % REVOKED_BITMAP_SIZE;
        let word = idx / 64;
        let bit = idx % 64;
        self.revoked[word] |= 1u64 << bit;
    }
}

impl Default for CapRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_resource(rt: ResourceType, id: u64) -> ResourceRef {
        ResourceRef::new(rt, id)
    }

    #[test]
    fn test_permissions_basic() {
        assert!(Permissions::ALL.contains(Permissions::READ));
        assert!(Permissions::ALL.contains(Permissions::WRITE));
        assert!(Permissions::ALL.contains(Permissions::EXECUTE));
        assert!(Permissions::ALL.contains(Permissions::GRANT));
        assert!(!Permissions::READ.contains(Permissions::WRITE));
        assert!(Permissions::NONE.is_empty());
        assert!(!Permissions::READ.is_empty());
    }

    #[test]
    fn test_permissions_subset() {
        assert!(Permissions::READ.is_subset_of(Permissions::ALL));
        assert!(Permissions::NONE.is_subset_of(Permissions::READ));
        assert!(!Permissions::ALL.is_subset_of(Permissions::READ));
        let rw = Permissions::READ.union(Permissions::WRITE);
        assert!(Permissions::READ.is_subset_of(rw));
        assert!(Permissions::WRITE.is_subset_of(rw));
        assert!(!Permissions::EXECUTE.is_subset_of(rw));
    }

    #[test]
    fn test_permissions_intersect() {
        let rw = Permissions::READ.union(Permissions::WRITE);
        let rx = Permissions::READ.union(Permissions::EXECUTE);
        let result = rw.intersect(rx);
        assert!(result.contains(Permissions::READ));
        assert!(!result.contains(Permissions::WRITE));
        assert!(!result.contains(Permissions::EXECUTE));
    }

    #[test]
    fn test_permissions_from_bits() {
        assert_eq!(Permissions::from_bits(0b1111), Permissions::ALL);
        assert_eq!(Permissions::from_bits(0b0001), Permissions::READ);
        // High bits are masked off
        assert_eq!(Permissions::from_bits(0xFFFF).bits(), 0b1111);
    }

    #[test]
    fn test_resource_type_from_u8() {
        assert_eq!(ResourceType::from_u8(0), Some(ResourceType::TensorBuffer));
        assert_eq!(ResourceType::from_u8(7), Some(ResourceType::Device));
        assert_eq!(ResourceType::from_u8(8), None);
        assert_eq!(ResourceType::from_u8(255), None);
    }

    #[test]
    fn test_capability_not_expired() {
        let cap = Capability {
            id: 1,
            resource: make_resource(ResourceType::TensorBuffer, 42),
            permissions: Permissions::READ,
            expires: 0, // no expiry
            owner: 1,
            parent: 0,
        };
        assert!(!cap.is_expired(1000));
        assert!(!cap.is_expired(u64::MAX));
    }

    #[test]
    fn test_capability_expired() {
        let cap = Capability {
            id: 1,
            resource: make_resource(ResourceType::TensorBuffer, 42),
            permissions: Permissions::READ,
            expires: 100,
            owner: 1,
            parent: 0,
        };
        assert!(!cap.is_expired(99));
        assert!(cap.is_expired(100));
        assert!(cap.is_expired(101));
    }

    #[test]
    fn test_capability_check_success() {
        let res = make_resource(ResourceType::OnnxModel, 1);
        let cap = Capability {
            id: 1,
            resource: res,
            permissions: Permissions::READ.union(Permissions::EXECUTE),
            expires: 0,
            owner: 1,
            parent: 0,
        };
        assert!(cap.check(&res, Permissions::READ, 0).is_ok());
        assert!(cap.check(&res, Permissions::EXECUTE, 0).is_ok());
    }

    #[test]
    fn test_capability_check_wrong_resource() {
        let res1 = make_resource(ResourceType::OnnxModel, 1);
        let res2 = make_resource(ResourceType::OnnxModel, 2);
        let cap = Capability {
            id: 1,
            resource: res1,
            permissions: Permissions::ALL,
            expires: 0,
            owner: 1,
            parent: 0,
        };
        assert_eq!(
            cap.check(&res2, Permissions::READ, 0),
            Err(CapError::WrongResource)
        );
    }

    #[test]
    fn test_capability_check_insufficient_perms() {
        let res = make_resource(ResourceType::TensorBuffer, 1);
        let cap = Capability {
            id: 1,
            resource: res,
            permissions: Permissions::READ,
            expires: 0,
            owner: 1,
            parent: 0,
        };
        assert_eq!(
            cap.check(&res, Permissions::WRITE, 0),
            Err(CapError::InsufficientPermissions)
        );
    }

    #[test]
    fn test_capability_check_expired() {
        let res = make_resource(ResourceType::TensorBuffer, 1);
        let cap = Capability {
            id: 1,
            resource: res,
            permissions: Permissions::ALL,
            expires: 50,
            owner: 1,
            parent: 0,
        };
        assert_eq!(
            cap.check(&res, Permissions::READ, 100),
            Err(CapError::Expired)
        );
    }

    #[test]
    fn test_registry_create_and_lookup() {
        let mut reg = CapRegistry::new();
        let res = make_resource(ResourceType::OnnxModel, 1);
        let id = reg
            .create(1, res, Permissions::READ.union(Permissions::EXECUTE), 0)
            .unwrap();
        assert!(id > 0);

        let cap = reg.lookup(id, 0).unwrap();
        assert_eq!(cap.owner, 1);
        assert_eq!(cap.resource, res);
        assert!(cap.permissions.contains(Permissions::READ));
    }

    #[test]
    fn test_registry_lookup_not_found() {
        let reg = CapRegistry::new();
        assert_eq!(reg.lookup(999, 0), Err(CapError::NotFound));
    }

    #[test]
    fn test_registry_revoke() {
        let mut reg = CapRegistry::new();
        let res = make_resource(ResourceType::TensorBuffer, 1);
        let id = reg.create(1, res, Permissions::ALL, 0).unwrap();
        assert!(reg.lookup(id, 0).is_ok());

        reg.revoke(id).unwrap();
        assert_eq!(reg.lookup(id, 0), Err(CapError::Revoked));
    }

    #[test]
    fn test_registry_revoke_cascades() {
        let mut reg = CapRegistry::new();
        let res = make_resource(ResourceType::TensorBuffer, 1);
        let parent_id = reg.create(1, res, Permissions::ALL, 0).unwrap();
        let child_id = reg.delegate(parent_id, 2, Permissions::READ, 0, 0).unwrap();

        // Revoking parent should cascade to child
        reg.revoke(parent_id).unwrap();
        assert_eq!(reg.lookup(parent_id, 0), Err(CapError::Revoked));
        assert_eq!(reg.lookup(child_id, 0), Err(CapError::Revoked));
    }

    #[test]
    fn test_registry_delegate() {
        let mut reg = CapRegistry::new();
        let res = make_resource(ResourceType::IpcEndpoint, 42);
        let parent_id = reg.create(1, res, Permissions::ALL, 0).unwrap();

        let child_id = reg.delegate(parent_id, 2, Permissions::READ, 0, 0).unwrap();

        let child = reg.lookup(child_id, 0).unwrap();
        assert_eq!(child.owner, 2);
        assert_eq!(child.resource, res);
        assert!(child.permissions.contains(Permissions::READ));
        assert!(!child.permissions.contains(Permissions::WRITE));
        assert_eq!(child.parent, parent_id);
    }

    #[test]
    fn test_registry_delegate_exceeds_permissions() {
        let mut reg = CapRegistry::new();
        let res = make_resource(ResourceType::TensorBuffer, 1);
        let parent_id = reg.create(1, res, Permissions::ALL, 0).unwrap();

        // Delegate only READ to task 2
        let child_id = reg
            .delegate(
                parent_id,
                2,
                Permissions::READ.union(Permissions::GRANT),
                0,
                0,
            )
            .unwrap();

        // Task 2 tries to delegate WRITE (which it doesn't have)
        let result = reg.delegate(child_id, 3, Permissions::WRITE, 0, 0);
        assert_eq!(result, Err(CapError::DelegationExceedsPermissions));
    }

    #[test]
    fn test_registry_delegate_without_grant_perm() {
        let mut reg = CapRegistry::new();
        let res = make_resource(ResourceType::TensorBuffer, 1);
        // Create with READ only (no GRANT)
        let id = reg.create(1, res, Permissions::READ, 0).unwrap();

        let result = reg.delegate(id, 2, Permissions::READ, 0, 0);
        assert_eq!(result, Err(CapError::InsufficientPermissions));
    }

    #[test]
    fn test_registry_delegate_expired_parent() {
        let mut reg = CapRegistry::new();
        let res = make_resource(ResourceType::TensorBuffer, 1);
        let parent_id = reg.create(1, res, Permissions::ALL, 50).unwrap();

        let result = reg.delegate(parent_id, 2, Permissions::READ, 0, 100);
        assert_eq!(result, Err(CapError::Expired));
    }

    #[test]
    fn test_registry_delegate_inherits_earlier_expiry() {
        let mut reg = CapRegistry::new();
        let res = make_resource(ResourceType::TensorBuffer, 1);
        let parent_id = reg.create(1, res, Permissions::ALL, 100).unwrap();

        // Child requests no expiry, but parent expires at 100
        let child_id = reg.delegate(parent_id, 2, Permissions::READ, 0, 0).unwrap();
        let child = reg.lookup(child_id, 0).unwrap();
        assert_eq!(child.expires, 100); // Capped to parent's expiry
    }

    #[test]
    fn test_registry_check() {
        let mut reg = CapRegistry::new();
        let res = make_resource(ResourceType::OnnxModel, 5);
        reg.create(1, res, Permissions::READ.union(Permissions::EXECUTE), 0)
            .unwrap();

        assert!(reg.check(1, &res, Permissions::READ, 0).is_ok());
        assert!(reg.check(1, &res, Permissions::EXECUTE, 0).is_ok());
        assert_eq!(
            reg.check(1, &res, Permissions::WRITE, 0),
            Err(CapError::InsufficientPermissions)
        );
        assert_eq!(
            reg.check(2, &res, Permissions::READ, 0),
            Err(CapError::InsufficientPermissions)
        );
    }

    #[test]
    fn test_registry_list_for_task() {
        let mut reg = CapRegistry::new();
        let res1 = make_resource(ResourceType::TensorBuffer, 1);
        let res2 = make_resource(ResourceType::OnnxModel, 2);
        reg.create(1, res1, Permissions::READ, 0).unwrap();
        reg.create(1, res2, Permissions::READ, 0).unwrap();
        reg.create(2, res1, Permissions::WRITE, 0).unwrap();

        let mut buf = [0u64; 10];
        let count = reg.list_for_task(1, &mut buf);
        assert_eq!(count, 2);

        let count = reg.list_for_task(2, &mut buf);
        assert_eq!(count, 1);

        let count = reg.list_for_task(99, &mut buf);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_registry_total_count() {
        let mut reg = CapRegistry::new();
        assert_eq!(reg.total_count(), 0);

        let res = make_resource(ResourceType::TensorBuffer, 1);
        reg.create(1, res, Permissions::READ, 0).unwrap();
        assert_eq!(reg.total_count(), 1);

        let id = reg.create(2, res, Permissions::WRITE, 0).unwrap();
        assert_eq!(reg.total_count(), 2);

        reg.revoke(id).unwrap();
        assert_eq!(reg.total_count(), 1);
    }

    #[test]
    fn test_registry_revoke_not_found() {
        let mut reg = CapRegistry::new();
        assert_eq!(reg.revoke(999), Err(CapError::NotFound));
    }

    #[test]
    fn test_registry_create_multiple_resources() {
        let mut reg = CapRegistry::new();
        for i in 0..10 {
            let res = make_resource(ResourceType::TensorBuffer, i);
            reg.create(1, res, Permissions::ALL, 0).unwrap();
        }
        assert_eq!(reg.total_count(), 10);
        assert_eq!(reg.count_for_task(1), 10);
    }

    // -- PeripheralCapability -----------------------------------------------

    #[test]
    fn test_peripheral_capability_from_u8() {
        assert_eq!(
            PeripheralCapability::from_u8(0x10),
            Some(PeripheralCapability::I2c)
        );
        assert_eq!(
            PeripheralCapability::from_u8(0x11),
            Some(PeripheralCapability::Spi)
        );
        assert_eq!(
            PeripheralCapability::from_u8(0x12),
            Some(PeripheralCapability::Gpio)
        );
        assert_eq!(
            PeripheralCapability::from_u8(0x13),
            Some(PeripheralCapability::Uart)
        );
        assert_eq!(
            PeripheralCapability::from_u8(0x14),
            Some(PeripheralCapability::Camera)
        );
        assert_eq!(
            PeripheralCapability::from_u8(0x15),
            Some(PeripheralCapability::Audio)
        );
        assert_eq!(PeripheralCapability::from_u8(0x00), None);
        assert_eq!(PeripheralCapability::from_u8(0xFF), None);
    }

    #[test]
    fn test_peripheral_capability_encode_decode() {
        let cap = PeripheralCapability::I2c;
        let id = cap.encode_instance_id(2);
        let (decoded_cap, decoded_instance) = PeripheralCapability::decode_instance_id(id).unwrap();
        assert_eq!(decoded_cap, PeripheralCapability::I2c);
        assert_eq!(decoded_instance, 2);
    }

    #[test]
    fn test_peripheral_capability_resource_ref() {
        let rref = PeripheralCapability::Camera.resource_ref(0);
        assert_eq!(rref.resource_type, ResourceType::Device);
        let (cap, inst) = PeripheralCapability::decode_instance_id(rref.instance_id).unwrap();
        assert_eq!(cap, PeripheralCapability::Camera);
        assert_eq!(inst, 0);
    }

    #[test]
    fn test_peripheral_capability_check_via_registry() {
        let mut reg = CapRegistry::new();
        let res = PeripheralCapability::Uart.resource_ref(1);
        let cap_id = reg
            .create(1, res, Permissions::READ.union(Permissions::WRITE), 0)
            .unwrap();
        assert!(cap_id > 0);
        assert!(reg.check(1, &res, Permissions::READ, 0).is_ok());
        assert!(reg.check(1, &res, Permissions::WRITE, 0).is_ok());
        // Different instance should fail
        let res2 = PeripheralCapability::Uart.resource_ref(2);
        assert_eq!(
            reg.check(1, &res2, Permissions::READ, 0),
            Err(CapError::InsufficientPermissions)
        );
    }
}
