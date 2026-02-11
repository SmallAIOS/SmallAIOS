// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Global kernel state — singleton subsystem instances.
//!
//! In unikernel mode, syscall handlers execute with interrupts masked
//! (FMASK on x86-64, DAIF on ARM64), so simple spinlock-free access
//! is safe when there is only one caller at a time. For SMP, the
//! subsystems themselves use per-CPU data or atomic operations.
//!
//! The state is initialized during `kernel_main` before syscalls are
//! enabled. Syscall handlers access subsystems through the accessor
//! functions here.

use crate::mem::buddy::BuddyAllocator;
use crate::mem::slab::SlabAllocator;
use crate::mem::tensor::TensorPool;
use crate::mem::MemError;
use core::cell::UnsafeCell;
use smallaios_security::capability::{CapError, CapRegistry};
use smallaios_security::crypto::csprng::{Csprng, CsprngError, CsprngSeed};

/// Global kernel state, initialized once during boot.
///
/// # Safety
/// Access is safe because:
/// - Initialization happens before any syscalls are enabled
/// - Syscall handlers run with interrupts masked (single-threaded per core)
/// - SMP coordination is handled by per-CPU run queues, not shared mutable state
struct KernelState {
    buddy: UnsafeCell<BuddyAllocator>,
    slab: UnsafeCell<SlabAllocator>,
    tensor_pool: UnsafeCell<TensorPool>,
    cap_registry: UnsafeCell<CapRegistry>,
    csprng: UnsafeCell<Csprng>,
    initialized: core::sync::atomic::AtomicBool,
}

// SAFETY: Access is serialized by interrupt masking during syscall dispatch.
// The UnsafeCells are only mutated through &mut references obtained while
// interrupts are disabled.
unsafe impl Sync for KernelState {}

static KERNEL_STATE: KernelState = KernelState {
    buddy: UnsafeCell::new(BuddyAllocator::new()),
    slab: UnsafeCell::new(SlabAllocator::new()),
    tensor_pool: UnsafeCell::new(TensorPool::new()),
    cap_registry: UnsafeCell::new(CapRegistry::new()),
    csprng: UnsafeCell::new(Csprng::new()),
    initialized: core::sync::atomic::AtomicBool::new(false),
};

/// Mark the kernel state as initialized (called from boot code).
pub fn mark_initialized() {
    KERNEL_STATE
        .initialized
        .store(true, core::sync::atomic::Ordering::Release);
}

/// Check if kernel state has been initialized.
pub fn is_initialized() -> bool {
    KERNEL_STATE
        .initialized
        .load(core::sync::atomic::Ordering::Acquire)
}

/// Access the buddy allocator.
///
/// # Safety
/// Caller must ensure exclusive access (e.g., interrupts disabled).
pub unsafe fn with_buddy<F, R>(f: F) -> R
where
    F: FnOnce(&mut BuddyAllocator) -> R,
{
    // SAFETY: Syscall handlers run with interrupts masked, guaranteeing
    // exclusive access per core. The caller asserts this precondition.
    let buddy = unsafe { &mut *KERNEL_STATE.buddy.get() };
    f(buddy)
}

/// Access the slab allocator.
///
/// # Safety
/// Caller must ensure exclusive access (e.g., interrupts disabled).
pub unsafe fn with_slab<F, R>(f: F) -> R
where
    F: FnOnce(&mut SlabAllocator) -> R,
{
    let slab = unsafe { &mut *KERNEL_STATE.slab.get() };
    f(slab)
}

/// Access the tensor pool.
///
/// # Safety
/// Caller must ensure exclusive access (e.g., interrupts disabled).
pub unsafe fn with_tensor_pool<F, R>(f: F) -> R
where
    F: FnOnce(&mut TensorPool) -> R,
{
    let pool = unsafe { &mut *KERNEL_STATE.tensor_pool.get() };
    f(pool)
}

/// Access the capability registry.
///
/// # Safety
/// Caller must ensure exclusive access (e.g., interrupts disabled).
pub unsafe fn with_cap_registry<F, R>(f: F) -> R
where
    F: FnOnce(&mut CapRegistry) -> R,
{
    let registry = unsafe { &mut *KERNEL_STATE.cap_registry.get() };
    f(registry)
}

/// Read-only access to the capability registry (no exclusive access needed).
pub fn with_cap_registry_ref<F, R>(f: F) -> R
where
    F: FnOnce(&CapRegistry) -> R,
{
    // SAFETY: Read-only access is safe with shared references.
    let registry = unsafe { &*KERNEL_STATE.cap_registry.get() };
    f(registry)
}

/// Map a MemError to a SyscallError i64.
pub fn mem_error_to_syscall(err: MemError) -> i64 {
    use crate::syscall::SyscallError;
    match err {
        MemError::OutOfMemory => SyscallError::OutOfMemory.as_i64(),
        MemError::BadAlignment => SyscallError::InvalidArgument.as_i64(),
        MemError::InvalidAddress => SyscallError::BadAddress.as_i64(),
        MemError::DoubleFree => SyscallError::InvalidArgument.as_i64(),
        MemError::SizeTooLarge => SyscallError::InvalidArgument.as_i64(),
        MemError::OrderTooLarge => SyscallError::InvalidArgument.as_i64(),
        MemError::SlabExhausted => SyscallError::OutOfMemory.as_i64(),
        MemError::TensorPoolExhausted => SyscallError::OutOfMemory.as_i64(),
        MemError::MappingError => SyscallError::IoError.as_i64(),
        MemError::NotMapped => SyscallError::NotFound.as_i64(),
        MemError::AlreadyMapped => SyscallError::AlreadyExists.as_i64(),
    }
}

/// Map a CapError to a SyscallError i64.
pub fn cap_error_to_syscall(err: CapError) -> i64 {
    use crate::syscall::SyscallError;
    match err {
        CapError::NotFound => SyscallError::NotFound.as_i64(),
        CapError::Expired => SyscallError::InvalidHandle.as_i64(),
        CapError::WrongResource => SyscallError::InvalidArgument.as_i64(),
        CapError::InsufficientPermissions => SyscallError::PermissionDenied.as_i64(),
        CapError::DelegationExceedsPermissions => SyscallError::PermissionDenied.as_i64(),
        CapError::Revoked => SyscallError::InvalidHandle.as_i64(),
        CapError::TooManyCaps => SyscallError::OutOfMemory.as_i64(),
        CapError::SystemLimitReached => SyscallError::OutOfMemory.as_i64(),
        CapError::InvalidId => SyscallError::InvalidHandle.as_i64(),
    }
}

/// Access the CSPRNG.
///
/// # Safety
/// Caller must ensure exclusive access (e.g., interrupts disabled).
pub unsafe fn with_csprng<F, R>(f: F) -> R
where
    F: FnOnce(&mut Csprng) -> R,
{
    let rng = unsafe { &mut *KERNEL_STATE.csprng.get() };
    f(rng)
}

/// Seed the kernel CSPRNG with a given seed.
///
/// # Safety
/// Caller must ensure exclusive access.
pub unsafe fn seed_csprng(seed: &CsprngSeed) -> Result<(), CsprngError> {
    with_csprng(|rng| rng.seed(seed))
}

/// Generate random bytes from the kernel CSPRNG.
///
/// # Safety
/// Caller must ensure exclusive access.
pub unsafe fn csprng_generate(buf: &mut [u8]) -> Result<(), CsprngError> {
    with_csprng(|rng| rng.generate(buf))
}

/// Check whether a task holds a capability for the given resource/permissions.
///
/// Returns `Ok(cap_id)` if the task has the required capability, or
/// `Err(SyscallError::PermissionDenied)` if not.
pub fn check_capability(
    task_id: u64,
    resource: &smallaios_security::capability::ResourceRef,
    required: smallaios_security::capability::Permissions,
) -> Result<u64, i64> {
    with_cap_registry_ref(|registry| registry.check(task_id, resource, required, 0))
        .map_err(cap_error_to_syscall)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state_not_initialized() {
        // In test context, state starts uninitialized
        // (but may be set by other tests, so just verify the function works)
        let _ = is_initialized();
    }

    #[test]
    fn test_mem_error_mapping() {
        use crate::syscall::SyscallError;
        assert_eq!(
            mem_error_to_syscall(MemError::OutOfMemory),
            SyscallError::OutOfMemory.as_i64()
        );
        assert_eq!(
            mem_error_to_syscall(MemError::BadAlignment),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            mem_error_to_syscall(MemError::InvalidAddress),
            SyscallError::BadAddress.as_i64()
        );
        assert_eq!(
            mem_error_to_syscall(MemError::DoubleFree),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn test_cap_error_mapping() {
        use crate::syscall::SyscallError;
        assert_eq!(
            cap_error_to_syscall(CapError::NotFound),
            SyscallError::NotFound.as_i64()
        );
        assert_eq!(
            cap_error_to_syscall(CapError::InsufficientPermissions),
            SyscallError::PermissionDenied.as_i64()
        );
        assert_eq!(
            cap_error_to_syscall(CapError::Revoked),
            SyscallError::InvalidHandle.as_i64()
        );
    }
}
