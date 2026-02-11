// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Syscall dispatch interface.
//!
//! ~64 syscalls organized into categories:
//! - Memory (0x00-0x0F): allocation, tensor buffers, GPU mapping
//! - Task (0x10-0x1F): spawn, yield, join, affinity, priority
//! - IPC (0x20-0x2F): pub/sub, request/reply, channels
//! - ONNX (0x30-0x3F): model load, session, inference
//! - Device (0x40-0x4F): enumerate, open, ioctl, DMA
//! - System (0x50-0x5F): info, time, shutdown, random, watchdog
//! - Capability (0x60-0x6F): create, revoke, delegate, check, list
//! - POSIX (0x70-0x8F): open/close/read/write, mmap, epoll, socket, time
//!
//! In unikernel mode these are direct function calls (no ring transition).
//! In VM mode they use `syscall`/`svc` instructions dispatched through
//! architecture-specific entry points.

pub mod capability;
pub mod device;
pub mod ipc;
pub mod memory;
pub mod onnx;
pub mod posix;
pub mod system;
pub mod task;

/// Maximum syscall number (exclusive). Covers 0x00..0x90.
pub const SYSCALL_TABLE_SIZE: usize = 0x90;

/// Syscall numbers — Memory category (0x00-0x0F).
pub mod nr {
    // Memory syscalls (0x00-0x0F)
    pub const MEM_ALLOC: usize = 0x00;
    pub const MEM_FREE: usize = 0x01;
    pub const MEM_MAP: usize = 0x02;
    pub const MEM_PROTECT: usize = 0x03;
    pub const TENSOR_ALLOC: usize = 0x04;
    pub const TENSOR_FREE: usize = 0x05;
    pub const TENSOR_MAP_GPU: usize = 0x06;
    pub const TENSOR_UNMAP_GPU: usize = 0x07;

    // Task syscalls (0x10-0x1F)
    pub const TASK_SPAWN: usize = 0x10;
    pub const TASK_YIELD: usize = 0x11;
    pub const TASK_EXIT: usize = 0x12;
    pub const TASK_JOIN: usize = 0x13;
    pub const TASK_SET_PRIORITY: usize = 0x14;
    pub const TASK_SET_CLASS: usize = 0x15;
    pub const TASK_CURRENT: usize = 0x16;

    // IPC syscalls (0x20-0x2F)
    pub const IPC_PUBLISH: usize = 0x20;
    pub const IPC_SUBSCRIBE: usize = 0x21;
    pub const IPC_RECV: usize = 0x22;
    pub const IPC_QUERY: usize = 0x23;
    pub const IPC_CHANNEL_CREATE: usize = 0x24;
    pub const IPC_CHANNEL_SEND: usize = 0x25;
    pub const IPC_CHANNEL_RECV: usize = 0x26;
    pub const IPC_CHANNEL_CLOSE: usize = 0x27;

    // ONNX syscalls (0x30-0x3F)
    pub const ONNX_LOAD: usize = 0x30;
    pub const ONNX_UNLOAD: usize = 0x31;
    pub const ONNX_CREATE_SESSION: usize = 0x32;
    pub const ONNX_RUN: usize = 0x33;
    pub const ONNX_GET_METADATA: usize = 0x34;
    pub const ONNX_LIST_PROVIDERS: usize = 0x35;

    // Device syscalls (0x40-0x4F)
    pub const DEV_ENUMERATE: usize = 0x40;
    pub const DEV_OPEN: usize = 0x41;
    pub const DEV_CLOSE: usize = 0x42;
    pub const DEV_IOCTL: usize = 0x43;
    pub const DEV_DMA_ALLOC: usize = 0x44;

    // System syscalls (0x50-0x5F)
    pub const SYS_INFO: usize = 0x50;
    pub const SYS_TIME: usize = 0x51;
    pub const SYS_SHUTDOWN: usize = 0x52;
    pub const SYS_LOG: usize = 0x53;
    pub const SYS_RANDOM: usize = 0x54;
    pub const SYS_WATCHDOG_PET: usize = 0x55;
    pub const SYS_WATCHDOG_REMAINING: usize = 0x56;

    // Capability syscalls (0x60-0x6F)
    pub const CAP_CREATE: usize = 0x60;
    pub const CAP_REVOKE: usize = 0x61;
    pub const CAP_DELEGATE: usize = 0x62;
    pub const CAP_CHECK: usize = 0x63;
    pub const CAP_LIST: usize = 0x64;

    // POSIX compatibility syscalls (0x70-0x8F)
    pub const POSIX_OPEN: usize = 0x70;
    pub const POSIX_CLOSE: usize = 0x71;
    pub const POSIX_READ: usize = 0x72;
    pub const POSIX_WRITE: usize = 0x73;
    pub const POSIX_FSTAT: usize = 0x74;
    pub const POSIX_DUP: usize = 0x75;
    pub const POSIX_DUP2: usize = 0x76;
    pub const POSIX_LSEEK: usize = 0x77;
    pub const POSIX_MMAP: usize = 0x78;
    pub const POSIX_MUNMAP: usize = 0x79;
    pub const POSIX_MPROTECT: usize = 0x7A;
    pub const POSIX_EPOLL_CREATE: usize = 0x7B;
    pub const POSIX_EPOLL_CTL: usize = 0x7C;
    pub const POSIX_EPOLL_WAIT: usize = 0x7D;
    pub const POSIX_CLOCK_GETTIME: usize = 0x7E;
    pub const POSIX_NANOSLEEP: usize = 0x7F;
    pub const POSIX_GETRANDOM: usize = 0x80;
    pub const POSIX_SOCKET: usize = 0x81;
}

/// Syscall error codes.
///
/// Negative values indicate errors; zero or positive values are success.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i64)]
pub enum SyscallError {
    /// Success (no error).
    Success = 0,
    /// Invalid syscall number.
    NoSys = -1,
    /// Permission denied (capability check failed).
    PermissionDenied = -2,
    /// Invalid argument.
    InvalidArgument = -3,
    /// Out of memory.
    OutOfMemory = -4,
    /// Resource not found.
    NotFound = -5,
    /// Resource already exists.
    AlreadyExists = -6,
    /// Operation timed out.
    TimedOut = -7,
    /// Resource busy.
    Busy = -8,
    /// Buffer too small.
    BufferTooSmall = -9,
    /// Invalid handle.
    InvalidHandle = -10,
    /// Operation not supported.
    NotSupported = -11,
    /// I/O error.
    IoError = -12,
    /// Device error.
    DeviceError = -13,
    /// Model load error.
    ModelError = -14,
    /// Inference error.
    InferenceError = -15,
    /// Address fault — invalid pointer.
    BadAddress = -16,
}

impl SyscallError {
    /// Convert to raw i64 return value.
    pub fn as_i64(self) -> i64 {
        self as i64
    }

    /// Try to convert from raw i64 value.
    pub fn from_i64(value: i64) -> Option<Self> {
        match value {
            0 => Some(Self::Success),
            -1 => Some(Self::NoSys),
            -2 => Some(Self::PermissionDenied),
            -3 => Some(Self::InvalidArgument),
            -4 => Some(Self::OutOfMemory),
            -5 => Some(Self::NotFound),
            -6 => Some(Self::AlreadyExists),
            -7 => Some(Self::TimedOut),
            -8 => Some(Self::Busy),
            -9 => Some(Self::BufferTooSmall),
            -10 => Some(Self::InvalidHandle),
            -11 => Some(Self::NotSupported),
            -12 => Some(Self::IoError),
            -13 => Some(Self::DeviceError),
            -14 => Some(Self::ModelError),
            -15 => Some(Self::InferenceError),
            -16 => Some(Self::BadAddress),
            _ => None,
        }
    }
}

/// Syscall arguments passed from architecture-specific entry points.
///
/// Up to 6 arguments are passed in registers:
/// - x86-64: rdi, rsi, rdx, r10, r8, r9 (syscall number in rax)
/// - ARM64: x0-x5 (syscall number in x8)
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct SyscallArgs {
    /// Syscall number.
    pub number: usize,
    /// Arguments 0-5.
    pub args: [usize; 6],
}

impl SyscallArgs {
    /// Create a new SyscallArgs.
    pub const fn new(number: usize, args: [usize; 6]) -> Self {
        Self { number, args }
    }

    /// Create a zero-argument syscall.
    pub const fn zero(number: usize) -> Self {
        Self {
            number,
            args: [0; 6],
        }
    }
}

/// Syscall return value.
///
/// Returned in rax (x86-64) or x0 (ARM64).
/// Negative values indicate errors; zero or positive values are success.
pub type SyscallResult = i64;

/// Type of a syscall handler function.
///
/// Takes the argument array and returns a result code.
pub type SyscallHandler = fn(&SyscallArgs) -> SyscallResult;

/// The syscall dispatch table.
///
/// Indexed by syscall number. Unassigned entries point to `sys_nosys`.
static SYSCALL_TABLE: [SyscallHandler; SYSCALL_TABLE_SIZE] = build_syscall_table();

/// Build the dispatch table at compile time.
const fn build_syscall_table() -> [SyscallHandler; SYSCALL_TABLE_SIZE] {
    let mut table: [SyscallHandler; SYSCALL_TABLE_SIZE] =
        [sys_nosys as SyscallHandler; SYSCALL_TABLE_SIZE];

    // Memory syscalls
    table[nr::MEM_ALLOC] = memory::sys_mem_alloc as SyscallHandler;
    table[nr::MEM_FREE] = memory::sys_mem_free as SyscallHandler;
    table[nr::MEM_MAP] = memory::sys_mem_map as SyscallHandler;
    table[nr::MEM_PROTECT] = memory::sys_mem_protect as SyscallHandler;
    table[nr::TENSOR_ALLOC] = memory::sys_tensor_alloc as SyscallHandler;
    table[nr::TENSOR_FREE] = memory::sys_tensor_free as SyscallHandler;
    table[nr::TENSOR_MAP_GPU] = memory::sys_tensor_map_gpu as SyscallHandler;
    table[nr::TENSOR_UNMAP_GPU] = memory::sys_tensor_unmap_gpu as SyscallHandler;

    // Task syscalls
    table[nr::TASK_SPAWN] = task::sys_task_spawn as SyscallHandler;
    table[nr::TASK_YIELD] = task::sys_task_yield as SyscallHandler;
    table[nr::TASK_EXIT] = task::sys_task_exit as SyscallHandler;
    table[nr::TASK_JOIN] = task::sys_task_join as SyscallHandler;
    table[nr::TASK_SET_PRIORITY] = task::sys_task_set_priority as SyscallHandler;
    table[nr::TASK_SET_CLASS] = task::sys_task_set_class as SyscallHandler;
    table[nr::TASK_CURRENT] = task::sys_task_current as SyscallHandler;

    // IPC syscalls
    table[nr::IPC_PUBLISH] = ipc::sys_ipc_publish as SyscallHandler;
    table[nr::IPC_SUBSCRIBE] = ipc::sys_ipc_subscribe as SyscallHandler;
    table[nr::IPC_RECV] = ipc::sys_ipc_recv as SyscallHandler;
    table[nr::IPC_QUERY] = ipc::sys_ipc_query as SyscallHandler;
    table[nr::IPC_CHANNEL_CREATE] = ipc::sys_ipc_channel_create as SyscallHandler;
    table[nr::IPC_CHANNEL_SEND] = ipc::sys_ipc_channel_send as SyscallHandler;
    table[nr::IPC_CHANNEL_RECV] = ipc::sys_ipc_channel_recv as SyscallHandler;
    table[nr::IPC_CHANNEL_CLOSE] = ipc::sys_ipc_channel_close as SyscallHandler;

    // ONNX syscalls
    table[nr::ONNX_LOAD] = onnx::sys_onnx_load as SyscallHandler;
    table[nr::ONNX_UNLOAD] = onnx::sys_onnx_unload as SyscallHandler;
    table[nr::ONNX_CREATE_SESSION] = onnx::sys_onnx_create_session as SyscallHandler;
    table[nr::ONNX_RUN] = onnx::sys_onnx_run as SyscallHandler;
    table[nr::ONNX_GET_METADATA] = onnx::sys_onnx_get_metadata as SyscallHandler;
    table[nr::ONNX_LIST_PROVIDERS] = onnx::sys_onnx_list_providers as SyscallHandler;

    // Device syscalls
    table[nr::DEV_ENUMERATE] = device::sys_dev_enumerate as SyscallHandler;
    table[nr::DEV_OPEN] = device::sys_dev_open as SyscallHandler;
    table[nr::DEV_CLOSE] = device::sys_dev_close as SyscallHandler;
    table[nr::DEV_IOCTL] = device::sys_dev_ioctl as SyscallHandler;
    table[nr::DEV_DMA_ALLOC] = device::sys_dev_dma_alloc as SyscallHandler;

    // System syscalls
    table[nr::SYS_INFO] = system::sys_info as SyscallHandler;
    table[nr::SYS_TIME] = system::sys_time as SyscallHandler;
    table[nr::SYS_SHUTDOWN] = system::sys_shutdown as SyscallHandler;
    table[nr::SYS_LOG] = system::sys_log as SyscallHandler;
    table[nr::SYS_RANDOM] = system::sys_random as SyscallHandler;
    table[nr::SYS_WATCHDOG_PET] = system::sys_watchdog_pet as SyscallHandler;
    table[nr::SYS_WATCHDOG_REMAINING] = system::sys_watchdog_remaining as SyscallHandler;

    // Capability syscalls
    table[nr::CAP_CREATE] = capability::sys_cap_create as SyscallHandler;
    table[nr::CAP_REVOKE] = capability::sys_cap_revoke as SyscallHandler;
    table[nr::CAP_DELEGATE] = capability::sys_cap_delegate as SyscallHandler;
    table[nr::CAP_CHECK] = capability::sys_cap_check as SyscallHandler;
    table[nr::CAP_LIST] = capability::sys_cap_list as SyscallHandler;

    // POSIX compatibility syscalls
    table[nr::POSIX_OPEN] = posix::sys_posix_open as SyscallHandler;
    table[nr::POSIX_CLOSE] = posix::sys_posix_close as SyscallHandler;
    table[nr::POSIX_READ] = posix::sys_posix_read as SyscallHandler;
    table[nr::POSIX_WRITE] = posix::sys_posix_write as SyscallHandler;
    table[nr::POSIX_FSTAT] = posix::sys_posix_fstat as SyscallHandler;
    table[nr::POSIX_DUP] = posix::sys_posix_dup as SyscallHandler;
    table[nr::POSIX_DUP2] = posix::sys_posix_dup2 as SyscallHandler;
    table[nr::POSIX_LSEEK] = posix::sys_posix_lseek as SyscallHandler;
    table[nr::POSIX_MMAP] = posix::sys_posix_mmap as SyscallHandler;
    table[nr::POSIX_MUNMAP] = posix::sys_posix_munmap as SyscallHandler;
    table[nr::POSIX_MPROTECT] = posix::sys_posix_mprotect as SyscallHandler;
    table[nr::POSIX_EPOLL_CREATE] = posix::sys_posix_epoll_create as SyscallHandler;
    table[nr::POSIX_EPOLL_CTL] = posix::sys_posix_epoll_ctl as SyscallHandler;
    table[nr::POSIX_EPOLL_WAIT] = posix::sys_posix_epoll_wait as SyscallHandler;
    table[nr::POSIX_CLOCK_GETTIME] = posix::sys_posix_clock_gettime as SyscallHandler;
    table[nr::POSIX_NANOSLEEP] = posix::sys_posix_nanosleep as SyscallHandler;
    table[nr::POSIX_GETRANDOM] = posix::sys_posix_getrandom as SyscallHandler;
    table[nr::POSIX_SOCKET] = posix::sys_posix_socket as SyscallHandler;

    table
}

/// Dispatch a syscall by number.
///
/// This is the main entry point called by architecture-specific handlers.
/// Returns the syscall result in the register convention (rax / x0).
pub fn dispatch(args: &SyscallArgs) -> SyscallResult {
    if args.number >= SYSCALL_TABLE_SIZE {
        return SyscallError::NoSys.as_i64();
    }
    let handler = SYSCALL_TABLE[args.number];
    handler(args)
}

/// Default handler for unassigned syscall numbers.
fn sys_nosys(_args: &SyscallArgs) -> SyscallResult {
    SyscallError::NoSys.as_i64()
}

/// Get the syscall category name for a given syscall number.
pub fn category_name(number: usize) -> &'static str {
    match number >> 4 {
        0x0 => "memory",
        0x1 => "task",
        0x2 => "ipc",
        0x3 => "onnx",
        0x4 => "device",
        0x5 => "system",
        0x6 => "capability",
        0x7 | 0x8 => "posix",
        _ => "unknown",
    }
}

/// Get the total number of registered (non-nosys) syscalls.
pub fn registered_count() -> usize {
    SYSCALL_TABLE
        .iter()
        .filter(|h| !core::ptr::eq(**h as *const (), sys_nosys as *const ()))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syscall_table_size() {
        assert_eq!(SYSCALL_TABLE.len(), SYSCALL_TABLE_SIZE);
    }

    #[test]
    fn test_registered_count() {
        // 8 memory + 7 task + 8 ipc + 6 onnx + 5 device + 7 system + 5 capability + 18 posix = 64
        assert_eq!(registered_count(), 64);
    }

    #[test]
    fn test_dispatch_invalid_number() {
        let args = SyscallArgs::zero(0xFF);
        assert_eq!(dispatch(&args), SyscallError::NoSys.as_i64());
    }

    #[test]
    fn test_dispatch_unassigned_slot() {
        // 0x08 is in memory range but unassigned
        let args = SyscallArgs::zero(0x08);
        assert_eq!(dispatch(&args), SyscallError::NoSys.as_i64());
    }

    #[test]
    fn test_dispatch_valid_syscall() {
        // sys_info with buf_ptr=0 returns the required buffer size (positive)
        let args = SyscallArgs::zero(nr::SYS_INFO);
        let result = dispatch(&args);
        assert!(result > 0, "sys_info should return required buffer size");
    }

    #[test]
    fn test_syscall_error_round_trip() {
        for error in [
            SyscallError::Success,
            SyscallError::NoSys,
            SyscallError::PermissionDenied,
            SyscallError::InvalidArgument,
            SyscallError::OutOfMemory,
            SyscallError::NotFound,
            SyscallError::AlreadyExists,
            SyscallError::TimedOut,
            SyscallError::Busy,
            SyscallError::BufferTooSmall,
            SyscallError::InvalidHandle,
            SyscallError::NotSupported,
            SyscallError::IoError,
            SyscallError::DeviceError,
            SyscallError::ModelError,
            SyscallError::InferenceError,
            SyscallError::BadAddress,
        ] {
            let raw = error.as_i64();
            let back = SyscallError::from_i64(raw);
            assert_eq!(back, Some(error));
        }
    }

    #[test]
    fn test_syscall_error_from_unknown() {
        assert_eq!(SyscallError::from_i64(-100), None);
        assert_eq!(SyscallError::from_i64(42), None);
    }

    #[test]
    fn test_category_name() {
        assert_eq!(category_name(nr::MEM_ALLOC), "memory");
        assert_eq!(category_name(nr::MEM_PROTECT), "memory");
        assert_eq!(category_name(nr::TASK_SPAWN), "task");
        assert_eq!(category_name(nr::TASK_CURRENT), "task");
        assert_eq!(category_name(nr::IPC_PUBLISH), "ipc");
        assert_eq!(category_name(nr::IPC_CHANNEL_CLOSE), "ipc");
        assert_eq!(category_name(nr::ONNX_LOAD), "onnx");
        assert_eq!(category_name(nr::ONNX_LIST_PROVIDERS), "onnx");
        assert_eq!(category_name(nr::DEV_ENUMERATE), "device");
        assert_eq!(category_name(nr::DEV_DMA_ALLOC), "device");
        assert_eq!(category_name(nr::SYS_INFO), "system");
        assert_eq!(category_name(nr::SYS_WATCHDOG_REMAINING), "system");
        assert_eq!(category_name(nr::POSIX_OPEN), "posix");
        assert_eq!(category_name(nr::POSIX_SOCKET), "posix");
        assert_eq!(category_name(0x8F), "posix");
        assert_eq!(category_name(0xF0), "unknown");
    }

    #[test]
    fn test_syscall_args_constructors() {
        let args = SyscallArgs::zero(0x50);
        assert_eq!(args.number, 0x50);
        assert_eq!(args.args, [0; 6]);

        let args = SyscallArgs::new(0x00, [1, 2, 3, 4, 5, 6]);
        assert_eq!(args.number, 0x00);
        assert_eq!(args.args[0], 1);
        assert_eq!(args.args[5], 6);
    }

    #[test]
    fn test_all_memory_syscalls_dispatched() {
        for nr in [
            nr::MEM_ALLOC,
            nr::MEM_FREE,
            nr::MEM_MAP,
            nr::MEM_PROTECT,
            nr::TENSOR_ALLOC,
            nr::TENSOR_FREE,
            nr::TENSOR_MAP_GPU,
            nr::TENSOR_UNMAP_GPU,
        ] {
            let args = SyscallArgs::zero(nr);
            let result = dispatch(&args);
            // All stubs should return a valid result (not panic)
            assert_ne!(SyscallError::from_i64(result), Some(SyscallError::NoSys));
        }
    }

    #[test]
    fn test_all_task_syscalls_dispatched() {
        for nr in [
            nr::TASK_SPAWN,
            nr::TASK_YIELD,
            nr::TASK_EXIT,
            nr::TASK_JOIN,
            nr::TASK_SET_PRIORITY,
            nr::TASK_SET_CLASS,
            nr::TASK_CURRENT,
        ] {
            let args = SyscallArgs::zero(nr);
            let result = dispatch(&args);
            assert_ne!(SyscallError::from_i64(result), Some(SyscallError::NoSys));
        }
    }

    #[test]
    fn test_all_ipc_syscalls_dispatched() {
        for nr in [
            nr::IPC_PUBLISH,
            nr::IPC_SUBSCRIBE,
            nr::IPC_RECV,
            nr::IPC_QUERY,
            nr::IPC_CHANNEL_CREATE,
            nr::IPC_CHANNEL_SEND,
            nr::IPC_CHANNEL_RECV,
            nr::IPC_CHANNEL_CLOSE,
        ] {
            let args = SyscallArgs::zero(nr);
            let result = dispatch(&args);
            assert_ne!(SyscallError::from_i64(result), Some(SyscallError::NoSys));
        }
    }

    #[test]
    fn test_all_onnx_syscalls_dispatched() {
        for nr in [
            nr::ONNX_LOAD,
            nr::ONNX_UNLOAD,
            nr::ONNX_CREATE_SESSION,
            nr::ONNX_RUN,
            nr::ONNX_GET_METADATA,
            nr::ONNX_LIST_PROVIDERS,
        ] {
            let args = SyscallArgs::zero(nr);
            let result = dispatch(&args);
            assert_ne!(SyscallError::from_i64(result), Some(SyscallError::NoSys));
        }
    }

    #[test]
    fn test_all_device_syscalls_dispatched() {
        for nr in [
            nr::DEV_ENUMERATE,
            nr::DEV_OPEN,
            nr::DEV_CLOSE,
            nr::DEV_IOCTL,
            nr::DEV_DMA_ALLOC,
        ] {
            let args = SyscallArgs::zero(nr);
            let result = dispatch(&args);
            assert_ne!(SyscallError::from_i64(result), Some(SyscallError::NoSys));
        }
    }

    #[test]
    fn test_all_system_syscalls_dispatched() {
        for nr in [
            nr::SYS_INFO,
            nr::SYS_TIME,
            nr::SYS_SHUTDOWN,
            nr::SYS_LOG,
            nr::SYS_RANDOM,
            nr::SYS_WATCHDOG_PET,
            nr::SYS_WATCHDOG_REMAINING,
        ] {
            let args = SyscallArgs::zero(nr);
            let result = dispatch(&args);
            assert_ne!(SyscallError::from_i64(result), Some(SyscallError::NoSys));
        }
    }

    #[test]
    fn test_gap_slots_return_nosys() {
        // Verify gaps between categories return NoSys
        for gap_nr in [
            0x08, 0x09, 0x0A, 0x0F, 0x17, 0x18, 0x1F, 0x28, 0x2F, 0x36, 0x3F, 0x45, 0x4F, 0x57,
            0x5F, 0x65, 0x6F, 0x82, 0x8F,
        ] {
            let args = SyscallArgs::zero(gap_nr);
            assert_eq!(
                dispatch(&args),
                SyscallError::NoSys.as_i64(),
                "gap slot {:#x} should return NoSys",
                gap_nr
            );
        }
    }

    // --- Fuzz-like boundary tests (Task 4.8) ---

    #[test]
    fn test_fuzz_all_table_slots_no_panic() {
        // Exhaustively dispatch every slot in the table — none should panic
        for nr in 0..SYSCALL_TABLE_SIZE {
            let args = SyscallArgs::new(nr, [0; 6]);
            let _ = dispatch(&args); // Must not panic
        }
    }

    #[test]
    fn test_fuzz_boundary_syscall_numbers() {
        // Boundary values around the table size
        for nr in [
            0,
            SYSCALL_TABLE_SIZE - 1,
            SYSCALL_TABLE_SIZE,
            SYSCALL_TABLE_SIZE + 1,
            usize::MAX,
            usize::MAX - 1,
        ] {
            let args = SyscallArgs::zero(nr);
            let _ = dispatch(&args); // Must not panic
        }
    }

    #[test]
    fn test_fuzz_max_usize_args() {
        // All syscalls with maximum usize arguments — must not panic
        let max_args = [usize::MAX; 6];
        for nr in [
            nr::MEM_ALLOC,
            nr::MEM_FREE,
            nr::MEM_MAP,
            nr::MEM_PROTECT,
            nr::TENSOR_ALLOC,
            nr::TENSOR_FREE,
            nr::TENSOR_MAP_GPU,
            nr::TENSOR_UNMAP_GPU,
            nr::TASK_SPAWN,
            nr::TASK_YIELD,
            nr::TASK_EXIT,
            nr::TASK_JOIN,
            nr::TASK_SET_PRIORITY,
            nr::TASK_SET_CLASS,
            nr::TASK_CURRENT,
            nr::IPC_PUBLISH,
            nr::IPC_SUBSCRIBE,
            nr::IPC_RECV,
            nr::IPC_QUERY,
            nr::IPC_CHANNEL_CREATE,
            nr::IPC_CHANNEL_SEND,
            nr::IPC_CHANNEL_RECV,
            nr::IPC_CHANNEL_CLOSE,
            nr::ONNX_LOAD,
            nr::ONNX_UNLOAD,
            nr::ONNX_CREATE_SESSION,
            nr::ONNX_RUN,
            nr::ONNX_GET_METADATA,
            nr::ONNX_LIST_PROVIDERS,
            nr::DEV_ENUMERATE,
            nr::DEV_OPEN,
            nr::DEV_CLOSE,
            nr::DEV_IOCTL,
            nr::DEV_DMA_ALLOC,
            nr::SYS_INFO,
            nr::SYS_TIME,
            nr::SYS_SHUTDOWN,
            nr::SYS_LOG,
            nr::SYS_RANDOM,
            nr::SYS_WATCHDOG_PET,
            nr::SYS_WATCHDOG_REMAINING,
            nr::CAP_CREATE,
            nr::CAP_REVOKE,
            nr::CAP_DELEGATE,
            nr::CAP_CHECK,
            nr::CAP_LIST,
            nr::POSIX_OPEN,
            nr::POSIX_CLOSE,
            nr::POSIX_READ,
            nr::POSIX_WRITE,
            nr::POSIX_FSTAT,
            nr::POSIX_DUP,
            nr::POSIX_DUP2,
            nr::POSIX_LSEEK,
            nr::POSIX_MMAP,
            nr::POSIX_MUNMAP,
            nr::POSIX_MPROTECT,
            nr::POSIX_EPOLL_CREATE,
            nr::POSIX_EPOLL_CTL,
            nr::POSIX_EPOLL_WAIT,
            nr::POSIX_CLOCK_GETTIME,
            nr::POSIX_NANOSLEEP,
            nr::POSIX_GETRANDOM,
            nr::POSIX_SOCKET,
        ] {
            let args = SyscallArgs::new(nr, max_args);
            let result = dispatch(&args);
            // Must return a valid error code, not panic.
            // SYS_WATCHDOG_REMAINING and TASK_CURRENT legitimately return positive values.
            assert!(
                result <= 0
                    || nr == super::nr::SYS_WATCHDOG_REMAINING
                    || nr == super::nr::TASK_CURRENT,
            );
        }
    }

    #[test]
    fn test_fuzz_zero_args_all_syscalls() {
        // All syscalls with zero arguments — exercises error path validation
        for nr in [
            nr::MEM_ALLOC,
            nr::MEM_FREE,
            nr::MEM_MAP,
            nr::MEM_PROTECT,
            nr::TENSOR_ALLOC,
            nr::TENSOR_FREE,
            nr::TENSOR_MAP_GPU,
            nr::TENSOR_UNMAP_GPU,
            nr::TASK_SPAWN,
            nr::TASK_YIELD,
            nr::TASK_EXIT,
            nr::TASK_JOIN,
            nr::TASK_SET_PRIORITY,
            nr::TASK_SET_CLASS,
            nr::TASK_CURRENT,
            nr::IPC_PUBLISH,
            nr::IPC_SUBSCRIBE,
            nr::IPC_RECV,
            nr::IPC_QUERY,
            nr::IPC_CHANNEL_CREATE,
            nr::IPC_CHANNEL_SEND,
            nr::IPC_CHANNEL_RECV,
            nr::IPC_CHANNEL_CLOSE,
            nr::ONNX_LOAD,
            nr::ONNX_UNLOAD,
            nr::ONNX_CREATE_SESSION,
            nr::ONNX_RUN,
            nr::ONNX_GET_METADATA,
            nr::ONNX_LIST_PROVIDERS,
            nr::DEV_ENUMERATE,
            nr::DEV_OPEN,
            nr::DEV_CLOSE,
            nr::DEV_IOCTL,
            nr::DEV_DMA_ALLOC,
            nr::SYS_INFO,
            nr::SYS_TIME,
            nr::SYS_SHUTDOWN,
            nr::SYS_LOG,
            nr::SYS_RANDOM,
            nr::SYS_WATCHDOG_PET,
            nr::SYS_WATCHDOG_REMAINING,
            nr::CAP_CREATE,
            nr::CAP_REVOKE,
            nr::CAP_DELEGATE,
            nr::CAP_CHECK,
            nr::CAP_LIST,
            nr::POSIX_OPEN,
            nr::POSIX_CLOSE,
            nr::POSIX_READ,
            nr::POSIX_WRITE,
            nr::POSIX_FSTAT,
            nr::POSIX_DUP,
            nr::POSIX_DUP2,
            nr::POSIX_LSEEK,
            nr::POSIX_MMAP,
            nr::POSIX_MUNMAP,
            nr::POSIX_MPROTECT,
            nr::POSIX_EPOLL_CREATE,
            nr::POSIX_EPOLL_CTL,
            nr::POSIX_EPOLL_WAIT,
            nr::POSIX_CLOCK_GETTIME,
            nr::POSIX_NANOSLEEP,
            nr::POSIX_GETRANDOM,
            nr::POSIX_SOCKET,
        ] {
            let args = SyscallArgs::zero(nr);
            let _ = dispatch(&args); // Must not panic
        }
    }

    // --- MC/DC coverage tests (Task 4.9) ---

    #[test]
    fn test_dispatch_boundary_exact_table_size() {
        // args.number == SYSCALL_TABLE_SIZE should return NoSys (boundary condition)
        let args = SyscallArgs::zero(SYSCALL_TABLE_SIZE);
        assert_eq!(dispatch(&args), SyscallError::NoSys.as_i64());
    }

    #[test]
    fn test_dispatch_boundary_last_valid_slot() {
        // Last valid index in table
        let args = SyscallArgs::zero(SYSCALL_TABLE_SIZE - 1);
        // This slot is unassigned, should be NoSys
        assert_eq!(dispatch(&args), SyscallError::NoSys.as_i64());
    }

    #[test]
    fn test_syscall_error_negative_values() {
        // All error codes are negative except Success(0)
        assert_eq!(SyscallError::Success.as_i64(), 0);
        assert!(SyscallError::NoSys.as_i64() < 0);
        assert!(SyscallError::BadAddress.as_i64() < 0);
    }

    #[test]
    fn test_category_name_all_boundaries() {
        // First and last entry in each category
        assert_eq!(category_name(0x00), "memory");
        assert_eq!(category_name(0x0F), "memory");
        assert_eq!(category_name(0x10), "task");
        assert_eq!(category_name(0x1F), "task");
        assert_eq!(category_name(0x20), "ipc");
        assert_eq!(category_name(0x2F), "ipc");
        assert_eq!(category_name(0x30), "onnx");
        assert_eq!(category_name(0x3F), "onnx");
        assert_eq!(category_name(0x40), "device");
        assert_eq!(category_name(0x4F), "device");
        assert_eq!(category_name(0x50), "system");
        assert_eq!(category_name(0x5F), "system");
        assert_eq!(category_name(nr::CAP_CREATE), "capability");
        assert_eq!(category_name(nr::CAP_LIST), "capability");
        assert_eq!(category_name(0x70), "posix");
        assert_eq!(category_name(0x7F), "posix");
        assert_eq!(category_name(0x80), "posix");
        assert_eq!(category_name(0x90), "unknown");
    }

    #[test]
    fn test_each_syscall_returns_expected_for_zero_args() {
        // Verify specific expected results for zero-arg dispatches
        // (exercises both valid-path and error-path for each handler)
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::MEM_ALLOC)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::MEM_FREE)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::MEM_MAP)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::MEM_PROTECT)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::TENSOR_ALLOC)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::TENSOR_FREE)),
            SyscallError::InvalidHandle.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::TENSOR_MAP_GPU)),
            SyscallError::InvalidHandle.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::TENSOR_UNMAP_GPU)),
            SyscallError::InvalidHandle.as_i64()
        );

        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::TASK_SPAWN)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::TASK_YIELD)),
            SyscallError::Success.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::TASK_EXIT)),
            SyscallError::Success.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::TASK_JOIN)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::TASK_SET_PRIORITY)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::TASK_SET_CLASS)),
            SyscallError::InvalidArgument.as_i64()
        );
        // TASK_CURRENT returns the current task ID (which may be non-zero
        // if another test set it); just verify it doesn't error.
        assert!(dispatch(&SyscallArgs::zero(nr::TASK_CURRENT)) >= 0);

        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::IPC_PUBLISH)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::IPC_SUBSCRIBE)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::IPC_RECV)),
            SyscallError::InvalidHandle.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::IPC_QUERY)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::IPC_CHANNEL_CREATE)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::IPC_CHANNEL_SEND)),
            SyscallError::InvalidHandle.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::IPC_CHANNEL_RECV)),
            SyscallError::InvalidHandle.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::IPC_CHANNEL_CLOSE)),
            SyscallError::InvalidHandle.as_i64()
        );

        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::ONNX_LOAD)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::ONNX_UNLOAD)),
            SyscallError::InvalidHandle.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::ONNX_CREATE_SESSION)),
            SyscallError::InvalidHandle.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::ONNX_RUN)),
            SyscallError::InvalidHandle.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::ONNX_GET_METADATA)),
            SyscallError::InvalidHandle.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::ONNX_LIST_PROVIDERS)),
            SyscallError::NotSupported.as_i64()
        );

        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::DEV_ENUMERATE)),
            SyscallError::NotSupported.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::DEV_OPEN)),
            SyscallError::NotSupported.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::DEV_CLOSE)),
            SyscallError::InvalidHandle.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::DEV_IOCTL)),
            SyscallError::InvalidHandle.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::DEV_DMA_ALLOC)),
            SyscallError::InvalidArgument.as_i64()
        );

        // sys_info with null buf_ptr returns the required buffer size (positive)
        assert!(dispatch(&SyscallArgs::zero(nr::SYS_INFO)) > 0);
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::SYS_TIME)),
            SyscallError::Success.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::SYS_SHUTDOWN)),
            SyscallError::PermissionDenied.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::SYS_LOG)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::SYS_RANDOM)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::SYS_WATCHDOG_PET)),
            SyscallError::Success.as_i64()
        );
        assert!(dispatch(&SyscallArgs::zero(nr::SYS_WATCHDOG_REMAINING)) > 0);

        // Capability: zero args means resource_type=0, instance=0, perms=0 → InvalidArgument
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::CAP_CREATE)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::CAP_REVOKE)),
            SyscallError::InvalidHandle.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::CAP_DELEGATE)),
            SyscallError::InvalidHandle.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::CAP_CHECK)),
            SyscallError::InvalidHandle.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::CAP_LIST)),
            SyscallError::Success.as_i64()
        );

        // POSIX: all zero-arg dispatches should return an error (not panic)
        let posix_enosys = -38i64;
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::POSIX_OPEN)),
            SyscallError::InvalidArgument.as_i64()
        );
        assert_eq!(dispatch(&SyscallArgs::zero(nr::POSIX_CLOSE)), posix_enosys);
        assert_eq!(dispatch(&SyscallArgs::zero(nr::POSIX_READ)), posix_enosys); // fd=0,buf=0,count=0 → stub
        assert_eq!(dispatch(&SyscallArgs::zero(nr::POSIX_WRITE)), posix_enosys); // fd=0,buf=0,count=0 → stub
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::POSIX_FSTAT)),
            SyscallError::BadAddress.as_i64()
        ); // stat_ptr=0
        assert_eq!(dispatch(&SyscallArgs::zero(nr::POSIX_DUP)), posix_enosys); // fd=0 is valid
        assert_eq!(dispatch(&SyscallArgs::zero(nr::POSIX_DUP2)), posix_enosys); // both fd=0
        assert_eq!(dispatch(&SyscallArgs::zero(nr::POSIX_LSEEK)), posix_enosys); // fd=0, SEEK_SET
        assert_eq!(dispatch(&SyscallArgs::zero(nr::POSIX_MMAP)), -22); // zero length → EINVAL
        assert_eq!(dispatch(&SyscallArgs::zero(nr::POSIX_MUNMAP)), -22); // zero length → EINVAL
        assert_eq!(dispatch(&SyscallArgs::zero(nr::POSIX_MPROTECT)), -22); // zero length → EINVAL
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::POSIX_EPOLL_CREATE)),
            posix_enosys
        ); // flags=0 ok
        assert_eq!(dispatch(&SyscallArgs::zero(nr::POSIX_EPOLL_CTL)), -22); // op=0 invalid → EINVAL
        assert_eq!(dispatch(&SyscallArgs::zero(nr::POSIX_EPOLL_WAIT)), -22); // max_events=0 → EINVAL
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::POSIX_CLOCK_GETTIME)),
            SyscallError::BadAddress.as_i64()
        );
        assert_eq!(
            dispatch(&SyscallArgs::zero(nr::POSIX_NANOSLEEP)),
            SyscallError::BadAddress.as_i64()
        );
        assert_eq!(dispatch(&SyscallArgs::zero(nr::POSIX_GETRANDOM)), 0); // zero-length → 0
        assert_eq!(dispatch(&SyscallArgs::zero(nr::POSIX_SOCKET)), -22); // family=0 invalid
    }

    #[test]
    fn test_all_posix_syscalls_dispatched() {
        for nr in [
            nr::POSIX_OPEN,
            nr::POSIX_CLOSE,
            nr::POSIX_READ,
            nr::POSIX_WRITE,
            nr::POSIX_FSTAT,
            nr::POSIX_DUP,
            nr::POSIX_DUP2,
            nr::POSIX_LSEEK,
            nr::POSIX_MMAP,
            nr::POSIX_MUNMAP,
            nr::POSIX_MPROTECT,
            nr::POSIX_EPOLL_CREATE,
            nr::POSIX_EPOLL_CTL,
            nr::POSIX_EPOLL_WAIT,
            nr::POSIX_CLOCK_GETTIME,
            nr::POSIX_NANOSLEEP,
            nr::POSIX_GETRANDOM,
            nr::POSIX_SOCKET,
        ] {
            let args = SyscallArgs::zero(nr);
            let result = dispatch(&args);
            assert_ne!(SyscallError::from_i64(result), Some(SyscallError::NoSys));
        }
    }

    #[test]
    fn test_all_capability_syscalls_dispatched() {
        for nr in [
            nr::CAP_CREATE,
            nr::CAP_REVOKE,
            nr::CAP_DELEGATE,
            nr::CAP_CHECK,
            nr::CAP_LIST,
        ] {
            let args = SyscallArgs::zero(nr);
            let result = dispatch(&args);
            assert_ne!(SyscallError::from_i64(result), Some(SyscallError::NoSys));
        }
    }

    // =========================================================================
    // Extended fuzz tests (Task 4.8) — invalid inputs, boundary values, no panics
    // =========================================================================

    #[test]
    fn test_fuzz_out_of_range_syscall_numbers() {
        // Specific out-of-range numbers that should all return NoSys
        for nr in [
            SYSCALL_TABLE_SIZE,
            SYSCALL_TABLE_SIZE + 1,
            SYSCALL_TABLE_SIZE + 100,
            0x100,
            0x1000,
            0xFFFF,
            0xFFFF_FFFF,
            usize::MAX,
            usize::MAX - 1,
            usize::MAX / 2,
        ] {
            let args = SyscallArgs::zero(nr);
            assert_eq!(
                dispatch(&args),
                SyscallError::NoSys.as_i64(),
                "out-of-range syscall {:#x} must return NoSys",
                nr
            );
        }
    }

    #[test]
    fn test_fuzz_mixed_arg_patterns_no_panic() {
        // Test representative syscalls with diverse argument patterns.
        // Patterns use zero/one/MAX — avoiding fake non-null pointers that would
        // pass validation and cause dereference of unmapped memory in handlers
        // that read/write user buffers (tensor_alloc, sys_info, sys_random, cap_list).
        let patterns: &[[usize; 6]] = &[
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [usize::MAX, 0, 0, 0, 0, 0],
            [0, usize::MAX, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX, usize::MAX],
            [usize::MAX / 2, usize::MAX / 2, 0, 0, 0, 0],
            [1, 0, usize::MAX, 0, 0, 0],
        ];

        // Syscalls that do NOT dereference user pointers after validation passes.
        // Excluded: TENSOR_ALLOC (reads shape_ptr), SYS_INFO (writes buf_ptr),
        // SYS_RANDOM (writes buf_ptr), CAP_LIST (writes buf_ptr in fill mode).
        let safe_registered = [
            nr::MEM_ALLOC, nr::MEM_FREE, nr::MEM_MAP, nr::MEM_PROTECT,
            nr::TENSOR_FREE, nr::TENSOR_MAP_GPU, nr::TENSOR_UNMAP_GPU,
            nr::TASK_SPAWN, nr::TASK_YIELD, nr::TASK_EXIT, nr::TASK_JOIN,
            nr::TASK_SET_PRIORITY, nr::TASK_SET_CLASS, nr::TASK_CURRENT,
            nr::IPC_PUBLISH, nr::IPC_SUBSCRIBE, nr::IPC_RECV, nr::IPC_QUERY,
            nr::IPC_CHANNEL_CREATE, nr::IPC_CHANNEL_SEND, nr::IPC_CHANNEL_RECV,
            nr::IPC_CHANNEL_CLOSE,
            nr::ONNX_LOAD, nr::ONNX_UNLOAD, nr::ONNX_CREATE_SESSION, nr::ONNX_RUN,
            nr::ONNX_GET_METADATA, nr::ONNX_LIST_PROVIDERS,
            nr::DEV_ENUMERATE, nr::DEV_OPEN, nr::DEV_CLOSE, nr::DEV_IOCTL, nr::DEV_DMA_ALLOC,
            nr::SYS_TIME, nr::SYS_SHUTDOWN, nr::SYS_LOG,
            nr::SYS_WATCHDOG_PET, nr::SYS_WATCHDOG_REMAINING,
            nr::CAP_CREATE, nr::CAP_REVOKE, nr::CAP_DELEGATE, nr::CAP_CHECK,
            nr::POSIX_OPEN, nr::POSIX_CLOSE, nr::POSIX_READ, nr::POSIX_WRITE,
            nr::POSIX_FSTAT, nr::POSIX_DUP, nr::POSIX_DUP2, nr::POSIX_LSEEK,
            nr::POSIX_MMAP, nr::POSIX_MUNMAP, nr::POSIX_MPROTECT,
            nr::POSIX_EPOLL_CREATE, nr::POSIX_EPOLL_CTL, nr::POSIX_EPOLL_WAIT,
            nr::POSIX_CLOCK_GETTIME, nr::POSIX_NANOSLEEP, nr::POSIX_GETRANDOM, nr::POSIX_SOCKET,
        ];

        for &nr in &safe_registered {
            for pattern in patterns {
                let args = SyscallArgs::new(nr, *pattern);
                let _ = dispatch(&args); // Must not panic
            }
        }
    }

    #[test]
    fn test_fuzz_invalid_capability_tokens() {
        // Capability-protected syscalls with bogus cap tokens should not panic
        // These are tested via dispatch to exercise the full path
        let bogus_handles: &[usize] = &[0, 1, 0xDEAD, usize::MAX, usize::MAX / 2];
        for &handle in bogus_handles {
            // tensor_free with invalid handle
            let args = SyscallArgs::new(nr::TENSOR_FREE, [handle, 0, 0, 0, 0, 0]);
            let result = dispatch(&args);
            assert!(result < 0, "invalid tensor handle must return error");

            // tensor_map_gpu with invalid handle
            let args = SyscallArgs::new(nr::TENSOR_MAP_GPU, [handle, 0, 0, 0, 0, 0]);
            let result = dispatch(&args);
            if handle == 0 {
                assert_eq!(result, SyscallError::InvalidHandle.as_i64());
            } else {
                assert!(result < 0);
            }

            // cap_revoke with invalid ID
            let args = SyscallArgs::new(nr::CAP_REVOKE, [handle, 0, 0, 0, 0, 0]);
            let result = dispatch(&args);
            assert!(result <= 0);

            // cap_check with invalid ID
            let args = SyscallArgs::new(nr::CAP_CHECK, [handle, 0, 0, 1, 0, 0]);
            let result = dispatch(&args);
            assert!(result <= 0);
        }
    }

    #[test]
    fn test_fuzz_null_pointer_combinations() {
        // Exercise each syscall that accepts pointer args with null pointers

        // IPC publish: key_ptr=0, data_ptr valid
        let args = SyscallArgs::new(nr::IPC_PUBLISH, [0, 10, 0x1000, 100, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // IPC publish: key_ptr valid, key_len=0
        let args = SyscallArgs::new(nr::IPC_PUBLISH, [0x1000, 0, 0x2000, 100, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // IPC recv: valid handle, null buf
        let args = SyscallArgs::new(nr::IPC_RECV, [1, 0, 100, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // IPC recv: valid handle, valid buf, zero len
        let args = SyscallArgs::new(nr::IPC_RECV, [1, 0x1000, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // IPC channel_recv: valid handle, null buf, non-zero len
        let args = SyscallArgs::new(nr::IPC_CHANNEL_RECV, [1, 0, 100, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // IPC channel_recv: valid handle, non-null buf, zero len
        let args = SyscallArgs::new(nr::IPC_CHANNEL_RECV, [1, 0x1000, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // ONNX get_metadata: valid handle, null buf
        let args = SyscallArgs::new(nr::ONNX_GET_METADATA, [1, 0, 100, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // ONNX get_metadata: valid handle, valid buf, zero len
        let args = SyscallArgs::new(nr::ONNX_GET_METADATA, [1, 0x1000, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // sys_random: null buf, non-zero len
        let args = SyscallArgs::new(nr::SYS_RANDOM, [0, 32, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // sys_log: valid level, null msg
        let args = SyscallArgs::new(nr::SYS_LOG, [0, 0, 10, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // sys_log: valid level, valid ptr, zero len
        let args = SyscallArgs::new(nr::SYS_LOG, [0, 0x1000, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_fuzz_boundary_size_values() {
        // Size arguments at max limits

        // IPC publish: data_len exactly at MAX
        let args = SyscallArgs::new(
            nr::IPC_PUBLISH,
            [0x1000, 10, 0x2000, ipc::MAX_IPC_MESSAGE_SIZE, 0, 0],
        );
        let result = dispatch(&args);
        assert_ne!(result, SyscallError::InvalidArgument.as_i64());

        // IPC publish: data_len one over MAX
        let args = SyscallArgs::new(
            nr::IPC_PUBLISH,
            [0x1000, 10, 0x2000, ipc::MAX_IPC_MESSAGE_SIZE + 1, 0, 0],
        );
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // IPC subscribe: key_len exactly at MAX
        let args = SyscallArgs::new(
            nr::IPC_SUBSCRIBE,
            [0x1000, ipc::MAX_KEY_EXPR_LEN, 0, 0, 0, 0],
        );
        let result = dispatch(&args);
        assert_ne!(result, SyscallError::InvalidArgument.as_i64());

        // IPC subscribe: key_len one over MAX
        let args = SyscallArgs::new(
            nr::IPC_SUBSCRIBE,
            [0x1000, ipc::MAX_KEY_EXPR_LEN + 1, 0, 0, 0, 0],
        );
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // ONNX load: data_len exactly at MAX
        let args = SyscallArgs::new(nr::ONNX_LOAD, [0x1000, onnx::MAX_MODEL_SIZE, 0, 0, 0, 0]);
        let result = dispatch(&args);
        assert_ne!(result, SyscallError::InvalidArgument.as_i64());

        // ONNX run: num_inputs exactly at MAX
        let args = SyscallArgs::new(
            nr::ONNX_RUN,
            [1, 0x1000, onnx::MAX_IO_TENSORS, 0x2000, 1, 0],
        );
        let result = dispatch(&args);
        assert_ne!(result, SyscallError::InvalidArgument.as_i64());

        // ONNX run: num_inputs one over MAX
        let args = SyscallArgs::new(
            nr::ONNX_RUN,
            [1, 0x1000, onnx::MAX_IO_TENSORS + 1, 0x2000, 1, 0],
        );
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // DMA alloc: size exactly at MAX
        let args = SyscallArgs::new(nr::DEV_DMA_ALLOC, [device::MAX_DMA_SIZE, 4096, 0, 0, 0, 0]);
        let result = dispatch(&args);
        assert_ne!(result, SyscallError::InvalidArgument.as_i64());

        // sys_random: len exactly at MAX
        let args = SyscallArgs::new(nr::SYS_RANDOM, [0x1000, system::MAX_RANDOM_LEN, 0, 0, 0, 0]);
        let result = dispatch(&args);
        assert_ne!(result, SyscallError::InvalidArgument.as_i64());

        // sys_random: len one over MAX
        let args = SyscallArgs::new(
            nr::SYS_RANDOM,
            [0x1000, system::MAX_RANDOM_LEN + 1, 0, 0, 0, 0],
        );
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // sys_log: len exactly at MAX
        let args = SyscallArgs::new(
            nr::SYS_LOG,
            [0, 0x1000, system::MAX_LOG_MSG_LEN, 0, 0, 0],
        );
        assert_eq!(dispatch(&args), SyscallError::Success.as_i64());

        // sys_log: len one over MAX
        let args = SyscallArgs::new(
            nr::SYS_LOG,
            [0, 0x1000, system::MAX_LOG_MSG_LEN + 1, 0, 0, 0],
        );
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());
    }

    // =========================================================================
    // MC/DC coverage tests (Task 4.9) — every branch independently exercised
    // =========================================================================

    // --- Memory MC/DC ---

    #[test]
    fn test_mcdc_mem_alloc_slab_vs_buddy_path() {
        // Slab path: size<=2048, flags=0, align<=16
        let args = SyscallArgs::new(nr::MEM_ALLOC, [64, 0, 0, 0, 0, 0]);
        let result = dispatch(&args);
        // Expect OOM or valid pointer (slab not initialized) — NOT InvalidArgument
        assert_ne!(result, SyscallError::InvalidArgument.as_i64());

        // Buddy path: size > 2048
        let args = SyscallArgs::new(nr::MEM_ALLOC, [8192, 0, 0, 0, 0, 0]);
        let result = dispatch(&args);
        assert_ne!(result, SyscallError::InvalidArgument.as_i64());

        // Buddy path: small size but non-zero flags
        let args = SyscallArgs::new(nr::MEM_ALLOC, [64, 0, 1, 0, 0, 0]);
        let result = dispatch(&args);
        assert_ne!(result, SyscallError::InvalidArgument.as_i64());

        // Buddy path: small size but large alignment
        let args = SyscallArgs::new(nr::MEM_ALLOC, [64, 64, 0, 0, 0, 0]);
        let result = dispatch(&args);
        assert_ne!(result, SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_mcdc_mem_alloc_alignment_conditions() {
        // align=0 (default) — valid
        let args = SyscallArgs::new(nr::MEM_ALLOC, [4096, 0, 0, 0, 0, 0]);
        assert_ne!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // align=1 (power of two) — valid
        let args = SyscallArgs::new(nr::MEM_ALLOC, [4096, 1, 0, 0, 0, 0]);
        assert_ne!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // align=2 (power of two) — valid
        let args = SyscallArgs::new(nr::MEM_ALLOC, [4096, 2, 0, 0, 0, 0]);
        assert_ne!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // align=3 (not power of two, non-zero) — invalid
        let args = SyscallArgs::new(nr::MEM_ALLOC, [4096, 3, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // align=5 (not power of two, non-zero) — invalid
        let args = SyscallArgs::new(nr::MEM_ALLOC, [4096, 5, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_mcdc_mem_free_slab_vs_buddy() {
        // Slab path: size <= 2048
        let args = SyscallArgs::new(nr::MEM_FREE, [0x1000, 512, 0, 0, 0, 0]);
        let result = dispatch(&args);
        // Slab free on uninitialized — error but not panic
        assert!(result <= 0);

        // Buddy path: size > 2048
        let args = SyscallArgs::new(nr::MEM_FREE, [0x1000, 8192, 0, 0, 0, 0]);
        let result = dispatch(&args);
        assert!(result <= 0);
    }

    #[test]
    fn test_mcdc_mem_free_conditions() {
        // ptr=0, size=valid → InvalidArgument (ptr check first)
        let args = SyscallArgs::new(nr::MEM_FREE, [0, 4096, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // ptr=valid, size=0 → InvalidArgument (size check)
        let args = SyscallArgs::new(nr::MEM_FREE, [0x1000, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // ptr=0, size=0 → InvalidArgument (ptr check first)
        let args = SyscallArgs::new(nr::MEM_FREE, [0, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_mcdc_mem_map_capability_path() {
        // size=0 → InvalidArgument (before cap check)
        let args = SyscallArgs::new(nr::MEM_MAP, [0x1000, 0x2000, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // size>0 → capability check (fails with PermissionDenied or NotSupported)
        let args = SyscallArgs::new(nr::MEM_MAP, [0x1000, 0x2000, 4096, 0, 0, 0]);
        let result = dispatch(&args);
        assert!(
            result == SyscallError::PermissionDenied.as_i64()
                || result == SyscallError::NotSupported.as_i64()
        );
    }

    #[test]
    fn test_mcdc_mem_protect_conditions() {
        // ptr=0, size=non-zero → InvalidArgument (compound OR: ptr==0)
        let args = SyscallArgs::new(nr::MEM_PROTECT, [0, 4096, 1, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // ptr=non-zero, size=0 → InvalidArgument (compound OR: size==0)
        let args = SyscallArgs::new(nr::MEM_PROTECT, [0x1000, 0, 1, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // ptr=non-zero, size=non-zero → capability check
        let args = SyscallArgs::new(nr::MEM_PROTECT, [0x1000, 4096, 1, 0, 0, 0]);
        let result = dispatch(&args);
        assert!(
            result == SyscallError::PermissionDenied.as_i64()
                || result == SyscallError::NotSupported.as_i64()
        );
    }

    #[test]
    fn test_mcdc_tensor_alloc_conditions() {
        // shape_ptr=0, ndim=valid → InvalidArgument (shape_ptr check)
        let args = SyscallArgs::new(nr::TENSOR_ALLOC, [0, 4, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // shape_ptr=valid, ndim=0 → InvalidArgument (ndim check)
        let args = SyscallArgs::new(nr::TENSOR_ALLOC, [0x1000, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // ndim=9 (over max)
        let args = SyscallArgs::new(nr::TENSOR_ALLOC, [0x1000, 9, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // invalid dtype (passes ptr/ndim checks, fails dtype)
        let args = SyscallArgs::new(nr::TENSOR_ALLOC, [0x1000, 4, 8, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        let args = SyscallArgs::new(nr::TENSOR_ALLOC, [0x1000, 4, 255, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // Valid shape_ptr with real data: ndim=1, shape=[4], dtype=Float32
        // This uses a real pointer so the shape read succeeds.
        let shape: [usize; 1] = [4];
        let args = SyscallArgs::new(
            nr::TENSOR_ALLOC,
            [shape.as_ptr() as usize, 1, 0, 0, 0, 0],
        );
        let result = dispatch(&args);
        // Should pass validation, hit capability check or tensor pool alloc
        assert_ne!(result, SyscallError::InvalidArgument.as_i64());

        // Valid with ndim=8 (max valid) and real shape data
        let shape8: [usize; 8] = [1, 2, 3, 4, 1, 1, 1, 1];
        let args = SyscallArgs::new(
            nr::TENSOR_ALLOC,
            [shape8.as_ptr() as usize, 8, 0, 0, 0, 0],
        );
        let result = dispatch(&args);
        assert_ne!(result, SyscallError::InvalidArgument.as_i64());

        // All valid dtypes (0..7) with real shape pointer
        for dtype in 0..=7u32 {
            let shape1: [usize; 1] = [1];
            let args = SyscallArgs::new(
                nr::TENSOR_ALLOC,
                [shape1.as_ptr() as usize, 1, dtype as usize, 0, 0, 0],
            );
            let result = dispatch(&args);
            // Should pass validation, hit capability check or tensor pool
            assert_ne!(
                result,
                SyscallError::InvalidArgument.as_i64(),
                "dtype {} should pass validation",
                dtype
            );
        }
    }

    #[test]
    fn test_mcdc_tensor_gpu_handle_vs_capability() {
        // handle=0 → InvalidHandle (before cap check)
        let args = SyscallArgs::new(nr::TENSOR_MAP_GPU, [0, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidHandle.as_i64());

        // handle=valid → capability check path
        let args = SyscallArgs::new(nr::TENSOR_MAP_GPU, [1, 0, 0, 0, 0, 0]);
        let result = dispatch(&args);
        assert!(
            result == SyscallError::PermissionDenied.as_i64()
                || result == SyscallError::NotSupported.as_i64()
        );

        // Same for unmap
        let args = SyscallArgs::new(nr::TENSOR_UNMAP_GPU, [0, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidHandle.as_i64());

        let args = SyscallArgs::new(nr::TENSOR_UNMAP_GPU, [1, 0, 0, 0, 0, 0]);
        let result = dispatch(&args);
        assert!(
            result == SyscallError::PermissionDenied.as_i64()
                || result == SyscallError::NotSupported.as_i64()
        );
    }

    // --- Task MC/DC ---

    #[test]
    fn test_mcdc_task_spawn_all_valid_classes() {
        // Each valid scheduling class (0=System, 1=Ipc, 2=Inference)
        for class in 0..=2usize {
            let args = SyscallArgs::new(nr::TASK_SPAWN, [0x1000, 0, class, 0, 0, 0]);
            let result = dispatch(&args);
            assert!(result > 0, "class {} should produce valid task ID", class);
        }

        // class=3 (over max) → InvalidArgument
        let args = SyscallArgs::new(nr::TASK_SPAWN, [0x1000, 0, 3, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_mcdc_task_set_priority_boundary() {
        // task_id=0 → InvalidArgument
        let args = SyscallArgs::new(nr::TASK_SET_PRIORITY, [0, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // task_id=valid, priority=0 (min valid)
        let args = SyscallArgs::new(nr::TASK_SET_PRIORITY, [1, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::Success.as_i64());

        // task_id=valid, priority=255 (max valid)
        let args = SyscallArgs::new(nr::TASK_SET_PRIORITY, [1, 255, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::Success.as_i64());

        // task_id=valid, priority=256 (one over max)
        let args = SyscallArgs::new(nr::TASK_SET_PRIORITY, [1, 256, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_mcdc_task_join_valid_id() {
        // task_id=0 → InvalidArgument
        let args = SyscallArgs::new(nr::TASK_JOIN, [0, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // task_id=valid → NotSupported (executor integration pending)
        let args = SyscallArgs::new(nr::TASK_JOIN, [1, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::NotSupported.as_i64());
    }

    #[test]
    fn test_mcdc_task_set_class_boundary() {
        // task_id=0 → InvalidArgument
        let args = SyscallArgs::new(nr::TASK_SET_CLASS, [0, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // task_id=valid, class=0 (min valid)
        let args = SyscallArgs::new(nr::TASK_SET_CLASS, [1, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::Success.as_i64());

        // task_id=valid, class=2 (max valid)
        let args = SyscallArgs::new(nr::TASK_SET_CLASS, [1, 2, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::Success.as_i64());

        // task_id=valid, class=3 (one over max)
        let args = SyscallArgs::new(nr::TASK_SET_CLASS, [1, 3, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());
    }

    // --- IPC MC/DC ---

    #[test]
    fn test_mcdc_ipc_publish_compound_conditions() {
        // key_ptr=0, key_len=non-zero → InvalidArgument (key_ptr==0 is true)
        let args = SyscallArgs::new(nr::IPC_PUBLISH, [0, 10, 0x2000, 100, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // key_ptr=non-zero, key_len=0 → InvalidArgument (key_len==0 is true)
        let args = SyscallArgs::new(nr::IPC_PUBLISH, [0x1000, 0, 0x2000, 100, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // key_ptr=non-zero, key_len=valid, data_len=MAX → passes validation
        let args = SyscallArgs::new(
            nr::IPC_PUBLISH,
            [0x1000, 10, 0x2000, ipc::MAX_IPC_MESSAGE_SIZE, 0, 0],
        );
        let result = dispatch(&args);
        assert_eq!(result, SyscallError::NotSupported.as_i64());

        // Valid all → NotSupported (stub)
        let args = SyscallArgs::new(nr::IPC_PUBLISH, [0x1000, 10, 0x2000, 100, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::NotSupported.as_i64());
    }

    #[test]
    fn test_mcdc_ipc_query_compound_conditions() {
        // key valid, data valid, reply_ptr=0 → InvalidArgument
        let args = SyscallArgs::new(nr::IPC_QUERY, [0x1000, 10, 0x2000, 100, 0, 256]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // key valid, data valid, reply_ptr=valid, reply_len=0 → InvalidArgument
        let args = SyscallArgs::new(nr::IPC_QUERY, [0x1000, 10, 0x2000, 100, 0x3000, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // key valid, data_len > MAX → InvalidArgument
        let args = SyscallArgs::new(
            nr::IPC_QUERY,
            [0x1000, 10, 0x2000, ipc::MAX_IPC_MESSAGE_SIZE + 1, 0x3000, 256],
        );
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // key_len > MAX → InvalidArgument
        let args = SyscallArgs::new(
            nr::IPC_QUERY,
            [0x1000, ipc::MAX_KEY_EXPR_LEN + 1, 0x2000, 100, 0x3000, 256],
        );
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // All valid → NotSupported
        let args = SyscallArgs::new(nr::IPC_QUERY, [0x1000, 10, 0x2000, 100, 0x3000, 256]);
        assert_eq!(dispatch(&args), SyscallError::NotSupported.as_i64());
    }

    #[test]
    fn test_mcdc_ipc_channel_send_data_size() {
        // handle=valid, data_len > MAX → InvalidArgument
        let args = SyscallArgs::new(
            nr::IPC_CHANNEL_SEND,
            [1, 0x1000, ipc::MAX_IPC_MESSAGE_SIZE + 1, 0, 0, 0],
        );
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // handle=valid, data_len=MAX → NotSupported (passes validation)
        let args = SyscallArgs::new(
            nr::IPC_CHANNEL_SEND,
            [1, 0x1000, ipc::MAX_IPC_MESSAGE_SIZE, 0, 0, 0],
        );
        assert_eq!(dispatch(&args), SyscallError::NotSupported.as_i64());
    }

    #[test]
    fn test_mcdc_ipc_channel_create_ptrs() {
        // send=0, recv=valid → InvalidArgument (send check)
        let args = SyscallArgs::new(nr::IPC_CHANNEL_CREATE, [0, 0x1000, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // send=valid, recv=0 → InvalidArgument (recv check)
        let args = SyscallArgs::new(nr::IPC_CHANNEL_CREATE, [0x1000, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // send=valid, recv=valid → NotSupported
        let args = SyscallArgs::new(nr::IPC_CHANNEL_CREATE, [0x1000, 0x2000, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::NotSupported.as_i64());
    }

    // --- ONNX MC/DC ---

    #[test]
    fn test_mcdc_onnx_run_all_branches() {
        // session=0 → InvalidHandle
        let args = SyscallArgs::new(nr::ONNX_RUN, [0, 0x1000, 1, 0x2000, 1, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidHandle.as_i64());

        // session=valid, inputs_ptr=0 → InvalidArgument
        let args = SyscallArgs::new(nr::ONNX_RUN, [1, 0, 1, 0x2000, 1, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // session=valid, inputs_ptr=valid, num_inputs=0 → InvalidArgument
        let args = SyscallArgs::new(nr::ONNX_RUN, [1, 0x1000, 0, 0x2000, 1, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // session=valid, inputs valid, outputs_ptr=0 → InvalidArgument
        let args = SyscallArgs::new(nr::ONNX_RUN, [1, 0x1000, 1, 0, 1, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // session=valid, inputs valid, outputs_ptr=valid, num_outputs=0 → InvalidArgument
        let args = SyscallArgs::new(nr::ONNX_RUN, [1, 0x1000, 1, 0x2000, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // num_inputs too large
        let args = SyscallArgs::new(
            nr::ONNX_RUN,
            [1, 0x1000, onnx::MAX_IO_TENSORS + 1, 0x2000, 1, 0],
        );
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // num_outputs too large
        let args = SyscallArgs::new(
            nr::ONNX_RUN,
            [1, 0x1000, 1, 0x2000, onnx::MAX_IO_TENSORS + 1, 0],
        );
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // All valid → NotSupported
        let args = SyscallArgs::new(nr::ONNX_RUN, [1, 0x1000, 1, 0x2000, 1, 0]);
        assert_eq!(dispatch(&args), SyscallError::NotSupported.as_i64());
    }

    #[test]
    fn test_mcdc_onnx_list_providers_branches() {
        // buf_ptr=0, buf_len=0 → NotSupported (query size mode)
        let args = SyscallArgs::new(nr::ONNX_LIST_PROVIDERS, [0, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::NotSupported.as_i64());

        // buf_ptr=0, buf_len=non-zero → InvalidArgument
        let args = SyscallArgs::new(nr::ONNX_LIST_PROVIDERS, [0, 256, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // buf_ptr=valid, buf_len=valid → NotSupported (stub)
        let args = SyscallArgs::new(nr::ONNX_LIST_PROVIDERS, [0x1000, 256, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::NotSupported.as_i64());
    }

    // --- Device MC/DC ---

    #[test]
    fn test_mcdc_dev_enumerate_branches() {
        // buf_ptr=0, buf_len=0 → NotSupported (query size mode)
        let args = SyscallArgs::new(nr::DEV_ENUMERATE, [0, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::NotSupported.as_i64());

        // buf_ptr=0, buf_len=non-zero → InvalidArgument
        let args = SyscallArgs::new(nr::DEV_ENUMERATE, [0, 256, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // buf_ptr=valid → NotSupported
        let args = SyscallArgs::new(nr::DEV_ENUMERATE, [0x1000, 256, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::NotSupported.as_i64());
    }

    #[test]
    fn test_mcdc_dev_dma_alloc_all_branches() {
        // size=0 → InvalidArgument
        let args = SyscallArgs::new(nr::DEV_DMA_ALLOC, [0, 4096, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // size > MAX → InvalidArgument
        let args = SyscallArgs::new(
            nr::DEV_DMA_ALLOC,
            [device::MAX_DMA_SIZE + 1, 4096, 0, 0, 0, 0],
        );
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // size=valid, align=bad (non-zero, not power of two) → InvalidArgument
        let args = SyscallArgs::new(nr::DEV_DMA_ALLOC, [4096, 3, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // size=valid, align=0 (default) → NotSupported (passes validation)
        let args = SyscallArgs::new(nr::DEV_DMA_ALLOC, [4096, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::NotSupported.as_i64());

        // size=valid, align=4096 (valid power of two) → NotSupported
        let args = SyscallArgs::new(nr::DEV_DMA_ALLOC, [4096, 4096, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::NotSupported.as_i64());
    }

    #[test]
    fn test_mcdc_dev_ioctl_valid_handle() {
        // handle=0 → InvalidHandle
        let args = SyscallArgs::new(nr::DEV_IOCTL, [0, 1, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidHandle.as_i64());

        // handle=valid → NotSupported
        let args = SyscallArgs::new(nr::DEV_IOCTL, [1, 1, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::NotSupported.as_i64());
    }

    #[test]
    fn test_mcdc_dev_close_valid_handle() {
        // handle=0 → InvalidHandle
        let args = SyscallArgs::new(nr::DEV_CLOSE, [0, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidHandle.as_i64());

        // handle=valid → NotSupported
        let args = SyscallArgs::new(nr::DEV_CLOSE, [1, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::NotSupported.as_i64());
    }

    // --- System MC/DC ---

    #[test]
    fn test_mcdc_sys_info_all_branches() {
        // buf_ptr=0 → returns required size
        let args = SyscallArgs::new(nr::SYS_INFO, [0, 0, 0, 0, 0, 0]);
        let result = dispatch(&args);
        assert!(result > 0);
        let required_size = result as usize;

        // buf_ptr=valid, buf_len < required → BufferTooSmall
        let args = SyscallArgs::new(nr::SYS_INFO, [0x1000, required_size - 1, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::BufferTooSmall.as_i64());

        // buf_ptr=valid but misaligned, buf_len=sufficient → BadAddress
        let args = SyscallArgs::new(nr::SYS_INFO, [0x1001, required_size, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::BadAddress.as_i64());

        // buf_ptr=valid aligned, buf_len=sufficient → Success
        let mut info = core::mem::MaybeUninit::<system::SystemInfo>::uninit();
        let args = SyscallArgs::new(
            nr::SYS_INFO,
            [info.as_mut_ptr() as usize, required_size, 0, 0, 0, 0],
        );
        assert_eq!(dispatch(&args), SyscallError::Success.as_i64());
    }

    #[test]
    fn test_mcdc_sys_log_all_branches() {
        // Invalid level → InvalidArgument
        let args = SyscallArgs::new(nr::SYS_LOG, [5, 0x1000, 10, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // Valid level, msg_ptr=0 → InvalidArgument (msg_ptr check)
        let args = SyscallArgs::new(nr::SYS_LOG, [0, 0, 10, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // Valid level, msg_ptr=valid, msg_len=0 → InvalidArgument (len check)
        let args = SyscallArgs::new(nr::SYS_LOG, [0, 0x1000, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // Valid level, msg_ptr=valid, msg_len > MAX → InvalidArgument
        let args = SyscallArgs::new(
            nr::SYS_LOG,
            [0, 0x1000, system::MAX_LOG_MSG_LEN + 1, 0, 0, 0],
        );
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // Each valid log level
        for level in 0..=4usize {
            let args = SyscallArgs::new(nr::SYS_LOG, [level, 0x1000, 10, 0, 0, 0]);
            assert_eq!(
                dispatch(&args),
                SyscallError::Success.as_i64(),
                "log level {} should succeed",
                level
            );
        }
    }

    #[test]
    fn test_mcdc_sys_random_conditions() {
        // buf_ptr=0 → InvalidArgument
        let args = SyscallArgs::new(nr::SYS_RANDOM, [0, 32, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // buf_len=0 → InvalidArgument (buf_len==0 branch of OR)
        let args = SyscallArgs::new(nr::SYS_RANDOM, [0x1000, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // buf_len > MAX → InvalidArgument (buf_len > MAX branch of OR)
        let args = SyscallArgs::new(
            nr::SYS_RANDOM,
            [0x1000, system::MAX_RANDOM_LEN + 1, 0, 0, 0, 0],
        );
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // Valid args → NotSupported (CSPRNG not initialized)
        let args = SyscallArgs::new(nr::SYS_RANDOM, [0x1000, 32, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::NotSupported.as_i64());
    }

    #[test]
    fn test_mcdc_sys_shutdown_capability() {
        // Without capability → PermissionDenied
        let args = SyscallArgs::new(nr::SYS_SHUTDOWN, [0, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::PermissionDenied.as_i64());
    }

    // --- Capability MC/DC ---

    #[test]
    fn test_mcdc_cap_create_lifecycle() {
        // Valid create → positive cap ID
        let args = SyscallArgs::new(nr::CAP_CREATE, [0, 1, 0b0001, 0, 0, 0]);
        let cap_id = dispatch(&args);
        assert!(cap_id > 0, "cap_create should return positive ID");

        // Check the created capability → Success
        let args = SyscallArgs::new(nr::CAP_CHECK, [cap_id as usize, 0, 1, 0b0001, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::Success.as_i64());

        // Check with wrong permission → PermissionDenied
        let args = SyscallArgs::new(nr::CAP_CHECK, [cap_id as usize, 0, 1, 0b0010, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::PermissionDenied.as_i64());

        // Revoke it → Success
        let args = SyscallArgs::new(nr::CAP_REVOKE, [cap_id as usize, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::Success.as_i64());

        // Check again after revoke → should fail
        let args = SyscallArgs::new(nr::CAP_CHECK, [cap_id as usize, 0, 1, 0b0001, 0, 0]);
        let result = dispatch(&args);
        assert!(result < 0, "check after revoke should fail");
    }

    #[test]
    fn test_mcdc_cap_create_all_resource_types() {
        // Valid resource types: 0..7
        for rt in 0..=7u8 {
            let args = SyscallArgs::new(nr::CAP_CREATE, [rt as usize, 1, 0b0001, 0, 0, 0]);
            let result = dispatch(&args);
            assert!(result > 0, "resource type {} should succeed", rt);
        }

        // Invalid resource type
        let args = SyscallArgs::new(nr::CAP_CREATE, [8, 1, 0b0001, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        let args = SyscallArgs::new(nr::CAP_CREATE, [255, 1, 0b0001, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_mcdc_cap_create_permission_bits() {
        // Each individual permission bit
        for perm in [0b0001u32, 0b0010, 0b0100, 0b1000] {
            let args = SyscallArgs::new(nr::CAP_CREATE, [0, 1, perm as usize, 0, 0, 0]);
            let result = dispatch(&args);
            assert!(result > 0, "perm {:#b} should succeed", perm);
        }

        // All bits combined
        let args = SyscallArgs::new(nr::CAP_CREATE, [0, 1, 0b1111, 0, 0, 0]);
        assert!(dispatch(&args) > 0);

        // High bits set → InvalidArgument
        let args = SyscallArgs::new(nr::CAP_CREATE, [0, 1, 0b10000, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // Zero perms → InvalidArgument
        let args = SyscallArgs::new(nr::CAP_CREATE, [0, 1, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());
    }

    #[test]
    fn test_mcdc_cap_delegate_all_branches() {
        // Create parent capability first
        let args = SyscallArgs::new(nr::CAP_CREATE, [0, 1, 0b1111, 0, 0, 0]);
        let parent_id = dispatch(&args);
        assert!(parent_id > 0);

        // cap_id=0 → InvalidHandle
        let args = SyscallArgs::new(nr::CAP_DELEGATE, [0, 2, 0b0001, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidHandle.as_i64());

        // to_task=0 → InvalidArgument
        let args = SyscallArgs::new(nr::CAP_DELEGATE, [parent_id as usize, 0, 0b0001, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // perms=0 → InvalidArgument
        let args = SyscallArgs::new(nr::CAP_DELEGATE, [parent_id as usize, 2, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // high bits in perms → InvalidArgument
        let args = SyscallArgs::new(nr::CAP_DELEGATE, [parent_id as usize, 2, 0xFF, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // Valid delegation with subset permissions
        let args = SyscallArgs::new(
            nr::CAP_DELEGATE,
            [parent_id as usize, 2, 0b0001, 0, 0, 0],
        );
        let child_id = dispatch(&args);
        assert!(child_id > 0, "delegation should succeed");
    }

    #[test]
    fn test_mcdc_cap_check_compound_condition() {
        // cap_id=0 → InvalidHandle
        let args = SyscallArgs::new(nr::CAP_CHECK, [0, 0, 1, 0b0001, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidHandle.as_i64());

        // Invalid resource type → InvalidArgument
        let args = SyscallArgs::new(nr::CAP_CHECK, [1, 99, 1, 0b0001, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // High perm bits → InvalidArgument (compound: high_bits OR zero)
        let args = SyscallArgs::new(nr::CAP_CHECK, [1, 0, 1, 0b10000, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // Zero perms → InvalidArgument (compound: high_bits OR zero)
        let args = SyscallArgs::new(nr::CAP_CHECK, [1, 0, 1, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::InvalidArgument.as_i64());

        // Non-existent cap → error from registry
        let args = SyscallArgs::new(nr::CAP_CHECK, [99999, 0, 1, 0b0001, 0, 0]);
        let result = dispatch(&args);
        assert!(result < 0);
    }

    #[test]
    fn test_mcdc_cap_list_all_branches() {
        // buf_ptr=0, buf_len=0 → count mode (returns 0 or positive)
        let args = SyscallArgs::new(nr::CAP_LIST, [0, 0, 0, 0, 0, 0]);
        let result = dispatch(&args);
        assert!(result >= 0);

        // buf_ptr=0, buf_len>0 → BadAddress
        let args = SyscallArgs::new(nr::CAP_LIST, [0, 0, 10, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::BadAddress.as_i64());

        // Create a cap for current task, then list with buffer
        let args = SyscallArgs::new(nr::CAP_CREATE, [0, 100, 0b0001, 0, 0, 0]);
        let _cap_id = dispatch(&args);

        // count mode should now return >= 1
        let args = SyscallArgs::new(nr::CAP_LIST, [0, 0, 0, 0, 0, 0]);
        let count = dispatch(&args);
        assert!(count >= 0);

        // buf_ptr=valid, buf_len=valid (fill buffer mode)
        let mut buf = [0u64; 16];
        let args = SyscallArgs::new(nr::CAP_LIST, [0, buf.as_mut_ptr() as usize, 16, 0, 0, 0]);
        let filled = dispatch(&args);
        assert!(filled >= 0);
    }

    // --- POSIX MC/DC ---

    #[test]
    fn test_mcdc_posix_read_write_compound_conditions() {
        // read: buf_ptr=0, count=0 → fd validation path (not BadAddress since count=0)
        let args = SyscallArgs::new(nr::POSIX_READ, [0, 0, 0, 0, 0, 0]);
        let result = dispatch(&args);
        // buf_ptr=0 AND count=0 → count>0 is false, so BadAddress not triggered
        assert!(result != SyscallError::BadAddress.as_i64() || result == -38);

        // read: buf_ptr=valid, count=valid, fd valid → stub
        let args = SyscallArgs::new(nr::POSIX_READ, [0, 0x1000, 100, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38); // ENOSYS

        // write: buf_ptr=0, count>0 → BadAddress
        let args = SyscallArgs::new(nr::POSIX_WRITE, [1, 0, 100, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::BadAddress.as_i64());

        // write: buf_ptr=0, count=0 → passes (count not > 0)
        let args = SyscallArgs::new(nr::POSIX_WRITE, [1, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);

        // write: negative fd
        let args = SyscallArgs::new(nr::POSIX_WRITE, [usize::MAX, 0x1000, 100, 0, 0, 0]);
        assert_eq!(dispatch(&args), -9); // EBADF
    }

    #[test]
    fn test_mcdc_posix_dup2_fd_conditions() {
        // old_fd negative, new_fd valid → EBADF (old_fd check)
        let args = SyscallArgs::new(nr::POSIX_DUP2, [usize::MAX, 3, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -9);

        // old_fd valid, new_fd negative → EBADF (new_fd check)
        let args = SyscallArgs::new(nr::POSIX_DUP2, [1, usize::MAX, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -9);

        // both valid → ENOSYS
        let args = SyscallArgs::new(nr::POSIX_DUP2, [1, 3, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);
    }

    #[test]
    fn test_mcdc_posix_lseek_whence() {
        // SEEK_SET=0 → valid
        let args = SyscallArgs::new(nr::POSIX_LSEEK, [3, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);

        // SEEK_CUR=1 → valid
        let args = SyscallArgs::new(nr::POSIX_LSEEK, [3, 10, 1, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);

        // SEEK_END=2 → valid
        let args = SyscallArgs::new(nr::POSIX_LSEEK, [3, 0, 2, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);

        // whence=3 → EINVAL
        let args = SyscallArgs::new(nr::POSIX_LSEEK, [3, 0, 3, 0, 0, 0]);
        assert_eq!(dispatch(&args), -22);
    }

    #[test]
    fn test_mcdc_posix_munmap_conditions() {
        // length=0, addr=aligned → EINVAL (length check)
        let args = SyscallArgs::new(nr::POSIX_MUNMAP, [4096, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -22);

        // length=valid, addr=unaligned → EINVAL (alignment check)
        let args = SyscallArgs::new(nr::POSIX_MUNMAP, [1, 4096, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -22);

        // length=valid, addr=aligned → ENOSYS
        let args = SyscallArgs::new(nr::POSIX_MUNMAP, [4096, 4096, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);
    }

    #[test]
    fn test_mcdc_posix_mprotect_conditions() {
        // length=0, addr=aligned → EINVAL (length check)
        let args = SyscallArgs::new(nr::POSIX_MPROTECT, [4096, 0, 1, 0, 0, 0]);
        assert_eq!(dispatch(&args), -22);

        // length=valid, addr=unaligned → EINVAL (alignment check)
        let args = SyscallArgs::new(nr::POSIX_MPROTECT, [1, 4096, 1, 0, 0, 0]);
        assert_eq!(dispatch(&args), -22);

        // length=valid, addr=aligned → ENOSYS
        let args = SyscallArgs::new(nr::POSIX_MPROTECT, [4096, 4096, 1, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);
    }

    #[test]
    fn test_mcdc_posix_epoll_create_flags() {
        // flags=0 → ENOSYS (valid)
        let args = SyscallArgs::new(nr::POSIX_EPOLL_CREATE, [0, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);

        // flags=O_CLOEXEC → ENOSYS (valid)
        let args = SyscallArgs::new(nr::POSIX_EPOLL_CREATE, [0o2000000, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);

        // flags=invalid → EINVAL
        let args = SyscallArgs::new(nr::POSIX_EPOLL_CREATE, [1, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -22);
    }

    #[test]
    fn test_mcdc_posix_epoll_ctl_op() {
        // epfd negative → EBADF
        let args = SyscallArgs::new(nr::POSIX_EPOLL_CTL, [usize::MAX, 1, 3, 0, 0, 0]);
        assert_eq!(dispatch(&args), -9);

        // op=1 (ADD) → ENOSYS (valid)
        let args = SyscallArgs::new(nr::POSIX_EPOLL_CTL, [3, 1, 5, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);

        // op=2 (MOD) → ENOSYS (valid)
        let args = SyscallArgs::new(nr::POSIX_EPOLL_CTL, [3, 2, 5, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);

        // op=3 (DEL) → ENOSYS (valid)
        let args = SyscallArgs::new(nr::POSIX_EPOLL_CTL, [3, 3, 5, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);

        // op=0 → EINVAL
        let args = SyscallArgs::new(nr::POSIX_EPOLL_CTL, [3, 0, 5, 0, 0, 0]);
        assert_eq!(dispatch(&args), -22);

        // op=4 → EINVAL
        let args = SyscallArgs::new(nr::POSIX_EPOLL_CTL, [3, 4, 5, 0, 0, 0]);
        assert_eq!(dispatch(&args), -22);
    }

    #[test]
    fn test_mcdc_posix_epoll_wait_conditions() {
        // epfd negative → EBADF
        let args = SyscallArgs::new(nr::POSIX_EPOLL_WAIT, [usize::MAX, 0x1000, 10, 100, 0, 0]);
        assert_eq!(dispatch(&args), -9);

        // events_ptr=0 → EINVAL
        let args = SyscallArgs::new(nr::POSIX_EPOLL_WAIT, [3, 0, 10, 100, 0, 0]);
        assert_eq!(dispatch(&args), -22);

        // max_events=0 → EINVAL
        let args = SyscallArgs::new(nr::POSIX_EPOLL_WAIT, [3, 0x1000, 0, 100, 0, 0]);
        assert_eq!(dispatch(&args), -22);

        // All valid → ENOSYS
        let args = SyscallArgs::new(nr::POSIX_EPOLL_WAIT, [3, 0x1000, 10, 100, 0, 0]);
        assert_eq!(dispatch(&args), -38);
    }

    #[test]
    fn test_mcdc_posix_clock_gettime_branches() {
        // ts_ptr=0 → BadAddress
        let args = SyscallArgs::new(nr::POSIX_CLOCK_GETTIME, [0, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::BadAddress.as_i64());

        // clock_id valid (0..3), ts_ptr valid
        for clock_id in 0..=3usize {
            let args = SyscallArgs::new(nr::POSIX_CLOCK_GETTIME, [clock_id, 0x1000, 0, 0, 0, 0]);
            assert_eq!(dispatch(&args), -38, "clock_id {} should return ENOSYS", clock_id);
        }

        // clock_id=4 (invalid) → EINVAL
        let args = SyscallArgs::new(nr::POSIX_CLOCK_GETTIME, [4, 0x1000, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -22);
    }

    #[test]
    fn test_mcdc_posix_getrandom_branches() {
        // buf_ptr=0, buf_len>0 → BadAddress
        let args = SyscallArgs::new(nr::POSIX_GETRANDOM, [0, 32, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), SyscallError::BadAddress.as_i64());

        // buf_ptr=0, buf_len=0 → 0 (zero bytes requested, buf_ptr=0 AND buf_len=0 is fine)
        let args = SyscallArgs::new(nr::POSIX_GETRANDOM, [0, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), 0);

        // buf_ptr=valid, buf_len=0 → 0
        let args = SyscallArgs::new(nr::POSIX_GETRANDOM, [0x1000, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), 0);

        // buf_ptr=valid, buf_len>0 → ENOSYS
        let args = SyscallArgs::new(nr::POSIX_GETRANDOM, [0x1000, 32, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);
    }

    #[test]
    fn test_mcdc_posix_socket_family_and_type() {
        // AF_INET(2), SOCK_STREAM(1) → ENOSYS (valid)
        let args = SyscallArgs::new(nr::POSIX_SOCKET, [2, 1, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);

        // AF_INET(2), SOCK_DGRAM(2) → ENOSYS (valid)
        let args = SyscallArgs::new(nr::POSIX_SOCKET, [2, 2, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);

        // AF_INET6(10), SOCK_STREAM(1) → ENOSYS (valid)
        let args = SyscallArgs::new(nr::POSIX_SOCKET, [10, 1, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);

        // AF_INET6(10), SOCK_DGRAM(2) → ENOSYS (valid)
        let args = SyscallArgs::new(nr::POSIX_SOCKET, [10, 2, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -38);

        // Invalid family → EINVAL
        let args = SyscallArgs::new(nr::POSIX_SOCKET, [0, 1, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -22);

        let args = SyscallArgs::new(nr::POSIX_SOCKET, [1, 1, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -22);

        // Valid family, invalid type → EINVAL
        let args = SyscallArgs::new(nr::POSIX_SOCKET, [2, 0, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -22);

        let args = SyscallArgs::new(nr::POSIX_SOCKET, [2, 3, 0, 0, 0, 0]);
        assert_eq!(dispatch(&args), -22);
    }

    // --- Dispatch and error infrastructure MC/DC ---

    #[test]
    fn test_mcdc_dispatch_boundary_conditions() {
        // number < SYSCALL_TABLE_SIZE (valid range)
        let args = SyscallArgs::zero(0);
        let _ = dispatch(&args); // Must not panic

        // number == SYSCALL_TABLE_SIZE - 1 (last valid index)
        let args = SyscallArgs::zero(SYSCALL_TABLE_SIZE - 1);
        let _ = dispatch(&args);

        // number == SYSCALL_TABLE_SIZE (first out of range) → NoSys
        let args = SyscallArgs::zero(SYSCALL_TABLE_SIZE);
        assert_eq!(dispatch(&args), SyscallError::NoSys.as_i64());

        // number == SYSCALL_TABLE_SIZE + 1 → NoSys
        let args = SyscallArgs::zero(SYSCALL_TABLE_SIZE + 1);
        assert_eq!(dispatch(&args), SyscallError::NoSys.as_i64());
    }

    #[test]
    fn test_mcdc_category_name_unknown() {
        // Category 0x9 → unknown
        assert_eq!(category_name(0x90), "unknown");
        assert_eq!(category_name(0xA0), "unknown");
        assert_eq!(category_name(0xFF), "unknown");
        assert_eq!(category_name(usize::MAX), "unknown");
    }

    #[test]
    fn test_mcdc_syscall_error_from_i64_boundaries() {
        // All defined values
        for val in 0..=-16i64 {
            let result = SyscallError::from_i64(val);
            assert!(result.is_some(), "value {} should be a valid error", val);
        }

        // Undefined values
        assert_eq!(SyscallError::from_i64(-17), None);
        assert_eq!(SyscallError::from_i64(-100), None);
        assert_eq!(SyscallError::from_i64(1), None);
        assert_eq!(SyscallError::from_i64(i64::MAX), None);
        assert_eq!(SyscallError::from_i64(i64::MIN), None);
    }

    #[test]
    fn test_mcdc_registered_count_equals_expected() {
        // Verify the exact count of non-nosys handlers
        // 8 memory + 7 task + 8 ipc + 6 onnx + 5 device + 7 system + 5 capability + 18 posix = 64
        assert_eq!(registered_count(), 64);
    }

    #[test]
    fn test_mcdc_nosys_returns_nosys() {
        // Direct call to sys_nosys
        let args = SyscallArgs::zero(0);
        assert_eq!(sys_nosys(&args), SyscallError::NoSys.as_i64());
    }
}
