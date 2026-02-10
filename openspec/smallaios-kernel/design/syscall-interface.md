# Syscall Interface Design

## Overview

SmallAIOS exposes a minimal, purpose-built syscall interface. In unikernel/container
mode, these are direct Rust function calls. In VM mode, they use the hardware
syscall mechanism (`syscall`/`sysret` on x86-64, `svc`/`eret` on ARM64).

## Syscall Calling Convention

### x86-64 (VM mode)

Follows the Linux syscall convention for toolchain compatibility:
- Syscall number: `rax`
- Arguments: `rdi`, `rsi`, `rdx`, `r10`, `r8`, `r9`
- Return value: `rax` (error: negative errno)
- Clobbered: `rcx`, `r11` (by `syscall` instruction)

### ARM64 (VM mode)

- Syscall number: `x8`
- Arguments: `x0` - `x5`
- Return value: `x0` (error: negative errno)
- Instruction: `svc #0`

### Container/Unikernel Mode

Direct function calls — no register convention, normal Rust ABI.

## Complete Syscall Table

### Memory Management (0x00 - 0x0F)

| Nr | Name | Args | Return | Description |
|---|---|---|---|---|
| 0x00 | `mem_alloc` | size: usize, align: usize, flags: u32 | *mut u8 | Allocate memory |
| 0x01 | `mem_free` | ptr: *mut u8, size: usize | () | Free memory |
| 0x02 | `mem_map` | phys: u64, virt: u64, size: usize, flags: u32 | result | Map physical to virtual |
| 0x03 | `mem_unmap` | virt: u64, size: usize | result | Unmap virtual range |
| 0x04 | `mem_protect` | ptr: *mut u8, size: usize, flags: u32 | result | Change memory protection |
| 0x05 | `tensor_alloc` | ndims: u32, shape: *const i64, dtype: u32 | TensorHandle | Allocate tensor buffer |
| 0x06 | `tensor_free` | handle: TensorHandle | () | Free tensor buffer |
| 0x07 | `tensor_data` | handle: TensorHandle | *mut u8 | Get raw data pointer |
| 0x08 | `tensor_map_gpu` | handle: TensorHandle, device: u32 | GpuPtr | Map tensor to GPU |
| 0x09 | `tensor_unmap_gpu` | handle: TensorHandle, device: u32 | () | Unmap tensor from GPU |

Flags for `mem_alloc`:
```rust
pub const MEM_HUGE_PAGE: u32  = 0x01;  // Use 2MB huge pages
pub const MEM_GIANT_PAGE: u32 = 0x02;  // Use 1GB giant pages
pub const MEM_DMA: u32        = 0x04;  // DMA-capable (pinned, for GPU)
pub const MEM_ZERO: u32       = 0x08;  // Zero-initialized
pub const MEM_NUMA_LOCAL: u32 = 0x10;  // Prefer local NUMA node
```

### Task Management (0x10 - 0x1F)

| Nr | Name | Args | Return | Description |
|---|---|---|---|---|
| 0x10 | `task_spawn` | entry: fn, arg: *const u8 | TaskId | Create async task |
| 0x11 | `task_yield` | — | () | Yield to scheduler |
| 0x12 | `task_exit` | code: i32 | ! | Exit current task |
| 0x13 | `task_join` | id: TaskId | i32 | Wait for task completion |
| 0x14 | `task_current` | — | TaskId | Get current task ID |
| 0x15 | `task_set_affinity` | id: TaskId, cpu_mask: u64 | result | Set CPU affinity |
| 0x16 | `task_set_priority` | id: TaskId, priority: u32 | result | Set task priority |
| 0x17 | `task_set_class` | id: TaskId, class: u32 | result | Set scheduling class (SYSTEM/IPC/INFERENCE) |

### IPC (0x20 - 0x2F)

| Nr | Name | Args | Return | Description |
|---|---|---|---|---|
| 0x20 | `ipc_publish` | key: *const u8, key_len: u32, data: *const u8, data_len: u32 | result | Publish message |
| 0x21 | `ipc_subscribe` | key_expr: *const u8, key_len: u32 | SubHandle | Subscribe to key expression |
| 0x22 | `ipc_unsubscribe` | handle: SubHandle | () | Remove subscription |
| 0x23 | `ipc_recv` | handle: SubHandle, buf: *mut u8, buf_len: u32 | usize | Receive next message |
| 0x24 | `ipc_query` | key: *const u8, key_len: u32, data: *const u8, data_len: u32, reply: *mut u8, reply_len: u32 | usize | Request/reply query |
| 0x25 | `ipc_declare_queryable` | key: *const u8, key_len: u32, handler: fn | QHandle | Register queryable |
| 0x26 | `ipc_undeclare_queryable` | handle: QHandle | () | Remove queryable |

### ONNX Runtime (0x30 - 0x3F)

| Nr | Name | Args | Return | Description |
|---|---|---|---|---|
| 0x30 | `onnx_load` | data: *const u8, len: usize | ModelHandle | Load ONNX model from bytes |
| 0x31 | `onnx_load_path` | path: *const u8, path_len: u32 | ModelHandle | Load ONNX model from path |
| 0x32 | `onnx_unload` | handle: ModelHandle | () | Unload model |
| 0x33 | `onnx_create_session` | model: ModelHandle, opts: *const SessionOpts | SessionHandle | Create session |
| 0x34 | `onnx_destroy_session` | handle: SessionHandle | () | Destroy session |
| 0x35 | `onnx_run` | session: SessionHandle, inputs: *const TensorHandle, n_in: u32, outputs: *mut TensorHandle, n_out: u32 | result | Run inference |
| 0x36 | `onnx_metadata` | model: ModelHandle, buf: *mut u8, buf_len: u32 | usize | Get model metadata |
| 0x37 | `onnx_input_info` | model: ModelHandle, idx: u32, info: *mut TensorInfo | result | Get input tensor info |
| 0x38 | `onnx_output_info` | model: ModelHandle, idx: u32, info: *mut TensorInfo | result | Get output tensor info |

### Device (0x40 - 0x4F)

| Nr | Name | Args | Return | Description |
|---|---|---|---|---|
| 0x40 | `dev_enumerate` | buf: *mut DeviceInfo, max: u32 | u32 | List devices |
| 0x41 | `dev_open` | id: DeviceId | DevHandle | Open device |
| 0x42 | `dev_close` | handle: DevHandle | () | Close device |
| 0x43 | `dev_ioctl` | handle: DevHandle, cmd: u32, arg: u64 | isize | Device control |
| 0x44 | `dev_dma_alloc` | size: usize, align: usize | DmaBuffer | Allocate DMA buffer |
| 0x45 | `dev_dma_free` | buf: DmaBuffer | () | Free DMA buffer |

### System (0x50 - 0x5F)

| Nr | Name | Args | Return | Description |
|---|---|---|---|---|
| 0x50 | `sys_info` | buf: *mut SystemInfo | result | Get system information |
| 0x51 | `sys_time` | — | u64 | Monotonic nanoseconds since boot |
| 0x52 | `sys_realtime` | — | u64 | Wall clock nanoseconds (UTC) |
| 0x53 | `sys_shutdown` | code: i32 | ! | Shutdown system |
| 0x54 | `sys_log` | level: u32, msg: *const u8, len: u32 | () | Write log message |
| 0x55 | `sys_random` | buf: *mut u8, len: u32 | result | Fill buffer with CSPRNG bytes |
| 0x56 | `sys_config` | key: *const u8, key_len: u32, val: *mut u8, val_len: u32 | usize | Read config value |
| 0x57 | `sys_metrics` | buf: *mut u8, buf_len: u32 | usize | Get system metrics |
| 0x58 | `sys_watchdog_pet` | — | result | Service (pet) hardware watchdog |
| 0x59 | `sys_watchdog_remaining` | — | u32 | Get remaining watchdog time (seconds) |

### Capability (0x60 - 0x6F)

| Nr | Name | Args | Return | Description |
|---|---|---|---|---|
| 0x60 | `cap_create` | resource: ResourceRef, perms: u32 | CapId | Create capability |
| 0x61 | `cap_revoke` | id: CapId | () | Revoke capability |
| 0x62 | `cap_delegate` | id: CapId, target: TaskId, perms: u32 | CapId | Delegate (subset) |
| 0x63 | `cap_check` | id: CapId, perm: u32 | bool | Check permission |
| 0x64 | `cap_list` | buf: *mut CapInfo, max: u32 | u32 | List held capabilities |

**Total: 49 syscalls**

## Error Codes

```rust
pub enum SyscallError {
    Success = 0,
    InvalidArgument = -1,      // EINVAL
    OutOfMemory = -2,          // ENOMEM
    PermissionDenied = -3,     // EACCES / EPERM
    NotFound = -4,             // ENOENT
    AlreadyExists = -5,        // EEXIST
    InvalidHandle = -6,        // EBADF
    WouldBlock = -7,           // EAGAIN
    TimedOut = -8,             // ETIMEDOUT
    NotSupported = -9,         // ENOSYS
    DeviceError = -10,         // EIO
    BufferTooSmall = -11,      // ERANGE
    Interrupted = -12,         // EINTR
    ResourceExhausted = -13,   // ENOSPC
    ConnectionReset = -14,     // ECONNRESET
    InvalidModel = -100,       // ONNX-specific: invalid model format
    UnsupportedOp = -101,      // ONNX-specific: unsupported operator
    ShapeMismatch = -102,      // ONNX-specific: tensor shape mismatch
    GpuError = -200,           // GPU-specific: device error
    GpuOutOfMemory = -201,     // GPU-specific: VRAM exhausted
}
```
