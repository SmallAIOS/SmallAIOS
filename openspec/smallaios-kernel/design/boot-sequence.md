# Boot Sequence Design

## Overview

SmallAIOS supports three boot paths depending on deployment mode. All paths
converge at the same kernel initialization entry point.

## Boot Path 1: Container Mode (Library OS)

This is the primary deployment mode. SmallAIOS runs as a statically-linked
Linux binary inside a container.

```
Container runtime (runc/crun)
    │
    ▼
ELF _start (arch/x86_64/entry.S or arch/aarch64/entry.S)
    │
    ▼
Rust main() — SmallAIOS entry point
    │
    ├── Parse auxiliary vector (AT_PAGESZ, AT_RANDOM, etc.)
    ├── Initialize allocators (mmap-based, from host kernel)
    ├── Read /config/smallaios.toml
    ├── Initialize POSIX compatibility layer
    │
    ▼
Common initialization path (see below)
```

In container mode, SmallAIOS uses the host kernel for:
- Memory allocation (via mmap syscalls)
- Thread creation (via clone/futex)
- Network I/O (via socket syscalls)
- GPU access (via ioctl to /dev/nvidia*)

The "kernel core" in this mode is a thin wrapper that delegates to host syscalls.

## Boot Path 2: MicroVM Mode (Real Kernel)

SmallAIOS boots as a guest kernel in a VM.

### x86-64 UEFI Boot

```
UEFI firmware
    │
    ▼
SmallAIOS UEFI application (PE32+ binary)
    │
    ├── Get memory map from UEFI (EFI_BOOT_SERVICES.GetMemoryMap)
    ├── Get framebuffer info (optional, for debug console)
    ├── Get ACPI RSDP pointer
    ├── ExitBootServices() — UEFI hands over control
    │
    ▼
arch/x86_64 early init (assembly)
    │
    ├── Set up initial page tables (identity map + kernel high map)
    ├── Set up GDT with kernel segments
    ├── Set up IDT with early exception handlers
    ├── Enable long mode features (NX, PCID, etc.)
    ├── Set up initial stack
    │
    ▼
arch/x86_64 Rust init
    │
    ├── Parse UEFI memory map → build physical frame allocator
    ├── Set up final page tables with proper memory map
    ├── Initialize Local APIC + I/O APIC
    ├── Calibrate TSC / APIC timer
    ├── Detect CPU features (CPUID)
    ├── Initialize per-CPU structures
    ├── Boot secondary CPUs (SIPI protocol):
    │   ├── Send INIT IPI
    │   ├── Wait 10ms
    │   ├── Send STARTUP IPI with trampoline address
    │   └── Each secondary CPU runs its own init sequence
    │
    ▼
Common initialization path (see below)
```

### ARM64 Boot

```
UEFI firmware (or bootloader with DTB)
    │
    ▼
SmallAIOS entry at EL1
    │
    ├── Parse DTB or UEFI memory map
    ├── Set up exception vector table (VBAR_EL1)
    ├── Configure MMU (TCR_EL1, MAIR_EL1, TTBR0/TTBR1)
    ├── Enable MMU
    ├── Initialize GICv3 (Distributor + Redistributor)
    ├── Set up ARM Generic Timer (CNTP_CTL_EL0)
    ├── Detect CPU features (ID_AA64* registers)
    ├── Boot secondary CPUs via PSCI:
    │   ├── PSCI CPU_ON call per secondary
    │   └── Each secondary runs init sequence
    │
    ▼
Common initialization path (see below)
```

## Common Initialization Path

After architecture-specific setup, all boot paths converge:

```
Step 1: Memory subsystem
    ├── Initialize buddy allocator (physical frames)
    ├── Initialize slab allocator (kernel objects)
    ├── Initialize kernel heap (#[global_allocator])
    ├── Initialize tensor buffer pool
    └── Log: "Memory: {total} MB, tensor pool: {pool} MB"

Step 2: Scheduler
    ├── Initialize per-CPU run queues
    ├── Create idle task per CPU
    ├── Start async executor
    └── Log: "Scheduler: {n} CPUs online"

Step 3: Device enumeration (VM/bare metal only)
    ├── PCIe bus scan (if applicable)
    ├── Virtio device detection (if applicable)
    ├── NVIDIA GPU initialization (if present)
    └── Log: "Devices: {list}"

Step 4: Security
    ├── Initialize capability registry
    ├── Create root capability set
    ├── Assign capabilities to subsystems
    └── Drop root capabilities

Step 5: ONNX runtime
    ├── Initialize operator registry
    ├── Register CPU execution provider
    ├── Register CUDA execution provider (if GPU present)
    ├── Load ONNX models from /models/ (or virtio-blk)
    ├── Create inference sessions (runs graph optimization)
    └── Log: "ONNX: loaded {n} models, providers: {list}"

Step 6: IPC
    ├── Initialize message router
    ├── Register built-in endpoints (health, metrics)
    ├── Register inference endpoints (one per model)
    ├── Start TCP listener on configured port
    └── Log: "IPC: listening on {addr}"

Step 7: Ready
    └── Log: "SmallAIOS ready in {elapsed} ms"
```

## Boot Time Budget

Target: **< 50ms** from entry to accepting inference requests (container mode).

| Phase | Budget | Notes |
|---|---|---|
| Container entry → Rust main | 1 ms | Static binary, no dynamic linking |
| Allocator init | 2 ms | mmap-based in container mode |
| Config parsing | 1 ms | Small TOML file |
| ONNX runtime init | 5 ms | Operator registration |
| Model loading | 10-30 ms | Depends on model size (mmap, no copy) |
| Graph optimization | 5-20 ms | Cached after first run |
| IPC init | 2 ms | Socket bind + listen |
| **Total** | **26-61 ms** | |

For MicroVM mode, add ~10ms for hardware init (page tables, APIC, etc.).

## Shutdown Sequence

```
1. Stop accepting new inference requests
2. Drain in-flight requests (with timeout)
3. Close IPC listeners
4. Unload ONNX models (free GPU memory)
5. Flush log buffer
6. Container mode: exit(0)
   VM mode: ACPI shutdown or HLT loop
```

Triggered by:
- SIGTERM (container mode — Kubernetes sends this)
- IPC command on `smallaios/v1/control/shutdown`
- sys_shutdown() syscall

## Early Debug Console

Before the IPC system is ready, SmallAIOS outputs boot logs to:
- **Container mode**: stdout (fd 1)
- **x86-64 VM**: COM1 serial port (0x3F8)
- **ARM64 VM**: PL011 UART (address from DTB)
- **UEFI**: EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL (before ExitBootServices)

Format: `[{timestamp_us}] {level} {module}: {message}`

Example:
```
[0000.000] INFO  boot: SmallAIOS v0.1.0 starting
[0000.001] INFO  mem: Physical memory: 2048 MB
[0000.002] INFO  mem: Tensor pool: 1024 MB (huge pages)
[0000.003] INFO  sched: 4 CPUs online, 4 worker threads
[0000.005] INFO  onnx: Loaded model "resnet50" (98 MB, 152 operators)
[0000.025] INFO  onnx: Session created: CPU provider, optimization level Full
[0000.027] INFO  ipc: Listening on tcp://0.0.0.0:7447
[0000.027] INFO  boot: SmallAIOS ready in 27 ms
```
