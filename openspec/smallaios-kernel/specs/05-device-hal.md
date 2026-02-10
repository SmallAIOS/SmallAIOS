# Spec 05: Device and Hardware Abstraction Layer

## Overview

The SmallAIOS HAL provides a uniform interface to hardware across three target
platforms: x86-64, ARM64, and NVIDIA GPU. The HAL is the only code that differs
per architecture — all higher layers (kernel, ONNX runtime, IPC) use the HAL
traits and are architecture-independent.

## HAL Trait Interface

```rust
/// Architecture-specific HAL implementation
pub trait Hal {
    /// Initialize the hardware platform
    fn init(boot_info: &BootInfo) -> Self;

    /// Memory management
    fn page_size() -> usize;
    fn map_page(virt: VirtAddr, phys: PhysAddr, flags: PageFlags);
    fn unmap_page(virt: VirtAddr);
    fn flush_tlb(virt: VirtAddr);
    fn flush_tlb_all();

    /// Interrupt control
    fn enable_interrupts();
    fn disable_interrupts();
    fn register_interrupt(vector: u32, handler: InterruptHandler);
    fn acknowledge_interrupt(vector: u32);
    fn send_ipi(cpu: CpuId, vector: u32);

    /// Timer
    fn timer_frequency() -> u64;  // Hz
    fn timer_current() -> u64;    // Ticks
    fn timer_set_deadline(ticks: u64);

    /// CPU features
    fn cpu_count() -> usize;
    fn cpu_id() -> CpuId;
    fn cpu_features() -> CpuFeatures;
    fn halt();  // Low-power wait

    /// Console (early boot, debugging)
    fn console_write(bytes: &[u8]);
}

/// GPU-specific HAL extension
pub trait GpuHal {
    fn gpu_enumerate() -> Vec<GpuDevice>;
    fn gpu_init(device: &GpuDevice) -> GpuContext;
    fn gpu_alloc(ctx: &GpuContext, size: usize) -> GpuPtr;
    fn gpu_free(ctx: &GpuContext, ptr: GpuPtr);
    fn gpu_copy_h2d(ctx: &GpuContext, host: *const u8, device: GpuPtr, size: usize);
    fn gpu_copy_d2h(ctx: &GpuContext, device: GpuPtr, host: *mut u8, size: usize);
    fn gpu_launch_kernel(ctx: &GpuContext, kernel: &GpuKernel, args: &GpuLaunchArgs);
    fn gpu_synchronize(ctx: &GpuContext);
}
```

## x86-64 Platform

### Boot

- **UEFI boot** (bare metal / VM): UEFI application entry → set up identity mapping →
  jump to kernel entry
- **Container entry**: Direct `_start` entry, host kernel provides memory map via
  auxiliary vector or custom protocol
- **Multiboot2** (QEMU/GRUB): Alternative boot protocol for testing

### CPU Initialization

```
1. Set up GDT (Global Descriptor Table) with kernel code/data segments
2. Set up IDT (Interrupt Descriptor Table) with 256 entries
3. Initialize APIC (Advanced Programmable Interrupt Controller)
   - Local APIC for timer and IPI
   - I/O APIC for external interrupts (PCIe MSI-X)
4. Enable SSE, AVX, AVX-512 if present (set CR4 bits, XCR0)
5. Set up syscall/sysret MSRs (for VM mode)
6. Initialize per-CPU data structures
7. Boot secondary CPUs via SIPI (Startup IPI)
```

### Memory Management

- **4-level paging** (PML4 → PDPT → PD → PT) for 48-bit virtual addresses
- **5-level paging** (PML5, for >48-bit VA) detected and used if available
- Page sizes: 4 KiB (normal), 2 MiB (huge), 1 GiB (giant)
- **PAT** (Page Attribute Table) for cache control on MMIO regions
- **NX bit** enforced: code pages are not writable, data pages are not executable

### CPU Feature Detection

```rust
pub struct X86Features {
    pub sse2: bool,        // Always true on x86-64
    pub sse4_1: bool,
    pub sse4_2: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512vnni: bool,  // INT8 inference
    pub amx_bf16: bool,    // Tile matrix multiply
    pub amx_int8: bool,
}
```

Detected at boot via `CPUID` instruction.

### Interrupt Handling

- **IDT**: 256 entries, first 32 reserved for CPU exceptions
- **Local APIC timer**: Used for scheduler ticks
- **MSI-X**: PCIe message-signaled interrupts for GPU and virtio devices
- All interrupt handlers follow: save registers → call Rust handler → restore → `iretq`

### Crate Structure

```
arch/x86_64/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── boot.rs         # UEFI/Multiboot2 entry point
    ├── gdt.rs          # Global Descriptor Table
    ├── idt.rs          # Interrupt Descriptor Table
    ├── apic.rs         # APIC (local + I/O)
    ├── paging.rs       # 4/5-level page tables
    ├── cpu.rs          # CPUID, feature detection, MSRs
    ├── simd.rs         # AVX/SSE state management (XSAVE)
    ├── serial.rs       # Early console via COM1
    └── entry.asm       # Assembly entry point, trampoline
```

## ARM64 (AArch64) Platform

### Boot

- **UEFI boot** (server/VM): Same UEFI application model as x86
- **Device Tree**: Parse DTB (Device Tree Blob) for hardware description
- **Container entry**: Direct entry, kernel at EL1

### CPU Initialization

```
1. Set up exception vector table (VBAR_EL1)
2. Configure MMU (TCR_EL1, MAIR_EL1, TTBR0_EL1/TTBR1_EL1)
3. Initialize GIC (Generic Interrupt Controller) v3/v4
   - Distributor, Redistributor, CPU interface
4. Enable NEON/SVE (CPACR_EL1)
5. Set up system timer (CNTFRQ_EL0, CNTP_CTL_EL0)
6. Boot secondary CPUs via PSCI (Power State Coordination Interface)
```

### Memory Management

- **4-level page tables** (4 KiB granule: L0 → L1 → L2 → L3)
- Page sizes: 4 KiB, 2 MiB (block descriptor at L2), 1 GiB (block at L1)
- Supports 16 KiB and 64 KiB granules (configurable at build time)
- **MAIR** attributes for cache policy control
- **PAN** (Privileged Access Never) enabled for security
- **BTI** (Branch Target Identification) enabled if available

### CPU Feature Detection

```rust
pub struct Aarch64Features {
    pub neon: bool,        // Always true on AArch64
    pub fp16: bool,        // Half-precision arithmetic
    pub dotprod: bool,     // INT8 dot products
    pub sve: bool,         // Scalable Vector Extension
    pub sve2: bool,
    pub sve_vector_length: usize,  // 128-2048 bits
    pub sme: bool,         // Scalable Matrix Extension
    pub bf16: bool,        // BFloat16 support
    pub i8mm: bool,        // INT8 matrix multiply
}
```

Detected via `ID_AA64ISAR*`, `ID_AA64PFR*`, `ID_AA64MMFR*` system registers.

### Crate Structure

```
arch/aarch64/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── boot.rs         # UEFI / DTB entry point
    ├── exception.rs    # Exception vector table
    ├── gic.rs          # GICv3/v4 driver
    ├── paging.rs       # AArch64 page tables
    ├── cpu.rs          # Feature detection, system registers
    ├── sve.rs          # SVE state management
    ├── psci.rs         # PSCI for SMP boot
    ├── uart.rs         # PL011 UART for early console
    └── entry.S         # Assembly entry, exception vectors
```

## NVIDIA GPU Platform

### Approach: Minimal GPU Driver

SmallAIOS does **not** use NVIDIA's proprietary kernel driver or CUDA runtime.
Instead, it implements a minimal GPU driver that:

1. Enumerates NVIDIA GPUs via PCIe configuration space
2. Maps GPU BAR (Base Address Register) MMIO regions
3. Initializes the GPU command processor
4. Manages GPU memory (VRAM allocation)
5. Submits compute kernels (PTX → SASS via `ptxas` at build time)
6. Handles GPU interrupts for completion notification

This is feasible because NVIDIA's GPU architecture is documented enough through:
- PCIe specification (BAR enumeration, MSI-X)
- NVIDIA's open-source `open-gpu-kernel-modules` (reference for register layout)
- PTX ISA specification (public, for kernel compilation)
- Nouveau project (open-source reference for initialization sequences)

### Clean Room Consideration

The NVIDIA open-gpu-kernel-modules are dual-licensed MIT/GPL. We reference
the **register definitions and initialization sequences** (factual hardware
interface data, not copyrightable expression) but write all code from scratch.

PTX assembly is compiled using `ptxas` (NVIDIA's assembler), which is a
build-time tool dependency, not a runtime dependency.

### GPU Initialization Sequence

```
1. PCIe enumeration: find devices with NVIDIA vendor ID (0x10DE)
2. Map BAR0 (MMIO registers) and BAR1 (GPU memory aperture)
3. Read GPU identification registers (architecture, SM count, memory size)
4. Initialize PMU (Power Management Unit)
5. Initialize memory controller (FB/VRAM configuration)
6. Set up GPU page tables (separate from CPU page tables)
7. Initialize command FIFO (ring buffer for GPU commands)
8. Initialize compute engine(s)
9. Test with a simple kernel launch
```

### Supported GPU Architectures

| Architecture | Compute Capability | Examples |
|---|---|---|
| Volta | 7.0, 7.2 | V100, Jetson Xavier |
| Turing | 7.5 | T4, RTX 2080 |
| Ampere | 8.0, 8.6, 8.7, 8.9 | A100, A10, RTX 3090 |
| Hopper | 9.0 | H100 |
| Blackwell | 10.0 | B200 |

### GPU Memory Management

```
┌─────────────────────────────────────┐
│          GPU VRAM                    │
├─────────────────────────────────────┤
│ Kernel code (PTX/SASS)     [fixed]  │
│ Constant memory            [fixed]  │
│ Tensor data pool           [dynamic]│
│ Workspace memory           [dynamic]│
│ Command buffers            [dynamic]│
└─────────────────────────────────────┘
```

- Simple bump allocator with free-list for GPU VRAM
- CPU-accessible via BAR1 aperture (for small transfers)
- DMA engine (CE - Copy Engine) for bulk transfers
- Pinned CPU memory regions for DMA source/destination

### Kernel Launch

```rust
pub struct GpuLaunchArgs {
    pub grid: [u32; 3],       // Grid dimensions (blocks)
    pub block: [u32; 3],      // Block dimensions (threads)
    pub shared_mem: u32,      // Dynamic shared memory bytes
    pub params: Vec<u64>,     // Kernel parameters
}
```

Kernels are written in PTX, compiled to SASS (native GPU ISA) at build time:

```
model.ptx → ptxas → model.cubin → embedded in SmallAIOS binary
```

### Crate Structure

```
arch/nvidia/
├── Cargo.toml
├── kernels/
│   ├── gemm_f32.ptx      # Matrix multiply (f32)
│   ├── gemm_f16.ptx      # Matrix multiply (f16, tensor cores)
│   ├── conv2d.ptx         # 2D convolution
│   ├── softmax.ptx        # Softmax
│   ├── layernorm.ptx      # Layer normalization
│   ├── elementwise.ptx    # Elementwise ops (add, mul, relu, etc.)
│   └── reduce.ptx         # Reduction ops
├── build.rs               # Compile PTX → SASS via ptxas
└── src/
    ├── lib.rs
    ├── pci.rs             # PCIe enumeration and BAR mapping
    ├── gpu.rs             # GPU initialization and management
    ├── memory.rs          # GPU memory allocator
    ├── fifo.rs            # Command FIFO (ring buffer)
    ├── compute.rs         # Compute engine and kernel launch
    ├── dma.rs             # DMA/Copy Engine
    ├── interrupt.rs       # GPU interrupt handling (MSI-X)
    └── ptx.rs             # PTX binary loader
```

## Virtio Devices (Container/VM Mode)

When running in a VM or container with KVM/QEMU, SmallAIOS uses virtio for I/O:

### virtio-blk

- Read-only block device for loading ONNX models from container image
- Simple request queue: read sectors → memory buffer
- No write support (immutable container image)

### virtio-net

- Minimal network device for IPC TCP transport
- Single TX queue, single RX queue
- No offloading features (minimal driver)

### virtio-console

- Kernel log output
- Debugging interface

### Virtio Driver Architecture

```rust
pub trait VirtioDevice {
    fn negotiate_features(&mut self, offered: u64) -> u64;
    fn setup_queues(&mut self);
    fn activate(&mut self);
}

pub struct VirtQueue {
    pub descriptors: &'static mut [VirtqDesc],
    pub avail: &'static mut VirtqAvail,
    pub used: &'static mut VirtqUsed,
}
```

All virtio drivers use MMIO transport (not PCI, to avoid full PCI stack in container mode).

## Hardware Requirements Summary

| Component | x86-64 | ARM64 | Notes |
|---|---|---|---|
| CPU | SSE2+ (AVX2 recommended) | NEON (SVE recommended) | SIMD required |
| RAM | 64 MB minimum, 1 GB+ recommended | Same | Depends on model size |
| GPU | Optional, NVIDIA CC 7.0+ | Optional, NVIDIA CC 7.0+ | For GPU inference |
| Boot | UEFI 2.0+ or container | UEFI or DTB or container | |
| Timer | APIC/TSC | ARM Generic Timer | Required |
| Interrupts | APIC + MSI-X | GICv3+ | Required |
