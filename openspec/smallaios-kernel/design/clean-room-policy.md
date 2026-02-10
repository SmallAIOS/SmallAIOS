# Clean Room Policy

## Purpose

SmallAIOS is developed under a **clean-room discipline** to ensure all code is
original work, free from licensing encumbrances, and independently copyrightable
under the Apache License 2.0.

## What "Clean Room" Means for SmallAIOS

### We DO reference:

1. **Open specifications and standards** (these describe interfaces, not implementations):
   - POSIX.1-2024 (IEEE 1003.1) — syscall semantics and behavior
   - ONNX IR specification — model format and operator semantics
   - UEFI specification — boot protocol
   - PCIe specification — device enumeration and configuration
   - NVIDIA PTX ISA — GPU instruction set architecture
   - ARM Architecture Reference Manual — instruction set, system registers
   - Intel SDM (Software Developer Manual) — x86 instructions, MSRs, paging
   - Zenoh protocol specification — wire format
   - OCI image/runtime specifications — container format
   - Virtio specification — virtual device interfaces
   - Protobuf encoding specification — wire format

2. **Public API documentation**:
   - NVIDIA CUDA Toolkit documentation (API semantics, not source)
   - Linux syscall man pages (behavioral specification)
   - ACPI specification (power management interfaces)

3. **Published research papers and textbooks**:
   - Operating system design (Tanenbaum, etc.)
   - GEMM optimization techniques (Goto, Van Zee, etc.)
   - Neural network inference optimization literature

### We DO NOT reference or copy:

1. **Linux kernel source code** (GPL-2.0 — license incompatible)
2. **NVIDIA proprietary driver source** (proprietary)
3. **Windows NT/kernel source** (proprietary)
4. **Any GPL-licensed OS kernel code** (Redox is MIT, but we still write original)
5. **Existing ONNX runtime source** (MIT license — compatible but we want clean room)
6. **cuDNN, cuBLAS source** (proprietary NVIDIA libraries)

### We MAY use as build-time tools (not linked into the binary):

1. **rustc / LLVM** — compiler toolchain (Apache 2.0 + MIT)
2. **ptxas** — NVIDIA PTX assembler (EULA permits use as a tool)
3. **cargo** — Rust build system (Apache 2.0 + MIT)
4. **Docker / Buildah** — container image building
5. **QEMU** — testing (GPL, but it's a tool, not linked)
6. **GDB** — debugging (GPL, but it's a tool, not linked)

### Permitted Rust crate dependencies (build-time only):

| Crate | License | Used For | Runtime? |
|---|---|---|---|
| `cc` | MIT/Apache-2.0 | Compile assembly files | Build only |
| `proc-macro2` | MIT/Apache-2.0 | Macro generation | Build only |

### No runtime crate dependencies.

The kernel binary has **zero** external Rust crate dependencies. All functionality
is implemented from scratch. This eliminates supply chain risk and ensures we
control every line of code.

## NVIDIA GPU: Special Considerations

The NVIDIA GPU driver is the most complex clean-room challenge. Our approach:

1. **Register definitions**: We reference register offsets and bit fields from:
   - NVIDIA's open-gpu-kernel-modules (MIT-licensed portion)
   - Nouveau project documentation (reverse-engineered, MIT/X11 licensed)
   - envytools documentation (public hardware documentation)

   Register offsets are factual hardware interface data. We document our source
   for each register definition.

2. **Initialization sequences**: We implement GPU initialization based on:
   - The NVIDIA open-gpu-kernel-modules MIT-licensed code (for reference)
   - Published GPU architecture documentation
   - UEFI GOP (for display/firmware initialization that precedes us)

   Our implementation is original code written from the specifications.

3. **Compute kernels**: Written in PTX assembly, which is a public ISA:
   - PTX specification is published by NVIDIA
   - Our kernels are original implementations of standard algorithms
   - Compiled to SASS using `ptxas` (a build tool, not linked)

4. **Memory management**: Implemented from PCIe BAR specifications and GPU
   architecture documentation. No reference to proprietary memory manager code.

## Contributor Requirements

All contributors must:

1. **Not have viewed** GPL-licensed kernel source code for the same subsystem
   within 6 months of contributing to that subsystem.

2. **Document references** for any non-trivial algorithm. Example:
   ```
   // Reference: "Anatomy of High-Performance Matrix Multiplication"
   // Goto & Van Zee, ACM TOMS 2008
   // Our implementation differs in: [specific differences]
   ```

3. **Sign off** commits with a Developer Certificate of Origin (DCO):
   ```
   Signed-off-by: Name <email>
   ```
   This certifies the contribution is original work or from a compatible license.

4. **Not copy-paste** code from any source. Reading documentation and writing
   original code is fine. Transliterating code line-by-line is not.

## Verification

- All code is reviewed for accidental similarity to known implementations
- Critical subsystems (memory manager, scheduler, GPU driver) have documented
  design rationale explaining architectural choices
- The SBOM (Software Bill of Materials) lists all references and tools used
- Regular audits compare our implementations against known codebases to verify
  independent origin

## License Compatibility Matrix

| Source License | Can Reference Spec? | Can Reference Code? | Can Link? |
|---|---|---|---|
| Apache-2.0 | Yes | Yes | Yes |
| MIT | Yes | Yes (with attribution) | Yes |
| BSD-2/3 | Yes | Yes (with attribution) | Yes |
| GPL-2.0 | Spec only | No | No |
| GPL-3.0 | Spec only | No | No |
| LGPL | Spec only | No | Dynamic only (impractical for us) |
| Proprietary | Public docs only | No | No |
| Public domain | Yes | Yes | Yes |
