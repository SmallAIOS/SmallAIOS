# Phase 4: NVIDIA GPU Support

## Objective

Implement the minimal NVIDIA GPU driver and CUDA execution provider so that
SmallAIOS can offload inference to NVIDIA GPUs. Target: run ResNet50 inference
on an NVIDIA T4 or A10 GPU.

## Dependencies

- Phase 3 complete (ONNX runtime with CPU execution provider)

## Tasks

### 4.1 PCIe Enumeration
- [ ] Scan PCIe configuration space (type 0/1 headers)
- [ ] Identify NVIDIA devices (vendor ID 0x10DE)
- [ ] Read BARs (Base Address Registers) for MMIO and memory aperture
- [ ] Set up MSI-X interrupts for GPU
- [ ] Map BAR0 (MMIO registers) into kernel virtual address space
- [ ] Map BAR1 (GPU memory aperture) for CPU-accessible GPU memory

### 4.2 GPU Identification
- [ ] Read GPU boot/identification registers
- [ ] Determine architecture (Volta, Turing, Ampere, Hopper)
- [ ] Query SM count, memory size, clock frequencies
- [ ] Compute capability detection
- [ ] Log GPU info at boot

### 4.3 GPU Initialization
- [ ] Power management unit (PMU) initialization
- [ ] Memory controller initialization
- [ ] GPU page table setup (separate from CPU page tables)
- [ ] Falcon microcontroller initialization (if required by architecture)
- [ ] Reference: open-gpu-kernel-modules register definitions (MIT portions)

### 4.4 GPU Memory Manager
- [ ] VRAM partitioning: static (weights, kernels) + dynamic (workspace)
- [ ] Bump allocator with free list for dynamic region
- [ ] Region reset between inference runs
- [ ] CPU-accessible mapping via BAR1 aperture (for small transfers)
- [ ] Pinned host memory management (for DMA)

### 4.5 DMA Engine (Copy Engine)
- [ ] Initialize Copy Engine (CE) via FIFO
- [ ] Host-to-Device transfer (pinned CPU memory → GPU VRAM)
- [ ] Device-to-Host transfer (GPU VRAM → pinned CPU memory)
- [ ] Device-to-Device transfer (intra-GPU copy)
- [ ] Async transfers with completion interrupts (MSI-X)
- [ ] Double-buffering for overlapping compute and transfer

### 4.6 Compute Engine
- [ ] Initialize Compute Engine via FIFO
- [ ] Command buffer construction (NVIDIA push buffer format)
- [ ] Kernel launch: set grid/block dims, parameters, shared memory
- [ ] Synchronization: wait for kernel completion (interrupt-based)
- [ ] Error handling: GPU faults, timeouts

### 4.7 PTX Kernel Development

Write PTX kernels for core inference operators:

- [ ] `gemm_f32.ptx`: Matrix multiply (f32, tiled, shared memory)
- [ ] `gemm_f16.ptx`: Matrix multiply (f16, using tensor cores HMMA)
- [ ] `conv2d.ptx`: 2D convolution (implicit GEMM approach)
- [ ] `elementwise.ptx`: Add, Mul, Relu, Sigmoid (fused, generic)
- [ ] `softmax.ptx`: Softmax (online algorithm, numerically stable)
- [ ] `layernorm.ptx`: Layer normalization (fused)
- [ ] `reduce.ptx`: Sum, Mean, Max reduction (tree-based)
- [ ] `transpose.ptx`: Tensor transpose (shared memory for coalescing)

Build-time compilation:
- [ ] `build.rs` invokes `ptxas` to compile PTX → SASS (cubin)
- [ ] Embed cubin binaries in the SmallAIOS binary
- [ ] Per-architecture compilation (sm_70, sm_75, sm_80, sm_90)

### 4.8 CUDA Execution Provider
- [ ] Register as execution provider with ONNX runtime
- [ ] Graph partitioning: decide which operators run on GPU vs CPU
- [ ] GPU memory planning (VRAM allocation for all tensors)
- [ ] Execution pipeline:
  1. Transfer input tensors to GPU
  2. Launch kernel sequence
  3. Transfer output tensors back to CPU
- [ ] Async execution with compute/transfer overlap
- [ ] Fallback to CPU EP for unsupported operators

### 4.9 Integration Test
- [ ] Run ResNet50 on NVIDIA GPU
- [ ] Compare output to CPU execution provider (must match within tolerance)
- [ ] Measure GPU inference latency
- [ ] Measure GPU memory usage
- [ ] Test on T4 (Turing), A10 (Ampere) if available

### 4.10 Container GPU Support
- [ ] Library OS mode: access GPU via host driver (`/dev/nvidia*` + ioctl)
- [ ] Test with NVIDIA Container Toolkit
- [ ] Docker compose example with GPU support
- [ ] Document GPU passthrough configuration

## Exit Criteria

- GPU detected and initialized on NVIDIA Turing+ hardware
- DMA transfers work (host↔device, verified with pattern test)
- PTX kernels compile and execute correctly
- GEMM kernel achieves ≥ 50% of theoretical GPU peak throughput
- ResNet50 produces correct output on GPU
- GPU inference latency < 5ms for ResNet50 (batch=1) on T4
- Graceful fallback when no GPU is present
- Works in Docker with NVIDIA Container Toolkit
