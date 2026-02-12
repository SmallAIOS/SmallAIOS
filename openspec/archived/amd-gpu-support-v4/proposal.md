# AMD GPU Support Proposal

## Summary

Add AMD GPU hardware abstraction layer (arch/amd crate) to SmallAIOS, enabling AI inference workloads on AMD Radeon and Instinct GPUs via ROCm/HIP-compatible abstractions.

## Motivation

SmallAIOS currently supports NVIDIA GPUs exclusively. AMD GPUs (RDNA for consumer/edge, CDNA for datacenter) represent a significant and growing share of the AI accelerator market. Adding AMD support enables:

- Datacenter deployment on AMD Instinct MI200/MI300 series
- Edge inference on RDNA-based APUs and discrete GPUs
- Multi-vendor GPU strategy reducing vendor lock-in
- Cost-competitive inference alternatives

## Scope

- PCIe enumeration for AMD GPUs (vendor ID 0x1002)
- GPU identification: RDNA 1/2/3, CDNA 1/2/3 architectures
- VRAM memory allocator with static/dynamic regions
- DMA engine for host-device transfers
- Wavefront-based compute engine (64-wide wavefronts for CDNA, 32-wide for RDNA)
- ROCm/HIP execution provider mapping ONNX operators to GPU kernels
- Comprehensive test suite

## Non-Goals

- Actual hardware register programming (stub implementation like NVIDIA crate)
- Display/graphics pipeline support
- Multi-GPU peer-to-peer (future work)
- ROCm userspace runtime integration (bare-metal only)

## Dependencies

- `smallaios-kernel` crate (same as NVIDIA crate)
- No external C dependencies (pure `#![no_std]` Rust)

## Timeline

Single implementation phase covering all modules.
