# Proposal: Intel GPU Support (intel-gpu-support-v5)

## Summary

Add Intel GPU hardware abstraction layer to SmallAIOS, enabling AI inference
workloads on Intel discrete and integrated GPUs. This covers the Xe
architecture family: Xe-LP (integrated), Xe-HPG (Arc consumer), Xe-HPC
(Data Center GPU Max / Ponte Vecchio), Xe-LPG (Meteor Lake), and Xe2
(Battlemage).

## Motivation

Intel's discrete GPU lineup (Arc A-series, Data Center GPU Max, Flex series)
is increasingly deployed in AI inference workloads. SmallAIOS currently
supports NVIDIA GPUs only. Adding Intel GPU support broadens hardware
compatibility and enables deployment on Intel-based edge and data center
platforms without requiring NVIDIA hardware.

Key Intel GPU targets:
- **Arc A770 / A750 / A580 / A380** -- Consumer/workstation discrete GPUs
  (Xe-HPG architecture, up to 32 Xe-cores / 512 EUs)
- **Data Center GPU Max 1550 (Ponte Vecchio)** -- HPC/AI data center GPU
  (Xe-HPC architecture, 128 Xe-cores / 1024 EUs, 128 GB HBM2e)
- **Flex 170** -- Data center visual compute / light inference
  (Xe-HPG derivative, 32 Xe-cores / 512 EUs)
- **Meteor Lake integrated** -- Ultra-low-power edge inference
  (Xe-LPG, up to 8 Xe-cores / 128 EUs)
- **Battlemage** -- Next-generation discrete (Xe2 architecture)

## Scope

1. **PCIe enumeration** -- Discover Intel GPUs (vendor 0x8086) on the PCI bus
2. **GPU identification** -- Map PCI device IDs to architecture, EU count,
   VRAM size, and capabilities
3. **VRAM/GTT memory management** -- Allocator for device-local and
   system-shared memory with static/dynamic region split
4. **DMA / Blitter engine** -- Asynchronous host-to-device, device-to-host,
   and device-to-device transfers via Intel's Blitter copy engine
5. **EU-based compute engine** -- Kernel launch using Intel's Execution Unit
   model (SIMD8/16/32, 8 threads per EU, workgroup dispatch)
6. **GPU initialization** -- BAR mapping, engine bring-up, GuC/HuC firmware
   loading references, power state management
7. **SPIR-V kernel registry** -- Kernel definitions using SPIR-V (Intel's
   shader/compute IL), equivalent to NVIDIA's PTX registry
8. **Level Zero execution provider** -- Top-level provider wiring all
   subsystems together for ONNX operator dispatch, equivalent to NVIDIA's
   CUDA provider

## Non-Goals

- Actual GPU hardware register programming (stub implementations)
- Display/rendering pipeline support
- OpenCL runtime compatibility
- Multi-GPU / multi-tile support (future work)

## Dependencies

- `smallaios-kernel` crate (core types)
- No external C dependencies (`#![no_std]` throughout)

## Success Criteria

- All modules compile under `#![no_std]` with `extern crate alloc`
- 150+ unit tests passing across all modules
- Zero clippy warnings with `-D warnings`
- Consistent API pattern with existing NVIDIA GPU crate
- Feature flags for each Intel GPU architecture generation
