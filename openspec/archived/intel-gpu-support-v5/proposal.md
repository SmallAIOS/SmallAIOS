# Intel GPU Support - Proposal

## Change ID
`intel-gpu-support-v5`

## Summary
Add Intel GPU Hardware Abstraction Layer (HAL) to SmallAIOS, enabling AI inference workloads on Intel Xe-LP, Xe-HPG, and Xe-HPC discrete and integrated GPUs.

## Motivation
Intel GPUs are widely deployed in client systems (integrated Xe-LP), gaming/workstation (Arc series, Xe-HPG), and data center (Ponte Vecchio, Xe-HPC). Supporting Intel GPUs enables SmallAIOS to run AI inference on the broadest possible hardware base, including cost-effective integrated GPU acceleration without dedicated discrete hardware.

## Scope
- PCIe enumeration for Intel GPU devices (vendor 0x8086)
- GPU identification: Xe-LP (Gen12), Xe-HPG (Alchemist/Arc), Xe-HPC (Ponte Vecchio)
- GPU initialization and lifecycle management
- Local memory / VRAM management with static/dynamic regions
- DMA/copy engine for host<->device transfers
- EU-based compute dispatch (Execution Units instead of CUDA cores)
- SPIR-V kernel registry (Intel's shader/compute ISA, analogous to NVIDIA's PTX)
- Level Zero-inspired execution provider for ONNX operator dispatch
- Comprehensive test suite

## Out of Scope
- Display/rendering functionality
- OpenCL runtime (we use Level Zero abstractions directly)
- Multi-GPU / tile-to-tile interconnect (future work)

## Dependencies
- `smallaios-kernel` crate (memory, error types)

## Risk Assessment
- Low: follows established pattern from arch/nvidia crate
- Medium: Intel GPU documentation is less openly available than NVIDIA's
