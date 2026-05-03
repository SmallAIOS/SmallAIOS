> **Status:** Roadmap stub. Smallest of the three FPGA follow-ups.
> Detailed specs/design/tasks deferred until at least one accelerator backend (`fpga-dpu-backend-v1` or `fpga-custom-npu-v1`) is landing.
> Sibling stubs: `fpga-dpu-backend-v1`, `fpga-custom-npu-v1`.

## Why

`fpga-accelerator-hal-v1` accepts only **statically loaded** bitstreams (preloaded by FSBL, baked into `BOOT.BIN`). That is sufficient for first-light but constrains us:

- Cannot swap bitstreams at runtime without rebooting (e.g., DPU overlay vs custom NPU vs debug overlay)
- Cannot do partial reconfiguration (loading only a region of the PL)
- Cannot recover from a corrupted PL load by reloading without a power cycle

The Zynq UltraScale+ Platform Management Unit (PMU) provides an Inter-Processor Interrupt (IPI) channel that the OS can use to trigger PL configuration via the PCAP / CSU DMA path. Linux's FPGA Manager exercises this; we reimplement enough of it for SmallAIOS.

## What Changes

- New `arch/aarch64-zynqmp::pmu` module: PMU IPI driver (request/response over IPI registers, message format per Xilinx XilPM)
- New `FpgaManager` API: `load_bitstream(&[u8]) -> Result<()>` (full PL reconfig); `load_partial(&[u8], region) -> Result<()>` (partial reconfig — gated behind a feature flag, may be deferred to a v2)
- Bitstream image format support: parse Xilinx `.bit` headers (rich metadata: target device, design name, build date) and validate target device against the running SOM. Accept raw `.bin` (configuration data with no header) only when accompanied by an out-of-band SmallAIOS manifest carrying target-device + signature metadata. Hand off raw configuration data to PMU after validation.
- IRQ-driven completion: wait for PMU response, surface configuration errors (CRC mismatch, unsupported format, partial-reconfig-not-allowed)
- `verified-boot` integration hook: bitstream signature check before handing off to PMU (gated behind the existing `verified-boot` feature; integrates with the project's PQC stack — ML-DSA-65 signatures)
- Documentation: when to use static (FSBL) vs dynamic (FpgaManager) loading; security implications of runtime PL reconfig
- New `just` recipes for runtime bitstream swap demos

Out of scope:
- DPU/NPU drivers themselves (those are separate changes)
- Bitstream encryption (separate concern, may piggy-back on `verified-boot`)
- Multi-tenant PL slicing (way out of scope)

## Capabilities

### New Capabilities

- `fpga-manager`: PMU-driven runtime bitstream loading and (optionally) partial reconfiguration. Detailed requirements TBD.

### Modified Capabilities

- `zynqmp-board`: the board crate gains a PMU IPI driver. May or may not be a spec change vs implementation detail — TBD when this change is fleshed out.

## Impact

**Code:** new `pmu` module in `arch/aarch64-zynqmp`; new `FpgaManager` API surface.

**Build:** no new feature flags by default; partial reconfig and verified-boot integration each behind their own feature.

**Dependencies:** no new runtime crate deps.

**Security:** runtime PL reconfig is a privileged operation. Must be gated behind the capability system; arbitrary user processes SHALL NOT be able to load bitstreams. `verified-boot` integration ensures only signed bitstreams load when the feature is on.

**Architecture:** preserves 4-layer model.

## Predecessors

- `fpga-accelerator-hal-v1` (provides board crate, AXI/DMA, GIC)

## Followers

- May enable: hot-swap demos, multi-overlay benchmarking, `fpga-custom-npu-v1` co-existence with DPU
