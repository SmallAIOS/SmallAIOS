## Context

SmallAIOS is a `#![no_std]` Rust unikernel for ONNX inference. The kernel provides stubbed device syscalls (0x40-0x4F) and HAL traits for bus controllers (`BusController`) and FPGA fabric (`FpgaFabric`), but has no USB support. PCIe enumeration exists in the GPU architecture crates (NVIDIA, AMD, Intel) and can discover xHCI controllers. The existing bus crate handles deterministic safety-critical protocols (CAN, ARINC, MIL-STD-1553, SpaceWire, CCSDS, DDS) — USB is fundamentally different: asynchronous, hot-pluggable, and non-deterministic.

The user has two specific SDR devices: an ADALM-PLUTO (Analog Devices, Zynq-7010 SoC, AD9363 RF transceiver, USB 2.0 vendor IIO interface) and a HackRF One (Great Scott Gadgets, LPC4320 MCU, USB 2.0 vendor-class with bulk streaming). Both connect via USB 2.0 High-Speed. Additionally, the user wants SmallAIOS to be connectable over USB — acting as a USB inference appliance that a host PC plugs into.

Key constraint: all implementations must be clean-room from public specifications (USB 2.0/3.x specs from usb.org, xHCI spec from Intel, IIOD protocol from libiio docs, HackRF vendor protocol from public firmware headers).

## Goals / Non-Goals

**Goals:**
- USB host controller stack sufficient to enumerate and communicate with USB 2.0 High-Speed devices
- xHCI host controller driver discovered via PCIe (reuse existing PCIe enumeration)
- USB device/gadget mode so SmallAIOS can present itself as a USB peripheral
- HackRF One driver: configure RF parameters via vendor control transfers, stream IQ samples via bulk endpoints
- ADALM-PLUTO driver: IIO protocol client over vendor USB bulk endpoints, attribute-based AD9363 configuration
- IQ sample pipeline from SDR to ONNX inference (ring buffer → windowing → tensor → model → Zenoh publish)
- USB inference gadget: host PC submits inference requests over USB, gets results back
- Integration with existing device syscalls and HAL trait system

**Non-Goals:**
- USB 1.x (UHCI/OHCI) support — legacy, not relevant for target platforms
- USB hub driver with multi-level topology — single-level (root hub ports) is sufficient initially
- USB class drivers beyond vendor-specific — no HID, mass storage, audio, video, CDC networking
- Isochronous transfer support — not needed for SDR (bulk is used) or inference gadget
- ADALM-PLUTO IIOD v1.x binary protocol — v0.x text protocol is simpler and sufficient
- FPGA-based SDR (running SmallAIOS directly on the PlutoSDR's Zynq) — future work
- Real-time DSP (demodulation, decoding) — only classification/detection via ONNX
- USB Power Delivery or USB-C alternate mode negotiation

## Decisions

### 1. Crate Structure: `usb` Crate + `sdr` Crate

**Decision**: Create two new workspace crates: `smallaios-usb` (USB core + xHCI + gadget framework) and `smallaios-sdr` (SDR device drivers + IQ pipeline). The `sdr` crate depends on `usb`.

**Rationale**: USB and SDR are separable concerns. The USB crate is a general-purpose peripheral bus stack reusable for future USB device classes. The SDR crate is domain-specific. Separating them keeps the USB crate clean and allows SDR-only or USB-only feature selection.

**Feature flags**:
- `usb`: `xhci` (default), `gadget`, `inference-gadget`
- `sdr`: `hackrf` (default), `pluto`, `iq-pipeline`

**Alternatives considered**:
- Everything in one crate: rejected — USB core is general-purpose, SDR is domain-specific
- SDR drivers in `bus` crate: rejected — USB devices are not deterministic buses, mixing them with CAN/ARINC would dilute the safety-critical focus of `bus`

### 2. USB Host Controller: xHCI Only

**Decision**: Support only xHCI (USB 3.x host controller interface). Do not implement EHCI, OHCI, or UHCI.

**Rationale**: All modern platforms targeted by SmallAIOS (Intel/AMD x86, Jetson, Zynq UltraScale+, RPi 4/5) have xHCI controllers. xHCI supports USB 2.0 devices natively through its transaction translator — there is no need for a separate EHCI driver to talk to USB 2.0 devices like the HackRF and PlutoSDR. The xHCI spec is publicly available from Intel.

**Alternatives considered**:
- EHCI + xHCI: rejected — doubles implementation effort, EHCI is legacy, xHCI handles USB 2.0
- Virtio-USB in QEMU: considered for testing — may add later, but real xHCI is the primary target

### 3. USB Descriptor Parsing: Zero-Copy, `#![no_std]`

**Decision**: Implement USB descriptor parsing as zero-copy views over raw byte buffers, similar to how the `net` crate parses packet headers.

**Rationale**: USB descriptors are nested and variable-length. A zero-copy approach avoids heap allocation (critical in `#![no_std]`) and is consistent with SmallAIOS's existing network packet parsing patterns. The descriptor parser validates lengths and types, then returns typed views.

**Key types**:
```
UsbDeviceDescriptor     — VID, PID, class, num_configurations
UsbConfigDescriptor     — num_interfaces, max_power
UsbInterfaceDescriptor  — class, subclass, protocol, num_endpoints
UsbEndpointDescriptor   — address, direction, transfer_type, max_packet_size
```

### 4. xHCI Driver Architecture: Ring-Based DMA

**Decision**: Implement the xHCI driver using the spec's ring-based DMA architecture: Command Ring (host→controller commands), Transfer Rings (per-endpoint data transfers), and Event Ring (controller→host completion notifications).

**Rationale**: This is how xHCI works — there is no alternative architecture. The key design decisions within xHCI are:
- **Scratchpad buffer allocation**: allocate from DMA-capable memory during init
- **Device context arrays**: statically allocate for max 16 device slots (sufficient for SDR + inference gadget use cases)
- **Event ring**: single interrupter with MSI-X interrupt for completion notification
- **Transfer rings**: 256 TRBs per ring, sufficient for multi-buffered bulk streaming

### 5. USB Gadget: Platform-Specific Device Controllers

**Decision**: Define a `UsbDeviceController` HAL trait. Provide implementations in architecture crates for platforms with USB OTG/device hardware (Zynq XUSB, Tegra XUSB, DWC3).

**Rationale**: USB device mode requires hardware support — not all platforms have it. The HAL trait abstracts the device controller, and the gadget framework composes USB descriptors and routes endpoint data to registered gadget functions. The inference gadget is one such function.

**UsbDeviceController trait**:
```rust
trait UsbDeviceController {
    fn init(&mut self) -> Result<(), HalError>;
    fn set_address(&mut self, addr: u8) -> Result<(), HalError>;
    fn configure_endpoint(&mut self, ep: &EndpointConfig) -> Result<(), HalError>;
    fn stall_endpoint(&mut self, ep_addr: u8) -> Result<(), HalError>;
    fn write_endpoint(&mut self, ep_addr: u8, data: &[u8]) -> Result<usize, HalError>;
    fn read_endpoint(&mut self, ep_addr: u8, buf: &mut [u8]) -> Result<usize, HalError>;
    fn poll_events(&mut self) -> UsbDeviceEvent;
}
```

### 6. HackRF One Driver: Stateless Vendor Requests

**Decision**: Implement the HackRF driver as a thin wrapper over USB vendor control transfers. Configuration state is maintained in the driver struct, not cached from the device.

**Rationale**: The HackRF protocol is simple — all 48 configuration commands are USB control transfers on EP0 (`bmRequestType = USB_TYPE_VENDOR | USB_RECIP_DEVICE`). IQ data flows on bulk endpoints (EP 0x81 IN for RX, EP 0x02 OUT for TX). The device firmware is stateful but the host driver need only track current mode (OFF/RX/TX) and configuration (frequency, sample rate, gains).

**Streaming design**: Submit 4 concurrent bulk IN transfers of 262,144 bytes each (matching libhackrf's `TRANSFER_COUNT` and `TRANSFER_BUFFER_SIZE`). Process completed transfers and resubmit in a ring pattern. IQ data is 8-bit signed interleaved (I,Q,I,Q...), 2 bytes per complex sample.

### 7. ADALM-PLUTO Driver: IIOD Text Protocol Client

**Decision**: Implement the IIOD v0.x text protocol over the vendor USB bulk interface. Do not implement the CDC/RNDIS/NCM network path.

**Rationale**: The PlutoSDR's primary data path is the vendor-specific IIO USB interface, not the virtual Ethernet. The IIOD v0.x protocol is ASCII-based and straightforward to parse:
- `PRINT\n` → returns XML context description
- `WRITE <dev> <channel> <attr>\n<len>\n<data>` → set attribute
- `READ <dev> <channel> <attr>\n` → get attribute
- `OPEN <dev> <samples> <mask>\n` → open streaming buffer
- `READBUF <dev> <bytes>\n` → read IQ samples

This avoids implementing XML parsing for the full IIO context — we hardcode knowledge of the AD9363's channel layout (voltage0/voltage1 for I/Q, altvoltage0/1 for LO) and only parse the attributes we need (frequency, sampling_frequency, gain, bandwidth).

**Alternatives considered**:
- IIOD v1.x binary: rejected — more complex, async multiplexed opcodes, not needed for single-device use
- CDC network + IP-based libiio: rejected — requires a full TCP/IP stack to the device, unnecessary overhead

### 8. IQ-to-ONNX Pipeline: Ring Buffer + Windowed Inference

**Decision**: Implement a configurable IQ ring buffer that feeds windowed chunks into ONNX models. The pipeline runs as a cooperative async task at INFERENCE priority.

**Rationale**: SDR IQ data is continuous and high-rate (up to 20 MSPS on both devices). The inference pipeline must decouple the USB streaming rate from the inference rate. A ring buffer absorbs bursts, and the inference task consumes fixed-size windows at its own pace. Dropped samples are acceptable (detection/classification, not recording).

**Pipeline stages**:
```
USB bulk RX → IQ Ring Buffer → Windowing → [Optional FFT] → Tensor Format → ONNX Inference → Zenoh Publish
```

**Configurable parameters**:
- Ring buffer depth (default: 1M samples = 2 MB for 8-bit, 4 MB for 16-bit)
- Window size (default: 1024 samples)
- Window overlap (default: 0 for classification, 50% for detection)
- Window function (Hann, Hamming, rectangular)
- FFT preprocessing (on/off — frequency domain vs. time domain input)
- Output key expression pattern: `sdr/{device}/{model}`

### 9. USB Inference Gadget: Vendor-Class Bulk Protocol

**Decision**: The inference gadget presents as a vendor-class USB device (class 0xFF) with a simple request/response protocol over bulk endpoints.

**Rationale**: Using vendor class avoids the complexity of fitting inference into an existing USB class (CDC, mass storage, etc.). The protocol is:

**Request format** (bulk OUT, host → SmallAIOS):
```
[4 bytes: request_id][2 bytes: model_name_len][model_name][4 bytes: tensor_size][tensor_data]
```

**Response format** (bulk IN, SmallAIOS → host):
```
[4 bytes: request_id][2 bytes: status][4 bytes: result_size][result_data]
```

This is deliberately simple — a host-side library (future, out of scope) would provide a friendlier API. Requests are bridged to Zenoh (`usb/inference/{model}`) so the same ONNX runtime serves USB, TCP, and QUIC clients.

### 10. Integration with Existing PCIe Enumeration

**Decision**: Extend the PCIe scanning in arch crates to detect xHCI controllers (class 0x0C, subclass 0x03, programming interface 0x30) alongside GPU devices.

**Rationale**: The arch/nvidia, arch/amd, and arch/intel_gpu crates already implement full PCIe bus scanning. Rather than duplicate this, we extend the PCI device classification to also capture xHCI controllers and pass them to the USB crate for initialization. On ARM64 platforms where xHCI may be platform-integrated (not PCIe), the HAL provides a platform-specific discovery path.

## Risks / Trade-offs

**[Risk] USB non-determinism in safety-critical context** → USB is inherently non-deterministic (hot-plug, variable latency, bus contention). Mitigation: classify USB as DAL B (mission computing) not DAL A (flight control). USB data is sensor input, not control output. Safety-critical control loops use CAN/ARINC/1553 buses.

**[Risk] xHCI complexity** → The xHCI spec is ~600 pages with complex ring and context management. Mitigation: implement minimal subset — only bulk and control transfers, skip isochronous and interrupt. Limit to 16 device slots. No hub support beyond root hub ports.

**[Risk] PlutoSDR IIO protocol fragility** → The IIOD protocol is not formally specified; behavior is defined by the libiio source code. Mitigation: target IIOD v0.x text protocol which is simple and stable. Hardcode AD9363 attribute paths rather than parsing XML context dynamically.

**[Risk] IQ data rate vs. USB bandwidth** → Both devices max at USB 2.0 HS (480 Mbps theoretical, ~40 MB/s practical). At 20 MSPS with 8-bit samples, HackRF needs 40 MB/s — right at the limit. Mitigation: document that sustained 20 MSPS requires careful USB bus management (no other high-bandwidth devices). Typical inference use cases need 2-5 MSPS which is well within budget.

**[Risk] USB gadget hardware availability** → Not all platforms have USB device-mode controllers. Mitigation: gadget support is behind a feature flag (`gadget`). Document supported platforms (Zynq, Jetson). x86 desktop/server platforms typically lack USB device mode — the inference gadget targets embedded deployments.

**[Trade-off] Separate `usb` and `sdr` crates vs. single crate** → Two crates adds workspace complexity but properly separates general-purpose USB from domain-specific SDR. The USB crate remains reusable for future USB device classes.

**[Trade-off] IIOD v0.x text vs. v1.x binary** → Text protocol is simpler but slightly less efficient. Acceptable because PlutoSDR bandwidth is limited by USB 2.0 anyway, not protocol overhead.

## Open Questions

1. **DWC3 vs. platform-specific gadget controllers**: The Synopsys DWC3 is used on both Zynq and Jetson. Should we implement one DWC3 driver usable across platforms, or separate per-platform implementations?

2. **USB suspend/resume**: Should SmallAIOS handle USB suspend/resume power management, or keep devices always active? Edge deployments may care about power; datacenter deployments don't.

3. **Multi-SDR support**: Should the IQ pipeline support multiple SDR devices simultaneously (e.g., HackRF on one frequency + PlutoSDR on another)? This multiplies ring buffers and inference tasks.

4. **ONNX model format for RF classification**: What input tensor format should be standardized? Options: raw IQ (Nx2 float32), magnitude/phase (Nx2 float32), spectrogram (NxM float32). This affects the pipeline preprocessing stage.

5. **USB inference gadget discovery**: Should the gadget expose a USB string descriptor advertising available ONNX models, or require the host to query via the bulk protocol?
