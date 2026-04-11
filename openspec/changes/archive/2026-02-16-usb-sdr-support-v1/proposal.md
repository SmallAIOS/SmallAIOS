## Why

SmallAIOS lacks any USB support — no host controller driver, no device enumeration, no USB protocol stack. This prevents two high-value deployment scenarios: (1) connecting USB peripherals such as software-defined radios (SDR) for edge AI on RF signals, and (2) exposing SmallAIOS itself as a USB device so it can function as a plug-in inference appliance (like a Coral USB Accelerator, but running the full ONNX stack). USB is the most common peripheral interface on edge platforms (Jetson, Zynq, Raspberry Pi) where SmallAIOS targets deployment.

## What Changes

- Add USB core stack: descriptor parsing, device enumeration, endpoint management, transfer types (control, bulk, interrupt, isochronous)
- Add xHCI (USB 3.x) host controller driver with PCIe discovery and MMIO register access
- Add USB device/gadget controller framework for platforms with USB device-mode hardware (Zynq, Jetson)
- Add USB inference gadget: SmallAIOS presents as a USB device accepting ONNX inference requests and returning results
- Add HackRF One SDR driver: vendor control transfers (VID `0x1D50`/PID `0x6089`) + bulk IQ streaming
- Add ADALM-PLUTO SDR driver: IIO protocol client over vendor USB interface (VID `0x0456`/PID `0xb673`)
- Add IQ-to-ONNX pipeline: ring buffer ingestion of IQ samples, windowing, and feeding into ONNX inference models for RF signal classification
- Extend device HAL with USB host controller and USB device controller traits
- Extend device syscalls (0x40-0x4F) to support USB device enumeration and I/O

## Capabilities

### New Capabilities

- `usb-core`: USB core protocol stack — descriptor parsing (device, configuration, interface, endpoint), device enumeration and address assignment, endpoint management, control/bulk/interrupt transfer state machines, USB 2.0 High-Speed and USB 3.x SuperSpeed support, hub driver for multi-device topologies
- `xhci-host`: xHCI (eXtensible Host Controller Interface) driver — PCIe discovery of xHCI controllers, capability/operational/runtime register access, device context management, transfer ring and event ring handling, command ring for device slot allocation, port status change detection, MSI-X interrupt support
- `usb-gadget`: USB device/gadget controller framework — device controller abstraction trait, gadget function registration, USB descriptors composition for device-mode presentation, endpoint configuration for IN/OUT directions, support for Zynq USB OTG and Tegra XUSB device controllers
- `usb-inference-gadget`: USB inference appliance function — SmallAIOS presents as a vendor-class USB device, host submits ONNX inference requests (model selection + input tensor) over bulk OUT, SmallAIOS returns inference results over bulk IN, zero-copy DMA integration for tensor data, Zenoh bridge so USB-submitted requests route through standard IPC
- `hackrf-driver`: HackRF One SDR device driver — USB vendor control transfers for all 48 configuration commands (SET_FREQ, SAMPLE_RATE_SET, SET_LNA_GAIN, SET_VGA_GAIN, SET_TRANSCEIVER_MODE, etc.), bulk IQ streaming on EP 0x81 (RX IN) and EP 0x02 (TX OUT), 8-bit signed I/Q sample format, multi-buffer async transfers for throughput
- `pluto-sdr-driver`: ADALM-PLUTO SDR device driver — USB composite device parsing to locate vendor-specific IIO interface, IIOD text protocol client (PRINT, OPEN, CLOSE, READ, WRITE, READBUF, WRITEBUF, TIMEOUT), IIO context discovery and attribute-based AD9363 configuration (frequency, sample rate, gain, bandwidth), bulk endpoint streaming for IQ data
- `sdr-onnx-pipeline`: SDR-to-ONNX inference pipeline — IQ sample ring buffer with configurable depth, windowing (Hann, Hamming, rectangular) for spectral analysis, FFT preprocessing for frequency-domain features, tensor formatting for ONNX input (real/imag interleaved or magnitude/phase), continuous streaming inference with result publishing to Zenoh key expressions (`sdr/{device}/{model}`)

### Modified Capabilities

- `05-device-hal`: Extend HAL with `UsbHostController` trait (init, port_status, device_attach, transfer_submit, transfer_poll) and `UsbDeviceController` trait (init, set_address, configure_endpoint, gadget_write, gadget_read). Add USB-specific error variants to `HalError`.
- `04-ipc-messaging`: Add USB inference gadget as a Zenoh transport endpoint — inference requests arriving over USB are published to Zenoh key expressions, enabling the same model serving pipeline regardless of whether the request came over TCP, QUIC, or USB.

## Impact

- **Rust workspace**: Add `usb` crate (core stack, xHCI host, gadget framework, inference gadget), add `sdr` crate (HackRF driver, PlutoSDR driver, IQ pipeline) with feature flags per device (`hackrf`, `pluto`)
- **Kernel HAL**: New `UsbHostController` and `UsbDeviceController` traits in `kernel/src/hal.rs`, new HalError variants (`UsbTransferError`, `UsbStall`, `UsbDeviceNotFound`, `UsbEndpointHalted`)
- **Device syscalls**: Implement stubbed syscalls 0x40-0x44 for USB device enumeration, open/close, ioctl, and DMA allocation
- **PCIe integration**: xHCI host controller discovered via existing PCIe enumeration in arch crates (class code 0x0C, subclass 0x03, prog-if 0x30)
- **Hardware dependencies**: Any xHCI-capable USB host (standard on x86, Jetson, many ARM64 SBCs), HackRF One (VID `0x1D50`/PID `0x6089`), ADALM-PLUTO (VID `0x0456`/PID `0xb673`)
- **Formal verification**: TLA+ models for USB enumeration state machine, xHCI transfer ring producer/consumer, and IQ ring buffer overflow protection
- **Safety certification**: USB core and xHCI driver subject to DO-178C DAL B (not DAL A — USB is inherently non-deterministic, unsuitable for primary flight controls, but appropriate for mission computing and sensor data)
