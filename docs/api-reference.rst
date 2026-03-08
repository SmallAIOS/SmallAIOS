API Reference
=============

API documentation is auto-generated from Rust doc comments using ``cargo doc``.

To generate locally:

.. code-block:: bash

   cargo doc --workspace --no-deps --open

The 18-crate workspace includes:

- ``smallaios-kernel`` — memory management, scheduler, syscall interface
- ``smallaios-security`` — capability system, PQC crypto stack
- ``smallaios-onnx-rt`` — ONNX protobuf parser, graph optimizer, operators
- ``smallaios-net`` — IPv4/IPv6, TCP/UDP, QUIC, HTTP/3, TLS 1.3
- ``smallaios-ipc`` — Zenoh-inspired pub/sub messaging
- ``smallaios-bus`` — CAN, ARINC 429/664, MIL-STD-1553, SpaceWire, DDS
- ``smallaios-peripheral`` — I2C, SPI, GPIO, UART, CSI camera, I2S audio
- ``smallaios-usb`` — USB core, xHCI host controller, gadget framework
- ``smallaios-sdr`` — HackRF One, ADALM-Pluto, IQ pipeline
- ``smallaios-container`` — Docker/K8s interface, health, metrics
- ``smallaios-posix`` — minimal POSIX compatibility layer
- ``smallaios-bench`` — benchmarks and performance reporting
- Architecture HALs: ``smallaios-arch-x86_64``, ``smallaios-arch-aarch64``, ``smallaios-arch-riscv64``, ``smallaios-arch-nvidia``, ``smallaios-arch-intel-gpu``, ``smallaios-arch-amd``
