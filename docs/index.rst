SmallAIOS
=========

A minimal, secure, Rust-based OS kernel purpose-built for AI inference workloads.

.. image:: https://github.com/SmallAIOS/SmallAIOS/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/SmallAIOS/SmallAIOS/actions/workflows/ci.yml

.. image:: https://codecov.io/gh/SmallAIOS/SmallAIOS/graph/badge.svg
   :target: https://codecov.io/gh/SmallAIOS/SmallAIOS

.. image:: https://img.shields.io/badge/license-Apache--2.0-blue.svg
   :target: https://github.com/SmallAIOS/SmallAIOS/blob/main/LICENSE

Key Features
------------

- **Unikernel architecture** — single address space, ~46 syscalls (vs Linux ~450)
- **Clean-room ONNX runtime** — ``#![no_std]`` Rust, 7 operators, CPU + CUDA providers
- **Post-quantum cryptography** — ML-KEM-768, ML-DSA-65, hybrid signatures
- **Multi-architecture** — x86-64, AArch64, RISC-V 64, NVIDIA Tegra
- **Safety-critical** — DO-178C DAL A compliance target, TLA+/SPIN/Kani verification
- **Tiny footprint** — <8 MB kernel, <600 KB Docker image, <50ms container boot

Quick Links
-----------

- :doc:`getting-started` — build, deploy, and run your first inference
- :doc:`architecture` — 4+1 architectural view model
- :doc:`requirements` — DO-178C requirements and traceability
- `GitHub Repository <https://github.com/SmallAIOS/SmallAIOS>`_

.. toctree::
   :maxdepth: 2
   :caption: Overview
   :hidden:

   architecture

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   getting-started
   bare-metal-deployment
   local-testing
   boot-security-matrix

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api-reference
   requirements
   traceability
   misra-rust-policy

.. toctree::
   :maxdepth: 2
   :caption: Compliance
   :hidden:

   nist/index

.. toctree::
   :maxdepth: 1
   :caption: Project
   :hidden:

   changelog
