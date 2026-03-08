Getting Started
===============

Prerequisites
-------------

- Rust nightly (pinned in ``rust-toolchain.toml``, currently ``nightly-2026-02-01``)
- ``make``
- QEMU (for VM testing): ``qemu-system-x86_64``, ``qemu-system-aarch64``, ``qemu-system-riscv64``
- Docker with Buildx (for container mode)

Building
--------

Container Mode (Library OS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   make build-container-x86    # x86_64-unknown-linux-musl
   make build-container-arm    # aarch64-unknown-linux-musl

Kernel Mode (VM / Bare Metal)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   make build-kernel-x86       # x86_64-unknown-none
   make build-kernel-arm       # aarch64-unknown-none

Running
-------

QEMU
~~~~

.. code-block:: bash

   make run-x86    # Boot in QEMU x86-64
   make run-arm    # Boot in QEMU ARM64

Docker
~~~~~~

.. code-block:: bash

   make docker-build           # Multi-arch container build
   docker run --rm smallaios:latest

The Docker image is ~600 KB (``FROM scratch``) and supports ``--health-check`` for readiness probes.

Testing
-------

.. code-block:: bash

   make test        # Run all unit tests (~4,100+ tests)
   make clippy      # Lint with clippy
   make fmt-check   # Verify formatting

First Inference
---------------

SmallAIOS boots directly to ONNX inference. In container mode, place an ONNX model
at the configured path and the runtime will parse, optimize, and execute it using
the built-in CPU execution provider.

The ONNX runtime supports: MatMul, Conv, Relu, Softmax, Add, Reshape, and GEMM operators.
