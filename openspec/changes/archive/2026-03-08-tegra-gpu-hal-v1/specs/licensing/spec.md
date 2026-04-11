## Licensing Strategy

### Overview

The Tegra GPU HAL involves three distinct licensing tiers that must be kept strictly separate. SmallAIOS is Apache-2.0 licensed. Reference material from the nvgpu project is MIT-licensed. NVIDIA firmware blobs have their own redistributable license. This spec defines the compliance strategy.

### License Tiers

| Tier | License | Scope | Files |
|------|---------|-------|-------|
| **SmallAIOS code** | Apache-2.0 | All driver logic, init sequences, state machines, error handling, tests | `arch/nvidia/src/tegra/*.rs` |
| **Register definitions** | Apache-2.0 (MIT provenance) | Register addresses, bit fields, constants derived from nvgpu reference | `arch/nvidia/src/tegra/regs.rs` |
| **Firmware blobs** | NVIDIA redistributable | Binary firmware for FECS, GPCCS, ACR | `arch/nvidia/firmware/gm20b/` |

### SmallAIOS Code (Apache-2.0)

All Rust source files carry the standard SmallAIOS header:

```rust
// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0
```

The implementation is clean-room: we reference MIT-licensed nvgpu source and public NVIDIA documentation (TRM, open-gpu-doc) for understanding register semantics, then write original Rust code. No code is copied from any source.

### Register Definitions (Apache-2.0 with MIT Provenance)

Register addresses and bit field definitions are factual/functional information. They are not copyrightable expression, but we document their provenance for transparency:

```rust
// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Register addresses and bit field definitions in this file are derived from
// publicly available NVIDIA documentation and the nvgpu project
// (https://github.com/NVIDIA/open-gpu-kernel-modules), which is licensed
// under MIT. See LICENSES/MIT-nvgpu.txt for the original license text.
//
// These definitions represent factual hardware interface specifications
// (register addresses, bit positions, field widths) which are functional
// elements not subject to copyright protection.
```

### Firmware Blobs (NVIDIA License)

Firmware blobs are binary artifacts redistributable under NVIDIA's license:

```
arch/nvidia/firmware/
  gm20b/
    acr_ucode.bin
    fecs_sig.bin
    gpccs_sig.bin
  LICENSE-NVIDIA
```

The `LICENSE-NVIDIA` file contains the NVIDIA Software License Agreement for firmware redistribution.

### License Files

Add to repository root:

```
LICENSES/
  MIT-nvgpu.txt          # MIT license text from nvgpu project
```

The existing `LICENSE` (Apache-2.0) remains the project-wide license. The `LICENSES/` directory holds third-party license texts referenced by provenance comments.

### What We Avoid

- **GPL v2 code:** The nvgpu repository contains `os/linux/` files that are GPL v2. We never reference, read, or derive from these files. Only `drivers/gpu/nvgpu/` (MIT) is used.
- **Proprietary NVIDIA driver code:** The closed-source NVIDIA Linux driver is not referenced.
- **Code copying:** No source code is copied from any external project. All Rust code is original.

### Compliance Checklist

- [ ] All `.rs` files have `SPDX-License-Identifier: Apache-2.0` header
- [ ] `regs.rs` has MIT provenance comment block
- [ ] `LICENSES/MIT-nvgpu.txt` exists with full MIT license text
- [ ] `arch/nvidia/firmware/LICENSE-NVIDIA` exists
- [ ] No references to `os/linux/` GPL v2 files anywhere in codebase
- [ ] README or NOTICE file documents the three-tier licensing approach
