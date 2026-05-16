# Tasks — cheri-capability-v1

## 0. Reference reading

- [ ] 0.1 Read the CHERI ISAv9 specification (Watson et al., "Capability Hardware Enhanced RISC Instructions: CHERI Instruction-Set Architecture", UCAM-CL-TR-987). Focus on the 128-bit capability representation, sealing/unsealing semantics, and the RISC-V instruction subset (`cgetbase`, `cgetlen`, `cseal`, `cunseal`, etc.).
- [ ] 0.2 Read the CHERIoT specification (Microsoft Research) for the embedded variant. Note the simplifications (no compressed-encoding 64-bit variant; pure 32-bit) and the implications for an embedded SmallAIOS port.
- [ ] 0.3 Read SRI's "CHERI for the Linux Kernel" porting notes and the FreeBSD-on-CHERI experience reports for porting lessons.
- [ ] 0.4 Survey the `cheri-rust` toolchain status as of the change date — what version of upstream Rust it tracks, what Cargo features are supported, what is broken.

## 1. Phase 1 — Alignment documentation

- [ ] 1.1 Create `docs/cheri-alignment.md` with an opening section: "This document is alignment-on-paper, not hardware-tested. SmallAIOS does not run on CHERI silicon today."
- [ ] 1.2 Add the field-by-field mapping table (CHERI capability fields ↔ SmallAIOS `Capability` struct fields). Cite `kernel/src/cap.rs` line numbers.
- [ ] 1.3 Add the permissions mapping table (CHERI `perms` bits ↔ SmallAIOS `Permissions` enum variants). Call out the CHERI-only permissions (`LOAD_CAP`, `STORE_CAP`, `SEAL`, etc.) that have no SmallAIOS analog today.
- [ ] 1.4 Add the otype/sealed-capability mapping section (CHERI sealing ↔ SmallAIOS PAC-signed handle from `aarch64-mte-pac-hardening-v1`).
- [ ] 1.5 Add a gap-analysis section: list the SmallAIOS capability-model changes that would be required for a hardware CHERI port (handle-pool model → carried-bounds model). Estimate effort.
- [ ] 1.6 Add an "unsafe surface area" section: enumerate the `unsafe` blocks in SmallAIOS that would compile-fail under `cheri-rust` and explain why each one is or isn't a real bug. Focus on the allocator (`kernel/src/mem/heap.rs`), DTB parsing (`kernel/src/mem/phys.rs`), and FFI surfaces (CUDA bindings in `arch/nvidia/`).
- [ ] 1.7 Add a "roadmap if silicon matures" section: bullet list of follow-up changes that would be needed in dependency order (`cheri-capability-v2` for the first hardware test, then SMMU integration, then performance benchmarking, then certification artifacts).

## 2. Phase 2 — Toolchain compile experiment

- [ ] 2.1 Set up a `cheri-rust` development environment on a Linux workstation (Docker image preferred — SRI publishes a `cheri-rust` toolchain image). Document the version + image hash in `notes/cheri-compile-experiment.md`.
- [ ] 2.2 Attempt to compile the `smallaios-security` crate's capability primitives (the `Capability` struct, the `require_capability` function, and the `Permissions` enum) for the `riscv64gc-unknown-cheri-elf` (or equivalent CHERI-RISC-V) target.
- [ ] 2.3 Document each compile error encountered, the hand-fix applied (if any), and the conclusion in `notes/cheri-compile-experiment.md`. Format: errors clustered by category (pointer arithmetic, raw FFI, etc.), counts per category.
- [ ] 2.4 If the capability primitives compile cleanly, attempt one syscall (`sys_mem_alloc`) as a stretch goal. Document the result.
- [ ] 2.5 Write the conclusion section: % CHERI-clean, biggest gaps, recommendation for next steps.

## 3. Phase 3+ — Deferred

- [ ] (Deferred indefinitely) Hardware port — requires production CHERI silicon.
- [ ] (Deferred indefinitely) SMMU + CHERI integration — requires hardware and a settled spec for CHERI + IOMMU interaction.
- [ ] (Deferred indefinitely) Performance benchmarking on CHERI silicon.
- [ ] (Deferred indefinitely) Certification artifacts for CHERI-on-SmallAIOS.

## 4. Verify + archive

- [ ] 4.1 Run `openspec validate cheri-capability-v1 --strict`.
- [ ] 4.2 Squash-merge to `develop`. (No code changes — the merge is doc-only.)
- [ ] 4.3 Open archive PR moving the change to `openspec/changes/archive/YYYY-MM-DD-cheri-capability-v1` and sync the spec deltas to main specs. The archived state retains the deferred Phase 3+ tasks as open research questions.
