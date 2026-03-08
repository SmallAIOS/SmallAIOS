## 1. Create Justfile

- [x] 1.1 Create `Justfile` with top-level settings (`set shell`, `set dotenv-load`) and variables (`cargo`, `docker`, `qemu_x86`, `qemu_arm`, `qemu_rv`, `build_std`, `max_kernel_size_mb`)
- [x] 1.2 Add container build recipes: `build-container-x86`, `build-container-arm` with GPU feature flag parameter
- [x] 1.3 Add kernel build recipes: `build-kernel-x86`, `build-kernel-arm`, `build-kernel-riscv` (release) and `build-kernel-x86-debug`, `build-kernel-arm-debug`, `build-kernel-riscv-debug`
- [x] 1.4 Add QEMU recipes: `run-x86`, `run-arm`, `run-riscv`, `debug-x86`, `run-x86-net`
- [x] 1.5 Add Docker recipes: `docker-build`, `docker-push`, `docker-local`, `docker-local-gpu`, `docker-local-clean`
- [x] 1.6 Add testing recipes: `test`, `clippy`, `fmt`, `fmt-check`
- [x] 1.7 Add formal verification recipes: `tla-verify`, `spin-verify`
- [x] 1.8 Add supply chain recipes: `deny`, `check-cycles`
- [x] 1.9 Add dependency analysis recipes: `depgraph`, `modgraph` (with optional crate parameter), `arch-check`, `dsm`, `arch`
- [x] 1.10 Add release recipes: `changelog`, `release-dry-run bump`, `release bump`
- [x] 1.11 Add deployment recipes: `deploy-netboot`, `deploy-rpi-sdcard dev`, `deploy-rpi-sdcard-update dev`, `deploy-jetson l4t`, `serial dev`
- [x] 1.12 Add size check recipes: `check-size-x86`, `check-size-arm`, `check-size-riscv`, `check-size-jetson`, `check-size`
- [x] 1.13 Add Jetson recipes: `dtb-jetson`, `build-kernel-jetson`, `sdcard-jetson`
- [x] 1.14 Add VMware recipe: `vmware-x86`
- [x] 1.15 Add `clean` recipe
- [x] 1.16 Add doc comments to all recipes for `just --list` output

## 2. Update CI Workflows

- [x] 2.1 Add `just` installation step to CI jobs that use `make` — use `taiki-e/install-action@just`
- [x] 2.2 Replace all `make <target>` invocations with `just <recipe>` in `.github/workflows/ci.yml`
- [x] 2.3 Update the `dependency-analysis` job to use `just depgraph`, `just arch-check`, `just dsm`

## 3. Update Documentation

- [x] 3.1 Update CLAUDE.md build commands section — replace all `make` examples with `just` equivalents
- [x] 3.2 Update CLAUDE.md dev tools section to include `just` installation (`cargo install just`)

## 4. Update Pre-commit Hooks

- [x] 4.1 Update `.pre-commit-config.yaml` — change `module-cycles` hook from `make arch-check` to `just arch-check`

## 5. Cleanup

- [x] 5.1 Delete `Makefile`
- [x] 5.2 Verify `just --list` shows all recipes with descriptions
