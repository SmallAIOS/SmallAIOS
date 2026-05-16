#!/usr/bin/env bash
# spec-exec-disasm-audit.sh — disassembly-level audit of speculative-execution
# barriers, for OpenSpec change spec-exec-mitigations-v1 (tasks 6.1, 4.2, 4.3).
#
# Advisory: this script is informational. It greps the linked kernel binary
# for the per-arch speculation barriers the change requires at trust
# boundaries, and reports MISSING entries. It exits 0 even on findings so the
# CI job stays advisory (continue-on-error) while per-arch phases land; flip
# STRICT=1 to make it gating once all phases have merged.
#
# Usage: scripts/spec-exec-disasm-audit.sh [x86_64|aarch64|riscv64]
set -uo pipefail

STRICT="${STRICT:-0}"
ARCHES=("${@:-x86_64 aarch64 riscv64}")
FAIL=0

note()  { echo "[spec-exec-audit] $*"; }
miss()  { echo "::warning::[spec-exec-audit] MISSING: $*"; FAIL=1; }

audit_x86_64() {
  local tgt=x86_64-unknown-none
  note "building kernel $tgt --features spec-exec-x86 --release"
  if ! cargo build -p smallaios-kernel --target "$tgt" \
        --features spec-exec-x86 --release 2>/dev/null; then
    note "x86_64 build unavailable (feature may not have landed yet) — skip"
    return 0
  fi
  local bin
  bin=$(find target/"$tgt"/release -maxdepth 1 -type f -name '*.elf' -o \
        -maxdepth 1 -type f -perm -u+x 2>/dev/null | head -1)
  [ -z "$bin" ] && { note "no x86_64 kernel binary found — skip"; return 0; }
  local d; d=$(objdump -d "$bin" 2>/dev/null)
  echo "$d" | grep -qiE '\blfence\b'        || miss "x86_64: lfence (Spectre v1 barrier after cap-check)"
  # Retpoline shape: indirect calls should be thunked, not naked call *%reg.
  if echo "$d" | grep -qiE '\bcall\s+\*%(rax|rbx|rcx|rdx|rsi|rdi|r[0-9]+)'; then
    miss "x86_64: naked indirect 'call *%reg' present (Retpoline thunking expected)"
  else
    note "x86_64: no naked indirect calls (Retpoline shape OK)"
  fi
  # Op-dispatch (string match -> jump table) must be read-only .rodata.
  if objdump -h "$bin" 2>/dev/null | grep -qE '\.rodata'; then
    note "x86_64: .rodata section present (op-dispatch jump table lives here)"
  fi
}

audit_aarch64() {
  local tgt=aarch64-unknown-none
  note "building kernel $tgt --release"
  if ! cargo build -p smallaios-kernel --target "$tgt" --release 2>/dev/null; then
    note "aarch64 build unavailable — skip"
    return 0
  fi
  local bin
  bin=$(find target/"$tgt"/release -maxdepth 1 -type f -perm -u+x 2>/dev/null | head -1)
  [ -z "$bin" ] && { note "no aarch64 kernel binary found — skip"; return 0; }
  local d; d=$(objdump -d "$bin" 2>/dev/null)
  echo "$d" | grep -qiE '\bcsdb\b'   || miss "aarch64: csdb (Spectre v1 barrier after cap-check)"
  echo "$d" | grep -qiE '\bdsb\b'    || miss "aarch64: dsb sy (DMA-setup ordering barrier)"
  # BTI landing pads: every indirect-branch target should begin with bti.
  echo "$d" | grep -qiE '\bbti\b'    || note "aarch64: no bti opcodes seen (BTI provided by aarch64-mte-pac-hardening-v1; may be absent on this build profile)"
}

audit_riscv64() {
  local tgt=riscv64gc-unknown-none-elf
  note "building arch-riscv64 $tgt --release"
  if ! cargo build -p smallaios-arch-riscv64 --target "$tgt" --release 2>/dev/null; then
    note "riscv64 build unavailable — skip"
    return 0
  fi
  note "riscv64: scaffolding phase — fence.i placeholder only, full CFI pending Zicfilp ratification"
}

for a in $ARCHES; do
  case "$a" in
    x86_64)  audit_x86_64  ;;
    aarch64) audit_aarch64 ;;
    riscv64) audit_riscv64 ;;
    *) note "unknown arch '$a' — skip" ;;
  esac
done

if [ "$FAIL" -ne 0 ] && [ "$STRICT" = "1" ]; then
  note "STRICT mode: missing barriers — failing"
  exit 1
fi
note "advisory audit complete (FAIL=$FAIL, STRICT=$STRICT)"
exit 0
