/* SmallAIOS ARM64 Linker Script — Tegra234 (Jetson Orin family) */
/*
 * Used for the `aarch64-unknown-none --features tegra234` build path
 * (e.g. when chain-loaded by U-Boot's `booti` from L4T's extlinux, or
 * dropped onto an SD card via NVIDIA's standard kernel-Image flow).
 *
 * The primary Phase-2 boot path is `aarch64-unknown-uefi` (PE/COFF .efi
 * loaded by the Orin's UEFI firmware) and uses rust's built-in UEFI
 * target — no custom linker script needed. This script exists for the
 * non-UEFI bare-metal symmetry case and to provide stable __text_start /
 * __bss_start / __stack_top symbols the boot assembly references.
 *
 * Layout follows the same pattern as `linker-tegra.ld` (Tegra X1) but
 * drops the framebuffer reservation — Orin's display pipeline is owned
 * by NVIDIA's UEFI/Bootflow stack and not exercised by the unikernel.
 *
 * Memory map:
 *   DRAM_BASE          = 0x80000000 (Tegra234 standard)
 *   KERNEL_LOAD_ADDR   = 0x80080000 (DRAM_BASE + 512 KiB, Image protocol)
 *
 * NOTE: this script does not currently reserve the ARM64 Linux Image
 * header at `.text.image_header` — the existing `image_header` module is
 * `#[cfg(feature = "tegra-x1")]`-only, so a `tegra234` build emits no
 * such section. If we ever want a hybrid PE/COFF + Image-stamped artifact
 * bootable from both UEFI and U-Boot (à la Linux's EFI stub pattern), a
 * follow-up sub-PR should: (1) generalize the `image_header` module to
 * cover `tegra234` as well, (2) re-add `KEEP(*(.text.image_header))`
 * here. Until then we explicitly do not, to avoid producing dead-section
 * artifacts and to keep the linker output predictable.
 */

ENTRY(_start)

SECTIONS
{
    . = 0x80080000;

    .text : ALIGN(0x1000)
    {
        __text_start = .;
        KEEP(*(.text.boot))
        *(.text .text.*)
        __text_end = .;
    }

    .rodata : ALIGN(0x1000)
    {
        __rodata_start = .;
        *(.rodata .rodata.*)
        __rodata_end = .;
    }

    .boot_sig : ALIGN(0x10)
    {
        __boot_sig_start = .;
        KEEP(*(.boot_sig))
        __boot_sig_end = .;
    }

    .data : ALIGN(0x1000)
    {
        *(.data .data.*)
    }

    .bss : ALIGN(0x1000)
    {
        __bss_start = .;
        *(.bss .bss.*)
        *(COMMON)
        __bss_end = .;
    }

    /* Stack: 64 KiB, 16-byte aligned. Matches linker-tegra.ld. */
    . = ALIGN(16);
    __stack_bottom = .;
    . += 0x10000;
    __stack_top = .;

    /DISCARD/ :
    {
        *(.comment)
        *(.eh_frame)
        *(.eh_frame_hdr)
        *(.note.*)
    }
}
