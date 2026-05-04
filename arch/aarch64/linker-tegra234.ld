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
 * The image header at .text.image_header reserves 64 bytes for the
 * ARM64 Linux boot protocol stamp (used by U-Boot's `booti`). A future
 * sub-PR may extend it to a hybrid PE/COFF + Image stamp so the same
 * artifact is bootable from both UEFI and U-Boot — see the EFI stub
 * pattern Linux uses. For now this script targets the Image-only path.
 */

ENTRY(_start)

SECTIONS
{
    . = 0x80080000;

    /* Image header must come first for ARM64 Linux boot protocol. */
    .text : ALIGN(0x1000)
    {
        __text_start = .;
        KEEP(*(.text.image_header))
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
