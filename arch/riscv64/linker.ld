/* SmallAIOS RISC-V 64-bit linker script for QEMU virt machine */

OUTPUT_ARCH(riscv)
ENTRY(_start)

/* QEMU virt machine loads kernel at 0x80200000 (after OpenSBI at 0x80000000) */
KERNEL_BASE = 0x80200000;

/* 128 MB physical memory on QEMU virt */
MEMORY
{
    RAM (rwx) : ORIGIN = 0x80200000, LENGTH = 128M
}

/* Stack: 64 KB per hart, 4 harts max = 256 KB */
KERNEL_STACK_SIZE = 0x10000;
MAX_HARTS = 4;

SECTIONS
{
    . = KERNEL_BASE;

    /* Text (code) section */
    .text ALIGN(4K) :
    {
        __text_start = .;
        KEEP(*(.text.entry))   /* Assembly entry point first */
        *(.text .text.*)
        __text_end = .;
    } > RAM

    /* Read-only data */
    .rodata ALIGN(4K) :
    {
        __rodata_start = .;
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
        __rodata_end = .;
    } > RAM

    /* Boot signature section (verified-boot) */
    .boot_sig ALIGN(16) :
    {
        __boot_sig_start = .;
        KEEP(*(.boot_sig))
        __boot_sig_end = .;
    } > RAM

    /* Initialized data */
    .data ALIGN(4K) :
    {
        __data_start = .;
        *(.data .data.*)
        *(.sdata .sdata.*)

        /* Global pointer for linker relaxation */
        PROVIDE(__global_pointer$ = . + 0x800);

        __data_end = .;
    } > RAM

    /* BSS (uninitialized data) — NOLOAD keeps zero-init data out of the ELF file */
    .bss (NOLOAD) : ALIGN(4K)
    {
        __bss_start = .;
        *(.bss .bss.*)
        *(.sbss .sbss.*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    . = ALIGN(4K);

    /* Per-hart kernel stacks — NOLOAD since boot code zeroes BSS anyway */
    .stacks (NOLOAD) :
    {
        __stacks_start = .;
        . += KERNEL_STACK_SIZE * MAX_HARTS;
        __stacks_end = .;
        __stack_top = .;
    } > RAM

    /* Heap starts after stacks */
    . = ALIGN(4K);
    __heap_start = .;

    /* End of kernel image */
    __kernel_end = .;

    /DISCARD/ :
    {
        *(.comment)
        *(.note*)
        *(.eh_frame*)
    }
}
