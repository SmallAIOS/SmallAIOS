/* SmallAIOS ARM64 Linker Script */
ENTRY(_start)

SECTIONS
{
    /* QEMU virt machine loads kernel at 0x40080000 */
    . = 0x40080000;

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

    /* Stack: 64 KiB, 16-byte aligned */
    . = ALIGN(16);
    __stack_bottom = .;
    . += 0x10000; /* 64 KiB */
    __stack_top = .;

    /DISCARD/ :
    {
        *(.comment)
        *(.eh_frame)
        *(.eh_frame_hdr)
        *(.note.*)
    }
}
