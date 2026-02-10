/* SmallAIOS x86-64 Linker Script */
ENTRY(_start)

SECTIONS
{
    . = 0x100000; /* 1 MiB */

    .text : ALIGN(0x1000)
    {
        *(.multiboot2)
        KEEP(*(.text.boot))
        *(.text .text.*)
    }

    .rodata : ALIGN(0x1000)
    {
        *(.rodata .rodata.*)
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
