/* SmallAIOS ARM64 Linker Script — Tegra X1 (Jetson Nano) */
/* U-Boot loads kernel Image at 0x80080000 (DRAM base + 512 KiB) */
ENTRY(_start)

SECTIONS
{
    . = 0x80080000;

    /* Image header must come first for U-Boot booti compatibility */
    .text : ALIGN(0x1000)
    {
        KEEP(*(.text.image_header))
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

    /* Framebuffer: 8 MiB at fixed DRAM offset 0x8F000000 */
    /* Reserved for HDMI display output (1920x1080 RGBA8888 + headroom) */
    . = 0x8F000000;
    .framebuffer (NOLOAD) : ALIGN(0x1000)
    {
        __fb_start = .;
        . += 0x800000; /* 8 MiB */
        __fb_end = .;
    }

    /DISCARD/ :
    {
        *(.comment)
        *(.eh_frame)
        *(.eh_frame_hdr)
        *(.note.*)
    }
}
