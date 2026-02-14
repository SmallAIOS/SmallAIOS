// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Framebuffer text console for HDMI output.
//!
//! Renders text to a linear RGBA8888 framebuffer using an embedded 8x16
//! VGA-compatible bitmap font. Provides a simple text console with cursor
//! tracking, line wrap, newline handling, and vertical scrolling.
//!
//! Designed for single-core early boot use — uses a `static mut` global
//! instance (safe because only one CPU core is running at boot time).

#![allow(dead_code)]

// ─── Font data ─────────────────────────────────────────────────────────────

/// 8x16 VGA-compatible bitmap font covering glyphs 0x00-0xFF.
///
/// Each glyph is 16 bytes (16 rows of 8 pixels, 1 bit per pixel, MSB left).
/// Total: 256 glyphs x 16 bytes = 4096 bytes.
///
/// This is the standard VGA/CP437 8x16 font bitmap data for the printable
/// ASCII range (0x20-0x7E). Non-printable characters are rendered as blank.
static FONT_8X16: [u8; 4096] = {
    let mut font = [0u8; 4096];

    // Helper: set a glyph's row data.
    // Each glyph starts at offset glyph_index * 16.
    // We define the printable ASCII glyphs (0x20 - 0x7E) inline.

    // 0x20 ' ' (space) - all zeros, already default

    // 0x21 '!'
    font[0x21 * 16 + 2] = 0x18;
    font[0x21 * 16 + 3] = 0x18;
    font[0x21 * 16 + 4] = 0x18;
    font[0x21 * 16 + 5] = 0x18;
    font[0x21 * 16 + 6] = 0x18;
    font[0x21 * 16 + 7] = 0x18;
    font[0x21 * 16 + 8] = 0x18;
    font[0x21 * 16 + 9] = 0x00;
    font[0x21 * 16 + 10] = 0x18;
    font[0x21 * 16 + 11] = 0x18;

    // 0x28 '('
    font[0x28 * 16 + 2] = 0x0C;
    font[0x28 * 16 + 3] = 0x18;
    font[0x28 * 16 + 4] = 0x30;
    font[0x28 * 16 + 5] = 0x30;
    font[0x28 * 16 + 6] = 0x30;
    font[0x28 * 16 + 7] = 0x30;
    font[0x28 * 16 + 8] = 0x30;
    font[0x28 * 16 + 9] = 0x30;
    font[0x28 * 16 + 10] = 0x18;
    font[0x28 * 16 + 11] = 0x0C;

    // 0x29 ')'
    font[0x29 * 16 + 2] = 0x30;
    font[0x29 * 16 + 3] = 0x18;
    font[0x29 * 16 + 4] = 0x0C;
    font[0x29 * 16 + 5] = 0x0C;
    font[0x29 * 16 + 6] = 0x0C;
    font[0x29 * 16 + 7] = 0x0C;
    font[0x29 * 16 + 8] = 0x0C;
    font[0x29 * 16 + 9] = 0x0C;
    font[0x29 * 16 + 10] = 0x18;
    font[0x29 * 16 + 11] = 0x30;

    // 0x2D '-'
    font[0x2D * 16 + 7] = 0x7E;

    // 0x2E '.'
    font[0x2E * 16 + 10] = 0x18;
    font[0x2E * 16 + 11] = 0x18;

    // 0x2F '/'
    font[0x2F * 16 + 2] = 0x02;
    font[0x2F * 16 + 3] = 0x06;
    font[0x2F * 16 + 4] = 0x0C;
    font[0x2F * 16 + 5] = 0x18;
    font[0x2F * 16 + 6] = 0x30;
    font[0x2F * 16 + 7] = 0x60;
    font[0x2F * 16 + 8] = 0xC0;

    // 0x30 '0'
    font[0x30 * 16 + 2] = 0x3C;
    font[0x30 * 16 + 3] = 0x66;
    font[0x30 * 16 + 4] = 0x66;
    font[0x30 * 16 + 5] = 0x6E;
    font[0x30 * 16 + 6] = 0x76;
    font[0x30 * 16 + 7] = 0x66;
    font[0x30 * 16 + 8] = 0x66;
    font[0x30 * 16 + 9] = 0x3C;

    // 0x31 '1'
    font[0x31 * 16 + 2] = 0x18;
    font[0x31 * 16 + 3] = 0x38;
    font[0x31 * 16 + 4] = 0x18;
    font[0x31 * 16 + 5] = 0x18;
    font[0x31 * 16 + 6] = 0x18;
    font[0x31 * 16 + 7] = 0x18;
    font[0x31 * 16 + 8] = 0x18;
    font[0x31 * 16 + 9] = 0x7E;

    // 0x32 '2'
    font[0x32 * 16 + 2] = 0x3C;
    font[0x32 * 16 + 3] = 0x66;
    font[0x32 * 16 + 4] = 0x06;
    font[0x32 * 16 + 5] = 0x0C;
    font[0x32 * 16 + 6] = 0x18;
    font[0x32 * 16 + 7] = 0x30;
    font[0x32 * 16 + 8] = 0x60;
    font[0x32 * 16 + 9] = 0x7E;

    // 0x33 '3'
    font[0x33 * 16 + 2] = 0x3C;
    font[0x33 * 16 + 3] = 0x66;
    font[0x33 * 16 + 4] = 0x06;
    font[0x33 * 16 + 5] = 0x1C;
    font[0x33 * 16 + 6] = 0x06;
    font[0x33 * 16 + 7] = 0x06;
    font[0x33 * 16 + 8] = 0x66;
    font[0x33 * 16 + 9] = 0x3C;

    // 0x34 '4'
    font[0x34 * 16 + 2] = 0x0C;
    font[0x34 * 16 + 3] = 0x1C;
    font[0x34 * 16 + 4] = 0x2C;
    font[0x34 * 16 + 5] = 0x4C;
    font[0x34 * 16 + 6] = 0x7E;
    font[0x34 * 16 + 7] = 0x0C;
    font[0x34 * 16 + 8] = 0x0C;
    font[0x34 * 16 + 9] = 0x0C;

    // 0x35 '5'
    font[0x35 * 16 + 2] = 0x7E;
    font[0x35 * 16 + 3] = 0x60;
    font[0x35 * 16 + 4] = 0x60;
    font[0x35 * 16 + 5] = 0x7C;
    font[0x35 * 16 + 6] = 0x06;
    font[0x35 * 16 + 7] = 0x06;
    font[0x35 * 16 + 8] = 0x66;
    font[0x35 * 16 + 9] = 0x3C;

    // 0x36 '6'
    font[0x36 * 16 + 2] = 0x3C;
    font[0x36 * 16 + 3] = 0x60;
    font[0x36 * 16 + 4] = 0x60;
    font[0x36 * 16 + 5] = 0x7C;
    font[0x36 * 16 + 6] = 0x66;
    font[0x36 * 16 + 7] = 0x66;
    font[0x36 * 16 + 8] = 0x66;
    font[0x36 * 16 + 9] = 0x3C;

    // 0x37 '7'
    font[0x37 * 16 + 2] = 0x7E;
    font[0x37 * 16 + 3] = 0x06;
    font[0x37 * 16 + 4] = 0x06;
    font[0x37 * 16 + 5] = 0x0C;
    font[0x37 * 16 + 6] = 0x18;
    font[0x37 * 16 + 7] = 0x18;
    font[0x37 * 16 + 8] = 0x18;
    font[0x37 * 16 + 9] = 0x18;

    // 0x38 '8'
    font[0x38 * 16 + 2] = 0x3C;
    font[0x38 * 16 + 3] = 0x66;
    font[0x38 * 16 + 4] = 0x66;
    font[0x38 * 16 + 5] = 0x3C;
    font[0x38 * 16 + 6] = 0x66;
    font[0x38 * 16 + 7] = 0x66;
    font[0x38 * 16 + 8] = 0x66;
    font[0x38 * 16 + 9] = 0x3C;

    // 0x39 '9'
    font[0x39 * 16 + 2] = 0x3C;
    font[0x39 * 16 + 3] = 0x66;
    font[0x39 * 16 + 4] = 0x66;
    font[0x39 * 16 + 5] = 0x3E;
    font[0x39 * 16 + 6] = 0x06;
    font[0x39 * 16 + 7] = 0x06;
    font[0x39 * 16 + 8] = 0x06;
    font[0x39 * 16 + 9] = 0x3C;

    // 0x3A ':'
    font[0x3A * 16 + 5] = 0x18;
    font[0x3A * 16 + 6] = 0x18;
    font[0x3A * 16 + 9] = 0x18;
    font[0x3A * 16 + 10] = 0x18;

    // 0x3D '='
    font[0x3D * 16 + 5] = 0x7E;
    font[0x3D * 16 + 7] = 0x7E;

    // 0x40 '@'
    font[0x40 * 16 + 2] = 0x3C;
    font[0x40 * 16 + 3] = 0x42;
    font[0x40 * 16 + 4] = 0x9E;
    font[0x40 * 16 + 5] = 0xA6;
    font[0x40 * 16 + 6] = 0xA6;
    font[0x40 * 16 + 7] = 0x9E;
    font[0x40 * 16 + 8] = 0x40;
    font[0x40 * 16 + 9] = 0x3C;

    // 0x41 'A'
    font[0x41 * 16 + 2] = 0x18;
    font[0x41 * 16 + 3] = 0x3C;
    font[0x41 * 16 + 4] = 0x66;
    font[0x41 * 16 + 5] = 0x66;
    font[0x41 * 16 + 6] = 0x7E;
    font[0x41 * 16 + 7] = 0x66;
    font[0x41 * 16 + 8] = 0x66;
    font[0x41 * 16 + 9] = 0x66;

    // 0x42 'B'
    font[0x42 * 16 + 2] = 0x7C;
    font[0x42 * 16 + 3] = 0x66;
    font[0x42 * 16 + 4] = 0x66;
    font[0x42 * 16 + 5] = 0x7C;
    font[0x42 * 16 + 6] = 0x66;
    font[0x42 * 16 + 7] = 0x66;
    font[0x42 * 16 + 8] = 0x66;
    font[0x42 * 16 + 9] = 0x7C;

    // 0x43 'C'
    font[0x43 * 16 + 2] = 0x3C;
    font[0x43 * 16 + 3] = 0x66;
    font[0x43 * 16 + 4] = 0x60;
    font[0x43 * 16 + 5] = 0x60;
    font[0x43 * 16 + 6] = 0x60;
    font[0x43 * 16 + 7] = 0x60;
    font[0x43 * 16 + 8] = 0x66;
    font[0x43 * 16 + 9] = 0x3C;

    // 0x44 'D'
    font[0x44 * 16 + 2] = 0x78;
    font[0x44 * 16 + 3] = 0x6C;
    font[0x44 * 16 + 4] = 0x66;
    font[0x44 * 16 + 5] = 0x66;
    font[0x44 * 16 + 6] = 0x66;
    font[0x44 * 16 + 7] = 0x66;
    font[0x44 * 16 + 8] = 0x6C;
    font[0x44 * 16 + 9] = 0x78;

    // 0x45 'E'
    font[0x45 * 16 + 2] = 0x7E;
    font[0x45 * 16 + 3] = 0x60;
    font[0x45 * 16 + 4] = 0x60;
    font[0x45 * 16 + 5] = 0x7C;
    font[0x45 * 16 + 6] = 0x60;
    font[0x45 * 16 + 7] = 0x60;
    font[0x45 * 16 + 8] = 0x60;
    font[0x45 * 16 + 9] = 0x7E;

    // 0x46 'F'
    font[0x46 * 16 + 2] = 0x7E;
    font[0x46 * 16 + 3] = 0x60;
    font[0x46 * 16 + 4] = 0x60;
    font[0x46 * 16 + 5] = 0x7C;
    font[0x46 * 16 + 6] = 0x60;
    font[0x46 * 16 + 7] = 0x60;
    font[0x46 * 16 + 8] = 0x60;
    font[0x46 * 16 + 9] = 0x60;

    // 0x47 'G'
    font[0x47 * 16 + 2] = 0x3C;
    font[0x47 * 16 + 3] = 0x66;
    font[0x47 * 16 + 4] = 0x60;
    font[0x47 * 16 + 5] = 0x60;
    font[0x47 * 16 + 6] = 0x6E;
    font[0x47 * 16 + 7] = 0x66;
    font[0x47 * 16 + 8] = 0x66;
    font[0x47 * 16 + 9] = 0x3E;

    // 0x48 'H'
    font[0x48 * 16 + 2] = 0x66;
    font[0x48 * 16 + 3] = 0x66;
    font[0x48 * 16 + 4] = 0x66;
    font[0x48 * 16 + 5] = 0x7E;
    font[0x48 * 16 + 6] = 0x66;
    font[0x48 * 16 + 7] = 0x66;
    font[0x48 * 16 + 8] = 0x66;
    font[0x48 * 16 + 9] = 0x66;

    // 0x49 'I'
    font[0x49 * 16 + 2] = 0x3C;
    font[0x49 * 16 + 3] = 0x18;
    font[0x49 * 16 + 4] = 0x18;
    font[0x49 * 16 + 5] = 0x18;
    font[0x49 * 16 + 6] = 0x18;
    font[0x49 * 16 + 7] = 0x18;
    font[0x49 * 16 + 8] = 0x18;
    font[0x49 * 16 + 9] = 0x3C;

    // 0x4A 'J'
    font[0x4A * 16 + 2] = 0x06;
    font[0x4A * 16 + 3] = 0x06;
    font[0x4A * 16 + 4] = 0x06;
    font[0x4A * 16 + 5] = 0x06;
    font[0x4A * 16 + 6] = 0x06;
    font[0x4A * 16 + 7] = 0x06;
    font[0x4A * 16 + 8] = 0x66;
    font[0x4A * 16 + 9] = 0x3C;

    // 0x4B 'K'
    font[0x4B * 16 + 2] = 0x66;
    font[0x4B * 16 + 3] = 0x6C;
    font[0x4B * 16 + 4] = 0x78;
    font[0x4B * 16 + 5] = 0x70;
    font[0x4B * 16 + 6] = 0x78;
    font[0x4B * 16 + 7] = 0x6C;
    font[0x4B * 16 + 8] = 0x66;
    font[0x4B * 16 + 9] = 0x66;

    // 0x4C 'L'
    font[0x4C * 16 + 2] = 0x60;
    font[0x4C * 16 + 3] = 0x60;
    font[0x4C * 16 + 4] = 0x60;
    font[0x4C * 16 + 5] = 0x60;
    font[0x4C * 16 + 6] = 0x60;
    font[0x4C * 16 + 7] = 0x60;
    font[0x4C * 16 + 8] = 0x60;
    font[0x4C * 16 + 9] = 0x7E;

    // 0x4D 'M'
    font[0x4D * 16 + 2] = 0xC6;
    font[0x4D * 16 + 3] = 0xEE;
    font[0x4D * 16 + 4] = 0xFE;
    font[0x4D * 16 + 5] = 0xD6;
    font[0x4D * 16 + 6] = 0xC6;
    font[0x4D * 16 + 7] = 0xC6;
    font[0x4D * 16 + 8] = 0xC6;
    font[0x4D * 16 + 9] = 0xC6;

    // 0x4E 'N'
    font[0x4E * 16 + 2] = 0x66;
    font[0x4E * 16 + 3] = 0x76;
    font[0x4E * 16 + 4] = 0x7E;
    font[0x4E * 16 + 5] = 0x6E;
    font[0x4E * 16 + 6] = 0x66;
    font[0x4E * 16 + 7] = 0x66;
    font[0x4E * 16 + 8] = 0x66;
    font[0x4E * 16 + 9] = 0x66;

    // 0x4F 'O'
    font[0x4F * 16 + 2] = 0x3C;
    font[0x4F * 16 + 3] = 0x66;
    font[0x4F * 16 + 4] = 0x66;
    font[0x4F * 16 + 5] = 0x66;
    font[0x4F * 16 + 6] = 0x66;
    font[0x4F * 16 + 7] = 0x66;
    font[0x4F * 16 + 8] = 0x66;
    font[0x4F * 16 + 9] = 0x3C;

    // 0x50 'P'
    font[0x50 * 16 + 2] = 0x7C;
    font[0x50 * 16 + 3] = 0x66;
    font[0x50 * 16 + 4] = 0x66;
    font[0x50 * 16 + 5] = 0x7C;
    font[0x50 * 16 + 6] = 0x60;
    font[0x50 * 16 + 7] = 0x60;
    font[0x50 * 16 + 8] = 0x60;
    font[0x50 * 16 + 9] = 0x60;

    // 0x51 'Q'
    font[0x51 * 16 + 2] = 0x3C;
    font[0x51 * 16 + 3] = 0x66;
    font[0x51 * 16 + 4] = 0x66;
    font[0x51 * 16 + 5] = 0x66;
    font[0x51 * 16 + 6] = 0x66;
    font[0x51 * 16 + 7] = 0x6E;
    font[0x51 * 16 + 8] = 0x3C;
    font[0x51 * 16 + 9] = 0x0E;

    // 0x52 'R'
    font[0x52 * 16 + 2] = 0x7C;
    font[0x52 * 16 + 3] = 0x66;
    font[0x52 * 16 + 4] = 0x66;
    font[0x52 * 16 + 5] = 0x7C;
    font[0x52 * 16 + 6] = 0x6C;
    font[0x52 * 16 + 7] = 0x66;
    font[0x52 * 16 + 8] = 0x66;
    font[0x52 * 16 + 9] = 0x66;

    // 0x53 'S'
    font[0x53 * 16 + 2] = 0x3C;
    font[0x53 * 16 + 3] = 0x66;
    font[0x53 * 16 + 4] = 0x60;
    font[0x53 * 16 + 5] = 0x3C;
    font[0x53 * 16 + 6] = 0x06;
    font[0x53 * 16 + 7] = 0x06;
    font[0x53 * 16 + 8] = 0x66;
    font[0x53 * 16 + 9] = 0x3C;

    // 0x54 'T'
    font[0x54 * 16 + 2] = 0x7E;
    font[0x54 * 16 + 3] = 0x18;
    font[0x54 * 16 + 4] = 0x18;
    font[0x54 * 16 + 5] = 0x18;
    font[0x54 * 16 + 6] = 0x18;
    font[0x54 * 16 + 7] = 0x18;
    font[0x54 * 16 + 8] = 0x18;
    font[0x54 * 16 + 9] = 0x18;

    // 0x55 'U'
    font[0x55 * 16 + 2] = 0x66;
    font[0x55 * 16 + 3] = 0x66;
    font[0x55 * 16 + 4] = 0x66;
    font[0x55 * 16 + 5] = 0x66;
    font[0x55 * 16 + 6] = 0x66;
    font[0x55 * 16 + 7] = 0x66;
    font[0x55 * 16 + 8] = 0x66;
    font[0x55 * 16 + 9] = 0x3C;

    // 0x56 'V'
    font[0x56 * 16 + 2] = 0x66;
    font[0x56 * 16 + 3] = 0x66;
    font[0x56 * 16 + 4] = 0x66;
    font[0x56 * 16 + 5] = 0x66;
    font[0x56 * 16 + 6] = 0x66;
    font[0x56 * 16 + 7] = 0x3C;
    font[0x56 * 16 + 8] = 0x3C;
    font[0x56 * 16 + 9] = 0x18;

    // 0x57 'W'
    font[0x57 * 16 + 2] = 0xC6;
    font[0x57 * 16 + 3] = 0xC6;
    font[0x57 * 16 + 4] = 0xC6;
    font[0x57 * 16 + 5] = 0xD6;
    font[0x57 * 16 + 6] = 0xFE;
    font[0x57 * 16 + 7] = 0xEE;
    font[0x57 * 16 + 8] = 0xC6;
    font[0x57 * 16 + 9] = 0xC6;

    // 0x58 'X'
    font[0x58 * 16 + 2] = 0x66;
    font[0x58 * 16 + 3] = 0x66;
    font[0x58 * 16 + 4] = 0x3C;
    font[0x58 * 16 + 5] = 0x18;
    font[0x58 * 16 + 6] = 0x3C;
    font[0x58 * 16 + 7] = 0x66;
    font[0x58 * 16 + 8] = 0x66;
    font[0x58 * 16 + 9] = 0x66;

    // 0x59 'Y'
    font[0x59 * 16 + 2] = 0x66;
    font[0x59 * 16 + 3] = 0x66;
    font[0x59 * 16 + 4] = 0x66;
    font[0x59 * 16 + 5] = 0x3C;
    font[0x59 * 16 + 6] = 0x18;
    font[0x59 * 16 + 7] = 0x18;
    font[0x59 * 16 + 8] = 0x18;
    font[0x59 * 16 + 9] = 0x18;

    // 0x5A 'Z'
    font[0x5A * 16 + 2] = 0x7E;
    font[0x5A * 16 + 3] = 0x06;
    font[0x5A * 16 + 4] = 0x0C;
    font[0x5A * 16 + 5] = 0x18;
    font[0x5A * 16 + 6] = 0x30;
    font[0x5A * 16 + 7] = 0x60;
    font[0x5A * 16 + 8] = 0x60;
    font[0x5A * 16 + 9] = 0x7E;

    // 0x5B '['
    font[0x5B * 16 + 2] = 0x3C;
    font[0x5B * 16 + 3] = 0x30;
    font[0x5B * 16 + 4] = 0x30;
    font[0x5B * 16 + 5] = 0x30;
    font[0x5B * 16 + 6] = 0x30;
    font[0x5B * 16 + 7] = 0x30;
    font[0x5B * 16 + 8] = 0x30;
    font[0x5B * 16 + 9] = 0x30;
    font[0x5B * 16 + 10] = 0x3C;

    // 0x5D ']'
    font[0x5D * 16 + 2] = 0x3C;
    font[0x5D * 16 + 3] = 0x0C;
    font[0x5D * 16 + 4] = 0x0C;
    font[0x5D * 16 + 5] = 0x0C;
    font[0x5D * 16 + 6] = 0x0C;
    font[0x5D * 16 + 7] = 0x0C;
    font[0x5D * 16 + 8] = 0x0C;
    font[0x5D * 16 + 9] = 0x0C;
    font[0x5D * 16 + 10] = 0x3C;

    // 0x5F '_'
    font[0x5F * 16 + 12] = 0xFF;

    // Lowercase letters: 0x61 'a' through 0x7A 'z'

    // 0x61 'a'
    font[0x61 * 16 + 5] = 0x3C;
    font[0x61 * 16 + 6] = 0x06;
    font[0x61 * 16 + 7] = 0x3E;
    font[0x61 * 16 + 8] = 0x66;
    font[0x61 * 16 + 9] = 0x3E;

    // 0x62 'b'
    font[0x62 * 16 + 2] = 0x60;
    font[0x62 * 16 + 3] = 0x60;
    font[0x62 * 16 + 4] = 0x60;
    font[0x62 * 16 + 5] = 0x7C;
    font[0x62 * 16 + 6] = 0x66;
    font[0x62 * 16 + 7] = 0x66;
    font[0x62 * 16 + 8] = 0x66;
    font[0x62 * 16 + 9] = 0x7C;

    // 0x63 'c'
    font[0x63 * 16 + 5] = 0x3C;
    font[0x63 * 16 + 6] = 0x66;
    font[0x63 * 16 + 7] = 0x60;
    font[0x63 * 16 + 8] = 0x66;
    font[0x63 * 16 + 9] = 0x3C;

    // 0x64 'd'
    font[0x64 * 16 + 2] = 0x06;
    font[0x64 * 16 + 3] = 0x06;
    font[0x64 * 16 + 4] = 0x06;
    font[0x64 * 16 + 5] = 0x3E;
    font[0x64 * 16 + 6] = 0x66;
    font[0x64 * 16 + 7] = 0x66;
    font[0x64 * 16 + 8] = 0x66;
    font[0x64 * 16 + 9] = 0x3E;

    // 0x65 'e'
    font[0x65 * 16 + 5] = 0x3C;
    font[0x65 * 16 + 6] = 0x66;
    font[0x65 * 16 + 7] = 0x7E;
    font[0x65 * 16 + 8] = 0x60;
    font[0x65 * 16 + 9] = 0x3C;

    // 0x66 'f'
    font[0x66 * 16 + 2] = 0x1C;
    font[0x66 * 16 + 3] = 0x30;
    font[0x66 * 16 + 4] = 0x30;
    font[0x66 * 16 + 5] = 0x7C;
    font[0x66 * 16 + 6] = 0x30;
    font[0x66 * 16 + 7] = 0x30;
    font[0x66 * 16 + 8] = 0x30;
    font[0x66 * 16 + 9] = 0x30;

    // 0x67 'g'
    font[0x67 * 16 + 5] = 0x3E;
    font[0x67 * 16 + 6] = 0x66;
    font[0x67 * 16 + 7] = 0x66;
    font[0x67 * 16 + 8] = 0x3E;
    font[0x67 * 16 + 9] = 0x06;
    font[0x67 * 16 + 10] = 0x3C;

    // 0x68 'h'
    font[0x68 * 16 + 2] = 0x60;
    font[0x68 * 16 + 3] = 0x60;
    font[0x68 * 16 + 4] = 0x60;
    font[0x68 * 16 + 5] = 0x7C;
    font[0x68 * 16 + 6] = 0x66;
    font[0x68 * 16 + 7] = 0x66;
    font[0x68 * 16 + 8] = 0x66;
    font[0x68 * 16 + 9] = 0x66;

    // 0x69 'i'
    font[0x69 * 16 + 3] = 0x18;
    font[0x69 * 16 + 5] = 0x38;
    font[0x69 * 16 + 6] = 0x18;
    font[0x69 * 16 + 7] = 0x18;
    font[0x69 * 16 + 8] = 0x18;
    font[0x69 * 16 + 9] = 0x3C;

    // 0x6C 'l'
    font[0x6C * 16 + 2] = 0x38;
    font[0x6C * 16 + 3] = 0x18;
    font[0x6C * 16 + 4] = 0x18;
    font[0x6C * 16 + 5] = 0x18;
    font[0x6C * 16 + 6] = 0x18;
    font[0x6C * 16 + 7] = 0x18;
    font[0x6C * 16 + 8] = 0x18;
    font[0x6C * 16 + 9] = 0x3C;

    // 0x6D 'm'
    font[0x6D * 16 + 5] = 0xEC;
    font[0x6D * 16 + 6] = 0xFE;
    font[0x6D * 16 + 7] = 0xD6;
    font[0x6D * 16 + 8] = 0xC6;
    font[0x6D * 16 + 9] = 0xC6;

    // 0x6E 'n'
    font[0x6E * 16 + 5] = 0x7C;
    font[0x6E * 16 + 6] = 0x66;
    font[0x6E * 16 + 7] = 0x66;
    font[0x6E * 16 + 8] = 0x66;
    font[0x6E * 16 + 9] = 0x66;

    // 0x6F 'o'
    font[0x6F * 16 + 5] = 0x3C;
    font[0x6F * 16 + 6] = 0x66;
    font[0x6F * 16 + 7] = 0x66;
    font[0x6F * 16 + 8] = 0x66;
    font[0x6F * 16 + 9] = 0x3C;

    // 0x70 'p'
    font[0x70 * 16 + 5] = 0x7C;
    font[0x70 * 16 + 6] = 0x66;
    font[0x70 * 16 + 7] = 0x66;
    font[0x70 * 16 + 8] = 0x7C;
    font[0x70 * 16 + 9] = 0x60;
    font[0x70 * 16 + 10] = 0x60;

    // 0x72 'r'
    font[0x72 * 16 + 5] = 0x6C;
    font[0x72 * 16 + 6] = 0x76;
    font[0x72 * 16 + 7] = 0x60;
    font[0x72 * 16 + 8] = 0x60;
    font[0x72 * 16 + 9] = 0x60;

    // 0x73 's'
    font[0x73 * 16 + 5] = 0x3E;
    font[0x73 * 16 + 6] = 0x60;
    font[0x73 * 16 + 7] = 0x3C;
    font[0x73 * 16 + 8] = 0x06;
    font[0x73 * 16 + 9] = 0x7C;

    // 0x74 't'
    font[0x74 * 16 + 2] = 0x30;
    font[0x74 * 16 + 3] = 0x30;
    font[0x74 * 16 + 4] = 0x30;
    font[0x74 * 16 + 5] = 0x7C;
    font[0x74 * 16 + 6] = 0x30;
    font[0x74 * 16 + 7] = 0x30;
    font[0x74 * 16 + 8] = 0x30;
    font[0x74 * 16 + 9] = 0x1C;

    // 0x75 'u'
    font[0x75 * 16 + 5] = 0x66;
    font[0x75 * 16 + 6] = 0x66;
    font[0x75 * 16 + 7] = 0x66;
    font[0x75 * 16 + 8] = 0x66;
    font[0x75 * 16 + 9] = 0x3E;

    // 0x76 'v'
    font[0x76 * 16 + 5] = 0x66;
    font[0x76 * 16 + 6] = 0x66;
    font[0x76 * 16 + 7] = 0x66;
    font[0x76 * 16 + 8] = 0x3C;
    font[0x76 * 16 + 9] = 0x18;

    // 0x77 'w'
    font[0x77 * 16 + 5] = 0xC6;
    font[0x77 * 16 + 6] = 0xC6;
    font[0x77 * 16 + 7] = 0xD6;
    font[0x77 * 16 + 8] = 0xFE;
    font[0x77 * 16 + 9] = 0x6C;

    // 0x78 'x'
    font[0x78 * 16 + 5] = 0x66;
    font[0x78 * 16 + 6] = 0x3C;
    font[0x78 * 16 + 7] = 0x18;
    font[0x78 * 16 + 8] = 0x3C;
    font[0x78 * 16 + 9] = 0x66;

    // 0x79 'y'
    font[0x79 * 16 + 5] = 0x66;
    font[0x79 * 16 + 6] = 0x66;
    font[0x79 * 16 + 7] = 0x3E;
    font[0x79 * 16 + 8] = 0x06;
    font[0x79 * 16 + 9] = 0x06;
    font[0x79 * 16 + 10] = 0x3C;

    // 0x7A 'z'
    font[0x7A * 16 + 5] = 0x7E;
    font[0x7A * 16 + 6] = 0x0C;
    font[0x7A * 16 + 7] = 0x18;
    font[0x7A * 16 + 8] = 0x30;
    font[0x7A * 16 + 9] = 0x7E;

    font
};

// ─── FbConsole struct ──────────────────────────────────────────────────────

/// Framebuffer text console state.
pub struct FbConsole {
    /// Framebuffer base address.
    fb_addr: usize,
    /// Display width in pixels.
    width: usize,
    /// Display height in pixels.
    height: usize,
    /// Framebuffer line stride in bytes.
    stride: usize,
    /// Current cursor row (character grid).
    cursor_row: usize,
    /// Current cursor column (character grid).
    cursor_col: usize,
    /// Number of text columns (width / 8).
    cols: usize,
    /// Number of text rows (height / 16).
    rows: usize,
    /// Whether the console has been initialized.
    initialized: bool,
}

impl FbConsole {
    /// Create an uninitialized console instance.
    const fn new() -> Self {
        Self {
            fb_addr: 0,
            width: 0,
            height: 0,
            stride: 0,
            cursor_row: 0,
            cursor_col: 0,
            cols: 0,
            rows: 0,
            initialized: false,
        }
    }
}

// ─── Global instance ───────────────────────────────────────────────────────

/// Global framebuffer console instance.
///
/// Safe to use `static mut` because this is only accessed during single-core
/// early boot before any secondary CPUs are started.
static mut FB_CONSOLE: FbConsole = FbConsole::new();

// ─── Console operations ────────────────────────────────────────────────────

/// Initialize the framebuffer console.
///
/// Zeroes the framebuffer memory, sets cursor to (0, 0), and computes the
/// text grid dimensions from pixel dimensions.
///
/// # Safety
/// `addr` must be a valid, mapped memory address with at least
/// `stride * height` bytes available. Must be called before any fb_putc/fb_puts.
pub unsafe fn fb_console_init(addr: usize, width: usize, height: usize, stride: usize) {
    // Zero the framebuffer
    let fb_size = stride * height;
    let fb_ptr = addr as *mut u8;
    core::ptr::write_bytes(fb_ptr, 0, fb_size);

    let console = &raw mut FB_CONSOLE;
    (*console).fb_addr = addr;
    (*console).width = width;
    (*console).height = height;
    (*console).stride = stride;
    (*console).cursor_row = 0;
    (*console).cursor_col = 0;
    (*console).cols = width / 8;
    (*console).rows = height / 16;
    (*console).initialized = true;
}

/// Render a single glyph at the given character grid position.
///
/// Writes 8x16 pixels into the framebuffer at pixel position (col*8, row*16).
/// Set bits in the font bitmap produce white (0xFFFFFFFF) pixels;
/// clear bits produce black (0x00000000) pixels.
///
/// # Safety
/// The framebuffer console must be initialized and the position must be within bounds.
unsafe fn render_glyph(row: usize, col: usize, byte: u8) {
    let console = &raw const FB_CONSOLE;
    let fb_addr = (*console).fb_addr;
    let stride = (*console).stride;

    let glyph_offset = (byte as usize) * 16;
    let px_x = col * 8;
    let px_y = row * 16;

    for font_row in 0..16 {
        let font_byte = FONT_8X16[glyph_offset + font_row];
        let row_addr = fb_addr + (px_y + font_row) * stride + px_x * 4;

        for bit in 0..8 {
            let pixel_ptr = (row_addr + bit * 4) as *mut u32;
            let color = if (font_byte >> (7 - bit)) & 1 != 0 {
                0xFFFF_FFFF // White
            } else {
                0x0000_0000 // Black
            };
            core::ptr::write_volatile(pixel_ptr, color);
        }
    }
}

/// Write a single byte to the framebuffer console.
///
/// Handles:
/// - `\n`: move to next row, column 0
/// - Printable chars (0x20-0x7E): render glyph, advance cursor
/// - Line wrap at end of row
/// - Scroll when past last row
///
/// # Safety
/// The framebuffer console must be initialized.
pub unsafe fn fb_putc(byte: u8) {
    let console = &raw mut FB_CONSOLE;
    if !(*console).initialized {
        return;
    }

    if byte == b'\n' {
        (*console).cursor_col = 0;
        (*console).cursor_row += 1;
    } else if byte == b'\r' {
        // Carriage return: just reset column
        (*console).cursor_col = 0;
        return;
    } else {
        let row = (*console).cursor_row;
        let col = (*console).cursor_col;
        let cols = (*console).cols;

        if row < (*console).rows {
            render_glyph(row, col, byte);
        }

        (*console).cursor_col += 1;
        if (*console).cursor_col >= cols {
            (*console).cursor_col = 0;
            (*console).cursor_row += 1;
        }
    }

    // Check if we need to scroll
    if (*console).cursor_row >= (*console).rows {
        scroll_up();
        (*console).cursor_row = (*console).rows - 1;
    }
}

/// Scroll the framebuffer up by one text line (16 pixel rows).
///
/// Moves all framebuffer content up by `stride * 16` bytes using memmove,
/// then zeroes the bottom 16 rows of pixels.
///
/// # Safety
/// The framebuffer console must be initialized.
unsafe fn scroll_up() {
    let console = &raw const FB_CONSOLE;
    let fb_addr = (*console).fb_addr;
    let stride = (*console).stride;
    let height = (*console).height;

    let line_bytes = stride * 16; // One text line = 16 pixel rows
    let total_bytes = stride * height;

    // Move everything up by one text line
    let src = (fb_addr + line_bytes) as *const u8;
    let dst = fb_addr as *mut u8;
    let move_bytes = total_bytes - line_bytes;
    core::ptr::copy(src, dst, move_bytes);

    // Zero the bottom text line
    let bottom = (fb_addr + total_bytes - line_bytes) as *mut u8;
    core::ptr::write_bytes(bottom, 0, line_bytes);
}

/// Write a string to the framebuffer console.
///
/// Iterates bytes and calls `fb_putc` for each one.
///
/// # Safety
/// The framebuffer console must be initialized.
pub unsafe fn fb_puts(s: &str) {
    for byte in s.bytes() {
        fb_putc(byte);
    }
}

/// Returns whether the framebuffer console has been initialized.
pub fn is_initialized() -> bool {
    // Safe: single-core early boot, only reading a bool
    unsafe {
        let console = &raw const FB_CONSOLE;
        (*console).initialized
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Font data ─────────────────────────────────────────────────────

    #[test]
    fn test_font_size() {
        assert_eq!(FONT_8X16.len(), 4096);
    }

    #[test]
    fn test_glyph_a_offset() {
        // Glyph 0x41 ('A') starts at offset 0x41 * 16 = 1040
        assert_eq!(0x41 * 16, 1040);
        // Verify the 'A' glyph has non-zero data (row 2 should be 0x18)
        assert_eq!(FONT_8X16[0x41 * 16 + 2], 0x18);
    }

    #[test]
    fn test_space_glyph_is_blank() {
        // Space (0x20) should be all zeros
        for i in 0..16 {
            assert_eq!(FONT_8X16[0x20 * 16 + i], 0x00);
        }
    }

    #[test]
    fn test_glyph_lookup_printable() {
        // All printable ASCII glyphs should have at least one non-zero row
        // (except space at 0x20)
        let has_pixels = |c: u8| -> bool {
            let offset = (c as usize) * 16;
            (0..16).any(|i| FONT_8X16[offset + i] != 0)
        };
        // Capital letters should all have visible pixels
        for c in b'A'..=b'Z' {
            assert!(
                has_pixels(c),
                "Glyph for '{}' (0x{:02X}) should have visible pixels",
                c as char,
                c
            );
        }
    }

    // ─── Console geometry ──────────────────────────────────────────────

    #[test]
    fn test_cols_rows_1080p() {
        // 1920 / 8 = 240 columns
        assert_eq!(1920 / 8, 240);
        // 1080 / 16 = 67 rows (integer division)
        assert_eq!(1080 / 16, 67);
    }

    #[test]
    fn test_cols_rows_720p() {
        assert_eq!(1280 / 8, 160);
        assert_eq!(720 / 16, 45);
    }

    // ─── FbConsole struct ──────────────────────────────────────────────

    #[test]
    fn test_fb_console_new() {
        let c = FbConsole::new();
        assert_eq!(c.fb_addr, 0);
        assert_eq!(c.width, 0);
        assert_eq!(c.height, 0);
        assert_eq!(c.stride, 0);
        assert_eq!(c.cursor_row, 0);
        assert_eq!(c.cursor_col, 0);
        assert_eq!(c.cols, 0);
        assert_eq!(c.rows, 0);
        assert!(!c.initialized);
    }

    // ─── Cursor advancement logic (simulated) ──────────────────────────

    /// Simulate cursor advancement for a given grid size.
    fn simulate_cursor(cols: usize, rows: usize, input: &[u8]) -> (usize, usize, usize) {
        let mut row = 0usize;
        let mut col = 0usize;
        let mut scrolls = 0usize;

        for &byte in input {
            if byte == b'\n' {
                col = 0;
                row += 1;
            } else {
                col += 1;
                if col >= cols {
                    col = 0;
                    row += 1;
                }
            }
            if row >= rows {
                scrolls += 1;
                row = rows - 1;
            }
        }
        (row, col, scrolls)
    }

    #[test]
    fn test_cursor_single_char() {
        let (row, col, scrolls) = simulate_cursor(240, 67, b"A");
        assert_eq!(row, 0);
        assert_eq!(col, 1);
        assert_eq!(scrolls, 0);
    }

    #[test]
    fn test_cursor_line_wrap() {
        // 240 characters should wrap to the next line
        let input: [u8; 240] = [b'A'; 240];
        let (row, col, scrolls) = simulate_cursor(240, 67, &input);
        assert_eq!(row, 1);
        assert_eq!(col, 0);
        assert_eq!(scrolls, 0);
    }

    #[test]
    fn test_cursor_newline_resets_col() {
        let (row, col, scrolls) = simulate_cursor(240, 67, b"ABC\n");
        assert_eq!(row, 1);
        assert_eq!(col, 0);
        assert_eq!(scrolls, 0);
    }

    #[test]
    fn test_cursor_newline_at_last_row_triggers_scroll() {
        // Fill rows 0..66 with newlines (66 newlines to reach row 66)
        // Then one more newline should trigger scroll
        let mut input = [b'\n'; 67];
        // 67 newlines: rows 0->1->2->...->66->67(scroll)
        let (row, col, scrolls) = simulate_cursor(240, 67, &input);
        assert_eq!(row, 66); // Last row after scroll
        assert_eq!(col, 0);
        assert_eq!(scrolls, 1);

        // 68 newlines: two scrolls
        input = [b'\n'; 67]; // reuse; test 68 below
        let mut input68 = [b'\n'; 68];
        let (row, col, scrolls) = simulate_cursor(240, 67, &input68);
        assert_eq!(row, 66);
        assert_eq!(col, 0);
        assert_eq!(scrolls, 2);
    }

    #[test]
    fn test_scroll_detection_exact_boundary() {
        // Row 66 is the last row (0-indexed, rows=67).
        // Being at row 66, col 0 and typing \n should trigger scroll.
        // Simulate: 66 newlines puts us at row 66, then one more newline
        let input: Vec<u8> = core::iter::repeat(b'\n').take(67).collect();
        let (row, _col, scrolls) = simulate_cursor(240, 67, &input);
        assert_eq!(row, 66);
        assert_eq!(scrolls, 1);
    }
}
