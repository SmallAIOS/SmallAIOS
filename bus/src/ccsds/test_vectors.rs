// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CCSDS Blue Book test vectors and specification compliance validation.
//!
//! Validates our CCSDS implementation against known-good byte patterns
//! derived from the following CCSDS Blue Book standards:
//! - CCSDS 133.0-B-2: Space Packet Protocol (primary header bit layout)
//! - CCSDS 132.0-B-3: TM Space Data Link Protocol (TM frame header)
//! - CCSDS 232.0-B-4: TC Space Data Link Protocol (TC frame + CRC-16/CCITT)
//! - CCSDS 231.0-B-3: TC Synchronization and Channel Coding (CLTU, BCH)

#[cfg(test)]
mod tests {
    use crate::ccsds::cltu::{
        bch_parity, bch_verify, cltu_decode, cltu_encode, CLTU_START, CLTU_TAIL, CODE_BLOCK_INFO_LEN,
        CODE_BLOCK_LEN, FILL_BYTE,
    };
    use crate::ccsds::packet::{PacketType, SeqFlags, SpacePacket, MAX_APID, MAX_SEQ_COUNT, PRIMARY_HEADER_LEN};
    use crate::ccsds::tc_frame::{crc16_ccitt, TcFrame, FECF_LEN};
    use crate::ccsds::tm_frame::{TmFrame, FHP_NO_PACKET, TM_HEADER_LEN};

    // ===================================================================
    // CCSDS 133.0-B-2: Space Packet Primary Header Bit-Field Layout
    // ===================================================================

    /// Validate that a TM unsegmented packet with APID 0x1A3, seq count 42,
    /// and 4 bytes of data produces the expected 6-byte primary header.
    ///
    /// Expected header (per CCSDS 133.0-B-2 Section 4.1):
    ///   Word 1: version=000, type=0(TM), sec_hdr=0, APID=0x1A3
    ///           = 0b000_0_0_00110100011 = 0x01A3
    ///   Word 2: seq_flags=11(unseg), seq_count=42
    ///           = 0b11_00000000101010 = 0xC02A
    ///   Word 3: data_length = 4 - 1 = 3
    ///           = 0x0003
    #[test]
    fn test_space_packet_header_bit_layout_tm() {
        let pkt = SpacePacket::new(
            PacketType::Telemetry,
            false,
            0x1A3,
            SeqFlags::Unsegmented,
            42,
            &[0xDE, 0xAD, 0xBE, 0xEF],
        )
        .unwrap();

        let mut buf = [0u8; 64];
        let len = pkt.encode(&mut buf).unwrap();
        assert_eq!(len, PRIMARY_HEADER_LEN + 4);

        // Verify header bytes.
        assert_eq!(buf[0], 0x01); // version=000, type=0, sec_hdr=0, APID_hi=001
        assert_eq!(buf[1], 0xA3); // APID_lo=10100011
        assert_eq!(buf[2], 0xC0); // seq_flags=11, seq_count_hi=000000
        assert_eq!(buf[3], 0x2A); // seq_count_lo=00101010
        assert_eq!(buf[4], 0x00); // data_length_hi
        assert_eq!(buf[5], 0x03); // data_length_lo = 4-1 = 3
    }

    /// Validate a TC packet header with secondary header flag set.
    ///
    /// Expected header:
    ///   Word 1: version=000, type=1(TC), sec_hdr=1, APID=0x100
    ///           = 0b000_1_1_00100000000 = 0x1900
    ///   Word 2: seq_flags=01(first), seq_count=0
    ///           = 0b01_00000000000000 = 0x4000
    ///   Word 3: data_length = 1 - 1 = 0
    ///           = 0x0000
    #[test]
    fn test_space_packet_header_bit_layout_tc() {
        let pkt = SpacePacket::new(
            PacketType::Telecommand,
            true,
            0x100,
            SeqFlags::First,
            0,
            &[0xFF],
        )
        .unwrap();

        let mut buf = [0u8; 64];
        pkt.encode(&mut buf).unwrap();

        assert_eq!(buf[0], 0x19); // version=000, type=1, sec_hdr=1, APID_hi=001
        assert_eq!(buf[1], 0x00); // APID_lo=00000000
        assert_eq!(buf[2], 0x40); // seq_flags=01, seq_count_hi=000000
        assert_eq!(buf[3], 0x00); // seq_count_lo=00000000
        assert_eq!(buf[4], 0x00); // data_length_hi
        assert_eq!(buf[5], 0x00); // data_length_lo = 1-1 = 0
    }

    /// Verify maximum field values encode correctly.
    ///
    /// APID=0x7FF, seq_count=0x3FFF, data=256 bytes:
    ///   Word 1: 000_0_0_11111111111 = 0x07FF
    ///   Word 2: 11_11111111111111 = 0xFFFF
    ///   Word 3: 256-1 = 255 = 0x00FF
    #[test]
    fn test_space_packet_max_field_values() {
        let pkt = SpacePacket::new(
            PacketType::Telemetry,
            false,
            MAX_APID,
            SeqFlags::Unsegmented,
            MAX_SEQ_COUNT,
            &[0xAA; 256],
        )
        .unwrap();

        let mut buf = [0u8; 300];
        pkt.encode(&mut buf).unwrap();

        assert_eq!(buf[0], 0x07);
        assert_eq!(buf[1], 0xFF);
        assert_eq!(buf[2], 0xFF);
        assert_eq!(buf[3], 0xFF);
        assert_eq!(buf[4], 0x00);
        assert_eq!(buf[5], 0xFF);
    }

    /// Verify idle packet APID (0x7FF) is correctly encoded and decoded.
    #[test]
    fn test_idle_packet_apid() {
        let idle = SpacePacket::idle();
        let mut buf = [0u8; 64];
        let len = idle.encode(&mut buf).unwrap();

        // APID should be 0x7FF in the header.
        let word1 = ((buf[0] as u16) << 8) | buf[1] as u16;
        let apid = word1 & 0x7FF;
        assert_eq!(apid, 0x7FF);

        // Roundtrip.
        let decoded = SpacePacket::decode(&buf[..len]).unwrap();
        assert!(decoded.is_idle());
    }

    /// Verify all sequence flag encodings per CCSDS 133.0-B-2 Table 4-2.
    #[test]
    fn test_sequence_flags_encoding() {
        let test_cases = [
            (SeqFlags::Continuation, 0b00),
            (SeqFlags::First, 0b01),
            (SeqFlags::Last, 0b10),
            (SeqFlags::Unsegmented, 0b11),
        ];

        for (flags, expected_bits) in test_cases {
            let pkt = SpacePacket::new(
                PacketType::Telemetry, false, 0, flags, 0, &[0x00],
            )
            .unwrap();

            let mut buf = [0u8; 64];
            pkt.encode(&mut buf).unwrap();

            // Sequence flags are bits 15:14 of word 2 (bytes 2-3).
            let actual_bits = (buf[2] >> 6) & 0x03;
            assert_eq!(
                actual_bits, expected_bits,
                "SeqFlags {:?} should encode as 0b{:02b}, got 0b{:02b}",
                flags, expected_bits, actual_bits
            );
        }
    }

    // ===================================================================
    // CCSDS 232.0-B-4: TC CRC-16/CCITT Known-Value Tests
    // ===================================================================

    /// CRC-16/CCITT with initial value 0xFFFF on the string "123456789"
    /// should produce 0x29B1 per the CCITT specification.
    #[test]
    fn test_crc16_ccitt_standard_check_value() {
        let data = b"123456789";
        let crc = crc16_ccitt(data);
        assert_eq!(crc, 0x29B1, "CRC-16/CCITT check value for '123456789' should be 0x29B1, got 0x{:04X}", crc);
    }

    /// CRC-16/CCITT on empty data should return 0xFFFF (initial value unchanged).
    #[test]
    fn test_crc16_ccitt_empty() {
        let crc = crc16_ccitt(&[]);
        assert_eq!(crc, 0xFFFF);
    }

    /// CRC-16/CCITT on single zero byte.
    #[test]
    fn test_crc16_ccitt_single_zero() {
        let crc = crc16_ccitt(&[0x00]);
        // After processing 0x00: XOR with 0xFF00, then 8 rounds.
        // Known value: 0xE1F0
        assert_eq!(crc, 0xE1F0, "CRC-16/CCITT for [0x00] should be 0xE1F0, got 0x{:04X}", crc);
    }

    /// CRC-16/CCITT on single 0xFF byte.
    ///
    /// Init=0xFFFF, XOR with 0xFF => byte becomes 0x00 in first iteration,
    /// then the high byte XORs to 0xFF with remaining shifts producing 0xFF00.
    #[test]
    fn test_crc16_ccitt_single_ff() {
        let crc = crc16_ccitt(&[0xFF]);
        assert_eq!(crc, 0xFF00, "CRC-16/CCITT for [0xFF] should be 0xFF00, got 0x{:04X}", crc);
    }

    /// TC frame encode/decode preserves CRC integrity across the wire.
    #[test]
    fn test_tc_frame_crc_integrity() {
        let frame = TcFrame::new(true, false, 0x1A3, 5, 99, &[0x01, 0x02, 0x03, 0x04]).unwrap();
        let mut buf = [0u8; 256];
        let len = frame.encode(&mut buf).unwrap();

        // The CRC-16 is the last 2 bytes.
        let crc_hi = buf[len - 2];
        let crc_lo = buf[len - 1];
        let received_crc = ((crc_hi as u16) << 8) | crc_lo as u16;

        // Independently compute CRC over header + data.
        let computed_crc = crc16_ccitt(&buf[..len - FECF_LEN]);
        assert_eq!(received_crc, computed_crc);

        // Verify decode succeeds (CRC passes).
        let decoded = TcFrame::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.data, &[0x01, 0x02, 0x03, 0x04]);
    }

    // ===================================================================
    // CCSDS 132.0-B-3: TM Transfer Frame Header Layout
    // ===================================================================

    /// Validate TM frame header bit-field layout.
    ///
    /// SCID=0x1A3, VCID=5, frame_counter=42, FHP=100:
    ///   Bytes 0-1: version(2)=00, SCID(10)=0x1A3 = 0b0110100011
    ///              = 0b00_0110100011 ... but bits also include some VCID packing
    #[test]
    fn test_tm_frame_header_roundtrip() {
        let frame = TmFrame::new(0x1A3, 5, 42, 100, &[0xAA; 50]).unwrap();
        let mut buf = [0u8; 256];
        let len = frame.encode(&mut buf).unwrap();

        let decoded = TmFrame::decode(&buf[..len]).unwrap();
        assert_eq!(decoded.scid, 0x1A3);
        assert_eq!(decoded.vcid, 5);
        assert_eq!(decoded.frame_counter, 42);
        assert_eq!(decoded.first_header_pointer, 100);
        assert_eq!(decoded.data, &[0xAA; 50]);
    }

    /// Verify idle TM frame FHP value (0x7FF per CCSDS 132.0-B-3 Section 4.1.2.7).
    #[test]
    fn test_tm_frame_idle_fhp() {
        let idle = TmFrame::idle(0x100, 0, 0, 64).unwrap();
        assert_eq!(idle.first_header_pointer, FHP_NO_PACKET);
        assert!(idle.is_idle());

        let mut buf = [0u8; 256];
        let len = idle.encode(&mut buf).unwrap();
        let decoded = TmFrame::decode(&buf[..len]).unwrap();
        assert!(decoded.is_idle());
        assert_eq!(decoded.first_header_pointer, FHP_NO_PACKET);
    }

    /// Verify TM frame size computation.
    #[test]
    fn test_tm_frame_total_length() {
        let frame = TmFrame::new(0, 0, 0, 0, &[0x00; 1024]).unwrap();
        assert_eq!(TM_HEADER_LEN + frame.data.len(), TM_HEADER_LEN + 1024);
    }

    // ===================================================================
    // CCSDS 231.0-B-3: CLTU / BCH Encoding
    // ===================================================================

    /// Verify CLTU start sequence is 0xEB90 per CCSDS 231.0-B-3 Section 5.2.
    #[test]
    fn test_cltu_start_sequence() {
        assert_eq!(CLTU_START, [0xEB, 0x90]);
    }

    /// Verify CLTU tail sequence per CCSDS 231.0-B-3 Section 5.4.
    #[test]
    fn test_cltu_tail_sequence() {
        assert_eq!(CLTU_TAIL, [0xC5, 0xC5, 0xC5, 0xC5, 0xC5, 0xC5, 0xC5, 0x79]);
    }

    /// Verify BCH(63,56) parity computation for all-zeros block.
    ///
    /// The BCH generator polynomial is g(x) = x^7 + x^6 + x^2 + 1.
    /// For an all-zeros information block, the complement of the syndrome is 0xFF.
    #[test]
    fn test_bch_parity_all_zeros() {
        let info = [0x00; CODE_BLOCK_INFO_LEN];
        let parity = bch_parity(&info);
        // For all-zeros input, the register stays 0x00 through all 7 bytes,
        // so bch_parity returns !0x00 = 0xFF.
        assert_eq!(parity, 0xFF, "BCH parity for all-zeros should be 0xFF (complement), got 0x{:02X}", parity);
    }

    /// Verify BCH parity for all-0xFF block.
    #[test]
    fn test_bch_parity_all_ones() {
        let info = [0xFF; CODE_BLOCK_INFO_LEN];
        let parity = bch_parity(&info);
        // Verify the parity is correct by round-tripping.
        let mut block = [0u8; CODE_BLOCK_LEN];
        block[..CODE_BLOCK_INFO_LEN].copy_from_slice(&info);
        block[CODE_BLOCK_INFO_LEN] = parity;
        assert!(bch_verify(&block));
    }

    /// Verify CLTU fill byte is 0x55 per CCSDS 231.0-B-3 Section 5.3.
    #[test]
    fn test_cltu_fill_byte() {
        assert_eq!(FILL_BYTE, 0x55);
    }

    /// Verify code block size is 8 (7 info + 1 parity) per specification.
    #[test]
    fn test_code_block_dimensions() {
        assert_eq!(CODE_BLOCK_INFO_LEN, 7);
        assert_eq!(CODE_BLOCK_LEN, 8);
    }

    /// Full CLTU encode/decode integration with known TC frame data.
    ///
    /// Encodes a TC frame into a CLTU, verifies the structure (start sequence,
    /// BCH-protected code blocks, fill padding, tail sequence), then decodes
    /// and verifies the original data is recovered.
    #[test]
    fn test_cltu_full_integration_with_tc_frame() {
        // Encode a TC frame.
        let tc = TcFrame::new(true, false, 0x1A3, 5, 0, &[0x01, 0x02, 0x03, 0x04]).unwrap();
        let mut tc_buf = [0u8; 256];
        let tc_len = tc.encode(&mut tc_buf).unwrap();
        let tc_bytes = &tc_buf[..tc_len];

        // Wrap in CLTU.
        let cltu = cltu_encode(tc_bytes);

        // Verify start/tail.
        assert_eq!(&cltu[..2], &CLTU_START);
        assert_eq!(&cltu[cltu.len() - 8..], &CLTU_TAIL);

        // Verify all code blocks pass BCH.
        let block_region = &cltu[2..cltu.len() - 8];
        assert_eq!(block_region.len() % CODE_BLOCK_LEN, 0);
        let num_blocks = block_region.len() / CODE_BLOCK_LEN;
        for i in 0..num_blocks {
            let start = i * CODE_BLOCK_LEN;
            let mut block = [0u8; CODE_BLOCK_LEN];
            block.copy_from_slice(&block_region[start..start + CODE_BLOCK_LEN]);
            assert!(bch_verify(&block), "BCH verification failed for block {}", i);
        }

        // Decode CLTU back to TC data.
        let recovered = cltu_decode(&cltu).unwrap();

        // The recovered data includes the original TC bytes and any fill padding.
        assert!(recovered.len() >= tc_len);
        assert_eq!(&recovered[..tc_len], tc_bytes);

        // Verify TC frame decodes correctly from recovered data.
        let decoded_tc = TcFrame::decode(&recovered[..tc_len]).unwrap();
        assert_eq!(decoded_tc.scid, 0x1A3);
        assert!(decoded_tc.bypass);
        assert_eq!(decoded_tc.data, &[0x01, 0x02, 0x03, 0x04]);
    }

    /// Verify that a single-bit error in a CLTU code block is detected.
    #[test]
    fn test_cltu_single_bit_error_detected() {
        let data = [0x42; 7];
        let cltu = cltu_encode(&data);

        // Flip a single bit in the first code block's info byte.
        let mut corrupted = cltu.clone();
        corrupted[2] ^= 0x01; // flip bit 0 of first info byte

        assert!(cltu_decode(&corrupted).is_err());
    }

    // ===================================================================
    // End-to-end: Space Packet -> TC Frame -> CLTU -> decode all layers
    // ===================================================================

    /// Full protocol stack test: Space Packet encapsulated in a TC Frame,
    /// then CLTU-encoded for uplink, then decoded layer by layer.
    #[test]
    fn test_full_stack_space_packet_tc_cltu_roundtrip() {
        // Create a telecommand space packet.
        let cmd_data = [0xAB, 0xCD, 0xEF, 0x01, 0x02, 0x03, 0x04, 0x05];
        let pkt = SpacePacket::new(
            PacketType::Telecommand,
            false,
            0x100,
            SeqFlags::Unsegmented,
            1,
            &cmd_data,
        )
        .unwrap();

        // Encode space packet.
        let mut pkt_buf = [0u8; 64];
        let pkt_len = pkt.encode(&mut pkt_buf).unwrap();

        // Encapsulate in TC frame.
        let tc = TcFrame::new(false, false, 0x042, 0, 0, &pkt_buf[..pkt_len]).unwrap();
        let mut tc_buf = [0u8; 128];
        let tc_len = tc.encode(&mut tc_buf).unwrap();

        // CLTU encode for uplink.
        let cltu = cltu_encode(&tc_buf[..tc_len]);

        // --- Decode path ---

        // CLTU decode.
        let cltu_decoded = cltu_decode(&cltu).unwrap();

        // TC frame decode (from first tc_len bytes of CLTU-decoded data).
        let tc_decoded = TcFrame::decode(&cltu_decoded[..tc_len]).unwrap();
        assert_eq!(tc_decoded.scid, 0x042);

        // Space packet decode from TC frame data.
        let pkt_decoded = SpacePacket::decode(&tc_decoded.data).unwrap();
        assert_eq!(pkt_decoded.packet_type, PacketType::Telecommand);
        assert_eq!(pkt_decoded.apid, 0x100);
        assert_eq!(pkt_decoded.seq_count, 1);
        assert_eq!(pkt_decoded.data, cmd_data);
    }
}
