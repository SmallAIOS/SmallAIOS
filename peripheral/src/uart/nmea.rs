// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! NMEA 0183 sentence parser. Parses $GPGGA and $GPRMC sentences.

#![allow(dead_code)]

/// NMEA parsing error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NmeaError {
    InvalidSentence,
    ChecksumMismatch { expected: u8, computed: u8 },
    FieldMissing,
    ParseError,
    UnsupportedSentence,
}

/// GPS fix quality (from GGA sentence).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixQuality {
    Invalid = 0,
    GpsFix = 1,
    DgpsFix = 2,
}

/// A parsed GPGGA sentence.
#[derive(Debug, Clone, Copy)]
pub struct GgaData {
    pub time_hhmmss: u32,
    pub latitude: i64,  // microdegrees (positive = N)
    pub longitude: i64, // microdegrees (positive = E)
    pub fix_quality: FixQuality,
    pub num_satellites: u8,
    pub hdop_x10: u16,
    pub altitude_mm: i32,
}

/// A parsed GPRMC sentence.
#[derive(Debug, Clone, Copy)]
pub struct RmcData {
    pub time_hhmmss: u32,
    pub status_valid: bool,
    pub latitude: i64,
    pub longitude: i64,
    pub speed_knots_x100: u32,
    pub course_x100: u32,
    pub date_ddmmyy: u32,
}

/// Compute XOR checksum over the bytes between '$' and '*' (exclusive).
pub fn compute_checksum(sentence: &[u8]) -> u8 {
    let start = if sentence.first() == Some(&b'$') {
        1
    } else {
        0
    };
    let end = sentence
        .iter()
        .position(|&b| b == b'*')
        .unwrap_or(sentence.len());
    let mut cksum: u8 = 0;
    for &byte in &sentence[start..end] {
        cksum ^= byte;
    }
    cksum
}

/// Validate checksum in a sentence like "$....*HH\r\n".
pub fn validate_checksum(sentence: &[u8]) -> Result<(), NmeaError> {
    let star_pos = sentence
        .iter()
        .position(|&b| b == b'*')
        .ok_or(NmeaError::InvalidSentence)?;
    if star_pos + 3 > sentence.len() {
        return Err(NmeaError::InvalidSentence);
    }
    let expected = parse_hex_u8(&sentence[star_pos + 1..star_pos + 3])?;
    let computed = compute_checksum(sentence);
    if expected != computed {
        return Err(NmeaError::ChecksumMismatch { expected, computed });
    }
    Ok(())
}

fn parse_hex_u8(bytes: &[u8]) -> Result<u8, NmeaError> {
    if bytes.len() < 2 {
        return Err(NmeaError::ParseError);
    }
    let hi = hex_nibble(bytes[0])?;
    let lo = hex_nibble(bytes[1])?;
    Ok((hi << 4) | lo)
}

fn hex_nibble(b: u8) -> Result<u8, NmeaError> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        _ => Err(NmeaError::ParseError),
    }
}

/// Split a sentence into fields (comma-separated, between $ prefix and * suffix).
fn split_fields(sentence: &[u8]) -> Result<([&[u8]; 20], usize), NmeaError> {
    let start = if sentence.first() == Some(&b'$') {
        1
    } else {
        0
    };
    let end = sentence
        .iter()
        .position(|&b| b == b'*')
        .unwrap_or(sentence.len());
    let data = &sentence[start..end];

    let mut fields: [&[u8]; 20] = [&[]; 20];
    let mut count = 0;
    let mut field_start = 0;

    for (i, &byte) in data.iter().enumerate() {
        if byte == b',' || i == data.len() - 1 {
            let field_end = if byte == b',' { i } else { i + 1 };
            if count < 20 {
                fields[count] = &data[field_start..field_end];
                count += 1;
            }
            field_start = i + 1;
        }
    }
    if count == 0 {
        return Err(NmeaError::InvalidSentence);
    }
    Ok((fields, count))
}

fn parse_u32(bytes: &[u8]) -> Result<u32, NmeaError> {
    if bytes.is_empty() {
        return Err(NmeaError::FieldMissing);
    }
    let mut val = 0u32;
    for &b in bytes {
        if b == b'.' {
            break;
        } // Stop at decimal point.
        if !b.is_ascii_digit() {
            return Err(NmeaError::ParseError);
        }
        val = val * 10 + (b - b'0') as u32;
    }
    Ok(val)
}

/// Parse a coordinate like "4807.038,N" into microdegrees.
/// Format: DDDMM.MMMMM where DDD is degrees, MM.MMMMM is minutes.
fn parse_coordinate(coord: &[u8], direction: &[u8]) -> Result<i64, NmeaError> {
    if coord.is_empty() {
        return Err(NmeaError::FieldMissing);
    }

    let dot_pos = coord.iter().position(|&b| b == b'.').unwrap_or(coord.len());
    if dot_pos < 3 {
        return Err(NmeaError::ParseError);
    }

    let deg_end = dot_pos - 2;
    let degrees = parse_u32(&coord[..deg_end])? as i64;
    let minutes_int = parse_u32(&coord[deg_end..dot_pos])? as i64;

    // Parse fractional minutes.
    let frac_part = if dot_pos + 1 < coord.len() {
        let frac_bytes = &coord[dot_pos + 1..];
        let mut frac = 0i64;
        let mut divisor = 1i64;
        for &b in frac_bytes {
            if !b.is_ascii_digit() {
                break;
            }
            frac = frac * 10 + (b - b'0') as i64;
            divisor *= 10;
        }
        (frac * 1_000_000) / divisor
    } else {
        0
    };

    // Convert: degrees + minutes/60, all in microdegrees.
    let microdeg = degrees * 1_000_000 + (minutes_int * 1_000_000 + frac_part) / 60;

    let sign = if direction == b"S" || direction == b"W" {
        -1
    } else {
        1
    };
    Ok(microdeg * sign)
}

/// Parse a GPGGA sentence.
pub fn parse_gga(sentence: &[u8]) -> Result<GgaData, NmeaError> {
    validate_checksum(sentence)?;
    let (fields, count) = split_fields(sentence)?;
    if count < 10 {
        return Err(NmeaError::FieldMissing);
    }
    if fields[0] != b"GPGGA" && fields[0] != b"GNGGA" {
        return Err(NmeaError::UnsupportedSentence);
    }

    let time = parse_u32(fields[1]).unwrap_or(0);
    let lat = parse_coordinate(fields[2], fields[3]).unwrap_or(0);
    let lon = parse_coordinate(fields[4], fields[5]).unwrap_or(0);
    let fix = match parse_u32(fields[6]).unwrap_or(0) {
        1 => FixQuality::GpsFix,
        2 => FixQuality::DgpsFix,
        _ => FixQuality::Invalid,
    };
    let sats = parse_u32(fields[7]).unwrap_or(0) as u8;
    let hdop = parse_u32(fields[8]).unwrap_or(0) as u16;
    let alt = parse_u32(fields[9]).unwrap_or(0) as i32 * 1000; // Convert m to mm.

    Ok(GgaData {
        time_hhmmss: time,
        latitude: lat,
        longitude: lon,
        fix_quality: fix,
        num_satellites: sats,
        hdop_x10: hdop,
        altitude_mm: alt,
    })
}

/// Parse a GPRMC sentence.
pub fn parse_rmc(sentence: &[u8]) -> Result<RmcData, NmeaError> {
    validate_checksum(sentence)?;
    let (fields, count) = split_fields(sentence)?;
    if count < 10 {
        return Err(NmeaError::FieldMissing);
    }
    if fields[0] != b"GPRMC" && fields[0] != b"GNRMC" {
        return Err(NmeaError::UnsupportedSentence);
    }

    let time = parse_u32(fields[1]).unwrap_or(0);
    let status = fields[2] == b"A";
    let lat = parse_coordinate(fields[3], fields[4]).unwrap_or(0);
    let lon = parse_coordinate(fields[5], fields[6]).unwrap_or(0);
    let speed = parse_u32(fields[7]).unwrap_or(0);
    let course = parse_u32(fields[8]).unwrap_or(0);
    let date = parse_u32(fields[9]).unwrap_or(0);

    Ok(RmcData {
        time_hhmmss: time,
        status_valid: status,
        latitude: lat,
        longitude: lon,
        speed_knots_x100: speed,
        course_x100: course,
        date_ddmmyy: date,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_checksum() {
        let sentence = b"$GPGGA,123456.00,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*6B";
        let cksum = compute_checksum(sentence);
        assert_eq!(cksum, 0x6B);
    }

    #[test]
    fn test_validate_checksum_valid() {
        let sentence = b"$GPGGA,123456.00,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*6B";
        assert!(validate_checksum(sentence).is_ok());
    }

    #[test]
    fn test_validate_checksum_invalid() {
        let sentence = b"$GPGGA,123456.00,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*FF";
        let result = validate_checksum(sentence);
        assert!(matches!(result, Err(NmeaError::ChecksumMismatch { .. })));
    }

    #[test]
    fn test_validate_checksum_no_star() {
        let sentence = b"$GPGGA,123456.00";
        assert_eq!(validate_checksum(sentence), Err(NmeaError::InvalidSentence));
    }

    #[test]
    fn test_parse_gga() {
        let sentence = b"$GPGGA,123456.00,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*6B";
        let gga = parse_gga(sentence).unwrap();
        assert_eq!(gga.time_hhmmss, 123456);
        assert_eq!(gga.fix_quality, FixQuality::GpsFix);
        assert_eq!(gga.num_satellites, 8);
        assert!(gga.latitude > 0); // North
        assert!(gga.longitude > 0); // East
    }

    #[test]
    fn test_parse_rmc() {
        // Compute a valid checksum for this sentence.
        let body = b"GPRMC,123519,A,4807.038,N,01131.000,E,022.4,084.4,230394,003.1,W";
        let mut cksum: u8 = 0;
        for &b in body.iter() {
            cksum ^= b;
        }
        let mut sentence = alloc::vec![b'$'];
        sentence.extend_from_slice(body);
        sentence.push(b'*');
        // Format checksum as hex.
        let hi = if cksum >> 4 < 10 {
            b'0' + (cksum >> 4)
        } else {
            b'A' + (cksum >> 4) - 10
        };
        let lo = if cksum & 0xF < 10 {
            b'0' + (cksum & 0xF)
        } else {
            b'A' + (cksum & 0xF) - 10
        };
        sentence.push(hi);
        sentence.push(lo);

        let rmc = parse_rmc(&sentence).unwrap();
        assert_eq!(rmc.time_hhmmss, 123519);
        assert!(rmc.status_valid);
        assert!(rmc.latitude > 0);
        assert!(rmc.longitude > 0);
        assert_eq!(rmc.date_ddmmyy, 230394);
    }

    #[test]
    fn test_parse_coordinate_north() {
        let lat = parse_coordinate(b"4807.038", b"N").unwrap();
        // 48 degrees + 7.038/60 minutes = 48.1173 degrees ≈ 48117300 microdeg
        assert!(lat > 48_000_000 && lat < 49_000_000);
    }

    #[test]
    fn test_parse_coordinate_south() {
        let lat = parse_coordinate(b"3348.000", b"S").unwrap();
        assert!(lat < 0);
    }

    #[test]
    fn test_hex_nibble() {
        assert_eq!(hex_nibble(b'0'), Ok(0));
        assert_eq!(hex_nibble(b'9'), Ok(9));
        assert_eq!(hex_nibble(b'A'), Ok(10));
        assert_eq!(hex_nibble(b'F'), Ok(15));
        assert_eq!(hex_nibble(b'a'), Ok(10));
        assert!(hex_nibble(b'G').is_err());
    }

    #[test]
    fn test_parse_gga_unsupported() {
        let sentence = b"$GPVTG,054.7,T,034.4,M,005.5,N,010.2,K*48";
        // This isn't GGA so it should fail.
        let result = parse_gga(sentence);
        assert!(result.is_err());
    }
}
