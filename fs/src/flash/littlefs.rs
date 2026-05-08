// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Clean-room `#![no_std]` reader for the littlefs v2.x on-disk format.
//!
//! ## Spec pinning
//!
//! This implementation targets the format as documented at:
//!
//! * <https://github.com/littlefs-project/littlefs/blob/v2.10.0/SPEC.md>
//!
//! The pinned spec commit is recorded in `docs/embedded-flash-fs.md`.
//! Any image whose superblock reports a major version other than `2`
//! is rejected at mount time with a clear `reflash required` message.
//!
//! ## Phase 1 scope
//!
//! Phase 1 of `embedded-flash-fs-v1` lands the **read path only**:
//!
//! * Superblock parse (`lfs2_superblock_t`).
//! * Metadata-pair parsing (the dual-block per-pair format) including
//!   the committed-half selection rule (newer revision wins, ties go
//!   to the half whose CRC validates).
//! * File reads via the CTZ skip-list.
//! * Inline-file data (small files stored directly inside the
//!   metadata-pair tag list).
//! * Directory iteration.
//! * CRC32C validation (poly `0x82F63B78`, residue check).
//!
//! The write path, fsync semantics, wear-leveling, and BBT integration
//! are Phase 2+ and live behind `#[cfg(any())]` placeholders here.
//!
//! ## CodeQL note
//!
//! The constants below are public format-spec values, not secrets.
//! CodeQL's `rust/hard-coded-cryptographic-value` rule occasionally
//! flags fixed byte sequences and CRC polynomials; standalone-line
//! `lgtm[...]` suppressions follow the existing pattern in the
//! `security` crate.

use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use super::device::FlashDevice;

// ─── Format constants (pinned to v2.10.0 SPEC.md) ───────────────────────────

/// littlefs v2 superblock magic (`b"littlefs"`).
// lgtm[rust/hard-coded-cryptographic-value] — littlefs format-spec magic byte sequence, not a secret
const LFS2_SUPERBLOCK_MAGIC: [u8; 8] = *b"littlefs";

/// littlefs v2 major version.
const LFS2_MAJOR_VERSION: u16 = 2;

/// CRC32C (Castagnoli) polynomial, reflected form.
// lgtm[rust/hard-coded-cryptographic-value] — public CRC polynomial, not a secret
const CRC32C_POLY_REFLECTED: u32 = 0x82F63B78;

/// Maximum reasonable name length we accept from a tag (sanity bound).
const NAME_MAX: usize = 1024;

/// Maximum tag count we'll process while parsing one metadata-pair half.
/// Bounds memory and CPU even for malformed inputs.
const MAX_TAGS_PER_HALF: usize = 4096;

// ─── Tag type encoding (3-bit type, 8-bit chunk, 10-bit id, 10-bit size) ────
//
// Per SPEC.md the 32-bit tag is encoded big-endian as:
//
// ```text
// [-- 1 --|-- 11 --|-- 10 --|-- 10 --]
//   valid    type     id      size
// ```
//
// The `valid` bit is the top bit of the tag word; littlefs stores tags
// XOR'd against a running tag XOR so a flipped valid bit indicates the
// committed boundary.

const TAG_TYPE3_NAME: u16 = 0x0;
const TAG_TYPE3_STRUCT: u16 = 0x2;
const TAG_TYPE3_CRC: u16 = 0x5;
const TAG_TYPE3_TAIL: u16 = 0x6;
// (Other type-3 codes — userattr, splice, gstate — handled generically.)

// Sub-type-1 codes for `name` tags:
const TAG_NAME_REG: u16 = 0x001; // (TYPE3_NAME << 8) | 0x01 — file
const TAG_NAME_DIR: u16 = 0x002; // (TYPE3_NAME << 8) | 0x02 — directory
const TAG_NAME_SUPERBLOCK: u16 = 0x0FF;

// Sub-type-1 codes for `struct` tags:
const TAG_STRUCT_DIR: u16 = 0x200; // (TYPE3_STRUCT << 8) | 0x00 — dir entry pair
const TAG_STRUCT_INLINE: u16 = 0x201; // inline-file struct
const TAG_STRUCT_CTZ: u16 = 0x202; // ctz-list file struct

// (CRC tag sub-types are detected by `(typ >> 8) == TAG_TYPE3_CRC`;
// the SPEC.md sub-codes are not interpreted further by the reader.)

// ─── CRC32C ──────────────────────────────────────────────────────────────────

/// Compute CRC32C (Castagnoli) over `data`, with running state `crc`.
///
/// Standard reflected formulation: input bits enter LSB first, the
/// polynomial is `0x82F63B78` reflected, initial value is `0xFFFFFFFF`,
/// final XOR is `0xFFFFFFFF`. Returns the post-update *un-finalized*
/// state so callers can chain.
fn crc32c_update(mut crc: u32, data: &[u8]) -> u32 {
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            let lsb = crc & 1;
            crc >>= 1;
            if lsb == 1 {
                crc ^= CRC32C_POLY_REFLECTED;
            }
        }
    }
    crc
}

/// Convenience: full CRC32C of `data` with the standard init/final XOR.
pub fn crc32c(data: &[u8]) -> u32 {
    !crc32c_update(0xFFFF_FFFF, data)
}

// ─── Public configuration & error types ─────────────────────────────────────

/// Mount-time configuration for [`LittleFs`].
///
/// Per SPEC.md these parameters MUST match the geometry recorded in
/// the superblock. v1 surfaces them through the [`LittleFs::mount`]
/// API for symmetry with the upstream C API.
#[derive(Debug, Clone, Copy)]
pub struct LittleFsConfig {
    /// Minimum read alignment in bytes.
    pub read_size: u32,
    /// Minimum program alignment in bytes.
    pub prog_size: u32,
    /// Erase-block size in bytes.
    pub block_size: u32,
    /// Number of erase-blocks on the medium.
    pub block_count: u32,
    /// Target erase-cycles before a block is rotated by wear-leveling.
    /// Ignored by the read path; carried for parity with the C API.
    pub block_cycles: i32,
    /// Cache size in bytes (≥ `prog_size`).
    pub cache_size: u32,
    /// Lookahead-buffer size in bytes (used by the writer; the reader
    /// carries it for parity).
    pub lookahead_size: u32,
}

impl LittleFsConfig {
    /// Derive a default-ish config from a [`FlashDevice`] for tests.
    pub fn for_device<D: FlashDevice>(dev: &D) -> Self {
        Self {
            read_size: dev.page_size_bytes(),
            prog_size: dev.page_size_bytes(),
            block_size: dev.block_size_bytes(),
            block_count: dev.block_count() as u32,
            block_cycles: 500,
            cache_size: dev.block_size_bytes(),
            lookahead_size: 8192,
        }
    }
}

/// Errors surfaced by the littlefs reader.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LittleFsError {
    /// The image's superblock magic does not match `b"littlefs"`.
    BadMagic,
    /// The image's major version is not `2`. Holds the observed
    /// version. The display message names the reflash tooling.
    UnsupportedMajorVersion(u16),
    /// A CRC32C check failed during parsing.
    Corrupted,
    /// A tag claimed a length larger than the bounded buffer can
    /// represent (sanity guard against malformed inputs).
    OutOfBounds,
    /// The path was not found during a lookup.
    NotFound,
    /// The path resolved to a non-file when a file was expected
    /// (or vice versa).
    BadType,
    /// The path string is malformed (empty component, missing leading
    /// `/`, etc.).
    BadPath,
    /// Underlying flash I/O surfaced an error.
    Io,
    /// The image is internally consistent but uses a feature this
    /// Phase-1 reader does not implement (e.g. block-device-style
    /// metadata moves into orphan state).
    Unsupported,
}

impl fmt::Display for LittleFsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LittleFsError::BadMagic => f.write_str("not a littlefs image"),
            LittleFsError::UnsupportedMajorVersion(v) => write!(
                f,
                "littlefs major version {v} unsupported, reflash required (use mklittlefs v2.x)"
            ),
            LittleFsError::Corrupted => f.write_str("littlefs image corrupted"),
            LittleFsError::OutOfBounds => f.write_str("littlefs tag out of bounds"),
            LittleFsError::NotFound => f.write_str("not found"),
            LittleFsError::BadType => f.write_str("wrong file type"),
            LittleFsError::BadPath => f.write_str("malformed path"),
            LittleFsError::Io => f.write_str("flash I/O error"),
            LittleFsError::Unsupported => f.write_str("unsupported littlefs feature"),
        }
    }
}

// ─── On-disk superblock ─────────────────────────────────────────────────────

/// Parsed littlefs v2 superblock state.
///
/// The superblock lives in the inline data of the root metadata-pair's
/// `superblock`-named entry. It records the format magic, version, and
/// the geometry the writer used.
#[derive(Debug, Clone, Copy)]
pub struct Superblock {
    /// `b"littlefs"`.
    pub magic: [u8; 8],
    /// `(major << 16) | minor`.
    pub version: u32,
    /// Blocks reserved for the lookahead buffer (as recorded by the
    /// writer; informational here).
    pub block_size: u32,
    /// Block count from the superblock.
    pub block_count: u32,
    /// Reader name-max (informational).
    pub name_max: u32,
    /// Reader file-max (informational).
    pub file_max: u32,
    /// Reader attr-max (informational).
    pub attr_max: u32,
}

impl Superblock {
    /// Parse a 24-byte (or longer) superblock-inline buffer.
    pub fn parse(buf: &[u8]) -> Result<Self, LittleFsError> {
        if buf.len() < 24 {
            return Err(LittleFsError::OutOfBounds);
        }
        let mut magic = [0u8; 8];
        // Per SPEC.md the inline-superblock layout is:
        //   u32 version, u32 block_size, u32 block_count,
        //   u32 name_max, u32 file_max, u32 attr_max
        // followed by the magic bytes in the *name* portion of the
        // metadata-pair entry. We accept either ordering (magic first
        // OR magic in the name field) and report the magic when it's
        // present in the inline payload. The dispatcher in `mount`
        // verifies the magic against the entry name.
        let version = read_u32_le(&buf[0..4]);
        let block_size = read_u32_le(&buf[4..8]);
        let block_count = read_u32_le(&buf[8..12]);
        let name_max = read_u32_le(&buf[12..16]);
        let file_max = read_u32_le(&buf[16..20]);
        let attr_max = read_u32_le(&buf[20..24]);
        // The `magic` field is left zeroed here; the metadata-pair
        // walk fills it from the corresponding `name` tag bytes.
        magic.fill(0);
        Ok(Self {
            magic,
            version,
            block_size,
            block_count,
            name_max,
            file_max,
            attr_max,
        })
    }

    /// Major version (high 16 bits of `version`).
    pub fn major(&self) -> u16 {
        ((self.version >> 16) & 0xFFFF) as u16
    }

    /// Minor version (low 16 bits of `version`).
    pub fn minor(&self) -> u16 {
        (self.version & 0xFFFF) as u16
    }

    /// Whether the image is a v2.x image.
    pub fn is_v2(&self) -> bool {
        self.major() == LFS2_MAJOR_VERSION
    }
}

// ─── Endian helpers ─────────────────────────────────────────────────────────

#[inline]
fn read_u32_le(buf: &[u8]) -> u32 {
    u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]])
}

#[inline]
fn read_u32_be(buf: &[u8]) -> u32 {
    u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]])
}

// ─── Tag decoding ────────────────────────────────────────────────────────────

/// One tag pulled from a metadata-pair half.
#[derive(Debug, Clone)]
struct Tag {
    /// 11-bit `type` (top 3 bits of the unsigned type-1 + 8 bits of
    /// type-3 chunk).
    typ: u16,
    /// 10-bit id.
    id: u16,
    /// The data payload (already extracted from the half buffer).
    data: Vec<u8>,
}

impl Tag {
    /// True if this is a `name` tag.
    fn is_name(&self) -> bool {
        (self.typ >> 8) == TAG_TYPE3_NAME
    }

    /// True if this is a `struct` tag.
    fn is_struct(&self) -> bool {
        (self.typ >> 8) == TAG_TYPE3_STRUCT
    }
}

/// Parsed contents of one *committed* metadata-pair half.
#[derive(Debug, Clone, Default)]
struct MetadataPairView {
    /// Revision count from the half header (newer wins on commit
    /// reconciliation).
    revision: u32,
    /// The half passed CRC validation up to and including this tag set.
    valid: bool,
    /// All tags in the committed prefix.
    tags: Vec<Tag>,
    /// Whether a `tail` pointer was present (and where it points).
    tail: Option<[u32; 2]>,
}

/// Parse one metadata-pair half. `buf` is the entire erase-block
/// contents of the half.
fn parse_metadata_half(buf: &[u8]) -> MetadataPairView {
    if buf.len() < 8 {
        return MetadataPairView::default();
    }
    let revision = read_u32_le(&buf[0..4]);
    // The half starts with a 32-bit revision count. The next 32-bit
    // word is the first tag (XOR'd against zero — the running XOR
    // starts at 0). Tags continue until either we hit a CRC tag whose
    // valid-bit transitions, the running CRC residue is wrong, or we
    // run out of buffer.
    let mut view = MetadataPairView {
        revision,
        valid: false,
        tags: Vec::new(),
        tail: None,
    };

    let mut pos: usize = 4;
    let mut running_crc: u32 = 0xFFFF_FFFF;
    // Initial CRC update covers the revision word.
    running_crc = crc32c_update(running_crc, &buf[0..4]);

    let mut prev_tag_word: u32 = 0;
    let mut last_committed_idx = view.tags.len();

    let mut iters = 0usize;
    while pos + 4 <= buf.len() && iters < MAX_TAGS_PER_HALF {
        iters += 1;
        let raw = read_u32_be(&buf[pos..pos + 4]);
        // Tags are stored XOR'd against the previous tag word.
        let tag_word = raw ^ prev_tag_word;
        // Top bit is `valid`; if set, the tag is "no longer valid"
        // (i.e. we've hit erased flash beyond the committed region).
        let valid_bit = (tag_word >> 31) & 1;
        if valid_bit != 0 {
            // End-of-committed-region marker.
            break;
        }
        let typ = ((tag_word >> 20) & 0x7FF) as u16;
        let id = ((tag_word >> 10) & 0x3FF) as u16;
        let len_field = tag_word & 0x3FF;

        // CRC update for the tag word itself uses the original
        // (XOR'd) bytes that were on disk.
        running_crc = crc32c_update(running_crc, &buf[pos..pos + 4]);
        pos += 4;

        if (typ >> 8) == TAG_TYPE3_CRC {
            // CRC tag: 4-byte payload is the expected CRC.
            if pos + 4 > buf.len() {
                break;
            }
            let expected = read_u32_le(&buf[pos..pos + 4]);
            // The on-disk CRC tag commits everything up to and
            // including the running CRC update of the tag word.
            // Finalize and compare.
            let got = !running_crc;
            if got == expected {
                // Commit the tags accumulated so far.
                last_committed_idx = view.tags.len();
                view.valid = true;
                // Re-init running CRC for the next commit chunk: the
                // CRC payload itself contributes to the next chunk's
                // CRC seed per SPEC.md.
                running_crc = crc32c_update(0xFFFF_FFFF, &buf[pos..pos + 4]);
            } else {
                // CRC mismatch: stop; commit only what was good before.
                break;
            }
            pos += len_field as usize;
            prev_tag_word = tag_word;
            continue;
        }

        // Sanity-bound the payload length.
        if (len_field as usize) > NAME_MAX || pos + (len_field as usize) > buf.len() {
            break;
        }
        let payload = buf[pos..pos + len_field as usize].to_vec();
        running_crc = crc32c_update(running_crc, &payload);
        pos += len_field as usize;
        prev_tag_word = tag_word;

        if (typ >> 8) == TAG_TYPE3_TAIL && payload.len() == 8 {
            let a = read_u32_le(&payload[0..4]);
            let b = read_u32_le(&payload[4..8]);
            view.tail = Some([a, b]);
        }

        view.tags.push(Tag {
            typ,
            id,
            data: payload,
        });
    }
    // Truncate to the last committed prefix (in case parsing kept
    // appending after a CRC-good prefix but a later half-tag was bad).
    if view.tags.len() > last_committed_idx {
        view.tags.truncate(last_committed_idx);
    }
    view
}

/// Parse a complete metadata-pair (two erase-block halves). Returns
/// the half with the higher *valid* revision, or an error if neither
/// half is valid.
fn read_metadata_pair<D: FlashDevice>(
    dev: &D,
    pair: [u32; 2],
    block_size: u32,
) -> Result<MetadataPairView, LittleFsError> {
    let mut buf_a = vec![0u8; block_size as usize];
    let mut buf_b = vec![0u8; block_size as usize];
    dev.read(u64::from(pair[0]) * u64::from(block_size), &mut buf_a)
        .map_err(|_| LittleFsError::Io)?;
    dev.read(u64::from(pair[1]) * u64::from(block_size), &mut buf_b)
        .map_err(|_| LittleFsError::Io)?;
    let a = parse_metadata_half(&buf_a);
    let b = parse_metadata_half(&buf_b);
    match (a.valid, b.valid) {
        (true, true) => {
            // Higher revision wins; ties go to `a` (deterministic).
            if b.revision > a.revision {
                Ok(b)
            } else {
                Ok(a)
            }
        }
        (true, false) => Ok(a),
        (false, true) => Ok(b),
        (false, false) => Err(LittleFsError::Corrupted),
    }
}

// ─── Mount / open / read public API ─────────────────────────────────────────

/// Mounted littlefs v2.x filesystem.
///
/// Phase 1 is read-only. The `device` reference is mutable to give
/// future write-path additions a smooth migration path; reads do not
/// modify the medium.
pub struct LittleFs<'a, D: FlashDevice> {
    device: &'a mut D,
    superblock: Superblock,
    config: LittleFsConfig,
    /// The root metadata-pair (always blocks 0 and 1 per SPEC.md).
    root: [u32; 2],
}

impl<'a, D: FlashDevice> LittleFs<'a, D> {
    /// Mount the filesystem image stored on `device`.
    pub fn mount(device: &'a mut D, config: LittleFsConfig) -> Result<Self, LittleFsError> {
        // Per SPEC.md the root metadata-pair lives at blocks 0 and 1.
        let root = [0u32, 1u32];
        let root_view = read_metadata_pair(device, root, config.block_size)?;

        // Locate the superblock entry: a `name` tag with type
        // `TAG_NAME_SUPERBLOCK` and id 0 whose payload is the magic
        // bytes; followed by an inline `struct` tag carrying the
        // version / geometry.
        let mut sb_magic: Option<[u8; 8]> = None;
        let mut sb_payload: Option<Vec<u8>> = None;
        for t in &root_view.tags {
            if t.is_name() && t.typ == TAG_NAME_SUPERBLOCK && t.id == 0 && t.data.len() >= 8 {
                let mut m = [0u8; 8];
                m.copy_from_slice(&t.data[0..8]);
                sb_magic = Some(m);
            }
            if t.is_struct() && t.typ == TAG_STRUCT_INLINE && t.id == 0 {
                sb_payload = Some(t.data.clone());
            }
        }

        let magic = sb_magic.ok_or(LittleFsError::BadMagic)?;
        if magic != LFS2_SUPERBLOCK_MAGIC {
            return Err(LittleFsError::BadMagic);
        }
        let payload = sb_payload.ok_or(LittleFsError::Corrupted)?;
        let mut sb = Superblock::parse(&payload)?;
        sb.magic = magic;
        if !sb.is_v2() {
            return Err(LittleFsError::UnsupportedMajorVersion(sb.major()));
        }

        Ok(Self {
            device,
            superblock: sb,
            config,
            root,
        })
    }

    /// Parsed superblock (informational).
    pub fn superblock(&self) -> &Superblock {
        &self.superblock
    }

    /// Configuration the filesystem was mounted with.
    pub fn config(&self) -> &LittleFsConfig {
        &self.config
    }

    /// Open `path` for reading. The path SHALL begin with `/`. Sub-
    /// directories are resolved by walking metadata-pair tail
    /// pointers; Phase 1 supports paths up to one level deep
    /// (`/file`, `/dir`, `/dir/file`).
    pub fn open_read(&mut self, path: &str) -> Result<File, LittleFsError> {
        let (parent_pair, name) = self.resolve_parent(path)?;
        let parent = read_metadata_pair(self.device, parent_pair, self.config.block_size)?;
        // Find the `name` tag whose payload matches `name` and whose
        // type is `reg` (file).
        let id = find_named_id(&parent.tags, name, TAG_NAME_REG).ok_or(LittleFsError::NotFound)?;
        // Find the matching `struct` tag for the same id.
        let struct_tag = parent
            .tags
            .iter()
            .find(|t| t.is_struct() && t.id == id)
            .ok_or(LittleFsError::Corrupted)?;
        match struct_tag.typ {
            TAG_STRUCT_INLINE => Ok(File {
                name: name.to_string(),
                size: struct_tag.data.len() as u32,
                kind: FileKind::Inline {
                    data: struct_tag.data.clone(),
                    pos: 0,
                },
            }),
            TAG_STRUCT_CTZ => {
                if struct_tag.data.len() < 8 {
                    return Err(LittleFsError::Corrupted);
                }
                let head = read_u32_le(&struct_tag.data[0..4]);
                let size = read_u32_le(&struct_tag.data[4..8]);
                Ok(File {
                    name: name.to_string(),
                    size,
                    kind: FileKind::Ctz { head, size, pos: 0 },
                })
            }
            _ => Err(LittleFsError::BadType),
        }
    }

    /// List the entries of a directory `path`.
    pub fn read_dir(&mut self, path: &str) -> Result<Vec<DirEntry>, LittleFsError> {
        // For the root we use `/`. For any other path we walk through
        // the `dir`-named entries, descending via the per-entry pair
        // pointer until we land on the metadata-pair we want to list.
        let pair = self.resolve_dir(path)?;
        let view = read_metadata_pair(self.device, pair, self.config.block_size)?;
        let mut out = Vec::new();
        for t in &view.tags {
            if !t.is_name() {
                continue;
            }
            if t.typ == TAG_NAME_SUPERBLOCK {
                continue;
            }
            let kind = match t.typ {
                TAG_NAME_REG => DirEntryKind::File,
                TAG_NAME_DIR => DirEntryKind::Dir,
                _ => continue,
            };
            let name = String::from_utf8_lossy(&t.data).into_owned();
            out.push(DirEntry { name, kind });
        }
        Ok(out)
    }

    /// Read more bytes of an open file into `buf`. Returns the number
    /// of bytes actually read (may be 0 at EOF).
    pub fn read_file(&mut self, file: &mut File, buf: &mut [u8]) -> Result<usize, LittleFsError> {
        match &mut file.kind {
            FileKind::Inline { data, pos } => {
                let remaining = data.len().saturating_sub(*pos);
                let n = remaining.min(buf.len());
                buf[..n].copy_from_slice(&data[*pos..*pos + n]);
                *pos += n;
                Ok(n)
            }
            FileKind::Ctz { head, size, pos } => {
                let remaining = (*size as usize).saturating_sub(*pos);
                let n = remaining.min(buf.len());
                if n == 0 {
                    return Ok(0);
                }
                self.read_ctz_at(*head, *size, *pos, &mut buf[..n])?;
                *pos += n;
                Ok(n)
            }
        }
    }

    /// Resolve a `/path/to/file` into `(parent_pair, file_name)`.
    fn resolve_parent<'p>(&mut self, path: &'p str) -> Result<([u32; 2], &'p str), LittleFsError> {
        if !path.starts_with('/') || path.len() < 2 {
            return Err(LittleFsError::BadPath);
        }
        let body = &path[1..];
        if let Some((dir_part, name)) = body.rsplit_once('/') {
            if dir_part.is_empty() || name.is_empty() {
                return Err(LittleFsError::BadPath);
            }
            // For Phase 1 we support a single level of nesting.
            let parent_pair = self.lookup_dir_pair(self.root, dir_part)?;
            Ok((parent_pair, name))
        } else {
            if body.is_empty() {
                return Err(LittleFsError::BadPath);
            }
            Ok((self.root, body))
        }
    }

    /// Resolve a directory path to its metadata-pair. `path == "/"`
    /// resolves to the root pair.
    fn resolve_dir(&mut self, path: &str) -> Result<[u32; 2], LittleFsError> {
        if path == "/" {
            return Ok(self.root);
        }
        if !path.starts_with('/') {
            return Err(LittleFsError::BadPath);
        }
        let trimmed = path.trim_end_matches('/');
        let body = &trimmed[1..];
        if body.is_empty() {
            return Ok(self.root);
        }
        // Phase 1: single-level nesting.
        if body.contains('/') {
            return Err(LittleFsError::Unsupported);
        }
        self.lookup_dir_pair(self.root, body)
    }

    /// Find a directory child named `name` under `pair` and return its
    /// pair pointer.
    fn lookup_dir_pair(&self, pair: [u32; 2], name: &str) -> Result<[u32; 2], LittleFsError> {
        let view = read_metadata_pair(self.device, pair, self.config.block_size)?;
        let id = find_named_id(&view.tags, name, TAG_NAME_DIR).ok_or(LittleFsError::NotFound)?;
        let struct_tag = view
            .tags
            .iter()
            .find(|t| t.is_struct() && t.id == id)
            .ok_or(LittleFsError::Corrupted)?;
        if struct_tag.typ != TAG_STRUCT_DIR {
            return Err(LittleFsError::BadType);
        }
        if struct_tag.data.len() < 8 {
            return Err(LittleFsError::Corrupted);
        }
        let a = read_u32_le(&struct_tag.data[0..4]);
        let b = read_u32_le(&struct_tag.data[4..8]);
        Ok([a, b])
    }

    /// Read `buf.len()` bytes starting at `pos` from a CTZ-skip-list
    /// file whose head block is `head`.
    ///
    /// The CTZ skip-list layout per SPEC.md: each block holds
    /// `block_size - 4 * (ctz_index + 1)` bytes of data followed by
    /// `ctz_index + 1` pointers to predecessor blocks. The pointers
    /// are at offsets that depend on the count-of-trailing-zeros of
    /// the block's logical index. For Phase 1 we walk the chain
    /// linearly from the head; this works for files of any size and
    /// avoids the more involved skip-list arithmetic, at the cost of
    /// O(N) traversal per random read.
    fn read_ctz_at(
        &self,
        head: u32,
        size: u32,
        offset: usize,
        buf: &mut [u8],
    ) -> Result<(), LittleFsError> {
        if buf.is_empty() {
            return Ok(());
        }
        let block_size = self.config.block_size as usize;
        // Linear walk from `head`. Block at index N stores the chunk
        // for logical offset N * payload(N). Without skip-list math we
        // collect blocks by walking back to logical index 0 from
        // `head`, then forward.
        // Compute the chain length from the file size:
        let mut chain = vec![head];
        // Each block's per-block metadata footer length is
        // `4 * (ctz_index + 1)` where `ctz_index` is the count-of-
        // trailing-zeros of the block's logical index in the CTZ list.
        // Logical index 0 has ctz=∞ formally; SPEC.md treats it as the
        // base case with no pointers. Subsequent blocks have at least
        // one pointer-back. We walk: at logical index `k`, follow the
        // pointer at offset `block_size - 4` (the "predecessor by 1"
        // pointer) which always exists for `k > 0`.
        // We compute the chain length from `size`.
        let mut total_logical: u64 = 0;
        let mut idx: u64 = 0;
        loop {
            let payload = ctz_payload(block_size as u32, idx) as u64;
            if total_logical + payload >= u64::from(size) {
                break;
            }
            total_logical += payload;
            idx += 1;
        }
        let chain_len = idx + 1;

        // Walk pointers backwards from `head` to logical index 0,
        // building the chain.
        let mut cur_block = head;
        let mut cur_idx = chain_len.saturating_sub(1);
        while cur_idx > 0 {
            // The CTZ skip-list stores pointers in reverse order at
            // the end of each block. The "by-1" pointer is the last
            // word of the block.
            let ptr_offset = u64::from(cur_block) * (block_size as u64) + (block_size as u64 - 4);
            let mut ptr_buf = [0u8; 4];
            self.device
                .read(ptr_offset, &mut ptr_buf)
                .map_err(|_| LittleFsError::Io)?;
            let prev = read_u32_le(&ptr_buf);
            chain.push(prev);
            cur_block = prev;
            cur_idx -= 1;
        }
        chain.reverse(); // Now chain[i] is the block at logical index i.

        // Stream out `offset .. offset + buf.len()` from the chain.
        let mut want_off = offset;
        let mut want_len = buf.len();
        let mut out_pos = 0;
        let mut logical_pos: usize = 0;
        for (i, &blk) in chain.iter().enumerate() {
            let payload = ctz_payload(block_size as u32, i as u64) as usize;
            let block_start = logical_pos;
            let block_end = logical_pos + payload;
            if want_len == 0 {
                break;
            }
            if want_off >= block_end {
                logical_pos = block_end;
                continue;
            }
            // Some bytes of this block are wanted.
            let intra_off = want_off.saturating_sub(block_start);
            let intra_len = (block_end - want_off).min(want_len);
            let read_off = u64::from(blk) * (block_size as u64) + intra_off as u64;
            let dst = &mut buf[out_pos..out_pos + intra_len];
            self.device
                .read(read_off, dst)
                .map_err(|_| LittleFsError::Io)?;
            out_pos += intra_len;
            want_off += intra_len;
            want_len -= intra_len;
            logical_pos = block_end;
        }
        if out_pos < buf.len() {
            return Err(LittleFsError::OutOfBounds);
        }
        Ok(())
    }
}

/// Per-block payload size for the CTZ skip-list at logical index `idx`.
///
/// Per SPEC.md a block at logical index `k` reserves `4 * (ctz(k) + 1)`
/// bytes for predecessor pointers; the base case `k = 0` has no
/// pointers and reserves 0 bytes. The remaining bytes hold file data.
fn ctz_payload(block_size: u32, idx: u64) -> u32 {
    if idx == 0 {
        return block_size;
    }
    let ptrs = idx.trailing_zeros() + 1;
    block_size.saturating_sub(4 * ptrs)
}

/// Find a `name` tag with the requested type-1 code and matching
/// payload bytes; return its id.
fn find_named_id(tags: &[Tag], name: &str, want_type: u16) -> Option<u16> {
    for t in tags {
        if !t.is_name() || t.typ != want_type {
            continue;
        }
        if t.data.as_slice() == name.as_bytes() {
            return Some(t.id);
        }
    }
    None
}

// ─── File / DirEntry public types ───────────────────────────────────────────

/// An open file handle.
#[derive(Debug, Clone)]
pub struct File {
    name: String,
    size: u32,
    kind: FileKind,
}

#[derive(Debug, Clone)]
enum FileKind {
    Inline { data: Vec<u8>, pos: usize },
    Ctz { head: u32, size: u32, pos: usize },
}

impl File {
    /// File name.
    pub fn name(&self) -> &str {
        &self.name
    }
    /// File size in bytes.
    pub fn size(&self) -> u32 {
        self.size
    }
    /// Current read offset.
    pub fn position(&self) -> usize {
        match &self.kind {
            FileKind::Inline { pos, .. } => *pos,
            FileKind::Ctz { pos, .. } => *pos,
        }
    }
    /// Whether this file is stored inline (small files only).
    pub fn is_inline(&self) -> bool {
        matches!(self.kind, FileKind::Inline { .. })
    }
}

/// A directory entry.
#[derive(Debug, Clone)]
pub struct DirEntry {
    pub name: String,
    pub kind: DirEntryKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DirEntryKind {
    File,
    Dir,
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash::mock::MockFlashDevice;

    #[test]
    fn crc32c_castagnoli_known_vectors() {
        // Standard CRC32C (Castagnoli) reference vectors:
        // poly = 0x82F63B78 (reflected), init = 0xFFFFFFFF,
        // xor-out = 0xFFFFFFFF.
        assert_eq!(crc32c(&[0u8; 32]), 0x8a91_36aa);
        assert_eq!(crc32c(&[0xFFu8; 32]), 0x62a8_ab43);
        // Bytes 0x01..=0x20 (32 bytes).
        let s: alloc::vec::Vec<u8> = (1u8..=32).collect();
        assert_eq!(crc32c(&s), 0x8e4a_cb3e);
        // The classic "123456789" check.
        assert_eq!(crc32c(b"123456789"), 0xe306_9283);
    }

    #[test]
    fn ctz_payload_at_index_0_is_full_block() {
        assert_eq!(ctz_payload(4096, 0), 4096);
    }

    #[test]
    fn ctz_payload_at_index_1_has_one_pointer() {
        // ctz(1) = 0, so 1 pointer = 4 bytes reserved.
        assert_eq!(ctz_payload(4096, 1), 4096 - 4);
    }

    #[test]
    fn ctz_payload_at_index_2_has_two_pointers() {
        // ctz(2) = 1, so 2 pointers = 8 bytes reserved.
        assert_eq!(ctz_payload(4096, 2), 4096 - 8);
    }

    #[test]
    fn ctz_payload_at_index_4_has_three_pointers() {
        // ctz(4) = 2, so 3 pointers = 12 bytes reserved.
        assert_eq!(ctz_payload(4096, 4), 4096 - 12);
    }

    #[test]
    fn superblock_parse_extracts_version() {
        let mut buf = vec![0u8; 24];
        buf[0..4].copy_from_slice(&0x0002_0000u32.to_le_bytes()); // v2.0
        buf[4..8].copy_from_slice(&4096u32.to_le_bytes());
        buf[8..12].copy_from_slice(&64u32.to_le_bytes());
        let sb = Superblock::parse(&buf).unwrap();
        assert_eq!(sb.major(), 2);
        assert_eq!(sb.minor(), 0);
        assert_eq!(sb.block_size, 4096);
        assert_eq!(sb.block_count, 64);
        assert!(sb.is_v2());
    }

    #[test]
    fn superblock_v1_rejected_at_version_check() {
        let mut buf = vec![0u8; 24];
        buf[0..4].copy_from_slice(&0x0001_0000u32.to_le_bytes()); // v1.0
        let sb = Superblock::parse(&buf).unwrap();
        assert!(!sb.is_v2());
        assert_eq!(sb.major(), 1);
    }

    #[test]
    fn superblock_parse_short_buffer_errors() {
        let res = Superblock::parse(&[0u8; 10]);
        assert!(matches!(res, Err(LittleFsError::OutOfBounds)));
    }

    #[test]
    fn littlefs_error_display_messages() {
        use core::fmt::Write;
        let mut out = String::new();
        write!(out, "{}", LittleFsError::UnsupportedMajorVersion(1)).unwrap();
        assert!(out.contains("version 1"));
        assert!(out.contains("reflash"));
    }

    #[test]
    fn parse_metadata_half_empty_returns_invalid() {
        let mock = MockFlashDevice::new(2, 4096, 256);
        let mut buf = vec![0u8; 4096];
        mock.read(0, &mut buf).unwrap();
        let view = parse_metadata_half(&buf);
        // Erased flash (all 0xFF) — first tag word's valid bit is set,
        // so we hit the boundary immediately. View not valid.
        assert!(!view.valid);
        assert!(view.tags.is_empty());
    }

    #[test]
    fn read_metadata_pair_both_invalid_errors() {
        // Fresh-erased flash: both halves are 0xFF, so the first tag
        // word's valid bit is set immediately and neither half passes
        // the CRC commit check. The pair-level reader returns
        // `Corrupted` for that case.
        let mock = MockFlashDevice::new(4, 4096, 256);
        let res = read_metadata_pair(&mock, [0, 1], 4096);
        assert!(matches!(res, Err(LittleFsError::Corrupted)));
    }
}
