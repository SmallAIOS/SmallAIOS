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

use super::device::{FlashDevice, FlashError};

mod alloc_mod;
mod bbt;
mod commit;
mod format;
mod write;

pub use format::crc32c;
pub use write::OpenFlags;

use alloc_mod::BlockAllocator;
use bbt::BadBlockTable;

/// Internal error raised by the metadata-pair commit builder when an
/// entry's tag payload exceeds the 10-bit length field of the tag word.
/// Surfaces to the public API as [`LittleFsError::Unsupported`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FlashCommitError {
    /// A tag's payload length exceeds `0x3FF` bytes.
    TagTooLarge,
}

/// Internal error raised by the block allocator when no free block is
/// available. Surfaces to the public API as [`LittleFsError::Io`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FlashAllocError {
    /// No free blocks remain in the allocator's pool.
    NoFreeBlocks,
}
use commit::{build_metadata_half, CommitEntry};
use format::{
    crc32c_update, read_u32_be, read_u32_le, LFS2_MAJOR_VERSION, LFS2_SUPERBLOCK_MAGIC,
    MAX_TAGS_PER_HALF, NAME_MAX, TAG_NAME_DIR, TAG_NAME_REG, TAG_NAME_SUPERBLOCK, TAG_STRUCT_CTZ,
    TAG_STRUCT_DIR, TAG_STRUCT_INLINE, TAG_TYPE3_CRC, TAG_TYPE3_NAME, TAG_TYPE3_STRUCT,
    TAG_TYPE3_TAIL,
};
use write::{MutableState, PairEntry, PairState};

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
/// Phase 2 adds the write path on top of the Phase 1 reader. The
/// `mutable` field is lazily initialised on the first write call:
/// reads do not allocate or scan free blocks.
pub struct LittleFs<'a, D: FlashDevice> {
    device: &'a mut D,
    superblock: Superblock,
    config: LittleFsConfig,
    /// The root metadata-pair (always blocks 0 and 1 per SPEC.md).
    root: [u32; 2],
    /// Mutable mount state; populated on the first write-path call.
    mutable: Option<MutableState>,
    /// Most recent `fadvise` hint applied to this mount. v1 honors the
    /// hint advisorily — it does not yet steer block-cache or
    /// write-batching decisions. The tests verify the hint round-trips
    /// cleanly (SEQUENTIAL/RANDOM/etc.) so a future cache layer can pick
    /// it up without an API break.
    fadvise: FadviseHint,
}

/// POSIX `fadvise`-style hint accepted on `/flash/`-backed file
/// descriptors.
///
/// Per `posix-vfs` spec, SmallAIOS routes the relevant subset of
/// `posix_fadvise` constants to the underlying littlefs cache layer:
///
/// * `Sequential` — caller will read in offset order. v1 is a no-op
///   placeholder; future versions can enable read-ahead and coalesce
///   appends into a single metadata-pair commit.
/// * `Random` — caller will read at random offsets. Disables any
///   read-ahead. v1 is a no-op (the reader has no prefetch yet).
/// * `WillNeed` — caller intends to access the data soon. Accepted as
///   a no-op (POSIX semantics: a hint, not a command).
/// * `DontNeed` — caller is finished with the data. Accepted as a
///   no-op.
/// * `BatchWrite` — SmallAIOS-specific extension — write-batching
///   directive used by the audit-log writer to coalesce short appends
///   into one metadata-pair commit. v1 accepts and round-trips the
///   hint; the actual coalescing logic lands when the audit-log
///   integration test suite ships.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FadviseHint {
    /// No hint set (mount default).
    #[default]
    Normal,
    /// Reads will be sequential.
    Sequential,
    /// Reads will be at random offsets.
    Random,
    /// Caller will need this data soon.
    WillNeed,
    /// Caller is done with this data.
    DontNeed,
    /// Coalesce short appends (SmallAIOS extension).
    BatchWrite,
}

impl<'a, D: FlashDevice> LittleFs<'a, D> {
    /// Test whether `device` already carries a valid littlefs v2 root
    /// metadata-pair. Used by the boot sequencer to decide between
    /// [`Self::mount`] and [`Self::format`].
    ///
    /// Returns `true` only if both halves of the root pair are readable
    /// and at least one half passes the CRC commit check AND records the
    /// `b"littlefs"` superblock magic. A device that has never been
    /// formatted (raw `0xFF` after manufacturing erase) returns `false`.
    pub fn is_formatted(device: &mut D, config: LittleFsConfig) -> bool {
        let root = [0u32, 1u32];
        let view = match read_metadata_pair(device, root, config.block_size) {
            Ok(v) => v,
            Err(_) => return false,
        };
        for t in &view.tags {
            if t.is_name() && t.typ == TAG_NAME_SUPERBLOCK && t.id == 0 && t.data.len() >= 8 {
                let mut m = [0u8; 8];
                m.copy_from_slice(&t.data[0..8]);
                if m == LFS2_SUPERBLOCK_MAGIC {
                    return true;
                }
            }
        }
        false
    }

    /// Format `device` as a fresh, empty littlefs v2 image.
    ///
    /// Writes a brand-new root metadata-pair at blocks 0 and 1 containing
    /// only the superblock entry. The two halves are written with
    /// successive revisions (0 in `blocks[0]`, 1 in `blocks[1]`) so the
    /// commit-half-selection rule picks `blocks[1]` as the active half on
    /// the subsequent mount; this matches the post-mkfs state the
    /// upstream `mklittlefs` tool produces.
    ///
    /// **Power-fail safety.** If the host loses power partway through
    /// `format`, the medium is left in a partially-formatted state that
    /// SHALL not satisfy [`Self::is_formatted`] (or, in the rare case
    /// only one half made it to flash, will satisfy it with a single
    /// committed half). On the next boot, the kernel SHALL re-run
    /// format-on-presence rather than try to mount the partial image.
    ///
    /// Caller MUST hold a `PhysicalPresenceProvider::asserted() == true`
    /// signal (or equivalent operator gesture) before invoking this; the
    /// VFS-mount sequencer handles the gating.
    pub fn format(device: &mut D, config: LittleFsConfig) -> Result<(), LittleFsError> {
        // Erase the two root-pair blocks.
        device.erase(0).map_err(|_| LittleFsError::Io)?;
        device.erase(1).map_err(|_| LittleFsError::Io)?;

        // Build the superblock entry payload: u32 version (major.minor),
        // u32 block_size, u32 block_count, u32 name_max, u32 file_max,
        // u32 attr_max — matches the test-fixture layout and the v2 SPEC.
        // Encoded as `(major << 16) | minor`; v2.0 → minor = 0.
        let version: u32 = u32::from(LFS2_MAJOR_VERSION) << 16;
        let mut sb_payload = Vec::with_capacity(24);
        sb_payload.extend_from_slice(&version.to_le_bytes());
        sb_payload.extend_from_slice(&config.block_size.to_le_bytes());
        sb_payload.extend_from_slice(&config.block_count.to_le_bytes());
        sb_payload.extend_from_slice(&255u32.to_le_bytes()); // name_max
        sb_payload.extend_from_slice(&0x7FFF_FFFFu32.to_le_bytes()); // file_max
        sb_payload.extend_from_slice(&1022u32.to_le_bytes()); // attr_max

        let entry = CommitEntry {
            typ_name: TAG_NAME_SUPERBLOCK,
            id: 0,
            name: b"littlefs".to_vec(),
            typ_struct: TAG_STRUCT_INLINE,
            struct_payload: sb_payload,
        };
        let entries = [entry];

        // Program both halves. Use a pad-up to prog_size on each side.
        let block_size = config.block_size as usize;
        let prog_size = config.prog_size.max(1) as usize;
        for (block, revision) in [(0u32, 0u32), (1u32, 1u32)] {
            let mut bytes = build_metadata_half(revision, &entries, None)
                .map_err(|_| LittleFsError::Unsupported)?;
            if bytes.len() > block_size {
                return Err(LittleFsError::Unsupported);
            }
            // Pad up to a prog_size multiple with `0xFF` (erased state),
            // matching what `commit_pair` does on subsequent writes.
            let pad_to = bytes.len().div_ceil(prog_size) * prog_size;
            bytes.resize(pad_to, 0xFF);
            let off = u64::from(block) * (block_size as u64);
            device.program(off, &bytes).map_err(|_| LittleFsError::Io)?;
        }
        Ok(())
    }

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
            mutable: None,
            fadvise: FadviseHint::default(),
        })
    }

    /// Apply a `fadvise` hint to this mount. v1 records the hint for
    /// inspection by the tests and any future cache layer; no behavioral
    /// change is observable today (POSIX permits a hint to be silently
    /// ignored).
    ///
    /// Phase 3 wires the API end-to-end so a `posix_fadvise` syscall on
    /// a `/flash/`-backed file descriptor round-trips cleanly through
    /// the VFS into the littlefs runtime.
    pub fn fadvise(&mut self, hint: FadviseHint) {
        self.fadvise = hint;
    }

    /// Currently active `fadvise` hint (test introspection).
    pub fn fadvise_hint(&self) -> FadviseHint {
        self.fadvise
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

    // ─── Phase 2 write path ─────────────────────────────────────────────────

    /// Lazily build the [`MutableState`] (block allocator + BBT + per-pair
    /// in-memory entries) needed for write operations.
    fn ensure_mutable(&mut self) -> Result<(), LittleFsError> {
        if self.mutable.is_some() {
            return Ok(());
        }
        let bbt = BadBlockTable::from_device(self.device);
        // Reserved blocks: root pair is always 0..1.
        let mut reserved = vec![self.root[0], self.root[1]];
        // Walk every metadata pair we know about and reconstruct an
        // in-memory PairState. Phase 2 supports root + one level of
        // sub-directories.
        let root_view = read_metadata_pair(self.device, self.root, self.config.block_size)?;
        let root_entries = entries_from_view(&root_view);
        // Track which half of the root pair held the active commit so
        // the next commit can use the other half.
        let mut buf_a = vec![0u8; self.config.block_size as usize];
        let mut buf_b = vec![0u8; self.config.block_size as usize];
        self.device
            .read(
                u64::from(self.root[0]) * u64::from(self.config.block_size),
                &mut buf_a,
            )
            .map_err(|_| LittleFsError::Io)?;
        self.device
            .read(
                u64::from(self.root[1]) * u64::from(self.config.block_size),
                &mut buf_b,
            )
            .map_err(|_| LittleFsError::Io)?;
        let view_a = parse_metadata_half(&buf_a);
        let view_b = parse_metadata_half(&buf_b);
        let (root_active_half, root_revision) =
            select_active_half(view_a.valid, view_a.revision, view_b.valid, view_b.revision);
        let root_tail = root_view.tail;
        // Reserve any blocks referenced by sub-dir or CTZ-file struct
        // tags so the allocator never re-uses them.
        struct PendingSubdir {
            blocks: [u32; 2],
            revision: u32,
            active_half: u8,
            entries: Vec<PairEntry>,
            tail: Option<[u32; 2]>,
        }
        let mut subdir_entries: Vec<PendingSubdir> = Vec::new();
        let mut subdir_pairs_pending: Vec<[u32; 2]> = Vec::new();
        for e in &root_entries {
            match e.typ_struct {
                t if t == TAG_STRUCT_DIR && e.struct_payload.len() == 8 => {
                    let a = read_u32_le(&e.struct_payload[0..4]);
                    let b = read_u32_le(&e.struct_payload[4..8]);
                    reserved.push(a);
                    reserved.push(b);
                    subdir_pairs_pending.push([a, b]);
                }
                t if t == TAG_STRUCT_CTZ && e.struct_payload.len() >= 8 => {
                    let head = read_u32_le(&e.struct_payload[0..4]);
                    let size = read_u32_le(&e.struct_payload[4..8]);
                    // Walk the CTZ chain (linear walk-back) to mark
                    // every block reserved.
                    let mut blocks_in_chain = Vec::new();
                    let chain_len = ctz_chain_length(self.config.block_size, size);
                    let mut cur = head;
                    let mut idx = chain_len.saturating_sub(1);
                    blocks_in_chain.push(cur);
                    while idx > 0 {
                        let off = u64::from(cur) * u64::from(self.config.block_size)
                            + u64::from(self.config.block_size - 4);
                        let mut pb = [0u8; 4];
                        self.device
                            .read(off, &mut pb)
                            .map_err(|_| LittleFsError::Io)?;
                        let prev = read_u32_le(&pb);
                        blocks_in_chain.push(prev);
                        cur = prev;
                        idx -= 1;
                    }
                    for blk in blocks_in_chain {
                        reserved.push(blk);
                    }
                }
                _ => {}
            }
        }
        // Build allocator with reserved blocks + bad blocks masked.
        let mut allocator = BlockAllocator::new(
            self.config.block_count,
            self.config.block_cycles.max(1) as u32,
            self.config.lookahead_size,
            &reserved,
        );
        for b in bbt.iter_bad() {
            allocator.mark_bad(b);
        }
        // Now load each subdir pair and add to state.
        for blocks in subdir_pairs_pending {
            let view = read_metadata_pair(self.device, blocks, self.config.block_size)?;
            let entries = entries_from_view(&view);
            let mut sub_a = vec![0u8; self.config.block_size as usize];
            let mut sub_b = vec![0u8; self.config.block_size as usize];
            self.device
                .read(
                    u64::from(blocks[0]) * u64::from(self.config.block_size),
                    &mut sub_a,
                )
                .map_err(|_| LittleFsError::Io)?;
            self.device
                .read(
                    u64::from(blocks[1]) * u64::from(self.config.block_size),
                    &mut sub_b,
                )
                .map_err(|_| LittleFsError::Io)?;
            let v_a = parse_metadata_half(&sub_a);
            let v_b = parse_metadata_half(&sub_b);
            let (active_half, revision) =
                select_active_half(v_a.valid, v_a.revision, v_b.valid, v_b.revision);
            subdir_entries.push(PendingSubdir {
                blocks,
                revision,
                active_half,
                entries,
                tail: view.tail,
            });
        }

        let mut state = MutableState {
            config: self.config,
            allocator,
            bbt,
            root: self.root,
            pairs: Vec::new(),
            commits: 0,
        };
        state.pairs.push(PairState {
            blocks: self.root,
            revision: root_revision,
            active_half: root_active_half,
            entries: root_entries,
            tail: root_tail,
            dirty: false,
        });
        for ps in subdir_entries {
            let PendingSubdir {
                blocks,
                revision,
                active_half,
                entries,
                tail,
            } = ps;
            state.pairs.push(PairState {
                blocks,
                revision,
                active_half,
                entries,
                tail,
                dirty: false,
            });
        }
        self.mutable = Some(state);
        Ok(())
    }

    /// Create a new empty regular file at `path`. Fails with
    /// `BadType` if a file/dir of that name already exists at the
    /// parent.
    pub fn create(&mut self, path: &str) -> Result<(), LittleFsError> {
        self.ensure_mutable()?;
        let (parent_pair_addr, name) = self.resolve_parent(path)?;
        let name_owned = name.to_string();
        let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
        let pair_idx = state
            .pairs
            .iter()
            .position(|p| p.blocks == parent_pair_addr)
            .ok_or(LittleFsError::NotFound)?;
        // Reject if any entry with this name already exists.
        if state
            .find_entry(pair_idx, &name_owned, TAG_NAME_REG)
            .is_some()
            || state
                .find_entry(pair_idx, &name_owned, TAG_NAME_DIR)
                .is_some()
        {
            return Err(LittleFsError::BadType);
        }
        let id = state.fresh_id(pair_idx)?;
        state.pairs[pair_idx].entries.push(PairEntry {
            id,
            typ_name: TAG_NAME_REG,
            name: name_owned.into_bytes(),
            typ_struct: TAG_STRUCT_INLINE,
            struct_payload: Vec::new(),
        });
        state.mark_dirty(pair_idx);
        state.commit_pair(self.device, pair_idx)?;
        Ok(())
    }

    /// Open `path` for writing. With [`OpenFlags::create_truncate`] (the
    /// default) creates the file if missing and truncates it to length 0.
    /// Returns a handle whose [`FileMut::write`] / [`FileMut::flush`] /
    /// [`FileMut::close`] methods materialise bytes into the file.
    pub fn open_write(&mut self, path: &str, flags: OpenFlags) -> Result<FileMut, LittleFsError> {
        self.ensure_mutable()?;
        let (parent_pair_addr, name) = self.resolve_parent(path)?;
        let name_owned = name.to_string();
        let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
        let pair_idx = state
            .pairs
            .iter()
            .position(|p| p.blocks == parent_pair_addr)
            .ok_or(LittleFsError::NotFound)?;
        let existing = state.find_entry(pair_idx, &name_owned, TAG_NAME_REG);
        let id = match existing {
            Some(idx) => {
                if flags.truncate {
                    // Zero-out the inline payload (or convert from CTZ).
                    let entry = &mut state.pairs[pair_idx].entries[idx];
                    entry.typ_struct = TAG_STRUCT_INLINE;
                    entry.struct_payload = Vec::new();
                    state.mark_dirty(pair_idx);
                }
                state.pairs[pair_idx].entries[idx].id
            }
            None => {
                if !flags.create {
                    return Err(LittleFsError::NotFound);
                }
                let id = state.fresh_id(pair_idx)?;
                state.pairs[pair_idx].entries.push(PairEntry {
                    id,
                    typ_name: TAG_NAME_REG,
                    name: name_owned.into_bytes(),
                    typ_struct: TAG_STRUCT_INLINE,
                    struct_payload: Vec::new(),
                });
                state.mark_dirty(pair_idx);
                id
            }
        };
        Ok(FileMut {
            pair_idx,
            id,
            buf: Vec::new(),
            append: flags.append,
            closed: false,
        })
    }

    /// Materialise the bytes accumulated in `file.buf` into the
    /// underlying entry. Inline files store the buffer directly inside
    /// the metadata-pair struct tag; once the buffer's size exceeds
    /// 1/8 of the cache size (per SPEC.md inline-file ceiling), the
    /// data is promoted to a CTZ skip-list.
    fn flush_filemut(&mut self, file: &mut FileMut) -> Result<(), LittleFsError> {
        let block_size = self.config.block_size;
        let cache_size = self.config.cache_size.max(block_size);
        let inline_ceiling = (cache_size / 8).max(64);
        // Collect the data we need to write before borrowing state.
        let new_buf = core::mem::take(&mut file.buf);
        let entry_pair_idx = file.pair_idx;
        let entry_id = file.id;
        // Determine current entry state and (size, append flag) without
        // holding a borrow on state across the CTZ build.
        // Two-phase: extract identifying info from state under one
        // borrow, then re-borrow self for the CTZ read.
        let (was_ctz, prev_inline_payload, prev_ctz): (bool, Option<Vec<u8>>, Option<(u32, u32)>) = {
            let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
            let entry_idx = state.pairs[entry_pair_idx]
                .entries
                .iter()
                .position(|e| e.id == entry_id)
                .ok_or(LittleFsError::NotFound)?;
            let entry = &state.pairs[entry_pair_idx].entries[entry_idx];
            if entry.typ_struct == TAG_STRUCT_CTZ && entry.struct_payload.len() >= 8 {
                let head = read_u32_le(&entry.struct_payload[0..4]);
                let size = read_u32_le(&entry.struct_payload[4..8]);
                (true, None, Some((head, size)))
            } else {
                (false, Some(entry.struct_payload.clone()), None)
            }
        };
        let mut combined: Vec<u8> = if file.append {
            if was_ctz {
                let (head, size) = prev_ctz.expect("ctz_old_head_size set above");
                let mut cur = vec![0u8; size as usize];
                self.read_ctz_at(head, size, 0, &mut cur)?;
                cur
            } else {
                prev_inline_payload.clone().unwrap_or_default()
            }
        } else {
            Vec::new()
        };
        combined.extend_from_slice(&new_buf);
        let ctz_old_head_size = prev_ctz;
        let _ = prev_inline_payload;

        let new_size = combined.len() as u32;
        if (combined.len() as u32) <= inline_ceiling && !was_ctz {
            // Stay inline.
            let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
            let entry_idx = state.pairs[entry_pair_idx]
                .entries
                .iter()
                .position(|e| e.id == entry_id)
                .ok_or(LittleFsError::NotFound)?;
            state.pairs[entry_pair_idx].entries[entry_idx].typ_struct = TAG_STRUCT_INLINE;
            state.pairs[entry_pair_idx].entries[entry_idx].struct_payload = combined;
            state.mark_dirty(entry_pair_idx);
            return Ok(());
        }

        // Promote / extend a CTZ chain. Phase 2 implements the simple
        // "single-block-per-logical-index, by-1 pointer at end of block"
        // layout the reader walks. We always rewrite the full chain
        // from scratch (rather than incrementally appending) to keep
        // the implementation simple and to ensure each block is in
        // its erased state before programming.
        // Free any old chain blocks back to the allocator first.
        if let Some((old_head, old_size)) = ctz_old_head_size {
            let mut to_free = Vec::new();
            let chain_len = ctz_chain_length(block_size, old_size);
            let mut cur = old_head;
            let mut idx = chain_len.saturating_sub(1);
            to_free.push(cur);
            while idx > 0 {
                let off = u64::from(cur) * u64::from(block_size) + u64::from(block_size - 4);
                let mut pb = [0u8; 4];
                self.device
                    .read(off, &mut pb)
                    .map_err(|_| LittleFsError::Io)?;
                let prev = read_u32_le(&pb);
                to_free.push(prev);
                cur = prev;
                idx -= 1;
            }
            let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
            for b in to_free {
                state.allocator.mark_free(b);
            }
        }

        // Allocate, erase, and program each block in the chain.
        let chain_len = ctz_chain_length(block_size, combined.len() as u32);
        let mut chain_blocks: Vec<u32> = Vec::with_capacity(chain_len as usize);
        for _ in 0..chain_len {
            let blk = {
                let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
                state
                    .allocator
                    .alloc_block()
                    .map_err(|_| LittleFsError::Io)?
            };
            chain_blocks.push(blk);
        }
        // Erase + program logical block at `i`.
        let mut data_off: usize = 0;
        let chain_count = chain_blocks.len();
        for i in 0..chain_count {
            let payload_size = ctz_payload(block_size, i as u64) as usize;
            let mut block_buf = vec![0xFFu8; block_size as usize];
            let take = payload_size.min(combined.len().saturating_sub(data_off));
            block_buf[..take].copy_from_slice(&combined[data_off..data_off + take]);
            data_off += take;
            // Write the by-1 pointer at the end of the block (only for
            // logical index > 0).
            if i > 0 {
                let prev_phys = chain_blocks[i - 1];
                let off = block_size as usize - 4;
                block_buf[off..off + 4].copy_from_slice(&prev_phys.to_le_bytes());
            }
            // Try erase + program with bad-block remap.
            let phys = chain_blocks[i];
            match self.write_full_block(phys, &block_buf) {
                Ok(()) => {}
                Err(LittleFsError::Io) => {
                    // The block went bad mid-write. Allocate another,
                    // update chain_blocks[i].
                    let replacement = {
                        let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
                        state
                            .allocator
                            .alloc_block()
                            .map_err(|_| LittleFsError::Io)?
                    };
                    chain_blocks[i] = replacement;
                    self.write_full_block(replacement, &block_buf)?;
                }
                Err(other) => return Err(other),
            }
        }
        let head = *chain_blocks.last().ok_or(LittleFsError::Unsupported)?;
        // Update the parent metadata-pair entry to point at the new
        // CTZ chain.
        {
            let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
            let entry_idx = state.pairs[entry_pair_idx]
                .entries
                .iter()
                .position(|e| e.id == entry_id)
                .ok_or(LittleFsError::NotFound)?;
            let mut payload = Vec::with_capacity(8);
            payload.extend_from_slice(&head.to_le_bytes());
            payload.extend_from_slice(&new_size.to_le_bytes());
            state.pairs[entry_pair_idx].entries[entry_idx].typ_struct = TAG_STRUCT_CTZ;
            state.pairs[entry_pair_idx].entries[entry_idx].struct_payload = payload;
            state.mark_dirty(entry_pair_idx);
        }
        Ok(())
    }

    /// Erase a block, then program a fully-formed block buffer to it.
    /// Surfaces `LittleFsError::Io` if the block goes bad (so the
    /// caller can re-allocate); other I/O errors bubble up.
    fn write_full_block(&mut self, block: u32, data: &[u8]) -> Result<(), LittleFsError> {
        match self.device.erase(u64::from(block)) {
            Ok(()) => {
                if let Some(state) = self.mutable.as_mut() {
                    state.allocator.note_erase(block);
                }
            }
            Err(FlashError::EraseFailure | FlashError::ProgramFailed) => {
                if let Some(state) = self.mutable.as_mut() {
                    state.bbt.mark_bad(block);
                    state.allocator.mark_bad(block);
                }
                let _ = self.device.mark_bad(u64::from(block));
                return Err(LittleFsError::Io);
            }
            Err(_) => return Err(LittleFsError::Io),
        }
        let prog_size = self.config.prog_size.max(1) as usize;
        let mut padded = data.to_vec();
        let pad_to = padded.len().div_ceil(prog_size) * prog_size;
        padded.resize(pad_to, 0xFF);
        let off = u64::from(block) * u64::from(self.config.block_size);
        match self.device.program(off, &padded) {
            Ok(()) => Ok(()),
            Err(FlashError::ProgramFailed | FlashError::ProgramOnDirty) => {
                if let Some(state) = self.mutable.as_mut() {
                    state.bbt.mark_bad(block);
                    state.allocator.mark_bad(block);
                }
                let _ = self.device.mark_bad(u64::from(block));
                Err(LittleFsError::Io)
            }
            Err(_) => Err(LittleFsError::Io),
        }
    }

    /// Append `bytes` to an open [`FileMut`].
    pub fn write(&mut self, file: &mut FileMut, bytes: &[u8]) -> Result<usize, LittleFsError> {
        if file.closed {
            return Err(LittleFsError::Io);
        }
        file.buf.extend_from_slice(bytes);
        // Keep the buffer bounded; flush partway if it grows past
        // 4 * block_size to avoid unbounded RAM growth.
        let block_size = self.config.block_size as usize;
        if file.buf.len() >= block_size * 4 {
            self.flush_filemut(file)?;
            file.append = true;
        }
        Ok(bytes.len())
    }

    /// Flush in-memory bytes to flash without committing the
    /// metadata pair. Returns once the file's data blocks are durably
    /// programmed; the parent metadata pair becomes durable only after
    /// the next [`fsync`](Self::fsync) or implicit commit.
    pub fn flush(&mut self, file: &mut FileMut) -> Result<(), LittleFsError> {
        self.flush_filemut(file)
    }

    /// Close `file` and commit the parent metadata pair. After this
    /// returns, the file's bytes are durable across power loss.
    pub fn close(&mut self, mut file: FileMut) -> Result<(), LittleFsError> {
        if file.closed {
            return Ok(());
        }
        self.flush_filemut(&mut file)?;
        file.closed = true;
        let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
        state.commit_pair(self.device, file.pair_idx)?;
        Ok(())
    }

    /// Force every dirty in-memory metadata-pair to flash. After a
    /// successful return, all data written via [`write`](Self::write) +
    /// [`flush`](Self::flush) is durable across power loss.
    pub fn fsync(&mut self) -> Result<(), LittleFsError> {
        if let Some(state) = self.mutable.as_mut() {
            state.commit_all(self.device)?;
        }
        Ok(())
    }

    /// Remove a regular file at `path`.
    pub fn remove(&mut self, path: &str) -> Result<(), LittleFsError> {
        self.ensure_mutable()?;
        let (parent_pair_addr, name) = self.resolve_parent(path)?;
        let name_owned = name.to_string();
        let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
        let pair_idx = state
            .pairs
            .iter()
            .position(|p| p.blocks == parent_pair_addr)
            .ok_or(LittleFsError::NotFound)?;
        let entry_idx = state
            .find_entry(pair_idx, &name_owned, TAG_NAME_REG)
            .ok_or(LittleFsError::NotFound)?;
        // If the entry is CTZ-backed, return its blocks to the
        // allocator pool.
        let entry = &state.pairs[pair_idx].entries[entry_idx];
        if entry.typ_struct == TAG_STRUCT_CTZ && entry.struct_payload.len() >= 8 {
            let head = read_u32_le(&entry.struct_payload[0..4]);
            let size = read_u32_le(&entry.struct_payload[4..8]);
            let block_size = self.config.block_size;
            let _ = entry;
            let mut to_free = Vec::new();
            let chain_len = ctz_chain_length(block_size, size);
            let mut cur = head;
            let mut idx = chain_len.saturating_sub(1);
            to_free.push(cur);
            while idx > 0 {
                let off = u64::from(cur) * u64::from(block_size) + u64::from(block_size - 4);
                let mut pb = [0u8; 4];
                self.device
                    .read(off, &mut pb)
                    .map_err(|_| LittleFsError::Io)?;
                let prev = read_u32_le(&pb);
                to_free.push(prev);
                cur = prev;
                idx -= 1;
            }
            let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
            for b in to_free {
                state.allocator.mark_free(b);
            }
        }
        let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
        state.pairs[pair_idx].entries.remove(entry_idx);
        state.mark_dirty(pair_idx);
        state.commit_pair(self.device, pair_idx)?;
        Ok(())
    }

    /// Atomic rename. The destination must not already exist (Phase 2
    /// surface: use [`Self::remove`] first if you want overwrite).
    /// The mutation runs as one metadata-pair commit, giving the
    /// littlefs power-fail safety: on crash recovery either the
    /// rename is fully visible or the previous state is intact.
    pub fn rename(&mut self, src: &str, dst: &str) -> Result<(), LittleFsError> {
        self.ensure_mutable()?;
        let (src_pair_addr, src_name) = self.resolve_parent(src)?;
        let (dst_pair_addr, dst_name) = self.resolve_parent(dst)?;
        if src_pair_addr != dst_pair_addr {
            // Cross-pair rename requires writing both pairs; Phase 2
            // surfaces it as `Unsupported`.
            return Err(LittleFsError::Unsupported);
        }
        let src_name = src_name.to_string();
        let dst_name = dst_name.to_string();
        let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
        let pair_idx = state
            .pairs
            .iter()
            .position(|p| p.blocks == src_pair_addr)
            .ok_or(LittleFsError::NotFound)?;
        // Source must exist as a file or directory.
        let src_idx = state
            .find_entry(pair_idx, &src_name, TAG_NAME_REG)
            .or_else(|| state.find_entry(pair_idx, &src_name, TAG_NAME_DIR))
            .ok_or(LittleFsError::NotFound)?;
        // Destination must not exist.
        if state
            .find_entry(pair_idx, &dst_name, TAG_NAME_REG)
            .is_some()
            || state
                .find_entry(pair_idx, &dst_name, TAG_NAME_DIR)
                .is_some()
        {
            return Err(LittleFsError::BadType);
        }
        state.pairs[pair_idx].entries[src_idx].name = dst_name.into_bytes();
        state.mark_dirty(pair_idx);
        state.commit_pair(self.device, pair_idx)?;
        Ok(())
    }

    /// Truncate `path` to `new_size` bytes. Phase 2 supports only
    /// `new_size <= current_size` (cannot grow the file via truncate).
    pub fn truncate(&mut self, path: &str, new_size: u32) -> Result<(), LittleFsError> {
        self.ensure_mutable()?;
        let (parent_pair_addr, name) = self.resolve_parent(path)?;
        let name_owned = name.to_string();
        // Read the current contents into memory, then rewrite.
        let cur_data = {
            let mut f = self.open_read(path)?;
            let cur_size = f.size();
            if new_size > cur_size {
                return Err(LittleFsError::Unsupported);
            }
            let mut buf = vec![0u8; cur_size as usize];
            self.read_file(&mut f, &mut buf)?;
            buf.truncate(new_size as usize);
            buf
        };
        let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
        let pair_idx = state
            .pairs
            .iter()
            .position(|p| p.blocks == parent_pair_addr)
            .ok_or(LittleFsError::NotFound)?;
        let entry_idx = state
            .find_entry(pair_idx, &name_owned, TAG_NAME_REG)
            .ok_or(LittleFsError::NotFound)?;
        // Free any old CTZ chain blocks.
        let prev_payload = state.pairs[pair_idx].entries[entry_idx]
            .struct_payload
            .clone();
        let prev_typ = state.pairs[pair_idx].entries[entry_idx].typ_struct;
        if prev_typ == TAG_STRUCT_CTZ && prev_payload.len() >= 8 {
            let old_head = read_u32_le(&prev_payload[0..4]);
            let old_size = read_u32_le(&prev_payload[4..8]);
            let block_size = self.config.block_size;
            let mut to_free = Vec::new();
            let chain_len = ctz_chain_length(block_size, old_size);
            let mut cur = old_head;
            let mut idx = chain_len.saturating_sub(1);
            to_free.push(cur);
            while idx > 0 {
                let off = u64::from(cur) * u64::from(block_size) + u64::from(block_size - 4);
                let mut pb = [0u8; 4];
                self.device
                    .read(off, &mut pb)
                    .map_err(|_| LittleFsError::Io)?;
                let prev = read_u32_le(&pb);
                to_free.push(prev);
                cur = prev;
                idx -= 1;
            }
            let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
            for b in to_free {
                state.allocator.mark_free(b);
            }
        }
        // Reset the entry's payload to inline-empty.
        let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
        state.pairs[pair_idx].entries[entry_idx].typ_struct = TAG_STRUCT_INLINE;
        state.pairs[pair_idx].entries[entry_idx].struct_payload = Vec::new();
        state.mark_dirty(pair_idx);
        // Commit the metadata pair so the entry is now empty inline.
        state.commit_pair(self.device, pair_idx)?;
        // If new_size > 0, write back through the normal write path.
        if !cur_data.is_empty() {
            let mut f = self.open_write(
                path,
                OpenFlags {
                    create: false,
                    truncate: false,
                    append: true,
                },
            )?;
            self.write(&mut f, &cur_data)?;
            self.close(f)?;
        }
        Ok(())
    }

    /// Create a sub-directory at `path`. Path must be one level deep
    /// (the Phase 1 reader supports only single-level directory
    /// nesting; deeper hierarchies surface as `Unsupported`).
    pub fn mkdir(&mut self, path: &str) -> Result<(), LittleFsError> {
        self.ensure_mutable()?;
        let (parent_pair_addr, name) = self.resolve_parent(path)?;
        if parent_pair_addr != self.root {
            return Err(LittleFsError::Unsupported);
        }
        let name_owned = name.to_string();
        let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
        let pair_idx = state
            .pairs
            .iter()
            .position(|p| p.blocks == parent_pair_addr)
            .ok_or(LittleFsError::NotFound)?;
        if state
            .find_entry(pair_idx, &name_owned, TAG_NAME_REG)
            .is_some()
            || state
                .find_entry(pair_idx, &name_owned, TAG_NAME_DIR)
                .is_some()
        {
            return Err(LittleFsError::BadType);
        }
        // Allocate two blocks for the new directory's metadata pair.
        let dir_a = state
            .allocator
            .alloc_block()
            .map_err(|_| LittleFsError::Io)?;
        let dir_b = state
            .allocator
            .alloc_block()
            .map_err(|_| LittleFsError::Io)?;
        let dir_blocks = [dir_a, dir_b];
        let id = state.fresh_id(pair_idx)?;
        let mut payload = Vec::with_capacity(8);
        payload.extend_from_slice(&dir_a.to_le_bytes());
        payload.extend_from_slice(&dir_b.to_le_bytes());
        state.pairs[pair_idx].entries.push(PairEntry {
            id,
            typ_name: TAG_NAME_DIR,
            name: name_owned.into_bytes(),
            typ_struct: TAG_STRUCT_DIR,
            struct_payload: payload,
        });
        state.mark_dirty(pair_idx);
        // Push a fresh empty directory pair-state.
        state.pairs.push(PairState {
            blocks: dir_blocks,
            revision: 0,
            active_half: 1, // so first commit lands in half[0].
            entries: Vec::new(),
            tail: None,
            dirty: true,
        });
        let new_pair_idx = state.pairs.len() - 1;
        // Commit the new (empty) directory pair first, then the
        // parent. Order matters for crash recovery: if the parent
        // commit succeeds before the child, a reader following the
        // pair pointer would see uninitialised flash.
        state.commit_pair(self.device, new_pair_idx)?;
        state.commit_pair(self.device, pair_idx)?;
        Ok(())
    }

    /// Remove an empty sub-directory at `path`.
    pub fn rmdir(&mut self, path: &str) -> Result<(), LittleFsError> {
        self.ensure_mutable()?;
        let (parent_pair_addr, name) = self.resolve_parent(path)?;
        if parent_pair_addr != self.root {
            return Err(LittleFsError::Unsupported);
        }
        let name_owned = name.to_string();
        let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
        let pair_idx = state
            .pairs
            .iter()
            .position(|p| p.blocks == parent_pair_addr)
            .ok_or(LittleFsError::NotFound)?;
        let entry_idx = state
            .find_entry(pair_idx, &name_owned, TAG_NAME_DIR)
            .ok_or(LittleFsError::NotFound)?;
        let entry = &state.pairs[pair_idx].entries[entry_idx];
        if entry.struct_payload.len() < 8 {
            return Err(LittleFsError::Corrupted);
        }
        let a = read_u32_le(&entry.struct_payload[0..4]);
        let b = read_u32_le(&entry.struct_payload[4..8]);
        let dir_blocks = [a, b];
        // Verify the dir is empty.
        let dir_idx = state
            .pairs
            .iter()
            .position(|p| p.blocks == dir_blocks)
            .ok_or(LittleFsError::NotFound)?;
        if !state.pairs[dir_idx].entries.is_empty() {
            return Err(LittleFsError::BadType);
        }
        // Free both pair blocks back to the allocator and drop the
        // sub-pair state.
        state.allocator.mark_free(a);
        state.allocator.mark_free(b);
        state.pairs.remove(dir_idx);
        let state = self.mutable.as_mut().ok_or(LittleFsError::Io)?;
        // After removal, the pair_idx index for the parent might still
        // be valid if it was lower than dir_idx — recompute it.
        let pair_idx = state
            .pairs
            .iter()
            .position(|p| p.blocks == parent_pair_addr)
            .ok_or(LittleFsError::NotFound)?;
        let entry_idx = state
            .find_entry(pair_idx, &name_owned, TAG_NAME_DIR)
            .ok_or(LittleFsError::NotFound)?;
        state.pairs[pair_idx].entries.remove(entry_idx);
        state.mark_dirty(pair_idx);
        state.commit_pair(self.device, pair_idx)?;
        Ok(())
    }
}

/// Compute the CTZ skip-list chain length for a file of `size` bytes
/// laid out with `block_size`-byte blocks.
fn ctz_chain_length(block_size: u32, size: u32) -> u64 {
    if size == 0 {
        return 1;
    }
    let mut total: u64 = 0;
    let mut idx: u64 = 0;
    loop {
        let payload = ctz_payload(block_size, idx) as u64;
        if total + payload >= u64::from(size) {
            return idx + 1;
        }
        total += payload;
        idx += 1;
    }
}

/// Convert a [`MetadataPairView`] (read-side type) into a `Vec` of
/// `PairEntry` (write-side type).
///
/// Pairs entries by id: every name tag is paired with the corresponding
/// struct tag of the same id (a v2 metadata-pair invariant). Tags that
/// don't pair (name without struct, struct without name, or the
/// trailing CRC) are dropped.
fn entries_from_view(view: &MetadataPairView) -> Vec<PairEntry> {
    use alloc::collections::BTreeMap;
    let mut by_id: BTreeMap<u16, (Option<&Tag>, Option<&Tag>)> = BTreeMap::new();
    for t in &view.tags {
        let slot = by_id.entry(t.id).or_insert((None, None));
        if t.is_name() {
            slot.0 = Some(t);
        } else if t.is_struct() {
            slot.1 = Some(t);
        }
    }
    let mut out = Vec::new();
    for (id, (n, s)) in by_id {
        if let (Some(n), Some(s)) = (n, s) {
            // Filter superblock entries — they are emitted via `mkfs`,
            // not via the per-mutation commit path.
            out.push(PairEntry {
                id,
                typ_name: n.typ,
                name: n.data.clone(),
                typ_struct: s.typ,
                struct_payload: s.data.clone(),
            });
        }
    }
    out
}

/// Pick the active half of a pair given each half's validity and
/// revision counters.
///
/// Returns `(active_half_index, active_revision)`.
fn select_active_half(va: bool, ra: u32, vb: bool, rb: u32) -> (u8, u32) {
    match (va, vb) {
        (true, true) => {
            if rb > ra {
                (1, rb)
            } else {
                (0, ra)
            }
        }
        (true, false) => (0, ra),
        (false, true) => (1, rb),
        (false, false) => (0, 0),
    }
}

/// Write-path file handle.
///
/// Phase 2 keeps this simple: the handle records the (pair_idx, id) of
/// the underlying entry plus an in-memory write buffer. Mutations
/// flow through [`LittleFs::write`] / [`LittleFs::flush`] /
/// [`LittleFs::close`] so the borrow checker is comfortable with
/// successive `&mut LittleFs` calls.
pub struct FileMut {
    pair_idx: usize,
    id: u16,
    buf: Vec<u8>,
    append: bool,
    closed: bool,
}

impl FileMut {
    /// Pair index in the in-memory mount state (test introspection).
    #[doc(hidden)]
    pub fn _pair_idx(&self) -> usize {
        self.pair_idx
    }
    /// Entry id (test introspection).
    #[doc(hidden)]
    pub fn _id(&self) -> u16 {
        self.id
    }
    /// Bytes still buffered in memory (test introspection).
    pub fn buffered(&self) -> usize {
        self.buf.len()
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
