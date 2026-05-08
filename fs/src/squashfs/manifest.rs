// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS squashfs manifest footer parser + ML-DSA-65 verifier.
//!
//! A SmallAIOS-produced squashfs partition has, after the natural EOF
//! of the squashfs image, a sealed manifest footer:
//!
//! ```text
//!   +------------------------+
//!   | magic = "SmAIOSFS\0"   |    9 bytes
//!   +------------------------+
//!   | version (= 1)          |    1 byte
//!   +------------------------+
//!   | hash_count (u32 LE)    |    4 bytes
//!   +------------------------+
//!   | hash[0..32 * count]    |    32 * count bytes (SHA-3-256 per 4 KiB)
//!   +------------------------+
//!   | pubkey_fpr[0..32]      |    32 bytes (SHA-3-256 of pubkey)
//!   +------------------------+
//!   | sig_len (u16 LE)       |    2 bytes
//!   +------------------------+
//!   | signature              |    sig_len bytes (ML-DSA-65)
//!   +------------------------+
//!   | pad to 4-byte align    |   0..3 bytes (zeros)
//!   +------------------------+
//!   | footer_length (u32 LE) |    4 bytes (size of THIS whole footer)
//!   +------------------------+
//! ```
//!
//! The footer is read first by SmallAIOS so the kernel can refuse the
//! mount before any data block is honored. External `mount -t squashfs
//! -o loop` ignores the trailing bytes (squashfs tolerates 4-byte-
//! aligned padding), so this footer does NOT break stock Linux loop-
//! back mounts.
//!
//! The signature covers the 32-byte hash array (concatenated, length =
//! `hash_count * 32` bytes). The kernel verifies the signature once at
//! mount; thereafter every block read goes through SHA-3-256 against
//! the corresponding entry in the array.

extern crate alloc;

use alloc::vec::Vec;

use smallaios_security::crypto::ml_dsa::{
    ml_dsa_65_verify, MlDsaPublicKey, MlDsaSignature, ML_DSA_65_PK_LEN,
};
use smallaios_security::crypto::sha3::{sha3_256, Sha3_256};

use super::SquashfsError;

/// Magic bytes at the start of the SmallAIOS manifest footer.
//
// lgtm[rust/hard-coded-cryptographic-value] — SmallAIOS manifest footer magic, public on-disk constant
pub const MANIFEST_MAGIC: [u8; 9] = *b"SmAIOSFS\0";

/// Size of one block-hash entry (SHA-3-256 = 32 bytes).
pub const BLOCK_HASH_LEN: usize = 32;

/// Logical block size used for the per-block hash array (4 KiB).
pub const HASH_BLOCK_SIZE: u64 = 4096;

/// Current footer version. Mounts MUST refuse any other version.
pub const MANIFEST_VERSION: u8 = 1;

/// Parsed manifest footer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Manifest {
    /// Concatenated SHA-3-256 hashes (one per [`HASH_BLOCK_SIZE`] of
    /// the squashfs blob).
    pub hashes: Vec<[u8; BLOCK_HASH_LEN]>,
    /// Public-key fingerprint (SHA-3-256 of the public key bytes).
    pub pubkey_fingerprint: [u8; 32],
    /// ML-DSA-65 signature over the concatenated hash array.
    pub signature: Vec<u8>,
    /// Length of the entire footer in bytes.
    pub footer_length: u32,
}

impl Manifest {
    /// Locate, parse, and signature-verify the manifest footer.
    ///
    /// `partition` is the entire squashfs partition bytes (including
    /// the appended footer).
    /// `public_key` is the ML-DSA-65 public key the kernel pinned at
    /// build time; the manifest's `pubkey_fingerprint` MUST match
    /// `SHA-3-256(public_key)`.
    ///
    /// Returns the parsed [`Manifest`] only after both the public-key
    /// fingerprint and the ML-DSA-65 signature pass. On any mismatch
    /// returns the appropriate [`SquashfsError`] variant; the caller
    /// SHALL fail the mount.
    pub fn parse_and_verify(partition: &[u8], public_key: &[u8]) -> Result<Self, SquashfsError> {
        if partition.len() < 4 {
            return Err(SquashfsError::ManifestMissing);
        }
        let footer_length_pos = partition.len() - 4;
        let footer_length = u32::from_le_bytes(partition[footer_length_pos..].try_into().unwrap());
        if footer_length < 16 || footer_length as usize > partition.len() {
            return Err(SquashfsError::ManifestMissing);
        }
        let footer_start = partition.len() - footer_length as usize;
        let footer = &partition[footer_start..];
        if footer[..MANIFEST_MAGIC.len()] != MANIFEST_MAGIC[..] {
            return Err(SquashfsError::ManifestMissing);
        }
        let mut p = MANIFEST_MAGIC.len();
        let version = footer[p];
        p += 1;
        if version != MANIFEST_VERSION {
            return Err(SquashfsError::ManifestCorrupt);
        }
        if p + 4 > footer.len() {
            return Err(SquashfsError::ManifestCorrupt);
        }
        let hash_count = u32::from_le_bytes(footer[p..p + 4].try_into().unwrap()) as usize;
        p += 4;
        let hashes_bytes = hash_count
            .checked_mul(BLOCK_HASH_LEN)
            .ok_or(SquashfsError::ManifestCorrupt)?;
        if p + hashes_bytes > footer.len() {
            return Err(SquashfsError::ManifestCorrupt);
        }
        let mut hashes: Vec<[u8; BLOCK_HASH_LEN]> = Vec::with_capacity(hash_count);
        for i in 0..hash_count {
            let off = p + i * BLOCK_HASH_LEN;
            let mut h = [0u8; BLOCK_HASH_LEN];
            h.copy_from_slice(&footer[off..off + BLOCK_HASH_LEN]);
            hashes.push(h);
        }
        p += hashes_bytes;

        if p + 32 > footer.len() {
            return Err(SquashfsError::ManifestCorrupt);
        }
        let mut pubkey_fpr = [0u8; 32];
        pubkey_fpr.copy_from_slice(&footer[p..p + 32]);
        p += 32;

        if p + 2 > footer.len() {
            return Err(SquashfsError::ManifestCorrupt);
        }
        let sig_len = u16::from_le_bytes(footer[p..p + 2].try_into().unwrap()) as usize;
        p += 2;
        if p + sig_len > footer.len() {
            return Err(SquashfsError::ManifestCorrupt);
        }
        let signature = footer[p..p + sig_len].to_vec();

        // Verify pubkey fingerprint matches the supplied public key.
        let expected_fpr = sha3_256(public_key);
        if expected_fpr.as_bytes() != &pubkey_fpr {
            return Err(SquashfsError::ManifestPublicKeyMismatch);
        }

        // Verify the ML-DSA-65 signature.
        if public_key.len() != ML_DSA_65_PK_LEN {
            return Err(SquashfsError::ManifestPublicKeyMismatch);
        }
        let mut pk_array = [0u8; ML_DSA_65_PK_LEN];
        pk_array.copy_from_slice(public_key);
        let pk = MlDsaPublicKey::from_bytes(pk_array);
        let signature_obj = MlDsaSignature::from_slice(&signature)
            .map_err(|_| SquashfsError::ManifestSignatureInvalid)?;
        // The signed message is the concatenated hash bytes.
        let message = build_signed_message(&hashes);
        ml_dsa_65_verify(&pk, &message, &signature_obj)
            .map_err(|_| SquashfsError::ManifestSignatureInvalid)?;

        Ok(Self {
            hashes,
            pubkey_fingerprint: pubkey_fpr,
            signature,
            footer_length,
        })
    }

    /// Length of the squashfs blob (everything before the footer).
    pub fn squashfs_length(&self, partition_size: usize) -> usize {
        partition_size - self.footer_length as usize
    }

    /// Verify a single 4 KiB-aligned block against the manifest.
    ///
    /// `block_index` is `byte_offset_in_squashfs / HASH_BLOCK_SIZE`.
    pub fn verify_block(&self, block_index: u64, bytes: &[u8]) -> Result<(), SquashfsError> {
        let idx = usize::try_from(block_index).map_err(|_| SquashfsError::SizeOverflow)?;
        if idx >= self.hashes.len() {
            return Err(SquashfsError::OutOfRange);
        }
        let mut h = Sha3_256::new();
        h.update(bytes)
            .map_err(|_| SquashfsError::DecompressFailed)?;
        let got = h.finalize().map_err(|_| SquashfsError::DecompressFailed)?;
        if got.as_bytes() != &self.hashes[idx] {
            return Err(SquashfsError::BlockHashMismatch);
        }
        Ok(())
    }

    /// Number of 4 KiB hashes in the manifest.
    pub fn hash_count(&self) -> usize {
        self.hashes.len()
    }

    /// Public-key fingerprint of the signing key.
    pub fn fingerprint(&self) -> &[u8; 32] {
        &self.pubkey_fingerprint
    }
}

/// Build the message that the manifest signature covers: the
/// concatenated 32-byte hashes.
pub fn build_signed_message(hashes: &[[u8; BLOCK_HASH_LEN]]) -> Vec<u8> {
    let mut msg = Vec::with_capacity(hashes.len() * BLOCK_HASH_LEN);
    for h in hashes {
        msg.extend_from_slice(h);
    }
    msg
}

/// Build an unsigned manifest footer (used by tests and the install
/// pipeline). The caller MUST sign `build_signed_message(&hashes)`
/// with the matching ML-DSA-65 secret key, then assemble the final
/// footer with [`assemble_signed_footer`].
pub fn build_hashes_for_blob(blob: &[u8]) -> Vec<[u8; BLOCK_HASH_LEN]> {
    let count = blob.len().div_ceil(HASH_BLOCK_SIZE as usize);
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let start = i * HASH_BLOCK_SIZE as usize;
        let end = core::cmp::min(start + HASH_BLOCK_SIZE as usize, blob.len());
        let mut block_buf = alloc::vec![0u8; HASH_BLOCK_SIZE as usize];
        block_buf[..end - start].copy_from_slice(&blob[start..end]);
        // For full coverage we hash the *padded* 4 KiB block (zero-pad
        // the trailing partial), so `verify_block` always sees a 4 KiB
        // window matching what was signed.
        let mut h = [0u8; BLOCK_HASH_LEN];
        h.copy_from_slice(sha3_256(&block_buf).as_bytes());
        out.push(h);
    }
    out
}

/// Assemble the final manifest footer bytes from already-computed
/// hashes, public-key fingerprint, and a pre-computed signature.
///
/// The returned footer is 4-byte aligned in length so callers can
/// simply append it to a 4-byte-aligned squashfs blob; the trailing
/// `u32` `footer_length` field is then at the very end of the
/// resulting partition where [`Manifest::parse_and_verify`] expects
/// it to be.
pub fn assemble_signed_footer(
    hashes: &[[u8; BLOCK_HASH_LEN]],
    pubkey_fingerprint: &[u8; 32],
    signature: &[u8],
) -> Vec<u8> {
    assemble_signed_footer_with_pad(hashes, pubkey_fingerprint, signature, 0)
}

/// Like [`assemble_signed_footer`] but pads the footer with extra
/// zero bytes (after the signature, before the trailing
/// `footer_length` u32) so the final footer length is at least
/// `min_footer_len` bytes. The trailing `footer_length` u32 still
/// records the actual full length, so [`Manifest::parse_and_verify`]
/// reads it correctly.
///
/// Useful for callers that need the final partition (squashfs blob
/// plus footer) to land on a particular alignment without having to
/// re-hash the blob.
pub fn assemble_signed_footer_with_pad(
    hashes: &[[u8; BLOCK_HASH_LEN]],
    pubkey_fingerprint: &[u8; 32],
    signature: &[u8],
    min_footer_len: usize,
) -> Vec<u8> {
    let mut footer = Vec::new();
    footer.extend_from_slice(&MANIFEST_MAGIC);
    footer.push(MANIFEST_VERSION);
    footer.extend_from_slice(&(hashes.len() as u32).to_le_bytes());
    for h in hashes {
        footer.extend_from_slice(h);
    }
    footer.extend_from_slice(pubkey_fingerprint);
    footer.extend_from_slice(&(signature.len() as u16).to_le_bytes());
    footer.extend_from_slice(signature);
    while footer.len() % 4 != 0 {
        footer.push(0u8);
    }
    while footer.len() + 4 < min_footer_len {
        footer.push(0u8);
    }
    while footer.len() % 4 != 0 {
        footer.push(0u8);
    }
    let footer_len = (footer.len() + 4) as u32;
    footer.extend_from_slice(&footer_len.to_le_bytes());
    footer
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_security::crypto::ml_dsa::{ml_dsa_65_keygen, ml_dsa_65_sign};

    #[test]
    fn missing_footer_rejected() {
        let blob = alloc::vec![0u8; 64];
        let pk = alloc::vec![0u8; ML_DSA_65_PK_LEN];
        assert_eq!(
            Manifest::parse_and_verify(&blob, &pk),
            Err(SquashfsError::ManifestMissing)
        );
    }

    #[test]
    fn round_trip_signature() {
        let seed = [42u8; 32];
        let kp = ml_dsa_65_keygen(&seed).unwrap();
        let pk_bytes = kp.public_key.as_bytes().to_vec();
        let blob: Vec<u8> = (0..16384u32).map(|i| (i & 0xFF) as u8).collect();
        let hashes = build_hashes_for_blob(&blob);
        let pubkey_fpr = *sha3_256(&pk_bytes).as_bytes();
        let message = build_signed_message(&hashes);
        let nonce = [0u8; 32];
        let sig = ml_dsa_65_sign(&kp.secret_key, &message, &nonce).unwrap();
        let footer = assemble_signed_footer(&hashes, &pubkey_fpr, sig.as_bytes());
        let mut partition = blob.clone();
        partition.extend_from_slice(&footer);
        let m = Manifest::parse_and_verify(&partition, &pk_bytes).unwrap();
        assert_eq!(m.hash_count(), hashes.len());
    }

    #[test]
    fn tampered_signature_rejected() {
        let seed = [11u8; 32];
        let kp = ml_dsa_65_keygen(&seed).unwrap();
        let pk_bytes = kp.public_key.as_bytes().to_vec();
        let blob: Vec<u8> = alloc::vec![0u8; 8192];
        let hashes = build_hashes_for_blob(&blob);
        let pubkey_fpr = *sha3_256(&pk_bytes).as_bytes();
        let message = build_signed_message(&hashes);
        let nonce = [1u8; 32];
        let sig = ml_dsa_65_sign(&kp.secret_key, &message, &nonce).unwrap();
        let mut bad_sig = sig.as_bytes().to_vec();
        bad_sig[100] ^= 0x55;
        let footer = assemble_signed_footer(&hashes, &pubkey_fpr, &bad_sig);
        let mut partition = blob.clone();
        partition.extend_from_slice(&footer);
        assert_eq!(
            Manifest::parse_and_verify(&partition, &pk_bytes),
            Err(SquashfsError::ManifestSignatureInvalid)
        );
    }

    #[test]
    fn pubkey_fingerprint_mismatch_rejected() {
        let seed = [1u8; 32];
        let kp = ml_dsa_65_keygen(&seed).unwrap();
        let pk_bytes = kp.public_key.as_bytes().to_vec();
        let blob: Vec<u8> = alloc::vec![0u8; 4096];
        let hashes = build_hashes_for_blob(&blob);
        let bogus_fpr = [0xAAu8; 32]; // Wrong fingerprint.
        let message = build_signed_message(&hashes);
        let nonce = [2u8; 32];
        let sig = ml_dsa_65_sign(&kp.secret_key, &message, &nonce).unwrap();
        let footer = assemble_signed_footer(&hashes, &bogus_fpr, sig.as_bytes());
        let mut partition = blob.clone();
        partition.extend_from_slice(&footer);
        assert_eq!(
            Manifest::parse_and_verify(&partition, &pk_bytes),
            Err(SquashfsError::ManifestPublicKeyMismatch)
        );
    }

    #[test]
    fn block_verify_round_trip() {
        let seed = [7u8; 32];
        let kp = ml_dsa_65_keygen(&seed).unwrap();
        let pk_bytes = kp.public_key.as_bytes().to_vec();
        let blob: Vec<u8> = (0..8192u32).map(|i| ((i * 7) & 0xFF) as u8).collect();
        let hashes = build_hashes_for_blob(&blob);
        let pubkey_fpr = *sha3_256(&pk_bytes).as_bytes();
        let message = build_signed_message(&hashes);
        let nonce = [3u8; 32];
        let sig = ml_dsa_65_sign(&kp.secret_key, &message, &nonce).unwrap();
        let footer = assemble_signed_footer(&hashes, &pubkey_fpr, sig.as_bytes());
        let mut partition = blob.clone();
        partition.extend_from_slice(&footer);
        let m = Manifest::parse_and_verify(&partition, &pk_bytes).unwrap();
        // Block 0 verifies.
        let mut padded = alloc::vec![0u8; 4096];
        padded.copy_from_slice(&blob[..4096]);
        m.verify_block(0, &padded).unwrap();
        // Block 1 verifies.
        let mut padded = alloc::vec![0u8; 4096];
        padded.copy_from_slice(&blob[4096..8192]);
        m.verify_block(1, &padded).unwrap();
        // Tampered byte fails.
        padded[0] ^= 1;
        assert_eq!(
            m.verify_block(1, &padded),
            Err(SquashfsError::BlockHashMismatch)
        );
    }
}
