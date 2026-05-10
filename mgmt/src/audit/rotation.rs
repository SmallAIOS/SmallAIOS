// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Hybrid (size OR age) audit-log rotation with hard `audit.max_total_disk_bytes`
//! failsafe.
//!
//! Phase 10 ships the policy engine + storage trait; the production-
//! side `KernelStorage` adapter that operates on `/data/audit/log.jsonl`
//! and `log.<n>.jsonl.gz` archives lands when the audit ring's
//! production sink is wired.

extern crate alloc;
use alloc::string::String;
use alloc::vec::Vec;

/// Rotation policy parameters from `mgmt/policy.toml audit.*`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RotationPolicy {
    /// Rotate the active log at this size (bytes). Default 64 MiB.
    pub rotate_size_bytes: u64,
    /// Rotate the active log at this age (hours). Default 24.
    pub rotate_age_hours: u32,
    /// Number of archives to retain after rotation. Default 8.
    pub keep_archives: u32,
    /// Hard failsafe — total disk usage cap. Default 512 MiB. Eviction
    /// of oldest archive when exceeded, regardless of `keep_archives`.
    pub max_total_disk_bytes: u64,
}

impl RotationPolicy {
    pub const fn v1_defaults() -> Self {
        Self {
            rotate_size_bytes: 64 * 1024 * 1024,
            rotate_age_hours: 24,
            keep_archives: 8,
            max_total_disk_bytes: 512 * 1024 * 1024,
        }
    }
}

impl Default for RotationPolicy {
    fn default() -> Self {
        Self::v1_defaults()
    }
}

/// One rotated archive's bookkeeping entry.
#[derive(Debug, Clone)]
pub struct ArchiveEntry {
    pub name: String,
    pub size_bytes: u64,
    pub mtime_unix: u64,
}

/// Result of a rotation evaluation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RotationOutcome {
    /// No rotation needed.
    NoOp,
    /// Rotate the active log to a fresh archive index.
    Rotate { reason: RotationReason },
    /// Failsafe: evict oldest archive to keep total under cap.
    Evict { archive: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationReason {
    SizeThreshold,
    AgeThreshold,
}

/// Pluggable storage abstraction; production implementation operates
/// on `/data/audit/`. Test implementations use in-memory state.
pub trait Storage {
    fn active_size_bytes(&self) -> u64;
    fn active_age_hours(&self, now_unix: u64) -> u32;
    fn archives(&self) -> Vec<ArchiveEntry>;
    fn rotate_active(&mut self) -> Result<(), RotationError>;
    fn evict_archive(&mut self, name: &str) -> Result<(), RotationError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationError {
    StorageUnavailable,
    UnknownArchive,
}

/// Decide what to do given the current storage state and policy.
pub fn evaluate<S: Storage>(
    storage: &S,
    policy: &RotationPolicy,
    now_unix: u64,
) -> RotationOutcome {
    // Failsafe first: if total exceeds cap, evict oldest.
    let total: u64 =
        storage.active_size_bytes() + storage.archives().iter().map(|a| a.size_bytes).sum::<u64>();
    if total > policy.max_total_disk_bytes {
        if let Some(oldest) = storage
            .archives()
            .iter()
            .min_by_key(|a| a.mtime_unix)
            .cloned()
        {
            return RotationOutcome::Evict {
                archive: oldest.name,
            };
        }
    }

    // Hybrid: size OR age.
    if storage.active_size_bytes() > policy.rotate_size_bytes {
        return RotationOutcome::Rotate {
            reason: RotationReason::SizeThreshold,
        };
    }
    if storage.active_age_hours(now_unix) > policy.rotate_age_hours {
        return RotationOutcome::Rotate {
            reason: RotationReason::AgeThreshold,
        };
    }

    RotationOutcome::NoOp
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    struct MockStorage {
        active_size: u64,
        active_age_h: u32,
        archives: Vec<ArchiveEntry>,
    }

    impl Storage for MockStorage {
        fn active_size_bytes(&self) -> u64 {
            self.active_size
        }
        fn active_age_hours(&self, _now_unix: u64) -> u32 {
            self.active_age_h
        }
        fn archives(&self) -> Vec<ArchiveEntry> {
            self.archives.clone()
        }
        fn rotate_active(&mut self) -> Result<(), RotationError> {
            self.active_size = 0;
            self.active_age_h = 0;
            Ok(())
        }
        fn evict_archive(&mut self, name: &str) -> Result<(), RotationError> {
            let before = self.archives.len();
            self.archives.retain(|a| a.name != name);
            if self.archives.len() == before {
                return Err(RotationError::UnknownArchive);
            }
            Ok(())
        }
    }

    #[test]
    fn noop_below_thresholds() {
        let s = MockStorage {
            active_size: 1,
            active_age_h: 1,
            archives: Vec::new(),
        };
        assert_eq!(
            evaluate(&s, &RotationPolicy::v1_defaults(), 0),
            RotationOutcome::NoOp
        );
    }

    #[test]
    fn size_threshold_rotates() {
        let s = MockStorage {
            active_size: 100 * 1024 * 1024,
            active_age_h: 1,
            archives: Vec::new(),
        };
        assert_eq!(
            evaluate(&s, &RotationPolicy::v1_defaults(), 0),
            RotationOutcome::Rotate {
                reason: RotationReason::SizeThreshold
            }
        );
    }

    #[test]
    fn age_threshold_rotates() {
        let s = MockStorage {
            active_size: 1,
            active_age_h: 30,
            archives: Vec::new(),
        };
        assert_eq!(
            evaluate(&s, &RotationPolicy::v1_defaults(), 0),
            RotationOutcome::Rotate {
                reason: RotationReason::AgeThreshold
            }
        );
    }

    #[test]
    fn failsafe_evicts_oldest_when_over_cap() {
        let s = MockStorage {
            active_size: 0,
            active_age_h: 0,
            archives: alloc::vec![
                ArchiveEntry {
                    name: "log.1.jsonl".to_string(),
                    size_bytes: 600 * 1024 * 1024,
                    mtime_unix: 100
                },
                ArchiveEntry {
                    name: "log.2.jsonl".to_string(),
                    size_bytes: 1,
                    mtime_unix: 200
                },
            ],
        };
        // 600 MiB + 1 byte > 512 MiB cap → evict oldest.
        assert_eq!(
            evaluate(&s, &RotationPolicy::v1_defaults(), 0),
            RotationOutcome::Evict {
                archive: "log.1.jsonl".to_string()
            }
        );
    }
}
