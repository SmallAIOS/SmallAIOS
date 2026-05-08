// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Argon2id per-tier parameter selection from available RAM.
//!
//! Spec: `openspec/changes/management-login-v1/specs/auth-shadow/spec.md`
//! Requirement "Argon2id per-tier parameter selection".
//!
//! At first boot the kernel measures available RAM and picks a tier. The
//! parameters are then embedded in each generated PHC string, so subsequent
//! verification is self-describing — see [`smallaios_security::argon2id`].
//!
//! | Tier      | RAM                       | m_cost  | t_cost | p_cost |
//! |-----------|---------------------------|---------|--------|--------|
//! | `tiny`    | ≤ 256 MiB                 | 8 MiB   | 3      | 1      |
//! | `default` | > 256 MiB and < 4 GiB     | 64 MiB  | 3      | 1      |
//! | `strong`  | ≥ 4 GiB                   | 128 MiB | 4      | 2      |

use smallaios_security::argon2id::Argon2idParams;

/// Threshold separating the `tiny` and `default` tiers (≤ 256 MiB → tiny).
pub const TINY_RAM_BYTES: u64 = 256 * 1024 * 1024;

/// Threshold separating the `default` and `strong` tiers (≥ 4 GiB → strong).
pub const STRONG_RAM_BYTES: u64 = 4 * 1024 * 1024 * 1024;

/// Pick an Argon2id parameter set for the running target based on available
/// RAM in bytes. Inputs at the boundaries fall in the lower tier:
///
/// - `available_ram_bytes <= 256 MiB`             → [`Argon2idParams::tiny`]
/// - `256 MiB < available_ram_bytes < 4 GiB`      → [`Argon2idParams::default_tier`]
/// - `available_ram_bytes >= 4 GiB`               → [`Argon2idParams::strong`]
///
/// The decision SHALL only be made when generating a *new* hash. When
/// verifying an existing hash the parameters in the PHC string take
/// precedence — this is enforced by [`smallaios_security::argon2id::argon2id_verify`].
pub fn select_tier(available_ram_bytes: u64) -> Argon2idParams {
    if available_ram_bytes <= TINY_RAM_BYTES {
        Argon2idParams::tiny()
    } else if available_ram_bytes < STRONG_RAM_BYTES {
        Argon2idParams::default_tier()
    } else {
        Argon2idParams::strong()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_for_under_256mib() {
        assert_eq!(select_tier(0), Argon2idParams::tiny());
        assert_eq!(select_tier(64 * 1024 * 1024), Argon2idParams::tiny());
        assert_eq!(select_tier(256 * 1024 * 1024), Argon2idParams::tiny());
    }

    #[test]
    fn default_for_between_256mib_and_4gib() {
        assert_eq!(
            select_tier(256 * 1024 * 1024 + 1),
            Argon2idParams::default_tier()
        );
        assert_eq!(
            select_tier(1024 * 1024 * 1024),
            Argon2idParams::default_tier()
        );
        assert_eq!(
            select_tier(4 * 1024 * 1024 * 1024 - 1),
            Argon2idParams::default_tier()
        );
    }

    #[test]
    fn strong_for_4gib_and_up() {
        assert_eq!(
            select_tier(4 * 1024 * 1024 * 1024),
            Argon2idParams::strong()
        );
        assert_eq!(
            select_tier(16 * 1024 * 1024 * 1024),
            Argon2idParams::strong()
        );
        assert_eq!(select_tier(u64::MAX), Argon2idParams::strong());
    }

    #[test]
    fn jetson_orin_nx_16gib_picks_strong() {
        // Spec scenario: "Jetson-class selects strong tier"
        // Jetson Orin NX 16 GiB → strong (m=128 MiB, t=4, p=2).
        let params = select_tier(16 * 1024 * 1024 * 1024);
        assert_eq!(params.m_cost_kib, 128 * 1024);
        assert_eq!(params.t_cost, 4);
        assert_eq!(params.p_cost, 2);
    }

    #[test]
    fn tiny_target_256mib_picks_tiny() {
        // Spec scenario: "Tiny board selects tiny tier"
        let params = select_tier(256 * 1024 * 1024);
        assert_eq!(params.m_cost_kib, 8 * 1024);
        assert_eq!(params.t_cost, 3);
        assert_eq!(params.p_cost, 1);
    }
}
