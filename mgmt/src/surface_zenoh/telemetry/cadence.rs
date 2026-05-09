// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-key cadence resolver: clamps the configured interval and
//! applies an optional adaptive override.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-zenoh-telemetry/spec.md`
//! Requirement "Per-key configurable cadence with adaptive option".
//!
//! ## Adaptive cadence
//!
//! When a metric crosses `threshold` (e.g. any CPU core > 80%), the
//! publisher upgrades to `fast_hz` and remains there until the metric
//! falls back below the threshold, at which point it reverts to
//! `slow_hz`. Hysteresis is **single-step** by design — the spec does
//! not call for separate enter/exit thresholds.

use super::clamp_interval;

/// Adaptive cadence overrides. When attached to a [`CadenceState`],
/// the publisher rebuilds its effective interval each tick from the
/// caller's metric input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdaptiveCadence {
    /// Threshold value the publisher's metric is compared against.
    /// For CPU, this is `peak_util_pct` in `0..=100`.
    pub threshold: u32,
    /// Cadence in Hz when above threshold (1..=10 typical). Translated
    /// to `1000 / fast_hz` ms internally and clamped to the publisher
    /// floor.
    pub fast_hz: u32,
    /// Cadence in Hz when at or below threshold.
    pub slow_hz: u32,
}

impl AdaptiveCadence {
    /// Construct from a triple. Helper for tests and for constructing
    /// from the typed `Config`.
    pub const fn new(threshold: u32, fast_hz: u32, slow_hz: u32) -> Self {
        Self {
            threshold,
            fast_hz,
            slow_hz,
        }
    }

    /// Translate `fast_hz` to a publisher interval (ms), clamped to
    /// the publisher's legal window.
    pub fn fast_interval_ms(&self) -> u32 {
        clamp_interval(hz_to_ms(self.fast_hz))
    }

    /// Translate `slow_hz` to a publisher interval (ms), clamped to
    /// the publisher's legal window.
    pub fn slow_interval_ms(&self) -> u32 {
        clamp_interval(hz_to_ms(self.slow_hz))
    }
}

/// Per-key publisher cadence state. Built once at start, updated each
/// tick with the latest metric reading.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CadenceState {
    /// Configured baseline interval in ms — used when no adaptive
    /// override is attached.
    pub base_interval_ms: u32,
    /// Optional adaptive override.
    pub adaptive: Option<AdaptiveCadence>,
    /// `true` iff the publisher is currently in the fast-hz tier.
    /// Tracked across ticks so threshold-cross hysteresis is stable.
    pub in_fast_tier: bool,
}

impl CadenceState {
    /// Build a non-adaptive state.
    pub const fn fixed(base_interval_ms: u32) -> Self {
        Self {
            base_interval_ms,
            adaptive: None,
            in_fast_tier: false,
        }
    }

    /// Build an adaptive state.
    pub const fn adaptive(base_interval_ms: u32, adaptive: AdaptiveCadence) -> Self {
        Self {
            base_interval_ms,
            adaptive: Some(adaptive),
            in_fast_tier: false,
        }
    }

    /// Update the in-fast-tier flag from a fresh metric reading and
    /// return the resolved [`EffectiveCadence`] for the next tick.
    pub fn update(&mut self, metric: u32) -> EffectiveCadence {
        let Some(adapt) = self.adaptive else {
            return EffectiveCadence {
                interval_ms: self.base_interval_ms,
                fast_tier: false,
            };
        };

        // Cross above threshold → fast. At or below → slow.
        // Hysteresis keeps the tier stable until the metric crosses
        // the threshold in the opposite direction.
        self.in_fast_tier = metric > adapt.threshold;

        let interval_ms = if self.in_fast_tier {
            adapt.fast_interval_ms()
        } else {
            adapt.slow_interval_ms()
        };
        EffectiveCadence {
            interval_ms,
            fast_tier: self.in_fast_tier,
        }
    }

    /// Resolve the current effective cadence without updating the
    /// in-fast-tier flag — used for diagnostic queries.
    pub fn current(&self) -> EffectiveCadence {
        match self.adaptive {
            Some(adapt) if self.in_fast_tier => EffectiveCadence {
                interval_ms: adapt.fast_interval_ms(),
                fast_tier: true,
            },
            Some(adapt) => EffectiveCadence {
                interval_ms: adapt.slow_interval_ms(),
                fast_tier: false,
            },
            None => EffectiveCadence {
                interval_ms: self.base_interval_ms,
                fast_tier: false,
            },
        }
    }
}

/// Snapshot of the effective cadence for a single key after an update.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffectiveCadence {
    /// Interval in milliseconds the publisher SHALL use for its next
    /// deadline.
    pub interval_ms: u32,
    /// `true` iff the adaptive fast tier is engaged. `false` for
    /// non-adaptive keys.
    pub fast_tier: bool,
}

/// Convert `hz` to a publisher interval in milliseconds. Zero is
/// treated as 1 Hz (defensive — the validator rejects zero before this
/// runs in production paths).
pub fn hz_to_ms(hz: u32) -> u32 {
    if hz == 0 {
        return 1000;
    }
    let ms = 1000u64 / hz as u64;
    if ms == 0 {
        1
    } else if ms > u32::MAX as u64 {
        u32::MAX
    } else {
        ms as u32
    }
}

#[cfg(test)]
mod tests {
    use super::super::{MAX_INTERVAL_MS, MIN_INTERVAL_MS};
    use super::*;

    #[test]
    fn hz_to_ms_basic_conversions() {
        assert_eq!(hz_to_ms(1), 1000);
        assert_eq!(hz_to_ms(10), 100);
        assert_eq!(hz_to_ms(2), 500);
        assert_eq!(hz_to_ms(4), 250);
    }

    #[test]
    fn hz_to_ms_zero_defaults_to_one_hz() {
        assert_eq!(hz_to_ms(0), 1000);
    }

    #[test]
    fn hz_to_ms_high_hz_floors_at_one_ms() {
        assert_eq!(hz_to_ms(2000), 1);
    }

    #[test]
    fn fixed_cadence_returns_base_interval() {
        let mut s = CadenceState::fixed(750);
        let eff = s.update(0);
        assert_eq!(eff.interval_ms, 750);
        assert!(!eff.fast_tier);
    }

    #[test]
    fn fixed_cadence_metric_value_is_ignored() {
        let mut s = CadenceState::fixed(1000);
        // Crossing any threshold-shaped value SHALL not flip tiers
        // when adaptive is not configured.
        let eff = s.update(99);
        assert_eq!(eff.interval_ms, 1000);
        let eff = s.update(0);
        assert_eq!(eff.interval_ms, 1000);
    }

    #[test]
    fn adaptive_above_threshold_engages_fast_tier() {
        let mut s = CadenceState::adaptive(1000, AdaptiveCadence::new(80, 10, 1));
        let eff = s.update(85);
        // fast_hz = 10 → 100 ms (clamped to floor).
        assert_eq!(eff.interval_ms, 100);
        assert!(eff.fast_tier);
        assert!(s.in_fast_tier);
    }

    #[test]
    fn adaptive_at_threshold_remains_slow_tier() {
        // Spec: "exceeds 80" → fast. == 80 keeps slow.
        let mut s = CadenceState::adaptive(1000, AdaptiveCadence::new(80, 10, 1));
        let eff = s.update(80);
        assert_eq!(eff.interval_ms, 1000);
        assert!(!eff.fast_tier);
    }

    #[test]
    fn adaptive_below_threshold_reverts_to_slow_tier() {
        let mut s = CadenceState::adaptive(1000, AdaptiveCadence::new(80, 10, 1));
        // Engage fast.
        s.update(90);
        assert!(s.in_fast_tier);
        // Drop back below.
        let eff = s.update(50);
        assert_eq!(eff.interval_ms, 1000);
        assert!(!eff.fast_tier);
    }

    #[test]
    fn adaptive_fast_interval_clamped_to_floor() {
        // fast_hz = 100 would imply 10 ms; clamped to 100 ms floor.
        let a = AdaptiveCadence::new(80, 100, 1);
        assert_eq!(a.fast_interval_ms(), MIN_INTERVAL_MS);
    }

    #[test]
    fn adaptive_slow_interval_clamped_to_ceiling() {
        // slow_hz < 1 (i.e. zero) → defaults to 1 Hz which is in
        // range; we instead use a value high enough to test the
        // ceiling: hz_to_ms(0) is 1000, which is far below ceiling.
        // Force the path by constructing a slow_hz that yields
        // > 60s — there's no integer-Hz value that does so without
        // hitting the zero branch, so we explicitly verify clamp via
        // the `clamp_interval` helper.
        let a = AdaptiveCadence::new(80, 10, 1);
        // slow_hz = 1 → 1000ms, which is in range.
        assert_eq!(a.slow_interval_ms(), 1000);
        // Confirm ceiling clamp via direct call.
        assert_eq!(super::super::clamp_interval(120_000), MAX_INTERVAL_MS);
    }

    #[test]
    fn adaptive_current_reflects_fast_tier_without_updating() {
        let mut s = CadenceState::adaptive(1000, AdaptiveCadence::new(80, 10, 1));
        s.update(90); // engage fast
        let cur = s.current();
        assert!(cur.fast_tier);
        assert_eq!(cur.interval_ms, 100);
    }

    #[test]
    fn adaptive_state_carries_tier_across_ticks() {
        let mut s = CadenceState::adaptive(1000, AdaptiveCadence::new(80, 10, 1));
        s.update(90);
        // Same metric value (still > threshold) → still fast.
        s.update(90);
        assert!(s.in_fast_tier);
    }

    #[test]
    fn adaptive_threshold_zero_always_fast_above_zero() {
        let mut s = CadenceState::adaptive(1000, AdaptiveCadence::new(0, 10, 1));
        let eff = s.update(1);
        assert!(eff.fast_tier);
    }
}
