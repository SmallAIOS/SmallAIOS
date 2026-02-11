// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! MIL-STD-1553B dual-redundant bus management (Bus A / Bus B failover).
//!
//! Implements automatic retry on the alternate bus when a transaction fails
//! on the primary bus. The manager tracks per-bus health and selects the
//! active bus based on error thresholds.

use crate::BusError;
use crate::mil1553::bc::{Bus, MessageResult};

/// Dual-bus health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusHealth {
    /// Bus is operational with no errors.
    Healthy,
    /// Bus is degraded (errors below threshold).
    Degraded,
    /// Bus has failed (errors at or above threshold).
    Failed,
}

/// Per-bus error tracking.
#[derive(Debug, Clone)]
struct BusTracker {
    /// Total timeout count.
    timeouts: u64,
    /// Total message error count.
    message_errors: u64,
    /// Total successful transactions.
    successes: u64,
    /// Consecutive failure count (reset on success).
    consecutive_failures: u32,
    /// Whether the bus is currently enabled.
    enabled: bool,
}

impl BusTracker {
    fn new() -> Self {
        Self {
            timeouts: 0,
            message_errors: 0,
            successes: 0,
            consecutive_failures: 0,
            enabled: true,
        }
    }

    fn record_success(&mut self) {
        self.successes += 1;
        self.consecutive_failures = 0;
    }

    fn record_timeout(&mut self) {
        self.timeouts += 1;
        self.consecutive_failures += 1;
    }

    fn record_error(&mut self) {
        self.message_errors += 1;
        self.consecutive_failures += 1;
    }

    fn total_errors(&self) -> u64 {
        self.timeouts + self.message_errors
    }
}

/// Dual-redundant bus manager.
///
/// Manages Bus A and Bus B with automatic failover when the active bus
/// exceeds an error threshold. Supports retry-on-alternate-bus strategy.
#[derive(Debug)]
pub struct DualBusManager {
    /// Bus A tracker.
    bus_a: BusTracker,
    /// Bus B tracker.
    bus_b: BusTracker,
    /// Currently active bus.
    active_bus: Bus,
    /// Error threshold for failover.
    failover_threshold: u32,
    /// Whether automatic failover is enabled.
    auto_failover: bool,
    /// Number of failover events that have occurred.
    failover_count: u32,
}

impl DualBusManager {
    /// Create a new dual-bus manager.
    ///
    /// `failover_threshold`: number of consecutive failures before switching buses.
    pub fn new(failover_threshold: u32) -> Self {
        Self {
            bus_a: BusTracker::new(),
            bus_b: BusTracker::new(),
            active_bus: Bus::A,
            failover_threshold,
            auto_failover: true,
            failover_count: 0,
        }
    }

    /// Returns the currently active bus.
    pub fn active_bus(&self) -> Bus {
        self.active_bus
    }

    /// Returns the alternate (standby) bus.
    pub fn standby_bus(&self) -> Bus {
        match self.active_bus {
            Bus::A => Bus::B,
            Bus::B => Bus::A,
        }
    }

    /// Returns the health status of a bus.
    pub fn bus_health(&self, bus: Bus) -> BusHealth {
        let tracker = self.tracker(bus);
        if !tracker.enabled {
            return BusHealth::Failed;
        }
        if tracker.consecutive_failures >= self.failover_threshold {
            BusHealth::Failed
        } else if tracker.total_errors() > 0 {
            BusHealth::Degraded
        } else {
            BusHealth::Healthy
        }
    }

    /// Returns the total timeout count for a bus.
    pub fn timeouts(&self, bus: Bus) -> u64 {
        self.tracker(bus).timeouts
    }

    /// Returns the total message error count for a bus.
    pub fn message_errors(&self, bus: Bus) -> u64 {
        self.tracker(bus).message_errors
    }

    /// Returns the total success count for a bus.
    pub fn successes(&self, bus: Bus) -> u64 {
        self.tracker(bus).successes
    }

    /// Returns the consecutive failure count for a bus.
    pub fn consecutive_failures(&self, bus: Bus) -> u32 {
        self.tracker(bus).consecutive_failures
    }

    /// Returns the number of failover events.
    pub fn failover_count(&self) -> u32 {
        self.failover_count
    }

    /// Enable or disable automatic failover.
    pub fn set_auto_failover(&mut self, enabled: bool) {
        self.auto_failover = enabled;
    }

    /// Manually force the active bus.
    pub fn force_bus(&mut self, bus: Bus) {
        self.active_bus = bus;
    }

    /// Disable a bus (mark as failed).
    pub fn disable_bus(&mut self, bus: Bus) {
        self.tracker_mut(bus).enabled = false;
        // If the disabled bus was active, switch to the other
        if self.active_bus == bus {
            self.active_bus = match bus {
                Bus::A => Bus::B,
                Bus::B => Bus::A,
            };
            self.failover_count += 1;
        }
    }

    /// Re-enable a previously disabled bus.
    pub fn enable_bus(&mut self, bus: Bus) {
        let tracker = self.tracker_mut(bus);
        tracker.enabled = true;
        tracker.consecutive_failures = 0;
    }

    /// Returns whether a bus is enabled.
    pub fn is_enabled(&self, bus: Bus) -> bool {
        self.tracker(bus).enabled
    }

    /// Record a transaction result and handle failover if needed.
    ///
    /// Returns the bus that should be used for the next attempt:
    /// - If the result is a failure and the alternate bus is available,
    ///   returns `Some(alternate_bus)` to suggest a retry.
    /// - Returns `None` if no retry is needed or possible.
    pub fn record_result(
        &mut self,
        bus: Bus,
        result: &MessageResult,
    ) -> Option<Bus> {
        match result {
            MessageResult::Success { .. } => {
                self.tracker_mut(bus).record_success();
                None
            }
            MessageResult::Timeout => {
                self.tracker_mut(bus).record_timeout();
                self.check_failover(bus)
            }
            MessageResult::MessageError => {
                self.tracker_mut(bus).record_error();
                self.check_failover(bus)
            }
            MessageResult::Busy => {
                // Busy is not a bus error, no failover needed
                None
            }
        }
    }

    /// Determine which bus to use for the next transaction.
    ///
    /// Returns the active bus if healthy, or the standby bus if the
    /// active bus has failed. Returns `Err` if both buses have failed.
    pub fn select_bus(&self) -> Result<Bus, BusError> {
        let active_health = self.bus_health(self.active_bus);
        if active_health != BusHealth::Failed {
            return Ok(self.active_bus);
        }
        let standby = self.standby_bus();
        let standby_health = self.bus_health(standby);
        if standby_health != BusHealth::Failed {
            return Ok(standby);
        }
        Err(BusError::BusOff)
    }

    /// Check if failover is needed after a failure on the given bus.
    fn check_failover(&mut self, failed_bus: Bus) -> Option<Bus> {
        if !self.auto_failover {
            return None;
        }

        let tracker = self.tracker(failed_bus);
        if tracker.consecutive_failures >= self.failover_threshold {
            let alternate = match failed_bus {
                Bus::A => Bus::B,
                Bus::B => Bus::A,
            };

            // Only fail over if the alternate bus is still healthy
            if self.tracker(alternate).enabled
                && self.tracker(alternate).consecutive_failures < self.failover_threshold
            {
                if self.active_bus == failed_bus {
                    self.active_bus = alternate;
                    self.failover_count += 1;
                }
                return Some(alternate);
            }
        }

        // Suggest retry on alternate bus even without failover
        let alternate = match failed_bus {
            Bus::A => Bus::B,
            Bus::B => Bus::A,
        };
        if self.tracker(alternate).enabled {
            Some(alternate)
        } else {
            None
        }
    }

    fn tracker(&self, bus: Bus) -> &BusTracker {
        match bus {
            Bus::A => &self.bus_a,
            Bus::B => &self.bus_b,
        }
    }

    fn tracker_mut(&mut self, bus: Bus) -> &mut BusTracker {
        match bus {
            Bus::A => &mut self.bus_a,
            Bus::B => &mut self.bus_b,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mil1553::word::StatusWord;
    use alloc::vec::Vec;

    fn success_result() -> MessageResult {
        MessageResult::Success {
            status: StatusWord::new(5).unwrap(),
            data: Vec::new(),
        }
    }

    #[test]
    fn test_new_manager() {
        let mgr = DualBusManager::new(3);
        assert_eq!(mgr.active_bus(), Bus::A);
        assert_eq!(mgr.standby_bus(), Bus::B);
        assert_eq!(mgr.failover_count(), 0);
        assert!(mgr.is_enabled(Bus::A));
        assert!(mgr.is_enabled(Bus::B));
    }

    #[test]
    fn test_bus_health_healthy() {
        let mgr = DualBusManager::new(3);
        assert_eq!(mgr.bus_health(Bus::A), BusHealth::Healthy);
        assert_eq!(mgr.bus_health(Bus::B), BusHealth::Healthy);
    }

    #[test]
    fn test_bus_health_degraded() {
        let mut mgr = DualBusManager::new(5);
        mgr.record_result(Bus::A, &MessageResult::Timeout);
        mgr.record_result(Bus::A, &success_result()); // Reset consecutive
        assert_eq!(mgr.bus_health(Bus::A), BusHealth::Degraded);
    }

    #[test]
    fn test_bus_health_failed() {
        let mut mgr = DualBusManager::new(3);
        for _ in 0..3 {
            mgr.record_result(Bus::A, &MessageResult::Timeout);
        }
        assert_eq!(mgr.bus_health(Bus::A), BusHealth::Failed);
    }

    #[test]
    fn test_success_resets_consecutive() {
        let mut mgr = DualBusManager::new(3);
        mgr.record_result(Bus::A, &MessageResult::Timeout);
        mgr.record_result(Bus::A, &MessageResult::Timeout);
        assert_eq!(mgr.consecutive_failures(Bus::A), 2);
        mgr.record_result(Bus::A, &success_result());
        assert_eq!(mgr.consecutive_failures(Bus::A), 0);
    }

    #[test]
    fn test_auto_failover() {
        let mut mgr = DualBusManager::new(3);
        assert_eq!(mgr.active_bus(), Bus::A);

        // 3 consecutive failures trigger failover
        for _ in 0..3 {
            mgr.record_result(Bus::A, &MessageResult::Timeout);
        }
        assert_eq!(mgr.active_bus(), Bus::B);
        assert_eq!(mgr.failover_count(), 1);
    }

    #[test]
    fn test_failover_suggests_alternate() {
        let mut mgr = DualBusManager::new(3);
        let retry = mgr.record_result(Bus::A, &MessageResult::Timeout);
        // Should suggest Bus::B as retry
        assert_eq!(retry, Some(Bus::B));
    }

    #[test]
    fn test_no_failover_on_success() {
        let mut mgr = DualBusManager::new(3);
        let retry = mgr.record_result(Bus::A, &success_result());
        assert_eq!(retry, None);
    }

    #[test]
    fn test_no_failover_on_busy() {
        let mut mgr = DualBusManager::new(3);
        let retry = mgr.record_result(Bus::A, &MessageResult::Busy);
        assert_eq!(retry, None);
    }

    #[test]
    fn test_disable_bus() {
        let mut mgr = DualBusManager::new(3);
        mgr.disable_bus(Bus::A);
        assert!(!mgr.is_enabled(Bus::A));
        assert_eq!(mgr.bus_health(Bus::A), BusHealth::Failed);
        // Active bus should switch to B
        assert_eq!(mgr.active_bus(), Bus::B);
    }

    #[test]
    fn test_disable_standby_bus() {
        let mut mgr = DualBusManager::new(3);
        mgr.disable_bus(Bus::B);
        assert!(!mgr.is_enabled(Bus::B));
        // Active bus stays A since B was standby
        assert_eq!(mgr.active_bus(), Bus::A);
    }

    #[test]
    fn test_enable_bus() {
        let mut mgr = DualBusManager::new(3);
        mgr.disable_bus(Bus::A);
        mgr.enable_bus(Bus::A);
        assert!(mgr.is_enabled(Bus::A));
        assert_eq!(mgr.consecutive_failures(Bus::A), 0);
    }

    #[test]
    fn test_force_bus() {
        let mut mgr = DualBusManager::new(3);
        mgr.force_bus(Bus::B);
        assert_eq!(mgr.active_bus(), Bus::B);
    }

    #[test]
    fn test_select_bus_active_healthy() {
        let mgr = DualBusManager::new(3);
        assert_eq!(mgr.select_bus().unwrap(), Bus::A);
    }

    #[test]
    fn test_select_bus_active_failed() {
        let mut mgr = DualBusManager::new(3);
        for _ in 0..3 {
            mgr.record_result(Bus::A, &MessageResult::Timeout);
        }
        // After failover, Bus::B is now active
        assert_eq!(mgr.select_bus().unwrap(), Bus::B);
    }

    #[test]
    fn test_select_bus_both_failed() {
        let mut mgr = DualBusManager::new(3);
        mgr.disable_bus(Bus::A);
        mgr.disable_bus(Bus::B);
        assert_eq!(mgr.select_bus().err(), Some(BusError::BusOff));
    }

    #[test]
    fn test_auto_failover_disabled() {
        let mut mgr = DualBusManager::new(3);
        mgr.set_auto_failover(false);
        for _ in 0..5 {
            mgr.record_result(Bus::A, &MessageResult::Timeout);
        }
        // Should stay on Bus A even after failures
        assert_eq!(mgr.active_bus(), Bus::A);
        assert_eq!(mgr.failover_count(), 0);
    }

    #[test]
    fn test_no_retry_when_alternate_disabled() {
        let mut mgr = DualBusManager::new(3);
        mgr.disable_bus(Bus::B);
        let retry = mgr.record_result(Bus::A, &MessageResult::Timeout);
        assert_eq!(retry, None);
    }

    #[test]
    fn test_counters() {
        let mut mgr = DualBusManager::new(100);
        mgr.record_result(Bus::A, &success_result());
        mgr.record_result(Bus::A, &MessageResult::Timeout);
        mgr.record_result(Bus::A, &MessageResult::MessageError);
        mgr.record_result(Bus::B, &success_result());
        mgr.record_result(Bus::B, &MessageResult::Timeout);

        assert_eq!(mgr.successes(Bus::A), 1);
        assert_eq!(mgr.timeouts(Bus::A), 1);
        assert_eq!(mgr.message_errors(Bus::A), 1);
        assert_eq!(mgr.successes(Bus::B), 1);
        assert_eq!(mgr.timeouts(Bus::B), 1);
    }

    #[test]
    fn test_double_failover() {
        let mut mgr = DualBusManager::new(2);
        // Fail Bus A
        for _ in 0..2 {
            mgr.record_result(Bus::A, &MessageResult::Timeout);
        }
        assert_eq!(mgr.active_bus(), Bus::B);
        assert_eq!(mgr.failover_count(), 1);

        // Recover Bus A
        mgr.enable_bus(Bus::A);
        // Fail Bus B
        for _ in 0..2 {
            mgr.record_result(Bus::B, &MessageResult::MessageError);
        }
        assert_eq!(mgr.active_bus(), Bus::A);
        assert_eq!(mgr.failover_count(), 2);
    }
}
