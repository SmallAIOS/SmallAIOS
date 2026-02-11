// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! QUIC congestion control (RFC 9002).
//!
//! Implements NewReno-based congestion control with:
//! - Slow start, congestion avoidance
//! - RTT estimation (smoothed RTT, RTT variance)
//! - Probe timeout (PTO) for loss detection
//! - Persistent congestion detection

/// Initial congestion window (14720 bytes per RFC 9002).
const INITIAL_WINDOW: u64 = 14_720;

/// Minimum congestion window (2 * max datagram size).
const MIN_WINDOW: u64 = 2 * 1472;

/// Loss reduction factor (NewReno uses 1/2).
const LOSS_REDUCTION_NUMERATOR: u64 = 1;
const LOSS_REDUCTION_DENOMINATOR: u64 = 2;

/// Persistent congestion threshold (RFC 9002 Section 7.6).
const PERSISTENT_CONGESTION_THRESHOLD: u64 = 3;

/// Default initial RTT estimate (333ms per RFC 9002).
const INITIAL_RTT_US: u64 = 333_000;

/// RTT measurement state.
#[derive(Debug)]
pub struct RttEstimator {
    /// Smoothed RTT (microseconds).
    smoothed_rtt: u64,
    /// RTT variance.
    rttvar: u64,
    /// Minimum RTT observed.
    min_rtt: u64,
    /// Latest RTT sample.
    latest_rtt: u64,
    /// Whether we have a valid RTT measurement.
    has_sample: bool,
}

impl Default for RttEstimator {
    fn default() -> Self {
        Self::new()
    }
}

impl RttEstimator {
    /// Create a new RTT estimator.
    pub fn new() -> Self {
        Self {
            smoothed_rtt: INITIAL_RTT_US,
            rttvar: INITIAL_RTT_US / 2,
            min_rtt: u64::MAX,
            latest_rtt: 0,
            has_sample: false,
        }
    }

    /// Update RTT with a new sample.
    pub fn update(&mut self, rtt_sample: u64, ack_delay: u64) {
        self.latest_rtt = rtt_sample;

        if rtt_sample < self.min_rtt {
            self.min_rtt = rtt_sample;
        }

        if !self.has_sample {
            self.smoothed_rtt = rtt_sample;
            self.rttvar = rtt_sample / 2;
            self.has_sample = true;
            return;
        }

        // Adjust for ACK delay (only in application data)
        let adjusted_rtt = if rtt_sample > self.min_rtt + ack_delay {
            rtt_sample - ack_delay
        } else {
            rtt_sample
        };

        // EWMA update per RFC 9002
        let abs_diff = self.smoothed_rtt.abs_diff(adjusted_rtt);
        self.rttvar = (3 * self.rttvar + abs_diff) / 4;
        self.smoothed_rtt = (7 * self.smoothed_rtt + adjusted_rtt) / 8;
    }

    /// Smoothed RTT.
    pub fn smoothed_rtt(&self) -> u64 {
        self.smoothed_rtt
    }

    /// RTT variance.
    pub fn rttvar(&self) -> u64 {
        self.rttvar
    }

    /// Minimum RTT.
    pub fn min_rtt(&self) -> u64 {
        if self.min_rtt == u64::MAX {
            INITIAL_RTT_US
        } else {
            self.min_rtt
        }
    }

    /// Latest RTT.
    pub fn latest_rtt(&self) -> u64 {
        self.latest_rtt
    }

    /// Whether we have a real RTT sample.
    pub fn has_sample(&self) -> bool {
        self.has_sample
    }

    /// Compute the loss detection delay (max(1.125 * smoothed_rtt, kGranularity)).
    pub fn loss_delay(&self) -> u64 {
        // 9/8 * smoothed_rtt, minimum 1ms
        let delay = (9 * self.smoothed_rtt) / 8;
        delay.max(1000) // 1ms minimum granularity
    }

    /// Compute the probe timeout (PTO).
    ///
    /// PTO = smoothed_rtt + max(4 * rttvar, kGranularity) + max_ack_delay
    pub fn pto(&self, max_ack_delay_us: u64) -> u64 {
        self.smoothed_rtt + (4 * self.rttvar).max(1000) + max_ack_delay_us
    }
}

/// Congestion controller state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CongestionState {
    /// Slow start phase.
    SlowStart,
    /// Congestion avoidance phase.
    CongestionAvoidance,
    /// Recovery after loss.
    Recovery,
}

/// NewReno congestion controller.
#[derive(Debug)]
pub struct CongestionController {
    /// Current state.
    state: CongestionState,
    /// Congestion window (bytes).
    congestion_window: u64,
    /// Slow start threshold.
    ssthresh: u64,
    /// Bytes in flight (sent but not yet acknowledged).
    bytes_in_flight: u64,
    /// Recovery start packet number.
    recovery_start_pn: Option<u64>,
    /// ECN-CE counter.
    ecn_ce_count: u64,
}

impl Default for CongestionController {
    fn default() -> Self {
        Self::new()
    }
}

impl CongestionController {
    /// Create a new congestion controller.
    pub fn new() -> Self {
        Self {
            state: CongestionState::SlowStart,
            congestion_window: INITIAL_WINDOW,
            ssthresh: u64::MAX,
            bytes_in_flight: 0,
            recovery_start_pn: None,
            ecn_ce_count: 0,
        }
    }

    /// Current congestion window.
    pub fn congestion_window(&self) -> u64 {
        self.congestion_window
    }

    /// Current state.
    pub fn state(&self) -> CongestionState {
        self.state
    }

    /// Bytes currently in flight.
    pub fn bytes_in_flight(&self) -> u64 {
        self.bytes_in_flight
    }

    /// Available congestion window (can send this many more bytes).
    pub fn available_window(&self) -> u64 {
        self.congestion_window.saturating_sub(self.bytes_in_flight)
    }

    /// Whether the congestion window allows sending.
    pub fn can_send(&self) -> bool {
        self.bytes_in_flight < self.congestion_window
    }

    /// Record that a packet was sent.
    pub fn on_packet_sent(&mut self, sent_bytes: u64) {
        self.bytes_in_flight += sent_bytes;
    }

    /// Record that a packet was acknowledged.
    pub fn on_packet_acked(&mut self, acked_bytes: u64, pn: u64) {
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(acked_bytes);

        // Don't increase window during recovery for packets sent before recovery
        if let Some(recovery_pn) = self.recovery_start_pn {
            if pn <= recovery_pn {
                return;
            }
            // Exiting recovery
            self.recovery_start_pn = None;
            self.state = if self.congestion_window < self.ssthresh {
                CongestionState::SlowStart
            } else {
                CongestionState::CongestionAvoidance
            };
        }

        match self.state {
            CongestionState::SlowStart => {
                self.congestion_window += acked_bytes;
                if self.congestion_window >= self.ssthresh {
                    self.state = CongestionState::CongestionAvoidance;
                }
            }
            CongestionState::CongestionAvoidance => {
                // Additive increase: cwnd += max_datagram_size * acked / cwnd
                self.congestion_window += (1472 * acked_bytes) / self.congestion_window;
            }
            CongestionState::Recovery => {
                // No increase during recovery
            }
        }
    }

    /// Record that a packet was lost.
    pub fn on_packet_lost(&mut self, lost_bytes: u64, largest_lost_pn: u64) {
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(lost_bytes);

        // Enter recovery if not already in it for this loss event
        if self.recovery_start_pn.is_none_or(|rpn| largest_lost_pn > rpn) {
            self.recovery_start_pn = Some(largest_lost_pn);
            self.state = CongestionState::Recovery;
            self.ssthresh = (self.congestion_window * LOSS_REDUCTION_NUMERATOR
                / LOSS_REDUCTION_DENOMINATOR)
                .max(MIN_WINDOW);
            self.congestion_window = self.ssthresh;
        }
    }

    /// Handle persistent congestion (no ACKs for > 3 * PTO).
    pub fn on_persistent_congestion(&mut self) {
        self.congestion_window = MIN_WINDOW;
        self.ssthresh = self.congestion_window;
        self.state = CongestionState::SlowStart;
        self.recovery_start_pn = None;
    }

    /// Handle ECN congestion experienced.
    pub fn on_ecn_ce(&mut self, ce_count: u64, sent_pn: u64) {
        if ce_count > self.ecn_ce_count {
            self.ecn_ce_count = ce_count;
            // Treat like a loss event
            self.on_packet_lost(0, sent_pn);
        }
    }

    /// Check for persistent congestion given PTO duration.
    pub fn is_persistent_congestion(
        &self,
        earliest_lost_time: u64,
        latest_lost_time: u64,
        pto: u64,
    ) -> bool {
        let duration = latest_lost_time.saturating_sub(earliest_lost_time);
        duration >= pto * PERSISTENT_CONGESTION_THRESHOLD
    }

    /// Reset bytes in flight (e.g., when retransmitting).
    pub fn remove_from_flight(&mut self, bytes: u64) {
        self.bytes_in_flight = self.bytes_in_flight.saturating_sub(bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- RTT Estimator ----

    #[test]
    fn test_rtt_new() {
        let rtt = RttEstimator::new();
        assert_eq!(rtt.smoothed_rtt(), INITIAL_RTT_US);
        assert!(!rtt.has_sample());
    }

    #[test]
    fn test_rtt_first_sample() {
        let mut rtt = RttEstimator::new();
        rtt.update(100_000, 0);
        assert_eq!(rtt.smoothed_rtt(), 100_000);
        assert_eq!(rtt.rttvar(), 50_000);
        assert_eq!(rtt.min_rtt(), 100_000);
        assert!(rtt.has_sample());
    }

    #[test]
    fn test_rtt_update_ewma() {
        let mut rtt = RttEstimator::new();
        rtt.update(100_000, 0);
        rtt.update(120_000, 0);
        // smoothed_rtt should be between 100k and 120k
        assert!(rtt.smoothed_rtt() > 100_000);
        assert!(rtt.smoothed_rtt() < 120_000);
    }

    #[test]
    fn test_rtt_min_rtt() {
        let mut rtt = RttEstimator::new();
        rtt.update(200_000, 0);
        rtt.update(100_000, 0);
        rtt.update(150_000, 0);
        assert_eq!(rtt.min_rtt(), 100_000);
    }

    #[test]
    fn test_rtt_ack_delay_adjustment() {
        let mut rtt = RttEstimator::new();
        rtt.update(50_000, 0); // First sample
        rtt.update(100_000, 20_000); // Second sample with ack delay
        // The adjusted RTT should be closer to 80k than 100k
        assert!(rtt.smoothed_rtt() < 90_000);
    }

    #[test]
    fn test_rtt_loss_delay() {
        let mut rtt = RttEstimator::new();
        rtt.update(100_000, 0);
        let delay = rtt.loss_delay();
        assert_eq!(delay, 112_500); // 9/8 * 100_000
    }

    #[test]
    fn test_rtt_loss_delay_minimum() {
        let mut rtt = RttEstimator::new();
        rtt.update(500, 0); // Very low RTT
        let delay = rtt.loss_delay();
        assert_eq!(delay, 1000); // 1ms minimum
    }

    #[test]
    fn test_rtt_pto() {
        let mut rtt = RttEstimator::new();
        rtt.update(100_000, 0);
        let pto = rtt.pto(25_000); // 25ms max ack delay
        // PTO = 100k + max(4*50k, 1k) + 25k = 100k + 200k + 25k = 325k
        assert_eq!(pto, 325_000);
    }

    // ---- Congestion Controller ----

    #[test]
    fn test_cc_new() {
        let cc = CongestionController::new();
        assert_eq!(cc.congestion_window(), INITIAL_WINDOW);
        assert_eq!(cc.state(), CongestionState::SlowStart);
        assert_eq!(cc.bytes_in_flight(), 0);
        assert!(cc.can_send());
    }

    #[test]
    fn test_cc_slow_start() {
        let mut cc = CongestionController::new();
        cc.on_packet_sent(1200);
        cc.on_packet_acked(1200, 0);
        // In slow start, window increases by acked_bytes
        assert_eq!(cc.congestion_window(), INITIAL_WINDOW + 1200);
        assert_eq!(cc.state(), CongestionState::SlowStart);
    }

    #[test]
    fn test_cc_transition_to_avoidance() {
        let mut cc = CongestionController::new();
        // Artificially set ssthresh
        cc.ssthresh = INITIAL_WINDOW + 500;

        cc.on_packet_sent(1200);
        cc.on_packet_acked(1200, 0);
        // Window now > ssthresh
        assert_eq!(cc.state(), CongestionState::CongestionAvoidance);
    }

    #[test]
    fn test_cc_on_loss() {
        let mut cc = CongestionController::new();
        cc.on_packet_sent(5000);
        cc.on_packet_lost(5000, 0);
        assert_eq!(cc.state(), CongestionState::Recovery);
        // Window halved (but at least MIN_WINDOW)
        let expected = (INITIAL_WINDOW / 2).max(MIN_WINDOW);
        assert_eq!(cc.congestion_window(), expected);
    }

    #[test]
    fn test_cc_no_double_reduction() {
        let mut cc = CongestionController::new();
        cc.on_packet_sent(5000);
        let initial_cwnd = cc.congestion_window();
        cc.on_packet_lost(2000, 5);
        let reduced = cc.congestion_window();
        assert!(reduced < initial_cwnd);

        // Loss of older packet during same recovery — no further reduction
        cc.on_packet_lost(1000, 3);
        assert_eq!(cc.congestion_window(), reduced);
    }

    #[test]
    fn test_cc_persistent_congestion() {
        let mut cc = CongestionController::new();
        cc.on_persistent_congestion();
        assert_eq!(cc.congestion_window(), MIN_WINDOW);
        assert_eq!(cc.state(), CongestionState::SlowStart);
    }

    #[test]
    fn test_cc_available_window() {
        let mut cc = CongestionController::new();
        cc.on_packet_sent(10_000);
        assert_eq!(cc.available_window(), INITIAL_WINDOW - 10_000);
    }

    #[test]
    fn test_cc_can_send() {
        let mut cc = CongestionController::new();
        assert!(cc.can_send());
        cc.on_packet_sent(INITIAL_WINDOW);
        assert!(!cc.can_send());
    }

    #[test]
    fn test_is_persistent_congestion() {
        let cc = CongestionController::new();
        let pto = 100_000;
        assert!(!cc.is_persistent_congestion(1000, 200_000, pto));
        assert!(cc.is_persistent_congestion(1000, 400_000, pto));
    }

    #[test]
    fn test_cc_recovery_exit() {
        let mut cc = CongestionController::new();
        cc.on_packet_sent(5000);
        cc.on_packet_lost(5000, 5);
        assert_eq!(cc.state(), CongestionState::Recovery);

        // ACK a packet sent AFTER recovery started
        cc.on_packet_sent(1000);
        cc.on_packet_acked(1000, 10);
        // Should exit recovery
        assert_ne!(cc.state(), CongestionState::Recovery);
    }

    #[test]
    fn test_cc_remove_from_flight() {
        let mut cc = CongestionController::new();
        cc.on_packet_sent(5000);
        cc.remove_from_flight(3000);
        assert_eq!(cc.bytes_in_flight(), 2000);
    }

    #[test]
    fn test_rtt_min_rtt_default() {
        let rtt = RttEstimator::new();
        assert_eq!(rtt.min_rtt(), INITIAL_RTT_US);
    }
}
