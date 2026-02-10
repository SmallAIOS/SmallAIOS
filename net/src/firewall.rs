// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Built-in packet filter / firewall for SmallAIOS.
//!
//! Provides a rule-based [`Firewall`] that evaluates inbound and outbound
//! packets against an ordered set of [`FirewallRule`]s.  Rules are matched by
//! highest priority first; the first matching rule determines the action.  If
//! no rule matches, the firewall's default action is used.

use alloc::vec::Vec;

use crate::NetError;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

/// Action to take on a matched packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirewallAction {
    /// Permit the packet through.
    Allow,
    /// Silently discard the packet.
    Drop,
    /// Discard the packet and send an ICMP unreachable (or TCP RST) back.
    Reject,
}

/// Direction of packet flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FirewallDirection {
    /// Packet arriving from the network.
    Inbound,
    /// Packet leaving toward the network.
    Outbound,
}

/// Transport protocol selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Protocol {
    /// Matches any protocol.
    Any,
    /// TCP (protocol number 6).
    Tcp,
    /// UDP (protocol number 17).
    Udp,
    /// ICMP (protocol number 1).
    Icmp,
}

// ---------------------------------------------------------------------------
// FirewallRule
// ---------------------------------------------------------------------------

/// A single firewall rule describing traffic to match and the action to take.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FirewallRule {
    /// Direction this rule applies to.
    pub direction: FirewallDirection,
    /// Protocol this rule matches (`Any` matches all protocols).
    pub protocol: Protocol,
    /// If set, match only packets from this source port.
    pub src_port: Option<u16>,
    /// If set, match only packets to this destination port.
    pub dst_port: Option<u16>,
    /// If set, match only packets from this source IPv4 address.
    pub src_addr: Option<[u8; 4]>,
    /// If set, match only packets to this destination IPv4 address.
    pub dst_addr: Option<[u8; 4]>,
    /// Action to take when the rule matches.
    pub action: FirewallAction,
    /// Priority (higher numeric value = higher priority).
    pub priority: u16,
}

impl FirewallRule {
    /// Check whether this rule matches the given packet metadata.
    fn matches(
        &self,
        direction: FirewallDirection,
        protocol: Protocol,
        src_addr: [u8; 4],
        dst_addr: [u8; 4],
        src_port: Option<u16>,
        dst_port: Option<u16>,
    ) -> bool {
        // Direction must match exactly.
        if self.direction != direction {
            return false;
        }

        // Protocol: rule `Any` matches everything; otherwise must match.
        if self.protocol != Protocol::Any && self.protocol != protocol {
            return false;
        }

        // Source address (if the rule specifies one).
        if let Some(rule_src) = self.src_addr {
            if rule_src != src_addr {
                return false;
            }
        }

        // Destination address (if the rule specifies one).
        if let Some(rule_dst) = self.dst_addr {
            if rule_dst != dst_addr {
                return false;
            }
        }

        // Source port (if the rule specifies one).
        if let Some(rule_sp) = self.src_port {
            match src_port {
                Some(sp) if sp == rule_sp => {}
                _ => return false,
            }
        }

        // Destination port (if the rule specifies one).
        if let Some(rule_dp) = self.dst_port {
            match dst_port {
                Some(dp) if dp == rule_dp => {}
                _ => return false,
            }
        }

        true
    }
}

// ---------------------------------------------------------------------------
// Firewall
// ---------------------------------------------------------------------------

/// A rule-based packet filter.
///
/// Rules are stored in insertion order but evaluated by descending priority.
/// The first matching rule wins.  If no rule matches, [`Firewall::default_action`]
/// is returned.
pub struct Firewall {
    /// Ordered list of firewall rules.
    rules: Vec<FirewallRule>,
    /// Action when no rule matches.
    default_action: FirewallAction,
}

impl Firewall {
    /// Maximum number of rules the firewall can hold.
    pub const MAX_RULES: usize = 256;

    /// Create a new [`Firewall`] with the given default action and no rules.
    pub fn new(default_action: FirewallAction) -> Self {
        Firewall {
            rules: Vec::new(),
            default_action,
        }
    }

    /// Return the current default action.
    pub fn default_action(&self) -> FirewallAction {
        self.default_action
    }

    /// Add a rule to the firewall.
    ///
    /// Returns [`NetError::TableFull`] if [`Self::MAX_RULES`] has been reached.
    pub fn add_rule(&mut self, rule: FirewallRule) -> Result<(), NetError> {
        if self.rules.len() >= Self::MAX_RULES {
            return Err(NetError::TableFull);
        }
        self.rules.push(rule);
        Ok(())
    }

    /// Remove the rule at `index`.
    ///
    /// # Panics
    ///
    /// Panics if `index` is out of bounds (same semantics as [`Vec::remove`]).
    pub fn remove_rule(&mut self, index: usize) {
        self.rules.remove(index);
    }

    /// Return the number of rules currently installed.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Evaluate a packet against the rule set.
    ///
    /// The rule with the highest `priority` value that matches is selected.  If
    /// multiple rules share the same priority, the first one added wins.  If no
    /// rule matches, [`Firewall::default_action`] is returned.
    pub fn evaluate(
        &self,
        direction: FirewallDirection,
        protocol: Protocol,
        src_addr: [u8; 4],
        dst_addr: [u8; 4],
        src_port: Option<u16>,
        dst_port: Option<u16>,
    ) -> FirewallAction {
        let mut best: Option<&FirewallRule> = None;

        for rule in &self.rules {
            if rule.matches(direction, protocol, src_addr, dst_addr, src_port, dst_port) {
                match best {
                    None => best = Some(rule),
                    Some(current) if rule.priority > current.priority => {
                        best = Some(rule);
                    }
                    _ => {}
                }
            }
        }

        best.map_or(self.default_action, |r| r.action)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rule(
        direction: FirewallDirection,
        protocol: Protocol,
        action: FirewallAction,
        priority: u16,
    ) -> FirewallRule {
        FirewallRule {
            direction,
            protocol,
            src_port: None,
            dst_port: None,
            src_addr: None,
            dst_addr: None,
            action,
            priority,
        }
    }

    // -- Default action -----------------------------------------------------

    #[test]
    fn test_default_allow() {
        let fw = Firewall::new(FirewallAction::Allow);
        let action = fw.evaluate(
            FirewallDirection::Inbound,
            Protocol::Tcp,
            [10, 0, 0, 1],
            [192, 168, 1, 1],
            Some(5000),
            Some(80),
        );
        assert_eq!(action, FirewallAction::Allow);
    }

    #[test]
    fn test_default_drop() {
        let fw = Firewall::new(FirewallAction::Drop);
        let action = fw.evaluate(
            FirewallDirection::Outbound,
            Protocol::Udp,
            [192, 168, 1, 1],
            [10, 0, 0, 1],
            Some(53),
            Some(53),
        );
        assert_eq!(action, FirewallAction::Drop);
    }

    // -- Add / remove rules -------------------------------------------------

    #[test]
    fn test_add_rule() {
        let mut fw = Firewall::new(FirewallAction::Drop);
        assert_eq!(fw.rule_count(), 0);
        fw.add_rule(make_rule(
            FirewallDirection::Inbound,
            Protocol::Tcp,
            FirewallAction::Allow,
            100,
        ))
        .unwrap();
        assert_eq!(fw.rule_count(), 1);
    }

    #[test]
    fn test_remove_rule() {
        let mut fw = Firewall::new(FirewallAction::Drop);
        fw.add_rule(make_rule(
            FirewallDirection::Inbound,
            Protocol::Tcp,
            FirewallAction::Allow,
            100,
        ))
        .unwrap();
        fw.add_rule(make_rule(
            FirewallDirection::Outbound,
            Protocol::Udp,
            FirewallAction::Allow,
            50,
        ))
        .unwrap();
        assert_eq!(fw.rule_count(), 2);
        fw.remove_rule(0);
        assert_eq!(fw.rule_count(), 1);
    }

    // -- Rule evaluation ----------------------------------------------------

    #[test]
    fn test_evaluate_matching_rule() {
        let mut fw = Firewall::new(FirewallAction::Drop);
        fw.add_rule(make_rule(
            FirewallDirection::Inbound,
            Protocol::Tcp,
            FirewallAction::Allow,
            100,
        ))
        .unwrap();

        let action = fw.evaluate(
            FirewallDirection::Inbound,
            Protocol::Tcp,
            [10, 0, 0, 1],
            [192, 168, 1, 1],
            Some(5000),
            Some(80),
        );
        assert_eq!(action, FirewallAction::Allow);
    }

    #[test]
    fn test_evaluate_no_match_uses_default() {
        let mut fw = Firewall::new(FirewallAction::Drop);
        fw.add_rule(make_rule(
            FirewallDirection::Inbound,
            Protocol::Tcp,
            FirewallAction::Allow,
            100,
        ))
        .unwrap();

        // Outbound UDP — does not match the inbound TCP rule.
        let action = fw.evaluate(
            FirewallDirection::Outbound,
            Protocol::Udp,
            [10, 0, 0, 1],
            [192, 168, 1, 1],
            Some(5000),
            Some(53),
        );
        assert_eq!(action, FirewallAction::Drop);
    }

    #[test]
    fn test_evaluate_priority_ordering() {
        let mut fw = Firewall::new(FirewallAction::Drop);

        // Low-priority allow.
        fw.add_rule(make_rule(
            FirewallDirection::Inbound,
            Protocol::Any,
            FirewallAction::Allow,
            10,
        ))
        .unwrap();

        // High-priority reject.
        fw.add_rule(make_rule(
            FirewallDirection::Inbound,
            Protocol::Any,
            FirewallAction::Reject,
            100,
        ))
        .unwrap();

        let action = fw.evaluate(
            FirewallDirection::Inbound,
            Protocol::Tcp,
            [10, 0, 0, 1],
            [192, 168, 1, 1],
            Some(5000),
            Some(80),
        );
        // Higher priority rule wins.
        assert_eq!(action, FirewallAction::Reject);
    }

    #[test]
    fn test_evaluate_same_priority_first_wins() {
        let mut fw = Firewall::new(FirewallAction::Drop);

        fw.add_rule(make_rule(
            FirewallDirection::Inbound,
            Protocol::Any,
            FirewallAction::Allow,
            50,
        ))
        .unwrap();

        fw.add_rule(make_rule(
            FirewallDirection::Inbound,
            Protocol::Any,
            FirewallAction::Reject,
            50,
        ))
        .unwrap();

        let action = fw.evaluate(
            FirewallDirection::Inbound,
            Protocol::Tcp,
            [10, 0, 0, 1],
            [192, 168, 1, 1],
            None,
            None,
        );
        // First rule added at priority 50 wins.
        assert_eq!(action, FirewallAction::Allow);
    }

    #[test]
    fn test_evaluate_src_addr_filter() {
        let mut fw = Firewall::new(FirewallAction::Drop);
        let mut rule = make_rule(
            FirewallDirection::Inbound,
            Protocol::Tcp,
            FirewallAction::Allow,
            100,
        );
        rule.src_addr = Some([10, 0, 0, 1]);
        fw.add_rule(rule).unwrap();

        // Matching source.
        let action = fw.evaluate(
            FirewallDirection::Inbound,
            Protocol::Tcp,
            [10, 0, 0, 1],
            [192, 168, 1, 1],
            Some(5000),
            Some(80),
        );
        assert_eq!(action, FirewallAction::Allow);

        // Non-matching source.
        let action = fw.evaluate(
            FirewallDirection::Inbound,
            Protocol::Tcp,
            [10, 0, 0, 2],
            [192, 168, 1, 1],
            Some(5000),
            Some(80),
        );
        assert_eq!(action, FirewallAction::Drop);
    }

    #[test]
    fn test_evaluate_dst_port_filter() {
        let mut fw = Firewall::new(FirewallAction::Drop);
        let mut rule = make_rule(
            FirewallDirection::Inbound,
            Protocol::Tcp,
            FirewallAction::Allow,
            100,
        );
        rule.dst_port = Some(443);
        fw.add_rule(rule).unwrap();

        // Matching dst port.
        let action = fw.evaluate(
            FirewallDirection::Inbound,
            Protocol::Tcp,
            [10, 0, 0, 1],
            [192, 168, 1, 1],
            Some(5000),
            Some(443),
        );
        assert_eq!(action, FirewallAction::Allow);

        // Non-matching dst port.
        let action = fw.evaluate(
            FirewallDirection::Inbound,
            Protocol::Tcp,
            [10, 0, 0, 1],
            [192, 168, 1, 1],
            Some(5000),
            Some(80),
        );
        assert_eq!(action, FirewallAction::Drop);
    }

    #[test]
    fn test_table_full() {
        let mut fw = Firewall::new(FirewallAction::Drop);
        for i in 0..Firewall::MAX_RULES {
            fw.add_rule(make_rule(
                FirewallDirection::Inbound,
                Protocol::Any,
                FirewallAction::Allow,
                i as u16,
            ))
            .unwrap();
        }
        assert_eq!(fw.rule_count(), Firewall::MAX_RULES);

        let result = fw.add_rule(make_rule(
            FirewallDirection::Inbound,
            Protocol::Any,
            FirewallAction::Allow,
            999,
        ));
        assert_eq!(result, Err(NetError::TableFull));
    }

    #[test]
    fn test_protocol_any_matches_all() {
        let mut fw = Firewall::new(FirewallAction::Drop);
        fw.add_rule(make_rule(
            FirewallDirection::Inbound,
            Protocol::Any,
            FirewallAction::Allow,
            100,
        ))
        .unwrap();

        for proto in [Protocol::Tcp, Protocol::Udp, Protocol::Icmp, Protocol::Any] {
            let action = fw.evaluate(
                FirewallDirection::Inbound,
                proto,
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                None,
                None,
            );
            assert_eq!(action, FirewallAction::Allow);
        }
    }

    #[test]
    fn test_max_rules_constant() {
        assert_eq!(Firewall::MAX_RULES, 256);
    }
}
