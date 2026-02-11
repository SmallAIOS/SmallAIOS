// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! FPGA fabric interrupt routing.
//!
//! Maps FPGA peripheral interrupt lines to CPU interrupt controller
//! sources (PLIC on RISC-V, GIC SPI on ARM). Supports shared interrupt
//! lines with handler chaining.

use core::fmt;

/// Maximum number of interrupt handlers that can be registered.
pub const MAX_HANDLERS: usize = 64;

/// Maximum number of handlers per shared interrupt line.
pub const MAX_SHARED_HANDLERS: usize = 8;

/// Interrupt controller type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptControllerType {
    /// RISC-V Platform-Level Interrupt Controller.
    Plic,
    /// ARM Generic Interrupt Controller (GICv2/v3).
    Gic,
}

/// Interrupt trigger type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriggerType {
    /// Level-sensitive (active high).
    LevelHigh,
    /// Level-sensitive (active low).
    LevelLow,
    /// Edge-triggered (rising edge).
    RisingEdge,
    /// Edge-triggered (falling edge).
    FallingEdge,
}

/// Interrupt routing entry.
#[derive(Debug, Clone, Copy)]
pub struct InterruptRoute {
    /// FPGA fabric interrupt line number.
    pub fabric_irq: u32,
    /// CPU interrupt controller source ID (PLIC source or GIC SPI number).
    pub controller_irq: u32,
    /// Priority (PLIC: 0-7, GIC: 0-255).
    pub priority: u8,
    /// Trigger type.
    pub trigger: TriggerType,
    /// Whether this handler is active.
    pub active: bool,
    /// Peripheral identifier for this handler.
    pub peripheral_id: u16,
}

/// Interrupt routing error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterruptError {
    /// Handler table is full.
    TableFull,
    /// Invalid interrupt number.
    InvalidIrq,
    /// Invalid priority value.
    InvalidPriority,
    /// Too many handlers on a shared line.
    TooManySharedHandlers,
    /// Handler not found for removal.
    HandlerNotFound,
    /// Controller type mismatch.
    ControllerMismatch,
}

impl fmt::Display for InterruptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterruptError::TableFull => write!(f, "interrupt handler table full"),
            InterruptError::InvalidIrq => write!(f, "invalid IRQ number"),
            InterruptError::InvalidPriority => write!(f, "invalid priority"),
            InterruptError::TooManySharedHandlers => write!(f, "too many shared handlers"),
            InterruptError::HandlerNotFound => write!(f, "handler not found"),
            InterruptError::ControllerMismatch => write!(f, "controller type mismatch"),
        }
    }
}

/// Interrupt dispatch result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DispatchResult {
    /// Interrupt was handled by a registered handler.
    Handled,
    /// Interrupt was not claimed by any handler (spurious).
    Unclaimed,
}

/// FPGA interrupt routing manager.
///
/// Maintains a table of interrupt routes mapping FPGA fabric interrupt
/// lines to CPU interrupt controller sources. Supports shared interrupt
/// lines where multiple peripherals share a single IRQ.
#[derive(Debug)]
pub struct InterruptRouter {
    /// Interrupt controller type.
    controller_type: InterruptControllerType,
    /// Routing entries.
    routes: [Option<InterruptRoute>; MAX_HANDLERS],
    /// Number of active routes.
    active_count: usize,
    /// Maximum priority value for this controller.
    max_priority: u8,
}

impl InterruptRouter {
    /// Create a new interrupt router for the given controller type.
    pub fn new(controller_type: InterruptControllerType) -> Self {
        let max_priority = match controller_type {
            InterruptControllerType::Plic => 7,
            InterruptControllerType::Gic => 255,
        };
        Self {
            controller_type,
            routes: [const { None }; MAX_HANDLERS],
            active_count: 0,
            max_priority,
        }
    }

    /// Interrupt controller type.
    pub fn controller_type(&self) -> InterruptControllerType {
        self.controller_type
    }

    /// Number of active routes.
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Register an interrupt handler for an FPGA peripheral.
    ///
    /// Maps `fabric_irq` to `controller_irq` on the CPU interrupt controller.
    /// Multiple peripherals can share the same `controller_irq` (shared line).
    pub fn register(
        &mut self,
        fabric_irq: u32,
        controller_irq: u32,
        priority: u8,
        trigger: TriggerType,
        peripheral_id: u16,
    ) -> Result<(), InterruptError> {
        if priority > self.max_priority {
            return Err(InterruptError::InvalidPriority);
        }

        // Check shared handler limit
        let shared_count = self.routes.iter().filter(|r| {
            r.as_ref()
                .is_some_and(|route| route.controller_irq == controller_irq && route.active)
        }).count();
        if shared_count >= MAX_SHARED_HANDLERS {
            return Err(InterruptError::TooManySharedHandlers);
        }

        // Find a free slot
        let slot = self.routes.iter().position(|r| r.is_none());
        let slot = match slot {
            Some(s) => s,
            None => return Err(InterruptError::TableFull),
        };

        self.routes[slot] = Some(InterruptRoute {
            fabric_irq,
            controller_irq,
            priority,
            trigger,
            active: true,
            peripheral_id,
        });
        self.active_count += 1;
        Ok(())
    }

    /// Unregister a handler by peripheral ID.
    pub fn unregister(&mut self, peripheral_id: u16) -> Result<(), InterruptError> {
        let mut found = false;
        for route in &mut self.routes {
            if let Some(ref r) = route {
                if r.peripheral_id == peripheral_id {
                    *route = None;
                    self.active_count -= 1;
                    found = true;
                }
            }
        }
        if found {
            Ok(())
        } else {
            Err(InterruptError::HandlerNotFound)
        }
    }

    /// Enable a handler by peripheral ID.
    pub fn enable(&mut self, peripheral_id: u16) -> Result<(), InterruptError> {
        for r in self.routes.iter_mut().flatten() {
            if r.peripheral_id == peripheral_id {
                r.active = true;
                return Ok(());
            }
        }
        Err(InterruptError::HandlerNotFound)
    }

    /// Disable a handler by peripheral ID.
    pub fn disable(&mut self, peripheral_id: u16) -> Result<(), InterruptError> {
        for r in self.routes.iter_mut().flatten() {
            if r.peripheral_id == peripheral_id {
                r.active = false;
                return Ok(());
            }
        }
        Err(InterruptError::HandlerNotFound)
    }

    /// Get all active handlers for a given controller IRQ (for shared line dispatch).
    ///
    /// Returns the peripheral IDs of all handlers registered on this IRQ.
    pub fn handlers_for_irq(&self, controller_irq: u32) -> [u16; MAX_SHARED_HANDLERS] {
        let mut result = [0u16; MAX_SHARED_HANDLERS];
        let mut idx = 0;
        for r in self.routes.iter().flatten() {
            if r.controller_irq == controller_irq && r.active && idx < MAX_SHARED_HANDLERS {
                result[idx] = r.peripheral_id;
                idx += 1;
            }
        }
        result
    }

    /// Count handlers for a given controller IRQ.
    pub fn handler_count_for_irq(&self, controller_irq: u32) -> usize {
        self.routes.iter().filter(|r| {
            r.as_ref()
                .is_some_and(|route| route.controller_irq == controller_irq && route.active)
        }).count()
    }

    /// Get a route by peripheral ID.
    pub fn get_route(&self, peripheral_id: u16) -> Option<&InterruptRoute> {
        self.routes.iter().filter_map(|r| r.as_ref()).find(|r| r.peripheral_id == peripheral_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn test_router_new_plic() {
        let router = InterruptRouter::new(InterruptControllerType::Plic);
        assert_eq!(router.controller_type(), InterruptControllerType::Plic);
        assert_eq!(router.active_count(), 0);
    }

    #[test]
    fn test_router_new_gic() {
        let router = InterruptRouter::new(InterruptControllerType::Gic);
        assert_eq!(router.controller_type(), InterruptControllerType::Gic);
    }

    #[test]
    fn test_register_handler() {
        let mut router = InterruptRouter::new(InterruptControllerType::Plic);
        router
            .register(0, 32, 5, TriggerType::LevelHigh, 1)
            .unwrap();
        assert_eq!(router.active_count(), 1);
    }

    #[test]
    fn test_register_invalid_priority_plic() {
        let mut router = InterruptRouter::new(InterruptControllerType::Plic);
        assert_eq!(
            router.register(0, 32, 8, TriggerType::LevelHigh, 1),
            Err(InterruptError::InvalidPriority)
        );
    }

    #[test]
    fn test_register_valid_priority_gic() {
        let mut router = InterruptRouter::new(InterruptControllerType::Gic);
        router
            .register(0, 32, 200, TriggerType::RisingEdge, 1)
            .unwrap();
    }

    #[test]
    fn test_register_shared_irq() {
        let mut router = InterruptRouter::new(InterruptControllerType::Plic);
        // Two peripherals sharing IRQ 32
        router
            .register(0, 32, 5, TriggerType::LevelHigh, 1)
            .unwrap();
        router
            .register(1, 32, 5, TriggerType::LevelHigh, 2)
            .unwrap();
        assert_eq!(router.handler_count_for_irq(32), 2);
    }

    #[test]
    fn test_register_too_many_shared() {
        let mut router = InterruptRouter::new(InterruptControllerType::Plic);
        for i in 0..MAX_SHARED_HANDLERS {
            router
                .register(i as u32, 32, 5, TriggerType::LevelHigh, i as u16 + 1)
                .unwrap();
        }
        assert_eq!(
            router.register(99, 32, 5, TriggerType::LevelHigh, 100),
            Err(InterruptError::TooManySharedHandlers)
        );
    }

    #[test]
    fn test_unregister() {
        let mut router = InterruptRouter::new(InterruptControllerType::Plic);
        router
            .register(0, 32, 5, TriggerType::LevelHigh, 1)
            .unwrap();
        router.unregister(1).unwrap();
        assert_eq!(router.active_count(), 0);
    }

    #[test]
    fn test_unregister_not_found() {
        let mut router = InterruptRouter::new(InterruptControllerType::Plic);
        assert_eq!(router.unregister(99), Err(InterruptError::HandlerNotFound));
    }

    #[test]
    fn test_enable_disable() {
        let mut router = InterruptRouter::new(InterruptControllerType::Plic);
        router
            .register(0, 32, 5, TriggerType::LevelHigh, 1)
            .unwrap();
        router.disable(1).unwrap();
        let route = router.get_route(1).unwrap();
        assert!(!route.active);

        router.enable(1).unwrap();
        let route = router.get_route(1).unwrap();
        assert!(route.active);
    }

    #[test]
    fn test_enable_not_found() {
        let mut router = InterruptRouter::new(InterruptControllerType::Plic);
        assert_eq!(router.enable(99), Err(InterruptError::HandlerNotFound));
    }

    #[test]
    fn test_handlers_for_irq() {
        let mut router = InterruptRouter::new(InterruptControllerType::Plic);
        router
            .register(0, 32, 5, TriggerType::LevelHigh, 10)
            .unwrap();
        router
            .register(1, 32, 5, TriggerType::LevelHigh, 20)
            .unwrap();
        router
            .register(2, 33, 3, TriggerType::RisingEdge, 30)
            .unwrap();

        let handlers = router.handlers_for_irq(32);
        assert_eq!(handlers[0], 10);
        assert_eq!(handlers[1], 20);
        assert_eq!(handlers[2], 0); // unused slot
    }

    #[test]
    fn test_handler_count_for_irq() {
        let mut router = InterruptRouter::new(InterruptControllerType::Gic);
        router.register(0, 64, 100, TriggerType::LevelHigh, 1).unwrap();
        router.register(1, 64, 100, TriggerType::LevelHigh, 2).unwrap();
        router.register(2, 65, 100, TriggerType::LevelHigh, 3).unwrap();
        assert_eq!(router.handler_count_for_irq(64), 2);
        assert_eq!(router.handler_count_for_irq(65), 1);
        assert_eq!(router.handler_count_for_irq(66), 0);
    }

    #[test]
    fn test_get_route() {
        let mut router = InterruptRouter::new(InterruptControllerType::Plic);
        router
            .register(5, 40, 7, TriggerType::FallingEdge, 42)
            .unwrap();
        let route = router.get_route(42).unwrap();
        assert_eq!(route.fabric_irq, 5);
        assert_eq!(route.controller_irq, 40);
        assert_eq!(route.priority, 7);
        assert_eq!(route.trigger, TriggerType::FallingEdge);
        assert_eq!(route.peripheral_id, 42);
        assert!(route.active);
    }

    #[test]
    fn test_get_route_not_found() {
        let router = InterruptRouter::new(InterruptControllerType::Plic);
        assert!(router.get_route(99).is_none());
    }

    #[test]
    fn test_trigger_type_variants() {
        assert_ne!(TriggerType::LevelHigh, TriggerType::LevelLow);
        assert_ne!(TriggerType::RisingEdge, TriggerType::FallingEdge);
    }

    #[test]
    fn test_dispatch_result_variants() {
        assert_ne!(DispatchResult::Handled, DispatchResult::Unclaimed);
    }

    #[test]
    fn test_error_display() {
        assert_eq!(format!("{}", InterruptError::TableFull), "interrupt handler table full");
        assert_eq!(format!("{}", InterruptError::InvalidIrq), "invalid IRQ number");
        assert_eq!(format!("{}", InterruptError::InvalidPriority), "invalid priority");
        assert_eq!(
            format!("{}", InterruptError::TooManySharedHandlers),
            "too many shared handlers"
        );
        assert_eq!(format!("{}", InterruptError::HandlerNotFound), "handler not found");
        assert_eq!(format!("{}", InterruptError::ControllerMismatch), "controller type mismatch");
    }
}
