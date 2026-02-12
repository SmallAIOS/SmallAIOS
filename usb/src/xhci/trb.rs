// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Transfer Request Block (TRB) types for xHCI.
//!
//! TRBs are 16-byte structures used for communication between the host
//! driver and the xHCI controller via shared memory rings.

/// TRB size in bytes.
pub const TRB_SIZE: usize = 16;

// ---------------------------------------------------------------------------
// TRB type codes (xHCI Table 6-91)
// ---------------------------------------------------------------------------

/// TRB type code (bits 15:10 of the Control field, dword 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TrbType {
    // Transfer TRB types
    Normal = 1,
    SetupStage = 2,
    DataStage = 3,
    StatusStage = 4,
    Isoch = 5,
    Link = 6,
    EventData = 7,
    NoOp = 8,

    // Command TRB types
    EnableSlotCommand = 9,
    DisableSlotCommand = 10,
    AddressDeviceCommand = 11,
    ConfigureEndpointCommand = 12,
    EvaluateContextCommand = 13,
    ResetEndpointCommand = 14,
    StopEndpointCommand = 15,
    SetTrDequeuePointerCommand = 16,
    ResetDeviceCommand = 17,
    NoOpCommand = 23,

    // Event TRB types
    TransferEvent = 32,
    CommandCompletionEvent = 33,
    PortStatusChangeEvent = 34,
    HostControllerEvent = 37,
}

impl TrbType {
    /// Parse a TRB type from the control field value.
    pub fn from_control(control: u32) -> Option<Self> {
        let code = ((control >> 10) & 0x3F) as u8;
        match code {
            1 => Some(TrbType::Normal),
            2 => Some(TrbType::SetupStage),
            3 => Some(TrbType::DataStage),
            4 => Some(TrbType::StatusStage),
            5 => Some(TrbType::Isoch),
            6 => Some(TrbType::Link),
            7 => Some(TrbType::EventData),
            8 => Some(TrbType::NoOp),
            9 => Some(TrbType::EnableSlotCommand),
            10 => Some(TrbType::DisableSlotCommand),
            11 => Some(TrbType::AddressDeviceCommand),
            12 => Some(TrbType::ConfigureEndpointCommand),
            13 => Some(TrbType::EvaluateContextCommand),
            14 => Some(TrbType::ResetEndpointCommand),
            15 => Some(TrbType::StopEndpointCommand),
            16 => Some(TrbType::SetTrDequeuePointerCommand),
            17 => Some(TrbType::ResetDeviceCommand),
            23 => Some(TrbType::NoOpCommand),
            32 => Some(TrbType::TransferEvent),
            33 => Some(TrbType::CommandCompletionEvent),
            34 => Some(TrbType::PortStatusChangeEvent),
            37 => Some(TrbType::HostControllerEvent),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Completion codes (xHCI Table 6-90)
// ---------------------------------------------------------------------------

/// TRB completion code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CompletionCode {
    Invalid = 0,
    Success = 1,
    DataBufferError = 2,
    BabbleDetected = 3,
    UsbTransactionError = 4,
    TrbError = 5,
    StallError = 6,
    ShortPacket = 13,
    EventRingFullError = 21,
    CommandRingStoppedError = 24,
    CommandAborted = 25,
    Stopped = 26,
    StoppedLengthInvalid = 27,
}

impl CompletionCode {
    /// Parse a completion code from byte value.
    pub fn from_byte(b: u8) -> Self {
        match b {
            0 => CompletionCode::Invalid,
            1 => CompletionCode::Success,
            2 => CompletionCode::DataBufferError,
            3 => CompletionCode::BabbleDetected,
            4 => CompletionCode::UsbTransactionError,
            5 => CompletionCode::TrbError,
            6 => CompletionCode::StallError,
            13 => CompletionCode::ShortPacket,
            21 => CompletionCode::EventRingFullError,
            24 => CompletionCode::CommandRingStoppedError,
            25 => CompletionCode::CommandAborted,
            26 => CompletionCode::Stopped,
            27 => CompletionCode::StoppedLengthInvalid,
            _ => CompletionCode::Invalid,
        }
    }

    /// Returns true if this code represents a successful completion.
    pub fn is_success(&self) -> bool {
        matches!(self, CompletionCode::Success | CompletionCode::ShortPacket)
    }
}

// ---------------------------------------------------------------------------
// Generic TRB
// ---------------------------------------------------------------------------

/// Raw 16-byte TRB (Transfer Request Block).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C, align(16))]
pub struct Trb {
    /// Parameter field (dwords 0-1, 64-bit).
    pub parameter: u64,
    /// Status field (dword 2).
    pub status: u32,
    /// Control field (dword 3): type, cycle bit, flags.
    pub control: u32,
}

impl Trb {
    /// Create a zeroed TRB.
    pub const fn zeroed() -> Self {
        Self {
            parameter: 0,
            status: 0,
            control: 0,
        }
    }

    /// Get the TRB type from the control field.
    pub fn trb_type(&self) -> Option<TrbType> {
        TrbType::from_control(self.control)
    }

    /// Get the cycle bit (bit 0 of control).
    pub fn cycle_bit(&self) -> bool {
        self.control & 1 != 0
    }

    /// Set the cycle bit.
    pub fn set_cycle_bit(&mut self, cycle: bool) {
        if cycle {
            self.control |= 1;
        } else {
            self.control &= !1;
        }
    }

    /// Build the control field from type, cycle bit, and flags.
    pub fn make_control(trb_type: TrbType, cycle: bool, flags: u32) -> u32 {
        let mut ctrl = ((trb_type as u32) << 10) | flags;
        if cycle {
            ctrl |= 1;
        }
        ctrl
    }

    // -- Normal TRB (for bulk/interrupt transfers) --------------------------

    /// Create a Normal TRB for a bulk transfer.
    ///
    /// `data_ptr`: physical address of the data buffer.
    /// `length`: transfer length in bytes.
    /// `cycle`: current producer cycle state.
    /// `ioc`: interrupt on completion.
    pub fn normal(data_ptr: u64, length: u32, cycle: bool, ioc: bool) -> Self {
        let mut flags = 0u32;
        if ioc {
            flags |= 1 << 5; // IOC bit
        }
        Self {
            parameter: data_ptr,
            status: length & 0x1FFFF, // bits 16:0 = TRB Transfer Length
            control: Self::make_control(TrbType::Normal, cycle, flags),
        }
    }

    // -- Setup Stage TRB (control transfer SETUP phase) ---------------------

    /// Create a Setup Stage TRB.
    ///
    /// `setup_data`: 8-byte USB setup packet encoded as u64 (little-endian).
    /// `transfer_type`: 0=No Data, 2=OUT Data, 3=IN Data.
    /// `cycle`: current producer cycle state.
    pub fn setup_stage(setup_data: u64, transfer_type: u8, cycle: bool) -> Self {
        let flags = (transfer_type as u32 & 0x3) << 16; // TRT field
        // IDT (Immediate Data) bit = bit 6
        let flags = flags | (1 << 6);
        Self {
            parameter: setup_data,
            status: 8, // Always 8 bytes for setup packet
            control: Self::make_control(TrbType::SetupStage, cycle, flags),
        }
    }

    // -- Data Stage TRB (control transfer DATA phase) -----------------------

    /// Create a Data Stage TRB.
    ///
    /// `data_ptr`: physical address of the data buffer.
    /// `length`: transfer length.
    /// `direction_in`: true for IN (device→host), false for OUT.
    /// `cycle`: current producer cycle state.
    pub fn data_stage(data_ptr: u64, length: u32, direction_in: bool, cycle: bool) -> Self {
        let mut flags = 0u32;
        if direction_in {
            flags |= 1 << 16; // DIR bit
        }
        Self {
            parameter: data_ptr,
            status: length & 0x1FFFF,
            control: Self::make_control(TrbType::DataStage, cycle, flags),
        }
    }

    // -- Status Stage TRB (control transfer STATUS phase) -------------------

    /// Create a Status Stage TRB.
    ///
    /// `direction_in`: true if status is IN direction (for OUT data transfers).
    /// `cycle`: current producer cycle state.
    /// `ioc`: interrupt on completion.
    pub fn status_stage(direction_in: bool, cycle: bool, ioc: bool) -> Self {
        let mut flags = 0u32;
        if direction_in {
            flags |= 1 << 16; // DIR bit
        }
        if ioc {
            flags |= 1 << 5; // IOC bit
        }
        Self {
            parameter: 0,
            status: 0,
            control: Self::make_control(TrbType::StatusStage, cycle, flags),
        }
    }

    // -- Link TRB (ring wrap-around) ----------------------------------------

    /// Create a Link TRB pointing to the ring segment start.
    ///
    /// `ring_base`: physical address of the ring start.
    /// `toggle_cycle`: true to toggle the cycle bit on wrap.
    /// `cycle`: current producer cycle state.
    pub fn link(ring_base: u64, toggle_cycle: bool, cycle: bool) -> Self {
        let mut flags = 0u32;
        if toggle_cycle {
            flags |= 1 << 1; // Toggle Cycle bit
        }
        Self {
            parameter: ring_base,
            status: 0,
            control: Self::make_control(TrbType::Link, cycle, flags),
        }
    }

    // -- Command TRBs -------------------------------------------------------

    /// Create an Enable Slot Command TRB.
    pub fn enable_slot_command(cycle: bool) -> Self {
        Self {
            parameter: 0,
            status: 0,
            control: Self::make_control(TrbType::EnableSlotCommand, cycle, 0),
        }
    }

    /// Create a Disable Slot Command TRB.
    pub fn disable_slot_command(slot_id: u8, cycle: bool) -> Self {
        let flags = (slot_id as u32) << 24; // Slot ID in bits 31:24
        Self {
            parameter: 0,
            status: 0,
            control: Self::make_control(TrbType::DisableSlotCommand, cycle, flags),
        }
    }

    /// Create an Address Device Command TRB.
    ///
    /// `input_context_ptr`: physical address of the Input Context.
    /// `slot_id`: device slot ID.
    /// `bsr`: Block Set-address Request (if true, skip SET_ADDRESS).
    pub fn address_device_command(
        input_context_ptr: u64,
        slot_id: u8,
        bsr: bool,
        cycle: bool,
    ) -> Self {
        let mut flags = (slot_id as u32) << 24;
        if bsr {
            flags |= 1 << 9; // BSR bit
        }
        Self {
            parameter: input_context_ptr,
            status: 0,
            control: Self::make_control(TrbType::AddressDeviceCommand, cycle, flags),
        }
    }

    /// Create a Configure Endpoint Command TRB.
    pub fn configure_endpoint_command(
        input_context_ptr: u64,
        slot_id: u8,
        cycle: bool,
    ) -> Self {
        let flags = (slot_id as u32) << 24;
        Self {
            parameter: input_context_ptr,
            status: 0,
            control: Self::make_control(TrbType::ConfigureEndpointCommand, cycle, flags),
        }
    }

    // -- Event TRB field extraction -----------------------------------------

    /// Extract completion code from an event TRB's status field.
    pub fn completion_code(&self) -> CompletionCode {
        CompletionCode::from_byte(((self.status >> 24) & 0xFF) as u8)
    }

    /// Extract slot ID from an event TRB's control field (bits 31:24).
    pub fn slot_id(&self) -> u8 {
        ((self.control >> 24) & 0xFF) as u8
    }

    /// Extract port ID from a Port Status Change Event (bits 31:24 of parameter).
    pub fn port_id(&self) -> u8 {
        ((self.parameter >> 24) & 0xFF) as u8
    }

    /// Extract TRB pointer from a Transfer Event (parameter field).
    pub fn trb_pointer(&self) -> u64 {
        self.parameter & !0xF // Low 4 bits are reserved
    }

    /// Extract transfer length residual from status field (bits 23:0).
    pub fn transfer_length_residual(&self) -> u32 {
        self.status & 0xFFFFFF
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trb_zeroed() {
        let trb = Trb::zeroed();
        assert_eq!(trb.parameter, 0);
        assert_eq!(trb.status, 0);
        assert_eq!(trb.control, 0);
        assert!(!trb.cycle_bit());
    }

    #[test]
    fn test_trb_cycle_bit() {
        let mut trb = Trb::zeroed();
        assert!(!trb.cycle_bit());
        trb.set_cycle_bit(true);
        assert!(trb.cycle_bit());
        trb.set_cycle_bit(false);
        assert!(!trb.cycle_bit());
    }

    #[test]
    fn test_trb_normal() {
        let trb = Trb::normal(0x1000_0000, 512, true, true);
        assert_eq!(trb.parameter, 0x1000_0000);
        assert_eq!(trb.status & 0x1FFFF, 512);
        assert!(trb.cycle_bit());
        assert_eq!(trb.trb_type(), Some(TrbType::Normal));
        // IOC bit (bit 5 of flags in control)
        assert!(trb.control & (1 << 5) != 0);
    }

    #[test]
    fn test_trb_setup_stage() {
        let setup_data = 0x0001_0006_0080u64; // GET_DESCRIPTOR example
        let trb = Trb::setup_stage(setup_data, 3, true); // IN data
        assert_eq!(trb.parameter, setup_data);
        assert_eq!(trb.status, 8);
        assert_eq!(trb.trb_type(), Some(TrbType::SetupStage));
        assert!(trb.cycle_bit());
    }

    #[test]
    fn test_trb_data_stage_in() {
        let trb = Trb::data_stage(0x2000_0000, 18, true, true);
        assert_eq!(trb.parameter, 0x2000_0000);
        assert_eq!(trb.status & 0x1FFFF, 18);
        assert_eq!(trb.trb_type(), Some(TrbType::DataStage));
        // DIR bit
        assert!(trb.control & (1 << 16) != 0);
    }

    #[test]
    fn test_trb_data_stage_out() {
        let trb = Trb::data_stage(0x3000_0000, 64, false, true);
        assert!(trb.control & (1 << 16) == 0); // DIR = OUT
    }

    #[test]
    fn test_trb_status_stage() {
        let trb = Trb::status_stage(true, true, true);
        assert_eq!(trb.trb_type(), Some(TrbType::StatusStage));
        assert!(trb.control & (1 << 5) != 0); // IOC
        assert!(trb.control & (1 << 16) != 0); // DIR = IN
    }

    #[test]
    fn test_trb_link() {
        let trb = Trb::link(0x5000_0000, true, true);
        assert_eq!(trb.parameter, 0x5000_0000);
        assert_eq!(trb.trb_type(), Some(TrbType::Link));
        assert!(trb.control & (1 << 1) != 0); // Toggle Cycle
    }

    #[test]
    fn test_trb_enable_slot_command() {
        let trb = Trb::enable_slot_command(true);
        assert_eq!(trb.trb_type(), Some(TrbType::EnableSlotCommand));
        assert!(trb.cycle_bit());
    }

    #[test]
    fn test_trb_disable_slot_command() {
        let trb = Trb::disable_slot_command(3, true);
        assert_eq!(trb.trb_type(), Some(TrbType::DisableSlotCommand));
        assert_eq!(trb.slot_id(), 3);
    }

    #[test]
    fn test_trb_address_device_command() {
        let trb = Trb::address_device_command(0xA000, 5, false, true);
        assert_eq!(trb.parameter, 0xA000);
        assert_eq!(trb.trb_type(), Some(TrbType::AddressDeviceCommand));
        assert_eq!(trb.slot_id(), 5);
    }

    #[test]
    fn test_trb_configure_endpoint_command() {
        let trb = Trb::configure_endpoint_command(0xB000, 2, true);
        assert_eq!(trb.trb_type(), Some(TrbType::ConfigureEndpointCommand));
        assert_eq!(trb.slot_id(), 2);
    }

    #[test]
    fn test_completion_code_success() {
        assert!(CompletionCode::Success.is_success());
        assert!(CompletionCode::ShortPacket.is_success());
        assert!(!CompletionCode::StallError.is_success());
        assert!(!CompletionCode::Invalid.is_success());
    }

    #[test]
    fn test_completion_code_from_byte() {
        assert_eq!(CompletionCode::from_byte(1), CompletionCode::Success);
        assert_eq!(CompletionCode::from_byte(6), CompletionCode::StallError);
        assert_eq!(CompletionCode::from_byte(13), CompletionCode::ShortPacket);
        assert_eq!(CompletionCode::from_byte(255), CompletionCode::Invalid);
    }

    #[test]
    fn test_trb_event_field_extraction() {
        // Simulate a command completion event.
        let mut trb = Trb::zeroed();
        trb.parameter = 0x1234_0000; // TRB pointer
        trb.status = 1u32 << 24; // Completion code = Success
        trb.control = Trb::make_control(TrbType::CommandCompletionEvent, true, 3 << 24); // Slot 3

        assert_eq!(trb.completion_code(), CompletionCode::Success);
        assert_eq!(trb.slot_id(), 3);
        assert_eq!(trb.trb_pointer(), 0x1234_0000);
    }

    #[test]
    fn test_trb_port_status_change_event() {
        let mut trb = Trb::zeroed();
        trb.parameter = 2u64 << 24; // Port ID = 2
        trb.control = Trb::make_control(TrbType::PortStatusChangeEvent, true, 0);

        assert_eq!(trb.port_id(), 2);
        assert_eq!(trb.trb_type(), Some(TrbType::PortStatusChangeEvent));
    }

    #[test]
    fn test_trb_type_from_control_unknown() {
        assert!(TrbType::from_control(0xFF << 10).is_none());
    }

    #[test]
    fn test_trb_size() {
        assert_eq!(core::mem::size_of::<Trb>(), TRB_SIZE);
    }
}
