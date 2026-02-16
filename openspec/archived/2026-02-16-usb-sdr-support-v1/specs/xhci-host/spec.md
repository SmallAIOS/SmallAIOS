# Delta for xHCI Host Controller Driver

## ADDED Requirements

### Requirement: xHCI Controller Discovery via PCIe
The xHCI driver SHALL discover xHCI host controllers by scanning the PCIe bus for devices with class code 0x0C, subclass 0x03, programming interface 0x30.

#### Scenario: Detect xHCI controller on PCIe bus
- WHEN SmallAIOS performs PCIe enumeration during boot
- THEN the xHCI driver MUST identify devices with class 0x0C/subclass 0x03/prog-if 0x30 as xHCI controllers
- AND MUST map the controller's BAR0 (memory-mapped registers) into kernel address space

#### Scenario: No xHCI controller present
- WHEN PCIe enumeration completes without finding an xHCI controller
- THEN the USB subsystem MUST gracefully report that no USB host is available
- AND MUST NOT panic or halt boot

### Requirement: xHCI Controller Initialization
The xHCI driver SHALL initialize the controller by resetting it, configuring operational registers, allocating device context arrays, and setting up command and event rings.

#### Scenario: Reset and initialize controller
- WHEN the xHCI driver initializes a discovered controller
- THEN it MUST write USBCMD.HCRST to reset the controller
- AND MUST wait for USBCMD.HCRST to clear and USBSTS.CNR to deassert (controller not ready → ready)
- AND MUST configure MaxSlotsEn, DCBAAP (Device Context Base Address Array Pointer), and CRCR (Command Ring Control Register)

#### Scenario: Allocate scratchpad buffers
- WHEN the controller's HCSPARAMS2 indicates scratchpad buffers are required
- THEN the driver MUST allocate the specified number of DMA-capable pages
- AND MUST populate the scratchpad buffer array at DCBAA slot 0

#### Scenario: Start controller
- WHEN initialization is complete
- THEN the driver MUST set USBCMD.RS (Run/Stop) to start the controller
- AND MUST verify USBSTS.HCH (Host Controller Halted) clears within 20ms

### Requirement: xHCI Command Ring
The xHCI driver SHALL implement the Command Ring for issuing commands to the controller (Enable Slot, Address Device, Configure Endpoint, etc.).

#### Scenario: Issue Enable Slot command
- WHEN a new device is detected on a port
- THEN the driver MUST enqueue an Enable Slot Command TRB on the Command Ring
- AND MUST ring the Command Ring doorbell (doorbell register 0)
- AND MUST wait for a Command Completion Event on the Event Ring with the assigned slot ID

#### Scenario: Issue Address Device command
- WHEN a slot has been enabled for a new device
- THEN the driver MUST allocate and initialize an Input Context with the slot context and EP0 context
- AND MUST enqueue an Address Device Command TRB pointing to the Input Context
- AND MUST receive the assigned USB device address in the Output Device Context

### Requirement: xHCI Transfer Rings
The xHCI driver SHALL implement per-endpoint Transfer Rings for queuing data transfers (control, bulk).

#### Scenario: Enqueue bulk transfer TRBs
- WHEN a driver submits a bulk transfer of N bytes
- THEN the xHCI driver MUST create one or more Normal TRBs on the endpoint's Transfer Ring
- AND MUST set the IOC (Interrupt on Completion) bit on the last TRB
- AND MUST ring the endpoint's doorbell register to notify the controller

#### Scenario: Handle Transfer Ring wrap-around
- WHEN the Transfer Ring's enqueue pointer reaches the last TRB slot
- THEN the driver MUST place a Link TRB pointing back to the start of the ring
- AND MUST toggle the Cycle State bit for the new cycle

#### Scenario: Process transfer completion
- WHEN the Event Ring contains a Transfer Event TRB
- THEN the driver MUST match the event to the originating Transfer TRB via the TRB pointer field
- AND MUST report the completion status and residual byte count to the requesting driver

### Requirement: xHCI Event Ring
The xHCI driver SHALL implement the Event Ring for receiving controller notifications (command completions, transfer completions, port status changes).

#### Scenario: Process port status change event
- WHEN the Event Ring contains a Port Status Change Event
- THEN the driver MUST read the affected port's PORTSC register
- AND MUST initiate device enumeration if a new device connection is detected (CCS bit set)

#### Scenario: Advance Event Ring Dequeue Pointer
- WHEN the driver has processed one or more events from the Event Ring
- THEN it MUST update the Event Ring Dequeue Pointer in the Interrupter register
- AND MUST clear the Event Handler Busy (EHB) bit to re-enable interrupts

### Requirement: xHCI Port Management
The xHCI driver SHALL detect device connection/disconnection on root hub ports and initiate port reset for new devices.

#### Scenario: Reset port on device connection
- WHEN a Port Status Change Event indicates a new device connection (CCS transition 0→1)
- THEN the driver MUST initiate a port reset by writing PORTSC.PR (Port Reset)
- AND MUST wait for the Port Reset Change (PRC) bit to indicate reset completion
- AND MUST determine the device speed from PORTSC.Port Speed field

#### Scenario: Handle device disconnection
- WHEN a Port Status Change Event indicates device disconnection (CCS transition 1→0)
- THEN the driver MUST disable the device slot via a Disable Slot Command
- AND MUST free all resources associated with the disconnected device
- AND MUST notify the USB core to remove the device from the registry

### Requirement: MSI-X Interrupt Support
The xHCI driver SHALL use MSI-X interrupts for event notification when available, falling back to polling if MSI-X is not supported.

#### Scenario: Configure MSI-X for xHCI events
- WHEN the xHCI controller's PCIe capability list includes MSI-X
- THEN the driver MUST configure MSI-X table entry 0 for the primary interrupter
- AND MUST enable MSI-X in the PCIe MSI-X capability register

#### Scenario: Fall back to polling mode
- WHEN MSI-X is not available
- THEN the driver MUST periodically poll the Event Ring for new events
- AND MUST use a polling interval of no more than 1ms to maintain responsiveness
