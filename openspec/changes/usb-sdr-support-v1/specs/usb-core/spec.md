# Delta for USB Core Protocol Stack

## ADDED Requirements

### Requirement: USB Device Descriptor Parsing
The USB core SHALL parse USB device descriptors from raw byte buffers using zero-copy views, extracting vendor ID, product ID, device class, and number of configurations per USB 2.0 specification chapter 9.

#### Scenario: Parse a valid device descriptor
- WHEN the USB core receives an 18-byte device descriptor from a control transfer
- THEN the parser MUST extract bLength, bDescriptorType, bcdUSB, bDeviceClass, bDeviceSubClass, bDeviceProtocol, bMaxPacketSize0, idVendor, idProduct, bcdDevice, iManufacturer, iProduct, iSerialNumber, and bNumConfigurations
- AND MUST validate that bLength == 18 and bDescriptorType == 0x01

#### Scenario: Reject truncated device descriptor
- WHEN a device descriptor buffer is shorter than 18 bytes
- THEN the parser MUST return an error indicating insufficient data
- AND MUST NOT attempt to read beyond the buffer boundary

#### Scenario: Parse vendor-specific device
- WHEN the device descriptor contains bDeviceClass == 0xFF (vendor-specific)
- THEN the parser MUST accept the descriptor and report the class as vendor-specific
- AND MUST rely on interface descriptors for further classification

### Requirement: USB Configuration Descriptor Parsing
The USB core SHALL parse configuration descriptors including all nested interface and endpoint descriptors as a contiguous descriptor chain.

#### Scenario: Parse a composite device configuration
- WHEN the USB core receives a configuration descriptor with wTotalLength indicating multiple interfaces
- THEN the parser MUST iterate through the descriptor chain, extracting each interface descriptor and its associated endpoint descriptors
- AND MUST correctly handle variable-length descriptors by using bLength to advance the parse pointer

#### Scenario: Identify interface by class/subclass/protocol
- WHEN parsing interface descriptors within a configuration
- THEN the parser MUST expose bInterfaceClass, bInterfaceSubClass, and bInterfaceProtocol for each interface
- AND MUST allow callers to search for interfaces matching specific class/subclass/protocol tuples

#### Scenario: Extract endpoint descriptors
- WHEN an interface descriptor is followed by endpoint descriptors
- THEN the parser MUST extract bEndpointAddress (including direction bit), bmAttributes (transfer type), and wMaxPacketSize for each endpoint
- AND MUST correctly distinguish bulk, control, interrupt, and isochronous transfer types from the bmAttributes field

### Requirement: USB Device Enumeration
The USB core SHALL enumerate newly connected USB devices by assigning addresses, reading descriptors, and selecting configurations.

#### Scenario: Enumerate a new device on port reset
- WHEN a USB device is detected on a root hub port and the port reset completes
- THEN the USB core MUST assign a unique device address (1-127) via SET_ADDRESS control transfer
- AND MUST read the device descriptor via GET_DESCRIPTOR(Device) at the new address
- AND MUST read the full configuration descriptor via GET_DESCRIPTOR(Configuration)

#### Scenario: Select device configuration
- WHEN device and configuration descriptors have been successfully read
- THEN the USB core MUST send SET_CONFIGURATION with the first (or only) configuration value
- AND MUST notify registered device drivers of the newly available device with its VID, PID, and interface list

#### Scenario: Handle enumeration failure
- WHEN a device fails to respond to SET_ADDRESS or GET_DESCRIPTOR within 5 seconds
- THEN the USB core MUST disable the port
- AND MUST log the failure with the port number and error reason

### Requirement: USB Control Transfer
The USB core SHALL implement USB control transfers (SETUP → DATA → STATUS) for device configuration and vendor-specific commands.

#### Scenario: Send a vendor control transfer
- WHEN a driver submits a control transfer with bmRequestType indicating vendor type and device recipient
- THEN the USB core MUST enqueue a SETUP stage TRB with the 8-byte setup packet, followed by optional DATA stage TRBs, followed by a STATUS stage TRB
- AND MUST return the completion status and any received data to the caller

#### Scenario: Handle control transfer stall
- WHEN a device responds to a control transfer with a STALL handshake
- THEN the USB core MUST report the stall to the calling driver
- AND MUST clear the stall condition on endpoint 0 via CLEAR_FEATURE(ENDPOINT_HALT)

### Requirement: USB Bulk Transfer
The USB core SHALL implement USB bulk transfers for high-throughput data streaming between host and device.

#### Scenario: Submit a bulk IN transfer
- WHEN a driver requests a bulk IN transfer of N bytes on a given endpoint
- THEN the USB core MUST enqueue Normal TRBs on the endpoint's transfer ring
- AND MUST return the received data and actual byte count upon completion

#### Scenario: Submit multiple concurrent bulk transfers
- WHEN a driver submits multiple bulk transfers on the same endpoint before previous transfers complete
- THEN the USB core MUST queue them on the transfer ring in submission order
- AND MUST complete them in order, notifying the driver of each completion individually

#### Scenario: Handle bulk transfer short packet
- WHEN a bulk IN transfer receives fewer bytes than requested (short packet)
- THEN the USB core MUST treat the short packet as a successful completion
- AND MUST report the actual number of bytes received

### Requirement: USB Endpoint Management
The USB core SHALL track endpoint state (halted, active, idle) and support endpoint reset operations.

#### Scenario: Reset a halted endpoint
- WHEN a bulk or interrupt endpoint enters the halted state due to a STALL or transfer error
- THEN the driver MUST be able to reset the endpoint via a CLEAR_FEATURE(ENDPOINT_HALT) request
- AND the USB core MUST reset the endpoint's transfer ring dequeue pointer after the halt is cleared

#### Scenario: Query endpoint status
- WHEN a driver queries the status of an endpoint
- THEN the USB core MUST return whether the endpoint is active, halted, or idle

### Requirement: USB Device Registry
The USB core SHALL maintain a registry of enumerated devices, allowing drivers to claim devices by VID/PID or class/subclass/protocol match.

#### Scenario: Register a device driver by VID/PID
- WHEN a device driver registers interest in VID `0x1D50` / PID `0x6089`
- AND a device with matching VID/PID is enumerated
- THEN the USB core MUST notify the driver and provide a device handle for communication

#### Scenario: Register a device driver by interface class
- WHEN a device driver registers interest in interface class 0xFF (vendor-specific) with specific subclass/protocol
- AND a device with a matching interface is enumerated
- THEN the USB core MUST notify the driver and provide the matching interface number

#### Scenario: Prevent double-claiming of interfaces
- WHEN two drivers attempt to claim the same interface on a device
- THEN the USB core MUST allow only the first driver to claim it
- AND MUST return an error to the second driver

### Requirement: Clean-Room Implementation
All USB core implementations SHALL be clean-room developed from the USB 2.0 and USB 3.x public specifications (usb.org) without reference to proprietary source code.

#### Scenario: Verify clean-room provenance
- WHEN the USB core module is submitted for review
- THEN the implementation MUST include a clean-room attestation document listing only the USB 2.0 specification (usb.org), USB 3.2 specification, and xHCI specification as reference sources
- AND MUST NOT contain code derived from proprietary USB stack implementations (Linux usbcore, libusb internals, etc.)
