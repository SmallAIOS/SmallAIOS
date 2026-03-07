# Delta for USB Device/Gadget Controller Framework

## ADDED Requirements

### Requirement: USB Device Controller HAL Trait
The USB gadget framework SHALL define a `UsbDeviceController` HAL trait abstracting platform-specific USB device controller hardware.

#### Scenario: Initialize device controller
- WHEN SmallAIOS boots on a platform with USB device-mode hardware (Zynq, Jetson)
- THEN the device controller driver MUST initialize the hardware, configure the PHY, and prepare EP0 for control transfers
- AND MUST report the controller's capabilities (supported speeds, max endpoints, max packet sizes)

#### Scenario: Handle SET_ADDRESS from host
- WHEN the host sends a SET_ADDRESS standard request during enumeration
- THEN the device controller MUST program the assigned USB address into hardware
- AND MUST acknowledge the request with a zero-length STATUS stage

#### Scenario: Configure endpoints for a gadget function
- WHEN a gadget function requests endpoint configuration (direction, transfer type, max packet size)
- THEN the device controller MUST allocate hardware endpoint resources
- AND MUST configure the endpoint's FIFO and DMA settings
- AND MUST return the assigned endpoint address to the gadget function

### Requirement: Gadget Function Registration
The USB gadget framework SHALL support registration of multiple gadget functions that compose into a USB composite device.

#### Scenario: Register a single gadget function
- WHEN a gadget function (e.g., inference gadget) registers with the framework
- THEN the framework MUST compose a device descriptor with the function's VID/PID and class information
- AND MUST include the function's interface and endpoint descriptors in the configuration descriptor

#### Scenario: Register multiple gadget functions as composite device
- WHEN multiple gadget functions register (e.g., inference gadget + serial console)
- THEN the framework MUST compose a composite device descriptor with bDeviceClass == 0xEF (miscellaneous), bDeviceSubClass == 0x02, bDeviceProtocol == 0x01 (IAD)
- AND MUST include Interface Association Descriptors grouping each function's interfaces

### Requirement: Gadget Descriptor Composition
The USB gadget framework SHALL automatically compose valid USB descriptors from registered gadget functions.

#### Scenario: Compose configuration descriptor
- WHEN the host requests GET_DESCRIPTOR(Configuration)
- THEN the framework MUST assemble a complete configuration descriptor chain including all interface and endpoint descriptors from all registered gadget functions
- AND MUST set wTotalLength to the correct total size

#### Scenario: Handle GET_DESCRIPTOR(String)
- WHEN the host requests a string descriptor
- THEN the framework MUST return UTF-16LE encoded strings for manufacturer, product, and serial number
- AND MUST support string index 0 (language ID array) returning US English (0x0409)

### Requirement: Gadget Endpoint Data Transfer
The USB gadget framework SHALL provide data transfer primitives for gadget functions to send and receive data on their endpoints.

#### Scenario: Write data to IN endpoint
- WHEN a gadget function submits data for transmission on a bulk IN endpoint
- THEN the framework MUST queue the data for DMA transfer to the host
- AND MUST notify the gadget function upon transfer completion

#### Scenario: Read data from OUT endpoint
- WHEN the host sends data on a bulk OUT endpoint
- THEN the framework MUST receive the data via DMA
- AND MUST deliver the received data to the owning gadget function with the actual byte count

#### Scenario: Handle endpoint stall from gadget function
- WHEN a gadget function requests to stall an endpoint (e.g., unsupported request)
- THEN the framework MUST set the endpoint's STALL condition in hardware
- AND MUST automatically clear the stall when the host sends CLEAR_FEATURE(ENDPOINT_HALT)

### Requirement: USB Device Event Handling
The USB gadget framework SHALL process USB bus events (reset, suspend, resume, speed negotiation) and notify gadget functions.

#### Scenario: Handle USB bus reset
- WHEN the host resets the USB bus
- THEN the framework MUST re-initialize the device controller to address 0
- AND MUST notify all registered gadget functions of the reset
- AND MUST prepare EP0 for re-enumeration

#### Scenario: Handle speed negotiation
- WHEN the device controller completes speed negotiation with the host
- THEN the framework MUST report the negotiated speed (High-Speed, Full-Speed, SuperSpeed) to gadget functions
- AND gadget functions MUST adjust their max packet sizes accordingly
