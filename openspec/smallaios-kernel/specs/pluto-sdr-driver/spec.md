# Delta for ADALM-PLUTO SDR Driver

## ADDED Requirements

### Requirement: ADALM-PLUTO Device Detection
The PlutoSDR driver SHALL detect ADALM-PLUTO devices by matching USB vendor ID `0x0456` (Analog Devices) and product ID `0xb673` during enumeration.

#### Scenario: Detect PlutoSDR on USB bus
- WHEN a USB device with VID `0x0456` and PID `0xb673` is enumerated
- THEN the PlutoSDR driver MUST claim the vendor-specific IIO interface (not the CDC, mass storage, or DFU interfaces)
- AND MUST identify the IIO interface by scanning interface descriptors for class 0xFF with the expected endpoint configuration (bulk IN + bulk OUT pair)

#### Scenario: Discover IIO context
- WHEN the PlutoSDR device is claimed
- THEN the driver MUST send the IIOD `PRINT` command over the vendor bulk endpoint
- AND MUST parse the response to confirm the presence of the `ad9361-phy` and `cf-ad9361-lpc` IIO devices

### Requirement: IIOD Text Protocol Client
The PlutoSDR driver SHALL implement the IIOD v0.x text protocol for communicating with the PlutoSDR's onboard iiod daemon over vendor USB bulk endpoints.

#### Scenario: Send IIOD command and receive response
- WHEN the driver sends an IIOD text command (e.g., `READ ad9361-phy INPUT voltage0 sampling_frequency\n`)
- THEN the driver MUST write the command bytes to the bulk OUT endpoint
- AND MUST read the response from the bulk IN endpoint
- AND MUST parse the response as an integer return code followed by optional data payload

#### Scenario: Write an IIO attribute
- WHEN the application sets a device attribute (e.g., set RX LO frequency to 915 MHz)
- THEN the driver MUST send `WRITE ad9361-phy OUTPUT altvoltage0 frequency\n9\n915000000` (attribute name, byte count, value)
- AND MUST verify the response return code is non-negative (success)

#### Scenario: Read an IIO attribute
- WHEN the application reads a device attribute (e.g., current sample rate)
- THEN the driver MUST send `READ cf-ad9361-lpc INPUT voltage0 sampling_frequency\n`
- AND MUST parse the response return code and value string

#### Scenario: Handle IIOD error response
- WHEN the IIOD daemon returns a negative return code
- THEN the driver MUST map the error code to a driver error type (e.g., -EINVAL → InvalidConfig, -EBUSY → DeviceBusy)
- AND MUST NOT proceed with dependent operations

### Requirement: AD9363 RF Configuration
The PlutoSDR driver SHALL configure the AD9363 RF transceiver via IIO attribute writes for frequency, sample rate, gain, and bandwidth.

#### Scenario: Set RX frequency
- WHEN the application requests an RX center frequency of 915 MHz
- THEN the driver MUST write `915000000` to attribute `frequency` on device `ad9361-phy`, output channel `altvoltage0` (RX LO)
- AND MUST verify the frequency is within the AD9363's range (325 MHz to 3.8 GHz)

#### Scenario: Set sample rate
- WHEN the application requests a sample rate of 2.5 MSPS
- THEN the driver MUST write `2500000` to attribute `sampling_frequency` on device `cf-ad9361-lpc`, input channel `voltage0`
- AND the AD9363 MUST automatically configure its decimation/interpolation filters

#### Scenario: Set gain mode and value
- WHEN the application requests manual gain of 30 dB on the RX channel
- THEN the driver MUST write `manual` to attribute `gain_control_mode` on `ad9361-phy`, input channel `voltage0`
- AND MUST write `30.000000` to attribute `hardwaregain` on the same channel
- AND MUST verify the gain is within the valid range (approximately -1 to 73 dB)

#### Scenario: Set RF bandwidth
- WHEN the application requests an RF bandwidth of 2 MHz
- THEN the driver MUST write `2000000` to attribute `rf_bandwidth` on `ad9361-phy`, input channel `voltage0`

#### Scenario: Reject out-of-range frequency
- WHEN the application requests a frequency of 10 GHz (exceeds AD9363 maximum of 3.8 GHz)
- THEN the driver MUST return an error indicating frequency out of range
- AND MUST NOT send the configuration to the device

### Requirement: PlutoSDR IQ Streaming
The PlutoSDR driver SHALL support continuous IQ sample streaming via the IIOD buffer protocol.

#### Scenario: Open streaming buffer
- WHEN the application requests to start IQ streaming with a buffer size of 32768 samples
- THEN the driver MUST send `OPEN cf-ad9361-lpc 32768 3 \n` (device, sample count, channel mask for I+Q)
- AND MUST verify the response indicates success

#### Scenario: Read IQ samples
- WHEN the streaming buffer is open
- THEN the driver MUST send `READBUF cf-ad9361-lpc <bytes>\n` to request a batch of IQ samples
- AND MUST receive the raw IQ sample data (16-bit signed integers, interleaved I,Q,I,Q)
- AND MUST deliver the samples to the registered callback

#### Scenario: Close streaming buffer
- WHEN the application requests to stop streaming
- THEN the driver MUST send `CLOSE cf-ad9361-lpc\n`
- AND MUST ensure no further READBUF commands are issued after close

#### Scenario: Handle streaming underrun
- WHEN READBUF returns fewer bytes than requested
- THEN the driver MUST deliver the available data to the callback
- AND MUST continue issuing READBUF commands for the remaining data

### Requirement: PlutoSDR Full-Duplex Support
The PlutoSDR driver SHALL support simultaneous transmit and receive since the AD9363 is a full-duplex transceiver.

#### Scenario: Configure independent TX and RX paths
- WHEN the application configures TX on one frequency and RX on another
- THEN the driver MUST independently configure `altvoltage1` (TX LO) and `altvoltage0` (RX LO)
- AND MUST support WRITEBUF for TX samples concurrently with READBUF for RX samples

### Requirement: IIOD Timeout Configuration
The PlutoSDR driver SHALL configure the IIOD communication timeout.

#### Scenario: Set IIOD timeout
- WHEN the driver initializes communication with the PlutoSDR
- THEN it MUST send `TIMEOUT 5000\n` to set a 5-second I/O timeout
- AND MUST handle timeout errors gracefully on subsequent commands
