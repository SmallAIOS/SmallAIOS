# Delta for HackRF One SDR Driver

## ADDED Requirements

### Requirement: HackRF One Device Detection
The HackRF driver SHALL detect HackRF One devices by matching USB vendor ID `0x1D50` and product ID `0x6089` during enumeration.

#### Scenario: Detect HackRF One on USB bus
- WHEN a USB device with VID `0x1D50` and PID `0x6089` is enumerated
- THEN the HackRF driver MUST claim the device
- AND MUST read the board ID via vendor request `BOARD_ID_READ` (code 14) to confirm it is a HackRF One (board ID 2)
- AND MUST read the firmware version via vendor request `VERSION_STRING_READ` (code 15)

#### Scenario: Reject non-HackRF device
- WHEN a USB device with VID `0x1D50` but a different PID is enumerated
- THEN the HackRF driver MUST NOT claim the device

### Requirement: HackRF RF Configuration
The HackRF driver SHALL configure the RF front-end (frequency, sample rate, gains, bandwidth) via USB vendor control transfers.

#### Scenario: Set center frequency
- WHEN the application requests a center frequency of 433 MHz
- THEN the driver MUST send vendor request `SET_FREQ` (code 16) with the frequency encoded as a 64-bit value in Hz in the data payload
- AND MUST verify the request completes without error

#### Scenario: Set sample rate
- WHEN the application requests a sample rate of 10 MSPS
- THEN the driver MUST send vendor request `SAMPLE_RATE_SET` (code 6) with the sample rate and baseband filter bandwidth encoded in the data payload
- AND MUST automatically set the baseband filter bandwidth to match the sample rate if not explicitly specified

#### Scenario: Set receiver gains
- WHEN the application configures LNA gain to 24 dB and VGA gain to 20 dB
- THEN the driver MUST send vendor request `SET_LNA_GAIN` (code 19) with value 24 (valid range 0-40, 8 dB steps)
- AND MUST send vendor request `SET_VGA_GAIN` (code 20) with value 20 (valid range 0-62, 2 dB steps)

#### Scenario: Enable RF amplifier
- WHEN the application enables the RF amplifier
- THEN the driver MUST send vendor request `AMP_ENABLE` (code 17) with wValue = 1
- AND MUST document that the RF amplifier adds approximately 11 dB of gain

#### Scenario: Reject invalid gain values
- WHEN the application requests an LNA gain of 45 dB (exceeds maximum 40 dB)
- THEN the driver MUST return an error indicating the gain value is out of range
- AND MUST NOT send the vendor request to the device

### Requirement: HackRF IQ Streaming
The HackRF driver SHALL support continuous IQ sample streaming via USB bulk transfers on endpoint 0x81 (RX) and endpoint 0x02 (TX).

#### Scenario: Start RX streaming
- WHEN the application requests to start receiving IQ samples
- THEN the driver MUST send vendor request `SET_TRANSCEIVER_MODE` (code 1) with wValue = 1 (RECEIVE)
- AND MUST submit 4 concurrent bulk IN transfers of 262,144 bytes each on endpoint 0x81
- AND MUST deliver received IQ data to the registered callback as it completes

#### Scenario: Process received IQ data
- WHEN a bulk IN transfer completes with IQ data
- THEN the driver MUST deliver the data buffer to the registered callback
- AND MUST immediately resubmit the transfer buffer for continuous streaming
- AND the data format MUST be 8-bit signed integers, interleaved I,Q,I,Q (2 bytes per complex sample)

#### Scenario: Stop RX streaming
- WHEN the application requests to stop receiving
- THEN the driver MUST send vendor request `SET_TRANSCEIVER_MODE` (code 1) with wValue = 0 (OFF)
- AND MUST cancel all outstanding bulk transfers
- AND MUST ensure no further callbacks are delivered after stop returns

#### Scenario: Start TX streaming
- WHEN the application requests to start transmitting IQ samples
- THEN the driver MUST send vendor request `SET_TRANSCEIVER_MODE` (code 1) with wValue = 2 (TRANSMIT)
- AND MUST accept IQ data buffers for bulk OUT transfer on endpoint 0x02

#### Scenario: Handle half-duplex constraint
- WHEN the application attempts to start TX while RX is active (or vice versa)
- THEN the driver MUST return an error indicating the HackRF One is half-duplex
- AND MUST NOT change the current transceiver mode

### Requirement: HackRF Sweep Mode
The HackRF driver SHALL support frequency sweep mode for wideband spectrum scanning.

#### Scenario: Initialize sweep mode
- WHEN the application configures a sweep from 100 MHz to 3 GHz with 20 MHz step size
- THEN the driver MUST send vendor request `INIT_SWEEP` (code 26) with the frequency ranges encoded in the data payload
- AND MUST set transceiver mode to SWEEP (5)

#### Scenario: Receive sweep data
- WHEN sweep mode is active and bulk IN data arrives
- THEN each buffer MUST begin with a 10-byte header containing the frequency (in Hz, 64-bit) and sample count (16-bit)
- AND the driver MUST deliver the header and IQ data to the registered callback

### Requirement: HackRF Device Reset
The HackRF driver SHALL support device reset for error recovery.

#### Scenario: Reset device after error
- WHEN the driver encounters an unrecoverable USB transfer error
- THEN it MUST send vendor request `RESET` (code 30)
- AND MUST re-initialize the device configuration after the USB bus reset completes
