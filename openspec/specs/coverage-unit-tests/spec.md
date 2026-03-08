# coverage-unit-tests Specification

## Purpose
TBD - created by archiving change test-coverage-v1. Update Purpose after archive.
## Requirements
### Requirement: UART Driver Unit Test Coverage
All UART driver files (ns16550a.rs, sifive.rs, pl011.rs, axi_uart_lite.rs) SHALL have at least 90% line coverage via mock-register-based unit tests.

#### Scenario: NS16550A initialization test
- GIVEN a mock register region of at least 0x100 bytes
- WHEN a Ns16550a driver is constructed with the mock region base address and initialized with a default UartConfig
- THEN the LCR register MUST be programmed with the correct word length, stop bits, and parity settings
- AND the IER register MUST be programmed to enable or disable interrupts per config
- AND the FCR register MUST enable the FIFO with the configured trigger level

#### Scenario: UART transmit test
- GIVEN an initialized UART driver with a mock register region
- WHEN data bytes are transmitted via the driver's send method
- THEN each byte MUST be written to the transmit holding register (THR)
- AND the driver MUST check the LSR transmit-empty bit before each write
- AND the test MUST verify the correct bytes appear in the THR offset of the mock region

#### Scenario: UART receive test
- GIVEN an initialized UART driver with a mock register region
- WHEN the RBR register contains a byte and the LSR data-ready bit is set
- THEN the driver's receive method MUST return the byte from RBR
- AND the test MUST verify the driver reads from the correct register offset

#### Scenario: UART error conditions
- GIVEN an initialized UART driver with a mock register region
- WHEN the LSR register indicates an overrun error, parity error, or framing error
- THEN the driver MUST return an appropriate error variant
- AND the error MUST be distinguishable by type (overrun vs parity vs framing)

#### Scenario: UART baud rate configuration
- GIVEN a UART driver and a mock register region
- WHEN the driver is configured with different baud rates (9600, 115200, 921600)
- THEN the divisor latch registers (DLL/DLM) MUST be programmed with the correct divisor value
- AND the DLAB bit in LCR MUST be set during divisor programming and cleared after

### Requirement: GPIO Driver Unit Test Coverage
All GPIO driver files (riscv_mmio.rs, arm_pl061.rs, axi_gpio.rs) SHALL have at least 90% line coverage via mock-register-based unit tests.

#### Scenario: GPIO pin direction configuration
- GIVEN a mock register region for a GPIO controller
- WHEN a pin is configured as output or input
- THEN the direction register MUST have the corresponding bit set or cleared
- AND the test MUST verify the correct bit position for pins 0 through the maximum pin count

#### Scenario: GPIO pin read/write
- GIVEN a GPIO controller with a mock register region and a pin configured as output
- WHEN the pin value is set high or low
- THEN the data output register MUST reflect the correct bit value
- AND reading a pin configured as input MUST return the value from the data input register

#### Scenario: GPIO interrupt configuration
- GIVEN a GPIO controller with a mock register region
- WHEN interrupts are enabled for a pin with a specified edge or level trigger
- THEN the interrupt enable register MUST have the corresponding bit set
- AND the interrupt type register (edge/level) MUST be correctly programmed
- AND the interrupt polarity register (rising/falling, high/low) MUST be correctly programmed

#### Scenario: GPIO error on invalid pin
- GIVEN a GPIO controller with a defined number of pins
- WHEN an operation is attempted on a pin number exceeding the maximum
- THEN the driver MUST return an error indicating an invalid pin

### Requirement: SPI Driver Unit Test Coverage
All SPI driver files (riscv_mmio.rs, arm_mmio.rs) SHALL have at least 90% line coverage via mock-register-based unit tests.

#### Scenario: SPI initialization
- GIVEN a mock register region for an SPI controller
- WHEN the controller is initialized with a specific clock divider, polarity, and phase
- THEN the control register MUST reflect the configured CPOL and CPHA bits
- AND the clock divider register MUST be programmed with the correct value

#### Scenario: SPI transfer
- GIVEN an initialized SPI controller with a mock register region
- WHEN a full-duplex transfer is initiated with transmit data
- THEN the transmit data MUST be written to the TX FIFO register
- AND received data MUST be read from the RX FIFO register
- AND the driver MUST wait for the transfer-complete status bit before returning

#### Scenario: SPI chip select control
- GIVEN an initialized SPI controller
- WHEN a chip select line is asserted or deasserted
- THEN the chip select register MUST reflect the correct active/inactive state

#### Scenario: SPI error conditions
- GIVEN an initialized SPI controller with a mock register region
- WHEN the status register indicates a FIFO overflow or underflow
- THEN the driver MUST return an appropriate error

### Requirement: I2C Driver Unit Test Coverage
All I2C driver files (riscv_mmio.rs, arm_mmio.rs, bitbang.rs) SHALL have at least 90% line coverage via mock-register-based unit tests.

#### Scenario: I2C write transaction
- GIVEN an initialized I2C controller with a mock register region
- WHEN a write transaction is initiated to a 7-bit slave address with data bytes
- THEN the address register MUST be programmed with the slave address and write bit
- AND each data byte MUST be written to the data register
- AND the driver MUST check the status register for ACK after each byte

#### Scenario: I2C read transaction
- GIVEN an initialized I2C controller with a mock register region
- WHEN a read transaction is initiated for N bytes from a slave address
- THEN the address register MUST be programmed with the slave address and read bit
- AND the driver MUST read N bytes from the data register
- AND a NACK MUST be sent after the final byte

#### Scenario: I2C NACK handling
- GIVEN an initialized I2C controller with a mock register region
- WHEN the status register indicates a NACK from the slave
- THEN the driver MUST return a NACK error
- AND the bus MUST be released (stop condition generated)

#### Scenario: I2C bitbang fallback
- GIVEN a bitbang I2C implementation using GPIO pins
- WHEN a write transaction is initiated
- THEN the SDA and SCL lines MUST be toggled in the correct I2C protocol sequence
- AND clock stretching MUST be respected by reading SCL before proceeding
- AND the ACK/NACK bit MUST be read from SDA after each byte

### Requirement: Camera Driver Unit Test Coverage
All camera driver files (tegra_vi.rs, broadcom_unicam.rs, fpga_csi.rs) SHALL have at least 90% line coverage via mock-register-based unit tests.

#### Scenario: CSI receiver initialization
- GIVEN a mock register region for a CSI receiver (Tegra VI, Unicam, or FPGA)
- WHEN the receiver is initialized with a lane count and data format
- THEN the lane configuration register MUST be programmed with the correct lane count
- AND the data format register MUST be programmed for the specified pixel format (RAW8, RAW10, YUV422)

#### Scenario: Frame capture start/stop
- GIVEN an initialized CSI receiver with a mock register region
- WHEN frame capture is started
- THEN the control register MUST have the capture-enable bit set
- AND when capture is stopped the capture-enable bit MUST be cleared

#### Scenario: DMA buffer configuration
- GIVEN an initialized CSI receiver with a mock register region
- WHEN a DMA buffer address and size are configured
- THEN the DMA base address register MUST contain the buffer physical address
- AND the DMA size register MUST contain the buffer size
- AND the buffer address MUST be aligned to the required boundary

#### Scenario: Camera error conditions
- GIVEN an initialized CSI receiver with a mock register region
- WHEN the status register indicates a CRC error, FIFO overflow, or frame sync loss
- THEN the driver MUST return an appropriate error variant
- AND the error MUST be distinguishable by type

### Requirement: Mock Register Pattern Consistency
All peripheral driver tests SHALL use a consistent mock register allocation pattern.

#### Scenario: Mock region allocation
- WHEN creating a mock register region for any peripheral driver test
- THEN the region MUST be allocated as a mutable byte array or Vec of sufficient size
- AND the base address MUST be derived from the array's pointer
- AND all register reads/writes MUST go through the same volatile access paths as production code

