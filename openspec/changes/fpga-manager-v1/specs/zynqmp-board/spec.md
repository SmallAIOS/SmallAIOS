## ADDED Requirements

### Requirement: PMU IPI Driver

The `arch/aarch64-zynqmp` crate SHALL provide a `pmu` module implementing an Inter-Processor Interrupt (IPI) driver for the Zynq UltraScale+ Platform Management Unit (PMU). The driver SHALL implement request/response messaging over the IPI registers using the message format defined by Xilinx XilPM, sufficient to trigger PL configuration via the PCAP / CSU DMA path.

#### Scenario: XilPM-format request/response round-trip

- **WHEN** the `pmu` module sends a request over the IPI registers
- **THEN** the request SHALL be encoded per the Xilinx XilPM message format
- **AND** the driver SHALL wait for the PMU's response on the IPI channel
- **AND** SHALL decode the XilPM response status and return it to the caller

#### Scenario: PMU error response is surfaced, not swallowed

- **WHEN** the PMU answers an IPI request with a non-success XilPM status code
- **THEN** the `pmu` module SHALL return an error carrying the observed status to the caller
- **AND** SHALL NOT silently retry or report success
