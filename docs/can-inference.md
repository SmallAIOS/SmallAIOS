# CAN Bus Inference

SmallAIOS can serve ONNX inference directly over a CAN bus, enabling
real-time AI for automotive ADAS, industrial robotics, drones, and
avionics.

## Architecture

```
 +------------+     +----------------------+     +----------------+
 | CAN frames | --> | CanInferenceAdapter  | --> | DataflowRunner |
 | (sensors)  |     |  - routing table     |     |                |
 +------------+     |  - batch buffers     |     |  Session::run  |
                    |  - decoders          |     +--------+-------+
                    +----------+-----------+              |
                               ^                          v
                               |               +----------+-----------+
                               |               | CanInferenceAdapter  |
                               +---------------+  - output routing    |
                                               |  - encoders          |
                                               +----------+-----------+
                                                          |
                                                          v
                                                   +------+------+
                                                   | CAN frames  |
                                                   | (actuators) |
                                                   +-------------+
```

The adapter sits on top of a `CanController` implementation (loopback,
MCP2515, or AXI CAN). Incoming frames are dispatched by CAN ID into
per-topic batch buffers. When a batch trigger fires (frame count or
time window), the adapter hands the tensor payload to the
`DataflowRunner`, which invokes `Session::run` on the target model.
Output tensors are then routed back through the adapter to emit CAN
frames on the bus.

## Use Cases

### Automotive ADAS
- Read steering angle, speed, yaw rate, accelerometer from CAN
- Run perception/prediction model
- Write steering and brake commands back to CAN

### Industrial Robotics
- Subscribe to encoder/force-torque sensor frames
- Run control policy network
- Publish actuator commands

### Drones
- IMU/GPS/barometer over CAN (ArduPilot DroneCAN)
- Flight controller policy network
- ESC commands

### Avionics (DO-178C / ARINC 825)
- CANaerospace NOD frames from sensors
- Flight envelope predictor
- Control surface commands as CANaerospace

## Configuration

Set environment variables in your container:

| Variable | Description | Example |
|----------|-------------|---------|
| `SMALLAIOS_BUS_BACKEND` | Set to `can` to enable | `can` |
| `SMALLAIOS_CAN_DEVICE` | CAN controller spec | `loopback`, `mcp2515:/dev/spidev0.0`, `axi:0x40000000` |
| `SMALLAIOS_CAN_ROUTING` | Path to routing TOML | `/etc/smallaios/can-routes.toml` |

## Routing Table Format

See `examples/can-routes.toml` for a complete automotive ADAS example.

Topics use the standard SmallAIOS pattern:
`smallaios/inference/<model>/{input,output,error}`

## Hardware Support

| Driver | Status | Use case |
|--------|--------|----------|
| `loopback` | Production | Testing, simulation |
| `mcp2515` | Production | SPI-attached CAN (Raspberry Pi, embedded Linux) |
| `axi_can` | Production | Xilinx FPGA AXI CAN IP |

## CANaerospace (ARINC 825)

For avionics applications, use the `canaerospace_float32` and
`canaerospace_int32` decoders to handle the ARINC 825 typed data
format directly. See the existing `bus::can::canaerospace` module
for the protocol primitives.

## Backpressure

If frames arrive faster than inference can process them, the runner
drops the oldest queued message and increments the
`inference_dropped_messages_total` counter. For hard real-time
deployments, the `OperatorBudget` mechanism enforces per-operator
time limits to prevent inference latency from exceeding the CAN
message period.

## See Also

- [Inference Bus Overview](inference-bus.md)
- [Scheduling Model](scheduling-model.md) — operator time budgets
- [Architecture](architecture.md) — Layer 2 HAL position of bus crate
