# Inference Bus — Pub/Sub Dataflow Runner

SmallAIOS exposes ONNX inference through two independent surfaces that can
run simultaneously inside the same container:

1. **HTTP API** — JSON over HTTP/1.1 on `SMALLAIOS_PORT` (default `8080`).
   Human-friendly; easy to debug with `curl`. Request/response semantics.
2. **Pub/Sub dataflow bus** — binary messages on a pub/sub fabric
   (Zenoh-style or DDS). Fire-and-forget; fan-in/fan-out across many
   producers and consumers; suitable for robotics, SDR, and sensor
   pipelines.

The bus is selected by the `SMALLAIOS_BUS_BACKEND` environment variable:

```
SMALLAIOS_BUS_BACKEND=none   # HTTP only (default)
SMALLAIOS_BUS_BACKEND=zenoh  # Zenoh-style key-expression pub/sub
SMALLAIOS_BUS_BACKEND=dds    # DDS via bus::dds::DdsZenohAdapter
```

When the bus is enabled, the container starts a **dataflow runner** in a
background thread that shares the same `ModelManager` as the HTTP server.
Both surfaces see the same loaded models; there is no extra copy.

## Topic Naming Convention

All inference topics live under a common prefix so clients can use
wildcards to subscribe to multiple models at once:

```
smallaios/inference/<model_name>/input    # client → runner: request tensors
smallaios/inference/<model_name>/output   # runner → client: result tensors
smallaios/inference/<model_name>/error    # runner → client: error report
smallaios/inference/_meta/models          # runner → clients: model list (heartbeat)
```

For example, a client that wants to invoke a model named `relu_demo`
publishes to `smallaios/inference/relu_demo/input` and subscribes to
`smallaios/inference/relu_demo/output`.

A monitoring dashboard can subscribe to `smallaios/inference/*/output`
to observe every model's results on one stream.

When the DDS backend is selected, DDS topic names are mapped 1:1 onto
Zenoh key expressions by `bus::dds::DdsZenohAdapter`, so the same
hierarchical layout works across both transports.

## Wire Format

Bus messages reuse the existing IPC binary inference protocol defined in
`ipc/src/inference_proto.rs`. Every message is a self-describing binary
frame — there is no JSON on the bus. The HTTP server still speaks JSON
for debuggability; the bus is tuned for throughput.

Frame layout (little-endian, `#[repr(C)]` compatible):

```
offset  size  field
------  ----  ----------------------------------------
0       4     magic       = 0x4F4E4E58  ("ONNX")
4       2     version     = 0x0001
6       2     request_type (see table below)
8       4     payload_len (bytes that follow)
12      N     payload     (type-specific, see protocol docs)
```

Request types:

| Value | Name              | Direction        |
|------:|-------------------|------------------|
| 1     | `LoadModel`       | client → runner  |
| 2     | `UnloadModel`     | client → runner  |
| 3     | `RunInference`    | client → runner  |
| 4     | `GetModelInfo`    | client → runner  |

For `RunInference`, the payload encodes the input tensor set: one or
more tensors, each prefixed by a small descriptor (dtype tag, rank, the
shape as `u32` dimensions, then the raw bytes).

## Example: Client Pseudocode

Because SmallAIOS does not ship with a Zenoh or DDS client crate itself
(the runner is the only thing that needs the transport), a consumer
application supplies its own client. The following is pseudocode
focusing on the on-wire contract rather than any specific library.

### Publishing a RunInference request

```text
# 1. Build the inference frame.
buf = bytearray()
buf += u32_le(0x4F4E4E58)     # magic "ONNX"
buf += u16_le(0x0001)         # version
buf += u16_le(3)              # request_type = RunInference
payload = encode_tensor_list([
    Tensor(dtype="f32", shape=[1, 4], data=[1.0, -2.0, 3.0, -4.0]),
])
buf += u32_le(len(payload))
buf += payload

# 2. Publish on the model's input topic.
bus.put("smallaios/inference/relu_demo/input", buf)
```

### Subscribing to output

```text
sub = bus.subscribe("smallaios/inference/relu_demo/output")
for msg in sub:
    assert u32_le(msg[0:4]) == 0x4F4E4E58  # "ONNX" magic
    tensors = decode_tensor_list(msg[12:])
    print("relu_demo output:", tensors[0])
```

### Wildcard monitoring

```text
# Zenoh key expression — wildcard segment matches any model name.
sub = bus.subscribe("smallaios/inference/*/output")
for msg in sub:
    handle(msg)

# DDS — same topic hierarchy, bridged through DdsZenohAdapter.
# Subscribers on the DDS side use a matching partition/topic filter.
```

### Handling errors

Errors are published on the `error` sibling topic as a short
tag-length-value payload (UTF-8 reason string + optional numeric code).
Clients should subscribe to both `output` and `error` for any model they
care about.

```text
err_sub = bus.subscribe("smallaios/inference/relu_demo/error")
for msg in err_sub:
    code, reason = decode_error(msg)
    log.warn(f"relu_demo inference failed: {reason} (code={code})")
```

## Backpressure

The runner maintains a bounded queue (default depth 16, configurable
via `DataflowRunnerConfig::max_queue_depth`). If the inference loop
cannot keep up with the input topic rate, the runner drops the oldest
queued message and increments a `dropped_total` counter exported via
the `/metrics` HTTP endpoint. This policy keeps memory bounded and
latency predictable at the cost of occasional dropped inputs — suitable
for live sensor streams where stale data is useless.

If strict delivery is required, use the HTTP API, which applies
TCP-level backpressure at the socket layer.

## When to Use HTTP vs the Bus

| Use case                            | HTTP | Bus |
|-------------------------------------|:----:|:---:|
| Interactive debugging with `curl`   |  ✓   |     |
| Single request → single response    |  ✓   |  ✓  |
| Fan-out: one input → many consumers |      |  ✓  |
| Fan-in: many producers → one model  |      |  ✓  |
| Real-time sensor streams            |      |  ✓  |
| Strict at-least-once delivery       |  ✓   |     |
| Cross-language ROS 2 / DDS clients  |      |  ✓  |
| Web/browser clients                 |  ✓   |     |

Both surfaces share the same loaded models, the same request path inside
the ONNX runtime, and the same Prometheus metric counters — the bus
simply bypasses JSON parsing and HTTP framing for hot-loop use cases.

## See Also

- `ipc/src/inference_proto.rs` — binary wire format definition
- `ipc/src/dataflow_runner.rs` — runner implementation (behind
  `onnx` feature)
- `bus/src/dds/` — DDS-to-Zenoh adapter
- `docs/scheduling-model.md` — where the runner sits in the scheduler
- `openspec/changes/dataflow-inference-v1/` — design notes
