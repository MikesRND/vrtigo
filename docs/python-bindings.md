# Python Bindings

VRTIGO provides Python bindings for packet parsing, readers, framing utilities,
and time-related types.

## Packet Types

- `DataPacket` and `ContextPacket` are owning packet objects. They are safe to
  retain after reading.
- `DataPacketView` and `ContextPacketView` are non-owning parsed views created
  with `parse(bytes)`.

## Ownership and Lifetime

- `DataPacket.from_bytes(data)` copies and owns the packet bytes.
- `DataPacketView.parse(data)` and `ContextPacketView.parse(data)` borrow the
  input `bytes`; keep the original `bytes` alive while using the view.

## Header, Payload, and Trailer Access

For data packets (`DataPacket` and `DataPacketView`):

- Header-derived metadata is exposed directly as properties such as `type`,
  `packet_count`, `size_bytes`, `size_words`, `has_stream_id`,
  `has_class_id`, `has_timestamp`, `has_trailer`, `stream_id`, `class_id`,
  and `timestamp`.
- `payload` returns copied `bytes`.
- `payload_view` returns a read-only zero-copy `memoryview` over the payload.
- `trailer_raw` returns the raw 32-bit trailer word, or `None` when the packet
  has no trailer.

## Time Types

The Python bindings also expose VRT time and scheduling helpers:

- `Timestamp` for VRT timestamps
- `Duration` for picosecond-accurate durations
- `SamplePeriod` for exact sample-rate derived intervals
- `SampleClock` for sample-driven time progression
- `StartTime` for scheduling stream start behavior

For details and examples, see [Timestamp Math](timestamp-math.md) and
[Python Timestamp Math](quickstart/python_timestamp_math.md).

## Reader and Framing Utilities

- `VRTFileReader` reads packets from files.
- `UDPVRTReader` reads packets from UDP sockets.
- Both readers return owning `DataPacket` or `ContextPacket` instances.
- `SampleFramer` consumes packet payloads and accumulates them into fixed-size
  sample frames.

For more detail, see [I/O Helpers](utils/io-helpers.md),
[Reading VRT Files](quickstart/file_reader.md), and
[Sample Framer](quickstart/sample_framer.md).

## Build and Test

For repository build commands, tests, and contributor guidance, see
[../DEVELOPMENT.md](../DEVELOPMENT.md).
