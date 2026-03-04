# Quickstart Guide

Get started with VRTIGO by exploring these executable examples. Each example is extracted from tested code, so you can be confident it compiles and works correctly.

## Examples

| Topic | Description |
|-------|-------------|
| [Data Packet](quickstart/data_packet.md) | Create a VRT signal data packet with timestamp and payload |
| [Context Packet](quickstart/context_packet.md) | Create a VRT context packet with signal metadata fields |
| [Reading VRT Files](quickstart/file_reader.md) | Read and process VRT packets from files |
| [Parsing Packets](quickstart/packet_parsing.md) | Parse and validate packets from raw buffers |
| [Sample Clock](quickstart/sample_clock.md) | Model sample time progression from a sample rate |
| [Sample Framer](quickstart/sample_framer.md) | Accumulate payload bytes into fixed-size sample frames |
| [Python Timestamp Math](quickstart/python_timestamp_math.md) | Duration, SamplePeriod, Timestamp arithmetic, and SampleClock from Python |


## Adding New Examples

Examples are generated from tested snippets in `tests/quickstart/`.
See [tests/quickstart/README.md](../tests/quickstart/README.md) for the marker
format and tooling details.

## See Also

- [CIF Field Access](cif_access.md) - Context packet field documentation
- [Python Bindings](python-bindings.md) - Python packet, reader, and time-type overview
