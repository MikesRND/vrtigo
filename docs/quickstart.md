# Quickstart Guide

Get started with VRTIGO by exploring these executable examples. Each example is extracted from tested code, so you can be confident it compiles and works correctly.

## Examples

| Topic | Description |
|-------|-------------|
| [Data Packet](quickstart/data_packet.md) | Create a VRT signal data packet with timestamp and payload |
| [Context Packet](quickstart/context_packet.md) | Create a VRT context packet with signal metadata fields |
| [Reading VRT Files](quickstart/file_reader.md) | Read and process VRT packets from files |
| [Parsing Packets](quickstart/packet_parsing.md) | Parse and validate packets from raw buffers |


## Adding New Examples

Examples are auto-generated from test files in `tests/quickstart/`. To add a new example:

1. Create a test file: `tests/quickstart/{name}_test.cpp`
2. Add `[TITLE]`, `[EXAMPLE]`, `[DESCRIPTION]`, and `[SNIPPET]` markers
3. Run `make autodocs` to generate `docs/quickstart/{name}.md`

See [tests/quickstart/README.md](../tests/quickstart/README.md) for the full marker format.

## See Also

- [CIF Field Access](cif_access.md) - Context packet field documentation
- [API Reference](api/) - Complete API documentation
