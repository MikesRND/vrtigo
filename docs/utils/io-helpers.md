# I/O Helpers

File, UDP, and PCAP utilities for reading and writing VRT packets.

## Writers

All writers support two primary overloads:
- `write_packet(span<const uint8_t>)` - raw bytes
- `write_packet(PacketVariant)` - dynamic packet variant

For typed packets or views, call `.as_bytes()`:
```cpp
writer.write_packet(typed_packet.as_bytes());
writer.write_packet(data_view.as_bytes());
```

### Writer API

| Method | VRTFileWriter | UDPVRTWriter | PCAPVRTWriter |
|--------|:-------------:|:------------:|:-------------:|
| `write_packet(span<uint8_t>)` | Y | Y | Y |
| `write_packet(span, ts_sec, ts_usec)` | - | - | Y |
| `write_packet(span, sockaddr_in&)` | - | Y | - |
| `write_packet(PacketVariant)` | Y | Y | Y |
| `write_packet(PacketVariant, sockaddr_in&)` | - | Y | - |
| `flush()` | Y | Y | Y |
| `packets_written()` | Y | Y | Y |
| `is_open()` | Y | Y | Y |

## Readers

All readers support:
- `read_next_packet()` - returns `expected<PacketVariant, ReaderError>`
- `read_next_raw()` - returns `span<const uint8_t>` (no parsing)

### Reader API

| Method | VRTFileReader | UDPVRTReader | PCAPVRTReader |
|--------|:-------------:|:------------:|:-------------:|
| `read_next_packet()` | Y | Y | Y |
| `read_next_raw()` | Y | Y | Y |
| `last_error()` | Y | - | - |
| `last_status()` | - | - | Y |
| `transport_status()` | - | Y | - |
| `packets_read()` | Y | Y | Y |
| `is_open()` | Y | Y | Y |
| `rewind()` | Y | - | Y |
| `tell()` / `size()` | Y | - | Y |
| `for_each_*` helpers | Y | Y | Y |

### Error Handling

After `read_next_raw()` returns an empty span, check the error accessor:

| Reader | Error Accessor | EOF Check |
|--------|----------------|-----------|
| VRTFileReader | `last_error()` | `.is_eof()` |
| UDPVRTReader | `transport_status()` | `.state == socket_closed` |
| PCAPVRTReader | `last_status()` | `== PCAPReadStatus::eof` |

## Span Lifetime

- **Writers**: `write_packet(span)` copies data; caller's buffer can be reused immediately
- **Readers**: `read_next_raw()` returns span valid only until next read call

## Namespaces

```cpp
vrtigo::utils::fileio::VRTFileWriter
vrtigo::utils::fileio::VRTFileReader
vrtigo::utils::netio::UDPVRTWriter
vrtigo::utils::netio::UDPVRTReader
vrtigo::utils::pcapio::PCAPVRTWriter
vrtigo::utils::pcapio::PCAPVRTReader
```

Or use the convenience aliases from `vrtigo_io.hpp`:
```cpp
vrtigo::VRTFileReader<>
vrtigo::VRTFileWriter<>
vrtigo::PCAPVRTReader<>
vrtigo::PCAPVRTWriter
```
