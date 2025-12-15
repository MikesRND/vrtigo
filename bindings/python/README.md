# VRTIGO Python Bindings

Python bindings for the VRTIGO VRT (VITA 49.2) packet library.

## Requirements

- Python 3.11+
- NumPy 1.24+
- CMake 3.24+
- C++20 compiler (GCC 13+ or Clang 16+)

## Quick Start

```bash
# Create venv and install dependencies from pyproject.toml
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]

# Build the module
make python

# Run tests
make python-test
```

## Building

From the repository root:

```bash
make python
```

Or manually with CMake:

```bash
cmake -B build -DVRTIGO_BUILD_PYTHON=ON
cmake --build build --target vrtigo_py vrtigo_stub
```

The module is built to `build/bindings/python/vrtigo*.so`.

## Usage

Set `PYTHONPATH` to use the built module:

```bash
export PYTHONPATH=build/bindings/python
python3
```

```python
import vrtigo

# Read packets from a file
reader = vrtigo.VRTFileReader('data.vrt')
for packet in reader:
    print(f"Stream {packet.stream_id:#x}: {packet.payload_size_bytes} bytes")

# Parse a packet from bytes
data = open('packet.bin', 'rb').read()
packet = vrtigo.DataPacket.from_bytes(data)
print(packet.type, packet.timestamp)

# Frame samples with callback
import numpy as np

def on_frame(samples: np.ndarray) -> bool:
    print(f"Got {len(samples)} samples")
    return True  # continue processing

framer = vrtigo.SampleFramer(
    frame_size=1024,
    dtype=np.int16,
    callback=on_frame
)

for packet in vrtigo.VRTFileReader('signal.vrt'):
    if isinstance(packet, vrtigo.DataPacket):
        framer.ingest(packet)
```

## API Overview

### Packet Types

| Class | Description |
|-------|-------------|
| `DataPacket` | Owning data packet (safe to store) |
| `ContextPacket` | Owning context packet (safe to store) |
| `DataPacketView` | Non-owning view (valid only during iteration) |
| `ContextPacketView` | Non-owning view (valid only during iteration) |

### Readers

| Class | Description |
|-------|-------------|
| `VRTFileReader` | Read packets from a file |
| `UDPVRTReader` | Read packets from UDP socket |

### Utilities

| Class | Description |
|-------|-------------|
| `SampleFramer` | Accumulate samples into fixed-size frames |
| `Timestamp` | VRT timestamp (integer + fractional) |
| `ClassId` | VRT class identifier (OUI, ICC, PCC) |

### Enums

- `PacketType` - signal_data, context, etc.
- `TsiType` - Integer timestamp type (none, utc, gps, other)
- `TsfType` - Fractional timestamp type (none, sample_count, real_time, free_running)
- `ValidationError` - Packet validation error codes

## Development

### Running Tests

```bash
make python-test

# Or manually
PYTHONPATH=build/bindings/python pytest bindings/python/tests -v
```

### Type Stubs

Type stubs (`vrtigo.pyi`) are generated automatically during build via nanobind's stubgen. They enable IDE autocomplete and type checking.

```bash
# Stubs are built with the module
make python

# Location
ls build/bindings/python/vrtigo.pyi
```

### Project Structure

```
bindings/python/
├── CMakeLists.txt      # Build configuration
├── pyproject.toml      # Python project metadata
├── .python-version     # Python version (3.11)
├── src/
│   ├── vrtigo_module.cpp
│   ├── core_bindings.hpp
│   ├── packet_view_bindings.hpp
│   ├── owning_packet_bindings.hpp
│   ├── reader_bindings.hpp
│   └── sample_framer_bindings.hpp
└── tests/
    ├── conftest.py
    ├── test_enums.py
    ├── test_packets.py
    ├── test_file_reader.py
    └── test_sample_framer.py
```
