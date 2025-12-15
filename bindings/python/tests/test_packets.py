"""
Test packet builder utilities for VRTIGO Python and C++ tests.

Generates valid VRT packets with configurable options. Can output:
- Raw bytes for Python tests
- Binary files for C++ tests
- Hex dumps for documentation

Usage:
    from test_packets import DataPacketBuilder, ContextPacketBuilder

    # Build a simple data packet
    pkt = DataPacketBuilder() \
        .with_stream_id(0x1234) \
        .with_payload(b'\\x00' * 64) \
        .build()

    # Build with int16 samples
    pkt = DataPacketBuilder() \
        .with_stream_id(0x1234) \
        .with_int16_samples([1, 2, 3, 4, 5, 6, 7, 8]) \
        .build()
"""

import struct
from dataclasses import dataclass, field
from typing import Optional, List, Union
from enum import IntEnum


class PacketType(IntEnum):
    """VRT packet types matching VITA 49.2"""
    SIGNAL_DATA_NO_ID = 0
    SIGNAL_DATA = 1
    EXTENSION_DATA_NO_ID = 2
    EXTENSION_DATA = 3
    CONTEXT = 4
    EXTENSION_CONTEXT = 5
    COMMAND = 6
    EXTENSION_COMMAND = 7


class TsiType(IntEnum):
    """Integer timestamp types"""
    NONE = 0
    UTC = 1
    GPS = 2
    OTHER = 3


class TsfType(IntEnum):
    """Fractional timestamp types"""
    NONE = 0
    SAMPLE_COUNT = 1
    REAL_TIME = 2
    FREE_RUNNING = 3


@dataclass
class ClassId:
    """VRT Class Identifier"""
    oui: int = 0
    icc: int = 0
    pcc: int = 0
    pbc: int = 0

    def to_bytes(self) -> bytes:
        """Encode as two 32-bit words (big-endian)"""
        word0 = ((self.pbc & 0x1F) << 27) | (self.oui & 0x00FFFFFF)
        word1 = ((self.icc & 0xFFFF) << 16) | (self.pcc & 0xFFFF)
        return struct.pack('>II', word0, word1)


@dataclass
class Timestamp:
    """VRT Timestamp"""
    tsi_type: TsiType = TsiType.NONE
    tsf_type: TsfType = TsfType.NONE
    tsi: int = 0  # Integer seconds
    tsf: int = 0  # Fractional (picoseconds for REAL_TIME)

    def to_bytes(self) -> bytes:
        """Encode timestamp fields (0, 1, or 2 words)"""
        result = b''
        if self.tsi_type != TsiType.NONE:
            result += struct.pack('>I', self.tsi & 0xFFFFFFFF)
        if self.tsf_type != TsfType.NONE:
            result += struct.pack('>Q', self.tsf & 0xFFFFFFFFFFFFFFFF)
        return result

    @property
    def word_count(self) -> int:
        count = 0
        if self.tsi_type != TsiType.NONE:
            count += 1
        if self.tsf_type != TsfType.NONE:
            count += 2
        return count


class DataPacketBuilder:
    """
    Builder for VRT data packets.

    Example:
        pkt = DataPacketBuilder() \\
            .with_stream_id(0x1234) \\
            .with_int16_samples([1, 2, 3, 4]) \\
            .build()
    """

    def __init__(self, packet_type: PacketType = PacketType.SIGNAL_DATA):
        self._packet_type = packet_type
        self._packet_count = 0
        self._stream_id: Optional[int] = None
        self._class_id: Optional[ClassId] = None
        self._timestamp: Optional[Timestamp] = None
        self._trailer: Optional[int] = None
        self._payload: bytes = b''

        # Auto-set stream_id presence based on packet type
        if packet_type in (PacketType.SIGNAL_DATA, PacketType.EXTENSION_DATA):
            self._stream_id = 0  # Will be set, default to 0

    def with_packet_count(self, count: int) -> 'DataPacketBuilder':
        """Set packet count (0-15)"""
        self._packet_count = count & 0xF
        return self

    def with_stream_id(self, stream_id: int) -> 'DataPacketBuilder':
        """Set stream identifier"""
        self._stream_id = stream_id & 0xFFFFFFFF
        return self

    def without_stream_id(self) -> 'DataPacketBuilder':
        """Remove stream identifier"""
        self._stream_id = None
        return self

    def with_class_id(self, oui: int = 0, icc: int = 0, pcc: int = 0, pbc: int = 0) -> 'DataPacketBuilder':
        """Set class identifier"""
        self._class_id = ClassId(oui, icc, pcc, pbc)
        return self

    def with_timestamp(self, tsi_type: TsiType = TsiType.UTC, tsf_type: TsfType = TsfType.REAL_TIME,
                       tsi: int = 0, tsf: int = 0) -> 'DataPacketBuilder':
        """Set timestamp"""
        self._timestamp = Timestamp(tsi_type, tsf_type, tsi, tsf)
        return self

    def with_trailer(self, trailer: int = 0) -> 'DataPacketBuilder':
        """Add trailer word"""
        self._trailer = trailer & 0xFFFFFFFF
        return self

    def with_payload(self, payload: bytes) -> 'DataPacketBuilder':
        """Set raw payload bytes (will be padded to word boundary)"""
        self._payload = payload
        return self

    def with_int8_samples(self, samples: List[int]) -> 'DataPacketBuilder':
        """Set payload as int8 samples (big-endian)"""
        self._payload = struct.pack(f'>{len(samples)}b', *samples)
        return self

    def with_int16_samples(self, samples: List[int]) -> 'DataPacketBuilder':
        """Set payload as int16 samples (big-endian)"""
        self._payload = struct.pack(f'>{len(samples)}h', *samples)
        return self

    def with_int32_samples(self, samples: List[int]) -> 'DataPacketBuilder':
        """Set payload as int32 samples (big-endian)"""
        self._payload = struct.pack(f'>{len(samples)}i', *samples)
        return self

    def with_float32_samples(self, samples: List[float]) -> 'DataPacketBuilder':
        """Set payload as float32 samples (big-endian)"""
        self._payload = struct.pack(f'>{len(samples)}f', *samples)
        return self

    def with_complex_int16_samples(self, samples: List[complex]) -> 'DataPacketBuilder':
        """Set payload as complex int16 samples (I/Q pairs, big-endian)"""
        flat = []
        for s in samples:
            flat.extend([int(s.real), int(s.imag)])
        self._payload = struct.pack(f'>{len(flat)}h', *flat)
        return self

    def build(self) -> bytes:
        """Build the complete packet"""
        # Calculate sizes
        header_words = 1
        if self._stream_id is not None:
            header_words += 1
        if self._class_id is not None:
            header_words += 2
        if self._timestamp is not None:
            header_words += self._timestamp.word_count

        # Pad payload to word boundary
        payload = self._payload
        if len(payload) % 4:
            payload += b'\x00' * (4 - len(payload) % 4)
        payload_words = len(payload) // 4

        trailer_words = 1 if self._trailer is not None else 0
        total_words = header_words + payload_words + trailer_words

        # Build header word
        header = (self._packet_type & 0xF) << 28
        if self._class_id is not None:
            header |= (1 << 27)  # C bit
        if self._trailer is not None:
            header |= (1 << 26)  # T bit
        # TSI (bits 23-22)
        if self._timestamp is not None:
            header |= (self._timestamp.tsi_type & 0x3) << 22
            header |= (self._timestamp.tsf_type & 0x3) << 20
        header |= (self._packet_count & 0xF) << 16
        header |= total_words & 0xFFFF

        # Assemble packet
        result = struct.pack('>I', header)

        if self._stream_id is not None:
            result += struct.pack('>I', self._stream_id)

        if self._class_id is not None:
            result += self._class_id.to_bytes()

        if self._timestamp is not None:
            result += self._timestamp.to_bytes()

        result += payload

        if self._trailer is not None:
            result += struct.pack('>I', self._trailer)

        return result


class ContextPacketBuilder:
    """
    Builder for VRT context packets.

    Example:
        pkt = ContextPacketBuilder() \\
            .with_stream_id(0x1234) \\
            .with_cif0(0x00000001) \\
            .build()
    """

    def __init__(self):
        self._packet_count = 0
        self._stream_id: int = 0  # Context packets always have stream ID
        self._class_id: Optional[ClassId] = None
        self._timestamp: Optional[Timestamp] = None
        self._change_indicator: bool = False
        self._cif0: int = 0
        self._cif1: Optional[int] = None
        self._cif2: Optional[int] = None
        self._cif3: Optional[int] = None
        self._context_fields: bytes = b''

    def with_packet_count(self, count: int) -> 'ContextPacketBuilder':
        self._packet_count = count & 0xF
        return self

    def with_stream_id(self, stream_id: int) -> 'ContextPacketBuilder':
        self._stream_id = stream_id & 0xFFFFFFFF
        return self

    def with_class_id(self, oui: int = 0, icc: int = 0, pcc: int = 0) -> 'ContextPacketBuilder':
        self._class_id = ClassId(oui, icc, pcc)
        return self

    def with_timestamp(self, tsi_type: TsiType = TsiType.UTC, tsf_type: TsfType = TsfType.REAL_TIME,
                       tsi: int = 0, tsf: int = 0) -> 'ContextPacketBuilder':
        self._timestamp = Timestamp(tsi_type, tsf_type, tsi, tsf)
        return self

    def with_change_indicator(self, changed: bool = True) -> 'ContextPacketBuilder':
        self._change_indicator = changed
        return self

    def with_cif0(self, cif0: int) -> 'ContextPacketBuilder':
        self._cif0 = cif0 & 0xFFFFFFFF
        return self

    def with_cif1(self, cif1: int) -> 'ContextPacketBuilder':
        self._cif1 = cif1 & 0xFFFFFFFF
        self._cif0 |= (1 << 1)  # CIF1 enable bit in CIF0
        return self

    def with_context_fields(self, fields: bytes) -> 'ContextPacketBuilder':
        """Set raw context field bytes"""
        self._context_fields = fields
        return self

    def build(self) -> bytes:
        """Build the complete context packet"""
        # Calculate sizes
        header_words = 1 + 1  # header + stream_id (always present for context)
        if self._class_id is not None:
            header_words += 2
        if self._timestamp is not None:
            header_words += self._timestamp.word_count

        cif_words = 1  # CIF0 always present
        if self._cif1 is not None:
            cif_words += 1
        if self._cif2 is not None:
            cif_words += 1
        if self._cif3 is not None:
            cif_words += 1

        context_words = len(self._context_fields) // 4
        total_words = header_words + cif_words + context_words

        # Build header word
        header = (PacketType.CONTEXT & 0xF) << 28
        if self._class_id is not None:
            header |= (1 << 27)  # C bit
        if self._timestamp is not None:
            header |= (self._timestamp.tsi_type & 0x3) << 22
            header |= (self._timestamp.tsf_type & 0x3) << 20
        header |= (self._packet_count & 0xF) << 16
        header |= total_words & 0xFFFF

        # Assemble packet
        result = struct.pack('>I', header)
        result += struct.pack('>I', self._stream_id)

        if self._class_id is not None:
            result += self._class_id.to_bytes()

        if self._timestamp is not None:
            result += self._timestamp.to_bytes()

        # CIF0 with change indicator
        cif0 = self._cif0
        if self._change_indicator:
            cif0 |= (1 << 31)
        result += struct.pack('>I', cif0)

        if self._cif1 is not None:
            result += struct.pack('>I', self._cif1)
        if self._cif2 is not None:
            result += struct.pack('>I', self._cif2)
        if self._cif3 is not None:
            result += struct.pack('>I', self._cif3)

        result += self._context_fields

        return result


def write_test_file(packets: List[bytes], filepath: str):
    """Write multiple packets to a file for testing"""
    with open(filepath, 'wb') as f:
        for pkt in packets:
            f.write(pkt)


def hex_dump(data: bytes, prefix: str = '') -> str:
    """Format bytes as hex dump for documentation"""
    lines = []
    for i in range(0, len(data), 16):
        chunk = data[i:i+16]
        hex_str = ' '.join(f'{b:02x}' for b in chunk)
        lines.append(f'{prefix}{i:04x}: {hex_str}')
    return '\n'.join(lines)


# Convenience functions for common packet types
def simple_data_packet(stream_id: int = 0x12345678, payload_bytes: int = 64) -> bytes:
    """Create a simple data packet with zero-filled payload"""
    return DataPacketBuilder() \
        .with_stream_id(stream_id) \
        .with_payload(b'\x00' * payload_bytes) \
        .build()


def data_packet_with_samples(stream_id: int, samples: List[int], dtype: str = 'int16') -> bytes:
    """Create a data packet with typed samples"""
    builder = DataPacketBuilder().with_stream_id(stream_id)
    if dtype == 'int8':
        builder.with_int8_samples(samples)
    elif dtype == 'int16':
        builder.with_int16_samples(samples)
    elif dtype == 'int32':
        builder.with_int32_samples(samples)
    else:
        raise ValueError(f'Unknown dtype: {dtype}')
    return builder.build()


def simple_context_packet(stream_id: int = 0x12345678) -> bytes:
    """Create a simple context packet"""
    return ContextPacketBuilder() \
        .with_stream_id(stream_id) \
        .with_cif0(0) \
        .build()
