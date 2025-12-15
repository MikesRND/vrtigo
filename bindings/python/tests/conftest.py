"""
Shared pytest fixtures for VRTIGO Python bindings tests.
"""

import os
import tempfile
import pytest

from test_packets import (
    DataPacketBuilder,
    ContextPacketBuilder,
    simple_data_packet,
    simple_context_packet,
    TsiType,
    TsfType,
)


@pytest.fixture
def sample_data_packet_bytes():
    """Minimal valid data packet bytes."""
    return simple_data_packet(stream_id=0x12345678, payload_bytes=64)


@pytest.fixture
def sample_context_packet_bytes():
    """Minimal valid context packet bytes."""
    return simple_context_packet(stream_id=0x12345678)


@pytest.fixture
def data_packet_with_timestamp_bytes():
    """Data packet with UTC/real-time timestamp."""
    return (
        DataPacketBuilder()
        .with_stream_id(0xABCDEF00)
        .with_timestamp(TsiType.UTC, TsfType.REAL_TIME, tsi=1700000000, tsf=500_000_000_000)
        .with_payload(b"\x00" * 32)
        .build()
    )


@pytest.fixture
def tmp_vrt_file():
    """Temporary file with 5 data packets."""
    packets = [simple_data_packet(stream_id=0x1000 + i, payload_bytes=64) for i in range(5)]

    with tempfile.NamedTemporaryFile(suffix=".vrt", delete=False) as f:
        for pkt in packets:
            f.write(pkt)
        filepath = f.name

    yield filepath
    os.unlink(filepath)


@pytest.fixture
def tmp_mixed_packet_file():
    """Temporary file with mixed data and context packets."""
    packets = [
        simple_data_packet(stream_id=0x1000),
        simple_context_packet(stream_id=0x1000),
        simple_data_packet(stream_id=0x1001),
        simple_context_packet(stream_id=0x1001),
        simple_data_packet(stream_id=0x1002),
    ]

    with tempfile.NamedTemporaryFile(suffix=".vrt", delete=False) as f:
        for pkt in packets:
            f.write(pkt)
        filepath = f.name

    yield filepath
    os.unlink(filepath)


@pytest.fixture
def tmp_empty_file():
    """Empty temporary file."""
    with tempfile.NamedTemporaryFile(suffix=".vrt", delete=False) as f:
        filepath = f.name

    yield filepath
    os.unlink(filepath)
