"""Tests for VRTFileReader"""

import tempfile
import os
import pytest

from test_packets import (
    DataPacketBuilder, ContextPacketBuilder,
    simple_data_packet, data_packet_with_samples, simple_context_packet,
    write_test_file
)


@pytest.fixture
def temp_vrt_file():
    """Create a temporary VRT file with test packets"""
    packets = [
        simple_data_packet(stream_id=0x1000 + i, payload_bytes=64)
        for i in range(5)
    ]

    with tempfile.NamedTemporaryFile(suffix='.vrt', delete=False) as f:
        for pkt in packets:
            f.write(pkt)
        filepath = f.name

    yield filepath
    os.unlink(filepath)


@pytest.fixture
def mixed_packet_file():
    """Create a file with mixed data and context packets"""
    packets = [
        simple_data_packet(stream_id=0x1000),
        simple_context_packet(stream_id=0x1000),
        simple_data_packet(stream_id=0x1001),
        simple_context_packet(stream_id=0x1001),
        simple_data_packet(stream_id=0x1002),
    ]

    with tempfile.NamedTemporaryFile(suffix='.vrt', delete=False) as f:
        for pkt in packets:
            f.write(pkt)
        filepath = f.name

    yield filepath
    os.unlink(filepath)


class TestVRTFileReaderBasics:
    """Basic VRTFileReader functionality"""

    def test_open_file(self, temp_vrt_file):
        """Can open a VRT file"""
        import vrtigo
        reader = vrtigo.VRTFileReader(temp_vrt_file)
        assert reader.is_open

    def test_file_not_found(self):
        """Raises error for missing file"""
        import vrtigo
        with pytest.raises(Exception):
            vrtigo.VRTFileReader('/nonexistent/path.vrt')

    def test_size_and_tell(self, temp_vrt_file):
        """Can get file size and position"""
        import vrtigo
        reader = vrtigo.VRTFileReader(temp_vrt_file)
        assert reader.size() > 0
        assert reader.tell() == 0


class TestVRTFileReaderIteration:
    """Iteration and packet reading"""

    def test_iterate_packets(self, temp_vrt_file):
        """Can iterate over packets"""
        import vrtigo
        reader = vrtigo.VRTFileReader(temp_vrt_file)

        packets = list(reader)
        assert len(packets) == 5

    def test_packet_types(self, temp_vrt_file):
        """Packets are correct type"""
        import vrtigo
        reader = vrtigo.VRTFileReader(temp_vrt_file)

        for pkt in reader:
            assert isinstance(pkt, vrtigo.DataPacket)

    def test_stream_ids(self, temp_vrt_file):
        """Packets have correct stream IDs"""
        import vrtigo
        reader = vrtigo.VRTFileReader(temp_vrt_file)

        for i, pkt in enumerate(reader):
            assert pkt.stream_id == 0x1000 + i

    def test_mixed_packets(self, mixed_packet_file):
        """Can read mixed data and context packets"""
        import vrtigo
        reader = vrtigo.VRTFileReader(mixed_packet_file)

        packets = list(reader)
        assert len(packets) == 5

        # Check types: data, context, data, context, data
        assert isinstance(packets[0], vrtigo.DataPacket)
        assert isinstance(packets[1], vrtigo.ContextPacket)
        assert isinstance(packets[2], vrtigo.DataPacket)
        assert isinstance(packets[3], vrtigo.ContextPacket)
        assert isinstance(packets[4], vrtigo.DataPacket)

    def test_packets_read_counter(self, temp_vrt_file):
        """Tracks packets read count"""
        import vrtigo
        reader = vrtigo.VRTFileReader(temp_vrt_file)

        assert reader.packets_read == 0
        list(reader)
        assert reader.packets_read == 5


class TestVRTFileReaderRewind:
    """Rewind functionality"""

    def test_rewind(self, temp_vrt_file):
        """Can rewind and re-read"""
        import vrtigo
        reader = vrtigo.VRTFileReader(temp_vrt_file)

        # Read all packets
        first_read = list(reader)

        # Rewind and read again
        reader.rewind()
        second_read = list(reader)

        assert len(first_read) == len(second_read)
        for a, b in zip(first_read, second_read):
            assert a.stream_id == b.stream_id


class TestVRTFileReaderStrictMode:
    """Strict mode and error handling"""

    def test_strict_mode_valid_packets(self, temp_vrt_file):
        """Strict mode works with valid packets"""
        import vrtigo
        reader = vrtigo.VRTFileReader(temp_vrt_file)

        # Should not raise
        pkt = reader.read_next_packet(strict=True)
        assert pkt is not None

    def test_eof_returns_none(self, temp_vrt_file):
        """EOF returns None in non-strict mode"""
        import vrtigo
        reader = vrtigo.VRTFileReader(temp_vrt_file)

        # Read all packets
        for _ in range(5):
            pkt = reader.read_next_packet()
            assert pkt is not None

        # EOF
        pkt = reader.read_next_packet()
        assert pkt is None


class TestVRTFileReaderPayload:
    """Payload access"""

    def test_payload_access(self):
        """Can access packet payload"""
        import vrtigo

        # Create packet with known payload
        samples = list(range(32))
        pkt_bytes = data_packet_with_samples(0x1234, samples, 'int16')

        with tempfile.NamedTemporaryFile(suffix='.vrt', delete=False) as f:
            f.write(pkt_bytes)
            filepath = f.name

        try:
            reader = vrtigo.VRTFileReader(filepath)
            pkt = reader.read_next_packet()

            assert pkt.payload_size_bytes == 64  # 32 samples * 2 bytes
            assert len(pkt.payload) == 64
        finally:
            os.unlink(filepath)
