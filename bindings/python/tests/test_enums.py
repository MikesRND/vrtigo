"""Tests for VRTIGO Python bindings - Phases 1-4"""

import struct


def test_import():
    """Verify module can be imported."""
    import vrtigo

    assert vrtigo is not None


# =============================================================================
# Phase 1: Core Enums
# =============================================================================


def test_packet_type_values():
    """Verify PacketType enum values."""
    import vrtigo

    assert vrtigo.PacketType.signal_data_no_id.value == 0
    assert vrtigo.PacketType.signal_data.value == 1
    assert vrtigo.PacketType.extension_data_no_id.value == 2
    assert vrtigo.PacketType.extension_data.value == 3
    assert vrtigo.PacketType.context.value == 4
    assert vrtigo.PacketType.extension_context.value == 5
    assert vrtigo.PacketType.command.value == 6
    assert vrtigo.PacketType.extension_command.value == 7


def test_packet_type_str():
    """Verify PacketType string conversion."""
    import vrtigo

    assert str(vrtigo.PacketType.signal_data) == "signal_data"
    assert str(vrtigo.PacketType.context) == "context"
    assert "PacketType.signal_data" in repr(vrtigo.PacketType.signal_data)


def test_tsi_type_values():
    """Verify TsiType enum values."""
    import vrtigo

    assert vrtigo.TsiType.none.value == 0
    assert vrtigo.TsiType.utc.value == 1
    assert vrtigo.TsiType.gps.value == 2
    assert vrtigo.TsiType.other.value == 3


def test_tsi_type_str():
    """Verify TsiType string conversion."""
    import vrtigo

    assert str(vrtigo.TsiType.utc) == "utc"
    assert str(vrtigo.TsiType.gps) == "gps"


def test_tsf_type_values():
    """Verify TsfType enum values."""
    import vrtigo

    assert vrtigo.TsfType.none.value == 0
    assert vrtigo.TsfType.sample_count.value == 1
    assert vrtigo.TsfType.real_time.value == 2
    assert vrtigo.TsfType.free_running.value == 3


def test_tsf_type_str():
    """Verify TsfType string conversion."""
    import vrtigo

    assert str(vrtigo.TsfType.real_time) == "real_time"
    assert str(vrtigo.TsfType.sample_count) == "sample_count"


def test_validation_error_values():
    """Verify ValidationError enum values."""
    import vrtigo

    assert vrtigo.ValidationError.none.value == 0
    assert vrtigo.ValidationError.buffer_too_small.value == 1
    assert vrtigo.ValidationError.packet_type_mismatch.value == 2


def test_validation_error_str():
    """Verify ValidationError provides human-readable messages."""
    import vrtigo

    assert "No error" in str(vrtigo.ValidationError.none)
    assert "Buffer" in str(vrtigo.ValidationError.buffer_too_small)


def test_helper_functions():
    """Verify helper functions work correctly."""
    import vrtigo

    # is_signal_data
    assert vrtigo.is_signal_data(vrtigo.PacketType.signal_data) is True
    assert vrtigo.is_signal_data(vrtigo.PacketType.signal_data_no_id) is True
    assert vrtigo.is_signal_data(vrtigo.PacketType.context) is False

    # has_stream_identifier
    assert vrtigo.has_stream_identifier(vrtigo.PacketType.signal_data) is True
    assert vrtigo.has_stream_identifier(vrtigo.PacketType.signal_data_no_id) is False
    assert vrtigo.has_stream_identifier(vrtigo.PacketType.context) is True


def test_constants():
    """Verify module constants are exposed."""
    import vrtigo

    assert vrtigo.VRT_WORD_SIZE == 4
    assert vrtigo.VRT_WORD_BITS == 32
    assert vrtigo.MAX_PACKET_WORDS == 65535
    assert vrtigo.MAX_PACKET_BYTES == 65535 * 4
    assert vrtigo.PICOSECONDS_PER_SECOND == 1_000_000_000_000


def test_enum_comparison():
    """Verify enum values can be compared."""
    import vrtigo

    assert vrtigo.PacketType.signal_data == vrtigo.PacketType.signal_data
    assert vrtigo.PacketType.signal_data != vrtigo.PacketType.context
    assert vrtigo.TsiType.utc != vrtigo.TsiType.gps


# =============================================================================
# Phase 2: ClassId
# =============================================================================


def test_class_id_construction():
    """Verify ClassId can be constructed."""
    import vrtigo

    cid = vrtigo.ClassId(oui=0x123456, icc=0x0001, pcc=0x0002)
    assert cid.oui == 0x123456
    assert cid.icc == 0x0001
    assert cid.pcc == 0x0002
    assert cid.pbc == 0


def test_class_id_with_pbc():
    """Verify ClassId with pad bit count."""
    import vrtigo

    cid = vrtigo.ClassId(oui=0xABCDEF, icc=0x1234, pcc=0x5678, pbc=5)
    assert cid.oui == 0xABCDEF
    assert cid.icc == 0x1234
    assert cid.pcc == 0x5678
    assert cid.pbc == 5


def test_class_id_from_words():
    """Verify ClassId can be created from wire format."""
    import vrtigo

    cid = vrtigo.ClassId(oui=0x123456, icc=0x0001, pcc=0x0002)
    cid2 = vrtigo.ClassId.from_words(cid.word0, cid.word1)
    assert cid == cid2


def test_class_id_word_encoding():
    """Verify ClassId word encoding matches VITA 49.2 spec."""
    import vrtigo

    cid = vrtigo.ClassId(oui=0x123456, icc=0xABCD, pcc=0x1234, pbc=3)
    # Word 0: [31:27] PBC | [26:24] Reserved | [23:0] OUI
    assert cid.word0 == (3 << 27) | 0x123456
    # Word 1: [31:16] ICC | [15:0] PCC
    assert cid.word1 == (0xABCD << 16) | 0x1234


def test_class_id_repr():
    """Verify ClassId has a useful repr."""
    import vrtigo

    cid = vrtigo.ClassId(oui=0x123456, icc=0x0001, pcc=0x0002)
    r = repr(cid)
    assert "ClassId" in r
    assert "123456" in r


# =============================================================================
# Phase 3: Timestamp (from packets only)
# =============================================================================


def test_timestamp_from_packet():
    """Verify Timestamp can be extracted from parsed packets."""
    import vrtigo

    # Create a packet with UTC/real_time timestamp
    packet_type = 1  # signal_data
    class_id_flag = 0
    trailer = 0
    tsi = 1  # utc
    tsf = 2  # real_time
    packet_count = 0
    packet_size = 6  # header + stream_id + tsi + tsf(2) + payload = 6 words

    header = (
        (packet_type << 28)
        | (class_id_flag << 27)
        | (trailer << 26)
        | (tsi << 22)
        | (tsf << 20)
        | (packet_count << 16)
        | packet_size
    )

    stream_id = 0x12345678
    tsi_val = 1000
    tsf_val = 500_000_000_000  # 0.5 seconds
    payload = 0xDEADBEEF

    packet_bytes = struct.pack(">IIIQI", header, stream_id, tsi_val, tsf_val, payload)

    pkt = vrtigo.DataPacketView.parse(packet_bytes)
    assert pkt.has_timestamp

    ts = pkt.timestamp
    assert ts is not None
    assert ts.tsi == 1000
    assert ts.tsf == 500_000_000_000
    assert ts.tsi_kind == vrtigo.TsiType.utc
    assert ts.tsf_kind == vrtigo.TsfType.real_time
    assert ts.has_tsi is True
    assert ts.has_tsf is True


# =============================================================================
# Phase 4: DataPacketView
# =============================================================================


def test_data_packet_parse_minimal():
    """Parse a minimal signal_data packet."""
    import vrtigo

    # Minimal packet: header + stream_id + 1 word payload = 3 words
    header = (1 << 28) | 3  # type=signal_data, size=3
    stream_id = 0xAABBCCDD
    payload = 0x11223344

    packet_bytes = struct.pack(">III", header, stream_id, payload)

    pkt = vrtigo.DataPacketView.parse(packet_bytes)
    assert pkt.type == vrtigo.PacketType.signal_data
    assert pkt.size_bytes == 12
    assert pkt.size_words == 3
    assert pkt.has_stream_id is True
    assert pkt.stream_id == 0xAABBCCDD
    assert pkt.has_class_id is False
    assert pkt.class_id is None
    assert pkt.has_timestamp is False
    assert pkt.timestamp is None
    assert pkt.payload_size_bytes == 4
    assert pkt.payload == b"\x11\x22\x33\x44"


def test_data_packet_with_class_id():
    """Parse a packet with class ID."""
    import vrtigo

    # header + stream_id + class_id(2) + payload = 5 words
    header = (1 << 28) | (1 << 27) | 5  # type=signal_data, C=1, size=5
    stream_id = 0x12345678
    class_word0 = 0x00ABCDEF  # OUI
    class_word1 = (0x1234 << 16) | 0x5678  # ICC | PCC
    payload = 0xDEADBEEF

    packet_bytes = struct.pack(
        ">IIIII", header, stream_id, class_word0, class_word1, payload
    )

    pkt = vrtigo.DataPacketView.parse(packet_bytes)
    assert pkt.has_class_id is True
    cid = pkt.class_id
    assert cid is not None
    assert cid.oui == 0xABCDEF
    assert cid.icc == 0x1234
    assert cid.pcc == 0x5678


def test_data_packet_parse_error():
    """Verify parse errors are raised as exceptions."""
    import vrtigo

    # Packet claiming size=10 words but only providing 1 word
    header = (1 << 28) | 10  # type=signal_data, size=10
    packet_bytes = struct.pack(">I", header)

    try:
        vrtigo.DataPacketView.parse(packet_bytes)
        assert False, "Should have raised an exception"
    except Exception as e:
        assert "Buffer" in str(e) or "buffer" in str(e)


def test_data_packet_repr():
    """Verify DataPacketView has useful repr."""
    import vrtigo

    header = (1 << 28) | 3
    stream_id = 0x12345678
    payload = 0xDEADBEEF

    packet_bytes = struct.pack(">III", header, stream_id, payload)
    pkt = vrtigo.DataPacketView.parse(packet_bytes)

    r = repr(pkt)
    assert "DataPacketView" in r
    assert "signal_data" in r
    assert "12345678" in r


def test_data_packet_payload_access():
    """Verify payload can be accessed as bytes."""
    import vrtigo

    header = (1 << 28) | 4  # 4 words = header + stream_id + 2 payload words
    stream_id = 0x12345678
    payload1 = 0xAABBCCDD
    payload2 = 0x11223344

    packet_bytes = struct.pack(">IIII", header, stream_id, payload1, payload2)
    pkt = vrtigo.DataPacketView.parse(packet_bytes)

    assert pkt.payload_size_bytes == 8
    assert pkt.payload_size_words == 2
    assert pkt.payload == b"\xaa\xbb\xcc\xdd\x11\x22\x33\x44"


if __name__ == "__main__":
    import sys

    # Run all test functions
    failed = 0
    for name, obj in list(globals().items()):
        if name.startswith("test_") and callable(obj):
            try:
                obj()
                print(f"PASS: {name}")
            except AssertionError as e:
                print(f"FAIL: {name}: {e}")
                failed += 1
            except Exception as e:
                print(f"ERROR: {name}: {e}")
                failed += 1

    sys.exit(1 if failed else 0)
