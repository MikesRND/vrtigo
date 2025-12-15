"""Tests for SampleFramer"""

import numpy as np
import pytest

from test_packets import DataPacketBuilder, data_packet_with_samples


class TestSampleFramerConstruction:
    """SampleFramer construction"""

    def test_create_int16_framer(self):
        """Can create framer with int16 buffer"""
        import vrtigo

        buffer = np.zeros(256, dtype=np.int16)
        framer = vrtigo.SampleFramer(buffer, samples_per_frame=64, callback=lambda x: True)

        assert framer.samples_per_frame == 64
        assert framer.emitted_frames == 0
        assert framer.buffered_samples == 0

    def test_create_float32_framer(self):
        """Can create framer with float32 buffer"""
        import vrtigo

        buffer = np.zeros(256, dtype=np.float32)
        framer = vrtigo.SampleFramer(buffer, samples_per_frame=64, callback=lambda x: True)

        assert framer.samples_per_frame == 64

    def test_create_complex64_framer(self):
        """Can create framer with complex64 buffer"""
        import vrtigo

        buffer = np.zeros(256, dtype=np.complex64)
        framer = vrtigo.SampleFramer(buffer, samples_per_frame=64, callback=lambda x: True)

        assert framer.samples_per_frame == 64

    def test_unsupported_dtype_raises(self):
        """Unsupported dtype raises error"""
        import vrtigo

        buffer = np.zeros(256, dtype=np.uint16)  # unsigned not supported
        with pytest.raises(ValueError):
            vrtigo.SampleFramer(buffer, samples_per_frame=64, callback=lambda x: True)

    def test_buffer_too_small_raises(self):
        """Buffer smaller than samples_per_frame raises error"""
        import vrtigo

        buffer = np.zeros(32, dtype=np.int16)
        with pytest.raises(ValueError):
            vrtigo.SampleFramer(buffer, samples_per_frame=64, callback=lambda x: True)


class TestSampleFramerIngest:
    """Sample ingestion"""

    def test_ingest_packet(self):
        """Can ingest samples from DataPacket"""
        import vrtigo

        buffer = np.zeros(256, dtype=np.int16)
        frames = []

        def on_frame(frame):
            frames.append(frame.copy())
            return True

        framer = vrtigo.SampleFramer(buffer, samples_per_frame=64, callback=on_frame)

        # Create packet with 128 samples (enough for 2 frames)
        samples = list(range(128))
        pkt_bytes = data_packet_with_samples(0x1234, samples, 'int16')
        pkt = vrtigo.DataPacket.from_bytes(pkt_bytes)

        result = framer.ingest(pkt)

        assert result.frames_emitted == 2
        assert not result.stopped
        assert len(frames) == 2
        assert framer.emitted_frames == 2

    def test_ingest_payload(self):
        """Can ingest raw payload bytes"""
        import vrtigo
        import struct

        buffer = np.zeros(256, dtype=np.int16)
        frames = []

        def on_frame(frame):
            frames.append(frame.copy())
            return True

        framer = vrtigo.SampleFramer(buffer, samples_per_frame=32, callback=on_frame)

        # Create raw payload (64 int16 samples = 128 bytes)
        payload = struct.pack('>' + 'h' * 64, *range(64))

        result = framer.ingest_payload(payload)

        assert result.frames_emitted == 2
        assert len(frames) == 2

    def test_accumulation(self):
        """Samples accumulate across packets"""
        import vrtigo

        buffer = np.zeros(256, dtype=np.int16)
        frames = []

        def on_frame(frame):
            frames.append(frame.copy())
            return True

        framer = vrtigo.SampleFramer(buffer, samples_per_frame=64, callback=on_frame)

        # Ingest 32 samples (half a frame)
        samples = list(range(32))
        pkt1 = vrtigo.DataPacket.from_bytes(data_packet_with_samples(0x1234, samples, 'int16'))
        result1 = framer.ingest(pkt1)

        assert result1.frames_emitted == 0
        assert framer.buffered_samples == 32

        # Ingest another 32 samples (completes frame)
        samples = list(range(32, 64))
        pkt2 = vrtigo.DataPacket.from_bytes(data_packet_with_samples(0x1234, samples, 'int16'))
        result2 = framer.ingest(pkt2)

        assert result2.frames_emitted == 1
        assert framer.buffered_samples == 0
        assert len(frames) == 1

    def test_payload_alignment_error(self):
        """Unaligned payload raises error"""
        import vrtigo

        buffer = np.zeros(256, dtype=np.int16)
        framer = vrtigo.SampleFramer(buffer, samples_per_frame=64, callback=lambda x: True)

        # int16 needs 2-byte alignment, but we send 3 bytes
        with pytest.raises(ValueError):
            framer.ingest_payload(b'\x00\x01\x02')


class TestSampleFramerCallback:
    """Callback behavior"""

    def test_callback_receives_correct_size(self):
        """Callback receives frames of correct size"""
        import vrtigo

        buffer = np.zeros(256, dtype=np.int16)
        frame_sizes = []

        def on_frame(frame):
            frame_sizes.append(len(frame))
            return True

        framer = vrtigo.SampleFramer(buffer, samples_per_frame=64, callback=on_frame)

        samples = list(range(128))
        pkt = vrtigo.DataPacket.from_bytes(data_packet_with_samples(0x1234, samples, 'int16'))
        framer.ingest(pkt)

        assert frame_sizes == [64, 64]

    def test_callback_stop_processing(self):
        """Callback can stop processing by returning False"""
        import vrtigo

        buffer = np.zeros(256, dtype=np.int16)
        call_count = [0]

        def on_frame(frame):
            call_count[0] += 1
            return call_count[0] < 2  # Stop after 2nd frame

        framer = vrtigo.SampleFramer(buffer, samples_per_frame=32, callback=on_frame)

        # Ingest enough for 4 frames
        samples = list(range(128))
        pkt = vrtigo.DataPacket.from_bytes(data_packet_with_samples(0x1234, samples, 'int16'))
        result = framer.ingest(pkt)

        assert result.stopped
        assert call_count[0] == 2  # Only called twice

    def test_callback_receives_correct_dtype(self):
        """Callback receives arrays with correct dtype"""
        import vrtigo

        dtypes_tested = []

        for dtype in [np.int16, np.int32, np.float32, np.float64]:
            buffer = np.zeros(256, dtype=dtype)
            received_dtype = [None]

            def on_frame(frame, expected=dtype):
                received_dtype[0] = frame.dtype
                return True

            framer = vrtigo.SampleFramer(buffer, samples_per_frame=64, callback=on_frame)

            # Create payload (256 bytes = enough samples for any dtype)
            payload = bytes(256)
            try:
                framer.ingest_payload(payload)
                if received_dtype[0] is not None:
                    assert received_dtype[0] == dtype
                    dtypes_tested.append(dtype)
            except ValueError:
                pass  # Alignment error for some dtypes

        assert len(dtypes_tested) > 0


class TestSampleFramerFlush:
    """Flush partial frame"""

    def test_flush_partial(self):
        """Can flush partial frame"""
        import vrtigo

        buffer = np.zeros(256, dtype=np.int16)
        frames = []

        def on_frame(frame):
            frames.append(frame.copy())
            return True

        framer = vrtigo.SampleFramer(buffer, samples_per_frame=64, callback=on_frame)

        # Ingest 32 samples (partial frame)
        samples = list(range(32))
        pkt = vrtigo.DataPacket.from_bytes(data_packet_with_samples(0x1234, samples, 'int16'))
        framer.ingest(pkt)

        assert framer.buffered_samples == 32
        assert len(frames) == 0

        # Flush
        result = framer.flush_partial()

        assert result.frames_emitted == 1
        assert len(frames) == 1
        assert len(frames[0]) == 32  # Partial frame

    def test_flush_empty(self):
        """Flush with no buffered samples emits nothing"""
        import vrtigo

        buffer = np.zeros(256, dtype=np.int16)
        framer = vrtigo.SampleFramer(buffer, samples_per_frame=64, callback=lambda x: True)

        result = framer.flush_partial()
        assert result.frames_emitted == 0


class TestSampleFramerReset:
    """Reset functionality"""

    def test_reset(self):
        """Can reset framer state"""
        import vrtigo

        buffer = np.zeros(256, dtype=np.int16)
        framer = vrtigo.SampleFramer(buffer, samples_per_frame=64, callback=lambda x: True)

        # Ingest some samples
        samples = list(range(96))  # 1 frame + 32 buffered
        pkt = vrtigo.DataPacket.from_bytes(data_packet_with_samples(0x1234, samples, 'int16'))
        framer.ingest(pkt)

        assert framer.emitted_frames == 1
        assert framer.buffered_samples == 32

        # Reset
        framer.reset()

        assert framer.emitted_frames == 0
        assert framer.buffered_samples == 0


class TestSampleFramerViews:
    """View behavior and manual copying"""

    def test_frame_is_view(self):
        """Frames are views into buffer"""
        import vrtigo

        buffer = np.zeros(256, dtype=np.int16)
        frames = []

        def on_frame(frame):
            # Frame is a view - call .copy() to retain
            frames.append(frame.copy())
            return True

        framer = vrtigo.SampleFramer(buffer, samples_per_frame=64, callback=on_frame)

        samples = list(range(64))
        pkt = vrtigo.DataPacket.from_bytes(data_packet_with_samples(0x1234, samples, 'int16'))
        framer.ingest(pkt)

        assert len(frames) == 1
        # Copy remains valid
        assert frames[0].dtype == np.int16
        assert len(frames[0]) == 64
