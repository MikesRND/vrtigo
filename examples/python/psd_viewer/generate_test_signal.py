#!/usr/bin/env python3
"""
Generate a test VRT file with a complex tone for PSD viewer testing.

Usage:
    python generate_test_signal.py output.vrt
    python generate_test_signal.py output.vrt --freq 10e6 --duration 1.0
"""

import argparse
import struct
import sys

import numpy as np


def build_data_packet(stream_id: int, payload: bytes, packet_count: int = 0) -> bytes:
    """Build a simple VRT data packet with float32 payload."""
    # Pad payload to word boundary
    if len(payload) % 4:
        payload += b'\x00' * (4 - len(payload) % 4)
    payload_words = len(payload) // 4

    # Header: packet_type=1 (SIGNAL_DATA), stream_id present
    header_words = 2  # header + stream_id
    total_words = header_words + payload_words

    header = (1 << 28)  # packet_type = 1 (signal data with stream id)
    header |= (packet_count & 0xF) << 16
    header |= total_words & 0xFFFF

    result = struct.pack('>I', header)
    result += struct.pack('>I', stream_id)
    result += payload
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate test VRT file with complex tone")
    parser.add_argument("output", help="Output VRT file path")
    parser.add_argument("--fs", type=float, default=100e6, help="Sample rate (Hz)")
    parser.add_argument("--freq", type=float, default=25e6, help="Tone frequency (Hz)")
    parser.add_argument("--duration", type=float, default=0.1, help="Duration (seconds)")
    parser.add_argument("--samples-per-packet", type=int, default=1024, help="Samples per packet")
    parser.add_argument("--stream-id", type=int, default=0x1234, help="Stream ID")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise amplitude (0 = none)")

    args = parser.parse_args()

    fs = args.fs
    freq = args.freq
    duration = args.duration
    samples_per_packet = args.samples_per_packet
    total_samples = int(fs * duration)

    print(f"Generating {duration}s of complex tone at {freq/1e6:.1f} MHz")
    print(f"  Sample rate: {fs/1e6:.1f} MHz")
    print(f"  Total samples: {total_samples}")
    print(f"  Samples per packet: {samples_per_packet}")
    print(f"  Number of packets: {total_samples // samples_per_packet}")

    with open(args.output, 'wb') as f:
        packet_count = 0
        for start in range(0, total_samples, samples_per_packet):
            end = min(start + samples_per_packet, total_samples)
            n_samples = end - start

            # Generate complex tone (unit amplitude = 0 dB)
            t = np.arange(start, end) / fs
            samples = np.exp(2j * np.pi * freq * t).astype(np.complex64)

            # Optional noise
            if args.noise > 0:
                noise = args.noise * (np.random.randn(n_samples) + 1j * np.random.randn(n_samples))
                samples += noise.astype(np.complex64)

            # Convert to big-endian float32 pairs (I, Q, I, Q, ...)
            # complex64 is stored as [real, imag] float32 pairs
            flat = samples.view(np.float32)
            payload = flat.astype('>f4').tobytes()

            packet = build_data_packet(args.stream_id, payload, packet_count % 16)
            f.write(packet)
            packet_count += 1

    print(f"Wrote {args.output} ({packet_count} packets)")


if __name__ == "__main__":
    main()
