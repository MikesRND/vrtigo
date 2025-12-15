#!/usr/bin/env python3
"""
UDP VRT transmitter for testing PSD viewer.

Generates a complex tone and transmits as VRT packets over UDP.

Usage:
    python udp_transmitter.py --port 5000
    python udp_transmitter.py --port 5000 --freq 10e6 --fs 25e6
"""

import argparse
import socket
import struct
import time

import numpy as np


def build_data_packet(stream_id: int, payload: bytes, packet_count: int = 0) -> bytes:
    """Build a simple VRT data packet with float32 payload."""
    if len(payload) % 4:
        payload += b'\x00' * (4 - len(payload) % 4)
    payload_words = len(payload) // 4

    header_words = 2
    total_words = header_words + payload_words

    header = (1 << 28)  # packet_type = 1 (signal data with stream id)
    header |= (packet_count & 0xF) << 16
    header |= total_words & 0xFFFF

    result = struct.pack('>I', header)
    result += struct.pack('>I', stream_id)
    result += payload
    return result


def main():
    parser = argparse.ArgumentParser(description="UDP VRT transmitter for PSD viewer testing")
    parser.add_argument("--port", type=int, default=5000, help="UDP port (default: 5000)")
    parser.add_argument("--host", default="127.0.0.1", help="Destination host (default: 127.0.0.1)")
    parser.add_argument("--fs", type=float, default=100e6, help="Sample rate (Hz)")
    parser.add_argument("--freq", type=float, default=25e6, help="Tone frequency (Hz)")
    parser.add_argument("--samples-per-packet", type=int, default=4096, help="Samples per packet")
    parser.add_argument("--stream-id", type=int, default=0x1234, help="Stream ID")
    parser.add_argument("--noise", type=float, default=0.0, help="Noise amplitude")

    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print(f"Transmitting to {args.host}:{args.port}")
    print(f"  Fs: {args.fs/1e6:.1f} MHz, Tone: {args.freq/1e6:.1f} MHz")
    print(f"  Samples/packet: {args.samples_per_packet}")
    print("Press Ctrl+C to stop")

    packet_count = 0
    sample_idx = 0
    samples_per_packet = args.samples_per_packet

    # Calculate packet rate to match sample rate
    packet_interval = samples_per_packet / args.fs
    overall_start = time.perf_counter()

    try:
        while True:
            loop_start = time.perf_counter()

            # Generate samples
            t = np.arange(sample_idx, sample_idx + samples_per_packet) / args.fs
            samples = np.exp(2j * np.pi * args.freq * t).astype(np.complex64)

            if args.noise > 0:
                noise = args.noise * (np.random.randn(samples_per_packet) + 1j * np.random.randn(samples_per_packet))
                samples += noise.astype(np.complex64)

            # Convert to big-endian
            flat = samples.view(np.float32)
            payload = flat.astype('>f4').tobytes()

            # Build and send packet
            packet = build_data_packet(args.stream_id, payload, packet_count % 16)
            sock.sendto(packet, (args.host, args.port))

            packet_count += 1
            sample_idx += samples_per_packet

            # Pace to match sample rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = packet_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass
    finally:
        total_time = time.perf_counter() - overall_start
        total_samples = packet_count * samples_per_packet
        actual_rate = total_samples / total_time if total_time > 0 else 0
        print(f"\nSent {packet_count} packets in {total_time:.1f}s")
        print(f"Actual rate: {actual_rate/1e6:.2f} MHz (target: {args.fs/1e6:.1f} MHz)")
        sock.close()


if __name__ == "__main__":
    main()
