#!/usr/bin/env python3
"""
Live PSD demo - runs transmitter and viewer together.

Usage:
    PYTHONPATH=build/bindings/python python demo.py
    PYTHONPATH=build/bindings/python python demo.py --freq 10e6
"""

import argparse
import os
import signal
import subprocess
import sys
import time

def main():
    parser = argparse.ArgumentParser(description="Live PSD demo")
    parser.add_argument("--port", type=int, default=5000, help="UDP port")
    parser.add_argument("--fs", type=float, default=1e6, help="Sample rate (Hz)")
    parser.add_argument("--freq", type=float, default=250e3, help="Tone frequency (Hz)")
    parser.add_argument("--fft-size", type=int, default=4096, help="FFT size")
    parser.add_argument("--noise", type=float, default=0.01, help="Noise amplitude")
    parser.add_argument("--avg", type=int, default=10, help="FFTs per display update")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable

    # Start transmitter
    tx_cmd = [
        python, os.path.join(script_dir, "udp_transmitter.py"),
        "--port", str(args.port),
        "--fs", str(args.fs),
        "--freq", str(args.freq),
        "--noise", str(args.noise),
    ]

    print(f"Starting transmitter: {args.freq/1e6:.1f} MHz tone @ {args.fs/1e6:.1f} MHz Fs")
    tx_proc = subprocess.Popen(tx_cmd)

    # Give transmitter time to start
    time.sleep(0.5)

    # Start viewer
    viewer_cmd = [
        python, os.path.join(script_dir, "psd_viewer.py"),
        "--udp", str(args.port),
        "--fs", str(args.fs),
        "--fft-size", str(args.fft_size),
        "--avg", str(args.avg),
    ]

    print(f"Starting viewer on UDP port {args.port}")
    print("Close the viewer window to exit")

    viewer_proc = subprocess.Popen(viewer_cmd)

    try:
        viewer_proc.wait()
    except KeyboardInterrupt:
        print("\nStopping...")
        viewer_proc.terminate()
        viewer_proc.wait()
    finally:
        # Send SIGINT so transmitter prints stats before exiting
        tx_proc.send_signal(signal.SIGINT)
        tx_proc.wait()
        print("Demo stopped")


if __name__ == "__main__":
    main()
