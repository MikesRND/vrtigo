#!/usr/bin/env python3
"""
Real-time PSD viewer for VRT streams.

Usage:
    python psd_viewer.py data.vrt              # File playback
    python psd_viewer.py --udp 4991            # Live UDP
    python psd_viewer.py --fft-size 8192       # Custom FFT size
    python psd_viewer.py --avg 50              # Fewer averages (faster update)
    python psd_viewer.py --fs 50e6             # Custom sample rate
"""

import argparse
import signal
import sys

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

try:
    import vrtigo
except ImportError:
    print("Error: vrtigo module not found.", file=sys.stderr)
    print("Set PYTHONPATH to include the built bindings:", file=sys.stderr)
    print("  PYTHONPATH=build/bindings/python python psd_viewer.py ...", file=sys.stderr)
    sys.exit(1)


class PSDViewer:
    """Real-time PSD display using pyqtgraph."""

    def __init__(self, fs: float = 100e6, fft_size: int = 4096, avg_count: int = 100):
        self.fs = fs
        self.fft_size = fft_size
        self.avg_count = avg_count

        # PSD state (initialized on first frame based on sample type)
        self.is_complex = None
        self.psd_accum = np.zeros(fft_size)
        self.freqs = np.zeros(fft_size)
        self.frame_count = 0

        # Window function (Blackman for good dynamic range)
        self.window = np.blackman(fft_size).astype(np.float32)
        # Window power normalization (so unit amplitude tone = 0 dB)
        self.window_power = np.sum(self.window ** 2) / fft_size

        # pyqtgraph setup
        self.app = pg.mkQApp("VRT PSD Viewer")
        self.win = pg.PlotWidget(title="VRT PSD")
        self.win.setLabel("bottom", "Frequency", "Hz")
        self.win.setLabel("left", "Power", "dB")
        self.win.showGrid(x=True, y=True, alpha=0.3)
        self.win.setYRange(-80, 10)  # Fixed Y axis
        self.win.enableAutoRange(axis='y', enable=False)
        self.curve = self.win.plot(pen=pg.mkPen("y", width=1))
        self.win.resize(1000, 600)
        self.win.show()

        # Stats
        self.total_frames = 0

    def on_frame(self, samples: np.ndarray) -> bool:
        """SampleFramer callback - accumulate PSD."""
        # Handle partial frames
        if len(samples) < self.fft_size:
            return True

        # Detect real vs complex on first frame
        if self.is_complex is None:
            self.is_complex = np.iscomplexobj(samples)
            self._setup_freq_axis()

        # Windowed FFT
        windowed = samples * self.window

        if self.is_complex:
            spectrum = np.fft.fft(windowed) / self.fft_size
            psd = np.abs(spectrum) ** 2 / self.window_power
        else:
            spectrum = np.fft.rfft(windowed) / self.fft_size
            psd = np.abs(spectrum) ** 2 / self.window_power

        self.psd_accum += psd
        self.frame_count += 1
        self.total_frames += 1

        if self.frame_count >= self.avg_count:
            self._update_display()

        return True

    def _setup_freq_axis(self):
        """Setup frequency axis based on sample type."""
        if self.is_complex:
            # Complex: -Fs/2 to +Fs/2
            self.freqs = np.fft.fftshift(np.fft.fftfreq(self.fft_size, 1 / self.fs))
            self.psd_accum = np.zeros(self.fft_size)
        else:
            # Real: 0 to Fs/2
            self.freqs = np.fft.rfftfreq(self.fft_size, 1 / self.fs)
            self.psd_accum = np.zeros(len(self.freqs))

    def _update_display(self):
        """Update the plot with averaged PSD."""
        psd_avg = self.psd_accum / self.frame_count

        if self.is_complex:
            psd_db = 10 * np.log10(np.fft.fftshift(psd_avg) + 1e-12)
        else:
            psd_db = 10 * np.log10(psd_avg + 1e-12)

        self.curve.setData(self.freqs, psd_db)
        self.app.processEvents()

        # Reset accumulator
        self.psd_accum.fill(0)
        self.frame_count = 0

    def flush(self):
        """Display any remaining accumulated data."""
        if self.frame_count > 0:
            self._update_display()

    def run_file(self, filename: str):
        """Process VRT file."""
        reader = vrtigo.VRTFileReader(filename)
        buffer = np.zeros(self.fft_size * 4, dtype=np.complex64)
        framer = vrtigo.SampleFramer(
            buffer, samples_per_frame=self.fft_size, callback=self.on_frame
        )

        print(f"Processing {filename}...")
        for packet in reader:
            if isinstance(packet, vrtigo.DataPacket):
                framer.ingest(packet)

        framer.flush_partial()
        self.flush()
        print(f"Processed {self.total_frames} frames")

        # Keep window open
        self.win.setTitle(f"VRT PSD - {filename} (complete)")
        self.app.exec()

        # Explicit cleanup to avoid nanobind warnings
        del framer, reader

    def run_udp(self, port: int):
        """Process live UDP stream."""
        reader = vrtigo.UDPVRTReader(port)
        reader.set_timeout(10)  # 10ms timeout for non-blocking behavior
        buffer = np.zeros(self.fft_size * 4, dtype=np.complex64)
        framer = vrtigo.SampleFramer(
            buffer, samples_per_frame=self.fft_size, callback=self.on_frame
        )

        print(f"Listening on UDP port {port}...")
        self.win.setTitle(f"VRT PSD - UDP:{port}")

        # Handle signals by quitting Qt app
        def quit_handler(*_):
            self.app.quit()
        signal.signal(signal.SIGINT, quit_handler)
        signal.signal(signal.SIGTERM, quit_handler)

        # Use timer for non-blocking UDP reads
        def poll_udp():
            try:
                packet = reader.read_next_packet()
                if packet is not None and isinstance(packet, vrtigo.DataPacket):
                    framer.ingest(packet)
            except TimeoutError:
                pass  # Expected on timeout
            except Exception as e:
                print(f"UDP error: {e}")

        timer = QtCore.QTimer()
        timer.timeout.connect(poll_udp)
        timer.start(1)  # Poll every 1ms

        self.app.exec()

        # Explicit cleanup to avoid nanobind warnings
        timer.stop()
        del framer, reader


def main():
    parser = argparse.ArgumentParser(
        description="Real-time PSD viewer for VRT streams"
    )
    parser.add_argument("file", nargs="?", help="VRT file to process")
    parser.add_argument("--udp", type=int, metavar="PORT", help="UDP port to listen on")
    parser.add_argument(
        "--fft-size",
        type=int,
        default=4096,
        help="FFT size (default: 4096)",
    )
    parser.add_argument(
        "--avg",
        type=int,
        default=100,
        help="Number of FFTs to average per display update (default: 100)",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=100e6,
        help="Sample rate in Hz (default: 100e6)",
    )

    args = parser.parse_args()

    if not args.file and not args.udp:
        parser.error("Must specify either a file or --udp PORT")

    if args.file and args.udp:
        parser.error("Cannot specify both file and --udp")

    viewer = PSDViewer(fs=args.fs, fft_size=args.fft_size, avg_count=args.avg)

    if args.udp:
        viewer.run_udp(args.udp)
    else:
        viewer.run_file(args.file)


if __name__ == "__main__":
    main()
