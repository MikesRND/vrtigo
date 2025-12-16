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
import threading
import time

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

        # Packet loss tracking / threading state
        self._stats_lock = threading.Lock()
        self._stats = {"received": 0, "lost": 0, "loss_events": 0, "skipped": 0}
        self._expected_packet_count = None
        self.udp_port = None  # Set in run_udp()
        self.udp_reader = None  # For diagnostics only

        # Background worker coordination
        self._pending_psd = None  # Latest (freqs, psd_db) pair
        self._psd_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._udp_thread = None
        self._defer_gui_updates = False

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
            psd_db = self._compute_psd_db()
            self._publish_psd(psd_db)
            self._reset_accumulator()

        return True

    def check_packet_loss(self, packet) -> int:
        """Check for packet loss via packet_count discontinuity.

        Returns number of packets lost (0 if none).
        """
        actual_count = packet.packet_count  # 0-15 (4-bit field)

        if self._expected_packet_count is None:
            self._expected_packet_count = (actual_count + 1) % 16
            self._increment_stats(received=1)
            return 0

        gap = (actual_count - self._expected_packet_count) % 16
        self._increment_stats(received=1, lost=gap, loss_events=1 if gap > 0 else 0)
        self._expected_packet_count = (actual_count + 1) % 16
        return gap

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

    def _compute_psd_db(self):
        """Compute PSD in dB from the accumulated frames."""
        psd_avg = self.psd_accum / max(1, self.frame_count)

        if self.is_complex:
            return 10 * np.log10(np.fft.fftshift(psd_avg) + 1e-12)
        return 10 * np.log10(psd_avg + 1e-12)

    def _reset_accumulator(self):
        self.psd_accum.fill(0)
        self.frame_count = 0

    def _publish_psd(self, psd_db: np.ndarray):
        """Publish PSD result either directly (file mode) or via handoff (UDP mode)."""
        if self._defer_gui_updates:
            with self._psd_lock:
                self._pending_psd = (self.freqs, psd_db)
            return

        self._update_display(self.freqs, psd_db)

    def _snapshot_stats(self):
        with self._stats_lock:
            return dict(self._stats)

    def _reset_stats(self):
        with self._stats_lock:
            for key in self._stats:
                self._stats[key] = 0
        self._expected_packet_count = None

    def _increment_stats(self, received=0, lost=0, loss_events=0, skipped=0):
        with self._stats_lock:
            self._stats["received"] += received
            self._stats["lost"] += lost
            self._stats["loss_events"] += loss_events
            self._stats["skipped"] += skipped

    def _update_display(self, freqs: np.ndarray, psd_db: np.ndarray):
        """Update the plot with averaged PSD."""
        self.curve.setData(freqs, psd_db)

        if self.udp_port is not None:
            stats = self._snapshot_stats()
            total = stats["received"] + stats["lost"]
            loss_pct = 100 * stats["lost"] / max(1, total)
            skipped = stats.get("skipped", 0)
            title = (
                f"VRT PSD - UDP:{self.udp_port} | "
                f"Rx: {stats['received']} | Lost: {stats['lost']} ({loss_pct:.1f}%) | "
                f"Skip: {skipped}"
            )
            self.win.setTitle(title)

        if not self._defer_gui_updates:
            self.app.processEvents()

    def _drain_pending_psd(self):
        """Render the latest PSD published by the worker thread."""
        payload = None
        with self._psd_lock:
            if self._pending_psd is not None:
                payload = self._pending_psd
                self._pending_psd = None

        if payload is None:
            return

        freqs, psd_db = payload
        self._update_display(freqs, psd_db)

    def flush(self):
        """Display any remaining accumulated data."""
        if self.frame_count > 0:
            psd_db = self._compute_psd_db()
            self._reset_accumulator()
            self._update_display(self.freqs, psd_db)

    def run_file(self, filename: str):
        """Process VRT file."""
        self._defer_gui_updates = False
        self._reset_stats()
        self.total_frames = 0
        self._pending_psd = None
        self.udp_port = None
        self._stop_event.clear()

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

    def run_udp(self, port: int, recv_buffer_mb: float = 16.0):
        """Process live UDP stream with a single worker thread."""
        self._defer_gui_updates = True
        self._stop_event.clear()
        self._reset_stats()
        self._pending_psd = None
        self.total_frames = 0
        self.udp_port = port
        start_time = time.time()
        first_packet_time = None
        last_packet_time = None

        try:
            reader = vrtigo.UDPVRTReader(port)
        except Exception as e:
            print(f"Error: failed to bind UDP reader: {e}", file=sys.stderr)
            return

        self.udp_reader = reader  # Diagnostics only
        reader.set_timeout(100)  # 100ms timeout for blocking reads in thread

        # Increase socket receive buffer to handle high packet rates
        buffer_bytes = int(recv_buffer_mb * 1024 * 1024)
        try:
            reader.set_receive_buffer_size(buffer_bytes)
            print(f"Socket receive buffer: {recv_buffer_mb} MB")
        except RuntimeError as e:
            print(f"Warning: Could not set receive buffer size: {e}")

        print(f"Listening on UDP port {port}...")
        self.win.setTitle(f"VRT PSD - UDP:{port}")

        # Handle signals by quitting Qt app
        def quit_handler(*_):
            self.app.quit()
        signal.signal(signal.SIGINT, quit_handler)
        signal.signal(signal.SIGTERM, quit_handler)

        # Framer lives on the worker thread; buffer stays alive via closure
        buffer = np.zeros(self.fft_size * 4, dtype=np.complex64)
        framer = vrtigo.SampleFramer(
            buffer, samples_per_frame=self.fft_size, callback=self.on_frame
        )

        def udp_worker():
            nonlocal first_packet_time, last_packet_time
            while not self._stop_event.is_set():
                try:
                    packet = reader.read_next_packet()

                    if packet is None:
                        self._increment_stats(skipped=1)
                        continue

                    if not isinstance(packet, vrtigo.DataPacket):
                        continue

                    now = time.time()
                    if first_packet_time is None:
                        first_packet_time = now
                    last_packet_time = now

                    self.check_packet_loss(packet)
                    framer.ingest(packet)
                except TimeoutError:
                    continue  # Keep waiting for data
                except Exception as e:
                    if not self._stop_event.is_set():
                        print(f"UDP error: {e}")
                    break

            try:
                framer.flush_partial()
            except Exception:
                pass

        self._udp_thread = threading.Thread(target=udp_worker, daemon=True)
        self._udp_thread.start()

        # GUI timer: pull latest PSD from worker and render
        timer = QtCore.QTimer()
        timer.timeout.connect(self._drain_pending_psd)
        timer.start(33)  # ~30 fps display updates

        self.app.exec()

        # Cleanup
        self._stop_event.set()
        timer.stop()
        if self._udp_thread:
            self._udp_thread.join(timeout=1.0)
            self._udp_thread = None

        end_time = time.time()
        run_start = first_packet_time if first_packet_time is not None else start_time
        run_end = last_packet_time if last_packet_time is not None else end_time
        duration = max(0.0, run_end - run_start)

        stats = self._snapshot_stats()
        rx = stats["received"]
        lost = stats["lost"]
        total = rx + lost
        loss_pct = 100 * lost / max(1, total)

        actual_pkt_rate = rx / duration if duration > 0 else 0.0
        actual_sample_rate_mhz = (rx * 128) / duration / 1e6 if duration > 0 else 0.0
        expected_pkt_rate = self.fs / 128
        expected_sample_rate_mhz = self.fs / 1e6

        print(
            f"Run summary: Rx={rx} Lost={lost} ({loss_pct:.1f}%) | "
            f"actual ~{actual_pkt_rate:.0f} pkt/s ({actual_sample_rate_mhz:.3f} MHz) "
            f"vs expected ~{expected_pkt_rate:.0f} pkt/s ({expected_sample_rate_mhz:.3f} MHz)"
        )

        self.udp_reader = None
        self._defer_gui_updates = False
        del reader, framer


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
    parser.add_argument(
        "--recv-buffer",
        type=float,
        default=16.0,
        metavar="MB",
        help="Socket receive buffer size in MB (default: 16)",
    )

    args = parser.parse_args()

    if not args.file and not args.udp:
        parser.error("Must specify either a file or --udp PORT")

    if args.file and args.udp:
        parser.error("Cannot specify both file and --udp")

    viewer = PSDViewer(fs=args.fs, fft_size=args.fft_size, avg_count=args.avg)

    if args.udp:
        viewer.run_udp(args.udp, recv_buffer_mb=args.recv_buffer)
    else:
        viewer.run_file(args.file)


if __name__ == "__main__":
    main()
