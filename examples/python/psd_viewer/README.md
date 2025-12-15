# PSD Viewer

Real-time Power Spectral Density viewer for VRT streams.

## Installation

```bash
pip install -r requirements.txt
```

Requires the vrtigo Python bindings to be built:
```bash
# From repository root
make python
```

## Usage

### View PSD from a VRT file

```bash
PYTHONPATH=build/bindings/python python psd_viewer.py recording.vrt
```

### Live UDP stream

```bash
PYTHONPATH=build/bindings/python python psd_viewer.py --udp 4991
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--fft-size` | 4096 | FFT size in samples |
| `--avg` | 100 | Number of FFTs to average per display update |
| `--fs` | 100e6 | Sample rate in Hz |
| `--udp PORT` | - | Listen on UDP port instead of reading file |

### Examples

```bash
# 8K FFT with faster updates (fewer averages)
python psd_viewer.py data.vrt --fft-size 8192 --avg 50

# 50 MHz sample rate
python psd_viewer.py data.vrt --fs 50e6

# Live UDP on port 5000
python psd_viewer.py --udp 5000 --fs 25e6
```

## Live Demo

Run transmitter and viewer together (1 MHz sample rate, real-time):

```bash
# From repository root
PYTHONPATH=build/bindings/python python examples/python/psd_viewer/demo.py
PYTHONPATH=build/bindings/python python examples/python/psd_viewer/demo.py --freq 100e3 --noise 0.05
```

## UDP Transmitter

Stream a test tone over UDP:

```bash
python udp_transmitter.py --port 5000 --freq 25e6
python udp_transmitter.py --port 5000 --freq 10e6 --noise 0.01
```

Options: `--port`, `--host`, `--fs`, `--freq`, `--noise`, `--samples-per-packet`

## Test Signal Generator

Generate a test VRT file with a complex tone:

```bash
python generate_test_signal.py test.vrt
python generate_test_signal.py test.vrt --freq 10e6 --duration 0.5 --noise 0.01
```

Options: `--fs`, `--freq`, `--duration`, `--samples-per-packet`, `--stream-id`, `--noise`

## WSL Notes

- Requires WSL2 with WSLg (Windows 11) or an X server
- If display issues occur, try: `export QT_QPA_PLATFORM=xcb`
