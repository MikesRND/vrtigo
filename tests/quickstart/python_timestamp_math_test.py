# [TITLE]
# Python Timestamp Math
# [/TITLE]
#
# Quickstart for picosecond-precision timestamp arithmetic from Python:
# Duration, SamplePeriod, Timestamp, and SampleClock.

import vrtigo

# [TEXT]
# All examples use `import vrtigo`.
# [/TEXT]


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------

# [EXAMPLE]
# Duration &mdash; Creating Time Intervals
# [/EXAMPLE]

# [DESCRIPTION]
# `Duration` represents a signed time interval with exact picosecond precision
# and a &plusmn;68 year range. Create from any common unit.
# [/DESCRIPTION]

def test_duration_creation():
    # [SNIPPET]
    d1 = vrtigo.Duration.from_seconds(2)
    d2 = vrtigo.Duration.from_milliseconds(500)
    d3 = vrtigo.Duration.from_microseconds(100)
    d4 = vrtigo.Duration.from_nanoseconds(250)
    d5 = vrtigo.Duration.from_picoseconds(1000)

    # From floating-point seconds (raises ValueError on NaN/inf)
    d6 = vrtigo.Duration.from_seconds_float(1.5)

    # Inspect components
    d6.seconds       # 1 (integer seconds)
    d6.picoseconds   # 500000000000 (subsecond picos)
    d6.to_seconds    # 1.5 (full-precision float)
    # [/SNIPPET]

    assert d1.seconds == 2
    assert d2.picoseconds == 500_000_000_000
    assert d3.picoseconds == 100_000_000
    assert d4.picoseconds == 250_000
    assert d5.picoseconds == 1000
    assert d6.seconds == 1
    assert d6.picoseconds == 500_000_000_000


# [EXAMPLE]
# Duration Arithmetic
# [/EXAMPLE]

# [DESCRIPTION]
# Durations support addition, subtraction, scalar multiplication/division,
# negation, and absolute value. All arithmetic saturates on overflow.
# [/DESCRIPTION]

def test_duration_arithmetic():
    # [SNIPPET]
    a = vrtigo.Duration.from_seconds(10)
    b = vrtigo.Duration.from_milliseconds(250)

    total  = a + b          # 10.25 seconds
    diff   = a - b          # 9.75 seconds
    scaled = b * 4          # 1 second
    halved = a // 2         # 5 seconds
    neg    = -a             # -10 seconds
    mag    = abs(neg)       # 10 seconds
    # [/SNIPPET]

    assert total.to_seconds == 10.25
    assert diff.to_seconds == 9.75
    assert scaled.seconds == 1 and scaled.picoseconds == 0
    assert halved.seconds == 5
    assert neg.is_negative
    assert mag == a


# [EXAMPLE]
# Duration Comparison and Predicates
# [/EXAMPLE]

# [DESCRIPTION]
# Durations are fully comparable and expose `is_zero`, `is_negative`,
# and `is_positive` predicates.
# [/DESCRIPTION]

def test_duration_comparison():
    # [SNIPPET]
    pos = vrtigo.Duration.from_seconds(5)
    neg = vrtigo.Duration.from_seconds(-3)
    z   = vrtigo.Duration.zero()

    assert pos > neg
    assert neg < z
    assert z.is_zero
    assert neg.is_negative
    assert pos.is_positive
    # [/SNIPPET]


# ---------------------------------------------------------------------------
# SamplePeriod
# ---------------------------------------------------------------------------

# [EXAMPLE]
# SamplePeriod &mdash; Sample Rate Specification
# [/EXAMPLE]

# [DESCRIPTION]
# `SamplePeriod` defines the time between consecutive samples and tracks
# whether the period is exactly representable in integer picoseconds.
# [/DESCRIPTION]

def test_sample_period_creation():
    # [SNIPPET]
    # From sample rate
    sp = vrtigo.SamplePeriod.from_rate_hz(10e6)    # 10 MHz
    sp.picoseconds  # 100000 (100 ps per sample)
    sp.rate_hz      # 10000000.0
    sp.is_exact     # True -- integer MHz rates are exact

    # From period in seconds
    sp2 = vrtigo.SamplePeriod.from_seconds(1e-6)   # 1 us period = 1 MHz

    # From rational ratio (rate_hz = numerator / denominator)
    sp3 = vrtigo.SamplePeriod.from_ratio(10_000_000, 1)  # 10 MHz, exact

    # Convert to Duration
    d = sp2.to_duration()
    d.to_seconds  # 1e-6
    # [/SNIPPET]

    assert sp.picoseconds == 100_000
    assert sp.is_exact
    assert sp2.picoseconds == 1_000_000
    assert sp3.picoseconds == 100_000
    assert abs(d.to_seconds - 1e-6) < 1e-18


# [EXAMPLE]
# SamplePeriod Exactness
# [/EXAMPLE]

# [DESCRIPTION]
# Not all sample rates map exactly to an integer number of picoseconds.
# `is_exact` and `error_ppm` let you detect and quantify rounding.
# [/DESCRIPTION]

def test_sample_period_exactness():
    # [SNIPPET]
    exact = vrtigo.SamplePeriod.from_rate_hz(10e6)
    exact.is_exact            # True
    exact.error_ppm           # 0.0

    approx = vrtigo.SamplePeriod.from_rate_hz(3.0)
    approx.is_exact           # may be False
    approx.error_picoseconds  # small nonzero value
    # [/SNIPPET]

    assert exact.is_exact
    assert exact.error_ppm == 0.0


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------

# [EXAMPLE]
# Timestamp &mdash; Construction
# [/EXAMPLE]

# [DESCRIPTION]
# A `Timestamp` wraps a VITA 49 timestamp: integer seconds (TSI) and
# fractional picoseconds (TSF), plus type metadata (`TsiType`, `TsfType`).
# [/DESCRIPTION]

def test_timestamp_construction():
    # [SNIPPET]
    ts = vrtigo.Timestamp(
        1700000000,                   # tsi: seconds since epoch
        500_000_000_000,              # tsf: 0.5 seconds in picoseconds
        vrtigo.TsiType.utc,           # integer timestamp type
        vrtigo.TsfType.real_time      # fractional timestamp type
    )

    ts.tsi       # 1700000000
    ts.tsf       # 500000000000
    ts.tsi_kind  # TsiType.utc
    ts.tsf_kind  # TsfType.real_time
    # [/SNIPPET]

    assert ts.tsi == 1700000000
    assert ts.tsf == 500_000_000_000
    assert ts.tsi_kind == vrtigo.TsiType.utc
    assert ts.tsf_kind == vrtigo.TsfType.real_time


# [EXAMPLE]
# UTC Helpers
# [/EXAMPLE]

# [DESCRIPTION]
# `Timestamp.now()` captures the current UTC wall-clock time.
# `from_utc_seconds()` creates a UTC timestamp from integer seconds.
# `to_datetime()` converts to a Python `datetime` (UTC timestamps only).
# [/DESCRIPTION]

def test_timestamp_utc_helpers():
    # [SNIPPET]
    import datetime

    # Current wall-clock time
    now = vrtigo.Timestamp.now()

    # From integer UTC seconds
    ts = vrtigo.Timestamp.from_utc_seconds(1000000000)

    # Convert to Python datetime (loses sub-microsecond precision)
    dt = ts.to_datetime()
    # datetime.datetime(2001, 9, 9, 1, 46, 40, tzinfo=datetime.timezone.utc)
    # [/SNIPPET]

    assert now.tsi_kind == vrtigo.TsiType.utc
    assert now.tsi > 1700000000
    assert ts.tsi == 1000000000
    assert ts.tsf == 0
    assert dt.year == 2001
    assert dt.month == 9
    assert dt.day == 9


# [EXAMPLE]
# Timestamp + Duration Arithmetic
# [/EXAMPLE]

# [DESCRIPTION]
# Add or subtract a `Duration` to shift a timestamp forward or backward.
# Requires `tsf_kind == real_time`; raises `TypeError` otherwise.
# [/DESCRIPTION]

def test_timestamp_duration_arithmetic():
    # [SNIPPET]
    ts = vrtigo.Timestamp.from_utc_seconds(1000)
    d  = vrtigo.Duration.from_milliseconds(1500)

    later = ts + d
    later.tsi  # 1001
    later.tsf  # 500000000000 (0.5 s in picos)

    earlier = later - d
    earlier.tsi  # 1000
    earlier.tsf  # 0
    # [/SNIPPET]

    assert later.tsi == 1001
    assert later.tsf == 500_000_000_000
    assert earlier.tsi == 1000
    assert earlier.tsf == 0


# [EXAMPLE]
# Timestamp Difference
# [/EXAMPLE]

# [DESCRIPTION]
# Subtracting two timestamps yields a `Duration`. Both must have
# `tsf_kind == real_time` and matching `tsi_kind`.
# [/DESCRIPTION]

def test_timestamp_difference():
    # [SNIPPET]
    ts1 = vrtigo.Timestamp(1000, 0,
                           vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
    ts2 = vrtigo.Timestamp(1005, 250_000_000_000,
                           vrtigo.TsiType.utc, vrtigo.TsfType.real_time)

    diff = ts2 - ts1
    diff.to_seconds   # 5.25
    diff.seconds      # 5
    diff.picoseconds  # 250000000000
    # [/SNIPPET]

    assert abs(diff.to_seconds - 5.25) < 1e-12
    assert diff.seconds == 5
    assert diff.picoseconds == 250_000_000_000


# ---------------------------------------------------------------------------
# SampleClock
# ---------------------------------------------------------------------------

# [EXAMPLE]
# SampleClock &mdash; Synthetic Timestamp Generator
# [/EXAMPLE]

# [DESCRIPTION]
# `SampleClock` produces deterministic timestamps at a fixed sample rate.
# Use it to generate VRT packet timestamps from sample counts.
# [/DESCRIPTION]

def test_sample_clock_basic():
    # [SNIPPET]
    # 1 MHz sample rate, start at epoch zero
    clock = vrtigo.SampleClock(1e-6)

    t0 = clock.now()     # 0.0 seconds
    t1 = clock.tick()    # advance 1 sample -> 1 us
    t2 = clock.tick(99)  # advance 99 more -> 100 us total

    diff = t2 - t0
    diff.to_seconds        # 0.0001 (100 us)

    clock.elapsed_samples  # 100
    clock.period.rate_hz   # 1000000.0
    # [/SNIPPET]

    assert t0.tsi == 0 and t0.tsf == 0
    assert clock.elapsed_samples == 100
    assert abs(diff.to_seconds - 0.0001) < 1e-15


# [EXAMPLE]
# SampleClock Start Time Options
# [/EXAMPLE]

# [DESCRIPTION]
# Control where the clock starts: current wall-clock, next second boundary,
# a specific timestamp, or with a delay offset.
# [/DESCRIPTION]

def test_sample_clock_start_time():
    # [SNIPPET]
    # Start at current UTC wall-clock time
    clock1 = vrtigo.SampleClock(1e-6, vrtigo.StartTime.now())

    # Start at next whole-second boundary (PPS alignment)
    clock2 = vrtigo.SampleClock(1e-6, vrtigo.StartTime.at_next_second())

    # Start at a specific timestamp
    ts = vrtigo.Timestamp(1700000000, 0,
                          vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
    clock3 = vrtigo.SampleClock(1e-6, vrtigo.StartTime.absolute(ts))

    # Start 500 ms from now
    clock4 = vrtigo.SampleClock(
        1e-6,
        vrtigo.StartTime.now_plus(vrtigo.Duration.from_milliseconds(500))
    )
    # [/SNIPPET]

    assert clock1.now().tsi > 1700000000
    assert clock3.now().tsi == 1700000000


# [EXAMPLE]
# Advance Without Returning a Timestamp
# [/EXAMPLE]

# [DESCRIPTION]
# `advance()` moves the clock forward without allocating a return value.
# `reset()` re-resolves the start time and clears elapsed samples.
# [/DESCRIPTION]

def test_sample_clock_advance_and_reset():
    # [SNIPPET]
    clock = vrtigo.SampleClock(1e-6)

    clock.advance(1000)          # skip forward 1000 samples
    ts = clock.now()             # read time without advancing
    clock.elapsed_samples        # 1000

    clock.reset()                # clears elapsed, re-resolves start
    clock.elapsed_samples        # 0
    # [/SNIPPET]

    assert clock.elapsed_samples == 0


# [EXAMPLE]
# Putting It Together
# [/EXAMPLE]

# [DESCRIPTION]
# Generate a sequence of timestamps for a 10 MHz signal, starting at a
# known time, and compute elapsed time for each packet of 1024 samples.
# [/DESCRIPTION]

def test_end_to_end():
    # [SNIPPET]
    period = vrtigo.SamplePeriod.from_rate_hz(10e6)
    start  = vrtigo.Timestamp.from_utc_seconds(1700000000)
    clock  = vrtigo.SampleClock(period.seconds, vrtigo.StartTime.absolute(start))

    timestamps = []
    for pkt in range(5):
        ts = clock.tick(1024)
        elapsed = ts - start
        timestamps.append(elapsed.to_seconds)
    # elapsed times: 0.0001024, 0.0002048, 0.0003072, ...
    # [/SNIPPET]

    assert len(timestamps) == 5
    for i, t in enumerate(timestamps):
        expected = (i + 1) * 1024 * 1e-7  # 1024 samples at 100ns each
        assert abs(t - expected) < 1e-15
