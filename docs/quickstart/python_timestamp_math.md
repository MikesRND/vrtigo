# Python Timestamp Math

*Auto-generated from `tests/quickstart/python_timestamp_math_test.py`. All examples are tested.*

---

All examples use `import vrtigo`.

## Duration &mdash; Creating Time Intervals

`Duration` represents a signed time interval with exact picosecond precision
and a &plusmn;68 year range. Create from any common unit.

```python
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
```

## Duration Arithmetic

Durations support addition, subtraction, scalar multiplication/division,
negation, and absolute value. All arithmetic saturates on overflow.

```python
    a = vrtigo.Duration.from_seconds(10)
    b = vrtigo.Duration.from_milliseconds(250)

    total  = a + b          # 10.25 seconds
    diff   = a - b          # 9.75 seconds
    scaled = b * 4          # 1 second
    halved = a // 2         # 5 seconds
    neg    = -a             # -10 seconds
    mag    = abs(neg)       # 10 seconds
```

## Duration Comparison and Predicates

Durations are fully comparable and expose `is_zero`, `is_negative`,
and `is_positive` predicates.

```python
    pos = vrtigo.Duration.from_seconds(5)
    neg = vrtigo.Duration.from_seconds(-3)
    z   = vrtigo.Duration.zero()

    assert pos > neg
    assert neg < z
    assert z.is_zero
    assert neg.is_negative
    assert pos.is_positive
```

## SamplePeriod &mdash; Sample Rate Specification

`SamplePeriod` defines the time between consecutive samples and tracks
whether the period is exactly representable in integer picoseconds.

```python
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
```

## SamplePeriod Exactness

Not all sample rates map exactly to an integer number of picoseconds.
`is_exact` and `error_ppm` let you detect and quantify rounding.

```python
    exact = vrtigo.SamplePeriod.from_rate_hz(10e6)
    exact.is_exact            # True
    exact.error_ppm           # 0.0

    approx = vrtigo.SamplePeriod.from_rate_hz(3.0)
    approx.is_exact           # may be False
    approx.error_picoseconds  # small nonzero value
```

## Timestamp &mdash; Construction

A `Timestamp` wraps a VITA 49 timestamp: integer seconds (TSI) and
fractional picoseconds (TSF), plus type metadata (`TsiType`, `TsfType`).

```python
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
```

## UTC Helpers

`Timestamp.now()` captures the current UTC wall-clock time.
`from_utc_seconds()` creates a UTC timestamp from integer seconds.
`to_datetime()` converts to a Python `datetime` (UTC timestamps only).

```python
    import datetime

    # Current wall-clock time
    now = vrtigo.Timestamp.now()

    # From integer UTC seconds
    ts = vrtigo.Timestamp.from_utc_seconds(1000000000)

    # Convert to Python datetime (loses sub-microsecond precision)
    dt = ts.to_datetime()
    # datetime.datetime(2001, 9, 9, 1, 46, 40, tzinfo=datetime.timezone.utc)
```

## Timestamp + Duration Arithmetic

Add or subtract a `Duration` to shift a timestamp forward or backward.
Requires `tsf_kind == real_time`; raises `TypeError` otherwise.

```python
    ts = vrtigo.Timestamp.from_utc_seconds(1000)
    d  = vrtigo.Duration.from_milliseconds(1500)

    later = ts + d
    later.tsi  # 1001
    later.tsf  # 500000000000 (0.5 s in picos)

    earlier = later - d
    earlier.tsi  # 1000
    earlier.tsf  # 0
```

## Timestamp Difference

Subtracting two timestamps yields a `Duration`. Both must have
`tsf_kind == real_time` and matching `tsi_kind`.

```python
    ts1 = vrtigo.Timestamp(1000, 0,
                           vrtigo.TsiType.utc, vrtigo.TsfType.real_time)
    ts2 = vrtigo.Timestamp(1005, 250_000_000_000,
                           vrtigo.TsiType.utc, vrtigo.TsfType.real_time)

    diff = ts2 - ts1
    diff.to_seconds   # 5.25
    diff.seconds      # 5
    diff.picoseconds  # 250000000000
```

## SampleClock &mdash; Synthetic Timestamp Generator

`SampleClock` produces deterministic timestamps at a fixed sample rate.
Use it to generate VRT packet timestamps from sample counts.

```python
    # 1 MHz sample rate, start at epoch zero
    clock = vrtigo.SampleClock(1e-6)

    t0 = clock.now()     # 0.0 seconds
    t1 = clock.tick()    # advance 1 sample -> 1 us
    t2 = clock.tick(99)  # advance 99 more -> 100 us total

    diff = t2 - t0
    diff.to_seconds        # 0.0001 (100 us)

    clock.elapsed_samples  # 100
    clock.period.rate_hz   # 1000000.0
```

## SampleClock Start Time Options

Control where the clock starts: current wall-clock, next second boundary,
a specific timestamp, or with a delay offset.

```python
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
```

## Advance Without Returning a Timestamp

`advance()` moves the clock forward without allocating a return value.
`reset()` re-resolves the start time and clears elapsed samples.

```python
    clock = vrtigo.SampleClock(1e-6)

    clock.advance(1000)          # skip forward 1000 samples
    ts = clock.now()             # read time without advancing
    clock.elapsed_samples        # 1000

    clock.reset()                # clears elapsed, re-resolves start
    clock.elapsed_samples        # 0
```

## Putting It Together

Generate a sequence of timestamps for a 10 MHz signal, starting at a
known time, and compute elapsed time for each packet of 1024 samples.

```python
    period = vrtigo.SamplePeriod.from_rate_hz(10e6)
    start  = vrtigo.Timestamp.from_utc_seconds(1700000000)
    clock  = vrtigo.SampleClock(period.seconds, vrtigo.StartTime.absolute(start))

    timestamps = []
    for pkt in range(5):
        ts = clock.tick(1024)
        elapsed = ts - start
        timestamps.append(elapsed.to_seconds)
    # elapsed times: 0.0001024, 0.0002048, 0.0003072, ...
```

