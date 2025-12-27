# Timestamp Math Operations

Precise timestamp arithmetic for sample-accurate timing in VITA 49 applications.

## Core Types

| Type | Size | Range | Use Case |
|------|------|-------|----------|
| `Duration` | 12 bytes | ±68 years | Long spans, storage |
| `ShortDuration` | 8 bytes | ±106 days | Fast arithmetic, accumulators |
| `SamplePeriod` | 16 bytes | Any rate | Sample timing specification |

## Sample-Based Offsets

Calculate timestamps from sample counts without manual unit conversion:

```cpp
auto period = SamplePeriod::from_rate_hz(10e6);  // 10 MHz sample rate
UtcRealTimestamp start(1700000000, 0);

// Offset by sample count
auto ts_1000 = start.offset_samples(1000, *period);   // +1000 samples
auto ts_back = start.offset_samples(-500, *period);   // -500 samples

// Equivalent manual approach (more verbose)
auto offset = ShortDuration::from_samples(1000, *period);
auto ts_1000_v2 = start + offset;
```

## Duration Arithmetic

```cpp
UtcRealTimestamp ts(1700000000, 0);

// Add/subtract Duration
ts += Duration::from_milliseconds(100);
ts -= Duration::from_microseconds(50);

// Timestamp difference yields Duration
UtcRealTimestamp later(1700000001, 500'000'000'000);
Duration diff = later - ts;  // ~1.5 seconds
```

## ShortDuration for Hot Paths

When accumulating offsets in tight loops, ShortDuration avoids the overhead of split-representation arithmetic:

```cpp
SamplePeriod period = SamplePeriod::from_rate_hz(100e6);  // 100 MHz
ShortDuration accumulator = ShortDuration::zero();

for (int64_t i = 0; i < num_samples; ++i) {
    accumulator += ShortDuration::from_samples(1, period);
    // ... process sample ...
}

// Convert to full Duration when needed
Duration total = accumulator.to_duration();
```

## Conversion Between Types

```cpp
Duration d = Duration::from_seconds(100);

// Duration -> ShortDuration (checked, may fail for large values)
if (auto sd = d.to_short_duration()) {
    // Use ShortDuration for fast arithmetic
}

// ShortDuration -> Duration (always succeeds)
ShortDuration sd = ShortDuration::from_picoseconds(1'000'000'000);
Duration d2 = sd.to_duration();
```

## Overflow Behavior

All arithmetic saturates rather than wrapping:

```cpp
auto ts = UtcRealTimestamp(UINT32_MAX - 1, 0);
ts += Duration::from_seconds(100);  // Saturates to max timestamp

if (saturated(ts)) {
    // Handle overflow condition
}
```

## Precision

- **Resolution**: 1 picosecond (10⁻¹² seconds)
- **Sample rates**: Exact for integer Hz rates up to 1 THz
- **Accumulation**: No drift over billions of samples when using exact periods
