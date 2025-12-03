# Sample Clock

*Auto-generated from `tests/quickstart/sample_clock_test.cpp`. All examples are tested.*

---

All examples assume `using namespace vrtigo;` and `using namespace std::chrono_literals;`.

## Basic Timestamp Generation

SampleClock generates deterministic timestamps at a fixed sample rate.
Create a clock with a sample period, then use `tick()` to advance time.

```cpp
    // Create clock at 1 MHz sample rate (1 microsecond period)
    SampleClock<> clock(1e-6);

    // Query current time without advancing
    auto t0 = clock.now(); // 0.000000 seconds

    // Advance by one sample
    auto t1 = clock.tick(); // 0.000001 seconds

    // Advance by multiple samples
    auto t2 = clock.tick(99); // 0.000100 seconds (100 samples total)
```

## Wall-Clock Start Time

Use `StartTime::now()` to start the clock from the current wall-clock time.
This is useful when timestamps need to reflect actual UTC time.

```cpp
    // Start clock at current UTC wall-clock time
    SampleClock<TsiType::utc> clock(1e-6, StartTime::now());

    // First timestamp reflects actual current time
    auto ts = clock.now();
    // ts.tsi() contains UTC seconds since epoch
```

## PPS Alignment

Use `StartTime::at_next_second()` to align the clock to the next whole-second
boundary. This is essential for PPS (pulse-per-second) synchronization where
timestamps must start exactly on second edges.

```cpp
    // Align clock to next second boundary (for PPS sync)
    SampleClock<TsiType::utc> clock(1e-6, StartTime::at_next_second());

    // First timestamp is exactly on a second boundary
    auto ts = clock.now();
    // ts.tsf() == 0 (no fractional seconds)
```

## Delayed Start with Offset

Use `StartTime::now_plus()` for a delayed start, or `StartTime::at_next_second_plus()`
to start at a fixed offset after a second boundary. The latter is useful for
systems that need processing time after PPS edges.

```cpp
    // Start 500ms from now (setup/coordination time)
    SampleClock<TsiType::utc> clock1(1e-6, StartTime::now_plus(500ms));

    // Start 100ms after next second boundary (PPS + processing offset)
    SampleClock<TsiType::utc> clock2(1e-6, StartTime::at_next_second_plus(100ms));

    auto ts = clock2.now();
    // ts.tsf() == 100ms in picoseconds
```

