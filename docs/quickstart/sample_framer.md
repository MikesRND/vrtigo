# Sample Framer

*Auto-generated from `tests/quickstart/sample_framer_test.cpp`. All examples are tested.*

---

All examples assume `using namespace vrtigo;` and `using namespace vrtigo::utils;`.

## Basic Sample Framing

SampleFramer accumulates payload bytes into fixed-size frames.
Your callback fires when each frame is complete.

```cpp
    // Create storage for a frame of samples (8 samples per frame)
    std::array<int32_t, 8> frame_buf{};

    // Define a simple callback that computes sum of samples in the frame
    int64_t sum = 0;
    auto on_frame = [&](std::span<const int32_t> frame) {
        for (int32_t sample : frame) {
            sum += sample;
        }
        return true;
    };

    // Create the framer
    SimpleSampleFramer<int32_t> framer(frame_buf, 8, on_frame);

    // First packet: 4 samples (not enough for frame)
    framer.ingest_payload(make_int32_payload(1, 4)); // [1,2,3,4]

    // Second packet: 4 samples (completes frame)
    framer.ingest_payload(make_int32_payload(5, 4)); // [5,6,7,8]

    // Callback fired with frame [1,2,3,4,5,6,7,8]
    // sum == 36
```

## Complex Float Samples

SampleFramer supports complex types via the template parameter.

```cpp
    std::array<std::complex<float>, 4> frame_buf{};
    std::vector<std::complex<float>> received;

    SimpleSampleFramer<std::complex<float>> framer(
        frame_buf, 4, [&](std::span<const std::complex<float>> frame) {
            received.insert(received.end(), frame.begin(), frame.end());
            return true;
        });

    auto payload = make_complex_float_payload(1.0f, 4);
    framer.ingest_payload(payload);

    // received[0] == (1.0f, 2.0f)
    // received[1] == (3.0f, 4.0f)
    // ...
```

## Stopping Early

Return `false` from the callback to stop processing. The framer returns
`FrameError::stop_requested` so you know processing was halted.

```cpp
    std::array<int16_t, 4> frame_buf{};
    size_t frames_seen = 0;

    SimpleSampleFramer<int16_t> framer(frame_buf, 4, [&](std::span<const int16_t>) {
        ++frames_seen;
        return frames_seen < 2; // stop after 2 frames
    });

    // Would produce 3 frames, but callback stops after 2
    auto payload = make_int16_payload(0, 12);
    auto result = framer.ingest_payload(payload);

    // result.error() == FrameError::stop_requested
    // frames_seen == 2
```

