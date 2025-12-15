#pragma once

#include "vrtigo/dynamic/data_packet.hpp"
#include "vrtigo/expected.hpp"
#include "vrtigo/sample_span.hpp"

#include <array>
#include <functional>
#include <span>
#include <stdexcept>

#include <cstdint>

namespace vrtigo::utils {

/**
 * Runtime errors from SampleFramer operations.
 *
 * Construction-time errors (format mismatch, buffer too small) throw
 * std::invalid_argument instead.
 */
enum class FrameError : uint8_t {
    payload_not_aligned, ///< Payload size not a multiple of sample size
    stop_requested       ///< Callback returned false to stop processing
};

/**
 * Convert FrameError to human-readable string.
 */
constexpr const char* frame_error_string(FrameError err) noexcept {
    switch (err) {
        case FrameError::payload_not_aligned:
            return "Payload size not aligned to sample size";
        case FrameError::stop_requested:
            return "Stop requested by callback";
        default:
            return "Unknown error";
    }
}

/**
 * Typed, endian-safe framer that accumulates VRT payload samples into
 * fixed-size frames using user-provided buffers.
 *
 * Features:
 * - Zero allocation: operates on user-provided span<SampleT> buffers
 * - Endian-safe: byte-swaps samples to host order per component
 * - Ping-pong support: optional double-buffering for producer/consumer decoupling
 * - Compile-time type safety via SampleT template parameter
 *
 * Callback signature: `bool(std::span<const SampleT>)`
 * Return false from callback to stop processing (surfaces as `stop_requested` error).
 *
 * @tparam SampleT Sample type: int8_t, int16_t, int32_t, float, double,
 *                 std::complex<int8_t>, std::complex<int16_t>, std::complex<int32_t>,
 *                 std::complex<float>, std::complex<double>
 * @tparam Callback Callable type for frame emission
 */
template <ValidSampleType SampleT, typename Callback>
class SampleFramer {
public:
    /**
     * Construct in linear (single buffer) mode.
     *
     * @param frame_buf Buffer for accumulating samples
     * @param samples_per_frame Number of samples per emitted frame
     * @param callback Function called when frame is complete
     * @throws std::invalid_argument if:
     *         - samples_per_frame == 0
     *         - frame_buf.size() < samples_per_frame
     */
    SampleFramer(std::span<SampleT> frame_buf, size_t samples_per_frame, Callback callback)
        : samples_per_frame_(samples_per_frame),
          callback_(std::move(callback)) {
        validate_construction(frame_buf.size(), 0, samples_per_frame);
        buffers_[0] = frame_buf;
        buffer_count_ = 1;
    }

    /**
     * Construct in ping-pong (double buffer) mode.
     *
     * If buf_b is empty, falls back to linear mode.
     *
     * @param buf_a First buffer for accumulating samples
     * @param buf_b Second buffer for ping-pong (empty = linear mode)
     * @param samples_per_frame Number of samples per emitted frame
     * @param callback Function called when frame is complete
     * @throws std::invalid_argument if:
     *         - samples_per_frame == 0
     *         - buf_a.size() < samples_per_frame
     *         - buf_b non-empty and buf_b.size() < samples_per_frame
     */
    SampleFramer(std::span<SampleT> buf_a, std::span<SampleT> buf_b, size_t samples_per_frame,
                 Callback callback)
        : samples_per_frame_(samples_per_frame),
          callback_(std::move(callback)) {
        validate_construction(buf_a.size(), buf_b.size(), samples_per_frame);
        buffers_[0] = buf_a;
        buffers_[1] = buf_b;
        buffer_count_ = buf_b.empty() ? 1 : 2;
    }

    /**
     * Ingest payload from a data packet.
     *
     * Reads samples from the packet's payload, performs endian conversion,
     * and accumulates into the frame buffer. Emits complete frames via callback.
     *
     * May emit multiple frames if the payload is large.
     *
     * @param pkt Data packet to process
     * @return Number of frames emitted, or error
     */
    expected<size_t, FrameError> ingest(const dynamic::DataPacketView& pkt) {
        return ingest_payload(pkt.payload());
    }

    /**
     * Ingest raw payload bytes.
     *
     * @param payload Raw payload bytes (network order)
     * @return Number of frames emitted, or error
     */
    expected<size_t, FrameError> ingest_payload(std::span<const uint8_t> payload) {
        constexpr size_t sample_size = detail::SampleTraits<SampleT>::sample_size;

        // Check alignment
        if (payload.size() % sample_size != 0) {
            return make_unexpected(FrameError::payload_not_aligned);
        }

        SampleSpanView<SampleT> samples(payload);
        size_t frames_emitted = 0;
        size_t sample_index = 0;

        while (sample_index < samples.count()) {
            // How many samples can we fit in current buffer?
            size_t space_available = samples_per_frame_ - fill_offset_;
            size_t samples_available = samples.count() - sample_index;
            size_t samples_to_read = std::min(space_available, samples_available);

            // Read samples into buffer
            auto dest = buffers_[active_index_].subspan(fill_offset_, samples_to_read);
            for (size_t i = 0; i < samples_to_read; ++i) {
                dest[i] = samples[sample_index + i];
            }

            fill_offset_ += samples_to_read;
            sample_index += samples_to_read;

            // Emit frame if complete
            if (fill_offset_ == samples_per_frame_) {
                auto frame_span =
                    std::span<const SampleT>(buffers_[active_index_].data(), samples_per_frame_);
                bool continue_processing = callback_(frame_span);
                if (!continue_processing) {
                    return make_unexpected(FrameError::stop_requested);
                }

                ++frames_emitted;
                ++emitted_frames_;
                active_index_ = (active_index_ + 1) % buffer_count_;
                fill_offset_ = 0;
            }
        }

        return frames_emitted;
    }

    /**
     * Flush any partial frame (no padding).
     *
     * Emits remaining samples as a shorter frame if any are buffered.
     *
     * @return 0 or 1 (frames emitted), or error
     */
    expected<size_t, FrameError> flush_partial() {
        if (fill_offset_ == 0) {
            return size_t{0};
        }

        auto frame_span = std::span<const SampleT>(buffers_[active_index_].data(), fill_offset_);
        bool continue_processing = callback_(frame_span);

        active_index_ = (active_index_ + 1) % buffer_count_;
        fill_offset_ = 0;
        ++emitted_frames_;

        if (!continue_processing) {
            return make_unexpected(FrameError::stop_requested);
        }

        return size_t{1};
    }

    /**
     * Reset state, clearing accumulated samples and frame count.
     */
    void reset() noexcept {
        fill_offset_ = 0;
        active_index_ = 0;
        emitted_frames_ = 0;
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// Get samples per complete frame
    size_t samples_per_frame() const noexcept { return samples_per_frame_; }

    /// Get frame size in bytes
    size_t frame_size_bytes() const noexcept { return samples_per_frame_ * sizeof(SampleT); }

    /// Get total frames emitted since construction or last reset
    size_t emitted_frames() const noexcept { return emitted_frames_; }

    /// Get number of samples currently buffered
    size_t buffered_samples() const noexcept { return fill_offset_; }

    /// Check if using ping-pong mode
    bool is_ping_pong() const noexcept { return buffer_count_ == 2; }

private:
    std::array<std::span<SampleT>, 2> buffers_{};
    size_t buffer_count_ = 1;
    size_t active_index_ = 0;
    size_t fill_offset_ = 0;
    size_t samples_per_frame_;
    size_t emitted_frames_ = 0;
    Callback callback_;

    void validate_construction(size_t buf_a_size, size_t buf_b_size, size_t samples_per_frame) {
        if (samples_per_frame == 0) {
            throw std::invalid_argument("samples_per_frame must be > 0");
        }
        if (buf_a_size < samples_per_frame) {
            throw std::invalid_argument("Buffer A too small for samples_per_frame");
        }
        if (buf_b_size > 0 && buf_b_size < samples_per_frame) {
            throw std::invalid_argument("Buffer B too small for samples_per_frame");
        }
    }
};

// ============================================================================
// Convenience type aliases
// ============================================================================

/**
 * Callback type using std::function for simpler API (with overhead).
 */
template <typename SampleT>
using FrameCallback = std::function<bool(std::span<const SampleT>)>;

/**
 * SampleFramer with std::function callback for easier use.
 *
 * Simpler to use but has std::function overhead. For zero-overhead,
 * use SampleFramer directly with a lambda or functor.
 */
template <ValidSampleType SampleT>
using SimpleSampleFramer = SampleFramer<SampleT, FrameCallback<SampleT>>;

} // namespace vrtigo::utils
