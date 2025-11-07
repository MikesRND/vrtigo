#pragma once

#include "../core/types.hpp"
#include "../core/endian.hpp"
#include "../core/concepts.hpp"
#include "../core/trailer.hpp"
#include <cstring>
#include <cstdio>
#include <span>
#include <stdexcept>

namespace vrtio {

template<
    packet_type Type,
    tsi_type TSI = tsi_type::none,
    tsf_type TSF = tsf_type::none,
    bool HasTrailer = false,
    size_t PayloadWords = 0
>
    requires (Type == packet_type::signal_data_no_stream ||
              Type == packet_type::signal_data_with_stream) &&
             ValidPayloadWords<PayloadWords>
class signal_packet {
public:
    // Compile-time packet configuration
    static constexpr packet_type packet_type_v = Type;
    static constexpr tsi_type tsi_type_v = TSI;
    static constexpr tsf_type tsf_type_v = TSF;
    static constexpr bool has_trailer = HasTrailer;
    static constexpr size_t payload_words = PayloadWords;

    // Derived constants
    static constexpr bool has_stream_id = (Type == packet_type::signal_data_with_stream);

    // Size calculation (in 32-bit words)
    static constexpr size_t header_words = 1;
    static constexpr size_t stream_id_words = has_stream_id ? 1 : 0;
    static constexpr size_t tsi_words = (TSI != tsi_type::none) ? 1 : 0;
    static constexpr size_t tsf_words = (TSF != tsf_type::none) ? 2 : 0;
    static constexpr size_t trailer_words = HasTrailer ? 1 : 0;

    static constexpr size_t total_words =
        header_words + stream_id_words + tsi_words +
        tsf_words + PayloadWords + trailer_words;

    // Compile-time check: ensure total packet size fits in 16-bit size field
    static_assert(total_words <= max_packet_words,
                  "Packet size exceeds maximum (65535 words). "
                  "Reduce payload size or remove optional fields.");

    static constexpr size_t size_bytes = total_words * vrt_word_size;
    static constexpr size_t payload_size_bytes = PayloadWords * vrt_word_size;

    // Field offsets (in 32-bit words)
    static constexpr size_t header_offset = 0;
    static constexpr size_t stream_id_offset = header_offset + header_words;
    static constexpr size_t tsi_offset = stream_id_offset + stream_id_words;
    static constexpr size_t tsf_offset = tsi_offset + tsi_words;
    static constexpr size_t payload_offset = tsf_offset + tsf_words;
    static constexpr size_t trailer_offset = payload_offset + PayloadWords;

    // Constructor: creates view over user-provided buffer
    // If init=true, initializes a new packet; otherwise just wraps existing data
    //
    // SAFETY WARNING: When init=false (parsing untrusted data), you MUST call
    // validate() before accessing packet fields. Nothing prevents unsafe usage!
    //
    // Example of UNSAFE pattern (DO NOT DO THIS):
    //   signal_packet packet(untrusted_buffer, false);
    //   auto id = packet.stream_id();  // DANGEROUS! No validation!
    //
    // Correct pattern:
    //   signal_packet packet(untrusted_buffer, false);
    //   if (packet.validate(buffer_size) == validation_error::none) {
    //       auto id = packet.stream_id();  // Safe after validation
    //   }
    //
    // Phase 2 Enhancement: This API will be replaced with type-safe factory
    // methods that enforce validation at compile time. See design/phase1_v1.4.md
    explicit signal_packet(uint8_t* buffer, bool init = true) noexcept
        : buffer_(buffer) {
        if (init) {
            init_header();
        }
    }

    // Prevent copying (packet is a view)
    signal_packet(const signal_packet&) = delete;
    signal_packet& operator=(const signal_packet&) = delete;

    // Allow moving
    signal_packet(signal_packet&&) noexcept = default;
    signal_packet& operator=(signal_packet&&) noexcept = default;

    // Header field accessors (always available)

    // Get packet count (4-bit field: valid range 0-15)
    uint8_t packet_count() const noexcept {
        return (read_u32(header_offset) >> 16) & 0x0F;
    }

    // Set packet count (4-bit field: valid range 0-15)
    // NOTE: Values > 15 will be silently truncated to 4 bits.
    // In debug builds, this will trigger an assertion for values > 15.
    void set_packet_count(uint8_t count) noexcept {
        #ifndef NDEBUG
        if (count > 15) {
            // Debug build: warn about truncation
            // This is a logic error - packet count should be in range 0-15
            std::fprintf(stderr,
                "WARNING: packet_count value %u exceeds 4-bit limit (15). "
                "Value will be truncated to %u.\n",
                static_cast<unsigned>(count),
                static_cast<unsigned>(count & 0x0F));
        }
        #endif

        uint32_t header = read_u32(header_offset);
        header = (header & 0xFFF0FFFF) | ((static_cast<uint32_t>(count) & 0x0F) << 16);
        write_u32(header_offset, header);
    }

    uint16_t packet_size_words() const noexcept {
        return read_u32(header_offset) & 0xFFFF;
    }

    // Stream ID accessors (only if has_stream_id)

    uint32_t stream_id() const noexcept requires(has_stream_id) {
        return read_u32(stream_id_offset);
    }

    void set_stream_id(uint32_t id) noexcept requires(has_stream_id) {
        write_u32(stream_id_offset, id);
    }

    // Integer timestamp accessors (only if TSI != none)

    uint32_t timestamp_integer() const noexcept requires(TSI != tsi_type::none) {
        return read_u32(tsi_offset);
    }

    void set_timestamp_integer(uint32_t ts) noexcept requires(TSI != tsi_type::none) {
        write_u32(tsi_offset, ts);
    }

    // Fractional timestamp accessors (only if TSF != none)

    uint64_t timestamp_fractional() const noexcept requires(TSF != tsf_type::none) {
        return read_u64(tsf_offset);
    }

    void set_timestamp_fractional(uint64_t ts) noexcept requires(TSF != tsf_type::none) {
        write_u64(tsf_offset, ts);
    }

    // Trailer accessors (only if HasTrailer)

    uint32_t trailer() const noexcept requires(HasTrailer) {
        return read_u32(trailer_offset);
    }

    void set_trailer(uint32_t t) noexcept requires(HasTrailer) {
        write_u32(trailer_offset, t);
    }

    // Individual trailer field getters (only if HasTrailer)

    /**
     * Get the count of associated context packets (bits 0-6)
     * @return Context packets count (0-127)
     */
    uint8_t trailer_context_packets() const noexcept requires(HasTrailer) {
        return trailer::get_context_packets(trailer());
    }

    /**
     * Check if reference is locked (bit 8)
     * @return true if reference lock is indicated
     */
    bool trailer_reference_lock() const noexcept requires(HasTrailer) {
        return trailer::get_bit<trailer::reference_lock_bit>(trailer());
    }

    /**
     * Check AGC/MGC status (bit 9)
     * @return true if AGC/MGC is active
     */
    bool trailer_agc_mgc() const noexcept requires(HasTrailer) {
        return trailer::get_bit<trailer::agc_mgc_bit>(trailer());
    }

    /**
     * Check if signal is detected (bit 10)
     * @return true if signal is detected
     */
    bool trailer_detected_signal() const noexcept requires(HasTrailer) {
        return trailer::get_bit<trailer::detected_signal_bit>(trailer());
    }

    /**
     * Check if spectral inversion is indicated (bit 11)
     * @return true if spectral inversion is present
     */
    bool trailer_spectral_inversion() const noexcept requires(HasTrailer) {
        return trailer::get_bit<trailer::spectral_inversion_bit>(trailer());
    }

    /**
     * Check if over-range condition is indicated (bit 12)
     * @return true if over-range occurred
     */
    bool trailer_over_range() const noexcept requires(HasTrailer) {
        return trailer::get_bit<trailer::over_range_bit>(trailer());
    }

    /**
     * Check if sample loss is indicated (bit 13)
     * @return true if sample loss occurred
     */
    bool trailer_sample_loss() const noexcept requires(HasTrailer) {
        return trailer::get_bit<trailer::sample_loss_bit>(trailer());
    }

    /**
     * Check if time is calibrated (bit 16)
     * @return true if calibrated time indicator is set
     */
    bool trailer_calibrated_time() const noexcept requires(HasTrailer) {
        return trailer::get_bit<trailer::calibrated_time_bit>(trailer());
    }

    /**
     * Check if data is valid (bit 17)
     * @return true if valid data indicator is set
     */
    bool trailer_valid_data() const noexcept requires(HasTrailer) {
        return trailer::get_bit<trailer::valid_data_bit>(trailer());
    }

    /**
     * Check if reference point is indicated (bit 18)
     * @return true if reference point indicator is set
     */
    bool trailer_reference_point() const noexcept requires(HasTrailer) {
        return trailer::get_bit<trailer::reference_point_bit>(trailer());
    }

    /**
     * Check if signal is detected (bit 19)
     * @return true if signal detected indicator is set
     */
    bool trailer_signal_detected() const noexcept requires(HasTrailer) {
        return trailer::get_bit<trailer::signal_detected_bit>(trailer());
    }

    /**
     * Check if any error conditions are present
     * @return true if over-range or sample loss occurred
     */
    bool trailer_has_errors() const noexcept requires(HasTrailer) {
        return trailer::has_errors(trailer());
    }

    // Individual trailer field setters (only if HasTrailer)

    /**
     * Set the count of associated context packets (bits 0-6)
     * @param count Context packets count (0-127)
     */
    void set_trailer_context_packets(uint8_t count) noexcept requires(HasTrailer) {
        uint32_t t = trailer::set_field<trailer::context_packets_shift,
                                        trailer::context_packets_mask>(trailer(), count);
        set_trailer(t);
    }

    /**
     * Set reference lock status (bit 8)
     * @param locked true to indicate reference lock
     */
    void set_trailer_reference_lock(bool locked) noexcept requires(HasTrailer) {
        set_trailer(trailer::set_bit<trailer::reference_lock_bit>(trailer(), locked));
    }

    /**
     * Set AGC/MGC status (bit 9)
     * @param active true to indicate AGC/MGC is active
     */
    void set_trailer_agc_mgc(bool active) noexcept requires(HasTrailer) {
        set_trailer(trailer::set_bit<trailer::agc_mgc_bit>(trailer(), active));
    }

    /**
     * Set detected signal status (bit 10)
     * @param detected true to indicate signal is detected
     */
    void set_trailer_detected_signal(bool detected) noexcept requires(HasTrailer) {
        set_trailer(trailer::set_bit<trailer::detected_signal_bit>(trailer(), detected));
    }

    /**
     * Set spectral inversion status (bit 11)
     * @param inverted true to indicate spectral inversion
     */
    void set_trailer_spectral_inversion(bool inverted) noexcept requires(HasTrailer) {
        set_trailer(trailer::set_bit<trailer::spectral_inversion_bit>(trailer(), inverted));
    }

    /**
     * Set over-range condition (bit 12)
     * @param over_range true to indicate over-range occurred
     */
    void set_trailer_over_range(bool over_range) noexcept requires(HasTrailer) {
        set_trailer(trailer::set_bit<trailer::over_range_bit>(trailer(), over_range));
    }

    /**
     * Set sample loss condition (bit 13)
     * @param loss true to indicate sample loss occurred
     */
    void set_trailer_sample_loss(bool loss) noexcept requires(HasTrailer) {
        set_trailer(trailer::set_bit<trailer::sample_loss_bit>(trailer(), loss));
    }

    /**
     * Set calibrated time indicator (bit 16)
     * @param calibrated true to indicate time is calibrated
     */
    void set_trailer_calibrated_time(bool calibrated) noexcept requires(HasTrailer) {
        set_trailer(trailer::set_bit<trailer::calibrated_time_bit>(trailer(), calibrated));
    }

    /**
     * Set valid data indicator (bit 17)
     * @param valid true to indicate data is valid
     */
    void set_trailer_valid_data(bool valid) noexcept requires(HasTrailer) {
        set_trailer(trailer::set_bit<trailer::valid_data_bit>(trailer(), valid));
    }

    /**
     * Set reference point indicator (bit 18)
     * @param ref_point true to indicate reference point
     */
    void set_trailer_reference_point(bool ref_point) noexcept requires(HasTrailer) {
        set_trailer(trailer::set_bit<trailer::reference_point_bit>(trailer(), ref_point));
    }

    /**
     * Set signal detected indicator (bit 19)
     * @param detected true to indicate signal is detected
     */
    void set_trailer_signal_detected(bool detected) noexcept requires(HasTrailer) {
        set_trailer(trailer::set_bit<trailer::signal_detected_bit>(trailer(), detected));
    }

    /**
     * Set trailer to indicate good status (valid data and calibrated time)
     */
    void set_trailer_good_status() noexcept requires(HasTrailer) {
        set_trailer(trailer::create_good_status());
    }

    /**
     * Clear all trailer bits
     */
    void clear_trailer() noexcept requires(HasTrailer) {
        set_trailer(0);
    }

    // Payload access

    std::span<uint8_t, payload_size_bytes> payload() noexcept {
        return std::span<uint8_t, payload_size_bytes>(
            buffer_ + payload_offset * vrt_word_size,
            payload_size_bytes
        );
    }

    std::span<const uint8_t, payload_size_bytes> payload() const noexcept {
        return std::span<const uint8_t, payload_size_bytes>(
            buffer_ + payload_offset * vrt_word_size,
            payload_size_bytes
        );
    }

    void set_payload(const uint8_t* data, size_t size) noexcept {
        auto dest = payload();
        size_t copy_size = std::min(size, dest.size());
        std::memcpy(dest.data(), data, copy_size);
    }

    // Full packet buffer access

    std::span<uint8_t, size_bytes> as_bytes() noexcept {
        return std::span<uint8_t, size_bytes>(buffer_, size_bytes);
    }

    std::span<const uint8_t, size_bytes> as_bytes() const noexcept {
        return std::span<const uint8_t, size_bytes>(buffer_, size_bytes);
    }

    // Raw buffer pointer
    uint8_t* data() noexcept { return buffer_; }
    const uint8_t* data() const noexcept { return buffer_; }

    // Validation: verify packet header matches template configuration
    //
    // CRITICAL: You MUST call this method when parsing untrusted data before
    // accessing any packet fields. The current API doesn't enforce this!
    //
    // Phase 2 Enhancement: This will be called automatically by the parse()
    // factory method, making validation impossible to forget. Until then,
    // manually calling validate() is the developer's responsibility.
    //
    // Returns: validation_error::none on success, or specific error code
    validation_error validate(size_t buffer_size) const noexcept {
        // Check 1: Buffer must be at least as large as our packet
        if (buffer_size < size_bytes) {
            return validation_error::buffer_too_small;
        }

        // Read and decode header
        uint32_t header = read_u32(header_offset);

        // Check 2: Packet type field (bits 31-28)
        uint8_t header_type = (header >> 28) & 0x0F;
        if (header_type != static_cast<uint8_t>(Type)) {
            return validation_error::packet_type_mismatch;
        }

        // Check 3: Trailer bit (bit 26)
        bool header_has_trailer = (header >> 26) & 0x01;
        if (header_has_trailer != HasTrailer) {
            return validation_error::trailer_bit_mismatch;
        }

        // Check 4: TSI field (bits 23-22)
        uint8_t header_tsi = (header >> 22) & 0x03;
        if (header_tsi != static_cast<uint8_t>(TSI)) {
            return validation_error::tsi_mismatch;
        }

        // Check 5: TSF field (bits 21-20)
        uint8_t header_tsf = (header >> 20) & 0x03;
        if (header_tsf != static_cast<uint8_t>(TSF)) {
            return validation_error::tsf_mismatch;
        }

        // Check 6: Size field (bits 15-0)
        uint16_t header_size = header & 0xFFFF;
        if (header_size != total_words) {
            return validation_error::size_field_mismatch;
        }

        return validation_error::none;
    }

    // Simplified validation without buffer size check
    validation_error validate() const noexcept {
        return validate(size_bytes);
    }

private:
    uint8_t* buffer_;  // View over user-provided memory

    // Initialize header with packet metadata
    void init_header() noexcept {
        uint32_t header = 0;

        // Packet type (bits 31-28)
        header |= (static_cast<uint32_t>(Type) << 28);

        // Class ID indicator (bit 27) - not used in Phase 1
        header |= (0 << 27);

        // Trailer indicator (bit 26)
        header |= (HasTrailer ? 1U : 0U) << 26;

        // Reserved (bit 25)
        header |= (0 << 25);

        // TSM (bit 24) - not used
        header |= (0 << 24);

        // TSI (bits 23-22)
        header |= (static_cast<uint32_t>(TSI) << 22);

        // TSF (bits 21-20)
        header |= (static_cast<uint32_t>(TSF) << 20);

        // Packet count (bits 19-16) - initialized to 0
        header |= (0 << 16);

        // Packet size in words (bits 15-0)
        header |= (total_words & 0xFFFF);

        write_u32(header_offset, header);
    }

    // Endianness-aware read/write helpers

    uint32_t read_u32(size_t word_offset) const noexcept {
        uint32_t value;
        std::memcpy(&value, buffer_ + word_offset * vrt_word_size, sizeof(value));
        return detail::network_to_host32(value);
    }

    void write_u32(size_t word_offset, uint32_t value) noexcept {
        value = detail::host_to_network32(value);
        std::memcpy(buffer_ + word_offset * vrt_word_size, &value, sizeof(value));
    }

    uint64_t read_u64(size_t word_offset) const noexcept {
        uint64_t value;
        std::memcpy(&value, buffer_ + word_offset * vrt_word_size, sizeof(value));
        return detail::network_to_host64(value);
    }

    void write_u64(size_t word_offset, uint64_t value) noexcept {
        value = detail::host_to_network64(value);
        std::memcpy(buffer_ + word_offset * vrt_word_size, &value, sizeof(value));
    }
};

// Type alias for common packet configurations

// Type 0: Signal data without stream ID
template<tsi_type TSI = tsi_type::none,
         tsf_type TSF = tsf_type::none,
         bool HasTrailer = false,
         size_t PayloadWords = 256>
using signal_packet_no_stream = signal_packet<
    packet_type::signal_data_no_stream,
    TSI, TSF, HasTrailer, PayloadWords
>;

// Type 1: Signal data with stream ID
template<tsi_type TSI = tsi_type::none,
         tsf_type TSF = tsf_type::none,
         bool HasTrailer = false,
         size_t PayloadWords = 256>
using signal_packet_with_stream = signal_packet<
    packet_type::signal_data_with_stream,
    TSI, TSF, HasTrailer, PayloadWords
>;

}  // namespace vrtio
