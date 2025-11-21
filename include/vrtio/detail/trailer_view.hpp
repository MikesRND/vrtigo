#pragma once

#include <optional>

#include <cstdint>
#include <cstring>

#include "endian.hpp"
#include "trailer.hpp"

namespace vrtigo {

/**
 * Read-only view over a trailer word stored in network byte order.
 *
 * The trailer implements VITA 49.2 Section 5.1.6 with enable/indicator bit pairing.
 * Each of the 8 named indicators has an enable bit (31-24) and an indicator bit (19-12).
 * When the enable bit is 0, the indicator value is undefined/invalid.
 */
class TrailerView {
public:
    explicit TrailerView(const uint8_t* word_ptr) noexcept : word_ptr_(word_ptr) {}

    /**
     * Get raw trailer word value (host byte order)
     */
    uint32_t raw() const noexcept {
        uint32_t value;
        std::memcpy(&value, word_ptr_, sizeof(value));
        return detail::network_to_host32(value);
    }

    // ========================================================================
    // Associated Context Packet Count (Rule 5.1.6-13)
    // ========================================================================

    /**
     * Get associated context packet count.
     * Returns count (0-127) if E bit is set, nullopt otherwise.
     * Per Rule 5.1.6-13: When E=1, count is valid. When E=0, count is undefined.
     */
    std::optional<uint8_t> context_packet_count() const noexcept {
        auto value = raw();
        bool e_bit = trailer::extract_bit<trailer::e_bit>(value);
        if (!e_bit) {
            return std::nullopt;
        }
        return static_cast<uint8_t>(
            trailer::extract_field<trailer::context_packet_count_shift,
                                   trailer::context_packet_count_mask>(value));
    }

    // ========================================================================
    // 8 Named Indicators (from Table 5.1.6-1)
    // Each returns indicator value if enabled, nullopt otherwise
    // ========================================================================

    /**
     * Calibrated Time Indicator (Enable bit 31, Indicator bit 19)
     */
    std::optional<bool> calibrated_time() const noexcept {
        return get_indicator<trailer::calibrated_time_enable_bit,
                             trailer::calibrated_time_indicator_bit>();
    }

    /**
     * Valid Data Indicator (Enable bit 30, Indicator bit 18)
     */
    std::optional<bool> valid_data() const noexcept {
        return get_indicator<trailer::valid_data_enable_bit, trailer::valid_data_indicator_bit>();
    }

    /**
     * Reference Lock Indicator (Enable bit 29, Indicator bit 17)
     */
    std::optional<bool> reference_lock() const noexcept {
        return get_indicator<trailer::reference_lock_enable_bit,
                             trailer::reference_lock_indicator_bit>();
    }

    /**
     * AGC/MGC Indicator (Enable bit 28, Indicator bit 16)
     */
    std::optional<bool> agc_mgc() const noexcept {
        return get_indicator<trailer::agc_mgc_enable_bit, trailer::agc_mgc_indicator_bit>();
    }

    /**
     * Detected Signal Indicator (Enable bit 27, Indicator bit 15)
     */
    std::optional<bool> detected_signal() const noexcept {
        return get_indicator<trailer::detected_signal_enable_bit,
                             trailer::detected_signal_indicator_bit>();
    }

    /**
     * Spectral Inversion Indicator (Enable bit 26, Indicator bit 14)
     */
    std::optional<bool> spectral_inversion() const noexcept {
        return get_indicator<trailer::spectral_inversion_enable_bit,
                             trailer::spectral_inversion_indicator_bit>();
    }

    /**
     * Over-range Indicator (Enable bit 25, Indicator bit 13)
     */
    std::optional<bool> over_range() const noexcept {
        return get_indicator<trailer::over_range_enable_bit, trailer::over_range_indicator_bit>();
    }

    /**
     * Sample Loss Indicator (Enable bit 24, Indicator bit 12)
     */
    std::optional<bool> sample_loss() const noexcept {
        return get_indicator<trailer::sample_loss_enable_bit, trailer::sample_loss_indicator_bit>();
    }

    // ========================================================================
    // Sample Frame and User-Defined Indicators (bits 11-8)
    // No enable bits - direct boolean access
    // ========================================================================

    /**
     * Sample Frame 1 indicator (bit 11)
     */
    bool sample_frame_1() const noexcept {
        return trailer::extract_bit<trailer::sample_frame_1_bit>(raw());
    }

    /**
     * Sample Frame 0 indicator (bit 10)
     */
    bool sample_frame_0() const noexcept {
        return trailer::extract_bit<trailer::sample_frame_0_bit>(raw());
    }

    /**
     * User Defined 1 indicator (bit 9)
     */
    bool user_defined_1() const noexcept {
        return trailer::extract_bit<trailer::user_defined_1_bit>(raw());
    }

    /**
     * User Defined 0 indicator (bit 8)
     */
    bool user_defined_0() const noexcept {
        return trailer::extract_bit<trailer::user_defined_0_bit>(raw());
    }

protected:
    const uint8_t* data_ptr() const noexcept { return word_ptr_; }

private:
    /**
     * Helper to get an indicator value if its enable bit is set
     */
    template <uint32_t EnableBit, uint32_t IndicatorBit>
    std::optional<bool> get_indicator() const noexcept {
        auto value = raw();
        bool enabled = trailer::extract_bit<EnableBit>(value);
        if (!enabled) {
            return std::nullopt;
        }
        return trailer::extract_bit<IndicatorBit>(value);
    }

    const uint8_t* word_ptr_;
};

/**
 * Mutable view over a trailer word with typed setters.
 *
 * Setters automatically handle enable/indicator bit pairing:
 * - set_X(value) sets enable bit to 1 and indicator bit to value
 * - clear_X() sets enable bit to 0 (making indicator invalid)
 */
class MutableTrailerView : public TrailerView {
public:
    explicit MutableTrailerView(uint8_t* word_ptr) noexcept
        : TrailerView(word_ptr),
          word_ptr_mut_(word_ptr) {}

    /**
     * Set raw trailer word value (host byte order)
     */
    void set_raw(uint32_t value) noexcept {
        value = detail::host_to_network32(value);
        std::memcpy(word_ptr_mut_, &value, sizeof(value));
    }

    /**
     * Clear entire trailer word to zero
     */
    void clear() noexcept { set_raw(0); }

    // ========================================================================
    // Associated Context Packet Count
    // ========================================================================

    /**
     * Set associated context packet count.
     * Sets E bit (bit 7) to 1 and count (bits 6-0) to specified value.
     * Count is clamped to 0-127 range.
     */
    void set_context_packet_count(uint8_t count) noexcept {
        modify([count](uint32_t value) {
            // Set E bit
            value = trailer::set_bit<trailer::e_bit>(value, true);
            // Set count field (bits 6-0)
            return trailer::set_field<trailer::context_packet_count_shift,
                                      trailer::context_packet_count_mask>(value, count);
        });
    }

    /**
     * Clear context packet count (sets E bit to 0, making count invalid)
     */
    void clear_context_packet_count() noexcept {
        modify([](uint32_t value) { return trailer::clear_bit<trailer::e_bit>(value); });
    }

    // ========================================================================
    // 8 Named Indicators - Setters
    // Each setter sets the enable bit AND the indicator bit
    // ========================================================================

    /**
     * Set Calibrated Time indicator
     * Sets enable bit 31 and indicator bit 19
     */
    void set_calibrated_time(bool value) noexcept {
        set_indicator<trailer::calibrated_time_enable_bit, trailer::calibrated_time_indicator_bit>(
            value);
    }

    /**
     * Set Valid Data indicator
     * Sets enable bit 30 and indicator bit 18
     */
    void set_valid_data(bool value) noexcept {
        set_indicator<trailer::valid_data_enable_bit, trailer::valid_data_indicator_bit>(value);
    }

    /**
     * Set Reference Lock indicator
     * Sets enable bit 29 and indicator bit 17
     */
    void set_reference_lock(bool value) noexcept {
        set_indicator<trailer::reference_lock_enable_bit, trailer::reference_lock_indicator_bit>(
            value);
    }

    /**
     * Set AGC/MGC indicator
     * Sets enable bit 28 and indicator bit 16
     */
    void set_agc_mgc(bool value) noexcept {
        set_indicator<trailer::agc_mgc_enable_bit, trailer::agc_mgc_indicator_bit>(value);
    }

    /**
     * Set Detected Signal indicator
     * Sets enable bit 27 and indicator bit 15
     */
    void set_detected_signal(bool value) noexcept {
        set_indicator<trailer::detected_signal_enable_bit, trailer::detected_signal_indicator_bit>(
            value);
    }

    /**
     * Set Spectral Inversion indicator
     * Sets enable bit 26 and indicator bit 14
     */
    void set_spectral_inversion(bool value) noexcept {
        set_indicator<trailer::spectral_inversion_enable_bit,
                      trailer::spectral_inversion_indicator_bit>(value);
    }

    /**
     * Set Over-range indicator
     * Sets enable bit 25 and indicator bit 13
     */
    void set_over_range(bool value) noexcept {
        set_indicator<trailer::over_range_enable_bit, trailer::over_range_indicator_bit>(value);
    }

    /**
     * Set Sample Loss indicator
     * Sets enable bit 24 and indicator bit 12
     */
    void set_sample_loss(bool value) noexcept {
        set_indicator<trailer::sample_loss_enable_bit, trailer::sample_loss_indicator_bit>(value);
    }

    // ========================================================================
    // 8 Named Indicators - Clear methods
    // Each clear method sets only the enable bit to 0
    // ========================================================================

    void clear_calibrated_time() noexcept { clear_enable<trailer::calibrated_time_enable_bit>(); }

    void clear_valid_data() noexcept { clear_enable<trailer::valid_data_enable_bit>(); }

    void clear_reference_lock() noexcept { clear_enable<trailer::reference_lock_enable_bit>(); }

    void clear_agc_mgc() noexcept { clear_enable<trailer::agc_mgc_enable_bit>(); }

    void clear_detected_signal() noexcept { clear_enable<trailer::detected_signal_enable_bit>(); }

    void clear_spectral_inversion() noexcept {
        clear_enable<trailer::spectral_inversion_enable_bit>();
    }

    void clear_over_range() noexcept { clear_enable<trailer::over_range_enable_bit>(); }

    void clear_sample_loss() noexcept { clear_enable<trailer::sample_loss_enable_bit>(); }

    // ========================================================================
    // Sample Frame and User-Defined - Setters
    // ========================================================================

    void set_sample_frame_1(bool value) noexcept {
        modify([value](uint32_t v) {
            return trailer::set_bit<trailer::sample_frame_1_bit>(v, value);
        });
    }

    void set_sample_frame_0(bool value) noexcept {
        modify([value](uint32_t v) {
            return trailer::set_bit<trailer::sample_frame_0_bit>(v, value);
        });
    }

    void set_user_defined_1(bool value) noexcept {
        modify([value](uint32_t v) {
            return trailer::set_bit<trailer::user_defined_1_bit>(v, value);
        });
    }

    void set_user_defined_0(bool value) noexcept {
        modify([value](uint32_t v) {
            return trailer::set_bit<trailer::user_defined_0_bit>(v, value);
        });
    }

    // ========================================================================
    // Sample Frame and User-Defined - Clear methods
    // ========================================================================

    void clear_sample_frame_1() noexcept {
        modify([](uint32_t v) { return trailer::clear_bit<trailer::sample_frame_1_bit>(v); });
    }

    void clear_sample_frame_0() noexcept {
        modify([](uint32_t v) { return trailer::clear_bit<trailer::sample_frame_0_bit>(v); });
    }

    void clear_user_defined_1() noexcept {
        modify([](uint32_t v) { return trailer::clear_bit<trailer::user_defined_1_bit>(v); });
    }

    void clear_user_defined_0() noexcept {
        modify([](uint32_t v) { return trailer::clear_bit<trailer::user_defined_0_bit>(v); });
    }

private:
    /**
     * Helper to set an indicator: sets enable bit to 1 and indicator bit to value
     */
    template <uint32_t EnableBit, uint32_t IndicatorBit>
    void set_indicator(bool value) noexcept {
        modify([value](uint32_t v) {
            // Set enable bit to 1
            v = trailer::set_bit<EnableBit>(v, true);
            // Set indicator bit to value
            return trailer::set_bit<IndicatorBit>(v, value);
        });
    }

    /**
     * Helper to clear an enable bit (makes corresponding indicator invalid)
     */
    template <uint32_t EnableBit>
    void clear_enable() noexcept {
        modify([](uint32_t v) { return trailer::clear_bit<EnableBit>(v); });
    }

    /**
     * Read-modify-write helper
     */
    template <typename Fn>
    void modify(Fn&& fn) noexcept {
        auto value = raw();
        value = fn(value);
        set_raw(value);
    }

    uint8_t* word_ptr_mut_;
};

/**
 * Value-type builder for composing trailer words with a fluent API.
 */
class TrailerBuilder {
public:
    constexpr TrailerBuilder() = default;
    explicit constexpr TrailerBuilder(uint32_t value) noexcept : value_(value) {}

    constexpr uint32_t value() const noexcept { return value_; }

    constexpr TrailerBuilder& raw(uint32_t value) noexcept {
        value_ = value;
        return *this;
    }

    constexpr TrailerBuilder& clear() noexcept {
        value_ = 0;
        return *this;
    }

    // ========================================================================
    // Context Packet Count
    // ========================================================================

    constexpr TrailerBuilder& context_packet_count(uint8_t count) noexcept {
        // Set E bit
        value_ = trailer::set_bit<trailer::e_bit>(value_, true);
        // Set count
        value_ = trailer::set_field<trailer::context_packet_count_shift,
                                    trailer::context_packet_count_mask>(value_, count);
        return *this;
    }

    // ========================================================================
    // Named Indicators
    // ========================================================================

    constexpr TrailerBuilder& calibrated_time(bool indicator_value) noexcept {
        value_ = trailer::set_bit<trailer::calibrated_time_enable_bit>(value_, true);
        value_ = trailer::set_bit<trailer::calibrated_time_indicator_bit>(value_, indicator_value);
        return *this;
    }

    constexpr TrailerBuilder& valid_data(bool indicator_value) noexcept {
        value_ = trailer::set_bit<trailer::valid_data_enable_bit>(value_, true);
        value_ = trailer::set_bit<trailer::valid_data_indicator_bit>(value_, indicator_value);
        return *this;
    }

    constexpr TrailerBuilder& reference_lock(bool indicator_value) noexcept {
        value_ = trailer::set_bit<trailer::reference_lock_enable_bit>(value_, true);
        value_ = trailer::set_bit<trailer::reference_lock_indicator_bit>(value_, indicator_value);
        return *this;
    }

    constexpr TrailerBuilder& agc_mgc(bool indicator_value) noexcept {
        value_ = trailer::set_bit<trailer::agc_mgc_enable_bit>(value_, true);
        value_ = trailer::set_bit<trailer::agc_mgc_indicator_bit>(value_, indicator_value);
        return *this;
    }

    constexpr TrailerBuilder& detected_signal(bool indicator_value) noexcept {
        value_ = trailer::set_bit<trailer::detected_signal_enable_bit>(value_, true);
        value_ = trailer::set_bit<trailer::detected_signal_indicator_bit>(value_, indicator_value);
        return *this;
    }

    constexpr TrailerBuilder& spectral_inversion(bool indicator_value) noexcept {
        value_ = trailer::set_bit<trailer::spectral_inversion_enable_bit>(value_, true);
        value_ =
            trailer::set_bit<trailer::spectral_inversion_indicator_bit>(value_, indicator_value);
        return *this;
    }

    constexpr TrailerBuilder& over_range(bool indicator_value) noexcept {
        value_ = trailer::set_bit<trailer::over_range_enable_bit>(value_, true);
        value_ = trailer::set_bit<trailer::over_range_indicator_bit>(value_, indicator_value);
        return *this;
    }

    constexpr TrailerBuilder& sample_loss(bool indicator_value) noexcept {
        value_ = trailer::set_bit<trailer::sample_loss_enable_bit>(value_, true);
        value_ = trailer::set_bit<trailer::sample_loss_indicator_bit>(value_, indicator_value);
        return *this;
    }

    // ========================================================================
    // Sample Frame and User-Defined
    // ========================================================================

    constexpr TrailerBuilder& sample_frame_1(bool value) noexcept {
        value_ = trailer::set_bit<trailer::sample_frame_1_bit>(value_, value);
        return *this;
    }

    constexpr TrailerBuilder& sample_frame_0(bool value) noexcept {
        value_ = trailer::set_bit<trailer::sample_frame_0_bit>(value_, value);
        return *this;
    }

    constexpr TrailerBuilder& user_defined_1(bool value) noexcept {
        value_ = trailer::set_bit<trailer::user_defined_1_bit>(value_, value);
        return *this;
    }

    constexpr TrailerBuilder& user_defined_0(bool value) noexcept {
        value_ = trailer::set_bit<trailer::user_defined_0_bit>(value_, value);
        return *this;
    }

    // ========================================================================
    // Utility Methods
    // ========================================================================

    TrailerBuilder& from_view(TrailerView view) noexcept {
        value_ = view.raw();
        return *this;
    }

    MutableTrailerView apply(MutableTrailerView view) const noexcept {
        view.set_raw(value_);
        return view;
    }

    constexpr operator uint32_t() const noexcept { return value_; }

private:
    uint32_t value_ = 0;
};

} // namespace vrtigo
