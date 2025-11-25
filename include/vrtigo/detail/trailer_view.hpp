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
    explicit TrailerView(const uint8_t* word_ptr) noexcept
        : word_ptr_(word_ptr),
          accessor_(std::span<const std::byte, trailer_fields::TrailerLayout::required_bytes>{
              reinterpret_cast<const std::byte*>(word_ptr_),
              trailer_fields::TrailerLayout::required_bytes}) {}

public:
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
        if (!accessor_.get<trailer_fields::EBit>()) {
            return std::nullopt;
        }
        return accessor_.get<trailer_fields::ContextPacketCount>();
    }

    // ========================================================================
    // 8 Named Indicators (from Table 5.1.6-1)
    // Each returns indicator value if enabled, nullopt otherwise
    // ========================================================================

    /**
     * Calibrated Time Indicator (Enable bit 31, Indicator bit 19)
     */
    std::optional<bool> calibrated_time() const noexcept {
        return get_indicator_bitfield<trailer_fields::CalibratedTimeEnable,
                                      trailer_fields::CalibratedTimeIndicator>();
    }

    /**
     * Valid Data Indicator (Enable bit 30, Indicator bit 18)
     */
    std::optional<bool> valid_data() const noexcept {
        return get_indicator_bitfield<trailer_fields::ValidDataEnable,
                                      trailer_fields::ValidDataIndicator>();
    }

    /**
     * Reference Lock Indicator (Enable bit 29, Indicator bit 17)
     */
    std::optional<bool> reference_lock() const noexcept {
        return get_indicator_bitfield<trailer_fields::ReferenceLockEnable,
                                      trailer_fields::ReferenceLockIndicator>();
    }

    /**
     * AGC/MGC Indicator (Enable bit 28, Indicator bit 16)
     */
    std::optional<bool> agc_mgc() const noexcept {
        return get_indicator_bitfield<trailer_fields::AgcMgcEnable,
                                      trailer_fields::AgcMgcIndicator>();
    }

    /**
     * Detected Signal Indicator (Enable bit 27, Indicator bit 15)
     */
    std::optional<bool> detected_signal() const noexcept {
        return get_indicator_bitfield<trailer_fields::DetectedSignalEnable,
                                      trailer_fields::DetectedSignalIndicator>();
    }

    /**
     * Spectral Inversion Indicator (Enable bit 26, Indicator bit 14)
     */
    std::optional<bool> spectral_inversion() const noexcept {
        return get_indicator_bitfield<trailer_fields::SpectralInversionEnable,
                                      trailer_fields::SpectralInversionIndicator>();
    }

    /**
     * Over-range Indicator (Enable bit 25, Indicator bit 13)
     */
    std::optional<bool> over_range() const noexcept {
        return get_indicator_bitfield<trailer_fields::OverRangeEnable,
                                      trailer_fields::OverRangeIndicator>();
    }

    /**
     * Sample Loss Indicator (Enable bit 24, Indicator bit 12)
     */
    std::optional<bool> sample_loss() const noexcept {
        return get_indicator_bitfield<trailer_fields::SampleLossEnable,
                                      trailer_fields::SampleLossIndicator>();
    }

    // ========================================================================
    // Sample Frame and User-Defined Indicators (bits 11-8)
    // Per Table 5.1.6-1: Each has corresponding enable bit (bits 23-20)
    // ========================================================================

    /**
     * Sample Frame 1 indicator (Enable bit 23, Indicator bit 11)
     */
    std::optional<bool> sample_frame_1() const noexcept {
        return get_indicator_bitfield<trailer_fields::SampleFrame1Enable,
                                      trailer_fields::SampleFrame1Indicator>();
    }

    /**
     * Sample Frame 0 indicator (Enable bit 22, Indicator bit 10)
     */
    std::optional<bool> sample_frame_0() const noexcept {
        return get_indicator_bitfield<trailer_fields::SampleFrame0Enable,
                                      trailer_fields::SampleFrame0Indicator>();
    }

    /**
     * User Defined 1 indicator (Enable bit 21, Indicator bit 9)
     */
    std::optional<bool> user_defined_1() const noexcept {
        return get_indicator_bitfield<trailer_fields::UserDefined1Enable,
                                      trailer_fields::UserDefined1Indicator>();
    }

    /**
     * User Defined 0 indicator (Enable bit 20, Indicator bit 8)
     */
    std::optional<bool> user_defined_0() const noexcept {
        return get_indicator_bitfield<trailer_fields::UserDefined0Enable,
                                      trailer_fields::UserDefined0Indicator>();
    }

protected:
    const uint8_t* data_ptr() const noexcept { return word_ptr_; }

private:
    /**
     * Helper to get an indicator value if its enable bit is set
     */
    template <typename EnableField, typename IndicatorField>
    std::optional<bool> get_indicator_bitfield() const noexcept {
        if (!accessor_.get<EnableField>()) {
            return std::nullopt;
        }
        return accessor_.get<IndicatorField>();
    }

    const uint8_t* word_ptr_;
    detail::ConstBitFieldAccessor<trailer_fields::TrailerLayout> accessor_;
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
          word_ptr_mut_(word_ptr),
          mutable_accessor_(std::span<std::byte, trailer_fields::TrailerLayout::required_bytes>{
              reinterpret_cast<std::byte*>(word_ptr),
              trailer_fields::TrailerLayout::required_bytes}) {}

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
        mutable_accessor_.set<trailer_fields::EBit>(true);
        mutable_accessor_.set<trailer_fields::ContextPacketCount>(count & 0x7F); // Clamp to 7 bits
    }

    /**
     * Clear context packet count (sets E bit to 0, making count invalid)
     */
    void clear_context_packet_count() noexcept {
        mutable_accessor_.set<trailer_fields::EBit>(false);
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
        set_indicator_bitfield<trailer_fields::CalibratedTimeEnable,
                               trailer_fields::CalibratedTimeIndicator>(value);
    }

    /**
     * Set Valid Data indicator
     * Sets enable bit 30 and indicator bit 18
     */
    void set_valid_data(bool value) noexcept {
        set_indicator_bitfield<trailer_fields::ValidDataEnable, trailer_fields::ValidDataIndicator>(
            value);
    }

    /**
     * Set Reference Lock indicator
     * Sets enable bit 29 and indicator bit 17
     */
    void set_reference_lock(bool value) noexcept {
        set_indicator_bitfield<trailer_fields::ReferenceLockEnable,
                               trailer_fields::ReferenceLockIndicator>(value);
    }

    /**
     * Set AGC/MGC indicator
     * Sets enable bit 28 and indicator bit 16
     */
    void set_agc_mgc(bool value) noexcept {
        set_indicator_bitfield<trailer_fields::AgcMgcEnable, trailer_fields::AgcMgcIndicator>(
            value);
    }

    /**
     * Set Detected Signal indicator
     * Sets enable bit 27 and indicator bit 15
     */
    void set_detected_signal(bool value) noexcept {
        set_indicator_bitfield<trailer_fields::DetectedSignalEnable,
                               trailer_fields::DetectedSignalIndicator>(value);
    }

    /**
     * Set Spectral Inversion indicator
     * Sets enable bit 26 and indicator bit 14
     */
    void set_spectral_inversion(bool value) noexcept {
        set_indicator_bitfield<trailer_fields::SpectralInversionEnable,
                               trailer_fields::SpectralInversionIndicator>(value);
    }

    /**
     * Set Over-range indicator
     * Sets enable bit 25 and indicator bit 13
     */
    void set_over_range(bool value) noexcept {
        set_indicator_bitfield<trailer_fields::OverRangeEnable, trailer_fields::OverRangeIndicator>(
            value);
    }

    /**
     * Set Sample Loss indicator
     * Sets enable bit 24 and indicator bit 12
     */
    void set_sample_loss(bool value) noexcept {
        set_indicator_bitfield<trailer_fields::SampleLossEnable,
                               trailer_fields::SampleLossIndicator>(value);
    }

    // ========================================================================
    // 8 Named Indicators - Clear methods
    // Each clear method sets only the enable bit to 0
    // ========================================================================

    void clear_calibrated_time() noexcept {
        mutable_accessor_.set<trailer_fields::CalibratedTimeEnable>(false);
    }

    void clear_valid_data() noexcept {
        mutable_accessor_.set<trailer_fields::ValidDataEnable>(false);
    }

    void clear_reference_lock() noexcept {
        mutable_accessor_.set<trailer_fields::ReferenceLockEnable>(false);
    }

    void clear_agc_mgc() noexcept { mutable_accessor_.set<trailer_fields::AgcMgcEnable>(false); }

    void clear_detected_signal() noexcept {
        mutable_accessor_.set<trailer_fields::DetectedSignalEnable>(false);
    }

    void clear_spectral_inversion() noexcept {
        mutable_accessor_.set<trailer_fields::SpectralInversionEnable>(false);
    }

    void clear_over_range() noexcept {
        mutable_accessor_.set<trailer_fields::OverRangeEnable>(false);
    }

    void clear_sample_loss() noexcept {
        mutable_accessor_.set<trailer_fields::SampleLossEnable>(false);
    }

    // ========================================================================
    // Sample Frame and User-Defined - Setters
    // Per Table 5.1.6-1: Each has corresponding enable bit (bits 23-20)
    // ========================================================================

    void set_sample_frame_1(bool value) noexcept {
        set_indicator_bitfield<trailer_fields::SampleFrame1Enable,
                               trailer_fields::SampleFrame1Indicator>(value);
    }

    void set_sample_frame_0(bool value) noexcept {
        set_indicator_bitfield<trailer_fields::SampleFrame0Enable,
                               trailer_fields::SampleFrame0Indicator>(value);
    }

    void set_user_defined_1(bool value) noexcept {
        set_indicator_bitfield<trailer_fields::UserDefined1Enable,
                               trailer_fields::UserDefined1Indicator>(value);
    }

    void set_user_defined_0(bool value) noexcept {
        set_indicator_bitfield<trailer_fields::UserDefined0Enable,
                               trailer_fields::UserDefined0Indicator>(value);
    }

    // ========================================================================
    // Sample Frame and User-Defined - Clear methods
    // ========================================================================

    void clear_sample_frame_1() noexcept {
        mutable_accessor_.set<trailer_fields::SampleFrame1Enable>(false);
    }

    void clear_sample_frame_0() noexcept {
        mutable_accessor_.set<trailer_fields::SampleFrame0Enable>(false);
    }

    void clear_user_defined_1() noexcept {
        mutable_accessor_.set<trailer_fields::UserDefined1Enable>(false);
    }

    void clear_user_defined_0() noexcept {
        mutable_accessor_.set<trailer_fields::UserDefined0Enable>(false);
    }

private:
    // Inherit type aliases from base class (TrailerView already defines these)

    /**
     * Helper to set an indicator: sets enable bit to 1 and indicator bit to value
     */
    template <typename EnableField, typename IndicatorField>
    void set_indicator_bitfield(bool value) noexcept {
        mutable_accessor_.set<EnableField>(true);
        mutable_accessor_.set<IndicatorField>(value);
    }

    uint8_t* word_ptr_mut_;
    detail::BitFieldAccessor<trailer_fields::TrailerLayout> mutable_accessor_;
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
        // Set E bit (bit 7)
        value_ |= (1U << 7);
        // Set count (bits 0-6), clamped to 7 bits
        value_ = (value_ & ~0x7FU) | (count & 0x7FU);
        return *this;
    }

    // ========================================================================
    // Named Indicators
    // ========================================================================

    constexpr TrailerBuilder& calibrated_time(bool indicator_value) noexcept {
        return set_indicator(31, 19, indicator_value);
    }

    constexpr TrailerBuilder& valid_data(bool indicator_value) noexcept {
        return set_indicator(30, 18, indicator_value);
    }

    constexpr TrailerBuilder& reference_lock(bool indicator_value) noexcept {
        return set_indicator(29, 17, indicator_value);
    }

    constexpr TrailerBuilder& agc_mgc(bool indicator_value) noexcept {
        return set_indicator(28, 16, indicator_value);
    }

    constexpr TrailerBuilder& detected_signal(bool indicator_value) noexcept {
        return set_indicator(27, 15, indicator_value);
    }

    constexpr TrailerBuilder& spectral_inversion(bool indicator_value) noexcept {
        return set_indicator(26, 14, indicator_value);
    }

    constexpr TrailerBuilder& over_range(bool indicator_value) noexcept {
        return set_indicator(25, 13, indicator_value);
    }

    constexpr TrailerBuilder& sample_loss(bool indicator_value) noexcept {
        return set_indicator(24, 12, indicator_value);
    }

    // ========================================================================
    // Sample Frame and User-Defined
    // Per Table 5.1.6-1: Each has corresponding enable bit (bits 23-20)
    // ========================================================================

    constexpr TrailerBuilder& sample_frame_1(bool value) noexcept {
        return set_indicator(23, 11, value);
    }

    constexpr TrailerBuilder& sample_frame_0(bool value) noexcept {
        return set_indicator(22, 10, value);
    }

    constexpr TrailerBuilder& user_defined_1(bool value) noexcept {
        return set_indicator(21, 9, value);
    }

    constexpr TrailerBuilder& user_defined_0(bool value) noexcept {
        return set_indicator(20, 8, value);
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
    /**
     * Constexpr helper to set enable/indicator bit pair
     * @param enable_bit Enable bit position (20-31)
     * @param indicator_bit Indicator bit position (8-19)
     * @param indicator_value Value for indicator bit
     */
    constexpr TrailerBuilder& set_indicator(uint32_t enable_bit, uint32_t indicator_bit,
                                            bool indicator_value) noexcept {
        // Set enable bit to 1
        value_ |= (1U << enable_bit);
        // Set indicator bit to value
        if (indicator_value) {
            value_ |= (1U << indicator_bit);
        } else {
            value_ &= ~(1U << indicator_bit);
        }
        return *this;
    }

    uint32_t value_ = 0; // HOST byte order (not network!)
};

} // namespace vrtigo
