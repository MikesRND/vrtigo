#pragma once

#include "endian.hpp"
#include "trailer.hpp"
#include <cstdint>
#include <cstring>

namespace vrtio {

/**
 * Immutable view over a trailer word stored in network byte order.
 *
 * The view performs endian conversion on access and exposes typed getters
 * for each of the defined trailer fields.
 */
class ConstTrailerView {
public:
    explicit ConstTrailerView(const uint8_t* word_ptr) noexcept
        : word_ptr_(word_ptr) {}

    uint32_t raw() const noexcept {
        uint32_t value;
        std::memcpy(&value, word_ptr_, sizeof(value));
        return detail::network_to_host32(value);
    }

    uint8_t context_packets() const noexcept {
        return trailer::get_context_packets(raw());
    }

    bool reference_lock() const noexcept {
        return trailer::get_bit<trailer::reference_lock_bit>(raw());
    }

    bool agc_mgc() const noexcept {
        return trailer::get_bit<trailer::agc_mgc_bit>(raw());
    }

    bool detected_signal() const noexcept {
        return trailer::get_bit<trailer::detected_signal_bit>(raw());
    }

    bool spectral_inversion() const noexcept {
        return trailer::get_bit<trailer::spectral_inversion_bit>(raw());
    }

    bool over_range() const noexcept {
        return trailer::get_bit<trailer::over_range_bit>(raw());
    }

    bool sample_loss() const noexcept {
        return trailer::get_bit<trailer::sample_loss_bit>(raw());
    }

    bool calibrated_time() const noexcept {
        return trailer::get_bit<trailer::calibrated_time_bit>(raw());
    }

    bool valid_data() const noexcept {
        return trailer::get_bit<trailer::valid_data_bit>(raw());
    }

    bool reference_point() const noexcept {
        return trailer::get_bit<trailer::reference_point_bit>(raw());
    }

    bool signal_detected() const noexcept {
        return trailer::get_bit<trailer::signal_detected_bit>(raw());
    }

    bool has_errors() const noexcept {
        return trailer::has_errors(raw());
    }

protected:
    const uint8_t* data_ptr() const noexcept { return word_ptr_; }

private:
    const uint8_t* word_ptr_;
};

/**
 * Mutable view over a trailer word with typed setters.
 */
class TrailerView : public ConstTrailerView {
public:
    explicit TrailerView(uint8_t* word_ptr) noexcept
        : ConstTrailerView(word_ptr), word_ptr_mut_(word_ptr) {}

    void set_raw(uint32_t value) noexcept {
        value = detail::host_to_network32(value);
        std::memcpy(word_ptr_mut_, &value, sizeof(value));
    }

    void clear() noexcept { set_raw(0); }

    void set_context_packets(uint8_t count) noexcept {
        modify([count](uint32_t value) {
            return trailer::set_field<trailer::context_packets_shift,
                                      trailer::context_packets_mask>(value, count);
        });
    }

    void set_reference_lock(bool locked) noexcept {
        modify([locked](uint32_t value) {
            return trailer::set_bit<trailer::reference_lock_bit>(value, locked);
        });
    }

    void set_agc_mgc(bool active) noexcept {
        modify([active](uint32_t value) {
            return trailer::set_bit<trailer::agc_mgc_bit>(value, active);
        });
    }

    void set_detected_signal(bool detected) noexcept {
        modify([detected](uint32_t value) {
            return trailer::set_bit<trailer::detected_signal_bit>(value, detected);
        });
    }

    void set_spectral_inversion(bool inverted) noexcept {
        modify([inverted](uint32_t value) {
            return trailer::set_bit<trailer::spectral_inversion_bit>(value, inverted);
        });
    }

    void set_over_range(bool over_range) noexcept {
        modify([over_range](uint32_t value) {
            return trailer::set_bit<trailer::over_range_bit>(value, over_range);
        });
    }

    void set_sample_loss(bool loss) noexcept {
        modify([loss](uint32_t value) {
            return trailer::set_bit<trailer::sample_loss_bit>(value, loss);
        });
    }

    void set_calibrated_time(bool calibrated) noexcept {
        modify([calibrated](uint32_t value) {
            return trailer::set_bit<trailer::calibrated_time_bit>(value, calibrated);
        });
    }

    void set_valid_data(bool valid) noexcept {
        modify([valid](uint32_t value) {
            return trailer::set_bit<trailer::valid_data_bit>(value, valid);
        });
    }

    void set_reference_point(bool ref_point) noexcept {
        modify([ref_point](uint32_t value) {
            return trailer::set_bit<trailer::reference_point_bit>(value, ref_point);
        });
    }

    void set_signal_detected(bool detected) noexcept {
        modify([detected](uint32_t value) {
            return trailer::set_bit<trailer::signal_detected_bit>(value, detected);
        });
    }

private:
    template<typename Fn>
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

    constexpr TrailerBuilder& context_packets(uint8_t count) noexcept {
        value_ = trailer::set_field<trailer::context_packets_shift,
                                    trailer::context_packets_mask>(value_, count);
        return *this;
    }

    constexpr TrailerBuilder& reference_lock(bool locked) noexcept {
        value_ = trailer::set_bit<trailer::reference_lock_bit>(value_, locked);
        return *this;
    }

    constexpr TrailerBuilder& agc_mgc(bool active) noexcept {
        value_ = trailer::set_bit<trailer::agc_mgc_bit>(value_, active);
        return *this;
    }

    constexpr TrailerBuilder& detected_signal(bool detected) noexcept {
        value_ = trailer::set_bit<trailer::detected_signal_bit>(value_, detected);
        return *this;
    }

    constexpr TrailerBuilder& spectral_inversion(bool inverted) noexcept {
        value_ = trailer::set_bit<trailer::spectral_inversion_bit>(value_, inverted);
        return *this;
    }

    constexpr TrailerBuilder& over_range(bool over_range) noexcept {
        value_ = trailer::set_bit<trailer::over_range_bit>(value_, over_range);
        return *this;
    }

    constexpr TrailerBuilder& sample_loss(bool loss) noexcept {
        value_ = trailer::set_bit<trailer::sample_loss_bit>(value_, loss);
        return *this;
    }

    constexpr TrailerBuilder& calibrated_time(bool calibrated) noexcept {
        value_ = trailer::set_bit<trailer::calibrated_time_bit>(value_, calibrated);
        return *this;
    }

    constexpr TrailerBuilder& valid_data(bool valid) noexcept {
        value_ = trailer::set_bit<trailer::valid_data_bit>(value_, valid);
        return *this;
    }

    constexpr TrailerBuilder& reference_point(bool ref_point) noexcept {
        value_ = trailer::set_bit<trailer::reference_point_bit>(value_, ref_point);
        return *this;
    }

    constexpr TrailerBuilder& signal_detected(bool detected) noexcept {
        value_ = trailer::set_bit<trailer::signal_detected_bit>(value_, detected);
        return *this;
    }

    TrailerBuilder& from_view(ConstTrailerView view) noexcept {
        value_ = view.raw();
        return *this;
    }

    TrailerView apply(TrailerView view) const noexcept {
        view.set_raw(value_);
        return view;
    }

    constexpr operator uint32_t() const noexcept { return value_; }

private:
    uint32_t value_ = 0;
};

}  // namespace vrtio
