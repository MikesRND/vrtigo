#pragma once

#include <algorithm>
#include <complex>
#include <span>

#include <cstring>

#include "detail/endian.hpp"
#include "detail/sample_traits.hpp"

namespace vrtigo {

namespace detail {

// Read a single scalar component from buffer with endian conversion
template <typename T>
[[nodiscard]] inline T read_component(const uint8_t* buffer) noexcept {
    T value;
    std::memcpy(&value, buffer, sizeof(T));

    if constexpr (sizeof(T) == 2) {
        uint16_t raw;
        std::memcpy(&raw, &value, sizeof(raw));
        raw = network_to_host16(raw);
        std::memcpy(&value, &raw, sizeof(value));
    } else if constexpr (sizeof(T) == 4) {
        uint32_t raw;
        std::memcpy(&raw, &value, sizeof(raw));
        raw = network_to_host32(raw);
        std::memcpy(&value, &raw, sizeof(value));
    } else if constexpr (sizeof(T) == 8) {
        uint64_t raw;
        std::memcpy(&raw, &value, sizeof(raw));
        raw = network_to_host64(raw);
        std::memcpy(&value, &raw, sizeof(value));
    }
    // sizeof(T) == 1: no swap needed

    return value;
}

// Write a single scalar component to buffer with endian conversion
template <typename T>
inline void write_component(uint8_t* buffer, T value) noexcept {
    if constexpr (sizeof(T) == 2) {
        uint16_t raw;
        std::memcpy(&raw, &value, sizeof(raw));
        raw = host_to_network16(raw);
        std::memcpy(&value, &raw, sizeof(value));
    } else if constexpr (sizeof(T) == 4) {
        uint32_t raw;
        std::memcpy(&raw, &value, sizeof(raw));
        raw = host_to_network32(raw);
        std::memcpy(&value, &raw, sizeof(value));
    } else if constexpr (sizeof(T) == 8) {
        uint64_t raw;
        std::memcpy(&raw, &value, sizeof(raw));
        raw = host_to_network64(raw);
        std::memcpy(&value, &raw, sizeof(value));
    }
    // sizeof(T) == 1: no swap needed

    std::memcpy(buffer, &value, sizeof(T));
}

// Read a sample (scalar or complex) from buffer
template <ValidSampleType T>
[[nodiscard]] inline T read_sample(const uint8_t* buffer) noexcept {
    if constexpr (SampleTraits<T>::sample_size != SampleTraits<T>::component_size) {
        // Complex type: read I then Q
        using Component = typename T::value_type;
        Component real = read_component<Component>(buffer);
        Component imag = read_component<Component>(buffer + sizeof(Component));
        return T(real, imag);
    } else {
        return read_component<T>(buffer);
    }
}

// Write a sample (scalar or complex) to buffer
template <ValidSampleType T>
inline void write_sample(uint8_t* buffer, T value) noexcept {
    if constexpr (SampleTraits<T>::sample_size != SampleTraits<T>::component_size) {
        // Complex type: write I then Q
        using Component = typename T::value_type;
        write_component<Component>(buffer, value.real());
        write_component<Component>(buffer + sizeof(Component), value.imag());
    } else {
        write_component<T>(buffer, value);
    }
}

} // namespace detail

/**
 * Read-only view over VRT payload samples.
 *
 * Provides endian-safe access to samples stored in network byte order.
 * Zero allocation - operates directly on the provided span.
 *
 * @tparam T Sample type (int8/16/32, float, double, or std::complex variants)
 *
 * Example:
 *   auto view = dynamic::DataPacketView::parse(buffer).value();
 *   auto samples = SampleSpanView<int16_t>(view.payload());
 *   for (size_t i = 0; i < samples.count(); ++i) {
 *       int16_t s = samples[i];
 *   }
 */
template <ValidSampleType T>
class SampleSpanView {
protected:
    std::span<const uint8_t> data_;

public:
    using value_type = T;
    using traits = detail::SampleTraits<T>;

    explicit SampleSpanView(std::span<const uint8_t> payload) noexcept : data_(payload) {}

    /**
     * Number of complete samples in the payload.
     * Trailing bytes that don't form a complete sample are ignored.
     */
    [[nodiscard]] size_t count() const noexcept { return data_.size() / traits::sample_size; }

    /**
     * Size of the underlying payload in bytes.
     */
    [[nodiscard]] size_t size_bytes() const noexcept { return data_.size(); }

    /**
     * Access a single sample by index (unchecked).
     * @param index Zero-based sample index. Must be < count().
     * @return Sample value in host byte order.
     */
    [[nodiscard]] T operator[](size_t index) const noexcept {
        return detail::read_sample<T>(data_.data() + index * traits::sample_size);
    }

    /**
     * Bulk read samples into user buffer.
     * @param dest Destination buffer for samples.
     * @return Number of samples copied (min of dest.size() and count()).
     */
    size_t read(std::span<T> dest) const noexcept { return read(dest, 0); }

    /**
     * Bulk read samples into user buffer starting at offset.
     * @param dest Destination buffer for samples.
     * @param start_index First sample index to read.
     * @return Number of samples copied. Returns 0 if start_index >= count().
     */
    size_t read(std::span<T> dest, size_t start_index) const noexcept {
        if (start_index >= count()) {
            return 0;
        }
        size_t available = count() - start_index;
        size_t to_copy = std::min(dest.size(), available);
        for (size_t i = 0; i < to_copy; ++i) {
            dest[i] = (*this)[start_index + i];
        }
        return to_copy;
    }
};

/**
 * Mutable view over VRT payload samples.
 *
 * Provides endian-safe read and write access to samples.
 * Inherits all read operations from SampleSpanView.
 *
 * @tparam T Sample type (int8/16/32, float, double, or std::complex variants)
 *
 * Example:
 *   Packet packet(buffer);
 *   auto samples = SampleSpan<float>(packet.payload());
 *   samples.set(0, 1.5f);
 */
template <ValidSampleType T>
class SampleSpan : public SampleSpanView<T> {
private:
    std::span<uint8_t> mutable_data_;

public:
    using typename SampleSpanView<T>::value_type;
    using typename SampleSpanView<T>::traits;

    explicit SampleSpan(std::span<uint8_t> payload) noexcept
        : SampleSpanView<T>(payload),
          mutable_data_(payload) {}

    /**
     * Set a single sample by index (unchecked).
     * @param index Zero-based sample index. Must be < count().
     * @param value Sample value in host byte order.
     */
    void set(size_t index, T value) noexcept {
        detail::write_sample<T>(mutable_data_.data() + index * traits::sample_size, value);
    }

    /**
     * Bulk write samples from user buffer.
     * @param src Source buffer of samples.
     * @return Number of samples written (min of src.size() and count()).
     */
    size_t write(std::span<const T> src) noexcept { return write(src, 0); }

    /**
     * Bulk write samples from user buffer starting at offset.
     * @param src Source buffer of samples.
     * @param start_index First sample index to write.
     * @return Number of samples written. Returns 0 if start_index >= count().
     */
    size_t write(std::span<const T> src, size_t start_index) noexcept {
        if (start_index >= this->count()) {
            return 0;
        }
        size_t available = this->count() - start_index;
        size_t to_copy = std::min(src.size(), available);
        for (size_t i = 0; i < to_copy; ++i) {
            set(start_index + i, src[i]);
        }
        return to_copy;
    }
};

} // namespace vrtigo
