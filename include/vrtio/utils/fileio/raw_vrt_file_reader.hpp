#pragma once

#include <array>
#include <span>
#include <string>
#include <utility>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vrtio/types.hpp>

#include "../../detail/endian.hpp"
#include "../../detail/header_decode.hpp"

namespace vrtio::utils::fileio {

/**
 * @brief Low-level VRT file reader returning raw packet bytes
 *
 * Reads VRT packets from binary files and returns raw bytes without parsing or validation.
 * Use this for maximum performance or when you need custom packet processing.
 *
 * **For most use cases, prefer VRTFileReader which returns validated, type-safe packet views.**
 *
 * Provides multiple API patterns:
 * - read_next(): User-provided buffer (no allocation)
 * - read_next_span(): Internal scratch buffer (no heap allocation)
 * - for_each_packet(): Streaming callback pattern
 *
 * The MaxPacketWords template parameter controls the internal scratch buffer size.
 * Note: The scratch buffer is MaxPacketWords * 4 bytes. For the default (65535 words),
 * this is 256KB. This makes the class non-copyable and best allocated on the heap
 * or with move semantics.
 *
 * @tparam MaxPacketWords Maximum packet size in 32-bit words (default: 65535)
 *
 * @warning This class is MOVE-ONLY due to the large internal buffer.
 *          Do not copy instances of this class.
 *
 * @note The MaxPacketWords parameter must be > 0 and <= 65535 (VRT maximum)
 * @see VRTFileReader for high-level type-safe packet reading
 */
template <uint16_t MaxPacketWords = 65535>
class RawVRTFileReader {
    // Compile-time validation
    static_assert(MaxPacketWords > 0, "MaxPacketWords must be positive");
    static_assert(MaxPacketWords <= max_packet_words,
                  "MaxPacketWords exceeds VRT specification maximum (65535)");

    // Warning if buffer is very large (>1MB)
    static_assert(
        MaxPacketWords * vrt_word_size <= 1024 * 1024 || MaxPacketWords == 65535,
        "WARNING: Scratch buffer exceeds 1MB. Consider heap allocation or smaller MaxPacketWords.");

public:
    /**
     * @brief Result of a packet read operation
     *
     * Contains detailed information about the read attempt including
     * success/failure status, packet metadata, and buffer requirements.
     */
    struct ReadResult {
        ValidationError error;       ///< Error code (none = success)
        PacketType type;             ///< Packet type from header
        size_t packet_size_bytes;    ///< Actual packet size in bytes
        size_t buffer_size_required; ///< Required buffer size (set when buffer too small)
        size_t file_offset;          ///< File offset where packet starts
        uint32_t header;             ///< Header word (host byte order, already converted)

        /// Check if read was successful
        bool is_valid() const noexcept { return error == ValidationError::none; }

        /// Check if this is end-of-file (not an error)
        bool is_eof() const noexcept {
            return error == ValidationError::buffer_too_small && packet_size_bytes == 0;
        }
    };

    /**
     * @brief Open a VRT file for reading
     *
     * @param filepath Path to VRT binary file
     * @throws std::runtime_error if file cannot be opened
     */
    explicit RawVRTFileReader(const char* filepath)
        : file_(nullptr),
          file_size_(0),
          current_offset_(0),
          packets_read_(0),
          last_error_{} {
        file_ = std::fopen(filepath, "rb");
        if (!file_) {
            throw std::runtime_error(std::string("Failed to open file: ") + filepath);
        }

        // Determine file size
        std::fseek(file_, 0, SEEK_END);
        file_size_ = std::ftell(file_);
        std::fseek(file_, 0, SEEK_SET);
    }

    /**
     * @brief Destructor - closes file handle
     */
    ~RawVRTFileReader() noexcept {
        if (file_) {
            std::fclose(file_);
        }
    }

    // Non-copyable due to large scratch buffer and FILE* ownership
    RawVRTFileReader(const RawVRTFileReader&) = delete;
    RawVRTFileReader& operator=(const RawVRTFileReader&) = delete;

    // Move-only semantics
    RawVRTFileReader(RawVRTFileReader&& other) noexcept
        : file_(other.file_),
          file_size_(other.file_size_),
          current_offset_(other.current_offset_),
          packets_read_(other.packets_read_),
          scratch_buffer_(std::move(other.scratch_buffer_)),
          last_error_(other.last_error_) {
        other.file_ = nullptr;
    }

    RawVRTFileReader& operator=(RawVRTFileReader&& other) noexcept {
        if (this != &other) {
            if (file_) {
                std::fclose(file_);
            }
            file_ = other.file_;
            file_size_ = other.file_size_;
            current_offset_ = other.current_offset_;
            packets_read_ = other.packets_read_;
            scratch_buffer_ = std::move(other.scratch_buffer_);
            last_error_ = other.last_error_;
            other.file_ = nullptr;
        }
        return *this;
    }

    /**
     * @brief Read next packet into user-provided buffer
     *
     * This method allows the user to provide their own buffer for zero-allocation
     * operation. The buffer must be at least as large as the packet being read.
     *
     * @param buffer User-provided buffer (must not be null)
     * @param buffer_size Size of user buffer in bytes
     * @return ReadResult with error code and packet information
     *
     * @note If buffer is too small, result.buffer_size_required will be set
     *       to the actual size needed, allowing smart resizing.
     */
    ReadResult read_next(uint8_t* buffer, size_t buffer_size) noexcept {
        ReadResult result{};
        result.file_offset = current_offset_;

        // Check for EOF
        if (current_offset_ >= file_size_) {
            result.error = ValidationError::buffer_too_small; // Signals EOF
            result.packet_size_bytes = 0;
            return result;
        }

        // Read header (4 bytes minimum)
        if (file_size_ - current_offset_ < vrt_word_size) {
            result.error = ValidationError::buffer_too_small;
            result.packet_size_bytes = 0;
            return result;
        }

        uint32_t header_raw;
        if (std::fread(&header_raw, vrt_word_size, 1, file_) != 1) {
            result.error = ValidationError::buffer_too_small;
            result.packet_size_bytes = 0;
            return result;
        }

        // Early validation: decode header
        uint32_t header_host = detail::network_to_host32(header_raw);
        auto decoded = detail::decode_header(header_host);
        result.header = header_host;
        result.type = decoded.type;

        // Validate packet type
        if (!detail::is_valid_packet_type(decoded.type)) {
            result.error = ValidationError::invalid_packet_type;
            current_offset_ += vrt_word_size;
            return result;
        }

        // Calculate packet size
        result.packet_size_bytes = decoded.size_words * vrt_word_size;
        result.buffer_size_required = result.packet_size_bytes;

        // Validate packet size
        if (decoded.size_words == 0 || decoded.size_words > MaxPacketWords) {
            result.error = ValidationError::size_field_mismatch;
            current_offset_ += vrt_word_size;
            return result;
        }

        // Check user buffer size
        if (buffer_size < result.packet_size_bytes) {
            result.error = ValidationError::buffer_too_small;
            // Rewind to start of packet for potential retry
            std::fseek(file_, current_offset_, SEEK_SET);
            return result;
        }

        // Check file has enough data
        if (current_offset_ + result.packet_size_bytes > file_size_) {
            result.error = ValidationError::buffer_too_small;
            std::fseek(file_, current_offset_, SEEK_SET);
            return result;
        }

        // Copy header to buffer
        std::memcpy(buffer, &header_raw, vrt_word_size);

        // Read rest of packet
        size_t remaining = result.packet_size_bytes - vrt_word_size;
        if (remaining > 0) {
            if (std::fread(buffer + vrt_word_size, 1, remaining, file_) != remaining) {
                result.error = ValidationError::buffer_too_small;
                std::fseek(file_, current_offset_, SEEK_SET);
                return result;
            }
        }

        // Success
        result.error = ValidationError::none;
        current_offset_ += result.packet_size_bytes;
        packets_read_++;

        return result;
    }

    /**
     * @brief Read next packet using internal scratch buffer
     *
     * Returns a span view into the internal scratch buffer. The span is valid
     * until the next call to read_next_span() or object destruction.
     *
     * @return Span of packet data, or empty span on error/EOF
     *
     * @note An empty span means either EOF or error. Use last_error() to distinguish:
     *       - EOF: last_error().is_eof() returns true
     *       - Error: last_error().error != none and !is_eof()
     *
     * @warning The returned span is invalidated by the next read_next_span() call!
     */
    std::span<const uint8_t> read_next_span() noexcept {
        last_error_ = read_next(scratch_buffer_.data(), scratch_buffer_.size());

        if (last_error_.is_valid()) {
            return std::span<const uint8_t>(scratch_buffer_.data(), last_error_.packet_size_bytes);
        }

        return std::span<const uint8_t>();
    }

    /**
     * @brief Get detailed error information from last read_next_span() call
     *
     * Use this to distinguish EOF from errors when read_next_span() returns empty span.
     *
     * @return ReadResult from last read_next_span() call
     */
    const ReadResult& last_error() const noexcept { return last_error_; }

    /**
     * @brief Stream all packets through a callback function
     *
     * Efficiently processes all packets in the file using the callback pattern.
     * The callback receives a span to each packet in sequence.
     *
     * @tparam Callback Function/lambda type with signature: bool(std::span<const uint8_t>, const
     * ReadResult&)
     * @param callback Function called for each packet. Return false to stop iteration.
     * @return Number of packets successfully processed
     *
     * @note The callback receives:
     *       - std::span<const uint8_t>: Packet data (valid only during callback)
     *       - const ReadResult&: Packet metadata (type, size, offset, etc.)
     *
     * Example:
     * @code
     * size_t count = reader.for_each_packet([](auto packet, auto& info) {
     *     std::cout << "Packet type: " << static_cast<int>(info.type) << "\n";
     *     return true;  // Continue processing
     * });
     * @endcode
     */
    template <typename Callback>
    size_t for_each_packet(Callback&& callback) noexcept {
        size_t processed = 0;

        while (true) {
            auto packet = read_next_span();

            // Check for EOF
            if (packet.empty() && last_error_.is_eof()) {
                break;
            }

            // Check for error
            if (packet.empty()) {
                break;
            }

            // Call user callback
            bool continue_processing = callback(packet, last_error_);
            processed++;

            if (!continue_processing) {
                break;
            }
        }

        return processed;
    }

    /**
     * @brief Rewind file to beginning for re-reading
     */
    void rewind() noexcept {
        if (file_) {
            std::fseek(file_, 0, SEEK_SET);
            current_offset_ = 0;
            packets_read_ = 0;
            last_error_ = ReadResult{};
        }
    }

    /**
     * @brief Get current file position in bytes
     */
    size_t tell() const noexcept { return current_offset_; }

    /**
     * @brief Get total file size in bytes
     */
    size_t size() const noexcept { return file_size_; }

    /**
     * @brief Get number of packets read so far
     */
    size_t packets_read() const noexcept { return packets_read_; }

    /**
     * @brief Check if file is still open
     */
    bool is_open() const noexcept { return file_ != nullptr; }

private:
    FILE* file_;            ///< File handle
    size_t file_size_;      ///< Total file size in bytes
    size_t current_offset_; ///< Current read position
    size_t packets_read_;   ///< Number of packets read
    std::array<uint8_t, MaxPacketWords * vrt_word_size> scratch_buffer_; ///< Internal buffer
    ReadResult last_error_; ///< Last error from read_next_span()
};

} // namespace vrtio::utils::fileio
