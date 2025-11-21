// Copyright (c) 2025 Michael Smith
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <span>
#include <stdexcept>
#include <string>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace vrtigo::utils::fileio {

/**
 * @brief Low-level VRT file writer with internal buffering
 *
 * Writes raw VRT packet bytes to a binary file with internal buffering
 * to minimize system calls. Large packets that exceed the buffer size
 * bypass the buffer and are written directly.
 *
 * This is a low-level writer that accepts raw bytes without validation.
 * For high-level packet writing with type safety, use VRTFileWriter.
 *
 * Buffer Strategy:
 * - Packets smaller than remaining buffer space are copied to buffer
 * - Packets larger than buffer size trigger flush, then direct write
 * - Automatic flush when buffer is full
 * - Manual flush() available to force write of partial buffer
 *
 * Error Handling:
 * - Constructor may throw on file creation failure
 * - After construction, all operations are noexcept
 * - Errors stored in sticky state (remains until clear_error())
 * - Returns false on error, true on success
 * - errno preserved in last_errno()
 *
 * Thread Safety:
 * - Not thread-safe: single thread should own this instance
 * - Safe to move between threads (move-only)
 *
 * @tparam MaxPacketWords Maximum packet size in 32-bit words (default 65535)
 */
template <size_t MaxPacketWords = 65535>
class RawVRTFileWriter {
public:
    static constexpr size_t max_packet_words = 65535; // VRT spec maximum
    static constexpr size_t max_packet_bytes = max_packet_words * 4;
    static constexpr size_t buffer_size_bytes = MaxPacketWords * 4;

    static_assert(MaxPacketWords > 0, "MaxPacketWords must be positive");
    static_assert(MaxPacketWords <= max_packet_words, "MaxPacketWords exceeds VRT maximum");

    // Warn if buffer is excessively large
    static_assert(buffer_size_bytes <= 1024 * 1024,
                  "Buffer size exceeds 1MB - consider reducing MaxPacketWords");

    /**
     * @brief Create writer for new file
     *
     * Creates or truncates the file at the given path.
     *
     * @param file_path Path to output file
     * @throws std::runtime_error if file cannot be created
     */
    explicit RawVRTFileWriter(const std::string& file_path)
        : fd_(-1),
          buffer_used_(0),
          packets_written_(0),
          bytes_written_(0),
          last_errno_(0) {
        fd_ = ::open(file_path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd_ < 0) {
            throw std::runtime_error("Failed to create file: " + file_path +
                                     " (errno=" + std::to_string(errno) + ")");
        }
    }

    /**
     * @brief Destructor flushes and closes file
     *
     * Any buffered data is written before closing. Errors during
     * flush are silently ignored.
     */
    ~RawVRTFileWriter() {
        if (fd_ >= 0) {
            flush(); // Best effort - ignore errors
            ::close(fd_);
        }
    }

    // Move-only (large buffer, file handle ownership)
    RawVRTFileWriter(const RawVRTFileWriter&) = delete;
    RawVRTFileWriter& operator=(const RawVRTFileWriter&) = delete;

    RawVRTFileWriter(RawVRTFileWriter&& other) noexcept
        : fd_(other.fd_),
          buffer_(std::move(other.buffer_)),
          buffer_used_(other.buffer_used_),
          packets_written_(other.packets_written_),
          bytes_written_(other.bytes_written_),
          last_errno_(other.last_errno_) {
        other.fd_ = -1;
        other.buffer_used_ = 0;
        other.packets_written_ = 0;
        other.bytes_written_ = 0;
        other.last_errno_ = 0;
    }

    RawVRTFileWriter& operator=(RawVRTFileWriter&& other) noexcept {
        if (this != &other) {
            // Clean up existing state
            if (fd_ >= 0) {
                flush();
                ::close(fd_);
            }

            // Move from other
            fd_ = other.fd_;
            buffer_ = std::move(other.buffer_);
            buffer_used_ = other.buffer_used_;
            packets_written_ = other.packets_written_;
            bytes_written_ = other.bytes_written_;
            last_errno_ = other.last_errno_;

            // Reset other
            other.fd_ = -1;
            other.buffer_used_ = 0;
            other.packets_written_ = 0;
            other.bytes_written_ = 0;
            other.last_errno_ = 0;
        }
        return *this;
    }

    /**
     * @brief Write packet from raw pointer
     *
     * @param data Pointer to packet bytes
     * @param size Size of packet in bytes
     * @return true on success, false on error
     *
     * @note size must be > 0 and a multiple of 4 (VRT packets are word-aligned)
     */
    bool write_packet(const uint8_t* data, size_t size) noexcept {
        if (!is_open() || has_error()) {
            return false;
        }

        if (size == 0 || size % 4 != 0) {
            return false; // Invalid packet size
        }

        // Large packet: flush buffer then direct write
        if (size > buffer_size_bytes) {
            if (!flush()) {
                return false;
            }
            return write_direct(data, size);
        }

        // Buffer is too full: flush first
        if (buffer_used_ + size > buffer_size_bytes) {
            if (!flush()) {
                return false;
            }
        }

        // Copy to buffer
        std::memcpy(buffer_.data() + buffer_used_, data, size);
        buffer_used_ += size;
        packets_written_++;
        bytes_written_ += size;

        return true;
    }

    /**
     * @brief Write packet from span
     *
     * @param packet Span containing packet bytes
     * @return true on success, false on error
     */
    bool write_packet(std::span<const uint8_t> packet) noexcept {
        return write_packet(packet.data(), packet.size());
    }

    /**
     * @brief Flush buffered data to disk
     *
     * Forces write of any buffered data. Called automatically when
     * buffer is full or on destruction.
     *
     * @return true on success, false on error
     */
    bool flush() noexcept {
        if (!is_open()) {
            return false;
        }

        if (buffer_used_ == 0) {
            return true; // Nothing to flush
        }

        ssize_t written = ::write(fd_, buffer_.data(), buffer_used_);
        if (written < 0 || static_cast<size_t>(written) != buffer_used_) {
            last_errno_ = errno;
            return false;
        }

        buffer_used_ = 0;
        return true;
    }

    /**
     * @brief Get number of packets written
     *
     * @return Total packets written successfully
     */
    [[nodiscard]] size_t packets_written() const noexcept { return packets_written_; }

    /**
     * @brief Get number of bytes written
     *
     * @return Total bytes written (including buffered but not flushed)
     */
    [[nodiscard]] size_t bytes_written() const noexcept { return bytes_written_; }

    /**
     * @brief Check if file is open
     *
     * @return true if file is open
     */
    [[nodiscard]] bool is_open() const noexcept { return fd_ >= 0; }

    /**
     * @brief Check if error state is set
     *
     * @return true if an error has occurred
     */
    [[nodiscard]] bool has_error() const noexcept { return last_errno_ != 0; }

    /**
     * @brief Get last error number
     *
     * @return errno value from last failed operation (0 if no error)
     */
    [[nodiscard]] int last_errno() const noexcept { return last_errno_; }

    /**
     * @brief Clear error state
     *
     * Resets error state to allow retry of operations.
     */
    void clear_error() noexcept { last_errno_ = 0; }

private:
    /**
     * @brief Write data directly to file (bypass buffer)
     *
     * Used for large packets that exceed buffer size.
     */
    bool write_direct(const uint8_t* data, size_t size) noexcept {
        ssize_t written = ::write(fd_, data, size);
        if (written < 0 || static_cast<size_t>(written) != size) {
            last_errno_ = errno;
            return false;
        }

        packets_written_++;
        bytes_written_ += size;
        return true;
    }

    int fd_;                                        ///< File descriptor
    std::array<uint8_t, buffer_size_bytes> buffer_; ///< Internal write buffer
    size_t buffer_used_;                            ///< Bytes used in buffer
    size_t packets_written_;                        ///< Total packets written
    size_t bytes_written_;                          ///< Total bytes written
    int last_errno_;                                ///< Last error number
};

} // namespace vrtigo::utils::fileio
