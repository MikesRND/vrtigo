// Copyright (c) 2025 Michael Smith
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

namespace vrtigo::utils::fileio {

/**
 * @brief Status codes for VRT file writer operations
 *
 * Represents the current state of a writer, including error conditions
 * that may arise during write or flush operations.
 */
enum class WriterStatus : uint8_t {
    ready,             ///< Writer is ready for write operations
    write_error,       ///< Write system call failed
    flush_error,       ///< Flush operation failed
    closed,            ///< File has been closed
    disk_full,         ///< Disk is full (ENOSPC)
    permission_denied, ///< Permission denied (EACCES/EPERM)
    invalid_packet     ///< Parse error variant rejected
};

/**
 * @brief Convert WriterStatus to human-readable string
 *
 * @param status The status code to convert
 * @return String representation of the status
 */
constexpr const char* writer_status_string(WriterStatus status) noexcept {
    switch (status) {
        case WriterStatus::ready:
            return "ready";
        case WriterStatus::write_error:
            return "write_error";
        case WriterStatus::flush_error:
            return "flush_error";
        case WriterStatus::closed:
            return "closed";
        case WriterStatus::disk_full:
            return "disk_full";
        case WriterStatus::permission_denied:
            return "permission_denied";
        case WriterStatus::invalid_packet:
            return "invalid_packet";
    }
    return "unknown";
}

} // namespace vrtigo::utils::fileio
