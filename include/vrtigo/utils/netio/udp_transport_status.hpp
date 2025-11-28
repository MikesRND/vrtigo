#pragma once

#include "vrtigo/types.hpp"

#include <cstddef>
#include <cstdint>

namespace vrtigo::utils::netio {

/**
 * @brief Status information for UDP datagram reception
 *
 * Tracks the state of the last UDP receive operation, including
 * error conditions, packet metadata, and truncation information.
 *
 * This status type is specific to UDP transport and does not
 * attempt to unify with file I/O semantics.
 */
struct UDPTransportStatus {
    /**
     * @brief State of the last UDP receive operation
     */
    enum class State : uint8_t {
        /** Packet successfully received and ready for parsing */
        packet_ready,

        /** Socket has been closed (orderly shutdown) */
        socket_closed,

        /** Fatal socket error occurred */
        socket_error,

        /** Datagram exceeded buffer size and was truncated */
        datagram_truncated,

        /** Receive timeout (SO_RCVTIMEO expired) - non-terminal */
        timeout,

        /** Receive interrupted by signal (EINTR) - non-terminal */
        interrupted
    };

    /** Current state */
    State state{State::packet_ready};

    /** Number of bytes actually received (may be less than actual_size if truncated) */
    size_t bytes_received{0};

    /**
     * Actual datagram size for truncated packets
     *
     * When state == datagram_truncated, this contains the full size of the
     * datagram that was sent. Compare with bytes_received to determine how
     * much buffer space is needed.
     */
    size_t actual_size{0};

    /**
     * VRT packet header in host byte order
     *
     * Only valid if bytes_received >= 4.
     * Used for error diagnostics and ParseError creation.
     */
    uint32_t header{0};

    /**
     * Packet type decoded from header
     *
     * Only valid if bytes_received >= 4, otherwise PacketType::signal_data_no_id.
     */
    PacketType packet_type{PacketType::signal_data_no_id};

    /** Platform errno value for socket_error state */
    int errno_value{0};

    /**
     * @brief Check if the socket is in a terminal error state
     *
     * @return true if socket is closed or has a fatal error
     */
    bool is_terminal() const noexcept {
        return state == State::socket_closed || state == State::socket_error;
    }

    /**
     * @brief Check if the last datagram was truncated
     *
     * When true, actual_size contains the full datagram size.
     * Caller can reallocate a larger buffer and create a new reader.
     *
     * @return true if datagram exceeded buffer and was truncated
     */
    bool is_truncated() const noexcept { return state == State::datagram_truncated; }
};

/**
 * @brief Convert UDPTransportStatus::State to human-readable string
 *
 * @param state The transport state to convert
 * @return String representation of the state
 */
constexpr const char* transport_state_string(UDPTransportStatus::State state) noexcept {
    switch (state) {
        case UDPTransportStatus::State::packet_ready:
            return "packet_ready";
        case UDPTransportStatus::State::socket_closed:
            return "socket_closed";
        case UDPTransportStatus::State::socket_error:
            return "socket_error";
        case UDPTransportStatus::State::datagram_truncated:
            return "datagram_truncated";
        case UDPTransportStatus::State::timeout:
            return "timeout";
        case UDPTransportStatus::State::interrupted:
            return "interrupted";
        default:
            return "unknown";
    }
}

} // namespace vrtigo::utils::netio
