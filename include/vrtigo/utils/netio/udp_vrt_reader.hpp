#pragma once

#include <array>
#include <chrono>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <unistd.h>

// Linux/POSIX socket headers
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

#include "../../detail/endian.hpp"
#include "../../detail/header_decode.hpp"
#include "../../detail/packet_parser.hpp"
#include "../../detail/packet_variant.hpp"
#include "../../dynamic.hpp"
#include "../../types.hpp"
#include "../detail/iteration_helpers.hpp"
#include "udp_transport_status.hpp"

namespace vrtigo::utils::netio {

/**
 * @brief Blocking UDP VRT packet reader (Linux/POSIX)
 *
 * Reads VRT packets from UDP datagrams with automatic validation and type-safe packet views.
 * This reader operates in BLOCKING mode, mirroring the semantics of VRTFileReader to maintain
 * API compatibility with shared iteration helpers.
 *
 * **IMPORTANT: BLOCKING-ONLY OPERATION**
 *
 * This reader relies on kernel-level blocking via recvmsg(). The underlying socket MUST remain
 * in blocking mode. Do NOT set the socket to non-blocking (O_NONBLOCK) as this will break the
 * std::nullopt == EOF contract and cause iteration helpers (for_each_validated_packet, etc.)
 * to terminate prematurely on transient EAGAIN errors.
 *
 * For non-blocking I/O, use a manual read loop with select/poll/epoll instead of the
 * for_each_* helpers, or consider implementing a separate non-blocking reader class.
 *
 * **UDP Datagram Mapping**
 *
 * This reader assumes each UDP datagram contains exactly one complete VRT packet.
 * The natural packet boundaries provided by UDP eliminate the buffering and reassembly
 * complexity required for stream-based protocols like TCP.
 *
 * **Datagram Truncation**
 *
 * If a datagram exceeds MaxPacketWords * 4 bytes, it will be detected via MSG_TRUNC and
 * returned as a ParseError. The actual datagram size is available in
 * transport_status().actual_size, allowing you to reallocate with a larger buffer if needed.
 *
 * @tparam MaxPacketWords Maximum packet size in 32-bit words (default: 65535)
 *
 * @warning This class is MOVE-ONLY due to the large internal scratch buffer.
 *
 * Example usage:
 * @code
 * // Create reader listening on UDP port 12345
 * UDPVRTReader<> reader(12345);
 *
 * // Read packets one at a time
 * while (auto pkt = reader.read_next_packet()) {
 *     std::visit([](auto&& p) {
 *         using T = std::decay_t<decltype(p)>;
 *         if constexpr (std::is_same_v<T, vrtigo::dynamic::DataPacket>) {
 *             auto payload = p.payload();
 *             // Process data...
 *         }
 *     }, *pkt);
 * }
 *
 * // Or use filtered iteration (same API as VRTFileReader)
 * reader.for_each_data_packet([](const vrtigo::dynamic::DataPacket& pkt) {
 *     // Process data packet
 *     return true; // continue
 * });
 * @endcode
 */
template <uint16_t MaxPacketWords = 65535>
class UDPVRTReader {
    static_assert(MaxPacketWords > 0, "MaxPacketWords must be positive");
    static_assert(MaxPacketWords <= max_packet_words,
                  "MaxPacketWords exceeds VRT specification maximum (65535)");

public:
    /**
     * @brief Create UDP reader listening on specified port
     *
     * Creates a UDP socket bound to INADDR_ANY (all interfaces) on the specified port.
     * The socket is configured for blocking operation.
     *
     * @param port UDP port to listen on
     * @throws std::runtime_error if socket creation or binding fails
     */
    explicit UDPVRTReader(uint16_t port)
        : socket_(-1),
          owns_socket_(true),
          scratch_buffer_{},
          status_{} {
        // Create UDP socket
        socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_ < 0) {
            throw std::runtime_error("Failed to create UDP socket");
        }

        // Bind to port
        struct sockaddr_in addr {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = INADDR_ANY;

        if (bind(socket_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
            close(socket_);
            throw std::runtime_error("Failed to bind UDP socket to port " + std::to_string(port));
        }

        // Socket is blocking by default - no need to change flags
    }

    /**
     * @brief Create UDP reader using existing socket
     *
     * Takes ownership of an existing socket file descriptor. The socket MUST be in blocking mode.
     *
     * @param socket_fd Existing socket file descriptor
     * @param take_ownership If true, socket will be closed in destructor
     *
     * @warning The socket must be configured for blocking operation.
     *          Non-blocking sockets will break iteration helper semantics.
     */
    explicit UDPVRTReader(int socket_fd, bool take_ownership = false)
        : socket_(socket_fd),
          owns_socket_(take_ownership),
          scratch_buffer_{},
          status_{} {
        if (socket_ < 0) {
            throw std::runtime_error("Invalid socket file descriptor");
        }
    }

    /**
     * @brief Destructor - closes socket if owned
     */
    ~UDPVRTReader() noexcept {
        if (owns_socket_ && socket_ >= 0) {
            close(socket_);
        }
    }

    // Non-copyable (due to socket and large buffer)
    UDPVRTReader(const UDPVRTReader&) = delete;
    UDPVRTReader& operator=(const UDPVRTReader&) = delete;

    // Move-only semantics
    UDPVRTReader(UDPVRTReader&& other) noexcept
        : socket_(other.socket_),
          owns_socket_(other.owns_socket_),
          scratch_buffer_(std::move(other.scratch_buffer_)),
          status_(other.status_) {
        other.socket_ = -1;
        other.owns_socket_ = false;
    }

    UDPVRTReader& operator=(UDPVRTReader&& other) noexcept {
        if (this != &other) {
            if (owns_socket_ && socket_ >= 0) {
                close(socket_);
            }
            socket_ = other.socket_;
            owns_socket_ = other.owns_socket_;
            scratch_buffer_ = std::move(other.scratch_buffer_);
            status_ = other.status_;
            other.socket_ = -1;
            other.owns_socket_ = false;
        }
        return *this;
    }

    /**
     * @brief Read next packet as validated view
     *
     * Blocks waiting for the next UDP datagram, validates it, and returns a type-safe variant
     * containing the appropriate packet view.
     *
     * @return ParseResult<PacketVariant> containing valid packet or error info,
     *         or std::nullopt on socket closure, timeout, or fatal error
     *
     * @note In blocking mode, this will wait indefinitely for data unless a timeout is set.
     * @note Datagram truncation (exceeding buffer size) returns ParseError with
     *       ValidationError::buffer_too_small. Check transport_status().actual_size for
     *       the required buffer size.
     * @note The returned view references the internal scratch buffer and is valid
     *       until the next read operation.
     */
    std::optional<vrtigo::ParseResult<vrtigo::PacketVariant>> read_next_packet() noexcept {
        auto bytes = read_next_datagram();

        if (bytes.empty()) {
            // Check for terminal conditions (socket closed or fatal error)
            if (status_.is_terminal()) {
                return std::nullopt;
            }

            // Non-terminal conditions: timeout, interrupted, truncation
            // For timeout/interrupted, just return nullopt and let caller retry
            if (status_.state == UDPTransportStatus::State::timeout ||
                status_.state == UDPTransportStatus::State::interrupted) {
                return std::nullopt;
            }

            // Datagram truncated - return ParseError so iteration continues
            if (status_.is_truncated()) {
                if (status_.bytes_received >= 4) {
                    // We have header - create proper ParseError
                    auto decoded = vrtigo::detail::decode_header(status_.header);
                    return vrtigo::ParseResult<vrtigo::PacketVariant>{vrtigo::ParseError{
                        ValidationError::buffer_too_small, status_.packet_type, decoded,
                        std::span<const uint8_t>() // No partial data
                    }};
                }
                // Truncated with no header - return generic ParseError
                // Use dummy header with packet size from actual_size
                vrtigo::detail::DecodedHeader dummy{};
                dummy.type = PacketType::signal_data_no_id;
                dummy.size_words =
                    static_cast<uint16_t>(std::min(status_.actual_size / 4, size_t(65535)));
                dummy.has_class_id = false;
                dummy.trailer_included = false;
                return vrtigo::ParseResult<vrtigo::PacketVariant>{vrtigo::ParseError{
                    ValidationError::buffer_too_small, PacketType::signal_data_no_id, dummy,
                    std::span<const uint8_t>()}};
            }

            // Should never reach here
            return std::nullopt;
        }

        // Validate minimum packet size
        if (bytes.size() < 4) {
            // Malformed datagram - return ParseError so iteration continues
            vrtigo::detail::DecodedHeader dummy{};
            dummy.type = PacketType::signal_data_no_id;
            dummy.size_words = static_cast<uint16_t>(bytes.size() / 4);
            dummy.has_class_id = false;
            dummy.trailer_included = false;
            return vrtigo::ParseResult<vrtigo::PacketVariant>{
                vrtigo::ParseError{ValidationError::buffer_too_small, PacketType::signal_data_no_id,
                                   dummy, std::span<const uint8_t>(bytes.data(), bytes.size())}};
        }

        // Parse and validate the packet
        return vrtigo::parse_packet(bytes);
    }

    /**
     * @brief Iterate over all packets with automatic validation
     *
     * Processes all packets received on the socket, automatically validating each one.
     * The callback receives a PacketVariant for each packet.
     *
     * This method blocks indefinitely waiting for packets. To stop iteration, return false
     * from the callback or close the socket from another thread.
     *
     * @tparam Callback Function type with signature: bool(const PacketVariant&)
     * @param callback Function called for each packet. Return false to stop iteration.
     * @return Number of packets processed
     */
    template <typename Callback>
    size_t for_each_validated_packet(Callback&& callback) noexcept {
        return detail::for_each_validated_packet(*this, std::forward<Callback>(callback));
    }

    /**
     * @brief Iterate over data packets only (signal/extension data)
     *
     * Processes only valid data packets (types 0-3), skipping context packets
     * and invalid packets. The callback receives a validated dynamic::DataPacket.
     *
     * @tparam Callback Function type with signature: bool(const vrtigo::dynamic::DataPacket&)
     * @param callback Function called for each data packet. Return false to stop.
     * @return Number of data packets processed
     */
    template <typename Callback>
    size_t for_each_data_packet(Callback&& callback) noexcept {
        return detail::for_each_data_packet(*this, std::forward<Callback>(callback));
    }

    /**
     * @brief Iterate over context packets only (context/extension context)
     *
     * Processes only valid context packets (types 4-5), skipping data packets
     * and invalid packets. The callback receives a validated dynamic::ContextPacket.
     *
     * @tparam Callback Function type with signature: bool(const vrtigo::dynamic::ContextPacket&)
     * @param callback Function called for each context packet. Return false to stop.
     * @return Number of context packets processed
     */
    template <typename Callback>
    size_t for_each_context_packet(Callback&& callback) noexcept {
        return detail::for_each_context_packet(*this, std::forward<Callback>(callback));
    }

    /**
     * @brief Iterate over packets with a specific stream ID
     *
     * Processes only packets that have a stream ID matching the given value.
     * Skips packets without stream IDs (types 0, 2) and invalid packets.
     *
     * @tparam Callback Function type with signature: bool(const PacketVariant&)
     * @param stream_id_filter The stream ID to filter by
     * @param callback Function called for each matching packet. Return false to stop.
     * @return Number of matching packets processed
     */
    template <typename Callback>
    size_t for_each_packet_with_stream_id(uint32_t stream_id_filter, Callback&& callback) noexcept {
        return detail::for_each_packet_with_stream_id(*this, stream_id_filter,
                                                      std::forward<Callback>(callback));
    }

    /**
     * @brief Get transport status from last receive operation
     *
     * Provides detailed information about the last datagram reception, including
     * error conditions and truncation details.
     *
     * When a datagram is truncated (exceeds buffer size), transport_status().actual_size
     * contains the full datagram size needed.
     *
     * @return Status of last receive operation
     */
    const UDPTransportStatus& transport_status() const noexcept { return status_; }

    /**
     * @brief Set receive timeout for blocking operations
     *
     * Sets SO_RCVTIMEO socket option to limit blocking time.
     * After timeout, recvmsg() returns EAGAIN and is treated as a socket error.
     *
     * @param timeout Timeout duration (zero = no timeout, infinite blocking)
     * @return true on success, false on failure
     */
    bool try_set_timeout(std::chrono::milliseconds timeout) noexcept {
        struct timeval tv {};
        tv.tv_sec = timeout.count() / 1000;
        tv.tv_usec = (timeout.count() % 1000) * 1000;

        return setsockopt(socket_, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv)) >= 0;
    }

    /**
     * @brief Set socket receive buffer size
     *
     * Sets SO_RCVBUF socket option to control kernel buffer size.
     * Larger buffers can help prevent packet loss under high load.
     *
     * @param bytes Requested buffer size in bytes
     * @return true on success, false on failure
     */
    bool try_set_receive_buffer_size(size_t bytes) noexcept {
        int size = static_cast<int>(bytes);
        return setsockopt(socket_, SOL_SOCKET, SO_RCVBUF, &size, sizeof(size)) >= 0;
    }

    /**
     * @brief Check if socket is still valid
     *
     * @return true if socket is open and not in error state
     */
    bool is_open() const noexcept { return socket_ >= 0 && !status_.is_terminal(); }

    /**
     * @brief Get underlying socket file descriptor
     *
     * Use this for advanced socket configuration or integration with event loops.
     *
     * @return Socket file descriptor
     */
    int socket_fd() const noexcept { return socket_; }

    /**
     * @brief Get the port the socket is bound to
     *
     * Useful when binding to port 0 (ephemeral port) to discover the assigned port.
     *
     * @return Port number, or 0 on error
     */
    uint16_t socket_port() const noexcept {
        struct sockaddr_in addr {};
        socklen_t addr_len = sizeof(addr);

        if (getsockname(socket_, reinterpret_cast<struct sockaddr*>(&addr), &addr_len) < 0) {
            return 0;
        }

        return ntohs(addr.sin_port);
    }

private:
    /**
     * @brief Read next UDP datagram into scratch buffer
     *
     * Blocks waiting for a datagram (in blocking mode). Detects truncation via MSG_TRUNC.
     * Updates status_ with result information.
     *
     * @return Span of datagram bytes, or empty span on error/truncation
     */
    std::span<const uint8_t> read_next_datagram() noexcept {
        // Clear previous state
        status_.header = 0;
        status_.packet_type = PacketType::signal_data_no_id;
        status_.bytes_received = 0;
        status_.actual_size = 0;
        status_.errno_value = 0;

        // Set up msghdr for recvmsg (to detect MSG_TRUNC)
        struct iovec iov {};
        iov.iov_base = scratch_buffer_.data();
        iov.iov_len = scratch_buffer_.size();

        struct msghdr msg {};
        msg.msg_iov = &iov;
        msg.msg_iovlen = 1;

        // Blocking receive with MSG_TRUNC to detect truncation
        // MSG_TRUNC makes recvmsg return the actual datagram size even if truncated
        ssize_t bytes;
        while (true) {
            bytes = recvmsg(socket_, &msg, MSG_TRUNC);

            if (bytes >= 0) {
                break; // Success
            }

            // Handle errors
            status_.errno_value = errno;

            // EINTR: interrupted by signal - retry immediately
            if (errno == EINTR) {
                continue;
            }

            // EAGAIN/EWOULDBLOCK: timeout or would block - non-terminal
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                status_.state = UDPTransportStatus::State::timeout;
                return {};
            }

            // All other errors are fatal
            status_.state = UDPTransportStatus::State::socket_error;
            return {};
        }

        if (bytes == 0) {
            // Socket closed (shouldn't happen with UDP, but handle it)
            status_.state = UDPTransportStatus::State::socket_closed;
            return {};
        }

        // Check for truncation
        if (msg.msg_flags & MSG_TRUNC) {
            status_.state = UDPTransportStatus::State::datagram_truncated;
            status_.actual_size = static_cast<size_t>(bytes);
            // bytes_received is what actually fit in the buffer
            status_.bytes_received = std::min(scratch_buffer_.size(), static_cast<size_t>(bytes));

            // Try to decode header if we received at least 4 bytes
            if (scratch_buffer_.size() >= 4) {
                uint32_t network_header;
                std::memcpy(&network_header, scratch_buffer_.data(), 4);
                status_.header = vrtigo::detail::network_to_host32(network_header);

                // Decode packet type from header
                auto decoded = vrtigo::detail::decode_header(status_.header);
                status_.packet_type = decoded.type;
            }

            return {}; // Don't return truncated data
        }

        // Successful receive
        status_.state = UDPTransportStatus::State::packet_ready;
        status_.bytes_received = static_cast<size_t>(bytes);

        // Decode header if we have at least 4 bytes
        if (bytes >= 4) {
            uint32_t network_header;
            std::memcpy(&network_header, scratch_buffer_.data(), 4);
            status_.header = vrtigo::detail::network_to_host32(network_header);

            // Decode packet type from header
            auto decoded = vrtigo::detail::decode_header(status_.header);
            status_.packet_type = decoded.type;
        }

        return std::span<const uint8_t>(scratch_buffer_.data(), static_cast<size_t>(bytes));
    }

    int socket_;       ///< UDP socket file descriptor
    bool owns_socket_; ///< Whether to close socket in destructor
    std::array<uint8_t, MaxPacketWords * 4> scratch_buffer_; ///< Internal datagram buffer
    UDPTransportStatus status_;                              ///< Status of last receive operation
};

} // namespace vrtigo::utils::netio
