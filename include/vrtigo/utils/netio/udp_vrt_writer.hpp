// Copyright (c) 2025 Michael Smith
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unistd.h>

// Linux/POSIX socket headers
#include "vrtigo/detail/packet_variant.hpp"
#include "vrtigo/utils/netio/udp_transport_status.hpp"

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>

namespace vrtigo::utils::netio {

/**
 * @brief UDP VRT packet writer (Linux/POSIX)
 *
 * Writes VRT packets as UDP datagrams with automatic MTU validation.
 * Each packet is sent as a complete datagram (no fragmentation).
 *
 * Connection Modes:
 * - Bound mode: Connect to single endpoint, use send()
 * - Unbound mode: Specify destination per packet with sendto()
 *
 * Supported Packet Types:
 * - Raw bytes (std::span<const uint8_t>)
 * - PacketVariant (runtime packets)
 *
 * MTU Enforcement:
 * - Default MTU: 1500 bytes
 * - Packets exceeding MTU are rejected (no fragmentation)
 * - Configurable via set_mtu()
 * - Returns false for oversized packets
 *
 * Blocking Mode:
 * - Always uses blocking sockets (consistent with UDPVRTReader)
 * - SO_SNDTIMEO can be set for timeout support
 *
 * Thread Safety:
 * - Not thread-safe: single thread should own this instance
 * - Safe to move between threads (move-only)
 *
 * Example usage:
 * @code
 * // Bound mode - single destination
 * UDPVRTWriter writer("192.168.1.100", 12345);
 *
 * // Create packet using typed::SignalDataPacketBuilder
 * using PacketType = vrtigo::typed::SignalDataPacketBuilder<64>;
 * alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};
 * PacketType packet(buffer);
 * packet.set_stream_id(0x1234);
 * packet.set_packet_count(1);
 *
 * // Option 1: Write raw bytes
 * writer.write_packet(packet.as_bytes());
 *
 * // Option 2: Convert to variant
 * auto result = vrtigo::dynamic::DataPacketView::parse(packet.as_bytes());
 * if (result) {
 *     vrtigo::PacketVariant variant = result.value();
 *     writer.write_packet(variant);
 * }
 *
 * // Unbound mode - per-packet destination
 * UDPVRTWriter multi_writer(0);  // bind to any local port
 *
 * sockaddr_in dest1 {};
 * dest1.sin_family = AF_INET;
 * dest1.sin_port = htons(12345);
 * inet_pton(AF_INET, "192.168.1.100", &dest1.sin_addr);
 *
 * sockaddr_in dest2 {};
 * dest2.sin_family = AF_INET;
 * dest2.sin_port = htons(12345);
 * inet_pton(AF_INET, "192.168.1.101", &dest2.sin_addr);
 *
 * // Write to multiple destinations using raw bytes
 * auto bytes = packet.as_bytes();
 * multi_writer.write_packet(bytes, dest1);
 * multi_writer.write_packet(bytes, dest2);
 * @endcode
 */
class UDPVRTWriter {
public:
    static constexpr size_t default_mtu = 1500; ///< Default MTU in bytes

    /**
     * @brief Create writer in bound mode (single destination)
     *
     * Connects to a single UDP endpoint. All packets are sent to this destination.
     *
     * @param host Destination hostname or IP address
     * @param port Destination UDP port
     * @throws std::runtime_error if socket creation or DNS resolution fails
     */
    explicit UDPVRTWriter(const std::string& host, uint16_t port)
        : socket_(-1),
          bound_mode_(true),
          mtu_(default_mtu),
          packets_sent_(0),
          bytes_sent_(0) {
        // Create UDP socket
        socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_ < 0) {
            throw std::runtime_error("Failed to create UDP socket");
        }

        // Resolve destination address
        struct sockaddr_in addr {};
        if (!resolve_address(host, port, addr)) {
            ::close(socket_);
            throw std::runtime_error("Failed to resolve address: " + host);
        }

        // Connect socket to destination (bound mode)
        if (::connect(socket_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
            ::close(socket_);
            throw std::runtime_error("Failed to connect UDP socket to " + host + ":" +
                                     std::to_string(port));
        }

        dest_addr_ = addr;
        status_.state = UDPTransportStatus::State::packet_ready;
    }

    /**
     * @brief Create writer in unbound mode (per-packet destination)
     *
     * Binds to a local port but does not connect to a destination.
     * Caller must specify destination for each write_packet() call.
     *
     * @param local_port Local port to bind (0 = any port)
     * @throws std::runtime_error if socket creation or binding fails
     */
    explicit UDPVRTWriter(uint16_t local_port = 0)
        : socket_(-1),
          bound_mode_(false),
          mtu_(default_mtu),
          packets_sent_(0),
          bytes_sent_(0) {
        // Create UDP socket
        socket_ = ::socket(AF_INET, SOCK_DGRAM, 0);
        if (socket_ < 0) {
            throw std::runtime_error("Failed to create UDP socket");
        }

        // Bind to local port
        struct sockaddr_in addr {};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(local_port);
        addr.sin_addr.s_addr = INADDR_ANY;

        if (::bind(socket_, reinterpret_cast<struct sockaddr*>(&addr), sizeof(addr)) < 0) {
            ::close(socket_);
            throw std::runtime_error("Failed to bind UDP socket to port " +
                                     std::to_string(local_port));
        }

        status_.state = UDPTransportStatus::State::packet_ready;
    }

    /**
     * @brief Destructor closes socket
     */
    ~UDPVRTWriter() {
        if (socket_ >= 0) {
            ::close(socket_);
        }
    }

    // Move-only (socket ownership)
    UDPVRTWriter(const UDPVRTWriter&) = delete;
    UDPVRTWriter& operator=(const UDPVRTWriter&) = delete;

    UDPVRTWriter(UDPVRTWriter&& other) noexcept
        : socket_(other.socket_),
          bound_mode_(other.bound_mode_),
          dest_addr_(other.dest_addr_),
          mtu_(other.mtu_),
          packets_sent_(other.packets_sent_),
          bytes_sent_(other.bytes_sent_),
          status_(other.status_) {
        other.socket_ = -1;
        other.packets_sent_ = 0;
        other.bytes_sent_ = 0;
    }

    UDPVRTWriter& operator=(UDPVRTWriter&& other) noexcept {
        if (this != &other) {
            // Clean up existing state
            if (socket_ >= 0) {
                ::close(socket_);
            }

            // Move from other
            socket_ = other.socket_;
            bound_mode_ = other.bound_mode_;
            dest_addr_ = other.dest_addr_;
            mtu_ = other.mtu_;
            packets_sent_ = other.packets_sent_;
            bytes_sent_ = other.bytes_sent_;
            status_ = other.status_;

            // Reset other
            other.socket_ = -1;
            other.packets_sent_ = 0;
            other.bytes_sent_ = 0;
        }
        return *this;
    }

    /**
     * @brief Write raw packet bytes (bound mode)
     *
     * Sends packet to connected destination. Only valid in bound mode.
     *
     * @param bytes The packet bytes to write
     * @return true on success, false on I/O error
     * @note The span contents are copied; caller's buffer can be reused immediately after return.
     */
    bool write_packet(std::span<const uint8_t> bytes) noexcept {
        if (!bound_mode_) {
            // Bound mode required for this method
            status_.state = UDPTransportStatus::State::socket_error;
            status_.errno_value = ENOTCONN;
            return false;
        }

        // Check MTU
        if (bytes.size() > mtu_) {
            status_.state = UDPTransportStatus::State::socket_error;
            status_.errno_value = EMSGSIZE;
            return false;
        }

        // Send datagram
        ssize_t sent = ::send(socket_, bytes.data(), bytes.size(), 0);
        if (sent < 0) {
            status_.state = map_errno_to_state(errno);
            status_.errno_value = errno;
            return false;
        }

        if (static_cast<size_t>(sent) != bytes.size()) {
            // Partial send (should not happen with UDP)
            status_.state = UDPTransportStatus::State::socket_error;
            status_.errno_value = EIO;
            return false;
        }

        packets_sent_++;
        bytes_sent_ += bytes.size();
        status_.state = UDPTransportStatus::State::packet_ready;
        return true;
    }

    /**
     * @brief Write raw packet bytes to specific destination (unbound mode)
     *
     * Sends packet to specified destination. Can be used in both bound
     * and unbound modes, but typically used in unbound mode for
     * per-packet destination control.
     *
     * @param bytes The packet bytes to write
     * @param dest Destination address
     * @return true on success, false on I/O error
     * @note The span contents are copied; caller's buffer can be reused immediately after return.
     */
    bool write_packet(std::span<const uint8_t> bytes, const struct sockaddr_in& dest) noexcept {
        // Check MTU
        if (bytes.size() > mtu_) {
            status_.state = UDPTransportStatus::State::socket_error;
            status_.errno_value = EMSGSIZE;
            return false;
        }

        // Send datagram
        ssize_t sent = ::sendto(socket_, bytes.data(), bytes.size(), 0,
                                reinterpret_cast<const struct sockaddr*>(&dest), sizeof(dest));
        if (sent < 0) {
            status_.state = map_errno_to_state(errno);
            status_.errno_value = errno;
            return false;
        }

        if (static_cast<size_t>(sent) != bytes.size()) {
            // Partial send (should not happen with UDP)
            status_.state = UDPTransportStatus::State::socket_error;
            status_.errno_value = EIO;
            return false;
        }

        packets_sent_++;
        bytes_sent_ += bytes.size();
        status_.state = UDPTransportStatus::State::packet_ready;
        return true;
    }

    /**
     * @brief Write packet from variant (bound mode)
     *
     * Sends packet to connected destination. Only valid in bound mode.
     *
     * @param packet The packet variant to write (always valid)
     * @return true on success, false on I/O error
     */
    bool write_packet(const vrtigo::PacketVariant& packet) noexcept {
        return vrtigo::detail::visit_packet_bytes(
            packet, [this](std::span<const uint8_t> bytes) { return this->write_packet(bytes); });
    }

    /**
     * @brief Write packet to specific destination (unbound mode)
     *
     * Sends packet to specified destination. Can be used in both bound
     * and unbound modes, but typically used in unbound mode for
     * per-packet destination control.
     *
     * @param packet The packet variant to write (always valid)
     * @param dest Destination address
     * @return true on success, false on I/O error
     */
    bool write_packet(const vrtigo::PacketVariant& packet,
                      const struct sockaddr_in& dest) noexcept {
        return vrtigo::detail::visit_packet_bytes(packet,
                                                  [this, &dest](std::span<const uint8_t> bytes) {
                                                      return this->write_packet(bytes, dest);
                                                  });
    }

    /**
     * @brief Set maximum transmission unit
     *
     * Packets larger than MTU will be rejected. Default is 1500 bytes.
     *
     * @param mtu Maximum packet size in bytes
     */
    void set_mtu(size_t mtu) noexcept { mtu_ = mtu; }

    /**
     * @brief Set send timeout
     *
     * Sets SO_SNDTIMEO socket option. By default, send operations block
     * indefinitely. This allows timeout on slow/blocked sends.
     *
     * @param milliseconds Timeout in milliseconds (0 = infinite)
     */
    void set_send_timeout(int milliseconds) noexcept {
        struct timeval tv {};
        tv.tv_sec = milliseconds / 1000;
        tv.tv_usec = (milliseconds % 1000) * 1000;

        ::setsockopt(socket_, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    }

    /**
     * @brief Get number of packets sent
     *
     * @return Total packets sent successfully
     */
    [[nodiscard]] size_t packets_sent() const noexcept { return packets_sent_; }

    /**
     * @brief Get number of bytes sent
     *
     * @return Total bytes sent (UDP payload only)
     */
    [[nodiscard]] size_t bytes_sent() const noexcept { return bytes_sent_; }

    /**
     * @brief Get transport status
     *
     * @return Current UDP transport status
     */
    [[nodiscard]] const UDPTransportStatus& transport_status() const noexcept { return status_; }

    /**
     * @brief Flush operation (no-op for UDP)
     *
     * UDP datagrams are sent immediately, no buffering occurs.
     * This method exists for concept compatibility but always returns true.
     *
     * @return Always true
     */
    bool flush() noexcept {
        return true; // No buffering in UDP
    }

private:
    /**
     * @brief Resolve hostname to sockaddr_in
     *
     * @param host Hostname or IP address
     * @param port Port number
     * @param out Output address structure
     * @return true on success, false on failure
     */
    static bool resolve_address(const std::string& host, uint16_t port,
                                struct sockaddr_in& out) noexcept {
        struct addrinfo hints {};
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_DGRAM;

        struct addrinfo* result = nullptr;
        int ret = ::getaddrinfo(host.c_str(), nullptr, &hints, &result);
        if (ret != 0 || result == nullptr) {
            return false;
        }

        // Use first result
        std::memcpy(&out, result->ai_addr, sizeof(struct sockaddr_in));
        out.sin_port = htons(port);

        ::freeaddrinfo(result);
        return true;
    }

    /**
     * @brief Map errno to UDPTransportStatus::State
     *
     * @param err errno value
     * @return Corresponding transport state
     */
    static UDPTransportStatus::State map_errno_to_state(int err) noexcept {
        switch (err) {
            case EAGAIN:
#if EAGAIN != EWOULDBLOCK
            case EWOULDBLOCK:
#endif
                return UDPTransportStatus::State::timeout;
            case EINTR:
                return UDPTransportStatus::State::interrupted;
            case EMSGSIZE:
            case ENETUNREACH:
            case EHOSTUNREACH:
            case ECONNREFUSED:
                return UDPTransportStatus::State::socket_error;
            default:
                return UDPTransportStatus::State::socket_error;
        }
    }

    int socket_;                   ///< Socket file descriptor
    bool bound_mode_;              ///< True if connected to single destination
    struct sockaddr_in dest_addr_; ///< Destination address (bound mode)
    size_t mtu_;                   ///< Maximum transmission unit
    size_t packets_sent_;          ///< Total packets sent
    size_t bytes_sent_;            ///< Total bytes sent
    UDPTransportStatus status_;    ///< Transport status
};

} // namespace vrtigo::utils::netio
