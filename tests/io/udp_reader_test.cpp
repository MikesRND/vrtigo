#include <array>
#include <chrono>
#include <thread>

#include <arpa/inet.h>
#include <cstring>
#include <gtest/gtest.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vrtio/utils/netio/udp_vrt_reader.hpp>

#include "test_utils.hpp"

using namespace vrtio;
using namespace vrtio::utils::netio;
// PacketVariant and related types are now in vrtio namespace

// =============================================================================
// RAII Thread Guard
// =============================================================================

/**
 * @brief RAII guard for std::thread to ensure joining even on test failure
 */
class ThreadGuard {
public:
    explicit ThreadGuard(std::thread&& t) : thread_(std::move(t)) {}

    ~ThreadGuard() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    ThreadGuard(const ThreadGuard&) = delete;
    ThreadGuard& operator=(const ThreadGuard&) = delete;

    void join() {
        if (thread_.joinable()) {
            thread_.join();
        }
    }

private:
    std::thread thread_;
};

// =============================================================================
// Test Fixture
// =============================================================================

class UDPReaderTest : public ::testing::Test {
protected:
    int sender_socket_ = -1;

    void SetUp() override {
        // Create sender socket
        sender_socket_ = socket(AF_INET, SOCK_DGRAM, 0);
        ASSERT_GE(sender_socket_, 0) << "Failed to create sender socket: " << strerror(errno);
    }

    void TearDown() override {
        if (sender_socket_ >= 0) {
            close(sender_socket_);
        }
    }

    /**
     * @brief Send raw bytes via UDP to localhost:port
     */
    void send_udp_packet(const uint8_t* data, size_t len, uint16_t port) {
        struct sockaddr_in dest {};
        dest.sin_family = AF_INET;
        dest.sin_port = htons(port);
        dest.sin_addr.s_addr = inet_addr("127.0.0.1");

        ssize_t sent = sendto(sender_socket_, data, len, 0,
                              reinterpret_cast<struct sockaddr*>(&dest), sizeof(dest));

        ASSERT_EQ(sent, static_cast<ssize_t>(len)) << "Failed to send packet: " << strerror(errno);
    }

    /**
     * @brief Send a VRT packet vector via UDP
     */
    void send_vrt_packet(const std::vector<uint8_t>& packet, uint16_t port) {
        send_udp_packet(packet.data(), packet.size(), port);
    }
};

// =============================================================================
// Basic Sanity Tests
// =============================================================================

TEST_F(UDPReaderTest, ConstructReaderEphemeralPort) {
    EXPECT_NO_THROW({
        UDPVRTReader<> reader(uint16_t(0)); // Port 0 = kernel assigns ephemeral port
        EXPECT_TRUE(reader.is_open());

        uint16_t port = reader.socket_port();
        // Ephemeral ports are typically in the range 32768-60999
        // Just verify we got *some* port assigned
        EXPECT_GT(port, 0) << "Kernel should assign a non-zero ephemeral port";
    });
}

TEST_F(UDPReaderTest, ConstructReaderFixedPort) {
    uint16_t free_port;

    {
        // Bind to ephemeral port first to find a free port
        UDPVRTReader<> temp_reader(uint16_t(0));
        free_port = temp_reader.socket_port();
        ASSERT_GT(free_port, 0);
        // temp_reader destructor closes socket here
    }

    // Now create reader on that specific port
    // (This proves we can bind to a specific port, not just that the port works)
    EXPECT_NO_THROW({
        UDPVRTReader<> reader(free_port);
        EXPECT_TRUE(reader.is_open());
        EXPECT_EQ(reader.socket_port(), free_port);
    });
}

TEST_F(UDPReaderTest, ReceiveSinglePacket) {
    // Create reader on ephemeral port
    UDPVRTReader<> reader(uint16_t(0));
    reader.try_set_timeout(std::chrono::milliseconds(1000));

    uint16_t port = reader.socket_port();
    ASSERT_GT(port, 0) << "Should get ephemeral port from kernel";

    // Send a minimal VRT packet in background thread
    auto packet_data = test_utils::create_minimal_vrt_packet(0x12345678);

    ThreadGuard sender(std::thread([this, packet_data, port]() {
        // Small delay to ensure reader is blocking on recvmsg
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        send_vrt_packet(packet_data, port);
    }));

    // Receive the packet (blocking)
    auto pkt = reader.read_next_packet();

    // Wait for sender thread
    // ThreadGuard auto-joins in destructor

    // Verify we got a packet
    ASSERT_TRUE(pkt.has_value()) << "Should receive a packet";
    ASSERT_TRUE(is_data_packet(*pkt)) << "Should be a data packet";

    // Verify packet contents
    const auto& view = std::get<RuntimeDataPacket>(*pkt);
    EXPECT_TRUE(view.is_valid());
    EXPECT_EQ(view.type(), PacketType::signal_data);
    EXPECT_TRUE(view.has_stream_id());

    auto stream_id = view.stream_id();
    ASSERT_TRUE(stream_id.has_value());
    EXPECT_EQ(*stream_id, 0x12345678);

    // Check payload size and content
    auto payload = view.payload();
    EXPECT_EQ(payload.size(), 4); // 1 word = 4 bytes

    // Verify payload content (0xDEADBEEF in network byte order)
    EXPECT_EQ(payload[0], 0xDE);
    EXPECT_EQ(payload[1], 0xAD);
    EXPECT_EQ(payload[2], 0xBE);
    EXPECT_EQ(payload[3], 0xEF);
}

TEST_F(UDPReaderTest, ReceiveMultiplePackets) {
    UDPVRTReader<> reader(uint16_t(0));
    reader.try_set_timeout(std::chrono::milliseconds(1000));

    uint16_t port = reader.socket_port();
    ASSERT_GT(port, 0) << "Should get ephemeral port from kernel";

    const size_t NUM_PACKETS = 5;

    // Send multiple packets in background
    ThreadGuard sender(std::thread([this, port]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        for (size_t i = 0; i < NUM_PACKETS; ++i) {
            auto packet = test_utils::create_minimal_vrt_packet(0x1000 + i);
            send_vrt_packet(packet, port);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }));

    // Receive packets
    size_t received = 0;
    std::vector<uint32_t> stream_ids;

    for (size_t i = 0; i < NUM_PACKETS; ++i) {
        auto pkt = reader.read_next_packet();
        if (!pkt)
            break;

        if (is_data_packet(*pkt)) {
            const auto& view = std::get<RuntimeDataPacket>(*pkt);
            auto sid = view.stream_id();
            EXPECT_TRUE(sid.has_value());

            if (sid) {
                stream_ids.push_back(*sid);
            }

            received++;
        }
    }

    // ThreadGuard auto-joins in destructor

    EXPECT_EQ(received, NUM_PACKETS) << "Should receive all packets";

    // Verify stream IDs are in expected range
    for (auto sid : stream_ids) {
        EXPECT_GE(sid, 0x1000);
        EXPECT_LT(sid, 0x1000 + NUM_PACKETS);
    }
}

TEST_F(UDPReaderTest, TimeoutWhenNoData) {
    UDPVRTReader<> reader(uint16_t(0)); // Ephemeral port
    reader.try_set_timeout(std::chrono::milliseconds(200));

    auto start = std::chrono::steady_clock::now();
    auto pkt = reader.read_next_packet();
    auto elapsed = std::chrono::steady_clock::now() - start;

    // Should timeout (return nullopt)
    EXPECT_FALSE(pkt.has_value()) << "Should timeout with no data";

    // Should have waited approximately the timeout duration
    EXPECT_GE(elapsed, std::chrono::milliseconds(150));
    EXPECT_LE(elapsed, std::chrono::milliseconds(300));
}

TEST_F(UDPReaderTest, IterationHelper) {
    UDPVRTReader<> reader(uint16_t(0));
    reader.try_set_timeout(std::chrono::milliseconds(500));

    uint16_t port = reader.socket_port();
    ASSERT_GT(port, 0) << "Should get ephemeral port from kernel";

    const size_t NUM_PACKETS = 3;

    // Send packets in background
    ThreadGuard sender(std::thread([this, port]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        for (size_t i = 0; i < NUM_PACKETS; ++i) {
            auto packet = test_utils::create_minimal_vrt_packet(0x2000 + i);
            send_vrt_packet(packet, port);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }));

    // Use for_each_data_packet helper
    size_t count = 0;
    reader.for_each_data_packet([&count](const RuntimeDataPacket& view) {
        EXPECT_TRUE(view.is_valid());
        EXPECT_TRUE(view.has_stream_id());
        count++;
        return count < NUM_PACKETS; // Stop after NUM_PACKETS
    });

    // ThreadGuard auto-joins in destructor

    EXPECT_EQ(count, NUM_PACKETS) << "Should iterate over all packets";
}

TEST_F(UDPReaderTest, TransportStatus) {
    UDPVRTReader<> reader(uint16_t(0));
    reader.try_set_timeout(std::chrono::milliseconds(500));

    uint16_t port = reader.socket_port();
    ASSERT_GT(port, 0) << "Should get ephemeral port from kernel";

    // Send a packet
    auto packet_data = test_utils::create_minimal_vrt_packet();

    ThreadGuard sender(std::thread([this, packet_data, port]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        send_vrt_packet(packet_data, port);
    }));

    auto pkt = reader.read_next_packet();

    // ThreadGuard auto-joins in destructor

    const auto& status = reader.transport_status();

    if (pkt.has_value()) {
        EXPECT_EQ(status.state, UDPTransportStatus::State::packet_ready);
        EXPECT_EQ(status.bytes_received, 12); // 3 words = 12 bytes
        EXPECT_FALSE(status.is_truncated());
        EXPECT_FALSE(status.is_terminal());
    }
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(UDPReaderTest, TruncatedDatagram) {
    // Create reader with very small buffer (only 2 words)
    UDPVRTReader<2> reader(uint16_t(0));
    reader.try_set_timeout(std::chrono::milliseconds(500));

    uint16_t port = reader.socket_port();
    ASSERT_GT(port, 0) << "Should get ephemeral port from kernel";

    // Send a 3-word packet (will be truncated)
    auto packet_data = test_utils::create_minimal_vrt_packet();

    ThreadGuard sender(std::thread([this, packet_data, port]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        send_vrt_packet(packet_data, port);
    }));

    auto pkt = reader.read_next_packet();

    // ThreadGuard auto-joins in destructor

    const auto& status = reader.transport_status();

    // Should detect truncation
    EXPECT_TRUE(status.is_truncated()) << "Should detect datagram truncation";

    if (status.is_truncated()) {
        EXPECT_GT(status.actual_size, status.bytes_received);
        EXPECT_EQ(status.actual_size, 12);   // Original 3-word packet
        EXPECT_EQ(status.bytes_received, 8); // Only 2 words received
    }

    // Should get InvalidPacket or nullopt
    if (pkt.has_value()) {
        if (!is_valid(*pkt)) {
            const auto& invalid = std::get<InvalidPacket>(*pkt);
            EXPECT_EQ(invalid.error, ValidationError::buffer_too_small);
        }
    }
}

TEST_F(UDPReaderTest, LargePayload) {
    UDPVRTReader<> reader(uint16_t(0));
    reader.try_set_timeout(std::chrono::milliseconds(500));

    uint16_t port = reader.socket_port();
    ASSERT_GT(port, 0) << "Should get ephemeral port from kernel";

    // Create packet with 100 payload words
    const uint16_t PAYLOAD_WORDS = 100;
    auto packet_data = test_utils::create_vrt_packet_with_payload(0x99999999, PAYLOAD_WORDS);

    ThreadGuard sender(std::thread([this, packet_data, port]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        send_vrt_packet(packet_data, port);
    }));

    auto pkt = reader.read_next_packet();

    // ThreadGuard auto-joins in destructor

    ASSERT_TRUE(pkt.has_value()) << "Should receive large packet";
    ASSERT_TRUE(is_data_packet(*pkt));

    const auto& view = std::get<RuntimeDataPacket>(*pkt);
    EXPECT_TRUE(view.is_valid());

    auto payload = view.payload();
    EXPECT_EQ(payload.size(), PAYLOAD_WORDS * 4); // 100 words = 400 bytes

    // Verify stream ID
    auto sid = view.stream_id();
    ASSERT_TRUE(sid.has_value());
    EXPECT_EQ(*sid, 0x99999999);
}

TEST_F(UDPReaderTest, InvalidHeaderSize) {
    UDPVRTReader<> reader(uint16_t(0));
    reader.try_set_timeout(std::chrono::milliseconds(500));

    uint16_t port = reader.socket_port();
    ASSERT_GT(port, 0) << "Should get ephemeral port from kernel";

    // Send invalid packet (header says size 10, but actual size is 3 words)
    std::vector<uint8_t> bad_packet(12);
    uint32_t bad_header = 0x1000000A; // Size = 10 words (but we only send 3)

    bad_packet[0] = (bad_header >> 24) & 0xFF;
    bad_packet[1] = (bad_header >> 16) & 0xFF;
    bad_packet[2] = (bad_header >> 8) & 0xFF;
    bad_packet[3] = bad_header & 0xFF;

    // Fill rest with dummy data
    for (size_t i = 4; i < 12; ++i) {
        bad_packet[i] = 0xAA;
    }

    ThreadGuard sender(std::thread([this, bad_packet, port]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        send_vrt_packet(bad_packet, port);
    }));

    auto pkt = reader.read_next_packet();

    // ThreadGuard auto-joins in destructor

    // Should get InvalidPacket due to size mismatch
    if (pkt.has_value() && !is_valid(*pkt)) {
        const auto& invalid = std::get<InvalidPacket>(*pkt);
        EXPECT_NE(invalid.error, ValidationError::none);
    }
}

TEST_F(UDPReaderTest, TimeoutIsNonTerminal) {
    UDPVRTReader<> reader(uint16_t(0));
    reader.try_set_timeout(std::chrono::milliseconds(100));

    uint16_t port = reader.socket_port();
    ASSERT_GT(port, 0);

    // First call: timeout (no data)
    auto pkt1 = reader.read_next_packet();
    EXPECT_FALSE(pkt1.has_value()) << "Should timeout with no data";
    EXPECT_TRUE(reader.is_open()) << "Reader should still be open after timeout";

    // Second call: send data and verify it can still receive
    auto packet_data = test_utils::create_minimal_vrt_packet();

    ThreadGuard sender(std::thread([this, packet_data, port]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        send_vrt_packet(packet_data, port);
    }));

    auto pkt2 = reader.read_next_packet();
    ASSERT_TRUE(pkt2.has_value()) << "Should receive packet after timeout";
    EXPECT_TRUE(is_valid(*pkt2)) << "Packet should be valid";
}
