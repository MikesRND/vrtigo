#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <vrtigo/detail/endian.hpp>

// Test byte swapping functions
TEST(EndianTest, ByteSwap16) {
    uint16_t value = 0x1234;
    uint16_t swapped = vrtigo::detail::byteswap16(value);
    EXPECT_EQ(swapped, 0x3412);

    // Double swap should return original
    EXPECT_EQ(vrtigo::detail::byteswap16(swapped), value);
}

TEST(EndianTest, ByteSwap32) {
    uint32_t value = 0x12345678;
    uint32_t swapped = vrtigo::detail::byteswap32(value);
    EXPECT_EQ(swapped, 0x78563412);

    // Double swap should return original
    EXPECT_EQ(vrtigo::detail::byteswap32(swapped), value);
}

TEST(EndianTest, ByteSwap64) {
    uint64_t value = 0x123456789ABCDEF0ULL;
    uint64_t swapped = vrtigo::detail::byteswap64(value);
    EXPECT_EQ(swapped, 0xF0DEBC9A78563412ULL);

    // Double swap should return original
    EXPECT_EQ(vrtigo::detail::byteswap64(swapped), value);
}

// Test network byte order conversions
TEST(EndianTest, HostToNetwork32) {
    uint32_t value = 0xDEADBEEF;
    uint32_t network = vrtigo::detail::host_to_network32(value);
    uint32_t host = vrtigo::detail::network_to_host32(network);

    // Round-trip should preserve value
    EXPECT_EQ(host, value);
}

TEST(EndianTest, HostToNetwork64) {
    uint64_t value = 0xCAFEBABEDEADBEEFULL;
    uint64_t network = vrtigo::detail::host_to_network64(value);
    uint64_t host = vrtigo::detail::network_to_host64(network);

    // Round-trip should preserve value
    EXPECT_EQ(host, value);
}

// Test that network byte order is big-endian
TEST(EndianTest, NetworkIsBigEndian) {
    uint32_t value = 0x12345678;
    uint32_t network = vrtigo::detail::host_to_network32(value);

    // Convert to byte array
    uint8_t bytes[4];
    std::memcpy(bytes, &network, 4);

    if (vrtigo::detail::is_little_endian) {
        // On little-endian, bytes should be swapped to big-endian
        EXPECT_EQ(bytes[0], 0x12); // MSB
        EXPECT_EQ(bytes[1], 0x34);
        EXPECT_EQ(bytes[2], 0x56);
        EXPECT_EQ(bytes[3], 0x78); // LSB
    } else {
        // On big-endian, bytes should remain unchanged
        EXPECT_EQ(bytes[0], 0x12);
        EXPECT_EQ(bytes[1], 0x34);
        EXPECT_EQ(bytes[2], 0x56);
        EXPECT_EQ(bytes[3], 0x78);
    }
}

// Test constexpr nature of endian functions
TEST(EndianTest, ConstexprFunctions) {
    // These should compile as constexpr
    constexpr uint32_t swapped = vrtigo::detail::byteswap32(0x12345678);
    constexpr uint32_t network = vrtigo::detail::host_to_network32(0xDEADBEEF);

    EXPECT_EQ(swapped, 0x78563412);
    // network value depends on platform endianness, just verify it compiles
    (void)network;
}

// Test platform detection
TEST(EndianTest, PlatformDetection) {
    // At least one should be true
    EXPECT_TRUE(vrtigo::detail::is_little_endian || vrtigo::detail::is_big_endian);

    // Both cannot be true
    EXPECT_FALSE(vrtigo::detail::is_little_endian && vrtigo::detail::is_big_endian);

// Most modern systems are little-endian
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    EXPECT_TRUE(vrtigo::detail::is_little_endian);
#endif
}

// Test edge cases
TEST(EndianTest, EdgeCases) {
    // Zero should swap to zero
    EXPECT_EQ(vrtigo::detail::byteswap16(0), 0);
    EXPECT_EQ(vrtigo::detail::byteswap32(0), 0);
    EXPECT_EQ(vrtigo::detail::byteswap64(0), 0);

    // Max values
    EXPECT_EQ(vrtigo::detail::byteswap16(0xFFFF), 0xFFFF);
    EXPECT_EQ(vrtigo::detail::byteswap32(0xFFFFFFFF), 0xFFFFFFFF);
    EXPECT_EQ(vrtigo::detail::byteswap64(0xFFFFFFFFFFFFFFFFULL), 0xFFFFFFFFFFFFFFFFULL);

    // Powers of 2
    EXPECT_EQ(vrtigo::detail::byteswap32(0x00000001), 0x01000000);
    EXPECT_EQ(vrtigo::detail::byteswap32(0x80000000), 0x00000080);
}

// Test real-world VRT packet field conversion
TEST(EndianTest, VRTFieldConversion) {
    // Stream ID example
    uint32_t stream_id = 0x12345678;
    uint32_t network_stream = vrtigo::detail::host_to_network32(stream_id);

    // Simulate writing to buffer and reading back
    uint8_t buffer[4]{};
    std::memcpy(buffer, &network_stream, 4);

    uint32_t read_network;
    std::memcpy(&read_network, buffer, 4);
    uint32_t recovered = vrtigo::detail::network_to_host32(read_network);

    EXPECT_EQ(recovered, stream_id);

    // Verify bytes are in network order (big-endian)
    EXPECT_EQ(buffer[0], 0x12);
    EXPECT_EQ(buffer[1], 0x34);
    EXPECT_EQ(buffer[2], 0x56);
    EXPECT_EQ(buffer[3], 0x78);
}

// Test 64-bit timestamp conversion
TEST(EndianTest, TimestampConversion) {
    // Picosecond timestamp
    uint64_t timestamp = 999999999999ULL; // Max picoseconds in a second
    uint64_t network_ts = vrtigo::detail::host_to_network64(timestamp);

    // Simulate write/read
    uint8_t buffer[8]{};
    std::memcpy(buffer, &network_ts, 8);

    uint64_t read_network;
    std::memcpy(&read_network, buffer, 8);
    uint64_t recovered = vrtigo::detail::network_to_host64(read_network);

    EXPECT_EQ(recovered, timestamp);

    // Verify bytes are in network order (big-endian)
    // timestamp = 999999999999 = 0x00 00 00 E8 D4 A5 0F FF
    EXPECT_EQ(buffer[0], 0x00);
    EXPECT_EQ(buffer[1], 0x00);
    EXPECT_EQ(buffer[2], 0x00);
    EXPECT_EQ(buffer[3], 0xE8);
    EXPECT_EQ(buffer[4], 0xD4);
    EXPECT_EQ(buffer[5], 0xA5);
    EXPECT_EQ(buffer[6], 0x0F);
    EXPECT_EQ(buffer[7], 0xFF);
}
