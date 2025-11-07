#include <vrtio/packet/signal_packet.hpp>
#include <vrtio/packet/builder.hpp>
#include <vrtio/core/trailer.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <cstring>

// Test fixture for trailer field manipulation
class TrailerFieldTest : public ::testing::Test {
protected:
    using PacketType = vrtio::SignalPacket<
        vrtio::packet_type::signal_data_with_stream,
        vrtio::TimeStampUTC,
        true,   // HasTrailer
        128
    >;

    std::vector<uint8_t> buffer;

    void SetUp() override {
        buffer.resize(PacketType::size_bytes);
        PacketType packet(buffer.data());
    }
};

// Test individual bit getters and setters
TEST_F(TrailerFieldTest, ValidDataBit) {
    PacketType packet(buffer.data());

    // Initially should be false (trailer is zero-initialized)
    EXPECT_FALSE(packet.trailer_valid_data());

    // Set to true
    packet.set_trailer_valid_data(true);
    EXPECT_TRUE(packet.trailer_valid_data());
    EXPECT_EQ(packet.trailer(), 1U << 17);

    // Set to false
    packet.set_trailer_valid_data(false);
    EXPECT_FALSE(packet.trailer_valid_data());
    EXPECT_EQ(packet.trailer(), 0U);
}

TEST_F(TrailerFieldTest, CalibratedTimeBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer_calibrated_time());

    packet.set_trailer_calibrated_time(true);
    EXPECT_TRUE(packet.trailer_calibrated_time());
    EXPECT_EQ(packet.trailer(), 1U << 16);

    packet.set_trailer_calibrated_time(false);
    EXPECT_FALSE(packet.trailer_calibrated_time());
}

TEST_F(TrailerFieldTest, OverRangeBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer_over_range());

    packet.set_trailer_over_range(true);
    EXPECT_TRUE(packet.trailer_over_range());
    EXPECT_EQ(packet.trailer(), 1U << 12);

    packet.set_trailer_over_range(false);
    EXPECT_FALSE(packet.trailer_over_range());
}

TEST_F(TrailerFieldTest, SampleLossBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer_sample_loss());

    packet.set_trailer_sample_loss(true);
    EXPECT_TRUE(packet.trailer_sample_loss());
    EXPECT_EQ(packet.trailer(), 1U << 13);
}

TEST_F(TrailerFieldTest, ReferenceLockBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer_reference_lock());

    packet.set_trailer_reference_lock(true);
    EXPECT_TRUE(packet.trailer_reference_lock());
    EXPECT_EQ(packet.trailer(), 1U << 8);
}

TEST_F(TrailerFieldTest, AgcMgcBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer_agc_mgc());

    packet.set_trailer_agc_mgc(true);
    EXPECT_TRUE(packet.trailer_agc_mgc());
    EXPECT_EQ(packet.trailer(), 1U << 9);
}

TEST_F(TrailerFieldTest, DetectedSignalBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer_detected_signal());

    packet.set_trailer_detected_signal(true);
    EXPECT_TRUE(packet.trailer_detected_signal());
    EXPECT_EQ(packet.trailer(), 1U << 10);
}

TEST_F(TrailerFieldTest, SpectralInversionBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer_spectral_inversion());

    packet.set_trailer_spectral_inversion(true);
    EXPECT_TRUE(packet.trailer_spectral_inversion());
    EXPECT_EQ(packet.trailer(), 1U << 11);
}

TEST_F(TrailerFieldTest, ReferencePointBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer_reference_point());

    packet.set_trailer_reference_point(true);
    EXPECT_TRUE(packet.trailer_reference_point());
    EXPECT_EQ(packet.trailer(), 1U << 18);
}

TEST_F(TrailerFieldTest, SignalDetectedBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer_signal_detected());

    packet.set_trailer_signal_detected(true);
    EXPECT_TRUE(packet.trailer_signal_detected());
    EXPECT_EQ(packet.trailer(), 1U << 19);
}

TEST_F(TrailerFieldTest, ContextPacketsField) {
    PacketType packet(buffer.data());

    EXPECT_EQ(packet.trailer_context_packets(), 0);

    // Set to various values
    packet.set_trailer_context_packets(5);
    EXPECT_EQ(packet.trailer_context_packets(), 5);
    EXPECT_EQ(packet.trailer(), 5U);

    packet.set_trailer_context_packets(127);
    EXPECT_EQ(packet.trailer_context_packets(), 127);
    EXPECT_EQ(packet.trailer(), 127U);

    packet.set_trailer_context_packets(0);
    EXPECT_EQ(packet.trailer_context_packets(), 0);
}

// Test multiple bits set simultaneously
TEST_F(TrailerFieldTest, MultipleBitsSet) {
    PacketType packet(buffer.data());

    packet.set_trailer_valid_data(true);
    packet.set_trailer_calibrated_time(true);

    EXPECT_TRUE(packet.trailer_valid_data());
    EXPECT_TRUE(packet.trailer_calibrated_time());

    uint32_t expected = (1U << 17) | (1U << 16);
    EXPECT_EQ(packet.trailer(), expected);
}

TEST_F(TrailerFieldTest, ErrorFlags) {
    PacketType packet(buffer.data());

    // No errors initially
    EXPECT_FALSE(packet.trailer_has_errors());

    // Set over-range
    packet.set_trailer_over_range(true);
    EXPECT_TRUE(packet.trailer_has_errors());

    // Clear over-range, set sample loss
    packet.set_trailer_over_range(false);
    packet.set_trailer_sample_loss(true);
    EXPECT_TRUE(packet.trailer_has_errors());

    // Both errors
    packet.set_trailer_over_range(true);
    EXPECT_TRUE(packet.trailer_has_errors());

    // Clear all errors
    packet.set_trailer_over_range(false);
    packet.set_trailer_sample_loss(false);
    EXPECT_FALSE(packet.trailer_has_errors());
}

TEST_F(TrailerFieldTest, GoodStatus) {
    PacketType packet(buffer.data());

    packet.set_trailer_good_status();

    EXPECT_TRUE(packet.trailer_valid_data());
    EXPECT_TRUE(packet.trailer_calibrated_time());
    EXPECT_EQ(packet.trailer(), vrtio::trailer::status_all_good);
}

TEST_F(TrailerFieldTest, ClearTrailer) {
    PacketType packet(buffer.data());

    // Set multiple bits
    packet.set_trailer_valid_data(true);
    packet.set_trailer_calibrated_time(true);
    packet.set_trailer_over_range(true);

    EXPECT_NE(packet.trailer(), 0U);

    // Clear
    packet.clear_trailer();
    EXPECT_EQ(packet.trailer(), 0U);
    EXPECT_FALSE(packet.trailer_valid_data());
    EXPECT_FALSE(packet.trailer_calibrated_time());
    EXPECT_FALSE(packet.trailer_over_range());
}

// Test that setting one bit doesn't affect others
TEST_F(TrailerFieldTest, BitIndependence) {
    PacketType packet(buffer.data());

    // Set multiple bits
    packet.set_trailer_valid_data(true);
    packet.set_trailer_context_packets(42);
    packet.set_trailer_over_range(true);

    // Verify initial state
    EXPECT_TRUE(packet.trailer_valid_data());
    EXPECT_EQ(packet.trailer_context_packets(), 42);
    EXPECT_TRUE(packet.trailer_over_range());

    // Toggle one bit
    packet.set_trailer_calibrated_time(true);

    // Verify others unchanged
    EXPECT_TRUE(packet.trailer_valid_data());
    EXPECT_EQ(packet.trailer_context_packets(), 42);
    EXPECT_TRUE(packet.trailer_over_range());
    EXPECT_TRUE(packet.trailer_calibrated_time());

    // Clear one bit
    packet.set_trailer_over_range(false);

    // Verify others unchanged
    EXPECT_TRUE(packet.trailer_valid_data());
    EXPECT_EQ(packet.trailer_context_packets(), 42);
    EXPECT_FALSE(packet.trailer_over_range());
    EXPECT_TRUE(packet.trailer_calibrated_time());
}

// Test builder pattern with individual fields
TEST_F(TrailerFieldTest, BuilderIndividualFields) {
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x12345678)
        .timestamp_integer(1000000)
        .trailer_valid_data(true)
        .trailer_calibrated_time(true)
        .packet_count(12)  // Max 15 (4 bits)
        .build();

    EXPECT_EQ(packet.stream_id(), 0x12345678U);
    EXPECT_EQ(packet.timestamp_integer(), 1000000U);
    EXPECT_TRUE(packet.trailer_valid_data());
    EXPECT_TRUE(packet.trailer_calibrated_time());
    EXPECT_EQ(packet.packet_count(), 12);
}

TEST_F(TrailerFieldTest, BuilderGoodStatus) {
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x11111111)
        .trailer_good_status()
        .build();

    EXPECT_TRUE(packet.trailer_valid_data());
    EXPECT_TRUE(packet.trailer_calibrated_time());
    EXPECT_EQ(packet.trailer(), vrtio::trailer::status_all_good);
}

TEST_F(TrailerFieldTest, BuilderStatusMethod) {
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x22222222)
        .trailer_status(true, true, false, false)
        .build();

    EXPECT_TRUE(packet.trailer_valid_data());
    EXPECT_TRUE(packet.trailer_calibrated_time());
    EXPECT_FALSE(packet.trailer_over_range());
    EXPECT_FALSE(packet.trailer_sample_loss());
}

TEST_F(TrailerFieldTest, BuilderWithErrors) {
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x33333333)
        .trailer_status(true, true, true, true)
        .build();

    EXPECT_TRUE(packet.trailer_valid_data());
    EXPECT_TRUE(packet.trailer_calibrated_time());
    EXPECT_TRUE(packet.trailer_over_range());
    EXPECT_TRUE(packet.trailer_sample_loss());
    EXPECT_TRUE(packet.trailer_has_errors());
}

TEST_F(TrailerFieldTest, BuilderContextPackets) {
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x44444444)
        .trailer_context_packets(10)
        .trailer_valid_data(true)
        .build();

    EXPECT_EQ(packet.trailer_context_packets(), 10);
    EXPECT_TRUE(packet.trailer_valid_data());
}

TEST_F(TrailerFieldTest, BuilderReferenceLock) {
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x55555555)
        .trailer_reference_lock(true)
        .trailer_valid_data(true)
        .build();

    EXPECT_TRUE(packet.trailer_reference_lock());
    EXPECT_TRUE(packet.trailer_valid_data());
}

// Test network byte order preservation
TEST_F(TrailerFieldTest, NetworkByteOrder) {
    PacketType packet(buffer.data());

    // Set a specific trailer pattern
    uint32_t test_value = 0x12345678;
    packet.set_trailer(test_value);

    // Read it back
    EXPECT_EQ(packet.trailer(), test_value);

    // Verify individual bits work correctly with this value
    // Bit 16 should be set (0x00010000 is part of 0x12345678)
    // Actually 0x12345678 has bits: 28,24,22,20,18,14,12,11,10,6,5,4,3
    bool bit16 = (test_value >> 16) & 0x01;
    EXPECT_EQ(packet.trailer_calibrated_time(), bit16);
}

// Test constexpr helper functions directly
TEST(TrailerHelperTest, GetBit) {
    uint32_t value = 0x00030000;  // Bits 16 and 17 set

    EXPECT_TRUE(vrtio::trailer::get_bit<16>(value));
    EXPECT_TRUE(vrtio::trailer::get_bit<17>(value));
    EXPECT_FALSE(vrtio::trailer::get_bit<15>(value));
    EXPECT_FALSE(vrtio::trailer::get_bit<18>(value));
}

TEST(TrailerHelperTest, SetBit) {
    uint32_t value = 0;

    value = vrtio::trailer::set_bit<17>(value, true);
    EXPECT_EQ(value, 1U << 17);

    value = vrtio::trailer::set_bit<16>(value, true);
    EXPECT_EQ(value, (1U << 17) | (1U << 16));

    value = vrtio::trailer::set_bit<17>(value, false);
    EXPECT_EQ(value, 1U << 16);
}

TEST(TrailerHelperTest, GetField) {
    uint32_t value = 0x0000007F;  // Context packets = 127

    uint32_t field = vrtio::trailer::get_field<
        vrtio::trailer::context_packets_shift,
        vrtio::trailer::context_packets_mask
    >(value);

    EXPECT_EQ(field, 127U);
}

TEST(TrailerHelperTest, SetField) {
    uint32_t value = 0;

    value = vrtio::trailer::set_field<
        vrtio::trailer::context_packets_shift,
        vrtio::trailer::context_packets_mask
    >(value, 42);

    EXPECT_EQ(value, 42U);

    // Verify it doesn't affect other bits
    value = 0xFFFFFFFF;
    value = vrtio::trailer::set_field<
        vrtio::trailer::context_packets_shift,
        vrtio::trailer::context_packets_mask
    >(value, 0);

    EXPECT_EQ(value & 0x7F, 0U);
    EXPECT_EQ(value & ~0x7FU, 0xFFFFFF80U);
}

TEST(TrailerHelperTest, IsValidData) {
    EXPECT_FALSE(vrtio::trailer::is_valid_data(0));
    EXPECT_TRUE(vrtio::trailer::is_valid_data(1U << 17));
    EXPECT_TRUE(vrtio::trailer::is_valid_data(0xFFFFFFFF));
}

TEST(TrailerHelperTest, IsCalibratedTime) {
    EXPECT_FALSE(vrtio::trailer::is_calibrated_time(0));
    EXPECT_TRUE(vrtio::trailer::is_calibrated_time(1U << 16));
    EXPECT_TRUE(vrtio::trailer::is_calibrated_time(0xFFFFFFFF));
}

TEST(TrailerHelperTest, HasErrors) {
    EXPECT_FALSE(vrtio::trailer::has_errors(0));
    EXPECT_TRUE(vrtio::trailer::has_errors(1U << 12));  // over-range
    EXPECT_TRUE(vrtio::trailer::has_errors(1U << 13));  // sample loss
    EXPECT_TRUE(vrtio::trailer::has_errors((1U << 12) | (1U << 13)));
    EXPECT_FALSE(vrtio::trailer::has_errors((1U << 16) | (1U << 17)));  // good status
}

TEST(TrailerHelperTest, GetContextPackets) {
    EXPECT_EQ(vrtio::trailer::get_context_packets(0), 0);
    EXPECT_EQ(vrtio::trailer::get_context_packets(42), 42);
    EXPECT_EQ(vrtio::trailer::get_context_packets(127), 127);
    EXPECT_EQ(vrtio::trailer::get_context_packets(0xFFFFFFFF), 127);  // Masked to 7 bits
}

TEST(TrailerHelperTest, CreateGoodStatus) {
    uint32_t status = vrtio::trailer::create_good_status();
    EXPECT_EQ(status, (1U << 17) | (1U << 16));
    EXPECT_TRUE(vrtio::trailer::is_valid_data(status));
    EXPECT_TRUE(vrtio::trailer::is_calibrated_time(status));
    EXPECT_FALSE(vrtio::trailer::has_errors(status));
}

// Test edge cases
TEST_F(TrailerFieldTest, ContextPacketsBoundary) {
    PacketType packet(buffer.data());

    // Test maximum value (7 bits = 127)
    packet.set_trailer_context_packets(127);
    EXPECT_EQ(packet.trailer_context_packets(), 127);

    // Test that values > 127 are masked (truncated to 7 bits)
    packet.set_trailer_context_packets(128);
    EXPECT_EQ(packet.trailer_context_packets(), 0);  // 128 & 0x7F = 0

    packet.set_trailer_context_packets(255);
    EXPECT_EQ(packet.trailer_context_packets(), 127);  // 255 & 0x7F = 127
}

TEST_F(TrailerFieldTest, AllBitsSet) {
    PacketType packet(buffer.data());

    // Set all defined bits
    packet.set_trailer(0xFFFFFFFF);

    // Verify all getters return true/max value
    EXPECT_EQ(packet.trailer_context_packets(), 127);
    EXPECT_TRUE(packet.trailer_reference_lock());
    EXPECT_TRUE(packet.trailer_agc_mgc());
    EXPECT_TRUE(packet.trailer_detected_signal());
    EXPECT_TRUE(packet.trailer_spectral_inversion());
    EXPECT_TRUE(packet.trailer_over_range());
    EXPECT_TRUE(packet.trailer_sample_loss());
    EXPECT_TRUE(packet.trailer_calibrated_time());
    EXPECT_TRUE(packet.trailer_valid_data());
    EXPECT_TRUE(packet.trailer_reference_point());
    EXPECT_TRUE(packet.trailer_signal_detected());
    EXPECT_TRUE(packet.trailer_has_errors());
}

// Test roundtrip through serialization
TEST_F(TrailerFieldTest, SerializationRoundtrip) {
    // Create and configure packet
    [[maybe_unused]] auto packet1 = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0xABCDEF00)
        .timestamp_integer(999999)
        .trailer_valid_data(true)
        .trailer_calibrated_time(true)
        .trailer_context_packets(25)
        .trailer_reference_lock(true)
        .packet_count(10)
        .build();

    // Simulate receiving: parse into a new buffer
    std::vector<uint8_t> rx_buffer(buffer.begin(), buffer.end());
    PacketType packet2(rx_buffer.data(), false);

    // Verify all fields match
    EXPECT_EQ(packet2.stream_id(), 0xABCDEF00U);
    EXPECT_EQ(packet2.timestamp_integer(), 999999U);
    EXPECT_TRUE(packet2.trailer_valid_data());
    EXPECT_TRUE(packet2.trailer_calibrated_time());
    EXPECT_EQ(packet2.trailer_context_packets(), 25);
    EXPECT_TRUE(packet2.trailer_reference_lock());
    EXPECT_EQ(packet2.packet_count(), 10);
}
