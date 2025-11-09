#include <vrtio/packet/data_packet.hpp>
#include <vrtio/packet/builder.hpp>
#include <vrtio/core/trailer.hpp>
#include <vrtio/core/trailer_view.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <array>
#include <cstring>

// Test fixture for trailer field manipulation
class TrailerFieldTest : public ::testing::Test {
protected:
    using PacketType = vrtio::SignalDataPacket<
        vrtio::TimeStampUTC,
        vrtio::Trailer::Included,  // Trailer included
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
    EXPECT_FALSE(packet.trailer().valid_data());

    // Set to true
    packet.trailer().set_valid_data(true);
    EXPECT_TRUE(packet.trailer().valid_data());
    EXPECT_EQ(packet.trailer().raw(), 1U << 17);

    // Set to false
    packet.trailer().set_valid_data(false);
    EXPECT_FALSE(packet.trailer().valid_data());
    EXPECT_EQ(packet.trailer().raw(), 0U);
}

TEST_F(TrailerFieldTest, CalibratedTimeBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer().calibrated_time());

    packet.trailer().set_calibrated_time(true);
    EXPECT_TRUE(packet.trailer().calibrated_time());
    EXPECT_EQ(packet.trailer().raw(), 1U << 16);

    packet.trailer().set_calibrated_time(false);
    EXPECT_FALSE(packet.trailer().calibrated_time());
}

TEST_F(TrailerFieldTest, OverRangeBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer().over_range());

    packet.trailer().set_over_range(true);
    EXPECT_TRUE(packet.trailer().over_range());
    EXPECT_EQ(packet.trailer().raw(), 1U << 12);

    packet.trailer().set_over_range(false);
    EXPECT_FALSE(packet.trailer().over_range());
}

TEST_F(TrailerFieldTest, SampleLossBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer().sample_loss());

    packet.trailer().set_sample_loss(true);
    EXPECT_TRUE(packet.trailer().sample_loss());
    EXPECT_EQ(packet.trailer().raw(), 1U << 13);
}

TEST_F(TrailerFieldTest, ReferenceLockBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer().reference_lock());

    packet.trailer().set_reference_lock(true);
    EXPECT_TRUE(packet.trailer().reference_lock());
    EXPECT_EQ(packet.trailer().raw(), 1U << 8);
}

TEST_F(TrailerFieldTest, AgcMgcBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer().agc_mgc());

    packet.trailer().set_agc_mgc(true);
    EXPECT_TRUE(packet.trailer().agc_mgc());
    EXPECT_EQ(packet.trailer().raw(), 1U << 9);
}

TEST_F(TrailerFieldTest, DetectedSignalBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer().detected_signal());

    packet.trailer().set_detected_signal(true);
    EXPECT_TRUE(packet.trailer().detected_signal());
    EXPECT_EQ(packet.trailer().raw(), 1U << 10);
}

TEST_F(TrailerFieldTest, SpectralInversionBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer().spectral_inversion());

    packet.trailer().set_spectral_inversion(true);
    EXPECT_TRUE(packet.trailer().spectral_inversion());
    EXPECT_EQ(packet.trailer().raw(), 1U << 11);
}

TEST_F(TrailerFieldTest, ReferencePointBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer().reference_point());

    packet.trailer().set_reference_point(true);
    EXPECT_TRUE(packet.trailer().reference_point());
    EXPECT_EQ(packet.trailer().raw(), 1U << 18);
}

TEST_F(TrailerFieldTest, SignalDetectedBit) {
    PacketType packet(buffer.data());

    EXPECT_FALSE(packet.trailer().signal_detected());

    packet.trailer().set_signal_detected(true);
    EXPECT_TRUE(packet.trailer().signal_detected());
    EXPECT_EQ(packet.trailer().raw(), 1U << 19);
}

TEST_F(TrailerFieldTest, ContextPacketsField) {
    PacketType packet(buffer.data());

    EXPECT_EQ(packet.trailer().context_packets(), 0);

    // Set to various values
    packet.trailer().set_context_packets(5);
    EXPECT_EQ(packet.trailer().context_packets(), 5);
    EXPECT_EQ(packet.trailer().raw(), 5U);

    packet.trailer().set_context_packets(127);
    EXPECT_EQ(packet.trailer().context_packets(), 127);
    EXPECT_EQ(packet.trailer().raw(), 127U);

    packet.trailer().set_context_packets(0);
    EXPECT_EQ(packet.trailer().context_packets(), 0);
}

// Test multiple bits set simultaneously
TEST_F(TrailerFieldTest, MultipleBitsSet) {
    PacketType packet(buffer.data());

    packet.trailer().set_valid_data(true);
    packet.trailer().set_calibrated_time(true);

    EXPECT_TRUE(packet.trailer().valid_data());
    EXPECT_TRUE(packet.trailer().calibrated_time());

    uint32_t expected = (1U << 17) | (1U << 16);
    EXPECT_EQ(packet.trailer().raw(), expected);
}

TEST_F(TrailerFieldTest, ErrorFlags) {
    PacketType packet(buffer.data());

    // No errors initially
    EXPECT_FALSE(packet.trailer().has_errors());

    // Set over-range
    packet.trailer().set_over_range(true);
    EXPECT_TRUE(packet.trailer().has_errors());

    // Clear over-range, set sample loss
    packet.trailer().set_over_range(false);
    packet.trailer().set_sample_loss(true);
    EXPECT_TRUE(packet.trailer().has_errors());

    // Both errors
    packet.trailer().set_over_range(true);
    EXPECT_TRUE(packet.trailer().has_errors());

    // Clear all errors
    packet.trailer().set_over_range(false);
    packet.trailer().set_sample_loss(false);
    EXPECT_FALSE(packet.trailer().has_errors());
}

TEST_F(TrailerFieldTest, GoodStatus) {
    PacketType packet(buffer.data());

    packet.trailer().set_valid_data(true);
    packet.trailer().set_calibrated_time(true);

    EXPECT_TRUE(packet.trailer().valid_data());
    EXPECT_TRUE(packet.trailer().calibrated_time());
    EXPECT_EQ(packet.trailer().raw(), vrtio::trailer::status_all_good);
}

TEST_F(TrailerFieldTest, ClearTrailer) {
    PacketType packet(buffer.data());

    // Set multiple bits
    packet.trailer().set_valid_data(true);
    packet.trailer().set_calibrated_time(true);
    packet.trailer().set_over_range(true);

    EXPECT_NE(packet.trailer().raw(), 0U);

    // Clear
    packet.trailer().clear();
    EXPECT_EQ(packet.trailer().raw(), 0U);
    EXPECT_FALSE(packet.trailer().valid_data());
    EXPECT_FALSE(packet.trailer().calibrated_time());
    EXPECT_FALSE(packet.trailer().over_range());
}

// Test that setting one bit doesn't affect others
TEST_F(TrailerFieldTest, BitIndependence) {
    PacketType packet(buffer.data());

    // Set multiple bits
    packet.trailer().set_valid_data(true);
    packet.trailer().set_context_packets(42);
    packet.trailer().set_over_range(true);

    // Verify initial state
    EXPECT_TRUE(packet.trailer().valid_data());
    EXPECT_EQ(packet.trailer().context_packets(), 42);
    EXPECT_TRUE(packet.trailer().over_range());

    // Toggle one bit
    packet.trailer().set_calibrated_time(true);

    // Verify others unchanged
    EXPECT_TRUE(packet.trailer().valid_data());
    EXPECT_EQ(packet.trailer().context_packets(), 42);
    EXPECT_TRUE(packet.trailer().over_range());
    EXPECT_TRUE(packet.trailer().calibrated_time());

    // Clear one bit
    packet.trailer().set_over_range(false);

    // Verify others unchanged
    EXPECT_TRUE(packet.trailer().valid_data());
    EXPECT_EQ(packet.trailer().context_packets(), 42);
    EXPECT_FALSE(packet.trailer().over_range());
    EXPECT_TRUE(packet.trailer().calibrated_time());
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
    EXPECT_TRUE(packet.trailer().valid_data());
    EXPECT_TRUE(packet.trailer().calibrated_time());
    EXPECT_EQ(packet.packet_count(), 12);
}

TEST_F(TrailerFieldTest, BuilderGoodStatus) {
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x11111111)
        .trailer_valid_data(true)
        .trailer_calibrated_time(true)
        .build();

    EXPECT_TRUE(packet.trailer().valid_data());
    EXPECT_TRUE(packet.trailer().calibrated_time());
    EXPECT_EQ(packet.trailer().raw(), vrtio::trailer::status_all_good);
}

TEST_F(TrailerFieldTest, BuilderWithErrors) {
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x33333333)
        .trailer_valid_data(true)
        .trailer_calibrated_time(true)
        .trailer_over_range(true)
        .trailer_sample_loss(true)
        .build();

    EXPECT_TRUE(packet.trailer().valid_data());
    EXPECT_TRUE(packet.trailer().calibrated_time());
    EXPECT_TRUE(packet.trailer().over_range());
    EXPECT_TRUE(packet.trailer().sample_loss());
    EXPECT_TRUE(packet.trailer().has_errors());
}

TEST_F(TrailerFieldTest, BuilderContextPackets) {
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x44444444)
        .trailer_context_packets(10)
        .trailer_valid_data(true)
        .build();

    EXPECT_EQ(packet.trailer().context_packets(), 10);
    EXPECT_TRUE(packet.trailer().valid_data());
}

TEST_F(TrailerFieldTest, BuilderReferenceLock) {
    auto packet = vrtio::PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x55555555)
        .trailer_reference_lock(true)
        .trailer_valid_data(true)
        .build();

    EXPECT_TRUE(packet.trailer().reference_lock());
    EXPECT_TRUE(packet.trailer().valid_data());
}

// Test network byte order preservation
TEST_F(TrailerFieldTest, NetworkByteOrder) {
    PacketType packet(buffer.data());

    // Set a specific trailer pattern
    uint32_t test_value = 0x12345678;
    packet.trailer().set_raw(test_value);

    // Read it back
    EXPECT_EQ(packet.trailer().raw(), test_value);

    // Verify individual bits work correctly with this value
    // Bit 16 should be set (0x00010000 is part of 0x12345678)
    // Actually 0x12345678 has bits: 28,24,22,20,18,14,12,11,10,6,5,4,3
    bool bit16 = (test_value >> 16) & 0x01;
    EXPECT_EQ(packet.trailer().calibrated_time(), bit16);
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

TEST(TrailerViewTest, MutableViewUpdatesBuffer) {
    alignas(4) std::array<uint8_t, 4> buffer{};
    vrtio::TrailerView view(buffer.data());
    EXPECT_EQ(view.raw(), 0U);
    view.set_valid_data(true);
    view.set_calibrated_time(true);
    EXPECT_TRUE(view.valid_data());
    EXPECT_TRUE(view.calibrated_time());

    vrtio::ConstTrailerView const_view(buffer.data());
    EXPECT_TRUE(const_view.valid_data());
    EXPECT_TRUE(const_view.calibrated_time());
}

TEST(TrailerViewTest, StatusHelpers) {
    alignas(4) std::array<uint8_t, 4> buffer{};
    vrtio::TrailerView view(buffer.data());
    view.set_valid_data(true);
    view.set_calibrated_time(true);
    view.set_over_range(true);
    view.set_sample_loss(false);
    EXPECT_TRUE(view.valid_data());
    EXPECT_TRUE(view.calibrated_time());
    EXPECT_TRUE(view.over_range());
    EXPECT_FALSE(view.sample_loss());

    view.clear();
    EXPECT_EQ(view.raw(), 0U);
    view.set_valid_data(true);
    view.set_calibrated_time(true);
    EXPECT_EQ(view.raw(), vrtio::trailer::status_all_good);
}

TEST(TrailerBuilderTest, FluentValue) {
    auto builder = vrtio::TrailerBuilder{}
        .valid_data(true)
        .calibrated_time(true)
        .context_packets(12)
        .over_range(true);

    alignas(4) std::array<uint8_t, 4> buffer{};
    builder.apply(vrtio::TrailerView(buffer.data()));

    vrtio::ConstTrailerView view(buffer.data());
    EXPECT_TRUE(view.valid_data());
    EXPECT_TRUE(view.calibrated_time());
    EXPECT_EQ(view.context_packets(), 12);
    EXPECT_TRUE(view.over_range());
    EXPECT_TRUE(view.has_errors());
}

// Test edge cases
TEST_F(TrailerFieldTest, ContextPacketsBoundary) {
    PacketType packet(buffer.data());

    // Test maximum value (7 bits = 127)
    packet.trailer().set_context_packets(127);
    EXPECT_EQ(packet.trailer().context_packets(), 127);

    // Test that values > 127 are masked (truncated to 7 bits)
    packet.trailer().set_context_packets(128);
    EXPECT_EQ(packet.trailer().context_packets(), 0);  // 128 & 0x7F = 0

    packet.trailer().set_context_packets(255);
    EXPECT_EQ(packet.trailer().context_packets(), 127);  // 255 & 0x7F = 127
}

TEST_F(TrailerFieldTest, AllBitsSet) {
    PacketType packet(buffer.data());

    // Set all defined bits
    packet.trailer().set_raw(0xFFFFFFFF);

    // Verify all getters return true/max value
    EXPECT_EQ(packet.trailer().context_packets(), 127);
    EXPECT_TRUE(packet.trailer().reference_lock());
    EXPECT_TRUE(packet.trailer().agc_mgc());
    EXPECT_TRUE(packet.trailer().detected_signal());
    EXPECT_TRUE(packet.trailer().spectral_inversion());
    EXPECT_TRUE(packet.trailer().over_range());
    EXPECT_TRUE(packet.trailer().sample_loss());
    EXPECT_TRUE(packet.trailer().calibrated_time());
    EXPECT_TRUE(packet.trailer().valid_data());
    EXPECT_TRUE(packet.trailer().reference_point());
    EXPECT_TRUE(packet.trailer().signal_detected());
    EXPECT_TRUE(packet.trailer().has_errors());
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
    EXPECT_TRUE(packet2.trailer().valid_data());
    EXPECT_TRUE(packet2.trailer().calibrated_time());
    EXPECT_EQ(packet2.trailer().context_packets(), 25);
    EXPECT_TRUE(packet2.trailer().reference_lock());
    EXPECT_EQ(packet2.packet_count(), 10);
}
