#include <array>
#include <vector>

#include <cstring>
#include <gtest/gtest.h>
#include <vrtigo.hpp>

// Test fixture for trailer field manipulation
class TrailerTest : public ::testing::Test {
protected:
    using PacketType = vrtigo::SignalDataPacket<vrtigo::NoClassId, vrtigo::UtcRealTimestamp,
                                                vrtigo::Trailer::included, 128>;

    alignas(4) std::array<uint8_t, PacketType::size_bytes()> buffer{};

    void SetUp() override {
        PacketType packet(buffer);
        packet.trailer().clear(); // Start with zeroed trailer
    }
};

// ============================================================================
// Context Packet Count Tests (E bit + 7-bit count)
// ============================================================================

TEST_F(TrailerTest, ContextPacketCount_InitiallyInvalid) {
    PacketType packet(buffer);

    // When E bit is 0, count should return nullopt
    EXPECT_FALSE(packet.trailer().context_packet_count().has_value());
}

TEST_F(TrailerTest, ContextPacketCount_SetAndGet) {
    PacketType packet(buffer);

    // Set count (should set E bit and count field)
    packet.trailer().set_context_packet_count(42);

    // Should now have a value
    ASSERT_TRUE(packet.trailer().context_packet_count().has_value());
    EXPECT_EQ(*packet.trailer().context_packet_count(), 42);

    // E bit (bit 7) should be set
    EXPECT_TRUE(packet.trailer().raw() & (1U << 7));
}

TEST_F(TrailerTest, ContextPacketCount_Clear) {
    PacketType packet(buffer);

    packet.trailer().set_context_packet_count(10);
    ASSERT_TRUE(packet.trailer().context_packet_count().has_value());

    // Clear should set E bit to 0
    packet.trailer().clear_context_packet_count();
    EXPECT_FALSE(packet.trailer().context_packet_count().has_value());
}

TEST_F(TrailerTest, ContextPacketCount_MaxValue) {
    PacketType packet(buffer);

    // Count is 7 bits, max value is 127
    packet.trailer().set_context_packet_count(127);
    ASSERT_TRUE(packet.trailer().context_packet_count().has_value());
    EXPECT_EQ(*packet.trailer().context_packet_count(), 127);
}

// ============================================================================
// Named Indicators (8 indicators with enable/indicator bit pairing)
// ============================================================================

TEST_F(TrailerTest, CalibratedTime_EnableIndicatorPairing) {
    PacketType packet(buffer);

    // Initially invalid (enable bit 31 = 0)
    EXPECT_FALSE(packet.trailer().calibrated_time().has_value());

    // Set to true (sets enable bit 31 and indicator bit 19)
    packet.trailer().set_calibrated_time(true);
    ASSERT_TRUE(packet.trailer().calibrated_time().has_value());
    EXPECT_TRUE(*packet.trailer().calibrated_time());

    // Verify bits
    uint32_t raw = packet.trailer().raw();
    EXPECT_TRUE(raw & (1U << 31)); // Enable bit
    EXPECT_TRUE(raw & (1U << 19)); // Indicator bit

    // Set to false (enable still set, indicator cleared)
    packet.trailer().set_calibrated_time(false);
    ASSERT_TRUE(packet.trailer().calibrated_time().has_value());
    EXPECT_FALSE(*packet.trailer().calibrated_time());

    raw = packet.trailer().raw();
    EXPECT_TRUE(raw & (1U << 31));  // Enable still set
    EXPECT_FALSE(raw & (1U << 19)); // Indicator cleared
}

TEST_F(TrailerTest, CalibratedTime_Clear) {
    PacketType packet(buffer);

    packet.trailer().set_calibrated_time(true);
    ASSERT_TRUE(packet.trailer().calibrated_time().has_value());

    // Clear should disable the indicator
    packet.trailer().clear_calibrated_time();
    EXPECT_FALSE(packet.trailer().calibrated_time().has_value());

    // Enable bit should be cleared
    EXPECT_FALSE(packet.trailer().raw() & (1U << 31));
}

TEST_F(TrailerTest, ValidData_EnableIndicatorPairing) {
    PacketType packet(buffer);

    EXPECT_FALSE(packet.trailer().valid_data().has_value());

    packet.trailer().set_valid_data(true);
    ASSERT_TRUE(packet.trailer().valid_data().has_value());
    EXPECT_TRUE(*packet.trailer().valid_data());

    // Verify enable bit 30 and indicator bit 18
    uint32_t raw = packet.trailer().raw();
    EXPECT_TRUE(raw & (1U << 30));
    EXPECT_TRUE(raw & (1U << 18));
}

TEST_F(TrailerTest, ReferenceLock_EnableIndicatorPairing) {
    PacketType packet(buffer);

    EXPECT_FALSE(packet.trailer().reference_lock().has_value());

    packet.trailer().set_reference_lock(true);
    ASSERT_TRUE(packet.trailer().reference_lock().has_value());
    EXPECT_TRUE(*packet.trailer().reference_lock());

    // Verify enable bit 29 and indicator bit 17
    uint32_t raw = packet.trailer().raw();
    EXPECT_TRUE(raw & (1U << 29));
    EXPECT_TRUE(raw & (1U << 17));
}

TEST_F(TrailerTest, AgcMgc_EnableIndicatorPairing) {
    PacketType packet(buffer);

    EXPECT_FALSE(packet.trailer().agc_mgc().has_value());

    packet.trailer().set_agc_mgc(true);
    ASSERT_TRUE(packet.trailer().agc_mgc().has_value());
    EXPECT_TRUE(*packet.trailer().agc_mgc());

    // Verify enable bit 28 and indicator bit 16
    uint32_t raw = packet.trailer().raw();
    EXPECT_TRUE(raw & (1U << 28));
    EXPECT_TRUE(raw & (1U << 16));
}

TEST_F(TrailerTest, DetectedSignal_EnableIndicatorPairing) {
    PacketType packet(buffer);

    EXPECT_FALSE(packet.trailer().detected_signal().has_value());

    packet.trailer().set_detected_signal(true);
    ASSERT_TRUE(packet.trailer().detected_signal().has_value());
    EXPECT_TRUE(*packet.trailer().detected_signal());

    // Verify enable bit 27 and indicator bit 15
    uint32_t raw = packet.trailer().raw();
    EXPECT_TRUE(raw & (1U << 27));
    EXPECT_TRUE(raw & (1U << 15));
}

TEST_F(TrailerTest, SpectralInversion_EnableIndicatorPairing) {
    PacketType packet(buffer);

    EXPECT_FALSE(packet.trailer().spectral_inversion().has_value());

    packet.trailer().set_spectral_inversion(true);
    ASSERT_TRUE(packet.trailer().spectral_inversion().has_value());
    EXPECT_TRUE(*packet.trailer().spectral_inversion());

    // Verify enable bit 26 and indicator bit 14
    uint32_t raw = packet.trailer().raw();
    EXPECT_TRUE(raw & (1U << 26));
    EXPECT_TRUE(raw & (1U << 14));
}

TEST_F(TrailerTest, OverRange_EnableIndicatorPairing) {
    PacketType packet(buffer);

    EXPECT_FALSE(packet.trailer().over_range().has_value());

    packet.trailer().set_over_range(true);
    ASSERT_TRUE(packet.trailer().over_range().has_value());
    EXPECT_TRUE(*packet.trailer().over_range());

    // Verify enable bit 25 and indicator bit 13
    uint32_t raw = packet.trailer().raw();
    EXPECT_TRUE(raw & (1U << 25));
    EXPECT_TRUE(raw & (1U << 13));
}

TEST_F(TrailerTest, SampleLoss_EnableIndicatorPairing) {
    PacketType packet(buffer);

    EXPECT_FALSE(packet.trailer().sample_loss().has_value());

    packet.trailer().set_sample_loss(true);
    ASSERT_TRUE(packet.trailer().sample_loss().has_value());
    EXPECT_TRUE(*packet.trailer().sample_loss());

    // Verify enable bit 24 and indicator bit 12
    uint32_t raw = packet.trailer().raw();
    EXPECT_TRUE(raw & (1U << 24));
    EXPECT_TRUE(raw & (1U << 12));
}

// ============================================================================
// Sample Frame and User-Defined Indicators (bits 11-8, enable bits 23-20)
// ============================================================================

TEST_F(TrailerTest, SampleFrame1_DirectAccess) {
    PacketType packet(buffer);

    // Initially not set (enable bit is 0)
    EXPECT_FALSE(packet.trailer().sample_frame_1().has_value());

    // Set to true (sets enable bit 23 and indicator bit 11)
    packet.trailer().set_sample_frame_1(true);
    ASSERT_TRUE(packet.trailer().sample_frame_1().has_value());
    EXPECT_TRUE(*packet.trailer().sample_frame_1());
    EXPECT_TRUE(packet.trailer().raw() & (1U << 11)); // Indicator bit
    EXPECT_TRUE(packet.trailer().raw() & (1U << 23)); // Enable bit

    // Set to false (enable bit stays 1, indicator bit becomes 0)
    packet.trailer().set_sample_frame_1(false);
    ASSERT_TRUE(packet.trailer().sample_frame_1().has_value());
    EXPECT_FALSE(*packet.trailer().sample_frame_1());
}

TEST_F(TrailerTest, SampleFrame0_DirectAccess) {
    PacketType packet(buffer);

    EXPECT_FALSE(packet.trailer().sample_frame_0().has_value());

    packet.trailer().set_sample_frame_0(true);
    ASSERT_TRUE(packet.trailer().sample_frame_0().has_value());
    EXPECT_TRUE(*packet.trailer().sample_frame_0());
    EXPECT_TRUE(packet.trailer().raw() & (1U << 10)); // Indicator bit
    EXPECT_TRUE(packet.trailer().raw() & (1U << 22)); // Enable bit

    packet.trailer().clear_sample_frame_0();
    EXPECT_FALSE(packet.trailer().sample_frame_0().has_value());
}

TEST_F(TrailerTest, UserDefined1_DirectAccess) {
    PacketType packet(buffer);

    EXPECT_FALSE(packet.trailer().user_defined_1().has_value());

    packet.trailer().set_user_defined_1(true);
    ASSERT_TRUE(packet.trailer().user_defined_1().has_value());
    EXPECT_TRUE(*packet.trailer().user_defined_1());
    EXPECT_TRUE(packet.trailer().raw() & (1U << 9));  // Indicator bit
    EXPECT_TRUE(packet.trailer().raw() & (1U << 21)); // Enable bit

    packet.trailer().clear_user_defined_1();
    EXPECT_FALSE(packet.trailer().user_defined_1().has_value());
}

TEST_F(TrailerTest, UserDefined0_DirectAccess) {
    PacketType packet(buffer);

    EXPECT_FALSE(packet.trailer().user_defined_0().has_value());

    packet.trailer().set_user_defined_0(true);
    ASSERT_TRUE(packet.trailer().user_defined_0().has_value());
    EXPECT_TRUE(*packet.trailer().user_defined_0());
    EXPECT_TRUE(packet.trailer().raw() & (1U << 8));  // Indicator bit
    EXPECT_TRUE(packet.trailer().raw() & (1U << 20)); // Enable bit

    packet.trailer().clear_user_defined_0();
    EXPECT_FALSE(packet.trailer().user_defined_0().has_value());
}

// ============================================================================
// TrailerBuilder Tests
// ============================================================================

TEST_F(TrailerTest, Builder_ContextPacketCount) {
    PacketType packet(buffer);

    vrtigo::TrailerBuilder().context_packet_count(25).apply(packet.trailer());

    ASSERT_TRUE(packet.trailer().context_packet_count().has_value());
    EXPECT_EQ(*packet.trailer().context_packet_count(), 25);
}

TEST_F(TrailerTest, Builder_NamedIndicators) {
    PacketType packet(buffer);

    vrtigo::TrailerBuilder().calibrated_time(true).valid_data(true).reference_lock(false).apply(
        packet.trailer());

    ASSERT_TRUE(packet.trailer().calibrated_time().has_value());
    EXPECT_TRUE(*packet.trailer().calibrated_time());

    ASSERT_TRUE(packet.trailer().valid_data().has_value());
    EXPECT_TRUE(*packet.trailer().valid_data());

    ASSERT_TRUE(packet.trailer().reference_lock().has_value());
    EXPECT_FALSE(*packet.trailer().reference_lock());
}

TEST_F(TrailerTest, Builder_SampleFrameAndUserDefined) {
    PacketType packet(buffer);

    vrtigo::TrailerBuilder()
        .sample_frame_1(true)
        .sample_frame_0(false)
        .user_defined_1(true)
        .user_defined_0(false)
        .apply(packet.trailer());

    ASSERT_TRUE(packet.trailer().sample_frame_1().has_value());
    EXPECT_TRUE(*packet.trailer().sample_frame_1());
    ASSERT_TRUE(packet.trailer().sample_frame_0().has_value());
    EXPECT_FALSE(*packet.trailer().sample_frame_0());
    ASSERT_TRUE(packet.trailer().user_defined_1().has_value());
    EXPECT_TRUE(*packet.trailer().user_defined_1());
    ASSERT_TRUE(packet.trailer().user_defined_0().has_value());
    EXPECT_FALSE(*packet.trailer().user_defined_0());
}

TEST_F(TrailerTest, Builder_ComplexTrailer) {
    PacketType packet(buffer);

    vrtigo::TrailerBuilder()
        .context_packet_count(10)
        .calibrated_time(true)
        .valid_data(true)
        .over_range(false)
        .sample_loss(false)
        .sample_frame_1(true)
        .apply(packet.trailer());

    ASSERT_TRUE(packet.trailer().context_packet_count().has_value());
    EXPECT_EQ(*packet.trailer().context_packet_count(), 10);

    ASSERT_TRUE(packet.trailer().calibrated_time().has_value());
    EXPECT_TRUE(*packet.trailer().calibrated_time());

    ASSERT_TRUE(packet.trailer().valid_data().has_value());
    EXPECT_TRUE(*packet.trailer().valid_data());

    ASSERT_TRUE(packet.trailer().over_range().has_value());
    EXPECT_FALSE(*packet.trailer().over_range());

    ASSERT_TRUE(packet.trailer().sample_frame_1().has_value());
    EXPECT_TRUE(*packet.trailer().sample_frame_1());
}

TEST_F(TrailerTest, Builder_ValueMethod) {
    uint32_t trailer_word = vrtigo::TrailerBuilder().calibrated_time(true).valid_data(true).value();

    // Should have enable bits 31,30 and indicator bits 19,18
    EXPECT_TRUE(trailer_word & (1U << 31));
    EXPECT_TRUE(trailer_word & (1U << 30));
    EXPECT_TRUE(trailer_word & (1U << 19));
    EXPECT_TRUE(trailer_word & (1U << 18));
}

TEST_F(TrailerTest, Builder_FromView) {
    PacketType packet(buffer);

    // Set up trailer
    packet.trailer().set_calibrated_time(true);
    packet.trailer().set_context_packet_count(42);

    // Copy to builder and modify
    auto new_trailer =
        vrtigo::TrailerBuilder().from_view(packet.trailer()).valid_data(true).value();

    // Should have original values plus new one
    PacketType packet2(buffer);
    packet2.trailer().set_raw(new_trailer);

    ASSERT_TRUE(packet2.trailer().calibrated_time().has_value());
    EXPECT_TRUE(*packet2.trailer().calibrated_time());

    ASSERT_TRUE(packet2.trailer().context_packet_count().has_value());
    EXPECT_EQ(*packet2.trailer().context_packet_count(), 42);

    ASSERT_TRUE(packet2.trailer().valid_data().has_value());
    EXPECT_TRUE(*packet2.trailer().valid_data());
}

// ============================================================================
// Clear Methods Tests
// ============================================================================

TEST_F(TrailerTest, ClearMethods_NamedIndicators) {
    PacketType packet(buffer);

    // Set multiple indicators
    packet.trailer().set_calibrated_time(true);
    packet.trailer().set_valid_data(true);
    packet.trailer().set_reference_lock(true);

    // Clear one
    packet.trailer().clear_valid_data();

    // Others should still be set
    ASSERT_TRUE(packet.trailer().calibrated_time().has_value());
    EXPECT_TRUE(*packet.trailer().calibrated_time());

    ASSERT_TRUE(packet.trailer().reference_lock().has_value());
    EXPECT_TRUE(*packet.trailer().reference_lock());

    // Cleared one should return nullopt
    EXPECT_FALSE(packet.trailer().valid_data().has_value());
}

TEST_F(TrailerTest, ClearMethod_EntireTrailer) {
    PacketType packet(buffer);

    // Set everything
    packet.trailer().set_context_packet_count(50);
    packet.trailer().set_calibrated_time(true);
    packet.trailer().set_valid_data(true);
    packet.trailer().set_sample_frame_1(true);

    // Clear entire trailer
    packet.trailer().clear();

    // Everything should be invalid/false
    EXPECT_FALSE(packet.trailer().context_packet_count().has_value());
    EXPECT_FALSE(packet.trailer().calibrated_time().has_value());
    EXPECT_FALSE(packet.trailer().valid_data().has_value());
    EXPECT_FALSE(packet.trailer().sample_frame_1());
    EXPECT_EQ(packet.trailer().raw(), 0U);
}

// ============================================================================
// Multiple Indicators Test
// ============================================================================

TEST_F(TrailerTest, MultipleIndicators_Independent) {
    PacketType packet(buffer);

    // Set multiple indicators independently
    packet.trailer().set_calibrated_time(true);
    packet.trailer().set_valid_data(false);
    packet.trailer().set_over_range(true);
    packet.trailer().set_sample_loss(false);
    packet.trailer().set_context_packet_count(7);

    // All should be independently accessible
    ASSERT_TRUE(packet.trailer().calibrated_time().has_value());
    EXPECT_TRUE(*packet.trailer().calibrated_time());

    ASSERT_TRUE(packet.trailer().valid_data().has_value());
    EXPECT_FALSE(*packet.trailer().valid_data());

    ASSERT_TRUE(packet.trailer().over_range().has_value());
    EXPECT_TRUE(*packet.trailer().over_range());

    ASSERT_TRUE(packet.trailer().sample_loss().has_value());
    EXPECT_FALSE(*packet.trailer().sample_loss());

    ASSERT_TRUE(packet.trailer().context_packet_count().has_value());
    EXPECT_EQ(*packet.trailer().context_packet_count(), 7);
}

// ============================================================================
// Endian Test
// ============================================================================

TEST_F(TrailerTest, EndianHandling) {
    PacketType packet(buffer);

    packet.trailer().set_calibrated_time(true);
    packet.trailer().set_context_packet_count(42);

    // Read back should work correctly (verifies endian handling)
    ASSERT_TRUE(packet.trailer().calibrated_time().has_value());
    EXPECT_TRUE(*packet.trailer().calibrated_time());

    ASSERT_TRUE(packet.trailer().context_packet_count().has_value());
    EXPECT_EQ(*packet.trailer().context_packet_count(), 42);
}

// ============================================================================
// VITA 49.2 Rule 5.1.6-13 Compliance
// ============================================================================

TEST_F(TrailerTest, Rule_5_1_6_13_Compliance) {
    PacketType packet(buffer);

    // Rule 5.1.6-13: When E bit is 0, count is undefined
    EXPECT_FALSE(packet.trailer().context_packet_count().has_value());

    // When E bit is 1, count shall be valid
    packet.trailer().set_context_packet_count(100);
    ASSERT_TRUE(packet.trailer().context_packet_count().has_value());
    EXPECT_EQ(*packet.trailer().context_packet_count(), 100);

    // Clearing E bit makes count undefined again
    packet.trailer().clear_context_packet_count();
    EXPECT_FALSE(packet.trailer().context_packet_count().has_value());
}
