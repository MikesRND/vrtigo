#include <gtest/gtest.h>
#include <vrtigo.hpp>
#include <vrtigo/detail/packet_concepts.hpp>

using namespace vrtigo;
using namespace vrtigo::field;

// ============================================================================
// PacketMetadataLike - shared read-only surface
// ============================================================================

TEST(PacketConceptsTest, AllPacketsSatisfyPacketMetadataLike) {
    // Compile-time packet builders
    using DataPkt = typed::DataPacketBuilder<PacketType::signal_data, 64>;
    using CtxPkt = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth>;

    static_assert(PacketMetadataLike<DataPkt>);
    static_assert(PacketMetadataLike<CtxPkt>);

    // Runtime packets
    static_assert(PacketMetadataLike<dynamic::DataPacketView>);
    static_assert(PacketMetadataLike<dynamic::ContextPacketView>);
}

// ============================================================================
// CompileTimePacketLike - static structure, mutable
// ============================================================================

TEST(PacketConceptsTest, DataPacketBuilderSatisfiesCompileTimePacketLike) {
    using PktType = typed::DataPacketBuilder<PacketType::signal_data, 64, UtcRealTimestamp,
                                             NoClassId, WithTrailer>;

    static_assert(PacketMetadataLike<PktType>);
    static_assert(CompileTimePacketLike<PktType>);
    static_assert(DataPacketBuilderLike<PktType>);

    // Should NOT satisfy context or dynamic concepts
    static_assert(!ContextPacketBuilderLike<PktType>);
    static_assert(!DynamicPacketViewLike<PktType>);
}

TEST(PacketConceptsTest, ContextPacketBuilderSatisfiesCompileTimePacketLike) {
    using PktType = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth, sample_rate>;

    static_assert(PacketMetadataLike<PktType>);
    static_assert(CompileTimePacketLike<PktType>);
    static_assert(ContextPacketBuilderLike<PktType>);

    // Should NOT satisfy data or dynamic concepts
    static_assert(!DataPacketBuilderLike<PktType>);
    static_assert(!DynamicPacketViewLike<PktType>);
}

// ============================================================================
// DynamicPacketViewLike - parsed, read-only
// ============================================================================

TEST(PacketConceptsTest, DynamicDataPacketSatisfiesDynamicPacketViewLike) {
    static_assert(PacketMetadataLike<dynamic::DataPacketView>);
    static_assert(DynamicPacketViewLike<dynamic::DataPacketView>);
    static_assert(DynamicDataPacketViewLike<dynamic::DataPacketView>);

    // Should NOT satisfy context or compile-time concepts
    static_assert(!DynamicContextPacketViewLike<dynamic::DataPacketView>);
    static_assert(!CompileTimePacketLike<dynamic::DataPacketView>);
}

TEST(PacketConceptsTest, DynamicContextPacketSatisfiesDynamicPacketViewLike) {
    static_assert(PacketMetadataLike<dynamic::ContextPacketView>);
    static_assert(DynamicPacketViewLike<dynamic::ContextPacketView>);
    static_assert(DynamicContextPacketViewLike<dynamic::ContextPacketView>);

    // Should NOT satisfy data or compile-time concepts
    static_assert(!DynamicDataPacketViewLike<dynamic::ContextPacketView>);
    static_assert(!CompileTimePacketLike<dynamic::ContextPacketView>);
}

// ============================================================================
// Mutual exclusivity tests
// ============================================================================

TEST(PacketConceptsTest, CTAndDynamicMutuallyExclusive) {
    using DataPkt = typed::DataPacketBuilder<PacketType::signal_data, 64>;
    using CtxPkt = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth>;

    // CT packet builders are NOT Dynamic
    static_assert(!DynamicPacketViewLike<DataPkt>);
    static_assert(!DynamicPacketViewLike<CtxPkt>);

    // Dynamic packets are NOT CT
    static_assert(!CompileTimePacketLike<dynamic::DataPacketView>);
    static_assert(!CompileTimePacketLike<dynamic::ContextPacketView>);
}

TEST(PacketConceptsTest, DataAndContextMutuallyExclusive) {
    using DataPkt = typed::DataPacketBuilder<PacketType::signal_data, 64>;
    using CtxPkt = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth>;

    // Data is not Context
    static_assert(DataPacketBuilderLike<DataPkt>);
    static_assert(!ContextPacketBuilderLike<DataPkt>);

    // Context is not Data
    static_assert(ContextPacketBuilderLike<CtxPkt>);
    static_assert(!DataPacketBuilderLike<CtxPkt>);

    // Same for dynamic
    static_assert(DynamicDataPacketViewLike<dynamic::DataPacketView>);
    static_assert(!DynamicContextPacketViewLike<dynamic::DataPacketView>);

    static_assert(DynamicContextPacketViewLike<dynamic::ContextPacketView>);
    static_assert(!DynamicDataPacketViewLike<dynamic::ContextPacketView>);
}

// ============================================================================
// Helper concepts for DataPacketBuilder
// ============================================================================

TEST(PacketConceptsTest, HelperConceptsForBuilder) {
    using WithStreamIdPkt = typed::DataPacketBuilder<PacketType::signal_data, 64>;
    using NoStreamIdPkt = typed::DataPacketBuilder<PacketType::signal_data_no_id, 64>;
    using WithTrailerPkt =
        typed::DataPacketBuilder<PacketType::signal_data, 64, NoTimestamp, NoClassId, WithTrailer>;

    // Stream ID
    static_assert(HasStreamId<WithStreamIdPkt>);
    static_assert(!HasStreamId<NoStreamIdPkt>);

    // Trailer
    static_assert(HasMutableTrailer<WithTrailerPkt>);
    static_assert(!HasMutableTrailer<NoStreamIdPkt>);

    // Payload
    static_assert(HasPayload<WithStreamIdPkt>);
    static_assert(HasPayload<NoStreamIdPkt>);

    // Packet count
    static_assert(HasPacketCount<WithStreamIdPkt>);
    static_assert(HasPacketCount<NoStreamIdPkt>);
}

// ============================================================================
// Non-packet types rejected
// ============================================================================

TEST(PacketConceptsTest, NonPacketTypesRejected) {
    static_assert(!PacketMetadataLike<int>);
    static_assert(!PacketMetadataLike<std::string>);
    static_assert(!PacketMetadataLike<std::vector<uint8_t>>);

    static_assert(!CompileTimePacketLike<int>);
    static_assert(!DynamicPacketViewLike<std::string>);
    static_assert(!DataPacketBuilderLike<std::vector<uint8_t>>);
    static_assert(!ContextPacketBuilderLike<void*>);
}

// ============================================================================
// Runtime behavior verification
// ============================================================================

TEST(PacketConceptsTest, RuntimeBehaviorConsistency) {
    // Create data packet
    using SignalType = typed::DataPacketBuilder<PacketType::signal_data, 32>;
    alignas(4) std::array<uint8_t, SignalType::size_bytes()> signal_buffer;
    SignalType signal_pkt(signal_buffer);

    EXPECT_NO_THROW({
        signal_pkt.set_stream_id(0x12345678);
        EXPECT_EQ(signal_pkt.stream_id(), 0x12345678);
        EXPECT_EQ(signal_pkt.as_bytes().data(), signal_buffer.data());
    });

    // Parse as runtime packet
    auto data_result = dynamic::DataPacketView::parse(signal_buffer);
    ASSERT_TRUE(data_result.has_value()) << data_result.error().message();
    const auto& view = data_result.value();
    EXPECT_NO_THROW({
        auto id = view.stream_id();
        ASSERT_TRUE(id.has_value());
        EXPECT_EQ(*id, 0x12345678);
    });

    // Create context packet
    using ContextType = typed::ContextPacketBuilder<NoTimestamp, NoClassId, bandwidth>;
    alignas(4) std::array<uint8_t, ContextType::size_bytes()> context_buffer;
    ContextType context_pkt(context_buffer);

    EXPECT_NO_THROW({
        context_pkt.set_stream_id(0xABCDEF00);
        EXPECT_EQ(context_pkt.stream_id(), 0xABCDEF00);
    });

    // Parse as runtime context packet
    auto ctx_result = dynamic::ContextPacketView::parse(context_buffer);
    ASSERT_TRUE(ctx_result.has_value()) << ctx_result.error().message();
    const auto& ctx_view = ctx_result.value();
    EXPECT_FALSE(ctx_view.change_indicator());
}
