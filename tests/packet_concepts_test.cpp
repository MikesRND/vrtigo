#include <gtest/gtest.h>
#include <vrtigo.hpp>

using namespace vrtigo;
using namespace vrtigo::field;

// ============================================================================
// PacketMetadataLike - shared read-only surface
// ============================================================================

TEST(PacketConceptsTest, AllPacketsSatisfyPacketMetadataLike) {
    // Compile-time packets
    using DataPkt = DataPacket<PacketType::signal_data, NoClassId, NoTimestamp, Trailer::none, 64>;
    using CtxPkt = ContextPacket<NoTimestamp, NoClassId, bandwidth>;

    static_assert(PacketMetadataLike<DataPkt>);
    static_assert(PacketMetadataLike<CtxPkt>);

    // Runtime packets
    static_assert(PacketMetadataLike<RuntimeDataPacket>);
    static_assert(PacketMetadataLike<RuntimeContextPacket>);
}

// ============================================================================
// CompileTimePacketLike - static structure, mutable
// ============================================================================

TEST(PacketConceptsTest, DataPacketSatisfiesCompileTimePacketLike) {
    using PktType =
        DataPacket<PacketType::signal_data, NoClassId, UtcRealTimestamp, Trailer::included, 64>;

    static_assert(PacketMetadataLike<PktType>);
    static_assert(CompileTimePacketLike<PktType>);
    static_assert(DataPacketLike<PktType>);

    // Should NOT satisfy context or runtime concepts
    static_assert(!ContextPacketLike<PktType>);
    static_assert(!RuntimePacketLike<PktType>);
}

TEST(PacketConceptsTest, ContextPacketSatisfiesCompileTimePacketLike) {
    using PktType = ContextPacket<NoTimestamp, NoClassId, bandwidth, sample_rate>;

    static_assert(PacketMetadataLike<PktType>);
    static_assert(CompileTimePacketLike<PktType>);
    static_assert(ContextPacketLike<PktType>);

    // Should NOT satisfy data or runtime concepts
    static_assert(!DataPacketLike<PktType>);
    static_assert(!RuntimePacketLike<PktType>);
}

// ============================================================================
// RuntimePacketLike - parsed, read-only
// ============================================================================

TEST(PacketConceptsTest, RuntimeDataPacketSatisfiesRuntimePacketLike) {
    static_assert(PacketMetadataLike<RuntimeDataPacket>);
    static_assert(RuntimePacketLike<RuntimeDataPacket>);
    static_assert(RuntimeDataPacketLike<RuntimeDataPacket>);

    // Should NOT satisfy context or compile-time concepts
    static_assert(!RuntimeContextPacketLike<RuntimeDataPacket>);
    static_assert(!CompileTimePacketLike<RuntimeDataPacket>);
}

TEST(PacketConceptsTest, RuntimeContextPacketSatisfiesRuntimePacketLike) {
    static_assert(PacketMetadataLike<RuntimeContextPacket>);
    static_assert(RuntimePacketLike<RuntimeContextPacket>);
    static_assert(RuntimeContextPacketLike<RuntimeContextPacket>);

    // Should NOT satisfy data or compile-time concepts
    static_assert(!RuntimeDataPacketLike<RuntimeContextPacket>);
    static_assert(!CompileTimePacketLike<RuntimeContextPacket>);
}

// ============================================================================
// Mutual exclusivity tests
// ============================================================================

TEST(PacketConceptsTest, CTAndRTMutuallyExclusive) {
    using DataPkt = DataPacket<PacketType::signal_data, NoClassId, NoTimestamp, Trailer::none, 64>;
    using CtxPkt = ContextPacket<NoTimestamp, NoClassId, bandwidth>;

    // CT packets are NOT RT
    static_assert(!RuntimePacketLike<DataPkt>);
    static_assert(!RuntimePacketLike<CtxPkt>);

    // RT packets are NOT CT
    static_assert(!CompileTimePacketLike<RuntimeDataPacket>);
    static_assert(!CompileTimePacketLike<RuntimeContextPacket>);
}

TEST(PacketConceptsTest, DataAndContextMutuallyExclusive) {
    using DataPkt = DataPacket<PacketType::signal_data, NoClassId, NoTimestamp, Trailer::none, 64>;
    using CtxPkt = ContextPacket<NoTimestamp, NoClassId, bandwidth>;

    // Data is not Context
    static_assert(DataPacketLike<DataPkt>);
    static_assert(!ContextPacketLike<DataPkt>);

    // Context is not Data
    static_assert(ContextPacketLike<CtxPkt>);
    static_assert(!DataPacketLike<CtxPkt>);

    // Same for runtime
    static_assert(RuntimeDataPacketLike<RuntimeDataPacket>);
    static_assert(!RuntimeContextPacketLike<RuntimeDataPacket>);

    static_assert(RuntimeContextPacketLike<RuntimeContextPacket>);
    static_assert(!RuntimeDataPacketLike<RuntimeContextPacket>);
}

// ============================================================================
// Helper concepts for PacketBuilder
// ============================================================================

TEST(PacketConceptsTest, HelperConceptsForBuilder) {
    using WithStreamId =
        DataPacket<PacketType::signal_data, NoClassId, NoTimestamp, Trailer::none, 64>;
    using NoStreamId =
        DataPacket<PacketType::signal_data_no_id, NoClassId, NoTimestamp, Trailer::none, 64>;
    using WithTrailer =
        DataPacket<PacketType::signal_data, NoClassId, NoTimestamp, Trailer::included, 64>;

    // Stream ID
    static_assert(HasStreamId<WithStreamId>);
    static_assert(!HasStreamId<NoStreamId>);

    // Trailer
    static_assert(HasMutableTrailer<WithTrailer>);
    static_assert(!HasMutableTrailer<NoStreamId>);

    // Payload
    static_assert(HasPayload<WithStreamId>);
    static_assert(HasPayload<NoStreamId>);

    // Packet count
    static_assert(HasPacketCount<WithStreamId>);
    static_assert(HasPacketCount<NoStreamId>);
}

// ============================================================================
// Non-packet types rejected
// ============================================================================

TEST(PacketConceptsTest, NonPacketTypesRejected) {
    static_assert(!PacketMetadataLike<int>);
    static_assert(!PacketMetadataLike<std::string>);
    static_assert(!PacketMetadataLike<std::vector<uint8_t>>);

    static_assert(!CompileTimePacketLike<int>);
    static_assert(!RuntimePacketLike<std::string>);
    static_assert(!DataPacketLike<std::vector<uint8_t>>);
    static_assert(!ContextPacketLike<void*>);
}

// ============================================================================
// Legacy aliases (for backwards compatibility during transition)
// ============================================================================

TEST(PacketConceptsTest, LegacyAliasesWork) {
    using DataPkt = DataPacket<PacketType::signal_data, NoClassId, NoTimestamp, Trailer::none, 64>;

    // PacketLike is alias for PacketMetadataLike
    static_assert(PacketLike<DataPkt>);
    static_assert(PacketLike<RuntimeDataPacket>);

    // AnyPacket is alias for PacketMetadataLike
    static_assert(AnyPacket<DataPkt>);
    static_assert(AnyPacket<RuntimeContextPacket>);
}

// ============================================================================
// Runtime behavior verification
// ============================================================================

TEST(PacketConceptsTest, RuntimeBehaviorConsistency) {
    // Create data packet
    using SignalType =
        DataPacket<PacketType::signal_data, NoClassId, NoTimestamp, Trailer::none, 32>;
    alignas(4) std::array<uint8_t, SignalType::size_bytes()> signal_buffer;
    SignalType signal_pkt(signal_buffer);

    EXPECT_NO_THROW({
        signal_pkt.set_stream_id(0x12345678);
        EXPECT_EQ(signal_pkt.stream_id(), 0x12345678);
        EXPECT_EQ(signal_pkt.as_bytes().data(), signal_buffer.data());
    });

    // Parse as runtime packet
    auto data_result = RuntimeDataPacket::parse(signal_buffer);
    ASSERT_TRUE(data_result.ok()) << data_result.error().message();
    const auto& view = data_result.value();
    EXPECT_NO_THROW({
        auto id = view.stream_id();
        ASSERT_TRUE(id.has_value());
        EXPECT_EQ(*id, 0x12345678);
    });

    // Create context packet
    using ContextType = ContextPacket<NoTimestamp, NoClassId, bandwidth>;
    alignas(4) std::array<uint8_t, ContextType::size_bytes()> context_buffer;
    ContextType context_pkt(context_buffer);

    EXPECT_NO_THROW({
        context_pkt.set_stream_id(0xABCDEF00);
        EXPECT_EQ(context_pkt.stream_id(), 0xABCDEF00);
    });

    // Parse as runtime context packet
    auto ctx_result = RuntimeContextPacket::parse(context_buffer);
    ASSERT_TRUE(ctx_result.ok()) << ctx_result.error().message();
    const auto& ctx_view = ctx_result.value();
    EXPECT_FALSE(ctx_view.change_indicator());
}
