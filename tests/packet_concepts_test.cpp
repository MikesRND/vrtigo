#include <gtest/gtest.h>
#include <vrtio/core/packet_concepts.hpp>
#include <vrtio/packet/data_packet.hpp>
#include <vrtio/packet/data_packet_view.hpp>
#include <vrtio/packet/context_packet.hpp>
#include <vrtio/packet/context_packet_view.hpp>

using namespace vrtio;

// Test that SignalPacket satisfies FixedPacketLike concept
TEST(PacketConceptsTest, SignalPacketIsFixedPacketLike) {
    using PacketType = DataPacket<
        PacketType::SignalData,
        TimeStampUTC,
        Trailer::Included,
        64
    >;

    static_assert(PacketBase<PacketType>);
    static_assert(FixedPacketLike<PacketType>);
    static_assert(CompileTimePacket<PacketType>);
    static_assert(AnyPacketLike<PacketType>);

    // Should NOT satisfy variable packet concepts
    static_assert(!VariablePacketLike<PacketType>);
    static_assert(!RuntimePacketView<PacketType>);
}

// Test that SignalPacketView satisfies FixedPacketViewLike concept
TEST(PacketConceptsTest, SignalPacketViewIsFixedPacketViewLike) {
    static_assert(PacketBase<SignalPacketView>);
    static_assert(FixedPacketViewLike<SignalPacketView>);
    static_assert(RuntimePacketView<SignalPacketView>);
    static_assert(AnyPacketLike<SignalPacketView>);

    // Should NOT satisfy variable or compile-time concepts
    static_assert(!VariablePacketViewLike<SignalPacketView>);
    static_assert(!CompileTimePacket<SignalPacketView>);
}

// Test that ContextPacket satisfies VariablePacketLike concept
TEST(PacketConceptsTest, ContextPacketIsVariablePacketLike) {
    using PacketType = ContextPacket<
        true,                     // HasStreamId
        NoTimeStamp,
        NoClassId,
        cif0::BANDWIDTH | cif0::SAMPLE_RATE,
        0, 0, 0
    >;

    static_assert(PacketBase<PacketType>);
    static_assert(VariablePacketLike<PacketType>);
    static_assert(CompileTimePacket<PacketType>);
    static_assert(AnyPacketLike<PacketType>);

    // Should NOT satisfy fixed packet concepts
    static_assert(!FixedPacketLike<PacketType>);
    static_assert(!RuntimePacketView<PacketType>);
}

// Test that ContextPacketView satisfies VariablePacketViewLike concept
TEST(PacketConceptsTest, ContextPacketViewIsVariablePacketViewLike) {
    static_assert(PacketBase<ContextPacketView>);
    static_assert(VariablePacketViewLike<ContextPacketView>);
    static_assert(RuntimePacketView<ContextPacketView>);
    static_assert(AnyPacketLike<ContextPacketView>);

    // Should NOT satisfy fixed or compile-time concepts
    static_assert(!FixedPacketViewLike<ContextPacketView>);
    static_assert(!CompileTimePacket<ContextPacketView>);
}

// Test helper concepts for PacketBuilder
TEST(PacketConceptsTest, SignalPacketHelperConcepts) {
    using WithStreamId = DataPacket<PacketType::SignalData, NoTimeStamp, Trailer::None, 64>;
    using NoStreamId = DataPacket<PacketType::SignalDataNoId, NoTimeStamp, Trailer::None, 64>;
    using WithTimestamp = DataPacket<PacketType::SignalData, TimeStampUTC, Trailer::None, 64>;
    using WithTrailer = DataPacket<PacketType::SignalData, NoTimeStamp, Trailer::Included, 64>;

    // Stream ID
    static_assert(HasStreamId<WithStreamId>);
    static_assert(!HasStreamId<NoStreamId>);

    // Timestamp
    static_assert(HasTimestampInteger<WithTimestamp>);
    static_assert(HasTimestampFractional<WithTimestamp>);
    static_assert(!HasTimestampInteger<NoStreamId>);

    // Trailer
    static_assert(HasTrailer<WithTrailer>);
    static_assert(!HasTrailer<NoStreamId>);

    // Payload
    static_assert(HasPayload<WithStreamId>);
    static_assert(HasPayload<NoStreamId>);

    // Packet count
    static_assert(HasPacketCount<WithStreamId>);
    static_assert(HasPacketCount<NoStreamId>);
}

// Test that concepts properly distinguish packet categories
TEST(PacketConceptsTest, ConceptsMutuallyExclusive) {
    using SignalPkt = DataPacket<PacketType::SignalData, NoTimeStamp, Trailer::None, 64>;
    using ContextPkt = ContextPacket<true, NoTimeStamp, NoClassId, cif0::BANDWIDTH, 0, 0, 0>;

    // Signal is Fixed, not Variable
    static_assert(FixedPacketLike<SignalPkt>);
    static_assert(!VariablePacketLike<SignalPkt>);

    // Context is Variable, not Fixed
    static_assert(VariablePacketLike<ContextPkt>);
    static_assert(!FixedPacketLike<ContextPkt>);

    // Both are CompileTimePacket
    static_assert(CompileTimePacket<SignalPkt>);
    static_assert(CompileTimePacket<ContextPkt>);

    // Neither are RuntimePacketView
    static_assert(!RuntimePacketView<SignalPkt>);
    static_assert(!RuntimePacketView<ContextPkt>);
}

// Test that views are properly distinguished
TEST(PacketConceptsTest, ViewsMutuallyExclusive) {
    // SignalPacketView is FixedPacketViewLike, not VariablePacketViewLike
    static_assert(FixedPacketViewLike<SignalPacketView>);
    static_assert(!VariablePacketViewLike<SignalPacketView>);

    // ContextPacketView is VariablePacketViewLike, not FixedPacketViewLike
    static_assert(VariablePacketViewLike<ContextPacketView>);
    static_assert(!FixedPacketViewLike<ContextPacketView>);

    // Both are RuntimePacketView
    static_assert(RuntimePacketView<SignalPacketView>);
    static_assert(RuntimePacketView<ContextPacketView>);

    // Neither are CompileTimePacket
    static_assert(!CompileTimePacket<SignalPacketView>);
    static_assert(!CompileTimePacket<ContextPacketView>);
}

// Test that non-packet types are correctly rejected
TEST(PacketConceptsTest, NonPacketTypesRejected) {
    static_assert(!PacketBase<int>);
    static_assert(!PacketBase<std::string>);
    static_assert(!PacketBase<std::vector<uint8_t>>);

    static_assert(!FixedPacketLike<int>);
    static_assert(!VariablePacketLike<std::string>);
    static_assert(!RuntimePacketView<std::vector<uint8_t>>);

    static_assert(!AnyPacketLike<int>);
    static_assert(!AnyPacketLike<std::string>);
    static_assert(!AnyPacketLike<void*>);
}

// Runtime verification test (concepts are compile-time, but verify behavior)
TEST(PacketConceptsTest, RuntimeBehaviorConsistency) {
    // Create signal packet
    using SignalType = DataPacket<PacketType::SignalData, NoTimeStamp, Trailer::None, 32>;
    alignas(4) std::array<uint8_t, SignalType::size_bytes> signal_buffer;
    SignalType signal_pkt(signal_buffer.data());

    // Verify it has expected methods (concepts checked at compile-time)
    EXPECT_NO_THROW({
        signal_pkt.set_stream_id(0x12345678);
        EXPECT_EQ(signal_pkt.stream_id(), 0x12345678);
        EXPECT_EQ(signal_pkt.raw_bytes(), signal_buffer.data());
        auto err = signal_pkt.validate(signal_buffer.size());
        EXPECT_EQ(err, validation_error::none);
    });

    // Create signal packet view
    SignalPacketView view(signal_buffer.data(), signal_buffer.size());
    EXPECT_NO_THROW({
        EXPECT_TRUE(view.is_valid());
        EXPECT_EQ(view.error(), validation_error::none);
        auto id = view.stream_id();
        ASSERT_TRUE(id.has_value());
        EXPECT_EQ(*id, 0x12345678);
    });

    // Create context packet
    using ContextType = ContextPacket<true, NoTimeStamp, NoClassId, cif0::BANDWIDTH, 0, 0, 0>;
    alignas(4) std::array<uint8_t, ContextType::size_bytes> context_buffer;
    ContextType context_pkt(context_buffer.data());

    EXPECT_NO_THROW({
        context_pkt.set_stream_id(0xABCDEF00);
        EXPECT_EQ(context_pkt.stream_id(), 0xABCDEF00);
        EXPECT_EQ(context_pkt.cif0(), ContextType::cif0_value);
        EXPECT_NE(context_pkt.context_buffer(), nullptr);
    });

    // Create context packet view
    ContextPacketView ctx_view(context_buffer.data(), context_buffer.size());
    EXPECT_NO_THROW({
        auto err = ctx_view.error();
        EXPECT_EQ(err, validation_error::none);
        EXPECT_TRUE(ctx_view.is_valid());
        EXPECT_NE(ctx_view.context_buffer(), nullptr);
    });
}
