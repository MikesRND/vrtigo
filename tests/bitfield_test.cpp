// Simple test to verify the bitfield API compiles and works
#include "vrtigo/detail/bitfield.hpp"

#include <array>
#include <iomanip>
#include <iostream>

#include <cstring>

int main() {
    using namespace vrtigo::detail;

    // Test 1: Basic field operations
    {
        std::cout << "Test 1: Basic field operations\n";

        // Define a simple layout
        using TestField1 = BitField<uint32_t, 0, 8>;   // bits 7-0
        using TestField2 = BitField<uint32_t, 8, 8>;   // bits 15-8
        using TestField3 = BitField<uint32_t, 16, 16>; // bits 31-16
        using TestLayout = BitFieldLayout<TestField1, TestField2, TestField3>;

        std::array<std::byte, 4> buffer{};
        BitFieldAccessor<TestLayout, false> accessor(buffer); // Little endian for test

        // Set values
        accessor.set<TestField1>(0x12);
        accessor.set<TestField2>(0x34);
        accessor.set<TestField3>(0x5678);

        // Read back
        std::cout << "  Field1: 0x" << std::hex << static_cast<int>(accessor.get<TestField1>())
                  << "\n";
        std::cout << "  Field2: 0x" << std::hex << static_cast<int>(accessor.get<TestField2>())
                  << "\n";
        std::cout << "  Field3: 0x" << std::hex << accessor.get<TestField3>() << "\n";

        // Check raw buffer
        uint32_t raw_value;
        std::memcpy(&raw_value, buffer.data(), sizeof(raw_value));
        std::cout << "  Raw buffer: 0x" << std::hex << raw_value << "\n";
        std::cout << "  Expected:   0x56783412\n\n";
    }

    // Test 2: VRT Header example (removed - used bitfield_example.hpp functions)

    // Test 3: Single bit flags
    {
        std::cout << "Test 3: Single bit flags\n";

        using Flag0 = BitFlag<0>;
        using Flag7 = BitFlag<7>;
        using Flag15 = BitFlag<15>;
        using Flag31 = BitFlag<31>;
        using FlagLayout = BitFieldLayout<Flag0, Flag7, Flag15, Flag31>;

        std::array<std::byte, 4> flag_buffer{};
        BitFieldAccessor<FlagLayout, false> accessor(flag_buffer);

        accessor.set<Flag0>(true);
        accessor.set<Flag7>(true);
        accessor.set<Flag31>(true);

        std::cout << "  Flag0: " << accessor.get<Flag0>() << "\n";
        std::cout << "  Flag7: " << accessor.get<Flag7>() << "\n";
        std::cout << "  Flag15: " << accessor.get<Flag15>() << "\n";
        std::cout << "  Flag31: " << accessor.get<Flag31>() << "\n";

        uint32_t raw_flags;
        std::memcpy(&raw_flags, flag_buffer.data(), sizeof(raw_flags));
        std::cout << "  Raw flags: 0x" << std::hex << raw_flags << "\n";
        std::cout << "  Expected:  0x80000081\n\n";
    }

    // Test 4: Multi-word layout (VRT-like structure)
    {
        std::cout << "Test 4: Multi-word layout\n";

        // Define a 4-word structure similar to VRT header
        // Word 0: Header fields
        using PacketType = BitField<uint32_t, 28, 4, 0>;  // Word 0, bits 31-28
        using HasStreamId = BitField<uint32_t, 27, 1, 0>; // Word 0, bit 27
        using HasClassId = BitField<uint32_t, 26, 1, 0>;  // Word 0, bit 26
        using PacketSize = BitField<uint32_t, 0, 16, 0>;  // Word 0, bits 15-0

        // Word 1: Stream ID
        using StreamId = BitField<uint32_t, 0, 32, 1>; // Word 1, all bits

        // Word 2-3: Class ID (64-bit)
        using ClassOUI = BitField<uint32_t, 8, 24, 2>;  // Word 2, bits 31-8
        using ClassICC = BitField<uint32_t, 16, 16, 3>; // Word 3, bits 31-16
        using ClassPCC = BitField<uint32_t, 0, 16, 3>;  // Word 3, bits 15-0

        using MultiWordLayout = BitFieldLayout<PacketType, HasStreamId, HasClassId, PacketSize,
                                               StreamId, ClassOUI, ClassICC, ClassPCC>;

        // Check that layout correctly calculates 4 words needed
        static_assert(MultiWordLayout::num_words == 4, "Should need 4 words");

        // Create a 16-byte (4-word) buffer
        std::array<std::byte, 16> multi_buffer{};
        BitFieldAccessor<MultiWordLayout, true> multi_accessor(multi_buffer);

        // Set fields in different words
        multi_accessor.set<PacketType>(0x4);      // Word 0
        multi_accessor.set<HasStreamId>(true);    // Word 0
        multi_accessor.set<HasClassId>(true);     // Word 0
        multi_accessor.set<PacketSize>(1234);     // Word 0
        multi_accessor.set<StreamId>(0x12345678); // Word 1
        multi_accessor.set<ClassOUI>(0xABCDEF);   // Word 2
        multi_accessor.set<ClassICC>(0x1234);     // Word 3
        multi_accessor.set<ClassPCC>(0x5678);     // Word 3

        // Read back and verify
        std::cout << "  PacketType: " << std::hex
                  << static_cast<int>(multi_accessor.get<PacketType>()) << "\n";
        std::cout << "  StreamId: 0x" << std::hex << multi_accessor.get<StreamId>() << "\n";
        std::cout << "  ClassOUI: 0x" << std::hex << multi_accessor.get<ClassOUI>() << "\n";
        std::cout << "  ClassICC: 0x" << std::hex << multi_accessor.get<ClassICC>() << "\n";
        std::cout << "  ClassPCC: 0x" << std::hex << multi_accessor.get<ClassPCC>() << "\n";

        // Verify values
        assert(multi_accessor.get<PacketType>() == 0x4);
        assert(multi_accessor.get<HasStreamId>() == true);
        assert(multi_accessor.get<StreamId>() == 0x12345678);
        assert(multi_accessor.get<ClassOUI>() == 0xABCDEF);

        std::cout << "  Multi-word test passed!\n\n";
    }

    // Test 5: Full-width fields (no UB shift)
    {
        std::cout << "Test 5: Full-width fields\n";

        // Test that 64-bit full-width field works (was UB before fix)
        using FullWidth64_0 = BitField<uint64_t, 0, 64, 0>; // Bytes 0-7
        using FullWidth64_1 = BitField<uint64_t, 0, 64, 2>; // Bytes 8-15 (word 2, not 1!)
        using FullWidthLayout = BitFieldLayout<FullWidth64_0, FullWidth64_1>;

        std::array<std::byte, 16> full_buffer{}; // 2 * 8 bytes
        BitFieldAccessor<FullWidthLayout> full_accessor(full_buffer);

        full_accessor.set<FullWidth64_0>(0x123456789ABCDEF0ULL);
        full_accessor.set<FullWidth64_1>(0xDEADBEEFCAFEBABEULL);

        auto val64_0 = full_accessor.get<FullWidth64_0>();
        auto val64_1 = full_accessor.get<FullWidth64_1>();

        std::cout << "  64-bit field 0: 0x" << std::hex << val64_0 << "\n";
        std::cout << "  64-bit field 1: 0x" << std::hex << val64_1 << "\n";

        assert(val64_0 == 0x123456789ABCDEF0ULL);
        assert(val64_1 == 0xDEADBEEFCAFEBABEULL);

        // Also test 32-bit full-width (different layout)
        using FullWidth32 = BitField<uint32_t, 0, 32, 0>;
        using Layout32 = BitFieldLayout<FullWidth32>;
        std::array<std::byte, 4> buffer32{};
        BitFieldAccessor<Layout32> accessor32(buffer32);
        accessor32.set<FullWidth32>(0xDEADBEEF);
        assert(accessor32.get<FullWidth32>() == 0xDEADBEEF);

        std::cout << "  Full-width test passed!\n\n";
    }

    // Test 6: Runtime validation-as-data (dynamic::DataPacketView style)
    {
        std::cout << "Test 6: Runtime validation-as-data\n";

        using TestField = BitField<uint32_t, 0, 32, 0>;
        using TestLayout = BitFieldLayout<TestField>;

        // Correct size buffer - operations return valid optionals
        std::array<std::byte, 4> good_buffer{};
        std::span<std::byte> good_span(good_buffer);
        RuntimeBitFieldAccessor<TestLayout> good_accessor(good_span);

        assert(good_accessor.is_valid());
        assert(good_accessor.valid()); // Alias works too

        good_accessor.set<TestField>(0x12345678);
        [[maybe_unused]] auto value = good_accessor.get<TestField>();
        assert(value.has_value());
        assert(*value == 0x12345678);
        std::cout << "  Valid buffer: get() returns optional with value\n";

        // Too small buffer - validation state is data, operations return nullopt
        std::array<std::byte, 2> small_buffer{};
        std::span<std::byte> small_span(small_buffer);
        RuntimeBitFieldAccessor<TestLayout> small_accessor(small_span);

        assert(!small_accessor.is_valid());
        assert(!small_accessor.valid());

        // get() returns nullopt on invalid buffer (safe, no UB)
        [[maybe_unused]] auto invalid_value = small_accessor.get<TestField>();
        assert(!invalid_value.has_value());
        std::cout << "  Invalid buffer: get() returns nullopt\n";

        // set() is no-op on invalid buffer (safe, no UB)
        small_accessor.set<TestField>(0xDEADBEEF);
        [[maybe_unused]] auto still_invalid = small_accessor.get<TestField>();
        assert(!still_invalid.has_value());
        std::cout << "  Invalid buffer: set() is no-op\n";

        // get_multiple() returns tuple of optionals
        using Field1 = BitField<uint32_t, 0, 16, 0>;
        using Field2 = BitField<uint32_t, 16, 16, 0>;
        using MultiLayout = BitFieldLayout<Field1, Field2>;

        std::array<std::byte, 4> multi_buffer{};
        RuntimeBitFieldAccessor<MultiLayout> multi_accessor(multi_buffer);
        multi_accessor.set<Field1>(0x1234);
        multi_accessor.set<Field2>(0x5678);

        [[maybe_unused]] auto [val1, val2] = multi_accessor.get_multiple<Field1, Field2>();
        assert(val1.has_value() && *val1 == 0x1234);
        assert(val2.has_value() && *val2 == 0x5678);
        std::cout << "  get_multiple() returns tuple of optionals\n";

        std::cout << "  Runtime validation-as-data works! (dynamic::DataPacketView style)\n\n";
    }

    // Test 7: Mixed storage types with VRT 32-bit word alignment
    {
        std::cout << "Test 7: Mixed storage types\n";

        // uint64_t field at word 0 spans bytes 0-7 (VRT words 0-1)
        // uint32_t field at word 2 occupies bytes 8-11 (VRT word 2)
        // These don't overlap, so the layout is valid
        using Field64 = BitField<uint64_t, 0, 64, 0>; // Bytes 0-7
        using Field32 = BitField<uint32_t, 0, 32, 2>; // Bytes 8-11
        using MixedLayout = BitFieldLayout<Field64, Field32>;

        static_assert(MixedLayout::word_size == 4, "VRT uses 32-bit words");
        static_assert(MixedLayout::required_bytes == 12, "Need 12 bytes (0-11)");
        static_assert(MixedLayout::num_words == 3, "Need 3 VRT words (ceil(12/4))");

        std::array<std::byte, 12> mixed_buffer{};
        BitFieldAccessor<MixedLayout, true> mixed_accessor(mixed_buffer);

        mixed_accessor.set<Field64>(0x123456789ABCDEF0ULL);
        mixed_accessor.set<Field32>(0xDEADBEEF);

        assert(mixed_accessor.get<Field64>() == 0x123456789ABCDEF0ULL);
        assert(mixed_accessor.get<Field32>() == 0xDEADBEEF);

        std::cout << "  Mixed uint64_t and uint32_t fields work correctly\n";
        std::cout << "  64-bit field: 0x" << std::hex << mixed_accessor.get<Field64>() << "\n";
        std::cout << "  32-bit field: 0x" << std::hex << mixed_accessor.get<Field32>() << "\n";
        std::cout << "  num_words = " << std::dec << MixedLayout::num_words << " (correct)\n\n";
    }

    // Test 8: Multi-word spanning field at non-zero word_index
    {
        std::cout << "Test 8: Multi-word spanning field\n";

        // uint64_t at word_index=1 spans bytes 4-11
        // This should require 3 VRT words total (words 0, 1, 2)
        using Field64AtWord1 = BitField<uint64_t, 0, 64, 1>; // Bytes 4-11
        using SpanLayout = BitFieldLayout<Field64AtWord1>;

        static_assert(SpanLayout::required_bytes == 12, "Bytes 0-11 needed");
        static_assert(SpanLayout::num_words == 3, "3 VRT words (0,1,2)");

        std::array<std::byte, 12> span_buffer{};
        BitFieldAccessor<SpanLayout, true> span_accessor(span_buffer);

        span_accessor.set<Field64AtWord1>(0xCAFEBABEDEADBEEFULL);
        assert(span_accessor.get<Field64AtWord1>() == 0xCAFEBABEDEADBEEFULL);

        std::cout << "  uint64_t at word_index=1 works correctly\n";
        std::cout << "  Value: 0x" << std::hex << span_accessor.get<Field64AtWord1>() << "\n";
        std::cout << "  required_bytes = " << std::dec << SpanLayout::required_bytes << "\n";
        std::cout << "  num_words = " << SpanLayout::num_words << " (correctly spans 3 words)\n\n";
    }

    // Test 9: Compile-time validation (uncomment to test)
    /*
    {
        // This should fail at compile time - overlapping fields in same word (bit-level)
        using BadField1 = BitField<uint32_t, 0, 8, 0>;
        using BadField2 = BitField<uint32_t, 4, 8, 0>;  // Overlaps with BadField1 in word 0
        using BadLayout = BitFieldLayout<BadField1, BadField2>;
        // Compilation error: "BitField layout contains overlapping fields"
    }

    {
        // This should fail - overlapping byte ranges (multi-word spanning)
        using Field64 = BitField<uint64_t, 0, 64, 0>;   // Bytes 0-7 (words 0-1)
        using Field32 = BitField<uint32_t, 0, 32, 1>;   // Bytes 4-7 (word 1) - OVERLAPS!
        using BadLayout = BitFieldLayout<Field64, Field32>;
        // Compilation error: "BitField layout contains overlapping fields"
        // uint64_t at word 0 spans bytes 0-7, uint32_t at word 1 spans bytes 4-7
    }

    {
        // This should be OK - same bit positions but different words, no byte overlap
        using OkField1 = BitField<uint32_t, 0, 8, 0>;   // Word 0, bytes 0-3
        using OkField2 = BitField<uint32_t, 0, 8, 1>;   // Word 1, bytes 4-7 - OK!
        using OkLayout = BitFieldLayout<OkField1, OkField2>;
    }
    */

    std::cout << "All tests completed successfully!\n";
    return 0;
}