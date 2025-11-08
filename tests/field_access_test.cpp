#include <vrtio/fields.hpp>
#include <iostream>
#include <cassert>

using namespace vrtio;
using namespace vrtio::field;

int main() {
    // Test 1: Compile-time packet with fields
    using TestPacket = ContextPacket<
        true,                    // HasStreamId
        NoTimeStamp,             // TimeStampType
        NoClassId,               // ClassIdType
        (1U << 29) | (1U << 21) | (1U << 23),  // CIF0: bandwidth, sample_rate, gain
        0,                       // CIF1
        0,                       // CIF2
        false                    // HasTrailer
    >;

    uint8_t buffer[1024] = {};
    TestPacket packet(buffer);

    // Test set() function
    bool success = set(packet, bandwidth, 1'000'000ULL);
    assert(success && "Should be able to set bandwidth");

    success = set(packet, sample_rate, 2'000'000ULL);
    assert(success && "Should be able to set sample_rate");

    success = set(packet, gain, 42U);
    assert(success && "Should be able to set gain");

    // Test get() function
    auto bw = get(packet, bandwidth);
    assert(bw.has_value() && "Bandwidth should be present");
    assert(*bw == 1'000'000ULL && "Bandwidth value should match");

    auto sr = get(packet, sample_rate);
    assert(sr.has_value() && "Sample rate should be present");
    assert(*sr == 2'000'000ULL && "Sample rate value should match");

    auto g = get(packet, gain);
    assert(g.has_value() && "Gain should be present");
    assert(*g == 42U && "Gain value should match");

    // Test has() function
    assert(has(packet, bandwidth) && "has() should return true for bandwidth");
    assert(has(packet, sample_rate) && "has() should return true for sample_rate");
    assert(has(packet, gain) && "has() should return true for gain");
    assert(!has(packet, temperature) && "has() should return false for temperature");

    // Test get() with missing field
    auto temp = get(packet, temperature);
    assert(!temp.has_value() && "Temperature should not be present");

    // Test get_unchecked() for known field
    uint64_t bw_direct = get_unchecked(packet, bandwidth);
    assert(bw_direct == 1'000'000ULL && "get_unchecked should return correct value");

    std::cout << "✓ All field access API tests passed!\n";
    std::cout << "  - Bandwidth: " << *bw << " Hz\n";
    std::cout << "  - Sample Rate: " << *sr << " Hz\n";
    std::cout << "  - Gain: " << *g << "\n";

    return 0;
}
