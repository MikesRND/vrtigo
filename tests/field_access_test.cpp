#include <iostream>

#include <cassert>
#include <vrtigo.hpp>

using namespace vrtigo;
using namespace vrtigo::field;

// =============================================================================
// Basic Field Access API Tests
// Tests for operator[], and make_field_proxy_unchecked() functions
// Detailed FieldProxy tests are in tests/cif/proxy_test.cpp
// =============================================================================

int main() {
    // Test packet with multiple fields using new field-based API
    // Note: Context packets always have Stream ID per VITA 49.2 spec
    using TestPacket = typed::ContextPacket<NoTimestamp,                   // TimestampType
                                            NoClassId,                     // ClassIdType
                                            bandwidth, sample_rate, gain>; // Fields

    alignas(4) std::array<uint8_t, TestPacket::size_bytes()> buffer{};
    TestPacket packet(buffer);

    // =======================================================================
    // Test set_value() and value() via proxy (interpreted values)
    // =======================================================================

    packet[bandwidth].set_value(1'000'000.0);   // 1 MHz
    packet[sample_rate].set_value(2'000'000.0); // 2 MSPS
    packet[gain].set_encoded(42U);              // Gain has no interpreted support

    // =======================================================================
    // Test operator[] function
    // =======================================================================

    [[maybe_unused]] auto bw = packet[bandwidth];
    assert(bw.has_value() && "Bandwidth should be present");
    assert(bw.value() == 1'000'000.0 && "Bandwidth value should match");

    [[maybe_unused]] auto sr = packet[sample_rate];
    assert(sr.has_value() && "Sample rate should be present");
    assert(sr.value() == 2'000'000.0 && "Sample rate value should match");

    [[maybe_unused]] auto g = packet[gain];
    assert(g.has_value() && "Gain should be present");
    assert(g.encoded() == 42U && "Gain value should match");

    // =======================================================================
    // Test operator[] for presence checking
    // =======================================================================

    assert(packet[bandwidth] && "operator[] should return true for bandwidth");
    assert(packet[sample_rate] && "operator[] should return true for sample_rate");
    assert(packet[gain] && "operator[] should return true for gain");
    assert(!packet[temperature] && "operator[] should return false for temperature");

    // =======================================================================
    // Test operator[] with missing field
    // =======================================================================

    [[maybe_unused]] auto temp = packet[temperature];
    assert(!temp.has_value() && "Temperature should not be present");

    // =======================================================================
    // Test make_field_proxy_unchecked() for known field (compile-time packets only)
    // make_field_proxy_unchecked() returns raw values (Q52.12 for bandwidth)
    // =======================================================================

    [[maybe_unused]] uint64_t bw_direct = make_field_proxy_unchecked(packet, bandwidth);
    // 1 MHz in Q52.12 format = 1'000'000 * 4096 = 4'096'000'000
    assert(bw_direct == 4'096'000'000ULL &&
           "make_field_proxy_unchecked should return correct raw value");

    std::cout << "✓ All field access API tests passed!\n";
    std::cout << "  - operator[] correctly identifies present/absent fields\n";
    std::cout << "  - operator[] returns FieldProxy with correct values\n";
    std::cout << "  - make_field_proxy_unchecked() provides zero-overhead access\n";
    std::cout << "\nNote: Detailed FieldProxy tests are in tests/cif/proxy_test.cpp\n";

    return 0;
}
