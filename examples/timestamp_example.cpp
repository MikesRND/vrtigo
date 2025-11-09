#include <vrtio.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <ctime>

using namespace vrtio;

// Helper function to print timestamp details
void printTimeStamp(const TimeStampUTC& ts, const std::string& label) {
    std::cout << label << ":\n";
    std::cout << "  Seconds: " << ts.seconds() << "\n";
    std::cout << "  Picoseconds: " << ts.fractional() << "\n";

    // Convert to human-readable time
    std::time_t time = ts.toTimeT();
    std::tm* tm_info = std::gmtime(&time);
    std::cout << "  UTC time: " << std::put_time(tm_info, "%Y-%m-%d %H:%M:%S");

    // Add sub-second precision
    uint64_t microseconds = ts.fractional() / 1'000'000;
    std::cout << "." << std::setfill('0') << std::setw(6) << microseconds << " UTC\n";
    std::cout << std::endl;
}

int main() {
    std::cout << "VRTIO TimeStamp Examples\n";
    std::cout << "=======================\n\n";

    // Example 1: Creating timestamps
    std::cout << "1. Creating TimeStamps\n";
    std::cout << "----------------------\n";

    // From current time
    auto ts_now = TimeStampUTC::now();
    printTimeStamp(ts_now, "Current time");

    // From UTC seconds only
    auto ts_seconds = TimeStampUTC::fromUTCSeconds(1699000000);
    printTimeStamp(ts_seconds, "From UTC seconds (1699000000)");

    // From components (seconds + picoseconds)
    auto ts_components = TimeStampUTC::fromComponents(
        1699000000,
        123'456'789'012ULL  // 123.456789012 microseconds
    );
    printTimeStamp(ts_components, "From components");

    // From std::chrono
    auto sys_time = std::chrono::system_clock::now();
    auto ts_chrono = TimeStampUTC::fromChrono(sys_time);
    printTimeStamp(ts_chrono, "From std::chrono::system_clock");

    // Example 2: Using timestamps with packets
    std::cout << "2. Using TimeStamps with Packets\n";
    std::cout << "---------------------------------\n";

    // Define packet type with UTC timestamps and real-time picoseconds
    using PacketType = SignalDataPacket<
        TimeStampUTC,
        Trailer::None,  // No trailer
        256     // payload words
    >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Create packet using builder with unified timestamp
    auto packet = PacketBuilder<PacketType>(buffer.data())
        .stream_id(0x12345678)
        .timestamp(ts_now)
        .build();

    std::cout << "Created packet with timestamp:\n";
    std::cout << "  Stream ID: 0x" << std::hex << packet.stream_id() << std::dec << "\n";

    auto read_ts = packet.getTimeStamp();
    printTimeStamp(read_ts, "  Packet timestamp");

    // Example 3: Timestamp arithmetic
    std::cout << "3. TimeStamp Arithmetic\n";
    std::cout << "-----------------------\n";

    auto ts_base = TimeStampUTC::fromUTCSeconds(1700000000);
    printTimeStamp(ts_base, "Base timestamp");

    // Add duration
    auto ts_plus_1ms = ts_base + std::chrono::milliseconds(1);
    printTimeStamp(ts_plus_1ms, "Base + 1 millisecond");

    auto ts_plus_1s = ts_base + std::chrono::seconds(1);
    printTimeStamp(ts_plus_1s, "Base + 1 second");

    // Subtract duration
    auto ts_minus_500us = ts_base - std::chrono::microseconds(500);
    printTimeStamp(ts_minus_500us, "Base - 500 microseconds");

    // Difference between timestamps
    auto duration = ts_plus_1s - ts_base;
    std::cout << "Difference between (Base + 1s) and Base: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
              << " milliseconds\n\n";

    // Example 4: Timestamp comparisons
    std::cout << "4. TimeStamp Comparisons\n";
    std::cout << "------------------------\n";

    auto ts1 = TimeStampUTC::fromUTCSeconds(1700000000);
    auto ts2 = TimeStampUTC::fromUTCSeconds(1700000001);
    auto ts3 = TimeStampUTC::fromComponents(1700000000, 500'000'000'000ULL);

    std::cout << "ts1 (1700000000s): " << ts1.seconds() << "s\n";
    std::cout << "ts2 (1700000001s): " << ts2.seconds() << "s\n";
    std::cout << "ts3 (1700000000s + 500ms): " << ts3.seconds() << "s + "
              << ts3.fractional() / 1'000'000'000ULL << "ms\n";

    std::cout << "ts1 < ts2: " << (ts1 < ts2 ? "true" : "false") << "\n";
    std::cout << "ts1 < ts3: " << (ts1 < ts3 ? "true" : "false") << "\n";
    std::cout << "ts2 > ts3: " << (ts2 > ts3 ? "true" : "false") << "\n";
    std::cout << "ts1 == ts1: " << (ts1 == ts1 ? "true" : "false") << "\n\n";

    // Example 5: Precision demonstration
    std::cout << "5. Precision Demonstration\n";
    std::cout << "--------------------------\n";

    // Create timestamp with picosecond precision
    auto ts_precise = TimeStampUTC::fromComponents(
        1700000000,
        123'456'789'012ULL  // Exactly 123,456,789,012 picoseconds
    );

    std::cout << "Original timestamp:\n";
    std::cout << "  Picoseconds: " << ts_precise.fractional() << " ps\n";
    std::cout << "  = " << ts_precise.fractional() / 1000ULL << " nanoseconds\n";
    std::cout << "  = " << ts_precise.fractional() / 1'000'000ULL << " microseconds\n";

    // Convert to chrono (loses sub-nanosecond precision)
    auto chrono_time = ts_precise.toChrono();
    auto ts_from_chrono = TimeStampUTC::fromChrono(chrono_time);

    std::cout << "\nAfter chrono round-trip:\n";
    std::cout << "  Picoseconds: " << ts_from_chrono.fractional() << " ps\n";
    std::cout << "  Lost precision: "
              << (ts_precise.fractional() - ts_from_chrono.fractional()) << " ps\n\n";

    // Example 6: GPS Timestamps using TimeStamp<gps, real_time>
    std::cout << "6. GPS TimeStamps with Typed API\n";
    std::cout << "--------------------------------\n";

    // Use TimeStamp<gps, real_time> to configure packet structure correctly
    // This sets TSI=2 (GPS) and TSF=2 (real_time) in the packet header
    using GPSPacket = SignalDataPacket<
        TimeStamp<tsi_type::gps, tsf_type::real_time>,  // GPS timestamp configuration
        Trailer::None,  // No trailer
        256
    >;

    std::cout << "GPS Packet Configuration:\n";
    std::cout << "  TSI type: GPS (value = 2)\n";
    std::cout << "  TSF type: real_time (value = 2)\n";
    std::cout << "  Packet has TSI field: " << (GPSPacket::tsi_type_v != tsi_type::none ? "yes" : "no") << "\n";
    std::cout << "  Packet has TSF field: " << (GPSPacket::tsf_type_v != tsf_type::none ? "yes" : "no") << "\n\n";

    // Create packet with GPS timestamps
    alignas(4) std::array<uint8_t, GPSPacket::size_bytes> gps_buffer{};
    GPSPacket gps_packet(gps_buffer.data());

    // GPS timestamp values (domain-specific)
    uint32_t gps_seconds = 1234567890;  // GPS seconds since Jan 6, 1980
    uint64_t gps_picoseconds = 500'000'000'000ULL;  // 0.5 seconds

    // Create GPS timestamp and set it using typed API
    using GPSTimeStamp = TimeStamp<tsi_type::gps, tsf_type::real_time>;
    GPSTimeStamp gps_ts(gps_seconds, gps_picoseconds);
    gps_packet.setTimeStamp(gps_ts);

    std::cout << "Setting GPS timestamp values:\n";
    std::cout << "  GPS seconds: " << gps_seconds << " (since Jan 6, 1980)\n";
    std::cout << "  GPS picoseconds: " << gps_picoseconds << " (0.5 seconds)\n\n";

    // Read back using typed API
    auto gps_read_ts = gps_packet.getTimeStamp();
    std::cout << "Reading back with getTimeStamp():\n";
    std::cout << "  TSI (seconds): " << gps_read_ts.seconds() << "\n";
    std::cout << "  TSF (picoseconds): " << gps_read_ts.fractional() << "\n\n";

    // You can also set the stream ID and other fields as normal
    gps_packet.set_stream_id(0x6B512345);
    gps_packet.set_packet_count(7);

    std::cout << "Other packet fields work normally:\n";
    std::cout << "  Stream ID: 0x" << std::hex << gps_packet.stream_id() << std::dec << "\n";
    std::cout << "  Packet count: " << static_cast<int>(gps_packet.packet_count()) << "\n\n";

    // Builder pattern also works with GPS timestamps
    std::cout << "Using PacketBuilder with GPS timestamps:\n";
    alignas(4) std::array<uint8_t, GPSPacket::size_bytes> builder_buffer{};
    auto gps_ts_builder = GPSTimeStamp::fromComponents(987654321, 123456789012);
    auto built_packet = PacketBuilder<GPSPacket>(builder_buffer.data())
        .stream_id(0xABCD1234)
        .timestamp(gps_ts_builder)
        .packet_count(15)
        .build();

    auto built_ts = built_packet.getTimeStamp();
    std::cout << "  Built packet TSI: " << built_ts.seconds() << "\n";
    std::cout << "  Built packet TSF: " << built_ts.fractional() << "\n\n";

    // Important notes about GPS timestamps
    std::cout << "Important GPS timestamp notes:\n";
    std::cout << "  - GPS epoch: Jan 6, 1980 00:00:00\n";
    std::cout << "  - UTC epoch: Jan 1, 1970 00:00:00\n";
    std::cout << "  - GPS leads UTC by ~18 seconds (as of 2024)\n";
    std::cout << "  - GPS-to-UTC conversion requires leap second tables\n";
    std::cout << "  - No automatic conversions provided by the library\n\n";

    // Example 7: Other timestamp types (e.g., TAI - International Atomic Time)
    std::cout << "7. Other TimeStamp Types (TAI Example)\n";
    std::cout << "---------------------------------------\n";

    // TAI and other non-standard timestamps use tsi_type::other
    // The specific time reference is application-defined
    using TAIPacket = SignalDataPacket<
        TimeStamp<tsi_type::other, tsf_type::real_time>,  // "Other" TSI for TAI
        Trailer::None,  // No trailer
        128
    >;

    std::cout << "TAI Packet Configuration:\n";
    std::cout << "  TSI type: other (value = " << static_cast<int>(tsi_type::other) << ")\n";
    std::cout << "  TSF type: real_time (value = 2)\n\n";

    alignas(4) std::array<uint8_t, TAIPacket::size_bytes> tai_buffer{};
    TAIPacket tai_packet(tai_buffer.data());

    // TAI is ahead of UTC by 37 seconds (as of 2024)
    uint32_t tai_seconds = 1699000037;  // Example: UTC + 37 seconds
    auto tai_ts = TimeStamp<tsi_type::other, tsf_type::real_time>::fromComponents(tai_seconds, 0);
    tai_packet.setTimeStamp(tai_ts);

    std::cout << "TAI timestamp example:\n";
    std::cout << "  TAI seconds: " << tai_packet.getTimeStamp().seconds() << "\n";
    std::cout << "  TAI = UTC + 37 seconds (as of 2024)\n";
    std::cout << "  No leap seconds in TAI (continuous timescale)\n";
    std::cout << "  Note: TAI uses TSI type 'other' (3) in VRT standard\n\n";

    // Example 8: Real-time timestamp updates
    std::cout << "8. Real-time Updates (showing time progression)\n";
    std::cout << "------------------------------------------------\n";

    for (int i = 0; i < 3; ++i) {
        auto ts = TimeStampUTC::now();
        std::cout << "Update " << (i + 1) << ": "
                  << ts.seconds() << "s + "
                  << ts.fractional() / 1'000'000'000ULL << "ms\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\nExample completed successfully!\n";

    return 0;
}
