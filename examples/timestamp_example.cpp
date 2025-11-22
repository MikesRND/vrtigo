#include <chrono>
#include <iomanip>
#include <iostream>
#include <thread>

#include <ctime>
#include <vrtigo.hpp>

using namespace vrtigo;

// Helper function to print timestamp details
void printTimestamp(const UtcRealTimestamp& ts, const std::string& label) {
    std::cout << label << ":\n";
    std::cout << "  Seconds: " << ts.tsi() << "\n";
    std::cout << "  Picoseconds: " << ts.tsf() << "\n";

    // Convert to human-readable time
    std::time_t time = ts.to_time_t();
    std::tm* tm_info = std::gmtime(&time);
    std::cout << "  UTC time: " << std::put_time(tm_info, "%Y-%m-%d %H:%M:%S");

    // Add sub-second precision
    uint64_t microseconds = ts.tsf() / 1'000'000;
    std::cout << "." << std::setfill('0') << std::setw(6) << microseconds << " UTC\n";
    std::cout << std::endl;
}

int main() {
    std::cout << "VRTIGO Timestamp Examples\n";
    std::cout << "=======================\n\n";

    // Example 1: Creating timestamps
    std::cout << "1. Creating Timestamps\n";
    std::cout << "----------------------\n";

    // From current time
    auto ts_now = UtcRealTimestamp::now();
    printTimestamp(ts_now, "Current time");

    // From UTC seconds only
    auto ts_seconds = UtcRealTimestamp::from_utc_seconds(1699000000);
    printTimestamp(ts_seconds, "From UTC seconds (1699000000)");

    // From components (seconds + picoseconds)
    auto ts_components = UtcRealTimestamp(1699000000,
                                          123'456'789'012ULL // 123.456789012 microseconds
    );
    printTimestamp(ts_components, "From components");

    // From std::chrono
    auto sys_time = std::chrono::system_clock::now();
    auto ts_chrono = UtcRealTimestamp::from_chrono(sys_time);
    printTimestamp(ts_chrono, "From std::chrono::system_clock");

    // Example 2: Using timestamps with packets
    std::cout << "2. Using Timestamps with Packets\n";
    std::cout << "---------------------------------\n";

    // Define packet type with UTC timestamps and real-time picoseconds
    using PacketType = SignalDataPacket<vrtigo::NoClassId, UtcRealTimestamp,
                                        Trailer::none, // No trailer
                                        256            // payload words
                                        >;

    alignas(4) std::array<uint8_t, PacketType::size_bytes> buffer{};

    // Create packet using builder with unified timestamp
    auto packet =
        PacketBuilder<PacketType>(buffer.data()).stream_id(0x12345678).timestamp(ts_now).build();

    std::cout << "Created packet with timestamp:\n";
    std::cout << "  Stream ID: 0x" << std::hex << packet.stream_id() << std::dec << "\n";

    auto read_ts = packet.timestamp();
    printTimestamp(read_ts, "  Packet timestamp");

    // Example 3: Timestamp arithmetic
    std::cout << "3. Timestamp Arithmetic\n";
    std::cout << "-----------------------\n";

    auto ts_base = UtcRealTimestamp::from_utc_seconds(1700000000);
    printTimestamp(ts_base, "Base timestamp");

    // Add duration
    auto ts_plus_1ms = ts_base + std::chrono::milliseconds(1);
    printTimestamp(ts_plus_1ms, "Base + 1 millisecond");

    auto ts_plus_1s = ts_base + std::chrono::seconds(1);
    printTimestamp(ts_plus_1s, "Base + 1 second");

    // Subtract duration
    auto ts_minus_500us = ts_base - std::chrono::microseconds(500);
    printTimestamp(ts_minus_500us, "Base - 500 microseconds");

    // Difference between timestamps
    auto duration = ts_plus_1s - ts_base;
    std::cout << "Difference between (Base + 1s) and Base: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
              << " milliseconds\n\n";

    // Example 4: Timestamp comparisons
    std::cout << "4. Timestamp Comparisons\n";
    std::cout << "------------------------\n";

    auto ts1 = UtcRealTimestamp::from_utc_seconds(1700000000);
    auto ts2 = UtcRealTimestamp::from_utc_seconds(1700000001);
    auto ts3 = UtcRealTimestamp(1700000000, 500'000'000'000ULL);

    std::cout << "ts1 (1700000000s): " << ts1.tsi() << "s\n";
    std::cout << "ts2 (1700000001s): " << ts2.tsi() << "s\n";
    std::cout << "ts3 (1700000000s + 500ms): " << ts3.tsi() << "s + "
              << ts3.tsf() / 1'000'000'000ULL << "ms\n";

    std::cout << "ts1 < ts2: " << (ts1 < ts2 ? "true" : "false") << "\n";
    std::cout << "ts1 < ts3: " << (ts1 < ts3 ? "true" : "false") << "\n";
    std::cout << "ts2 > ts3: " << (ts2 > ts3 ? "true" : "false") << "\n";
    std::cout << "ts1 == ts1: " << (ts1 == ts1 ? "true" : "false") << "\n\n";

    // Example 5: Precision demonstration
    std::cout << "5. Precision Demonstration\n";
    std::cout << "--------------------------\n";

    // Create timestamp with picosecond precision
    auto ts_precise = UtcRealTimestamp(1700000000,
                                       123'456'789'012ULL // Exactly 123,456,789,012 picoseconds
    );

    std::cout << "Original timestamp:\n";
    std::cout << "  Picoseconds: " << ts_precise.tsf() << " ps\n";
    std::cout << "  = " << ts_precise.tsf() / 1000ULL << " nanoseconds\n";
    std::cout << "  = " << ts_precise.tsf() / 1'000'000ULL << " microseconds\n";

    // Convert to chrono (loses sub-nanosecond precision)
    auto chrono_time = ts_precise.to_chrono();
    auto ts_from_chrono = UtcRealTimestamp::from_chrono(chrono_time);

    std::cout << "\nAfter chrono round-trip:\n";
    std::cout << "  Picoseconds: " << ts_from_chrono.tsf() << " ps\n";
    std::cout << "  Lost precision: " << (ts_precise.tsf() - ts_from_chrono.tsf()) << " ps\n\n";

    // Example 6: GPS Timestamps using Timestamp<gps, real_time>
    std::cout << "6. GPS Timestamps with Typed API\n";
    std::cout << "--------------------------------\n";

    // Use Timestamp<gps, real_time> to configure packet structure correctly
    // This sets TSI=2 (GPS) and TSF=2 (real_time) in the packet header
    using GPSPacket =
        SignalDataPacket<vrtigo::NoClassId,
                         Timestamp<TsiType::gps, TsfType::real_time>, // GPS timestamp configuration
                         Trailer::none,                               // No trailer
                         256>;

    std::cout << "GPS Packet Configuration:\n";
    std::cout << "  TSI type: GPS (value = 2)\n";
    std::cout << "  TSF type: real_time (value = 2)\n";
    std::cout << "  Packet has timestamp: " << (GPSPacket::has_timestamp ? "yes" : "no") << "\n\n";

    // Create packet with GPS timestamps
    alignas(4) std::array<uint8_t, GPSPacket::size_bytes> gps_buffer{};
    GPSPacket gps_packet(gps_buffer.data());

    // GPS timestamp values (domain-specific)
    uint32_t gps_seconds = 1234567890;             // GPS seconds since Jan 6, 1980
    uint64_t gps_picoseconds = 500'000'000'000ULL; // 0.5 seconds

    // Create GPS timestamp and set it using typed API
    using GPSTimestamp = Timestamp<TsiType::gps, TsfType::real_time>;
    GPSTimestamp gps_ts(gps_seconds, gps_picoseconds);
    gps_packet.set_timestamp(gps_ts);

    std::cout << "Setting GPS timestamp values:\n";
    std::cout << "  GPS seconds: " << gps_seconds << " (since Jan 6, 1980)\n";
    std::cout << "  GPS picoseconds: " << gps_picoseconds << " (0.5 seconds)\n\n";

    // Read back using typed API
    auto gps_read_ts = gps_packet.timestamp();
    std::cout << "Reading back with timestamp():\n";
    std::cout << "  TSI (seconds): " << gps_read_ts.tsi() << "\n";
    std::cout << "  TSF (picoseconds): " << gps_read_ts.tsf() << "\n\n";

    // You can also set the stream ID and other fields as normal
    gps_packet.set_stream_id(0x6B512345);
    gps_packet.set_packet_count(7);

    std::cout << "Other packet fields work normally:\n";
    std::cout << "  Stream ID: 0x" << std::hex << gps_packet.stream_id() << std::dec << "\n";
    std::cout << "  Packet count: " << static_cast<int>(gps_packet.packet_count()) << "\n\n";

    // Builder pattern also works with GPS timestamps
    std::cout << "Using PacketBuilder with GPS timestamps:\n";
    alignas(4) std::array<uint8_t, GPSPacket::size_bytes> builder_buffer{};
    auto gps_ts_builder = GPSTimestamp(987654321, 123456789012);
    auto built_packet = PacketBuilder<GPSPacket>(builder_buffer.data())
                            .stream_id(0xABCD1234)
                            .timestamp(gps_ts_builder)
                            .packet_count(15)
                            .build();

    auto built_ts = built_packet.timestamp();
    std::cout << "  Built packet TSI: " << built_ts.tsi() << "\n";
    std::cout << "  Built packet TSF: " << built_ts.tsf() << "\n\n";

    // Important notes about GPS timestamps
    std::cout << "Important GPS timestamp notes:\n";
    std::cout << "  - GPS epoch: Jan 6, 1980 00:00:00\n";
    std::cout << "  - UTC epoch: Jan 1, 1970 00:00:00\n";
    std::cout << "  - GPS leads UTC by ~18 seconds (as of 2024)\n";
    std::cout << "  - GPS-to-UTC conversion requires leap second tables\n";
    std::cout << "  - No automatic conversions provided by the library\n\n";

    // Example 7: Other timestamp types (e.g., TAI - International Atomic Time)
    std::cout << "7. Other Timestamp Types (TAI Example)\n";
    std::cout << "---------------------------------------\n";

    // TAI and other non-standard timestamps use TsiType::other
    // The specific time reference is application-defined
    using TAIPacket =
        SignalDataPacket<vrtigo::NoClassId,
                         Timestamp<TsiType::other, TsfType::real_time>, // "Other" TSI for TAI
                         Trailer::none,                                 // No trailer
                         128>;

    std::cout << "TAI Packet Configuration:\n";
    std::cout << "  TSI type: other (value = " << static_cast<int>(TsiType::other) << ")\n";
    std::cout << "  TSF type: real_time (value = 2)\n\n";

    alignas(4) std::array<uint8_t, TAIPacket::size_bytes> tai_buffer{};
    TAIPacket tai_packet(tai_buffer.data());

    // TAI is ahead of UTC by 37 seconds (as of 2024)
    uint32_t tai_seconds = 1699000037; // Example: UTC + 37 seconds
    auto tai_ts = Timestamp<TsiType::other, TsfType::real_time>(tai_seconds, 0);
    tai_packet.set_timestamp(tai_ts);

    std::cout << "TAI timestamp example:\n";
    std::cout << "  TAI seconds: " << tai_packet.timestamp().tsi() << "\n";
    std::cout << "  TAI = UTC + 37 seconds (as of 2024)\n";
    std::cout << "  No leap seconds in TAI (continuous timescale)\n";
    std::cout << "  Note: TAI uses TSI type 'other' (3) in VRT standard\n\n";

    // Example 8: Real-time timestamp updates
    std::cout << "8. Real-time Updates (showing time progression)\n";
    std::cout << "------------------------------------------------\n";

    for (int i = 0; i < 3; ++i) {
        auto ts = UtcRealTimestamp::now();
        std::cout << "Update " << (i + 1) << ": " << ts.tsi() << "s + "
                  << ts.tsf() / 1'000'000'000ULL << "ms\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\nExample completed successfully!\n";

    return 0;
}
