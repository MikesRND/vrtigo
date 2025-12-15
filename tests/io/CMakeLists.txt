# I/O helper tests (always built)

set(TEST_DATA_PATH "${CMAKE_SOURCE_DIR}/tests/data")

vrtigo_add_gtest(file_reader_test file_reader_test.cpp TEST_DATA_DIR "${TEST_DATA_PATH}")
vrtigo_add_gtest(enhanced_reader_test enhanced_reader_test.cpp TEST_DATA_DIR "${TEST_DATA_PATH}")
vrtigo_add_gtest(pcap_reader_test pcap_reader_test.cpp)
vrtigo_add_gtest(pcap_writer_test pcap_writer_test.cpp)

# UDP reader tests (Linux/POSIX only)
if(UNIX)
    vrtigo_add_gtest(udp_reader_test udp_reader_test.cpp)
endif()

vrtigo_add_gtest(file_writer_test file_writer_test.cpp)

# UDP writer tests (Linux/POSIX only)
if(UNIX)
    vrtigo_add_gtest(udp_writer_test udp_writer_test.cpp)
endif()

vrtigo_add_gtest(sample_framer_test sample_framer_test.cpp)
