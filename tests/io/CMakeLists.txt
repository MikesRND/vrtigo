# I/O helper tests (always built)

# File reader tests
add_executable(file_reader_test file_reader_test.cpp)
target_link_libraries(file_reader_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME file_reader_test COMMAND file_reader_test)

# Pass test data directory to tests
target_compile_definitions(file_reader_test PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data"
)

# Set C++20 standard
set_target_properties(file_reader_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Enhanced reader tests
add_executable(enhanced_reader_test enhanced_reader_test.cpp)
target_link_libraries(enhanced_reader_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME enhanced_reader_test COMMAND enhanced_reader_test)

# Pass test data directory to tests
target_compile_definitions(enhanced_reader_test PRIVATE
    TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data"
)

# Set C++20 standard
set_target_properties(enhanced_reader_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# PCAP reader tests
add_executable(pcap_reader_test pcap_reader_test.cpp)
target_link_libraries(pcap_reader_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME pcap_reader_test COMMAND pcap_reader_test)

# Set C++20 standard
set_target_properties(pcap_reader_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# PCAP writer tests
add_executable(pcap_writer_test pcap_writer_test.cpp)
target_link_libraries(pcap_writer_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME pcap_writer_test COMMAND pcap_writer_test)

# Set C++20 standard
set_target_properties(pcap_writer_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# UDP reader tests (Linux/POSIX only)
if(UNIX)
    add_executable(udp_reader_test udp_reader_test.cpp)
    target_link_libraries(udp_reader_test PRIVATE vrtio GTest::gtest_main)
    add_test(NAME udp_reader_test COMMAND udp_reader_test)

    set_target_properties(udp_reader_test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
endif()

# File writer tests
add_executable(file_writer_test file_writer_test.cpp)
target_link_libraries(file_writer_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME file_writer_test COMMAND file_writer_test)

set_target_properties(file_writer_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# UDP writer tests (Linux/POSIX only)
if(UNIX)
    add_executable(udp_writer_test udp_writer_test.cpp)
    target_link_libraries(udp_writer_test PRIVATE vrtio GTest::gtest_main)
    add_test(NAME udp_writer_test COMMAND udp_writer_test)

    set_target_properties(udp_writer_test PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
endif()
