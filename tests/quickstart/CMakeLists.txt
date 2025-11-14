# Quickstart tests - executable documentation snippets

# Create Data Packet snippet test
add_executable(quickstart_data_packet data_packet_test.cpp)
target_link_libraries(quickstart_data_packet PRIVATE vrtio GTest::gtest_main)
add_test(NAME quickstart_data_packet COMMAND quickstart_data_packet)

# Create Context Packet snippet test
add_executable(quickstart_context_packet context_packet_test.cpp)
target_link_libraries(quickstart_context_packet PRIVATE vrtio GTest::gtest_main)
add_test(NAME quickstart_context_packet COMMAND quickstart_context_packet)

# Read VRT File snippet test
add_executable(quickstart_file_reader file_reader_test.cpp)
target_link_libraries(quickstart_file_reader PRIVATE vrtio GTest::gtest_main)
target_compile_definitions(quickstart_file_reader PRIVATE TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/tests/data")
add_test(NAME quickstart_file_reader COMMAND quickstart_file_reader)

# Set C++20 standard
set_target_properties(quickstart_data_packet quickstart_context_packet quickstart_file_reader
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)