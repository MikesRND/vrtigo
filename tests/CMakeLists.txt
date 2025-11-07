# Tests for VRTIO Phase 1

# Find GoogleTest
find_package(GTest REQUIRED)

# Enable testing
enable_testing()

# Round-trip tests
add_executable(roundtrip_test roundtrip_test.cpp)
target_link_libraries(roundtrip_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME roundtrip_test COMMAND roundtrip_test)

# Endian tests
add_executable(endian_test endian_test.cpp)
target_link_libraries(endian_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME endian_test COMMAND endian_test)

# Builder tests
add_executable(builder_test builder_test.cpp)
target_link_libraries(builder_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME builder_test COMMAND builder_test)

# Security tests
add_executable(security_test security_test.cpp)
target_link_libraries(security_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME security_test COMMAND security_test)

# Trailer tests
add_executable(trailer_test trailer_test.cpp)
target_link_libraries(trailer_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME trailer_test COMMAND trailer_test)

# Timestamp tests
add_executable(timestamp_test timestamp_test.cpp)
target_link_libraries(timestamp_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME timestamp_test COMMAND timestamp_test)

# Set C++20 standard for tests
set_target_properties(
    roundtrip_test endian_test builder_test security_test trailer_test timestamp_test
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)
