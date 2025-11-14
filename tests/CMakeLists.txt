# Tests for VRTIO

# Find or fetch GoogleTest
find_package(GTest QUIET)

if(NOT GTest_FOUND)
    message(STATUS "GTest not found, fetching from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    # FetchContent already creates GTest::gtest and GTest::gtest_main targets
endif()

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

# Signal packet view tests (runtime parser)
add_executable(signal_packet_view_test signal_packet_view_test.cpp)
target_link_libraries(signal_packet_view_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME signal_packet_view_test COMMAND signal_packet_view_test)

# Packet concepts tests
add_executable(packet_concepts_test packet_concepts_test.cpp)
target_link_libraries(packet_concepts_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME packet_concepts_test COMMAND packet_concepts_test)

# Context packet tests (refactored into focused test suites)
add_executable(context_basic_test context_basic_test.cpp)
target_link_libraries(context_basic_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME context_basic_test COMMAND context_basic_test)

add_executable(context_validation_test context_validation_test.cpp)
target_link_libraries(context_validation_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME context_validation_test COMMAND context_validation_test)

add_executable(context_variable_test context_variable_test.cpp)
target_link_libraries(context_variable_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME context_variable_test COMMAND context_variable_test)

add_executable(context_timestamp_test context_timestamp_test.cpp)
target_link_libraries(context_timestamp_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME context_timestamp_test COMMAND context_timestamp_test)

add_executable(context_integration_test context_integration_test.cpp)
target_link_libraries(context_integration_test PRIVATE vrtio GTest::gtest_main)
add_test(NAME context_integration_test COMMAND context_integration_test)

# Field access API tests
add_executable(field_access_test field_access_test.cpp)
target_link_libraries(field_access_test PRIVATE vrtio)
add_test(NAME field_access_test COMMAND field_access_test)

# Set C++20 standard for tests
set_target_properties(
    roundtrip_test endian_test builder_test security_test trailer_test timestamp_test
    signal_packet_view_test packet_concepts_test
    context_basic_test context_validation_test context_variable_test
    context_timestamp_test context_integration_test
    field_access_test
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)

# Add core tests subdirectory (header decode utilities, etc.)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/core)
    add_subdirectory(core)
endif()

# Add I/O tests subdirectory (always built)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/io)
    add_subdirectory(io)
endif()

# Add CIF field unit tests subdirectory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cif)
    add_subdirectory(cif)
endif()

# Add quickstart documentation tests subdirectory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/quickstart)
    add_subdirectory(quickstart)
endif()
