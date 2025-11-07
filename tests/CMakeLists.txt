# Tests for VRTIO Phase 1

include(FetchContent)

# Fetch GoogleTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest)

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

# Set C++20 standard for tests
set_target_properties(
    roundtrip_test endian_test builder_test security_test
    PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
)
