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
