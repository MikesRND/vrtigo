# Tests for VRTIGO

include(${CMAKE_SOURCE_DIR}/cmake/VrtigoBuild.cmake)

# Find or fetch GoogleTest
find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
    find_package(GTest QUIET)
endif()

if(NOT GTest_FOUND)
    if(NOT VRTIGO_FETCH_DEPENDENCIES)
        message(FATAL_ERROR "GTest not found. Enable VRTIGO_FETCH_DEPENDENCIES to fetch it automatically.")
    endif()

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
endif()

vrtigo_add_gtest(roundtrip_test roundtrip_test.cpp)
vrtigo_add_gtest(endian_test endian_test.cpp)
vrtigo_add_gtest(security_test security_test.cpp)
vrtigo_add_gtest(trailer_test trailer_test.cpp)
vrtigo_add_gtest(timestamp_test timestamp_test.cpp)
vrtigo_add_gtest(sample_clock_test sample_clock_test.cpp)
vrtigo_add_gtest(start_time_test start_time_test.cpp)
vrtigo_add_gtest(runtime_data_packet_test runtime_data_packet_test.cpp)
vrtigo_add_gtest(packet_concepts_test packet_concepts_test.cpp)
vrtigo_add_gtest(sample_span_test sample_span_test.cpp)
vrtigo_add_gtest(variable_packet_size_test variable_packet_size_test.cpp)

vrtigo_add_gtest(context_basic_test context_basic_test.cpp)
vrtigo_add_gtest(context_validation_test context_validation_test.cpp)
vrtigo_add_gtest(context_variable_test context_variable_test.cpp)
vrtigo_add_gtest(context_timestamp_test context_timestamp_test.cpp)
vrtigo_add_gtest(context_integration_test context_integration_test.cpp)

vrtigo_add_test_binary(field_access_test field_access_test.cpp)
vrtigo_add_test_binary(bitfield_test bitfield_test.cpp)

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

# Add CIF access pattern tests subdirectory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cif_access)
    add_subdirectory(cif_access)
endif()

# ============================================================================
# GPU Tests (requires CUDA)
# ============================================================================

# Set default CUDA architectures if not specified
# Use 'native' to detect local GPU; set CMAKE_CUDA_ARCHITECTURES explicitly for headless builds
if(VRTIGO_ENABLE_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# GPU syntax check - compile-only, no runtime needed
# Can run in nvidia/cuda container without actual GPU
if(VRTIGO_ENABLE_CUDA AND VRTIGO_BUILD_GPU_SYNTAX_CHECK)
    add_library(gpu_compile_check OBJECT
        gpu/compile_check.cu
    )
    target_link_libraries(gpu_compile_check PRIVATE vrtigo::vrtigo)
    target_compile_definitions(gpu_compile_check PRIVATE VRTIGO_ENABLE_CUDA)
    set_target_properties(gpu_compile_check PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )
endif()

# GPU runtime tests - require actual GPU
if(VRTIGO_ENABLE_CUDA AND VRTIGO_BUILD_GPU_TESTS)
    # Complex type unit tests
    add_executable(gpu_complex_test
        gpu/complex_test.cu
    )
    target_link_libraries(gpu_complex_test PRIVATE
        vrtigo::vrtigo
        GTest::gtest
        GTest::gtest_main
    )
    target_compile_definitions(gpu_complex_test PRIVATE VRTIGO_ENABLE_CUDA)
    set_target_properties(gpu_complex_test PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    add_test(NAME gpu_complex_test COMMAND gpu_complex_test)
    set_tests_properties(gpu_complex_test PROPERTIES LABELS "gpu")

    # Integration tests (end-to-end pipeline)
    add_executable(gpu_integration_test
        gpu/integration_test.cu
    )
    target_link_libraries(gpu_integration_test PRIVATE
        vrtigo::vrtigo
        GTest::gtest
        GTest::gtest_main
    )
    target_compile_definitions(gpu_integration_test PRIVATE VRTIGO_ENABLE_CUDA)
    set_target_properties(gpu_integration_test PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
    )

    add_test(NAME gpu_integration_test COMMAND gpu_integration_test)
    set_tests_properties(gpu_integration_test PROPERTIES LABELS "gpu")
endif()
