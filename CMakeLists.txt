cmake_minimum_required(VERSION 3.24)
project(vrtio VERSION 1.0.0 LANGUAGES CXX)

# C++20 required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only library
add_library(vrtio INTERFACE)
target_include_directories(vrtio INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Compiler warnings
target_compile_options(vrtio INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Optional: Build tests
option(VRTIO_BUILD_TESTS "Build tests" ON)
if(VRTIO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Optional: Build examples
option(VRTIO_BUILD_EXAMPLES "Build examples" ON)
if(VRTIO_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# Optional: Build benchmarks
option(VRTIO_BUILD_BENCHMARKS "Build benchmarks" OFF)
if(VRTIO_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Installation
install(DIRECTORY include/vrtio
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS vrtio
    EXPORT vrtioTargets
    INCLUDES DESTINATION include
)

install(EXPORT vrtioTargets
    FILE vrtioTargets.cmake
    NAMESPACE vrtio::
    DESTINATION lib/cmake/vrtio
)
