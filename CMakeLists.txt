cmake_minimum_required(VERSION 3.24)
project(vrtio VERSION 0.0.1 LANGUAGES CXX)

# C++20 required
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Header-only library
add_library(vrtio INTERFACE)
target_include_directories(vrtio INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Generate version header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/vrtio/version.hpp
    @ONLY
)

# Compiler warnings
target_compile_options(vrtio INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# I/O helpers are always available (no optional toggle)

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

# Custom target: Extract quickstart snippets to docs
add_custom_target(quickstart ALL
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/extract_quickstart.sh
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Extracting quickstart code snippets to docs/quickstart.md"
    VERBATIM
)

# Installation
install(DIRECTORY include/vrtio
    DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)

install(FILES include/vrtio.hpp
    DESTINATION include
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/vrtio/version.hpp
    DESTINATION include/vrtio
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

# Configure and install package config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vrtioConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/vrtioConfig.cmake
    INSTALL_DESTINATION lib/cmake/vrtio
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/vrtioConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/vrtioConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/vrtioConfigVersion.cmake
    DESTINATION lib/cmake/vrtio
)
