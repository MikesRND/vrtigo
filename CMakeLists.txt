cmake_minimum_required(VERSION 3.24)
project(
    vrtigo
    VERSION 0.0.1
    DESCRIPTION "Header-only C++20 library for VRT (VITA 49.2) packets"
    LANGUAGES CXX
)

if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(FATAL_ERROR "MSVC (cl) toolchain is not yet supported. Please use Clang or GCC.")
endif()

include(GNUInstallDirs)

# Header-only library
add_library(vrtigo INTERFACE)
add_library(vrtigo::vrtigo ALIAS vrtigo)
target_include_directories(vrtigo INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Generate version header
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/vrtigo/version.hpp
    @ONLY
)

# Require C++20 for consumers
target_compile_features(vrtigo INTERFACE cxx_std_20)
set_property(TARGET vrtigo PROPERTY CXX_EXTENSIONS OFF)

# Compiler warnings
target_compile_options(vrtigo INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
    $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# I/O helpers are always available (no optional toggle)

# Optional: Build tests
option(VRTIGO_BUILD_TESTS "Build tests" OFF)
# Allow fetching optional dependencies (e.g., GTest) via FetchContent
option(VRTIGO_FETCH_DEPENDENCIES "Fetch optional dependencies when not found locally" OFF)
if(VRTIGO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Custom target: Generate documentation from test files
add_custom_target(autodocs ALL
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/extract_docs.sh cif_access
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/extract_docs.sh quickstart
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating documentation from test files"
    VERBATIM
)

# Installation
install(DIRECTORY include/vrtigo
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(DIRECTORY include/tl
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.hpp"
)

install(FILES include/vrtigo.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/vrtigo/version.hpp
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vrtigo
)

install(TARGETS vrtigo
    EXPORT vrtigoTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT vrtigoTargets
    FILE vrtigoTargets.cmake
    NAMESPACE vrtigo::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vrtigo
)

# Configure and install package config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/vrtigoConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/vrtigoConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vrtigo
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/vrtigoConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/vrtigoConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/vrtigoConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/vrtigo
)
