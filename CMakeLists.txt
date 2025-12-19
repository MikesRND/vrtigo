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

# I/O helpers are always available (no optional toggle)

# Optional: Build tests
option(VRTIGO_BUILD_TESTS "Build tests" OFF)
# Allow fetching optional dependencies (e.g., GTest) via FetchContent
option(VRTIGO_FETCH_DEPENDENCIES "Fetch optional dependencies when not found locally" OFF)
if(VRTIGO_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Optional: CUDA GPU extensions
option(VRTIGO_ENABLE_CUDA "Enable CUDA GPU extensions" OFF)
option(VRTIGO_BUILD_GPU_TESTS "Build GPU tests" OFF)
option(VRTIGO_BUILD_GPU_SYNTAX_CHECK "Build GPU syntax check target" OFF)

# Track whether CUDA was enabled (for config file substitution)
set(VRTIGO_CUDA_ENABLED FALSE)

if(VRTIGO_ENABLE_CUDA)
    # Workaround for CMake not yet supporting newer CUDA versions (e.g., CUDA 13.x)
    # These internal variables may be missing with unsupported CUDA versions
    if(NOT DEFINED _CMAKE_CUDA_WHOLE_FLAG)
        set(_CMAKE_CUDA_WHOLE_FLAG "")
    endif()
    if(NOT DEFINED CMAKE_CUDA_COMPILE_OBJECT)
        set(CMAKE_CUDA_COMPILE_OBJECT
            "<CMAKE_CUDA_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -c <SOURCE> -o <OBJECT>")
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # GPU extension library (interface/header-only)
    add_library(vrtigo_gpu INTERFACE)
    add_library(vrtigo::gpu ALIAS vrtigo_gpu)
    # Export as vrtigo::gpu (not vrtigo::vrtigo_gpu)
    set_target_properties(vrtigo_gpu PROPERTIES EXPORT_NAME gpu)
    target_link_libraries(vrtigo_gpu INTERFACE vrtigo::vrtigo CUDA::cudart)
    target_compile_definitions(vrtigo_gpu INTERFACE VRTIGO_ENABLE_CUDA)
    # Require CUDA C++20 for consumers (concepts, std::span, etc.)
    target_compile_features(vrtigo_gpu INTERFACE cuda_std_20)

    # Mark CUDA as enabled for config file substitution
    set(VRTIGO_CUDA_ENABLED TRUE)
endif()

# Optional: Build Python bindings
option(VRTIGO_BUILD_PYTHON "Build Python bindings (requires Python and nanobind)" OFF)
if(VRTIGO_BUILD_PYTHON)
    add_subdirectory(bindings/python)
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

# Install GPU headers (always installed, CUDA optional at consumer build time)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/include/vrtigo/gpu)
    install(DIRECTORY include/vrtigo/gpu
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vrtigo
        FILES_MATCHING PATTERN "*.hpp"
    )
endif()

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

# Core library target
install(TARGETS vrtigo
    EXPORT vrtigoTargets
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# GPU extension target (if CUDA enabled at build time)
if(VRTIGO_ENABLE_CUDA)
    install(TARGETS vrtigo_gpu
        EXPORT vrtigoTargets
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
    # Record that CUDA was enabled for the config file
    set(VRTIGO_CUDA_ENABLED TRUE)
endif()

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
