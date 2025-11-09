cmake_minimum_required(VERSION 3.24)
project(vrtio_install_verification CXX)

# This is a standalone consumer project that verifies vrtio can be found
# and used via find_package() after installation

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find the installed vrtio package
# CMAKE_PREFIX_PATH should be set to the install location
find_package(vrtio REQUIRED)

# Create the minimal consumer executable
add_executable(minimal_consumer minimal_consumer.cpp)

# Link against vrtio (header-only, just adds include directories)
target_link_libraries(minimal_consumer PRIVATE vrtio::vrtio)

# Enable warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(minimal_consumer PRIVATE
        -Wall -Wextra -Wpedantic
    )
endif()
