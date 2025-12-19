cmake_minimum_required(VERSION 3.24)
project(vrtigo_install_verification CXX)

# This is a standalone consumer project that verifies vrtigo can be found
# and used via find_package() after installation

# Find the installed vrtigo package
# CMAKE_PREFIX_PATH should be set to the install location
find_package(vrtigo REQUIRED)

# Create the minimal consumer executable
add_executable(minimal_consumer minimal_consumer.cpp)
target_compile_features(minimal_consumer PRIVATE cxx_std_20)
set_target_properties(minimal_consumer PROPERTIES CXX_EXTENSIONS OFF)

# Link against vrtigo (header-only, just adds include directories)
target_link_libraries(minimal_consumer PRIVATE vrtigo::vrtigo)

# Enable warnings
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(minimal_consumer PRIVATE
        -Wall -Wextra -Wpedantic
    )
elseif(MSVC)
    target_compile_options(minimal_consumer PRIVATE /W4)
endif()

# Optional: GPU consumer (requires CUDA and vrtigo::gpu target)
option(VRTIGO_TEST_GPU "Build GPU install verification" OFF)
if(VRTIGO_TEST_GPU)
    # Check if vrtigo::gpu target exists (installed with CUDA support)
    if(TARGET vrtigo::gpu)
        enable_language(CUDA)

        add_executable(gpu_consumer gpu_consumer.cu)
        target_link_libraries(gpu_consumer PRIVATE vrtigo::gpu)
        set_target_properties(gpu_consumer PROPERTIES
            CUDA_STANDARD 20
            CUDA_STANDARD_REQUIRED ON
        )
    else()
        message(WARNING "VRTIGO_TEST_GPU=ON but vrtigo::gpu target not found. "
                        "Was vrtigo built with -DVRTIGO_ENABLE_CUDA=ON?")
    endif()
endif()
