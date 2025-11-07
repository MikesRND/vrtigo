# Examples for VRTIO

# Basic usage example
add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage PRIVATE vrtio)

# Trailer fields example
add_executable(trailer_example trailer_example.cpp)
target_link_libraries(trailer_example PRIVATE vrtio)

# Set C++20 standard
set_target_properties(basic_usage trailer_example PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
