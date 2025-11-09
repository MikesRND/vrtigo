# Examples for VRTIO

# Basic usage example
add_executable(basic_usage basic_usage.cpp)
target_link_libraries(basic_usage PRIVATE vrtio)

# Trailer fields example
add_executable(trailer_example trailer_example.cpp)
target_link_libraries(trailer_example PRIVATE vrtio)

# Timestamp example
add_executable(timestamp_example timestamp_example.cpp)
target_link_libraries(timestamp_example PRIVATE vrtio)

# Context packet example
add_executable(context_example context_example.cpp)
target_link_libraries(context_example PRIVATE vrtio)

# File parsing example
add_executable(file_parsing_example file_parsing_example.cpp)
target_link_libraries(file_parsing_example PRIVATE vrtio)

# Set C++20 standard
set_target_properties(basic_usage trailer_example timestamp_example context_example file_parsing_example PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Add custom targets to run examples
add_custom_target(run-basic-usage
    COMMAND basic_usage
    DEPENDS basic_usage
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running basic usage example"
)

add_custom_target(run-trailer-example
    COMMAND trailer_example
    DEPENDS trailer_example
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running trailer example"
)

add_custom_target(run-timestamp-example
    COMMAND timestamp_example
    DEPENDS timestamp_example
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running timestamp example"
)

add_custom_target(run-context-example
    COMMAND context_example
    DEPENDS context_example
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running context example"
)

add_custom_target(run-examples
    COMMAND ${CMAKE_COMMAND} -E echo "Running all examples..."
    COMMAND basic_usage
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND trailer_example
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND timestamp_example
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND context_example
    DEPENDS basic_usage trailer_example timestamp_example context_example
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running all examples"
)

# Add 'run' as a convenient alias
add_custom_target(run
    COMMAND ${CMAKE_COMMAND} -E echo "Running all examples..."
    COMMAND basic_usage
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND trailer_example
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND timestamp_example
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND context_example
    DEPENDS basic_usage trailer_example timestamp_example context_example
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    COMMENT "Running all examples"
)
