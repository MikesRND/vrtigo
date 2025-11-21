# Examples for VRTIGO

include(${CMAKE_SOURCE_DIR}/cmake/VrtigoBuild.cmake)

vrtigo_add_example(basic_usage basic_usage.cpp)
vrtigo_add_example(trailer_example trailer_example.cpp)
vrtigo_add_example(timestamp_example timestamp_example.cpp)
vrtigo_add_example(context_example context_example.cpp)
vrtigo_add_example(file_parsing_example file_parsing_example.cpp)

# UDP reader example (Linux/POSIX only)
if(UNIX)
    vrtigo_add_example(udp_reader_example udp_reader_example.cpp)
endif()

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
