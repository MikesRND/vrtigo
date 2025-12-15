cmake_minimum_required(VERSION 3.24)
project(vrtigo_python LANGUAGES CXX)

# Require C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python with NumPy support
find_package(Python 3.8 REQUIRED COMPONENTS Interpreter Development.Module NumPy)

# Fetch nanobind
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG        v2.4.0
)
FetchContent_MakeAvailable(nanobind)

# Build the Python module
nanobind_add_module(
    vrtigo_py
    STABLE_ABI
    NB_STATIC
    src/vrtigo_module.cpp
)

# Link to the main vrtigo library (header-only)
target_link_libraries(vrtigo_py PRIVATE vrtigo::vrtigo)

# Set output name to 'vrtigo' (importable as 'import vrtigo')
set_target_properties(vrtigo_py PROPERTIES OUTPUT_NAME vrtigo)

# Generate type stubs (.pyi) for IDE autocomplete
nanobind_add_stub(
    vrtigo_stub
    MODULE vrtigo
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/vrtigo.pyi
    PYTHON_PATH $<TARGET_FILE_DIR:vrtigo_py>
    DEPENDS vrtigo_py
    MARKER_FILE py.typed
)

# Install the module to the Python site-packages
install(TARGETS vrtigo_py LIBRARY DESTINATION .)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/vrtigo.pyi DESTINATION .)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/py.typed DESTINATION .)
