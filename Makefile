# VRTIO Makefile - Convenience wrapper for CMake

.PHONY: all build clean configure debug release examples help
.PHONY: build-core build-io build-all
.PHONY: test-core test-io test-all test-roundtrip test-endian test-builder test-security test-trailer test-timestamp test-context test-field-access test-header-decode test-file-reader
.PHONY: run-core run-io run-all run-basic-usage run-trailer-example run-timestamp-example run-context-example run-file-parsing
.PHONY: check-core check-io check-all ci-core ci-io ci-all
.PHONY: list-tests list-examples format install uninstall

# Default build directory
BUILD_DIR ?= build
BUILD_TYPE ?= Debug

# Default target
all: build

# ============================================================================
# Configuration Targets
# ============================================================================

# Configure CMake (default: core only, no I/O helpers)
configure:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON

# Configure with I/O helpers enabled
configure-io:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DVRTIO_BUILD_IO_HELPERS=ON

# Debug build (default)
debug:
	@$(MAKE) configure BUILD_TYPE=Debug

# Release build (optimized)
release:
	@$(MAKE) configure BUILD_TYPE=Release

# ============================================================================
# Build Targets
# ============================================================================

# Build with current configuration
build: configure
	@cmake --build $(BUILD_DIR)

# Build core library only (header-only, no I/O)
build-core: configure
	@cmake --build $(BUILD_DIR)
	@echo "✓ Core library built (header-only)"

# Build with I/O helpers enabled
build-io: configure-io
	@cmake --build $(BUILD_DIR)
	@echo "✓ All components built (core + I/O helpers)"

# Build everything (alias for build-io)
build-all: build-io

# Build only examples
examples: configure
	@cmake --build $(BUILD_DIR) --target basic_usage trailer_example timestamp_example context_example

# Quick rebuild
rebuild: clean build

# Clean build artifacts
clean:
	@rm -rf $(BUILD_DIR)

# ============================================================================
# Test Targets (NO BARE "make test")
# ============================================================================

# Run core tests only (9 tests)
test-core: configure
	@cmake --build $(BUILD_DIR)
	@echo "Running core tests..."
	@cd $(BUILD_DIR) && ctest --output-on-failure -R "(roundtrip|endian|builder|security|trailer|timestamp|context|field_access|header_decode)_test"
	@echo "✓ Core tests passed"

# Run I/O tests only (1 test)
test-io: configure-io
	@cmake --build $(BUILD_DIR)
	@echo "Running I/O tests..."
	@cd $(BUILD_DIR) && ctest --output-on-failure -R "file_reader_test"
	@echo "✓ I/O tests passed"

# Run all tests (core + I/O)
test-all: configure-io
	@cmake --build $(BUILD_DIR)
	@echo "Running all tests..."
	@cd $(BUILD_DIR) && ctest --output-on-failure
	@echo "✓ All tests passed"

# Individual test targets
test-roundtrip: configure
	@cmake --build $(BUILD_DIR) --target roundtrip_test
	@./$(BUILD_DIR)/tests/roundtrip_test

test-endian: configure
	@cmake --build $(BUILD_DIR) --target endian_test
	@./$(BUILD_DIR)/tests/endian_test

test-builder: configure
	@cmake --build $(BUILD_DIR) --target builder_test
	@./$(BUILD_DIR)/tests/builder_test

test-security: configure
	@cmake --build $(BUILD_DIR) --target security_test
	@./$(BUILD_DIR)/tests/security_test

test-trailer: configure
	@cmake --build $(BUILD_DIR) --target trailer_test
	@./$(BUILD_DIR)/tests/trailer_test

test-timestamp: configure
	@cmake --build $(BUILD_DIR) --target timestamp_test
	@./$(BUILD_DIR)/tests/timestamp_test

test-context: configure
	@cmake --build $(BUILD_DIR) --target context_test
	@./$(BUILD_DIR)/tests/context_test

test-field-access: configure
	@cmake --build $(BUILD_DIR) --target field_access_test
	@./$(BUILD_DIR)/tests/field_access_test

test-header-decode: configure
	@cmake --build $(BUILD_DIR) --target header_decode_test
	@./$(BUILD_DIR)/tests/core/header_decode_test

test-file-reader: configure-io
	@cmake --build $(BUILD_DIR) --target file_reader_test
	@./$(BUILD_DIR)/tests/io/file_reader_test

# ============================================================================
# Run Targets (NO BARE "make run")
# ============================================================================

# Run core examples only (4 examples)
run-core: build-core
	@echo "Running core examples..."
	@echo "\n=== Basic Usage ==="
	@./$(BUILD_DIR)/examples/basic_usage
	@echo "\n=== Trailer Example ==="
	@./$(BUILD_DIR)/examples/trailer_example
	@echo "\n=== Timestamp Example ==="
	@./$(BUILD_DIR)/examples/timestamp_example
	@echo "\n=== Context Example ==="
	@./$(BUILD_DIR)/examples/context_example
	@echo "\n✓ Core examples completed"

# Run I/O examples only (1 example)
run-io: build-io
	@echo "Running I/O examples..."
	@if [ -f "$(BUILD_DIR)/examples/file_parsing_example" ]; then \
		echo "\n=== File Parsing Example ==="; \
		./$(BUILD_DIR)/examples/file_parsing_example tests/data/VITA49SampleData.bin; \
		echo "\n✓ I/O examples completed"; \
	else \
		echo "Error: file_parsing_example not built"; \
		exit 1; \
	fi

# Run all examples (core + I/O)
run-all: build-io
	@echo "Running all examples..."
	@echo "\n=== Basic Usage ==="
	@./$(BUILD_DIR)/examples/basic_usage
	@echo "\n=== Trailer Example ==="
	@./$(BUILD_DIR)/examples/trailer_example
	@echo "\n=== Timestamp Example ==="
	@./$(BUILD_DIR)/examples/timestamp_example
	@echo "\n=== Context Example ==="
	@./$(BUILD_DIR)/examples/context_example
	@if [ -f "$(BUILD_DIR)/examples/file_parsing_example" ]; then \
		echo "\n=== File Parsing Example ==="; \
		./$(BUILD_DIR)/examples/file_parsing_example tests/data/VITA49SampleData.bin; \
	fi
	@echo "\n✓ All examples completed"

# Individual run targets
run-basic-usage: build-core
	@./$(BUILD_DIR)/examples/basic_usage

run-trailer-example: build-core
	@./$(BUILD_DIR)/examples/trailer_example

run-timestamp-example: build-core
	@./$(BUILD_DIR)/examples/timestamp_example

run-context-example: build-core
	@./$(BUILD_DIR)/examples/context_example

run-file-parsing: build-io
	@if [ -f "$(BUILD_DIR)/examples/file_parsing_example" ]; then \
		./$(BUILD_DIR)/examples/file_parsing_example tests/data/VITA49SampleData.bin; \
	else \
		echo "Error: file_parsing_example not built (requires VRTIO_BUILD_IO_HELPERS=ON)"; \
		exit 1; \
	fi

# ============================================================================
# Workflow Targets
# ============================================================================

# Check: Build + Test
check-core: build-core test-core
	@echo "✓ Core validation complete"

check-io: build-io test-io
	@echo "✓ I/O validation complete"

check-all: build-io test-all
	@echo "✓ Full validation complete"

# CI: Build + Test + Run
ci-core: check-core run-core
	@echo "✓ Core CI pipeline complete"

ci-io: check-io run-io
	@echo "✓ I/O CI pipeline complete"

ci-all: check-all run-all
	@echo "✓ Full CI pipeline complete"

# ============================================================================
# Utility Targets
# ============================================================================

# List all available tests
list-tests:
	@echo "Available tests:"
	@echo "  Core Tests (9):"
	@echo "    - test-roundtrip      (roundtrip_test)"
	@echo "    - test-endian         (endian_test)"
	@echo "    - test-builder        (builder_test)"
	@echo "    - test-security       (security_test)"
	@echo "    - test-trailer        (trailer_test)"
	@echo "    - test-timestamp      (timestamp_test)"
	@echo "    - test-context        (context_test)"
	@echo "    - test-field-access   (field_access_test)"
	@echo "    - test-header-decode  (header_decode_test)"
	@echo "  I/O Tests (1):"
	@echo "    - test-file-reader    (file_reader_test) [requires I/O]"
	@echo ""
	@echo "Aggregate targets:"
	@echo "    - test-core           Run all core tests"
	@echo "    - test-io             Run I/O tests only"
	@echo "    - test-all            Run all tests"

# List all available examples
list-examples:
	@echo "Available examples:"
	@echo "  Core Examples (4):"
	@echo "    - run-basic-usage        (basic_usage)"
	@echo "    - run-trailer-example    (trailer_example)"
	@echo "    - run-timestamp-example  (timestamp_example)"
	@echo "    - run-context-example    (context_example)"
	@echo "  I/O Examples (1):"
	@echo "    - run-file-parsing       (file_parsing_example) [requires I/O]"
	@echo ""
	@echo "Aggregate targets:"
	@echo "    - run-core            Run all core examples"
	@echo "    - run-io              Run I/O examples only"
	@echo "    - run-all             Run all examples"

# Install (requires sudo for system-wide)
install: build
	@cmake --install $(BUILD_DIR)

# Uninstall (requires sudo for system-wide)
uninstall:
	@if [ -f $(BUILD_DIR)/install_manifest.txt ]; then \
		echo "Uninstalling VRTIO..."; \
		xargs rm -f < $(BUILD_DIR)/install_manifest.txt; \
		echo "✓ VRTIO uninstalled"; \
	else \
		echo "Error: No install manifest found. Have you run 'make install' yet?"; \
		exit 1; \
	fi

# Format code (if clang-format is available)
format:
	@if command -v clang-format >/dev/null 2>&1; then \
		find include examples tests -name "*.hpp" -o -name "*.cpp" | xargs clang-format -i; \
		echo "Code formatted"; \
	else \
		echo "clang-format not found"; \
	fi

# ============================================================================
# Help Target
# ============================================================================

help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║                    VRTIO Build System                           ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "ARCHITECTURE:"
	@echo "  VRTIO has a two-tier architecture:"
	@echo "    • Core (header-only)  - Always available, zero dependencies"
	@echo "    • I/O Helpers (opt-in) - Optional, requires configuration"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "BUILD TARGETS"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make                  Build with current configuration"
	@echo "  make build-core       Build core library only (header-only)"
	@echo "  make build-io         Build with I/O helpers enabled"
	@echo "  make build-all        Build everything (alias for build-io)"
	@echo "  make clean            Remove build directory"
	@echo "  make rebuild          Clean and rebuild"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "TEST TARGETS"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make test-core        Run core tests only (9 tests)"
	@echo "  make test-io          Run I/O tests only (1 test)"
	@echo "  make test-all         Run all tests (10 tests)"
	@echo ""
	@echo "  Individual tests:"
	@echo "    make test-roundtrip      make test-security"
	@echo "    make test-endian         make test-trailer"
	@echo "    make test-builder        make test-timestamp"
	@echo "    make test-context        make test-header-decode"
	@echo "    make test-file-reader    [requires I/O]"
	@echo ""
	@echo "  make list-tests       List all available tests"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "RUN TARGETS (Examples)"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make run-core         Run core examples (4 examples)"
	@echo "  make run-io           Run I/O examples (1 example)"
	@echo "  make run-all          Run all examples (5 examples)"
	@echo ""
	@echo "  Individual examples:"
	@echo "    make run-basic-usage         make run-timestamp-example"
	@echo "    make run-trailer-example     make run-context-example"
	@echo "    make run-file-parsing        [requires I/O]"
	@echo ""
	@echo "  make list-examples    List all available examples"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "WORKFLOW TARGETS"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make check-core       Build + test core"
	@echo "  make check-io         Build + test I/O"
	@echo "  make check-all        Build + test everything"
	@echo ""
	@echo "  make ci-core          Full validation: build + test + run core"
	@echo "  make ci-io            Full validation: build + test + run I/O"
	@echo "  make ci-all           Full validation: build + test + run everything"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "CONFIGURATION"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make debug            Configure for debug build (default)"
	@echo "  make release          Configure for release build"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "UTILITIES"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make format           Format code with clang-format"
	@echo "  make install          Install library system-wide"
	@echo "  make uninstall        Uninstall library from system"
	@echo "  make help             Show this help"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "VARIABLES"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  BUILD_DIR=<dir>       Set build directory (default: build)"
	@echo "  BUILD_TYPE=<type>     Set build type: Debug|Release (default: Debug)"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "EXAMPLES"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  # Core development workflow"
	@echo "  make build-core && make test-core && make run-core"
	@echo ""
	@echo "  # I/O development workflow"
	@echo "  make build-io && make test-io && make run-io"
	@echo ""
	@echo "  # Full validation (CI/CD)"
	@echo "  make ci-all"
	@echo ""
	@echo "  # Release build with all components"
	@echo "  make BUILD_TYPE=Release build-io"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
