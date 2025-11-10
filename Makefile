# VRTIO Makefile - Convenience wrapper for CMake

.PHONY: all build clean configure debug release examples help check
.PHONY: test run list-tests list-examples format install uninstall
.PHONY: quick-check ci-coverage ci-debug ci-clang ci-install-verify ci-local ci-full clean-all rebuild
.PHONY: format-check format-fix format-diff tidy tidy-fix ci-format ci-tidy

# Default build directory
BUILD_DIR ?= build
BUILD_TYPE ?= Debug

# Parallel build jobs
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Default target (GNU standard)
all: build

# ============================================================================
# Configuration Targets
# ============================================================================

# Configure CMake
configure:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON

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
	@cmake --build $(BUILD_DIR) -j$(NPROC)

# Build only examples
examples: configure
	@cmake --build $(BUILD_DIR) --target basic_usage trailer_example timestamp_example context_example file_parsing_example

# Quick rebuild
rebuild: clean build

# Clean build artifacts
clean:
	@rm -rf $(BUILD_DIR)

# ============================================================================
# Test Targets
# ============================================================================

# Run all tests
test: configure
	@cmake --build $(BUILD_DIR)
	@echo "Running all tests..."
	@cd $(BUILD_DIR) && ctest --output-on-failure
	@echo "✓ All tests passed"

# ============================================================================
# Run Targets
# ============================================================================

# Run all examples
run: build
	@echo "Running all examples..."
	@echo "\n=== Basic Usage ==="
	@./$(BUILD_DIR)/examples/basic_usage
	@echo "\n=== Trailer Example ==="
	@./$(BUILD_DIR)/examples/trailer_example
	@echo "\n=== Timestamp Example ==="
	@./$(BUILD_DIR)/examples/timestamp_example
	@echo "\n=== Context Example ==="
	@./$(BUILD_DIR)/examples/context_example
	@echo "\n=== File Parsing Example ==="
	@./$(BUILD_DIR)/examples/file_parsing_example tests/data/VITA49SampleData.bin
	@echo "\n✓ All examples completed"

# ============================================================================
# Workflow Targets
# ============================================================================

# GNU standard test target (alias to test)
check: test
	@echo "✓ Validation complete"

# ============================================================================
# CI Validation Targets
# ============================================================================

# Local CI validation (essential checks - no coverage)
# Note: Some checks may be skipped if optional dependencies are missing
ci-local:
	@echo "Running essential CI checks locally..."
	@$(MAKE) quick-check
	@$(MAKE) ci-debug
	@$(MAKE) ci-clang
	@echo "\n✓ Local CI validation complete"

# Full CI validation (includes coverage + install verify)
# Note: Some checks may be skipped if optional dependencies are missing
ci-full:
	@echo "Running full CI validation..."
	@$(MAKE) ci-local
	@$(MAKE) ci-install-verify
	@$(MAKE) ci-coverage
	@echo "\n✓ Full CI validation complete - safe to push!"

# Quick validation - matches CI quick-check job (required gate)
# Run this before every push to ensure CI gate will pass
quick-check:
	@./scripts/ci/quick-check.sh build-quick

# Code coverage report - matches CI coverage job
# Generates HTML report in build-coverage/coverage_html/index.html
ci-coverage:
	@./scripts/ci/coverage.sh build-coverage

# Debug build check - matches CI debug-build job
ci-debug:
	@./scripts/ci/debug-build.sh build-debug

# Clang build check - matches CI clang-build job
ci-clang:
	@./scripts/ci/clang-build.sh build-clang

# Install verification - matches CI install-verification job
ci-install-verify:
	@./scripts/ci/install-verify.sh build-install

# Clean all build directories (including CI build dirs)
clean-all:
	@echo "Removing all build directories..."
	@rm -rf build build-* install-test
	@echo "✓ All build artifacts removed"

# ============================================================================
# Utility Targets
# ============================================================================

# List all available tests
list-tests:
	@echo "Available tests:"
	@echo "    - test-roundtrip      (roundtrip_test)"
	@echo "    - test-endian         (endian_test)"
	@echo "    - test-builder        (builder_test)"
	@echo "    - test-security       (security_test)"
	@echo "    - test-trailer        (trailer_test)"
	@echo "    - test-timestamp      (timestamp_test)"
	@echo "    - test-context        (context_test)"
	@echo "    - test-field-access   (field_access_test)"
	@echo "    - test-header-decode  (header_decode_test)"
	@echo "    - test-file-reader    (file_reader_test)"
	@echo ""
	@echo "Aggregate target:"
	@echo "    - test                Run all tests"

# List all available examples
list-examples:
	@echo "Available examples:"
	@echo "    - run-basic-usage        (basic_usage)"
	@echo "    - run-trailer-example    (trailer_example)"
	@echo "    - run-timestamp-example  (timestamp_example)"
	@echo "    - run-context-example    (context_example)"
	@echo "    - run-file-parsing       (file_parsing_example)"
	@echo ""
	@echo "Aggregate target:"
	@echo "    - run                 Run all examples"

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

# ============================================================================
# Code Quality Targets
# ============================================================================

# Check code formatting (CI-friendly, returns error if changes needed)
format-check:
	@./scripts/ci/format-check.sh

# Auto-fix code formatting
format-fix:
	@./scripts/ci/format-fix.sh

# Show formatting differences (uses git diff for colored output)
format-diff:
	@echo "Formatting differences (using clang-format):"
	@for file in $$(find include tests examples -type f \( -name "*.hpp" -o -name "*.cpp" \) 2>/dev/null); do \
		if command -v clang-format >/dev/null 2>&1; then \
			if ! clang-format --dry-run --Werror "$$file" &>/dev/null; then \
				echo "\n=== $$file ==="; \
				clang-format "$$file" | diff -u "$$file" - || true; \
			fi; \
		fi; \
	done

# Legacy alias for format-fix
format: format-fix

# Run clang-tidy static analysis
tidy:
	@./scripts/ci/clang-tidy.sh build

# Run clang-tidy with auto-fix (use with caution!)
tidy-fix:
	@echo "Running clang-tidy with auto-fix (this may modify your code)..."
	@./scripts/ci/clang-tidy.sh build --fix

# CI format check (matches CI job)
ci-format: format-check

# CI static analysis check (matches CI job)
ci-tidy: tidy

# ============================================================================
# Help Target
# ============================================================================

help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║                    VRTIO Build System                           ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "COMMON TARGETS"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make / make all       Build everything (tests + examples)"
	@echo "  make build            Same as 'all' (explicit alias)"
	@echo "  make test             Run all tests (21 tests)"
	@echo "  make check            Same as 'test' (GNU standard)"
	@echo "  make run              Run all examples (5 examples)"
	@echo "  make examples         Build examples only"
	@echo ""
	@echo "  make clean            Remove build directory"
	@echo "  make rebuild          Clean + build"
	@echo "  make clean-all        Remove all build dirs (including CI)"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "CI VALIDATION"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make quick-check      Fast validation (required CI gate)"
	@echo "  make ci-local         Essential checks (quick + debug + clang)"
	@echo "  make ci-full          Full validation (local + install + coverage)"
	@echo ""
	@echo "  Individual checks (see scripts/ci/*.sh):"
	@echo "    make ci-debug            Debug build (catch assertions)"
	@echo "    make ci-clang            Clang compiler test"
	@echo "    make ci-install-verify   Install/package verification"
	@echo "    make ci-coverage         Generate coverage report"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "ADVANCED"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Build modes:"
	@echo "    make debug               Configure debug build (default)"
	@echo "    make release             Configure release build"
	@echo ""
	@echo "  Install:"
	@echo "    make install             Install library system-wide"
	@echo "    make uninstall           Uninstall from system"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make format-check        Check code formatting (CI-friendly)"
	@echo "    make format-fix          Auto-fix code formatting"
	@echo "    make format-diff         Show formatting differences"
	@echo "    make tidy                Run clang-tidy static analysis"
	@echo "    make ci-format           Format check (matches CI)"
	@echo "    make ci-tidy             Static analysis (matches CI)"
	@echo ""
	@echo "  Utilities:"
	@echo "    make list-tests          List all test targets"
	@echo "    make list-examples       List all example targets"
	@echo ""
	@echo "  Run individual tests/examples:"
	@echo "    ctest -R <pattern>       Run tests matching pattern"
	@echo "    ./build/tests/<name>     Run specific test directly"
	@echo "    ./build/examples/<name>  Run specific example directly"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "CONFIGURATION & EXAMPLES"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Variables:"
	@echo "    BUILD_DIR=<dir>          Build directory (default: build)"
	@echo "    BUILD_TYPE=<type>        Debug|Release (default: Debug)"
	@echo ""
	@echo "  Common workflows:"
	@echo "    make                     # Quick build"
	@echo "    make test                # Build + test"
	@echo "    make quick-check         # Before git push"
	@echo "    make ci-local            # Before important PR"
	@echo "    make ci-full             # Before release"
	@echo ""
	@echo "    BUILD_TYPE=Release make  # Release build"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo ""
