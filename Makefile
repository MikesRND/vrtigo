# VRTIGO Makefile - Convenience wrapper for CMake

.PHONY: all clean configure debug release help check
.PHONY: test install uninstall autodocs
.PHONY: quick-check coverage debug-build clang-build install-verify ci-full clean-all
.PHONY: format-check format-fix format-diff clang-tidy clang-tidy-fix
.PHONY: python python-test python-stubs

# Default build directory
BUILD_DIR ?= build
BUILD_TYPE ?= Debug
VRTIGO_BUILD_TESTS ?= ON
VRTIGO_FETCH_DEPENDENCIES ?= ON

# Parallel build jobs
NPROC := $(shell nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# ============================================================================
# Configuration Targets
# ============================================================================

# Configure CMake
configure:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-DVRTIGO_BUILD_TESTS=$(VRTIGO_BUILD_TESTS) \
		-DVRTIGO_FETCH_DEPENDENCIES=$(VRTIGO_FETCH_DEPENDENCIES)

# Debug build (default)
debug:
	@$(MAKE) configure BUILD_TYPE=Debug

# Release build (optimized)
release:
	@$(MAKE) configure BUILD_TYPE=Release

# ============================================================================
# Build Targets
# ============================================================================

# Default target - build everything (GNU standard)
all: configure
	@cmake --build $(BUILD_DIR) -j$(NPROC)

# Generate documentation from test files
autodocs: configure
	@cmake --build $(BUILD_DIR) --target autodocs

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
	@cd $(BUILD_DIR) && \
		if [ "$(VERBOSE)" = "1" ]; then \
			ctest --output-on-failure -V; \
		else \
			ctest --output-on-failure; \
		fi
	@echo "✓ All tests passed"

# ============================================================================
# Workflow Targets
# ============================================================================

# GNU standard test target (alias to test)
check: test
	@echo "✓ Validation complete"

# ============================================================================
# CI Validation Targets
# ============================================================================

# Full CI validation - runs all CI checks locally
# Note: Some checks may be skipped if optional dependencies are missing
ci-full:
	@echo "Running full CI validation..."
	@$(MAKE) format-check
	@$(MAKE) quick-check
	@$(MAKE) debug-build
	@$(MAKE) clang-build
	@$(MAKE) clang-tidy
	@$(MAKE) install-verify
	@$(MAKE) python-test
	@$(MAKE) coverage
	@echo "\n✓ Full CI validation complete - safe to push!"

# Quick validation - matches CI quick-check job (required gate)
# Run this before every push to ensure CI gate will pass
quick-check:
	@./scripts/ci/quick-check.sh build-quick

# Code coverage report - matches CI coverage job
# Generates HTML report in build-coverage/coverage_html/index.html
coverage:
	@./scripts/ci/coverage.sh build-coverage

# Debug build check - matches CI debug-build job
debug-build:
	@./scripts/ci/debug-build.sh build-debug

# Clang build check - matches CI clang-build job
clang-build:
	@./scripts/ci/clang-build.sh build-clang

# Install verification - matches CI install-verification job
install-verify:
	@./scripts/ci/install-verify.sh build-install

# Clean all build directories (including CI build dirs)
clean-all:
	@echo "Removing all build directories..."
	@rm -rf build build-* install-test
	@echo "✓ All build artifacts removed"

# ============================================================================
# Utility Targets
# ============================================================================

# Install (requires sudo for system-wide)
install: all
	@cmake --install $(BUILD_DIR)

# Uninstall (requires sudo for system-wide)
uninstall:
	@if [ -f $(BUILD_DIR)/install_manifest.txt ]; then \
		echo "Uninstalling VRTIGO..."; \
		xargs rm -f < $(BUILD_DIR)/install_manifest.txt; \
		echo "✓ VRTIGO uninstalled"; \
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
	@for file in $$(find include tests -type f \( -name "*.hpp" -o -name "*.cpp" \) 2>/dev/null); do \
		if command -v clang-format >/dev/null 2>&1; then \
			if ! clang-format --dry-run --Werror "$$file" &>/dev/null; then \
				echo "\n=== $$file ==="; \
				clang-format "$$file" | diff -u "$$file" - || true; \
			fi; \
		fi; \
	done


# Run clang-tidy static analysis
clang-tidy:
	@./scripts/ci/clang-tidy.sh build

# Run clang-tidy with auto-fix (use with caution!)
clang-tidy-fix:
	@echo "Running clang-tidy with auto-fix (this may modify your code)..."
	@./scripts/ci/clang-tidy.sh build --fix

# ============================================================================
# Python Targets
# ============================================================================

# Build Python bindings
python:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake .. \
		-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
		-DVRTIGO_BUILD_PYTHON=ON
	@cmake --build $(BUILD_DIR) --target vrtigo_py vrtigo_stub -j$(NPROC)
	@echo "✓ Python bindings built"
	@echo "  Module: $(BUILD_DIR)/bindings/python/vrtigo*.so"
	@echo "  Stubs:  $(BUILD_DIR)/bindings/python/vrtigo.pyi"

# Run Python tests
python-test: python
	@echo "Running Python tests..."
	@PYTHONPATH=$(BUILD_DIR)/bindings/python pytest bindings/python/tests -v
	@echo "✓ Python tests passed"

# Generate Python stubs only
python-stubs: python
	@cmake --build $(BUILD_DIR) --target vrtigo_stub
	@echo "✓ Stubs generated: $(BUILD_DIR)/bindings/python/vrtigo.pyi"

# ============================================================================
# Help Target
# ============================================================================

help:
	@echo "╔════════════════════════════════════════════════════════════════╗"
	@echo "║                    VRTIGO Build System                           ║"
	@echo "╚════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "COMMON TARGETS"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make / make all       Build everything"
	@echo "  make test             Run all tests"
	@echo "  make autodocs         Generate docs from test files"
	@echo ""
	@echo "  make clean            Remove build directory"
	@echo "  make clean-all        Remove all build dirs"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "CI VALIDATION"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make ci-full          Run all CI checks locally"
	@echo ""
	@echo "  Code Quality:"
	@echo "    make format-check        Check code formatting"
	@echo "    make format-fix          Auto-fix code formatting"
	@echo "    make format-diff         Show formatting differences"
	@echo "    make clang-tidy          Run static analysis"
	@echo "    make clang-tidy-fix      Auto-fix static analysis issues"
	@echo "    make coverage            Generate coverage report"
	@echo ""
	@echo "  Other CI targets:"
	@echo "    make quick-check         Fast validation (Release build)"
	@echo "    make debug-build         Debug build (catch assertions)"
	@echo "    make clang-build         Clang compiler test"
	@echo "    make install-verify      Install/package verification"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "ADVANCED"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  Persistent Build modes:"
	@echo "    make debug               Configure debug build (default)"
	@echo "    make release             Configure release build"
	@echo ""
	@echo "  Install:"
	@echo "    make install             Install library system-wide"
	@echo "    make uninstall           Uninstall from system"
	@echo ""
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "PYTHON BINDINGS"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@echo "  make python               Build Python bindings + stubs"
	@echo "  make python-test          Run Python tests"
	@echo ""
