#!/bin/bash
# Check if all C++ files are formatted according to .clang-format
# Returns non-zero exit code if any files need formatting

set -euo pipefail

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Format Check (clang-format)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Find clang-format (try version-specific names first)
CLANG_FORMAT=""
for cmd in clang-format-18 clang-format-17 clang-format-16 clang-format-15 clang-format; do
    if command -v "$cmd" &>/dev/null; then
        CLANG_FORMAT="$cmd"
        break
    fi
done

if [ -z "$CLANG_FORMAT" ]; then
    echo -e "${RED}Error: clang-format not found${NC}"
    echo "Please install clang-format (version 15 or newer recommended)"
    exit 1
fi

VERSION=$($CLANG_FORMAT --version)
echo "Using: $CLANG_FORMAT"
echo "Version: $VERSION"
echo ""

# Find all C++ source files
FILES=$(find include tests examples -type f \( -name "*.hpp" -o -name "*.h" -o -name "*.cpp" -o -name "*.cc" \) 2>/dev/null || true)

if [ -z "$FILES" ]; then
    echo -e "${YELLOW}Warning: No C++ files found to check${NC}"
    exit 0
fi

FILE_COUNT=$(echo "$FILES" | wc -l)
echo "Checking $FILE_COUNT files..."
echo ""

# Check formatting
NEEDS_FORMAT=()
while IFS= read -r file; do
    if ! $CLANG_FORMAT --dry-run --Werror "$file" &>/dev/null; then
        NEEDS_FORMAT+=("$file")
    fi
done <<< "$FILES"

# Report results
if [ ${#NEEDS_FORMAT[@]} -eq 0 ]; then
    echo -e "${GREEN}✓ All files are properly formatted!${NC}"
    exit 0
else
    echo -e "${RED}✗ Found ${#NEEDS_FORMAT[@]} file(s) with formatting issues:${NC}"
    echo ""
    for file in "${NEEDS_FORMAT[@]}"; do
        echo "  - $file"
    done
    echo ""
    echo "To see the differences, run:"
    echo "  make format-diff"
    echo ""
    echo "To fix formatting automatically, run:"
    echo "  make format-fix"
    echo ""
    exit 1
fi
