#!/bin/bash
# Auto-format all C++ files according to .clang-format

set -euo pipefail

# Color output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Auto-Format (clang-format)"
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
FILES=$(find include tests -type f \( -name "*.hpp" -o -name "*.h" -o -name "*.cpp" -o -name "*.cc" \) 2>/dev/null || true)

if [ -z "$FILES" ]; then
    echo -e "${YELLOW}Warning: No C++ files found to format${NC}"
    exit 0
fi

FILE_COUNT=$(echo "$FILES" | wc -l)
echo "Formatting $FILE_COUNT files..."
echo ""

# Format files
FORMATTED=0
while IFS= read -r file; do
    if ! $CLANG_FORMAT --dry-run --Werror "$file" &>/dev/null; then
        echo -e "${BLUE}Formatting:${NC} $file"
        $CLANG_FORMAT -i "$file"
        FORMATTED=$((FORMATTED + 1))
    fi
done <<< "$FILES"

echo ""
if [ $FORMATTED -eq 0 ]; then
    echo -e "${GREEN}✓ All files were already formatted correctly${NC}"
else
    echo -e "${GREEN}✓ Formatted $FORMATTED file(s)${NC}"
fi
