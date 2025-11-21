#!/bin/bash
# Extract quickstart snippets and generate docs/quickstart.md
# Only processes files listed in quickstart.order

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
QUICKSTART_DIR="$PROJECT_ROOT/tests/quickstart"
OUTPUT_FILE="$PROJECT_ROOT/docs/quickstart.md"
ORDER_FILE="$QUICKSTART_DIR/quickstart.order"

# Start with header
cat > "$OUTPUT_FILE" << 'EOF'
# VRTIGO Quickstart Code Examples

This file contains executable code snippets automatically extracted from test files in `tests/quickstart/`.
All code here is tested and guaranteed to compile.

---

EOF

# Check if order file exists
if [ ! -f "$ORDER_FILE" ]; then
    echo "Error: $ORDER_FILE not found" >&2
    exit 1
fi

echo "Processing snippets from $ORDER_FILE..." >&2

# Process each line in order file
while IFS= read -r line; do
    # Skip comments and empty lines
    if [[ $line =~ ^[[:space:]]*# ]] || [[ -z "${line// }" ]]; then
        continue
    fi

    # Parse line: filename.cpp: Title
    if [[ $line =~ ^([^:]+\.cpp)[[:space:]]*:[[:space:]]*(.+)$ ]]; then
        filename="${BASH_REMATCH[1]}"
        title="${BASH_REMATCH[2]}"
        filepath="$QUICKSTART_DIR/$filename"

        if [ ! -f "$filepath" ]; then
            echo "Warning: $filepath not found, skipping..." >&2
            continue
        fi

        echo "  Processing $filename..." >&2

        # Extract description if present
        description=""
        in_desc=0
        while IFS= read -r line; do
            if [[ $line =~ \[QUICKSTART-DESC\] ]]; then
                in_desc=1
            elif [[ $line =~ \[/QUICKSTART-DESC\] ]]; then
                in_desc=0
            elif [ $in_desc -eq 1 ]; then
                # Strip leading whitespace and comment markers
                desc_line="${line#"${line%%[![:space:]]*}"}"  # Strip leading whitespace
                desc_line="${desc_line#// }"                   # Strip "// "
                desc_line="${desc_line#//}"                    # Strip "//" if no space
                if [ -n "$description" ]; then
                    description="$description"$'\n'"$desc_line"
                else
                    description="$desc_line"
                fi
            fi
        done < "$filepath"

        # Extract code snippet
        code=""
        in_snippet=0
        while IFS= read -r line; do
            if [[ $line =~ \[QUICKSTART\] ]]; then
                in_snippet=1
            elif [[ $line =~ \[/QUICKSTART\] ]]; then
                in_snippet=0
            elif [ $in_snippet -eq 1 ]; then
                if [ -n "$code" ]; then
                    code="$code"$'\n'"$line"
                else
                    code="$line"
                fi
            fi
        done < "$filepath"

        # Write to output file
        echo "## $title" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"

        # Write description if available
        if [ -n "$description" ]; then
            echo "$description" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
        fi

        # Write code
        echo '```cpp' >> "$OUTPUT_FILE"
        echo "$code" >> "$OUTPUT_FILE"
        echo '```' >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    fi
done < "$ORDER_FILE"

echo "Generated $OUTPUT_FILE" >&2