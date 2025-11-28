#!/bin/bash
# Unified documentation extraction script
# Extracts examples from test files and generates markdown documentation
#
# Usage: extract_docs.sh <target>
# Where target is: cif_access | quickstart

set -e

TARGET="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

# Configure based on target
case "$TARGET" in
    cif_access)
        SRC_DIR="$PROJECT_ROOT/tests/cif_access"
        DOCS_DIR="$PROJECT_ROOT/docs/cif_access"
        PATTERN="cif*_test.cpp"
        ;;
    quickstart)
        SRC_DIR="$PROJECT_ROOT/tests/quickstart"
        DOCS_DIR="$PROJECT_ROOT/docs/quickstart"
        PATTERN="*_test.cpp"
        ;;
    *)
        echo "Usage: $0 <cif_access|quickstart>" >&2
        exit 1
        ;;
esac

# Create docs directory if it doesn't exist
mkdir -p "$DOCS_DIR"

# Remove old generated docs to prevent stale files
rm -f "$DOCS_DIR"/*.md

echo "Extracting $TARGET documentation..." >&2

# Process each matching test file
for testfile in "$SRC_DIR"/$PATTERN; do
    [ -f "$testfile" ] || continue

    filename=$(basename "$testfile")
    basename="${filename%_test.cpp}"
    output_file="$DOCS_DIR/${basename}.md"

    echo "  Processing $filename..." >&2

    # Extract document title
    title=""
    in_title=0
    while IFS= read -r line; do
        if [[ $line =~ \[TITLE\] ]]; then
            in_title=1
        elif [[ $line =~ \[/TITLE\] ]]; then
            break
        elif [ $in_title -eq 1 ]; then
            # Strip leading whitespace and comment markers
            title_line="${line#"${line%%[![:space:]]*}"}"  # Strip leading whitespace
            title_line="${title_line#// }"                  # Strip "// "
            title_line="${title_line#//}"                   # Strip "//" if no space
            if [ -n "$title" ]; then
                title="$title"$'\n'"$title_line"
            else
                title="$title_line"
            fi
        fi
    done < "$testfile"

    # Default title if none found
    if [ -z "$title" ]; then
        title="Examples"
    fi

    # Start output file with header
    cat > "$output_file" << EOF
# $title

*Auto-generated from \`tests/$TARGET/$filename\`. All examples are tested.*

---

EOF

    # Extract all EXAMPLE/DESCRIPTION/SNIPPET triplets and TEXT blocks
    # We'll read the file once and track state
    in_example=0
    in_description=0
    in_snippet=0
    in_text=0
    example_heading=""
    description=""
    snippet=""
    text_block=""

    while IFS= read -r line; do
        # TEXT markers (standalone prose between examples)
        if [[ $line =~ \[TEXT\] ]]; then
            in_text=1
            text_block=""
        elif [[ $line =~ \[/TEXT\] ]]; then
            in_text=0
            # Output text block immediately as plain paragraph
            if [ -n "$text_block" ]; then
                echo "$text_block" >> "$output_file"
                echo "" >> "$output_file"
            fi
            text_block=""
        elif [ $in_text -eq 1 ]; then
            # Strip leading whitespace and comment markers
            text_line="${line#"${line%%[![:space:]]*}"}"
            text_line="${text_line#// }"
            text_line="${text_line#//}"
            if [ -n "$text_block" ]; then
                text_block="$text_block"$'\n'"$text_line"
            else
                text_block="$text_line"
            fi
        fi

        # EXAMPLE markers
        if [[ $line =~ \[EXAMPLE\] ]]; then
            in_example=1
            example_heading=""
        elif [[ $line =~ \[/EXAMPLE\] ]]; then
            in_example=0
        elif [ $in_example -eq 1 ]; then
            # Strip leading whitespace and comment markers
            heading_line="${line#"${line%%[![:space:]]*}"}"
            heading_line="${heading_line#// }"
            heading_line="${heading_line#//}"
            if [ -n "$example_heading" ]; then
                example_heading="$example_heading"$'\n'"$heading_line"
            else
                example_heading="$heading_line"
            fi
        fi

        # DESCRIPTION markers
        if [[ $line =~ \[DESCRIPTION\] ]]; then
            in_description=1
            description=""
        elif [[ $line =~ \[/DESCRIPTION\] ]]; then
            in_description=0
        elif [ $in_description -eq 1 ]; then
            # Strip leading whitespace and comment markers
            desc_line="${line#"${line%%[![:space:]]*}"}"
            desc_line="${desc_line#// }"
            desc_line="${desc_line#//}"
            if [ -n "$description" ]; then
                description="$description"$'\n'"$desc_line"
            else
                description="$desc_line"
            fi
        fi

        # SNIPPET markers
        if [[ $line =~ \[SNIPPET\] ]]; then
            in_snippet=1
            snippet=""
        elif [[ $line =~ \[/SNIPPET\] ]]; then
            in_snippet=0

            # We have a complete triplet, write it out
            if [ -n "$example_heading" ] && [ -n "$snippet" ]; then
                echo "## $example_heading" >> "$output_file"
                echo "" >> "$output_file"

                # Write description if available
                if [ -n "$description" ]; then
                    echo "$description" >> "$output_file"
                    echo "" >> "$output_file"
                fi

                # Write snippet
                echo '```cpp' >> "$output_file"
                echo "$snippet" >> "$output_file"
                echo '```' >> "$output_file"
                echo "" >> "$output_file"
            fi

            # Reset for next triplet
            example_heading=""
            description=""
            snippet=""
        elif [ $in_snippet -eq 1 ]; then
            if [ -n "$snippet" ]; then
                snippet="$snippet"$'\n'"$line"
            else
                snippet="$line"
            fi
        fi
    done < "$testfile"

    echo "    Generated $output_file" >&2
done

echo "$TARGET documentation generation complete" >&2
