#!/bin/bash
# Extract CIF access examples and generate individual markdown files
# Processes all cif*_test.cpp files in tests/cif_access/
# Generates docs/cif_access/{basename}.md for each test file

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
CIF_ACCESS_DIR="$PROJECT_ROOT/tests/cif_access"
DOCS_DIR="$PROJECT_ROOT/docs/cif_access"

# Create docs directory if it doesn't exist
mkdir -p "$DOCS_DIR"

echo "Extracting CIF access documentation..." >&2

# Process each cif*_test.cpp file
for testfile in "$CIF_ACCESS_DIR"/cif*_test.cpp; do
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
        title="CIF Access Examples"
    fi

    # Start output file with header
    cat > "$output_file" << EOF
# $title

*Auto-generated from \`tests/cif_access/$filename\`. All examples are tested.*

---

EOF

    # Extract all EXAMPLE/DESCRIPTION/SNIPPET triplets
    # We'll read the file once and track state
    in_example=0
    in_description=0
    in_snippet=0
    example_heading=""
    description=""
    snippet=""

    while IFS= read -r line; do
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

echo "CIF access documentation generation complete" >&2
