# Quickstart Test System

This directory contains executable documentation - test files with extractable code snippets that serve both as:
1. **Unit tests** - Verify code examples compile and work correctly
2. **Documentation source** - Provide snippets for README and tutorials

## Structure

Each test file contains snippets delimited by special markers:
```
// [quickstart:snippet-name]
... code snippet here ...
// [/quickstart:snippet-name]
```

## Current Snippets

- `data_packet_test.cpp`: Creates a VRT Signal Data Packet with UTC timestamp and incrementing payload sequence
- `context_packet_test.cpp`: Creates a VRT Context Packet with signal metadata fields
- `file_reader_test.cpp`: Reads VRT packets from a file using the high-level VRTFileReader API

## Extracting Snippets

To extract a snippet for documentation, you can use a simple script:

```bash
# Extract snippet between markers
sed -n '/\[quickstart:create-data-packet\]/,/\[\/quickstart:create-data-packet\]/p' create_packet_test.cpp | sed '1d;$d'
```

Or with a Python script:

```python
import re

def extract_snippet(filename, snippet_name):
    with open(filename, 'r') as f:
        content = f.read()

    pattern = rf'// \[quickstart:{snippet_name}\](.*?)// \[/quickstart:{snippet_name}\]'
    match = re.search(pattern, content, re.DOTALL)

    if match:
        # Remove the first and last lines (markers)
        lines = match.group(1).strip().split('\n')
        return '\n'.join(lines)
    return None

# Example usage:
code = extract_snippet('create_packet_test.cpp', 'create-data-packet')
print(code)
```

## Adding New Snippets

1. Create a new test file or add to existing one
2. Wrap snippet with markers: `[quickstart:name]` and `[/quickstart:name]`
3. Add basic assertions to verify correctness
4. Update CMakeLists.txt if adding new test file
5. Run test to ensure it compiles and passes

## Benefits

- **Always tested**: Snippets are part of CI/CD test suite
- **Self-contained**: Each snippet is complete and runnable
- **Versioned**: Snippets evolve with the codebase
- **Discoverable**: Developers can run tests to see examples in action