# Quickstart Test System

This directory contains executable documentation - test files with extractable code snippets that serve both as:
1. **Unit tests** - Verify code examples compile and work correctly
2. **Documentation source** - Provide snippets for tutorials in `docs/quickstart/`

## Marker Format

Each test file uses these markers to define extractable content:

```cpp
// [TITLE]
// Document Title Here
// [/TITLE]

// [EXAMPLE]
// Example Section Heading
// [/EXAMPLE]

// [DESCRIPTION]
// Multi-line description text explaining the example.
// [/DESCRIPTION]

TEST(TestSuite, TestName) {
    // [SNIPPET]
    // Actual code to extract
    auto result = create_something();
    // [/SNIPPET]

    // Additional test assertions (not extracted)
    ASSERT_TRUE(result.has_value());
}
```

A single test file can contain multiple `[EXAMPLE]`/`[DESCRIPTION]`/`[SNIPPET]` triplets.

### Narrative Text Between Examples

Use `[TEXT]` blocks to add connecting prose between examples:

```cpp
// [TEXT]
// This paragraph appears between examples without a heading or code block.
// Useful for transitions and narrative flow.
// [/TEXT]
```

## Current Test Files

- `data_packet_test.cpp` → `docs/quickstart/data_packet.md`
- `context_packet_test.cpp` → `docs/quickstart/context_packet.md`
- `file_reader_test.cpp` → `docs/quickstart/file_reader.md` (2 examples)
- `packet_parsing_test.cpp` → `docs/quickstart/packet_parsing.md` (4 examples)

## Generating Documentation

Documentation is auto-generated during build:

```bash
make autodocs
```

Or directly:

```bash
./scripts/extract_docs.sh quickstart
```

## Adding New Examples

1. Create a new test file: `tests/quickstart/{name}_test.cpp`
2. Add `[TITLE]` block at the top of the file
3. Add `[EXAMPLE]`, `[DESCRIPTION]`, and `[SNIPPET]` triplets
4. Add test assertions outside the `[SNIPPET]` markers
5. Update `tests/quickstart/CMakeLists.txt` to register the test
6. Run `make autodocs` to generate `docs/quickstart/{name}.md`

## Benefits

- **Always tested**: Snippets are part of CI/CD test suite
- **Self-contained**: Each snippet is complete and runnable
- **Versioned**: Snippets evolve with the codebase
- **Discoverable**: Developers can run tests to see examples in action
- **Consistent**: Same format as `cif_access` documentation
