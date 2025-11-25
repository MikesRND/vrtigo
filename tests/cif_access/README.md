# CIF Access Pattern Tests

This directory contains self-documenting tests that demonstrate how to access and use Context Information Fields (CIF). Each test file automatically generates its own markdown documentation page.

## Purpose

These tests serve dual purposes:
1. **Unit tests** - Verify CIF access patterns compile and work correctly
2. **Documentation source** - Auto-generate field-specific documentation pages

## Naming Convention

**Format:** `cif{N}_{bit}_{fieldname}_test.cpp` → `docs/cif_access/cif{N}_{bit}_{fieldname}.md`

**Examples:**
- `cif0_15_payload_format_test.cpp` → `docs/cif_access/cif0_15_payload_format.md`
- `cif0_20_bandwidth_test.cpp` → `docs/cif_access/cif0_20_bandwidth.md`
- `cif1_05_gain_test.cpp` → `docs/cif_access/cif1_05_gain.md`

**Why this convention?**
- Files sort naturally by CIF register and bit position
- Easy to locate documentation for specific fields
- Clear identification of which CIF field is documented

## Marker System

Each test file uses four types of markers to delimit extractable content:

### Document Title (once per file)
```cpp
// [TITLE]
// Payload Format Field (CIF0 bit 15)
// [/TITLE]
```

### Example Sections (multiple per file)
```cpp
TEST_F(ContextPacketTest, CIF0_15_ThreeTierAccess) {
    // [EXAMPLE]
    // Three-Tier Access Pattern
    // [/EXAMPLE]

    // [DESCRIPTION]
    // The Payload Format field uses a three-tier access pattern:
    // - `.bytes()` returns raw bytes (8 bytes)
    // - `.encoded()` returns generic FieldView<2>
    // - `.value()` returns structured PayloadFormat::View
    // [/DESCRIPTION]

    // [SNIPPET]
    using FormatContext = ContextPacket<NoTimestamp, NoClassId, data_payload_format>;

    alignas(4) std::array<uint8_t, FormatContext::size_bytes> buffer{};
    FormatContext packet(buffer.data());

    auto proxy = packet[data_payload_format];
    auto raw = proxy.bytes();        // Tier 1: Raw bytes
    auto encoded = proxy.encoded();  // Tier 2: FieldView<2>
    auto format = proxy.value();     // Tier 3: PayloadFormat::View
    // [/SNIPPET]

    // Assertions (not extracted to docs)
    EXPECT_EQ(raw.size(), 8);
}
```

## Marker Reference

| Marker | Purpose | Occurrences |
|--------|---------|-------------|
| `[TITLE]` | Document header/title | Once per file (at top) |
| `[EXAMPLE]` | Section heading | Multiple (one per section) |
| `[DESCRIPTION]` | Section description | Multiple (paired with SNIPPET) |
| `[SNIPPET]` | Code to extract | Multiple (paired with DESCRIPTION) |

All markers use `[TAG]` ... `[/TAG]` format.

## File Structure

```cpp
#include "../cif/context_test_fixture.hpp"
#include "vrtigo/detail/{fieldname}.hpp"

using namespace vrtigo;

// [TITLE]
// Field Name (CIF{N} bit {bit})
// [/TITLE]

TEST_F(ContextPacketTest, CIF{N}_{bit}_ExampleName1) {
    // [EXAMPLE]
    // Example 1 Heading
    // [/EXAMPLE]

    // [DESCRIPTION]
    // Description of what this example demonstrates...
    // [/DESCRIPTION]

    // [SNIPPET]
    // Extractable code snippet...
    // [/SNIPPET]

    // Assertions (not extracted)
    EXPECT_EQ(...);
}

TEST_F(ContextPacketTest, CIF{N}_{bit}_ExampleName2) {
    // [EXAMPLE], [DESCRIPTION], [SNIPPET] for second example...
}
```

## Generated Documentation

Each test file generates a corresponding markdown page in `docs/cif_access/`:

**docs/cif_access/cif0_15_payload_format.md:**
```markdown
# Payload Format Field (CIF0 bit 15)

*Auto-generated from `tests/cif_access/cif0_15_payload_format_test.cpp`. All examples are tested.*

---

## Three-Tier Access Pattern

The Payload Format field uses a three-tier access pattern:
- `.bytes()` returns raw bytes (8 bytes)
- `.encoded()` returns generic FieldView<2>
- `.value()` returns structured PayloadFormat::View

\`\`\`cpp
using FormatContext = ContextPacket<NoTimestamp, NoClassId, data_payload_format>;
...
\`\`\`
```

## Documentation Generation

Documentation is automatically generated during build via `scripts/extract_cif_access.sh`:
- Processes all `cif*_test.cpp` files in this directory
- Extracts marker-delimited content
- Generates individual markdown files in `docs/cif_access/`
- Sections appear in source code order

## Adding New CIF Field Documentation

1. **Identify the field**: Note CIF register (0/1/2/3) and bit position
2. **Create test file**: `tests/cif_access/cif{N}_{bit}_{fieldname}_test.cpp`
3. **Add title marker**: Include CIF register and bit in title
4. **Write test cases**: Each test case becomes one documentation section
5. **Add markers**: Wrap examples with `[EXAMPLE]`, `[DESCRIPTION]`, `[SNIPPET]`
6. **Add assertions**: Verify examples work correctly (not extracted to docs)
7. **Update CMakeLists.txt**: Add `vrtigo_add_gtest()` line
8. **Build and test**: Verify tests pass and docs generate correctly

## Benefits

- **Always tested**: All examples guaranteed to compile and pass tests
- **Self-contained**: Each field gets its own documentation page
- **Natural ordering**: Files sort by CIF register and bit position
- **Version controlled**: Docs evolve with code
- **Low maintenance**: Single source of truth for examples
