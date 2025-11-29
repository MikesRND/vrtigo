# VRTIGO Style Guide

Items not specified here should generally follow the C++ Core Guidelines

- https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines

## Minimize API
- Expose only necessary functionality
- Use detail namespaces for internal implementations
- Prefer private/protected members over public where possible

## Namespace Organization

### Structure
- `vrtigo` - Main namespace for shared types (PacketType, Timestamp, ClassId, etc.)
- `vrtigo::dynamic` - Runtime packet parsing (flexible, dynamic structure)
- `vrtigo::typed` - Compile-time packets (high-performance, static structure)
- `vrtigo::field` - Field tag definitions (kept flat for convenience)
- `vrtigo::cif`, `vrtigo::trailer` - Narrow structs/enums kept separate for clarity
- `vrtigo::utils::fileio` - High-level I/O helpers (allocates, uses exceptions)
- `vrtigo::utils::netio` - UDP transport helpers (may allocate/throw)
- `vrtigo::utils::detail` - Shared iteration helpers (still considered internal)
- `vrtigo::detail` - Implementation details (never access directly)


## Accessor Methods

### Getters
- Format: `field_name()` - no prefix, snake_case
- Return: by value for primitives/small types
- Attributes: `[[nodiscard]]` only for critical values
- Const: always `const noexcept`

### Setters
- Format: `set_field_name()` - snake_case with set_ prefix
- Parameters: pass by value (enables move semantics)
- Return: void
- Const: `noexcept`

### Const-Correctness
- Use const/non-const overloading for mutable access
- Prefer returning different types over same-type overloads

## View Objects Pattern

### Structure
- `ConstFooView`: read-only view, contains only getters
- `FooView`: mutable view, inherits from `ConstFooView`, adds setters
- Parent class returns `FooView` from non-const, `ConstFooView` from const methods

### Synchronization
- Getters defined once in `ConstFooView`, inherited by `FooView`
- Bit positions/masks centralized in single header
- Both views use same utility functions for bit manipulation
- Single source of truth for field access logic

### When to Use
- Complex bit-packed structures
- Safety-critical fields requiring enforced const-correctness
- Zero-cost abstraction needed

## Builder Pattern
- Methods return `*this` for chaining
- Named after field: `builder.field_name(value)`
- Terminal method: `build()` returns constructed object

## Buffer Access
- Use `std::span` with compile-time sizes where possible
- Const/non-const overloads return appropriate span types

## Enums

### Enum Classes
- Always use `enum class`, never plain `enum`
- Name: PascalCase (e.g., `PacketType`, `ValidationError`)
- Underlying type: specify explicitly (e.g., `: uint8_t`)

### Enum Values
- Format: snake_case (e.g., `signal_data`, `buffer_too_small`)
- No prefix/suffix (rely on scoped enum namespace)
- Special values: `none` for absent/null state

### String Conversion
- Provide `to_string()` or `*_string()` function for user-facing enums
- Format: `constexpr const char* enum_name_string(EnumName value) noexcept`
